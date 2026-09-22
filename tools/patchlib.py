"""Compare outputs on named patches, locating each output's crop origin itself.

Every output is a different crop of the alignment canvas, so the reference
crop's origin inside it moves from run to run. Hard-coding that origin (as
several earlier scripts did) silently compares *different places* between two
variants and produces contradictory numbers. This module locates the origin per
image by matching against the reference, then samples.

Usage as a library:
    from patchlib import locate, load, acutance
Usage as a script:
    patchcmp.py label=path [label=path ...] -- patch:name:x:y:size ...
"""
import os

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
REF = os.environ.get("REF_EXPORT",
      os.path.join(os.environ.get("STACK_DIR", "stack"), "AF导出.png"))

DEFAULT_PATCHES = [
    ("wall@33", 1560, 1160, 140),
    ("knee-edge", 1639, 1245, 140),
    ("smooth-body", 1660, 1230, 140),
    ("leaf", 1400, 3040, 200),
    ("face", 2000, 700, 140),
    ("flat-backdrop", 2600, 1200, 160),
]


def load(path):
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32)


def locate(img, ref=None, search=None):
    """Find where the reference crop starts inside `img`, to 1 px.

    The reference crop's origin sits near (1288, 490) in every output this tool
    produces (it is the frame minus the alignment padding), so a global search in
    a generous box around that is both fast and unambiguous. Locating it by
    starting from the geometric centre and descending, by contrast, latches onto
    the wrong minimum.
    """
    if ref is None:
        ref = load(REF)
    oy0, oy1, ox0, ox1 = search or (455, 570, 1255, 1345)
    g = 8
    ao = np.asarray(Image.fromarray(img.astype(np.uint8)).convert("L"), dtype=np.float32)[::g, ::g]
    aa = np.asarray(Image.fromarray(ref.astype(np.uint8)).convert("L"), dtype=np.float32)[::g, ::g]
    h, w = aa.shape
    best = None
    for oy in range(oy0, oy1):
        for ox in range(ox0, ox1):
            y0, x0 = oy // g, ox // g
            sub = ao[y0:y0 + h, x0:x0 + w]
            if sub.shape != aa.shape:
                continue
            mad = float(np.abs(sub - aa).mean())
            if best is None or mad < best[0]:
                best = (mad, ox, oy)
    if best is None:
        return None
    # Refine at full resolution around the coarse winner.
    _, ox, oy = best
    S = 256
    ry, rx = ref.shape[0] // 2 - S // 2, ref.shape[1] // 2 - S // 2
    probe = ref[ry:ry + S, rx:rx + S]
    best = None
    for dy in range(-6, 7):
        for dx in range(-6, 7):
            y, x = oy + dy, ox + dx
            if y < 0 or x < 0 or y + ry + S > img.shape[0] or x + rx + S > img.shape[1]:
                continue
            sub = img[y + ry:y + ry + S, x + rx:x + rx + S]
            mad = float(np.abs(sub - probe).mean())
            if best is None or mad < best[0]:
                best = (mad, x, y)
    return best[2], best[1], best[0]


def gray(a):
    return a[..., 0] * 0.299 + a[..., 1] * 0.587 + a[..., 2] * 0.114


def acutance(a, q=0.98):
    gy, gx = np.gradient(gray(a).astype(np.float32))
    m = np.hypot(gx, gy).ravel()
    return float(m[m >= np.quantile(m, q)].mean())


def gradient_energy(a):
    gy, gx = np.gradient(gray(a).astype(np.float32))
    return float(np.hypot(gx, gy).mean())


def texture_std(a):
    return float(a.reshape(-1, 3).std(0).mean())


def grain(a):
    """High-frequency residual: how much the image differs from its own 3x3 median.

    In a region without real single-pixel detail this is sensor/selection grain,
    which is the figure the per-band maximum inflates. Reported globally over the
    reference-sized area so it cannot be cherry-picked by patch choice.
    """
    g = gray(a).astype(np.float32)
    p = np.pad(g, 1, mode="edge")
    stack = np.stack(
        [p[dy:dy + g.shape[0], dx:dx + g.shape[1]] for dy in range(3) for dx in range(3)]
    )
    med = np.median(stack, axis=0)
    return float((g - med).std())


def main():
    import sys

    args = sys.argv[1:]
    patches = DEFAULT_PATCHES
    if "--" in args:
        i = args.index("--")
        spec = args[i + 1:]
        args = args[:i]
        if spec:
            patches = []
            for s in spec:
                _, name, x, y, sz = s.split(":")
                patches.append((name, int(x), int(y), int(sz)))

    ref = load(REF)
    variants = [("AF export", ref, (0, 0))]
    for a in args:
        lab, path = a.split("=", 1)
        img = load(path)
        oy, ox, mad = locate(img, ref)
        variants.append((lab, img, (oy, ox), mad))

    header = "%-20s %8s %7s %7s" % ("variant", "gradE", "grain", "match")
    for name, _, _, _ in patches:
        header += "%12s" % name
    print(header)
    for v in variants:
        lab, img, org = v[0], v[1], v[2]
        mad = v[3] if len(v) > 3 else 0.0
        crop = img[org[0]:org[0] + ref.shape[0], org[1]:org[1] + ref.shape[1]]
        line = "%-20s %8.3f %7.3f %7.2f" % (lab, gradient_energy(img), grain(crop), mad)
        for name, x, y, s in patches:
            line += "%12.2f" % acutance(img[org[0] + y:org[0] + y + s, org[1] + x:org[1] + x + s])
        print(line)


if __name__ == "__main__":
    main()
