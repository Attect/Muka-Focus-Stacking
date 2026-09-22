"""Measure the seam between a near object and the background.

The halo this targets is a *low-frequency* defect: just outside the object the
image is brighter (or darker) than the surrounding background should be, because
the object's defocused bloom is being used as the base band underneath its own
sharp detail. Edge-acutance figures cannot see it — they only look at the
strongest gradient — so this measures the edge *profile* instead:

  * transition width  — distance between the 20% and 80% crossings, in pixels
  * overshoot         — how far the profile rises above its own plateau on the
                        outside of the object. A clean edge gives ~0; a bloom
                        gives a positive number that grows with halo strength.

Usage:
    seam.py label=path [label=path ...] -- x:y:size[:x:y:size ...]
"""
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, "tools")
from patchlib import load, locate, REF  # noqa: E402

Image.MAX_IMAGE_PIXELS = None

# Default spots: the seam the user pointed at (boot against the backdrop), the
# head silhouette, and the base edge.
DEFAULT = [(1639, 1245, 220), (2010, 700, 220), (1850, 1450, 220)]


def strongest_edge(img, cx, cy, size, border=24):
    """Location and unit normal of the strongest edge inside the window."""
    h = size // 2
    sub = img[cy - h:cy + h, cx - h:cx + h, :3].astype(np.float32)
    g = sub @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    gy, gx = np.gradient(g)
    m = np.hypot(gx, gy)
    m[:border] = 0
    m[-border:] = 0
    m[:, :border] = 0
    m[:, -border:] = 0
    iy, ix = np.unravel_index(np.argmax(m), m.shape)
    nx, ny = float(gx[iy, ix]), float(gy[iy, ix])
    n = np.hypot(nx, ny) + 1e-9
    return (cx - h + ix, cy - h + iy), (nx / n, ny / n), float(m[iy, ix])


def profile(img, pt, normal, half=70, step=0.5):
    ts = np.arange(-half, half + 1e-9, step)
    xs = pt[0] + normal[0] * ts
    ys = pt[1] + normal[1] * ts
    x0 = np.clip(np.floor(xs).astype(int), 0, img.shape[1] - 2)
    y0 = np.clip(np.floor(ys).astype(int), 0, img.shape[0] - 2)
    fx = (xs - x0)[:, None]
    fy = (ys - y0)[:, None]
    rgb = (
        img[y0, x0, :3] * (1 - fx) * (1 - fy)
        + img[y0, x0 + 1, :3] * fx * (1 - fy)
        + img[y0 + 1, x0, :3] * (1 - fx) * fy
        + img[y0 + 1, x0 + 1, :3] * fx * fy
    )
    return ts, rgb @ np.array([0.299, 0.587, 0.114], dtype=np.float32)


def describe(ts, v):
    neg = v[ts <= -40].mean()
    pos = v[ts >= 40].mean()
    span = pos - neg
    sign = 1.0 if span >= 0 else -1.0
    norm = (v - neg) / (span + 1e-9)
    lo = np.where(sign * norm >= 0.2)[0]
    hi = np.where(sign * norm >= 0.8)[0]
    width = (ts[hi[0]] - ts[lo[-1]]) if len(lo) and len(hi) else float("nan")
    # overshoot: how far the profile rises above the plateau on each side
    out_neg = 1.0 - norm[(ts >= -34) & (ts <= -8)].min() * sign if span >= 0 else 0.0
    out_pos = norm[(ts >= 8) & (ts <= 34)].max() - 1.0
    return width, out_neg, max(out_pos, 0.0), neg, pos


def main():
    args = sys.argv[1:]
    spots = DEFAULT
    if "--" in args:
        i = args.index("--")
        spec = args[i + 1:]
        args = args[:i]
        spots = []
        for s in spec:
            _, x, y, sz = ([""] + s.split(":"))[-4:]
            spots.append((int(x), int(y), int(sz)))

    ref = load(REF)
    variants = [("AF export", ref, (0, 0))]
    for a in args:
        lab, path = a.split("=", 1)
        img = load(path)
        oy, ox, _ = locate(img, ref)
        variants.append((lab, img, (oy, ox)))

    for (cx, cy, size) in spots:
        pt, normal, m = strongest_edge(ref, cx, cy, size)
        print("=== seam near (%d,%d): AF edge at %s, normal (%.2f, %.2f), |grad| %.0f"
              % (cx, cy, pt, normal[0], normal[1], m))
        print("%-18s %9s %10s %10s" % ("variant", "width(px)", "overshoot-", "overshoot+"))
        for lab, img, org in variants:
            p = (pt[0] + org[1], pt[1] + org[0])
            ts, v = profile(img, p, normal)
            width, on, op, neg, pos = describe(ts, v)
            print("%-18s %9.1f %10.3f %10.3f" % (lab, width, on, op))
        print()


if __name__ == "__main__":
    main()
