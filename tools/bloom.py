"""Locate and quantify low-frequency disagreement with the reference.

A seam halo is a *wide, smooth* brightening just outside a foreground object, so
it survives low-pass filtering while ordinary sharpness differences do not. This
blurs both images with a big box, takes the difference, and reports where and how
strongly the variant is brighter than the reference.

That gives two things at once: a scalar to compare variants on, and the place to
look at when the number is bad.

Usage:
    bloom.py label=path [label=path ...] [--radius 12]
"""
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, "tools")
from patchlib import gray, load, locate, REF  # noqa: E402

Image.MAX_IMAGE_PIXELS = None


def box(a, r):
    h, w = a.shape
    c = np.pad(np.cumsum(np.cumsum(a, 0), 1), ((1, 0), (1, 0)))
    ys, xs = np.arange(h), np.arange(w)
    y0, y1 = np.clip(ys - r, 0, h), np.clip(ys + r + 1, 0, h)
    x0, x1 = np.clip(xs - r, 0, w), np.clip(xs + r + 1, 0, w)
    s = c[np.ix_(y1, x1)] - c[np.ix_(y0, x1)] - c[np.ix_(y1, x0)] + c[np.ix_(y0, x0)]
    return s / ((y1 - y0)[:, None] * (x1 - x0)[None, :]).astype(np.float32)


def main():
    args = sys.argv[1:]
    radius = 12
    if "--radius" in args:
        i = args.index("--radius")
        radius = int(args[i + 1])
        del args[i:i + 2]

    ref = load(REF)
    ref_lp = box(gray(ref), radius)
    # The reference includes its own black border; ignore a margin.
    m = 20
    ref_lp = ref_lp[m:-m, m:-m]

    print("%-18s %9s %9s %9s   %s" % ("variant", "p99(+)", "max(+)", "p1(-)", "location of the strongest brightening"))
    for a in args:
        lab, path = a.split("=", 1)
        img = load(path)
        oy, ox, _ = locate(img, ref)
        sub = img[oy + m:oy + ref.shape[0] - m, ox + m:ox + ref.shape[1] - m]
        d = box(gray(sub), radius) - ref_lp
        pos = d[d > 0]
        flat = d.ravel()
        k = int(np.argmax(flat))
        ky, kx = divmod(k, d.shape[1])
        print("%-18s %9.2f %9.2f %9.2f   AF-crop (%d, %d)"
              % (lab, np.percentile(pos, 99) if pos.size else 0.0, d.max(),
                 np.percentile(d, 1), kx + m, ky + m))


if __name__ == "__main__":
    main()
