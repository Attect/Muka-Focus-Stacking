"""Compare the grain on smooth subject surfaces across fusion settings.

`--band-power 0` takes the per-band maximum, which is what makes the result
sharp; it also lets each band pick a different frame, and on a smooth surface
without real single-pixel detail that mixture shows up as grain. `--band-power
N > 0` averages the candidates with an energy weighting instead. This reports the
grain in three regions plus the silhouette band, so the trade-off is visible in
one place.

Canvas mapping (verified): canvas = output + (12, 18) = reference + (506, 1304)

    python tools/grain.py "label=path" ["label=path" ...]
"""
import os
import sys

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from patchlib import REF, load  # noqa: E402

CANVAS_FROM_OUT = (12, 18)
CANVAS_FROM_REF = (506, 1304)
REGIONS = [
    ("plastic-smooth", 3050, 3570, 3150, 3670),
    ("leather-stitch", 2950, 3160, 3200, 3500),
    ("flat-backdrop", 2400, 2600, 1400, 1700),
]


def gray(x):
    return (0.299 * x[..., 0] + 0.587 * x[..., 1] + 0.114 * x[..., 2]).astype(np.float64)


def highpass(m, k=3):
    p = np.pad(m, ((k + 1, k), (k + 1, k)), mode="edge")
    c = np.cumsum(np.cumsum(p, 0, dtype=np.float64), 1, dtype=np.float64)
    kk = 2 * k + 1
    return np.abs(m - (c[kk:, kk:] - c[:-kk, kk:] - c[kk:, :-kk] + c[:-kk, :-kk]) / float(kk * kk))


def energy(g, off, y0, y1, x0, x1, r=16):
    w = g[y0 - off[0] : y1 - off[0], x0 - off[1] : x1 - off[1]]
    e = highpass(w)
    k = 2 * r + 1
    p = np.pad(e, ((r + 1, r), (r + 1, r)), mode="edge")
    c = np.cumsum(np.cumsum(p, 0, dtype=np.float64), 1, dtype=np.float64)
    b = (c[k:, k:] - c[:-k, k:] - c[k:, :-k] + c[:-k, :-k]) / float(k * k)
    return float(np.median(b[r:-r, r:-r]))


def main():
    variants = []
    for a in sys.argv[1:]:
        lab, _, p = a.partition("=")
        variants.append((lab, gray(load(p)), CANVAS_FROM_OUT))
    variants.append(("AF export", gray(load(REF)), CANVAS_FROM_REF))

    print("%-20s" % "" + "".join("%16s" % lab for lab, _, _ in variants))
    for name, y0, y1, x0, x1 in REGIONS:
        print("%-20s" % name + "".join("%16.2f" % energy(g, o, y0, y1, x0, x1) for _, g, o in variants))


if __name__ == "__main__":
    main()
