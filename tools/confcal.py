"""Compute the pipeline's reliability statistic at labelled points, offline.

The statistic is the peak-to-mean ratio of the per-frame focus response over the
stack. Two notes in the code contradict each other about it — one says genuine
focus information scores 1.20-1.52 against 1.09-1.13 for a featureless surface,
the other says it came out *backwards* (1.24 on a featureless surface against
1.12 on a textured face). Whether a per-pixel adaptive band combination is worth
building depends on which is true, so measure it on named regions.

Mirrors the pipeline: 1-px box prefilter, SML summed over a window of
`--focus-radius`, computed on the *unwarped* frames.

    python tools/confcal.py
"""
import os
import sys

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

D = os.environ.get("STACK_DIR", "stack")
R = 4


def box_blur(a, r):
    k = 2 * r + 1
    p = np.pad(a, ((r + 1, r), (r + 1, r)), mode="edge")
    c = np.cumsum(np.cumsum(p, 0, dtype=np.float64), 1, dtype=np.float64)
    return ((c[k:, k:] - c[:-k, k:] - c[k:, :-k] + c[:-k, :-k]) / float(k * k)).astype(np.float32)


POINTS = [
    ("wall-relief", 2862, 2770),
    ("wall-relief", 2800, 2790),
    ("wall-relief", 2900, 2760),
    ("flat-backdrop", 2400, 1500),
    ("flat-backdrop", 2500, 1600),
    ("plastic-smooth", 3300, 3400),
    ("plastic-smooth", 3400, 3500),
    ("plastic-smooth", 3200, 3300),
    ("leather-band", 3000, 3300),
    ("leather-band", 3050, 3250),
    ("stitch-holes", 3080, 3350),
    ("blotch", 3183, 3974),
    ("blotch", 4409, 3379),
    ("blotch", 2581, 4124),
    ("face", 1700, 2900),
]


def main():
    files = sorted(x for x in os.listdir(D) if x.upper().startswith("DSC") and x.upper().endswith(".JPG"))
    resp = np.zeros((len(files), len(POINTS)), dtype=np.float32)
    for k, f in enumerate(files):
        a = np.asarray(Image.open(os.path.join(D, f)).convert("RGB"), dtype=np.float32)
        g = 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]
        p = box_blur(g, 1)
        lap = np.zeros_like(p)
        lap[2:-2, 2:-2] = (
            np.abs(2 * p[2:-2, 2:-2] - p[2:-2, :-4] - p[2:-2, 4:])
            + np.abs(2 * p[2:-2, 2:-2] - p[:-4, 2:-2] - p[4:, 2:-2])
        )
        s = box_blur(lap, R)
        for j, (_, y, x) in enumerate(POINTS):
            resp[k, j] = s[y, x]
        if k % 10 == 0:
            print("  frame", k, flush=True)
        del a, g, p, lap, s

    print()
    print("%-16s %-12s %10s %10s %8s" % ("region", "canvas y,x", "peak/mean", "peak/med", "peak frame"))
    for j, (lab, y, x) in enumerate(POINTS):
        v = resp[:, j]
        print("%-16s %-12s %10.3f %10.3f %8d"
              % (lab, "%d,%d" % (y, x), v.max() / max(v.mean(), 1e-9),
                 v.max() / max(np.median(v), 1e-9), int(np.argmax(v))))
    print()
    for lo, hi, name in [(0, 3, "wall-relief"), (3, 5, "flat-backdrop"),
                         (5, 8, "plastic-smooth"), (8, 10, "leather-band"),
                         (10, 11, "stitch-holes"), (11, 14, "blotch"), (14, 15, "face")]:
        v = resp[:, lo:hi]
        pm = v.max(axis=0) / np.maximum(v.mean(axis=0), 1e-9)
        print("%-16s peak/mean  中位 %.3f  范围 %.3f..%.3f" % (name, np.median(pm), pm.min(), pm.max()))


if __name__ == "__main__":
    main()
