"""Dense comparison of the saved depth field against the offline focus measure.

For a grid of points it reports the frame the measure picks (mirroring the
pipeline: 1-px box prefilter, SML window of `--focus-radius`, measured on the
unwarped frame) and the depth the pipeline wrote. Where they disagree, the depth
pipeline moved the value; where they agree and the peak is barely above the
median, the measure had nothing to measure and the value is a coin flip.

    python tools/measure_agree.py DEPTH.png Y0 Y1 X0 X1 STEP
"""
import os
import sys

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

D = os.environ.get("STACK_DIR", "stack")
NFRAMES = 49.0
R = 4


def box_blur(a, r):
    k = 2 * r + 1
    p = np.pad(a, ((r + 1, r), (r + 1, r)), mode="edge")
    c = np.cumsum(np.cumsum(p, 0, dtype=np.float64), 1, dtype=np.float64)
    return ((c[k:, k:] - c[:-k, k:] - c[k:, :-k] + c[:-k, :-k]) / float(k * k)).astype(np.float32)


def main():
    dz = np.asarray(Image.open(sys.argv[1]), dtype=np.float32) / 65535.0 * NFRAMES
    y0, y1, x0, x1, step = (int(v) for v in sys.argv[2:7])
    files = sorted(x for x in os.listdir(D) if x.upper().startswith("DSC") and x.upper().endswith(".JPG"))
    print("frames %d, window radius %d" % (len(files), R))

    pts = [(y, x) for y in range(y0, y1 + 1, step) for x in range(x0, x1 + 1, step)]
    resp = np.zeros((len(files), len(pts)), dtype=np.float32)
    for k, f in enumerate(files):
        a = np.asarray(Image.open(os.path.join(D, f)).convert("RGB"), dtype=np.float32)
        g = 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]
        p = box_blur(g, 1)
        # Laplacian-ish SML, box-summed over the focus window
        lap = np.zeros_like(p)
        lap[2:-2, 2:-2] = (
            np.abs(2 * p[2:-2, 2:-2] - p[2:-2, :-4] - p[2:-2, 4:])
            + np.abs(2 * p[2:-2, 2:-2] - p[:-4, 2:-2] - p[4:, 2:-2])
        )
        s = box_blur(lap, R)
        for j, (y, x) in enumerate(pts):
            resp[k, j] = s[y, x]
        if k % 10 == 0:
            print("  frame", k, flush=True)
        del a, g, p, lap, s

    arg = resp.argmax(axis=0)
    peak = resp.max(axis=0)
    med = np.median(resp, axis=0)
    disc = peak / np.maximum(med, 1e-9)
    field = np.array([dz[y, x] for y, x in pts])

    agree = np.abs(arg - field)
    print()
    print("%-14s %-14s %-14s" % ("group", "n", "share"))
    print("%-14s %-14d %.1f%%" % ("all points", len(pts), 100.0))
    for lo, hi, lab in [(0, 2, "|diff| <= 1"), (2, 6, "2..5"), (6, 16, "6..15"), (16, 99, ">= 16")]:
        m = (agree >= lo) & (agree <= hi)
        print("%-14s %-14d %.1f%%   median peak/med %.2f" % (lab, m.sum(), 100 * m.mean(), np.median(disc[m]) if m.any() else float("nan")))
    print()
    print("peak/median discrimination by whether the measure agrees with the field:")
    for lo, hi, lab in [(0, 2, "agree"), (2, 6, "small diff"), (16, 99, "big diff")]:
        m = (agree >= lo) & (agree <= hi)
        if m.any():
            print("  %-11s n=%5d  disc p10 %.2f  median %.2f  p90 %.2f"
                  % (lab, m.sum(), np.percentile(disc[m], 10), np.median(disc[m]), np.percentile(disc[m], 90)))
    print()
    print("the measure's own discrimination distribution over the grid:")
    for q in [10, 25, 50, 75, 90]:
        print("   p%-3d %.2f" % (q, np.percentile(disc, q)))


if __name__ == "__main__":
    main()
