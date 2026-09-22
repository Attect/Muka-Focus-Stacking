"""Scan the wall strip beside a silhouette and compare it against the frames.

For each row along the silhouette we locate the true edge (largest luminance
drop in the reference export), then take a window that lies entirely inside the
*far* surface, just short of the edge. The high-frequency energy inside that
window is then compared between the outputs and against the per-frame maximum,
which is the ceiling any merge could reach.

The window deliberately excludes the edge: a window that touches it is dominated
by the edge's own step, and a 15 px smooth band next to a 15 px window then
disappears into it. That mistake cost a whole round earlier.

    python tools/edgescan.py Y0 Y1 X0 X1 [variants...]

Variants are `label=path`. The reference export is always included.
"""
import os
import sys

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from patchlib import REF, load, locate  # noqa: E402

D = os.environ.get("STACK_DIR", "stack")
DREF = (495, 1287)  # reference index = canvas - DREF


def highpass(m, k=3):
    p = np.pad(m, ((k + 1, k), (k + 1, k)), mode="edge")
    c = np.cumsum(np.cumsum(p, 0, dtype=np.float64), 1, dtype=np.float64)
    kk = 2 * k + 1
    b = (c[kk:, kk:] - c[:-kk, kk:] - c[kk:, :-kk] + c[:-kk, :-kk]) / float(kk * kk)
    return np.abs(m - b)


def gray(im):
    return (0.299 * im[..., 0] + 0.587 * im[..., 1] + 0.114 * im[..., 2]).astype(np.float64)


def main():
    y0, y1, x0, x1 = (int(v) for v in sys.argv[1:5])
    variants = []
    for a in sys.argv[5:]:
        lab, _, p = a.partition("=")
        variants.append((lab, p))

    ref = load(REF)
    G = {"AF export": (gray(ref), DREF)}
    for lab, p in variants:
        im = load(p)
        oy, ox, _ = locate(im, ref)
        G[lab] = (gray(im), (DREF[0] - oy, DREF[1] - ox))

    files = sorted(
        x for x in os.listdir(D) if x.upper().startswith("DSC") and x.upper().endswith(".JPG")
    )

    print("canvas y %d..%d" % (y0, y1))
    print("%6s %7s %8s" % ("y", "edge x", "win x") + "".join("%16s" % l for l in G) + "%14s" % "ceiling")

    tot = {l: [] for l in G}
    ceil_all = []
    for y in range(y0, y1 + 1, 20):
        seg = G["AF export"][0][y - DREF[0], x0:x1]
        ex = x0 + int(np.argmax(np.abs(np.diff(seg)))) + 1
        wa, wb = ex - 22, ex - 4  # strictly inside the far surface
        if wa < x0 + 4 or wb > x1 - 4:
            continue
        row = {}
        for lab, (g, off) in G.items():
            w = g[y - off[0] : y - off[0] + 24, wa - off[1] : wb - off[1]]
            e = highpass(w)
            row[lab] = float(e[4:-4, 4:-4].mean())
            tot[lab].append(row[lab])
        # ceiling: the same window on each original frame, best over frames
        best, bestk = 0.0, -1
        for k, f in enumerate(files):
            im = Image.open(os.path.join(D, f)).convert("L").crop((wa, y0 - 40, wb, y1 + 60))
            a = np.asarray(im, dtype=np.float64)
            w = a[y - (y0 - 40) : y - (y0 - 40) + 24, :]
            e = highpass(w)
            v = float(e[4:-4, 4:-4].mean())
            if v > best:
                best, bestk = v, k
        ceil_all.append(best)
        line = "%6d %7d %8d" % (y, ex, wa)
        for lab in G:
            line += "%16.2f" % row[lab]
        line += "%10.2f@%d" % (best, bestk)
        print(line)

    print()
    print("%-16s %10s %10s" % ("", "mean", "%% of ceiling"))
    c = np.mean(ceil_all)
    for lab in G:
        m = np.mean(tot[lab])
        print("%-16s %10.2f %9.0f%%" % (lab, m, 100 * m / c))
    print("%-16s %10.2f" % ("ceiling (per-frame max)", c))


if __name__ == "__main__":
    main()
