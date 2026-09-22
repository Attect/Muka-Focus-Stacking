"""Texture of the far surface as a function of distance from the silhouette.

Places the window at fixed distances *inside the wall*, measured from the
silhouette as located in an output image (where it is genuinely sharp — in the
frames the ramp is ~16 px wide, so an edge found there is meaningless).

Row by row, then averaged. The per-frame maximum at the same distance is the
ceiling: what the stack physically contains there.

    python tools/distdecay.py OUROUT.png Y0 Y1 X0 X1 [label=path ...]
"""
import os
import sys

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from patchlib import REF, load  # noqa: E402

D = os.environ.get("STACK_DIR", "stack")
CANVAS_FROM_OUT = (12, 18)
CANVAS_FROM_REF = (506, 1304)
DISTS = [8, 10, 12, 15, 18, 22, 26, 30, 36, 42, 50]
WIN = 12  # window is [edge-d-WIN, edge-d], wide enough for the highpass margin


def gray(x):
    return (0.299 * x[..., 0] + 0.587 * x[..., 1] + 0.114 * x[..., 2]).astype(np.float64)


def highpass(m, k=3):
    p = np.pad(m, ((k + 1, k), (k + 1, k)), mode="edge")
    c = np.cumsum(np.cumsum(p, 0, dtype=np.float64), 1, dtype=np.float64)
    kk = 2 * k + 1
    return np.abs(m - (c[kk:, kk:] - c[:-kk, kk:] - c[kk:, :-kk] + c[:-kk, :-kk]) / float(kk * kk))


def energy(g, off, ys, xs):
    vals = []
    for y, x0, x1 in zip(ys, xs[:, 0], xs[:, 1]):
        w = g[y - off[0] - 6 : y - off[0] + 6, x0 - off[1] : x1 - off[1]]
        if w.shape != (12, x1 - x0):
            continue
        e = highpass(w)
        vals.append(float(e[3:-3, 3:-3].mean()))
    return float(np.mean(vals)) if vals else float("nan")


def main():
    out_path = sys.argv[1]
    y0, y1, x0, x1 = (int(v) for v in sys.argv[2:6])
    variants = []
    for a in sys.argv[6:]:
        lab, _, p = a.partition("=")
        variants.append((lab, p))

    ours = load(out_path)
    go = gray(ours)
    edges, ys = [], []
    for y in range(y0, y1 + 1):
        row = go[y - CANVAS_FROM_OUT[0], x0 - CANVAS_FROM_OUT[1] : x1 - CANVAS_FROM_OUT[1]]
        d = np.abs(np.diff(row))
        i = int(np.argmax(d))
        if d[i] < 25:  # no silhouette on this row
            continue
        edges.append(x0 + i + 1)
        ys.append(y)
    edges = np.array(edges)
    print("rows used: %d, edge x %d..%d" % (len(ys), edges.min(), edges.max()))

    G = [("ours", go, CANVAS_FROM_OUT)]
    for lab, p in variants:
        G.append((lab, gray(load(p)), CANVAS_FROM_OUT))
    G.append(("AF export", gray(load(REF)), CANVAS_FROM_REF))

    files = sorted(
        x for x in os.listdir(D) if x.upper().startswith("DSC") and x.upper().endswith(".JPG")
    )

    print()
    print("%6s" % "dist" + "".join("%14s" % l for l, _, _ in G) + "%14s" % "ceiling")
    for d in DISTS:
        xs = np.stack([edges - d - WIN, edges - d], 1)
        line = "%6d" % d
        for lab, g, off in G:
            line += "%14.2f" % energy(g, off, ys, xs)
        best = 0.0
        for f in files:
            a = np.asarray(Image.open(os.path.join(D, f)).convert("L"), dtype=np.float64)
            v = energy(a, (0, 0), ys, xs)
            best = max(best, v)
        line += "%14.2f" % best
        print(line)


if __name__ == "__main__":
    main()
