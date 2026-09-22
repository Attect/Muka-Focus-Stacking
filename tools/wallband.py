"""Measure the wall strip beside a silhouette, in canvas coordinates.

Conversion, verified twice (locate + local template match with an RGB check):

    canvas = output + (0, 8) = reference export + (502, 1295)

The strip is placed relative to the *true* edge as seen in frame 25 (the
alignment reference frame, whose transform is the identity, so it shares the
canvas pixel grid exactly). The window lies entirely inside the far surface, a
few pixels short of the edge: a window that touches the edge is dominated by the
edge step itself and hides exactly the defect we are looking for.
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


def gray(x):
    return (0.299 * x[..., 0] + 0.587 * x[..., 1] + 0.114 * x[..., 2]).astype(np.float64)


def highpass(m, k=3):
    p = np.pad(m, ((k + 1, k), (k + 1, k)), mode="edge")
    c = np.cumsum(np.cumsum(p, 0, dtype=np.float64), 1, dtype=np.float64)
    kk = 2 * k + 1
    return np.abs(m - (c[kk:, kk:] - c[:-kk, kk:] - c[kk:, :-kk] + c[:-kk, :-kk]) / float(kk * kk))


def band(g, off, y, x0, x1, hh=24):
    w = g[y - off[0] : y - off[0] + hh, x0 - off[1] : x1 - off[1]]
    e = highpass(w)
    return float(e[4:-4, 4:-4].mean())


def main():
    ys = [int(v) for v in sys.argv[1].split(",")]
    xlo, xhi = (int(v) for v in sys.argv[2].split(":"))
    inset = int(sys.argv[3]) if len(sys.argv) > 3 else 6
    width = int(sys.argv[4]) if len(sys.argv) > 4 else 20
    variants = []
    for a in sys.argv[5:]:
        lab, _, p = a.partition("=")
        variants.append((lab, p))

    files = sorted(
        x for x in os.listdir(D) if x.upper().startswith("DSC") and x.upper().endswith(".JPG")
    )
    f25 = gray(np.asarray(Image.open(os.path.join(D, files[25])).convert("RGB")))

    G = [("AF export", gray(load(REF)), CANVAS_FROM_REF)]
    for lab, p in variants:
        G.append((lab, gray(load(p)), CANVAS_FROM_OUT))

    print("canvas coordinates; window = [edge-%d, edge-%d] inside the far surface"
          % (inset + width, inset))
    for y in ys:
        seg = f25[y, xlo:xhi]
        ex = xlo + int(np.argmax(np.abs(np.diff(seg)))) + 1
        x0, x1 = ex - inset - width, ex - inset
        line = "y=%4d edge=%4d win=[%4d,%4d] " % (y, ex, x0, x1)
        for lab, g, off in G:
            line += "%18s" % ("%s=%.2f" % (lab, band(g, off, y, x0, x1)))
        # ceiling: best over the original frames, each allowed a +-4 px local shift
        y0 = y - 4
        best, bk = 0.0, -1
        for k, f in enumerate(files):
            a = np.asarray(
                Image.open(os.path.join(D, f)).convert("L").crop((x0 - 6, y0, x1 + 6, y0 + 24 + 8)),
                dtype=np.float64,
            )
            for dy in range(0, 9):
                for dx in range(0, 13):
                    w = a[dy : dy + 24, dx : dx + (x1 - x0)]
                    if w.shape != (24, x1 - x0):
                        continue
                    e = highpass(w)
                    v = float(e[4:-4, 4:-4].mean())
                    if v > best:
                        best, bk = v, k
        line += "%16s" % ("ceil=%.2f@%d" % (best, bk))
        print(line)


if __name__ == "__main__":
    main()
