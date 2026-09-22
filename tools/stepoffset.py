"""Locate the depth step and the visible silhouette edge on the same rows.

The visible edge is taken from an *output* image: there the silhouette is sharp,
because it is assembled from the frames in which the model is in focus. Locating
it in an original frame does not work — wherever the far surface is the sharp
one, that frame shows a soft ramp tens of pixels wide whose steepest point is not
the silhouette.

The depth step is searched only within `+-LIMIT` px of that edge. Without the
limit the largest jump of the depth field is often found somewhere inside the
subject, where the field is noisy, and the answer becomes meaningless.

Canvas mapping, verified by maximising the texture correlation between the output
and the reference over 783 textured windows (a sharp peak, correlation 0.19;
every neighbouring offset collapses to ~0):

    canvas = output + (12, 18) = reference export + (506, 1304)

`(12, 18)` is the tool's own `output window ... at (x, y)`.

    python tools/stepoffset.py DEPTH.png OUROUT.png Y0 Y1 X0 X1 [STEP]
"""
import os
import sys

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from patchlib import load  # noqa: E402

CANVAS_FROM_OUT = (12, 18)
N = 49.0
LIMIT = 30


def main():
    depth_path = sys.argv[1]
    out_path = sys.argv[2] or None
    y0, y1, x0, x1 = (int(v) for v in sys.argv[3:7])
    step = int(sys.argv[7]) if len(sys.argv) > 7 else 10

    dz = np.asarray(Image.open(depth_path), dtype=np.float32) / 65535.0 * N
    img = load(out_path) if out_path else None
    off = CANVAS_FROM_OUT
    g = None
    if img is not None:
        g = (0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]).astype(np.float64)

    print("%6s %8s %9s %8s   %s" % ("y", "edge", "step", "off", "crossing"))
    offs = []
    for y in range(y0, y1 + 1, step):
        row = g[y - off[0], x0 - off[1] : x1 - off[1]]
        d = np.abs(np.diff(row))
        ei = int(np.argmax(d))
        if d[ei] < 25:
            continue
        ex = x0 + ei + 1
        lo, hi = max(x0, ex - LIMIT), min(x1, ex + LIMIT)
        ds = dz[y, lo:hi]
        si = int(np.argmax(np.abs(np.diff(ds))))
        sx = lo + si + 1
        left, right = dz[y, sx - 12], dz[y, sx + 12]
        cross = "wall->model" if left > right else "model->wall"
        offs.append(sx - ex)
        print("%6d %8d %9d %+8d   %s (%.1f -> %.1f)" % (y, ex, sx, sx - ex, cross, left, right))
    o = np.array(offs) if offs else np.array([0.0])
    print()
    print("offset (positive = step inside the model): median %+.1f  mean %+.1f  |median| %.1f  IQR %.1f  n=%d"
          % (np.median(o), o.mean(), abs(np.median(o)), np.percentile(o, 75) - np.percentile(o, 25), len(o)))


if __name__ == "__main__":
    main()
