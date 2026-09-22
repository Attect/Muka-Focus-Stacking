"""Compare a wall window against the original frames, with no window drift.

Two traps this avoids, both of which produced wrong conclusions earlier:

* the window is placed strictly inside the wall, at a fixed distance from the
  edge as located in the *output* (whose silhouette is sharp). Locating it in an
  original frame fails, because at the frames where the wall is focused the wall
  itself is sharp but the frame where it is not has a soft ramp whose steepest
  point sits ~25 px away;
* the frames are measured as they are, without a shift search. Allowing shifts
  lets a window slide onto the edge and import texture that is not there.

    python tools/wallwin.py Y0 Y1 X0 X1 [label=path ...]
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


def energy(g, off, y0, y1, x0, x1):
    w = g[y0 - off[0] : y1 - off[0], x0 - off[1] : x1 - off[1]]
    e = highpass(w)
    return float(e[4:-4, 4:-4].mean())


def main():
    y0, y1, x0, x1 = (int(v) for v in sys.argv[1:5])
    variants = []
    for a in sys.argv[5:]:
        lab, _, p = a.partition("=")
        variants.append((lab, p))
    print("canvas window y %d..%d  x %d..%d" % (y0, y1, x0, x1))
    for lab, p in [("AF export", REF)] + variants:
        g = gray(load(p))
        off = CANVAS_FROM_REF if p == REF else CANVAS_FROM_OUT
        print("  %-22s %.2f" % (lab, energy(g, off, y0, y1, x0, x1)))

    files = sorted(
        x for x in os.listdir(D) if x.upper().startswith("DSC") and x.upper().endswith(".JPG")
    )
    vals = []
    for f in files:
        a = np.asarray(Image.open(os.path.join(D, f)).convert("L").crop((x0, y0, x1, y1)), dtype=np.float64)
        e = highpass(a)
        vals.append(float(e[4:-4, 4:-4].mean()))
    vals = np.array(vals)
    order = np.argsort(-vals)[:8]
    print()
    print("original frames, no shift: max %.2f @%d | median %.2f" % (vals.max(), int(np.argmax(vals)), np.median(vals)))
    print("  top8: " + ", ".join("%d:%.2f" % (int(i), vals[i]) for i in order))


if __name__ == "__main__":
    main()
