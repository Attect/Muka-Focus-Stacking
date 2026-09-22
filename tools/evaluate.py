"""Evaluate an output against the reference export on the criteria that matter.

Four independent checks, because each one alone has misled this project at least
once:

  band       texture energy of the far surface at 8/12/18 px from the silhouette,
             plus the per-frame ceiling at the same place (the physical maximum)
  black      share of pixels below L=5 and L=10 — the bug-6 artefact. The floor
             is the per-pixel minimum over the stack
  sharp      gradient energy in six named regions (patchlib)
  global     mean ratio of high-frequency energy to the reference export

    python tools/evaluate.py Y0 Y1 X0 X1 [label=path ...]
"""
import os
import sys

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from patchlib import REF, load  # noqa: E402

D = os.environ.get("STACK_DIR", "stack")
CANVAS_FROM_OUT = (12, 18)
CANVAS_FROM_REF = (506, 1304)
DISTS = [8, 12, 18, 26]
WIN = 12


def gray(x):
    return (0.299 * x[..., 0] + 0.587 * x[..., 1] + 0.114 * x[..., 2]).astype(np.float64)


def highpass(m, k=3):
    p = np.pad(m, ((k + 1, k), (k + 1, k)), mode="edge")
    c = np.cumsum(np.cumsum(p, 0, dtype=np.float64), 1, dtype=np.float64)
    kk = 2 * k + 1
    return np.abs(m - (c[kk:, kk:] - c[:-kk, kk:] - c[kk:, :-kk] + c[:-kk, :-kk]) / float(kk * kk))


def band_energy(g, off, ys, xs):
    vals = []
    for y, x0, x1 in zip(ys, xs[:, 0], xs[:, 1]):
        w = g[y - off[0] - 9 : y - off[0] + 9, x0 - off[1] : x1 - off[1]]
        if w.shape != (18, x1 - x0):
            continue
        e = highpass(w)
        vals.append(float(e[3:-3, 3:-3].mean()))
    return float(np.mean(vals)) if vals else float("nan")


def main():
    y0, y1, x0, x1 = (int(v) for v in sys.argv[1:5])
    variants = []
    for a in sys.argv[5:]:
        lab, _, p = a.partition("=")
        variants.append((lab, p))

    ours = gray(load(variants[0][1]))
    edges, ys = [], []
    for y in range(y0, y1 + 1):
        row = ours[y - CANVAS_FROM_OUT[0], x0 - CANVAS_FROM_OUT[1] : x1 - CANVAS_FROM_OUT[1]]
        d = np.abs(np.diff(row))
        i = int(np.argmax(d))
        if d[i] < 25:
            continue
        edges.append(x0 + i + 1)
        ys.append(y)
    edges = np.array(edges)
    print("band rows %d, edge x %d..%d (located in %s)" % (len(ys), edges.min(), edges.max(), variants[0][0]))

    files = sorted(x for x in os.listdir(D) if x.upper().startswith("DSC") and x.upper().endswith(".JPG"))
    loaded = {f: np.asarray(Image.open(os.path.join(D, f)).convert("L"), dtype=np.float64) for f in files}
    mn = np.minimum.reduce(list(loaded.values()))

    print()
    print("%-18s" % "band dist" + "".join("%14s" % d for d in DISTS) + "%14s" % "ceiling")
    rows = []
    for d in DISTS:
        xs = np.stack([edges - d - WIN, edges - d], 1)
        line = "%-18s" % ("  %d px" % d)
        for lab, p in variants:
            g = gray(load(p))
            line += "%14.2f" % band_energy(g, CANVAS_FROM_OUT, ys, xs)
        best = 0.0
        for f in files:
            v = band_energy(loaded[f], (0, 0), ys, xs)
            best = max(best, v)
        line += "%14.2f" % best
        rows.append((d, line))
    for _, line in rows:
        print(line)

    print()
    print("%-18s %12s %12s %14s" % ("variant", "L<5", "L<10", "mean L"))
    for lab, p in variants:
        g = gray(load(p))
        print("%-18s %11.4f%% %11.4f%% %14.2f" % (lab, 100 * (g < 5).mean(), 100 * (g < 10).mean(), g.mean()))
    print("%-18s %11.4f%% %11.4f%% %14.2f" % ("per-pixel stack min", 100 * (mn < 5).mean(), 100 * (mn < 10).mean(), mn.mean()))
    gr = gray(load(REF))
    print("%-18s %11.4f%% %11.4f%% %14.2f" % ("AF export", 100 * (gr < 5).mean(), 100 * (gr < 10).mean(), gr.mean()))

    print()
    print("regional gradient energy (patchlib):")
    cmd = 'python "%s" ' % os.path.join(HERE, "patchlib.py") + ' '.join('"%s=%s"' % (l, p) for l, p in variants)
    print("  run: " + cmd)


if __name__ == "__main__":
    main()
