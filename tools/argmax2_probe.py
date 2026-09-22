"""Per-pixel argmax versus patch-median-response argmax, at full resolution.

Two statistics are built from the same focus responses:

  A. median over the patch of  argmax over frames of R  — what the tool does
  B. argmax over frames of  median over the patch of R  — what several of the
     probe scripts report, and what the true sharpness agrees with

If they disagree, the depth field is not wrong for a lack of information but
because the per-pixel argmax is being decided by noise. Sampling is at full
resolution: the depth map is written at the analysis resolution, which is the
canvas itself when `--analysis-scale 1` is in effect, so the old `/2` used by
several probe scripts pointed somewhere else entirely.

Usage:  argmax2_probe.py [name:af_x:af_y ...]
"""
import os
import sys
import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
D = os.environ.get("STACK_DIR", "stack")
AF_X, AF_Y, CV_X, CV_Y = 1288, 488, 20, 13
SCALE = 1
PREFILTER, RADIUS = 1, 16
files = sorted(f for f in os.listdir(D) if f.upper().startswith("DSC") and f.upper().endswith(".JPG"))
N = len(files)
POINTS = []
for a in sys.argv[1:]:
    p = a.split(":")
    POINTS.append((p[0], int(p[1]), int(p[2])))
if not POINTS:
    POINTS = [("wall", 1560, 1160), ("face", 2000, 700)]
PAD = 80
HALF = 16


def box_mean(a, r):
    if r <= 0:
        return a
    h, w = a.shape
    c = np.pad(np.cumsum(np.cumsum(a, axis=0), axis=1), ((1, 0), (1, 0)))
    ys, xs = np.arange(h), np.arange(w)
    y0, y1 = np.clip(ys - r, 0, h), np.clip(ys + r + 1, 0, h)
    x0, x1 = np.clip(xs - r, 0, w), np.clip(xs + r + 1, 0, w)
    s = c[np.ix_(y1, x1)] - c[np.ix_(y0, x1)] - c[np.ix_(y1, x0)] + c[np.ix_(y0, x0)]
    area = ((y1 - y0)[:, None] * (x1 - x0)[None, :]).astype(np.float32)
    return s / area


def response(g):
    if PREFILTER > 0:
        g = box_mean(g, PREFILTER)
    p = np.pad(g, 2, mode="edge")
    c = p[2:-2, 2:-2]
    ml = (np.abs(2 * c - p[2:-2, :-4] - p[2:-2, 4:])
          + np.abs(2 * c - p[:-4, 2:-2] - p[4:, 2:-2]))
    return box_mean(ml, RADIUS)


best = arg = None
series = {lab: np.zeros(N) for lab, _, _ in POINTS}
for i, f in enumerate(files):
    L = Image.open(os.path.join(D, f)).convert("L")
    if SCALE > 1:
        L = L.resize((L.width // SCALE, L.height // SCALE), Image.BOX)
    g_all = np.asarray(L, dtype=np.float32) / 255.0
    for lab, x, y in POINTS:
        cx, cy = (AF_X + x + CV_X) // SCALE, (AF_Y + y + CV_Y) // SCALE
        m = response(g_all[cy - PAD:cy + PAD, cx - PAD:cx + PAD])
        series[lab][i] = np.median(m[PAD - HALF:PAD + HALF, PAD - HALF:PAD + HALF])
        if best is None:
            best = {l: np.full((2 * PAD, 2 * PAD), -np.inf, np.float32) for l, _, _ in POINTS}
            arg = {l: np.zeros((2 * PAD, 2 * PAD), np.uint8) for l, _, _ in POINTS}
        b, a = best[lab], arg[lab]
        upd = m > b
        b[upd] = m[upd]
        a[upd] = i
    if (i + 1) % 10 == 0:
        print("  %d/%d" % (i + 1, N))

print()
for lab, x, y in POINTS:
    s = series[lab]
    o = np.argsort(s)[::-1]
    a = arg[lab][PAD - HALF:PAD + HALF, PAD - HALF:PAD + HALF]
    print("=== %s (AF %d,%d) ===" % (lab, x, y))
    print("  A  per-pixel argmax, patch median      : %5.1f   histogram %s"
          % (np.median(a), np.round(np.bincount(a.ravel(), minlength=N).reshape(-1, 10).sum(1) / a.size, 2)))
    print("  B  patch-median response, argmax       : %2d     range x%.2f  top %s"
          % (o[0], s.max() / max(s.min(), 1e-9), ",".join(str(int(v)) for v in o[:5])))
