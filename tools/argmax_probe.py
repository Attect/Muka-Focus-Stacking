"""Replicate the tool's per-pixel depth estimation and see where the argmax lands.

The tool takes, for every pixel, the frame with the strongest focus response.
That is a *maximum statistic*: where the response is weak the winner is decided
by whatever small systematic difference exists between frames. This script
measures that directly by building the argmax map twice -- once on the raw focus
response and once on a per-frame brightness-normalised one -- and comparing the
result with the tool's own saved depth map.

Usage:  argmax_probe.py [name:x:y ...]
"""
import os
import sys
import numpy as np
from PIL import Image

# The project directory, so the paths below work on any machine.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

Image.MAX_IMAGE_PIXELS = None
D = os.environ.get("STACK_DIR", "stack")
AF_X, AF_Y, CV_X, CV_Y = 1288, 488, 20, 13
PREFILTER, RADIUS = 1, 8
files = sorted(f for f in os.listdir(D) if f.upper().startswith("DSC") and f.upper().endswith(".JPG"))
N = len(files)

POINTS = []
for a in sys.argv[1:]:
    p = a.split(":")
    POINTS.append((p[0], int(p[1]), int(p[2])))
if not POINTS:
    POINTS = [("knee-edge", 1639, 1245)]
HALF = 20  # patch half size, reference pixels

depth_path = os.path.join(ROOT, "out/depth_fix3.png")
dw, dh = Image.open(depth_path).size
out_w, out_h = Image.open(os.path.join(ROOT, "out/cur.png")).size
print("depth map %dx%d   output %dx%d   canvas would be %dx%d"
      % (dw, dh, out_w, out_h, 7008 // 2, 4672 // 2))
# The depth map covers the output rect at analysis resolution.
ORIGIN = "output" if dw == out_w // 2 else "canvas"
print("depth map origin: %s" % ORIGIN)


def box_mean(a, r):
    if r <= 0:
        return a
    h, w = a.shape
    c = np.pad(np.cumsum(np.cumsum(a, axis=0), axis=1), ((1, 0), (1, 0)))
    ys, xs = np.arange(h), np.arange(w)
    y0i, y1i = np.clip(ys - r, 0, h), np.clip(ys + r + 1, 0, h)
    x0i, x1i = np.clip(xs - r, 0, w), np.clip(xs + r + 1, 0, w)
    s = c[np.ix_(y1i, x1i)] - c[np.ix_(y0i, x1i)] - c[np.ix_(y1i, x0i)] + c[np.ix_(y0i, x0i)]
    area = ((y1i - y0i)[:, None] * (x1i - x0i)[None, :]).astype(np.float32)
    return s / area


def sml(g):
    if PREFILTER > 0:
        g = box_mean(g, PREFILTER)
    p = np.pad(g, 2, mode="edge")
    c = p[2:-2, 2:-2]
    ml = (np.abs(2 * c - p[2:-2, :-4] - p[2:-2, 4:])
          + np.abs(2 * c - p[:-4, 2:-2] - p[4:, 2:-2]))
    return box_mean(ml, RADIUS)


best_raw = None
best_norm = None
amp = np.zeros((len(POINTS), N))
mean_lum = np.zeros(N)

for i, f in enumerate(files):
    L = Image.open(os.path.join(D, f)).convert("L")
    g = np.asarray(L.resize((L.width // 2, L.height // 2), Image.BOX), dtype=np.float32) / 255.0
    mean_lum[i] = g.mean()
    m_raw = sml(g)
    m_norm = m_raw / (g.mean() + 1e-9)
    for pi, (lab, x, y) in enumerate(POINTS):
        if ORIGIN == "output":
            ax, ay = (AF_X + x + CV_X - 20) // 2, (AF_Y + y + CV_Y - 13) // 2
        else:
            ax, ay = (AF_X + x + CV_X) // 2, (AF_Y + y + CV_Y) // 2
        amp[pi, i] = np.median(m_norm[ay - HALF:ay + HALF, ax - HALF:ax + HALF])
    if best_raw is None:
        best_raw = np.full(g.shape, -np.inf, np.float32)
        best_norm = np.full(g.shape, -np.inf, np.float32)
        arg_raw = np.zeros(g.shape, np.uint8)
        arg_norm = np.zeros(g.shape, np.uint8)
    upd = m_raw > best_raw
    best_raw[upd] = m_raw[upd]
    arg_raw[upd] = i
    upd = m_norm > best_norm
    best_norm[upd] = m_norm[upd]
    arg_norm[upd] = i
    if (i + 1) % 10 == 0:
        print("  %d/%d" % (i + 1, N))

print("\nmean luminance %.4f .. %.4f, drift %.2f%%, corr with frame index %.3f"
      % (mean_lum.min(), mean_lum.max(), 100 * (mean_lum.max() / mean_lum.min() - 1),
         np.corrcoef(np.arange(N), mean_lum)[0, 1]))

tool_depth = np.asarray(Image.open(depth_path), dtype=np.float32) / 65535.0 * (N - 1)

print()
for pi, (lab, x, y) in enumerate(POINTS):
    if ORIGIN == "output":
        ax, ay = (AF_X + x + CV_X - 20) // 2, (AF_Y + y + CV_Y - 13) // 2
    else:
        ax, ay = (AF_X + x + CV_X) // 2, (AF_Y + y + CV_Y) // 2
    sl = (slice(ay - HALF, ay + HALF), slice(ax - HALF, ax + HALF))
    print("=== %s (%d,%d)  analysis (%d,%d) ===" % (lab, x, y, ax, ay))
    print("  patch-median response argmax : raw %2d   normalised %2d"
          % (np.argmax(amp[pi]), np.argmax(amp[pi])))
    print("  per-pixel argmax patch median: raw %.1f   normalised %.1f"
          % (np.median(arg_raw[sl]), np.median(arg_norm[sl])))
    print("  tool's saved depth           : %.1f" % np.median(tool_depth[sl]))
    h = np.bincount(arg_raw[sl].ravel(), minlength=N)
    print("  argmax histogram (raw), 10-frame bins: %s"
          % np.round(h.reshape(-1, 10).sum(1) / h.sum(), 3))
    print()
