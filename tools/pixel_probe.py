"""Per-frame focus response at one exact pixel, against its true local sharpness.

`measure_probe` aggregates the response over a patch before ranking frames, and
`argmax_probe` aggregates the per-pixel argmax afterwards; those two statistics
disagree on a textured backdrop next to a big smooth object. This prints the raw
material both are built from: the response at a single pixel and the sharpness of
a small patch around it, frame by frame, so the discrepancy can be attributed.

Usage:  pixel_probe.py <canvas_x> <canvas_y>
"""
import os
import sys
import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
D = os.environ.get("STACK_DIR", "stack")
PREFILTER, RADIUS, STEP = 1, 8, 2
files = sorted(f for f in os.listdir(D) if f.upper().startswith("DSC") and f.upper().endswith(".JPG"))
N = len(files)
cx, cy = int(sys.argv[1]), int(sys.argv[2])
# analysis coordinates = canvas / 2 (the tool's default analysis scale)
ax, ay = cx // 2, cy // 2


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


def sml(g, r=RADIUS):
    if PREFILTER > 0:
        g = box_mean(g, PREFILTER)
    p = np.pad(g, 2, mode="edge")
    c = p[2:-2, 2:-2]
    ml = (np.abs(2 * c - p[2:-2, :-4] - p[2:-2, 4:])
          + np.abs(2 * c - p[:-4, 2:-2] - p[4:, 2:-2]))
    return box_mean(ml, r)


def acu(a, q=0.98):
    g = a[..., 0] * 0.299 + a[..., 1] * 0.587 + a[..., 2] * 0.114
    gy, gx = np.gradient(g.astype(np.float32))
    m = np.hypot(gx, gy).ravel()
    return float(m[m >= np.quantile(m, q)].mean())


resp = np.zeros((N, 3))   # three aggregation radii
sharp = np.zeros((N, 3))  # three patch sizes (half sizes 8, 16, 32)
for i, f in enumerate(files):
    im = Image.open(os.path.join(D, f))
    a = np.asarray(im.convert("RGB"), dtype=np.float32)
    L = im.convert("L")
    g = np.asarray(L.resize((L.width // 2, L.height // 2), Image.BOX), dtype=np.float32) / 255.0
    for j, r in enumerate([RADIUS, 2 * RADIUS, 4 * RADIUS]):
        resp[i, j] = sml(g, r)[ay, ax]
    for j, h in enumerate([8, 16, 32]):
        sharp[i, j] = acu(a[cy - h:cy + h, cx - h:cx + h])
    if (i + 1) % 10 == 0:
        print("  %d/%d" % (i + 1, N))

print()
print("canvas (%d,%d) -> analysis (%d,%d)" % (cx, cy, ax, ay))
for j, r in enumerate([RADIUS, 2 * RADIUS, 4 * RADIUS]):
    o = np.argsort(resp[:, j])[::-1]
    print("SML response, aggregation radius %-3d: argmax %2d  top: %s"
          % (r, o[0], ",".join("%d(%.5f)" % (k, resp[k, j]) for k in o[:6])))
for j, h in enumerate([8, 16, 32]):
    o = np.argsort(sharp[:, j])[::-1]
    print("true patch sharpness, half size %-3d: argmax %2d  top: %s"
          % (h, o[0], ",".join("%d(%.1f)" % (k, sharp[k, j]) for k in o[:6])))
print()
print("frame  response(r8)  response(r32)  sharp(+-16)")
for k in range(N):
    if k % 2 == 0 or k in (33, 31, 35):
        print("%5d  %11.5f  %12.5f  %10.2f" % (k, resp[k, 0], resp[k, 2], sharp[k, 1]))
