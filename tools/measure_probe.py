"""Which focus measure actually tracks true sharpness?

For every frame it computes the patch's real acutance (the truth, judged on the
full-resolution image) alongside several candidate focus measures, then reports
how well each candidate ranks the frames. A measure with real signal peaks at the
same frame as the acutance; one with none does not.

Motivation: at a dark, low-contrast surface the shipped measure (SML at half
resolution, prefilter 1, radius 8) comes out essentially flat across the stack,
so its argmax is decided by whatever small systematic difference exists between
frames — exposure drift is a prime suspect, because SML is linear in intensity
and this stack drifts about 1.7%.

Usage:  measure_probe.py [name:x:y ...]
"""
import os
import sys
import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
D = os.environ.get("STACK_DIR", "stack")
AF_X, AF_Y, CV_X, CV_Y = 1288, 488, 20, 13
files = sorted(f for f in os.listdir(D) if f.upper().startswith("DSC") and f.upper().endswith(".JPG"))
N = len(files)

POINTS = []
for a in sys.argv[1:]:
    parts = a.split(":")
    POINTS.append((parts[0], int(parts[1]), int(parts[2])))
if not POINTS:
    POINTS = [("knee-edge", 1639, 1245)]
HALF = 24

# (name, scale, prefilter, radius, norm, kind)
VARIANTS = [
    ("sml  half pf1 r8  (current)", 2, 1, 8, "none", "sml"),
    ("sml  half pf0 r8", 2, 0, 8, "none", "sml"),
    ("sml  half pf0 r8  mean-norm", 2, 0, 8, "mean", "sml"),
    ("sml  half pf0 r8  local-norm", 2, 0, 8, "local", "sml"),
    ("sml  half pf0 r3", 2, 0, 3, "none", "sml"),
    ("sml  full pf0 r16", 1, 0, 16, "none", "sml"),
    ("sml  full pf2 r16", 1, 2, 16, "none", "sml"),
    ("ten  half pf1 r8", 2, 1, 8, "none", "ten"),
    ("ten  half pf0 r8  local-norm", 2, 0, 8, "local", "ten"),
    ("band half pf0 r8", 2, 0, 8, "none", "band"),
    ("var  half pf0 r8", 2, 0, 8, "none", "var"),
]


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


def measure(g, kind, radius):
    if kind == "sml":
        p = np.pad(g, 2, mode="edge")
        c = p[2:-2, 2:-2]
        ml = (np.abs(2 * c - p[2:-2, :-4] - p[2:-2, 4:])
              + np.abs(2 * c - p[:-4, 2:-2] - p[4:, 2:-2]))
        return box_mean(ml, radius)
    if kind == "ten":
        p = np.pad(g, 1, mode="edge")
        gx = (p[:-2, 2:] + 2 * p[1:-1, 2:] + p[2:, 2:]
              - p[:-2, :-2] - 2 * p[1:-1, :-2] - p[2:, :-2])
        gy = (p[2:, :-2] + 2 * p[2:, 1:-1] + p[2:, 2:]
              - p[:-2, :-2] - 2 * p[:-2, 1:-1] - p[:-2, 2:])
        return box_mean(gx * gx + gy * gy, radius)
    if kind == "var":
        m = box_mean(g, radius)
        return (box_mean(g * g, radius) - m * m).clip(0.0)
    if kind == "band":
        return box_mean(np.abs(g - box_mean(g, radius)), radius)
    raise ValueError(kind)


def acutance(a, q=0.99):
    g = a[..., 0] * 0.299 + a[..., 1] * 0.587 + a[..., 2] * 0.114
    gy, gx = np.gradient(g.astype(np.float32))
    m = np.hypot(gx, gy).ravel()
    return float(m[m >= np.quantile(m, q)].mean())


truth = {lab: np.zeros(N) for lab, _, _ in POINTS}
scores = {name: {lab: np.zeros(N) for lab, _, _ in POINTS} for name, *_ in VARIANTS}
gray_mean = np.zeros(N)

for i, f in enumerate(files):
    im = Image.open(os.path.join(D, f)).convert("RGB")
    a = np.asarray(im, dtype=np.float32)
    L = im.convert("L")
    g_half = np.asarray(L.resize((L.width // 2, L.height // 2), Image.BOX), dtype=np.float32) / 255.0
    g_full = np.asarray(L, dtype=np.float32) / 255.0
    gray_mean[i] = g_full.mean()
    for lab, x, y in POINTS:
        cvx, cvy = AF_X + x + CV_X, AF_Y + y + CV_Y
        truth[lab][i] = acutance(a[cvy - HALF:cvy + HALF, cvx - HALF:cvx + HALF])
        hx, hy = cvx // 2, cvy // 2
        q = HALF // 2 + 60
        for (name, scale, pf, radius, norm, kind) in VARIANTS:
            if scale == 2:
                g = g_half[hy - q:hy + q, hx - q:hx + q]
                cxc, cyc = q, q
            else:
                g = g_full[cvy - 140:cvy + 140, cvx - 140:cvx + 140]
                cxc, cyc = 140, 140
            if pf > 0:
                g = box_mean(g, pf)
            if norm == "mean":
                g = g / (g.mean() + 1e-9)
            elif norm == "local":
                g = g / (box_mean(g, 16) + 1e-6)
            m = measure(g, kind, radius)
            sc = HALF // scale
            scores[name][lab][i] = np.median(m[cyc - sc:cyc + sc, cxc - sc:cxc + sc])
    if (i + 1) % 10 == 0:
        print("  %d/%d" % (i + 1, N))

print()
print("frame mean brightness: %.4f .. %.4f  (drift %.2f%%)"
      % (gray_mean.min(), gray_mean.max(), 100 * (gray_mean.max() / gray_mean.min() - 1)))
print("corr(frame index, mean brightness) = %.3f"
      % np.corrcoef(np.arange(N), gray_mean)[0, 1])
print()

for lab, _, _ in POINTS:
    t = truth[lab]
    print("=== %s  (%d,%d) ===" % (lab, dict((l, x) for l, x, _ in POINTS)[lab],
                                   dict((l, y) for l, _, y in POINTS)[lab]))
    order = np.argsort(t)[::-1]
    print("  truth acutance top frames: %s" % ", ".join("%d(%.1f)" % (i, t[i]) for i in order[:6]))
    for name, *_ in VARIANTS:
        s = scores[name][lab]
        o = np.argsort(s)[::-1]
        rc = np.corrcoef(np.argsort(np.argsort(t)), np.argsort(np.argsort(s)))[0, 1]
        print("  %-30s argmax %2d  rank-corr %+.3f  top: %s"
              % (name, o[0], rc, ",".join(str(int(v)) for v in o[:5])))
    print()
