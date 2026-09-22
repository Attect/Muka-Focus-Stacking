"""Which focus-measure variant actually has signal at a given pixel?

The shipped measure is the modified Laplacian with a step of 2 pixels. That step
makes it a band-pass whose response falls to *zero* at the Nyquist frequency, so
content living at the pixel scale — the relief of a plaster backdrop — is nearly
invisible to it. The sharpness of such content does change with focus; a
central-difference gradient (np.gradient, step 1) sees it, which is why the
"truth" column and the measure disagree there.

This compares variants by how well they rank the frames against the true local
sharpness, and by their dynamic range, at chosen points.

Usage:  step_probe.py [name:x:y ...]
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
    p = a.split(":")
    POINTS.append((p[0], int(p[1]), int(p[2])))
if not POINTS:
    POINTS = [("wall", 1560, 1160), ("knee", 1639, 1245), ("face", 2000, 700)]
R = 16
PAD = 80
# (label, kind, step, prefilter)
VARIANTS = [
    ("sml step2 pf1 (current)", "sml", 2, 1),
    ("sml step2 pf0", "sml", 2, 0),
    ("sml step1 pf1", "sml", 1, 1),
    ("sml step1 pf0", "sml", 1, 0),
    ("ten step1 pf0", "ten", 1, 0),
    ("var step1 pf0", "var", 1, 0),
]


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


def response(g, kind, step, prefilter, radius=R):
    if prefilter > 0:
        g = box_mean(g, prefilter)
    if kind == "sml":
        p = np.pad(g, step, mode="edge")
        c = p[step:-step, step:-step]
        ml = (np.abs(2 * c - p[step:-step, :-2 * step] - p[step:-step, 2 * step:])
              + np.abs(2 * c - p[:-2 * step, step:-step] - p[2 * step:, step:-step]))
        return box_mean(ml, radius)
    if kind == "ten":
        p = np.pad(g, 1, mode="edge")
        gx = (p[:-2, 2:] + 2 * p[1:-1, 2:] + p[2:, 2:]
              - p[:-2, :-2] - 2 * p[1:-1, :-2] - p[2:, :-2])
        gy = (p[2:, :-2] + 2 * p[2:, 1:-1] + p[2:, 2:]
              - p[:-2, :-2] - 2 * p[:-2, 1:-1] - p[:-2, 2:])
        return box_mean(gx * gx + gy * gy, radius)
    m = box_mean(g, radius)
    return (box_mean(g * g, radius) - m * m).clip(0.0)


def acu(a, q=0.98):
    g = a[..., 0] * 0.299 + a[..., 1] * 0.587 + a[..., 2] * 0.114
    gy, gx = np.gradient(g.astype(np.float32))
    m = np.hypot(gx, gy).ravel()
    return float(m[m >= np.quantile(m, q)].mean())


resp = {v[0]: {lab: np.zeros(N) for lab, _, _ in POINTS} for v in VARIANTS}
truth = {lab: np.zeros(N) for lab, _, _ in POINTS}
for i, f in enumerate(files):
    im = Image.open(os.path.join(D, f))
    a = np.asarray(im.convert("RGB"), dtype=np.float32)
    L = np.asarray(im.convert("L"), dtype=np.float32) / 255.0
    for lab, x, y in POINTS:
        cx, cy = AF_X + x + CV_X, AF_Y + y + CV_Y
        truth[lab][i] = acu(a[cy - 16:cy + 16, cx - 16:cx + 16])
        g = L[cy - PAD:cy + PAD, cx - PAD:cx + PAD]
        for (name, kind, step, pf) in VARIANTS:
            m = response(g, kind, step, pf)
            resp[name][lab][i] = np.median(m[PAD - 8:PAD + 8, PAD - 8:PAD + 8])
    if (i + 1) % 10 == 0:
        print("  %d/%d" % (i + 1, N))

print()
for lab, _, _ in POINTS:
    t = truth[lab]
    o = np.argsort(t)[::-1]
    print("=== %s ===" % lab)
    print("  truth: argmax %2d  top %s  dynamic range x%.1f"
          % (o[0], ",".join(str(int(v)) for v in o[:5]), t.max() / max(t.min(), 1e-6)))
    for name, *_ in VARIANTS:
        s = resp[name][lab]
        so = np.argsort(s)[::-1]
        rc = np.corrcoef(np.argsort(np.argsort(t)), np.argsort(np.argsort(s)))[0, 1]
        print("  %-26s argmax %2d  rank-corr %+.3f  range x%.2f  top %s"
              % (name, so[0], rc, s.max() / max(s.min(), 1e-9),
                 ",".join(str(int(v)) for v in so[:5])))
    print()
