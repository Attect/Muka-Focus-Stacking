"""Is the focus-breathing scale real, or is the estimator matching blur instead?

The tool estimates each frame's scale by maximising normalised cross-correlation
of high-passed *gradient magnitude* features. Gradient magnitude is precisely the
thing that changes with focus, so a frame that is softer than the reference can
score better by being rescaled to look softer still — the estimate then tracks
focus rather than geometry.

This script decides the question by measuring the scale twice: once the way the
tool does it, and once after low-passing every frame hard enough that the focus
difference between them is gone, leaving only geometry.
"""
import os
import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
D = os.environ.get("STACK_DIR", "stack")
files = sorted(f for f in os.listdir(D) if f.upper().startswith("DSC") and f.upper().endswith(".JPG"))
WORK = 900
REF = 25
TEST = [0, 6, 12, 18, 31, 38, 44, 49]
SCALES = np.arange(0.985, 1.0151, 0.001)


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


def load_small(i):
    im = Image.open(os.path.join(D, files[i])).convert("L")
    f = WORK / max(im.size)
    return np.asarray(im.resize((round(im.width * f), round(im.height * f)), Image.BOX),
                      dtype=np.float32) / 255.0


def gradient(a):
    p = np.pad(a, 1, mode="edge")
    gx = (p[:-2, 2:] + 2 * p[1:-1, 2:] + p[2:, 2:]
          - p[:-2, :-2] - 2 * p[1:-1, :-2] - p[2:, :-2])
    gy = (p[2:, :-2] + 2 * p[2:, 1:-1] + p[2:, 2:]
          - p[:-2, :-2] - 2 * p[:-2, 1:-1] - p[:-2, 2:])
    return np.hypot(gx, gy)


def rescale(a, s):
    im = Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8))
    w, h = im.size
    cx, cy = w / 2.0, h / 2.0
    out = im.transform((w, h), Image.AFFINE,
                       (1.0 / s, 0.0, cx * (1.0 - 1.0 / s),
                        0.0, 1.0 / s, cy * (1.0 - 1.0 / s)),
                       resample=Image.BICUBIC, fillcolor=0)
    return np.asarray(out, dtype=np.float32) / 255.0


def centre(a, frac=0.96):
    h, w = a.shape
    ch, cw = int(h * frac), int(w * frac)
    y0, x0 = (h - ch) // 2, (w - cw) // 2
    return a[y0:y0 + ch, x0:x0 + cw]


def align_shape(a, shape):
    """Crop `a` to exactly `shape`, centred. Guards against rounding drift."""
    ch, cw = shape
    y0, x0 = (a.shape[0] - ch) // 2, (a.shape[1] - cw) // 2
    return a[y0:y0 + ch, x0:x0 + cw]


def ncc(a, b):
    a = a - a.mean()
    b = b - b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 0 else 0.0


print("work size %d, reference frame %d, scale step 0.001\n" % (WORK, REF))
raw = {i: load_small(i) for i in [REF] + TEST}
print("frame-to-frame contrast (gradient energy at work res):")
for i in [REF] + TEST:
    print("   frame %2d  %.5f" % (i, float(gradient(raw[i]).mean())))
print()

for mode in ["gradient high-pass (what the tool uses)", "no blur", "blur r2", "blur r4", "blur r8"]:
    feats = {}
    for i in [REF] + TEST:
        g = raw[i]
        if mode.startswith("blur"):
            r = int(mode.split("r")[1])
            g = box_mean(g, r)
        elif mode.startswith("gradient"):
            gm = gradient(g)
            g = gm - box_mean(gm, 4)
        feats[i] = g
    ref = centre(feats[REF])
    shape = ref.shape
    line = "%-36s" % mode
    detail = []
    for i in TEST:
        best, bestv = None, -9.9
        for s in SCALES:
            v = ncc(ref, align_shape(centre(rescale(feats[i], s)), shape))
            if v > bestv:
                bestv, best = v, s
        detail.append((i, best, bestv))
    print(line, " ".join("f%d:%.3f(%.2f)" % (i, s, v) for i, s, v in detail))
