"""Test whether focus breathing introduces a scale change across the stack."""
import os
import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
D = os.environ.get("STACK_DIR", "stack")
files = sorted(f for f in os.listdir(D) if f.upper().startswith("DSC") and f.upper().endswith(".JPG"))


def load(name, tw=1200):
    im = Image.open(os.path.join(D, name)).convert("L")
    h = int(im.height * tw / im.width)
    return np.asarray(im.resize((tw, h), Image.LANCZOS), dtype=np.float32)


def grad(g):
    gy, gx = np.gradient(g)
    return np.hypot(gx, gy)


def ncc(a, b):
    a = a - a.mean()
    b = b - b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 0 else 0.0


def zoom(img, s):
    """Scale about the centre by factor s."""
    h, w = img.shape
    ih, iw = int(round(h * s)), int(round(w * s))
    im = Image.fromarray(img)
    im = im.resize((iw, ih), Image.BICUBIC)
    # centre crop to original size
    x0 = (iw - w) // 2
    y0 = (ih - h) // 2
    return np.asarray(im.crop((x0, y0, x0 + w, y0 + h)), dtype=np.float32)


REF_IDX = 24
ref = grad(load(files[REF_IDX]))
print("reference frame", files[REF_IDX])

for i in [0, 8, 16, 32, 40, 49]:
    g = load(files[i])
    best = (None, -2)
    for s in np.arange(0.985, 1.0151, 0.0025):
        v = ncc(ref, grad(zoom(g, s)))
        if v > best[1]:
            best = (s, v)
    base = ncc(ref, grad(g))
    print("  frame %2d vs ref: best scale=%.4f ncc=%.5f | no-scale ncc=%.5f  delta=%+.5f"
          % (i, best[0], best[1], base, best[1] - base))
