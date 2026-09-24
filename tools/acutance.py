"""Acutance-based patch evaluation.

On smooth surfaces every plain "detail energy" metric is dominated by noise, so
it cannot tell a sharp leaf from a soft one. What the eye reads as sharpness
there is the steepness of the silhouette: a defocused edge spreads over more
pixels and therefore has a lower peak gradient for the same contrast.

`acutance` = mean gradient magnitude over the strongest 0.5% of edge pixels.
"""
import os
import sys
import numpy as np
from PIL import Image

# The project directory, so the paths below work on any machine.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

Image.MAX_IMAGE_PIXELS = None
D = os.environ.get("STACK_DIR", "stack")
AF_X, AF_Y = 1288, 488
files = sorted(f for f in os.listdir(D) if f.upper().startswith("DSC") and f.upper().endswith(".JPG"))

PATCHES = [
    ("leaf", 1400, 3040, 300),
    ("turntable", 600, 3560, 300),
    ("legs", 1250, 1900, 300),
    ("face", 2000, 700, 300),
]


def gray(a):
    return a[..., 0] * 0.299 + a[..., 1] * 0.587 + a[..., 2] * 0.114


def acutance(a, q=0.995):
    g = gray(a).astype(np.float32)
    gy, gx = np.gradient(g)
    m = np.hypot(gx, gy).ravel()
    return float(m[m >= np.quantile(m, q)].mean())


def best_raw(x, y, s):
    best, bi = -1.0, 0
    for i, f in enumerate(files):
        a = np.asarray(Image.open(os.path.join(D, f)).convert("RGB"), dtype=np.float32)
        v = acutance(a[AF_Y + y:AF_Y + y + s, AF_X + x:AF_X + x + s])
        if v > best:
            best, bi = v, i
    return best, bi


VARIANTS = [
    ("AF export", None),
    ("hard pick", os.path.join(ROOT, "out/hard.png")),
    ("bracket denoise .15", os.path.join(ROOT, "out/fix2.png")),
    ("bracket denoise 0", os.path.join(ROOT, "out/nodenoise.png")),
    ("pyramid denoise .15", os.path.join(ROOT, "out/pyr.png")),
    ("pyramid denoise 0", os.path.join(ROOT, "out/pyr_nodenoise.png")),
]

af = np.asarray(Image.open(os.path.join(D, "AF导出.png")).convert("RGB"), dtype=np.float32)

print("%-20s" % "variant" + "".join("%13s" % p[0] for p in PATCHES))
hdr = "%-20s" % "BEST RAW FRAME"
for name, x, y, s in PATCHES:
    v, bi = best_raw(x, y, s)
    hdr += "%13s" % ("%.2f@%d" % (v, bi))
print(hdr)
print()

for label, path in VARIANTS:
    line = "%-20s" % label
    for name, x, y, s in PATCHES:
        if path is None:
            a = af[y:y + s, x:x + s]
        elif os.path.exists(path):
            o = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32)
            a = o[AF_Y + y:AF_Y + y + s, AF_X + x:AF_X + x + s]
        else:
            line += "%13s" % "-"
            continue
        line += "%13.2f" % acutance(a)
    print(line)
