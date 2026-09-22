"""Rank fusion variants by real detail inside specific patches.

Uses mid-frequency energy (image minus a blurred copy), which ignores both
single-pixel sensor noise and global shading, so it measures actual structure.
Each patch also reports the raw frame that maximises it, giving the achievable
ceiling for that patch.
"""
import os
import sys
import numpy as np
from PIL import Image, ImageFilter

Image.MAX_IMAGE_PIXELS = None
D = os.environ.get("STACK_DIR", "stack")
D = os.environ.get("STACK_DIR", "stack")
AF_X, AF_Y = 1288, 488
files = sorted(f for f in os.listdir(D) if f.upper().startswith("DSC") and f.upper().endswith(".JPG"))

PATCHES = [
    ("leaf", 1400, 3040, 300),
    ("turntable", 600, 3560, 300),
    ("legs", 1250, 1900, 300),
    ("face", 2000, 700, 300),
]


def mf(a):
    im = Image.fromarray(a.astype(np.uint8))
    low = np.asarray(im.filter(ImageFilter.GaussianBlur(2)), dtype=np.float32)
    d = np.asarray(im, dtype=np.float32) - low
    return float((d ** 2).mean() ** 0.5)


def best_raw(x, y, s):
    best, bi = -1.0, 0
    for i, f in enumerate(files):
        a = np.asarray(Image.open(os.path.join(D, f)).convert("RGB"), dtype=np.float32)
        v = mf(a[AF_Y + y:AF_Y + y + s, AF_X + x:AF_X + x + s])
        if v > best:
            best, bi = v, i
    return best, bi


VARIANTS = sys.argv[1:] or [
    ("AF export", None),
    ("hard pick", r"B:\FocusMerge\out\hard.png"),
    ("fix2 (bracket d15)", r"B:\FocusMerge\out\fix2.png"),
    ("pyramid", r"B:\FocusMerge\out\pyr.png"),
]

af = np.asarray(Image.open(os.path.join(D, "AF导出.png")).convert("RGB"), dtype=np.float32)

print("%-20s" % "variant" + "".join("%12s" % p[0] for p in PATCHES))
hdr = "%-20s" % "best raw frame"
for name, x, y, s in PATCHES:
    v, bi = best_raw(x, y, s)
    hdr += "%12s" % ("%.1f@%d" % (v, bi))
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
            line += "%12s" % "-"
            continue
        line += "%12.3f" % mf(a)
    print(line)
