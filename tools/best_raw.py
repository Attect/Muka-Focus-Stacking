"""Is the reference sharper than any raw frame could be?

For each problem patch, compare the gradient energy of the reference export
against the *best achievable* value: the maximum, over all frames, of the
gradient energy inside that patch. No fusion can exceed that without sharpening
the result, so if the reference sits well above it the reference was sharpened
in post production and matching it is a sharpening question, not a fusion one.
"""
import os
import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
D = os.environ.get("STACK_DIR", "stack")
AF_X, AF_Y = 1288, 488
CV_X, CV_Y = 20, 13
files = sorted(f for f in os.listdir(D) if f.upper().startswith("DSC") and f.upper().endswith(".JPG"))

PATCHES = [
    ("turntable A", 640, 3584),
    ("turntable B", 1152, 3840),
    ("yellow leaf", 1408, 3072),
    ("base front", 1792, 3840),
    ("leaf right", 3200, 3072),
    ("face", 2000, 700),
    ("skirt (textured)", 2100, 1700),
]
SIZE = 160


def ge(a):
    g = a[..., 0] * 0.299 + a[..., 1] * 0.587 + a[..., 2] * 0.114
    gy, gx = np.gradient(g.astype(np.float32))
    return float(np.hypot(gx, gy).mean())


af = np.asarray(Image.open(os.path.join(D, "AF导出.png")).convert("RGB"), dtype=np.float32)
ours = np.asarray(Image.open(r"B:\FocusMerge\out\fix1.png").convert("RGB"), dtype=np.float32)
oc = ours[AF_Y:AF_Y + af.shape[0], AF_X:AF_X + af.shape[1]]

print("loading frames ...")
best = np.zeros(len(PATCHES))
allv = [np.zeros(len(files)) for _ in PATCHES]
for i, f in enumerate(files):
    a = np.asarray(Image.open(os.path.join(D, f)).convert("RGB"), dtype=np.float32)
    for j, (_, x, y) in enumerate(PATCHES):
        px, py = x + CV_X, y + CV_Y
        allv[j][i] = ge(a[py:py + SIZE, px:px + SIZE])
    if (i + 1) % 25 == 0:
        print("  %d/%d" % (i + 1, len(files)))

print("\n%-18s %8s %8s %10s %8s" % ("patch", "AF", "bestraw", "bestframe", "ours"))
for j, (name, x, y) in enumerate(PATCHES):
    v = allv[j]
    afv = ge(af[y:y + SIZE, x:x + SIZE])
    ov = ge(oc[y:y + SIZE, x:x + SIZE])
    print("%-18s %8.3f %8.3f %10d %8.3f   AF/best=%.2fx ours/best=%.2fx"
          % (name, afv, v.max(), int(np.argmax(v)), ov, afv / v.max(), ov / v.max()))
