"""Reference-free quality metric.

For focus stacking the correct ceiling is not any single frame but the per-pixel
maximum gradient magnitude across the whole stack: whatever the fusion does, it
can never be sharper than "each pixel taken from its sharpest frame".

ideal_gradient_mean = mean over pixels of max over frames of |grad|(frame)

Comparing a fusion result against that number says how much detail the fusion
actually captured, without needing a reference image at all.
"""
import os
import numpy as np
from PIL import Image

# The project directory, so the paths below work on any machine.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

Image.MAX_IMAGE_PIXELS = None
D = os.environ.get("STACK_DIR", "stack")
X0, Y0, CW, CH = 1288, 488, 4275, 4002
files = sorted(f for f in os.listdir(D) if f.upper().startswith("DSC") and f.upper().endswith(".JPG"))


def gray(a):
    return a[..., 0] * 0.299 + a[..., 1] * 0.587 + a[..., 2] * 0.114


def grad(g):
    gy, gx = np.gradient(g.astype(np.float32))
    return np.hypot(gx, gy)


best = None
for i, f in enumerate(files):
    im = Image.open(os.path.join(D, f)).convert("RGB")
    a = np.asarray(im.crop((X0, Y0, X0 + CW, Y0 + CH)), dtype=np.float32)
    g = grad(gray(a))
    best = g if best is None else np.maximum(best, g)
    if (i + 1) % 10 == 0:
        print("  processed %d/%d" % (i + 1, len(files)))

ideal = float(best.mean())
print("\nideal (per-pixel max over the stack) gradient energy = %.4f" % ideal)

af = np.asarray(Image.open(os.path.join(D, "AF导出.png")).convert("RGB"), dtype=np.float32)
af_e = float(grad(gray(af)).mean())
print("AF导出                                             = %.4f  (%.1f%% of ideal)" % (af_e, 100 * af_e / ideal))

for name, p in [("focusmerge (interp)", os.path.join(ROOT, "out/m2.png")),
                ("focusmerge (max)", os.path.join(ROOT, "out/max.png"))]:
    if not os.path.exists(p):
        continue
    o = np.asarray(Image.open(p).convert("RGB"), dtype=np.float32)[Y0:Y0 + CH, X0:X0 + CW]
    e = float(grad(gray(o)).mean())
    print("%-50s = %.4f  (%.1f%% of ideal)" % (name, e, 100 * e / ideal))
