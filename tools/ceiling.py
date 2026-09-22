"""Establish the sharpness ceiling: gradient energy of each raw frame inside the
matched crop region, versus the reference export.

If no single raw frame reaches the reference's gradient energy, the reference
has been sharpened / locally contrast-enhanced in post production, and chasing
that number with a pure fusion tool is meaningless.
"""
import os
import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
D = os.environ.get("STACK_DIR", "stack")
X0, Y0, CW, CH = 1288, 488, 4275, 4002
files = sorted(f for f in os.listdir(D) if f.upper().startswith("DSC") and f.upper().endswith(".JPG"))


def ge(a):
    g = a[..., 0] * 0.299 + a[..., 1] * 0.587 + a[..., 2] * 0.114
    gy, gx = np.gradient(g.astype(np.float32))
    return float(np.hypot(gx, gy).mean())


af = np.asarray(Image.open(os.path.join(D, "AF导出.png")).convert("RGB"), dtype=np.float32)
print("AF导出          gradient energy %.4f  (this is the target)" % ge(af))

# The matched crop sits inside our aligned canvas, which is offset from the raw
# frame by the alignment shift. Frames were aligned with sub-pixel shifts only,
# so sampling each raw frame at the same crop is accurate enough for this.
vals = []
for f in files:
    im = Image.open(os.path.join(D, f)).convert("RGB")
    a = np.asarray(im.crop((X0, Y0, X0 + CW, Y0 + CH)), dtype=np.float32)
    vals.append(ge(a))

vals = np.array(vals)
print("\nraw frames inside the same crop region:")
print("  min %.4f  median %.4f  max %.4f  (frame %d)" %
      (vals.min(), np.median(vals), vals.max(), int(np.argmax(vals))))
print("  top 5:", np.round(np.sort(vals)[-5:], 4))
print("  best raw frame / AF导出 = %.3f" % (vals.max() / ge(af)))

# blurred reference comparison: is AF simply a sharpened version?
from PIL import ImageFilter
for r in [0.5, 0.8, 1.0]:
    b = np.asarray(Image.fromarray(af.astype(np.uint8)).filter(ImageFilter.GaussianBlur(r)), dtype=np.float32)
    print("  AF blurred by sigma %.1f -> %.4f" % (r, ge(b)))
