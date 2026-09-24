"""Diagnose why the comparison metrics differ: colour handling of the 16-bit
PNG, and where exactly the detail gap comes from."""
import os
import numpy as np
from PIL import Image

# The project directory, so the paths below work on any machine.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

Image.MAX_IMAGE_PIXELS = None
D = os.environ.get("STACK_DIR", "stack")

af = np.asarray(Image.open(D + r"\AF导出.png").convert("RGB"), dtype=np.float32)
png = np.asarray(Image.open(os.path.join(ROOT, "out/merged_pyramid.png")).convert("RGB"), dtype=np.float32)
prev = np.asarray(Image.open(os.path.join(ROOT, "out/merged_pyramid_prev.jpg")).convert("RGB"), dtype=np.float32)
print("shapes af", af.shape, "png", png.shape, "prev", prev.shape)
print("af   mean per ch", af.reshape(-1, 3).mean(0).round(2), "std", af.std().round(2))
print("png  mean per ch", png.reshape(-1, 3).mean(0).round(2), "std", png.std().round(2))
print("prev mean per ch", prev.reshape(-1, 3).mean(0).round(2), "std", prev.std().round(2))

# The PNG stats are over the whole frame including white background; compare
# against the 8-bit preview of the same run instead (downscaled, but colour
# handling identical).
x0, y0 = 1288, 496
cw, ch = af.shape[1], af.shape[0]
ours = png[y0:y0 + ch, x0:x0 + cw]
print("\nmatched crop")
print("  ours mean", ours.reshape(-1, 3).mean(0).round(2), "std", ours.std().round(2))
print("  af   mean", af.reshape(-1, 3).mean(0).round(2), "std", af.std().round(2))

# Saturation / hue comparison: convert to HSV and compare the S channel
def hsv(a):
    im = Image.fromarray(a.astype(np.uint8)).convert("HSV")
    return np.asarray(im, dtype=np.float32)


ho, ha = hsv(ours), hsv(af)
print("\nHSV comparison (ours vs af)")
for i, n in enumerate(["H", "S", "V"]):
    print("  %s mean %.2f vs %.2f   std %.2f vs %.2f" % (n, ho[..., i].mean(), ha[..., i].mean(),
                                                          ho[..., i].std(), ha[..., i].std()))

# Where is the detail difference concentrated? Row/column profile of gradient
def grad(g):
    gy, gx = np.gradient(g)
    return np.hypot(gx, gy)


go = grad(ours.mean(2))
ga = grad(af.mean(2))
print("\ngradient energy ours %.4f af %.4f ratio %.3f" % (go.mean(), ga.mean(), go.mean() / ga.mean()))
B = 400
r = (go[:go.shape[0] // B * B, :go.shape[1] // B * B].reshape(-1, B, go.shape[1] // B, B).mean((1, 3)) /
     (ga[:ga.shape[0] // B * B, :ga.shape[1] // B * B].reshape(-1, B, ga.shape[1] // B, B).mean((1, 3)) + 1e-6))
print("block ratio: min %.2f p25 %.2f median %.2f p75 %.2f max %.2f" %
      (r.min(), np.percentile(r, 25), np.median(r), np.percentile(r, 75), r.max()))

# High frequency content: difference between the image and a blurred version
from PIL import ImageFilter
def hf_energy(a):
    im = Image.fromarray(a.astype(np.uint8))
    blur = im.filter(ImageFilter.GaussianBlur(2))
    d = np.asarray(im, dtype=np.float32) - np.asarray(blur, dtype=np.float32)
    return float((d ** 2).mean() ** 0.5)


for name, arr in [("ours", ours), ("af", af)]:
    print("%-6s high-frequency RMS %.3f" % (name, hf_energy(arr.mean(2))))
