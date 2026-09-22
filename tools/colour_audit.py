"""Colour audit: is the difference in the background between our fusion and the
reference export real, or did the reference get colour graded?

Samples the same physical patch of the backdrop in raw frames, our output and
the reference export.
"""
import os
import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
D = os.environ.get("STACK_DIR", "stack")
X0, Y0 = 1288, 488
CW, CH = 4275, 4002


def patch(a, x, y, s=200):
    p = a[y:y + s, x:x + s].reshape(-1, 3)
    return p.mean(0), p.std(0)


af = np.asarray(Image.open(os.path.join(D, "AF导出.png")).convert("RGB"), dtype=np.float32)

targets = {
    "backdrop upper-left": (150, 150),
    "backdrop upper-right": (CW - 350, 150),
    "grass area": (CW // 2 - 700, CH - 400),
}
print("=== wall / backdrop patches (mean per channel, std) ===\n")
for label, (x, y) in targets.items():
    m, s = patch(af, x, y)
    print("%-22s %-12s mean %s std %s" % (label, "AF导出", np.round(m, 2), np.round(s, 2)))
    for f in ["DSC03537.JPG", "DSC03561.JPG", "DSC03586.JPG"]:
        a = np.asarray(Image.open(os.path.join(D, f)).convert("RGB"), dtype=np.float32)
        a = a[Y0:Y0 + CH, X0:X0 + CW]
        m, s = patch(a, x, y)
        print("%-22s %-12s mean %s std %s" % ("", f[:11], np.round(m, 2), np.round(s, 2)))
    for name, p in [("ours max", "out/max.png"), ("ours bracket", "out/br.png")]:
        if not os.path.exists(p):
            continue
        o = np.asarray(Image.open(p).convert("RGB"), dtype=np.float32)[Y0:Y0 + CH, X0:X0 + CW]
        m, s = patch(o, x, y)
        print("%-22s %-12s mean %s std %s" % ("", name, np.round(m, 2), np.round(s, 2)))
    print()

# where does the difference actually live? absolute difference maps
ours = np.asarray(Image.open("out/max.png").convert("RGB"), dtype=np.float32)[Y0:Y0 + CH, X0:X0 + CW]
diff = np.abs(ours - af).mean(2)
print("mean abs difference ours-vs-AF: %.2f  (median %.2f, p95 %.2f)"
      % (diff.mean(), np.median(diff), np.percentile(diff, 95)))
B = 500
h, w = diff.shape[0] // B * B, diff.shape[1] // B * B
blk = diff[:h, :w].reshape(h // B, B, w // B, B).mean((1, 3))
ij = np.unravel_index(np.argmax(blk), blk.shape)
print("worst block at (%d, %d) in crop coords, mean abs diff %.2f"
      % (ij[1] * B, ij[0] * B, blk[ij]))
