"""Score several output images against the reference export.

Locates the reference crop inside each output automatically (the outputs have
different origins -- a run with alignment crops a few pixels off the frame, one
without keeps the whole frame), then reports the same per-region figures used
throughout: global gradient energy, flat-background texture, and edge acutance
relative to the reference in the regions that matter.

Usage:  eval_all.py [ref:x:y label ...]   (regions optional)
"""
import os
import sys
import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
D = os.environ.get("STACK_DIR", "stack")
REF = np.asarray(Image.open(os.path.join(D, "AF导出.png")).convert("RGB"), dtype=np.float32)

REGIONS = [
    ("leaf", 1400, 3040, 300),
    ("turntable", 600, 3560, 300),
    ("legs", 1250, 1900, 300),
    ("face", 2000, 700, 300),
    ("body", 1612, 1240, 300),
]
IDEAL_GRAD = 7.77


def gray(a):
    return a[..., 0] * 0.299 + a[..., 1] * 0.587 + a[..., 2] * 0.114


def grad_energy(a):
    gy, gx = np.gradient(gray(a).astype(np.float32))
    return float(np.hypot(gx, gy).mean())


def acutance(a, q=0.995):
    gy, gx = np.gradient(gray(a).astype(np.float32))
    m = np.hypot(gx, gy).ravel()
    return float(m[m >= np.quantile(m, q)].mean())


def bg_std(a, x=150, y=150, s=200):
    return float(a[y:y + s, x:x + s].reshape(-1, 3).std(0).mean())


def find_offset(path):
    """Search the reference crop origin inside this output, 1 px precision."""
    im = Image.open(path).convert("RGB")
    if im.size[0] < REF.shape[1] + 40:
        return None
    g = 8
    ao = np.asarray(im.convert("L"), dtype=np.float32)[::g, ::g]
    aa = np.asarray(Image.fromarray(REF.astype(np.uint8)).convert("L"), dtype=np.float32)[::g, ::g]
    h, w = aa.shape
    best = None
    for oy in range(470, 530):
        for ox in range(1270, 1330):
            y0, x0 = oy // g, ox // g
            sub = ao[y0:y0 + h, x0:x0 + w]
            if sub.shape != aa.shape:
                continue
            mad = float(np.abs(sub - aa).mean())
            if best is None or mad < best[0]:
                best = (mad, ox, oy)
    return best


print("reference: gradE %.3f   bg_std %.2f" % (grad_energy(REF), bg_std(REF)))
print()
print("%-26s %8s %8s | %s" % ("output", "gradE", "bg_std", " ".join("%-9s" % r[0] for r in REGIONS)))
print("%-26s %8.3f %8.2f | %s"
      % ("AF export", grad_energy(REF), bg_std(REF), "".join("%-9.2f" % 1.0 for _ in REGIONS)))

for path in sys.argv[1:]:
    if not os.path.exists(path):
        print("%-26s  (missing)" % os.path.basename(path))
        continue
    found = find_offset(path)
    if found is None:
        print("%-26s  (smaller than reference)" % os.path.basename(path))
        continue
    mad, ox, oy = found
    a = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32)
    a = a[oy:oy + REF.shape[0], ox:ox + REF.shape[1]]
    cells = "".join("%-9.2f" % (acutance(a[y:y + s, x:x + s]) / acutance(REF[y:y + s, x:x + s]))
                    for _, x, y, s in REGIONS)
    print("%-26s %8.3f %8.2f | %s   (origin %d,%d, mad %.2f)"
          % (os.path.basename(path), grad_energy(a), bg_std(a), cells, ox, oy, mad))
