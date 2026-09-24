"""How much of an output is not present in any source frame?

A fused pixel should be built from values that were actually photographed. A
halo — the classic failure of unconstrained per-band winner selection — is
exactly a value the frames never contained, and it appears next to a strong
edge. This measures that directly: per-pixel minimum and maximum over the stack,
then the share of output pixels landing outside that range and by how much.

Usage:  halo.py <region> [image ...]
        <region> is  x,y,w,h  in reference-crop coordinates.
"""
import os
import sys
import numpy as np
from PIL import Image

# The project directory, so the paths below work on any machine.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

Image.MAX_IMAGE_PIXELS = None
D = os.environ.get("STACK_DIR", "stack")
AF_X, AF_Y, CV_X, CV_Y = 1288, 488, 20, 13
files = sorted(f for f in os.listdir(D) if f.upper().startswith("DSC") and f.upper().endswith(".JPG"))

x, y, w, h = [int(v) for v in sys.argv[1].split(",")]
images = sys.argv[2:] or [os.path.join(ROOT, "out/可莉-全清晰.png")]
# canvas coordinates of the region (raw frames are indexed in canvas coordinates)
cx, cy = AF_X + x + CV_X, AF_Y + y + CV_Y
STRIDE = 3

lo = np.full((h, w, 3), np.inf, np.float32)
hi = np.full((h, w, 3), -np.inf, np.float32)
for i in range(0, len(files), STRIDE):
    a = np.asarray(Image.open(os.path.join(D, files[i])).convert("RGB"), dtype=np.float32)
    p = a[cy:cy + h, cx:cx + w]
    np.minimum(lo, p, out=lo)
    np.maximum(hi, p, out=hi)
AF = np.asarray(Image.open(os.path.join(D, "AF导出.png")).convert("RGB"), dtype=np.float32)
print("range computed from %d frames over canvas %s" % (len(files[::STRIDE]), (cx, cy)))
print()
print("%-24s %10s %12s %14s" % ("image", "outside%", "mean excess", "max excess"))
for path in images:
    if not os.path.exists(path):
        print("%-24s (missing)" % os.path.basename(path))
        continue
    a = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32)
    # Locate the reference crop at 1 px precision: a small sample is not enough,
    # a two-pixel error alone would dominate the range test below.
    S = 160
    ry, rx = h // 2 - S // 2, w // 2 - S // 2
    ref = AF[y + ry:y + ry + S, x + rx:x + rx + S]
    best = None
    for oy in range(478, 502):
        for ox in range(1278, 1302):
            if oy + y + ry + S >= a.shape[0] or ox + x + rx + S >= a.shape[1]:
                continue
            sub = a[oy + y + ry:oy + y + ry + S, ox + x + rx:ox + x + rx + S]
            mad = float(np.abs(sub - ref).mean())
            if best is None or mad < best[0]:
                best = (mad, ox, oy)
    if best is None:
        print("%-24s (too small)" % os.path.basename(path))
        continue
    _, ox, oy = best
    p = a[oy + y:oy + y + h, ox + x:ox + x + w]
    over = np.maximum(p - hi, 0.0)
    under = np.maximum(lo - p, 0.0)
    excess = np.maximum(over, under)
    out_share = 100.0 * float((excess > 1.0).mean())
    print("%-24s %9.2f%% %12.3f %14.1f   (origin %d,%d)"
          % (os.path.basename(path), out_share, float(excess.mean()), float(excess.max()), ox, oy))
