"""Overlay which frames the depth field selected, for one region.

Pixels whose selected frame lies in a given "good" range are highlighted. If the
object being examined is not highlighted, its depth is wrong; if it is
highlighted but still soft, the problem is in the fusion instead.
"""
import os
import sys
import numpy as np
from PIL import Image, ImageDraw

Image.MAX_IMAGE_PIXELS = None
D = os.environ.get("STACK_DIR", "stack")
AF_X, AF_Y, CV_X, CV_Y, S = 1288, 488, 20, 13, 2
N = 50

cx, cy, half = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
good_lo, good_hi = int(sys.argv[4]), int(sys.argv[5])
depth_path = sys.argv[6]
x0, y0, size = cx - half, cy - half, 2 * half

dep = np.asarray(Image.open(depth_path), dtype=np.float32) / 65535.0 * (N - 1)
ax, ay = (AF_X + x0 + CV_X) // S, (AF_Y + y0 + CV_Y) // S
h = size // S
sel = dep[ay:ay + h, ax:ax + h]

ours = np.asarray(Image.open(r"B:\FocusMerge\out\final.png")
                  .crop((AF_X + x0, AF_Y + y0, AF_X + x0 + size, AF_Y + y0 + size)), dtype=np.float32)
ours_sel = dep[ay:ay + h, ax:ax + h]
# nearest-neighbour upsample so each depth sample maps to a 2x2 image block
mask = ((sel >= good_lo) & (sel <= good_hi)).astype(np.uint8)
mask_up = np.repeat(np.repeat(mask, S, 0), S, 1)[:size, :size]
overlay = ours.copy()
overlay[..., 0] = np.where(mask_up > 0, np.minimum(255, ours[..., 0] + 90), ours[..., 0] * 0.45)
overlay[..., 1] = np.where(mask_up > 0, ours[..., 1], ours[..., 1] * 0.45)
overlay[..., 2] = np.where(mask_up > 0, ours[..., 2], ours[..., 2] * 0.45)

af = Image.open(os.path.join(D, "AF导出.png")).crop((x0, y0, x0 + size, y0 + size))

zoom = 2
cell = size * zoom


def big(a):
    if a.ndim == 2:
        return Image.fromarray(a).convert("RGB").resize((cell, cell), Image.NEAREST)
    return Image.fromarray(a.astype(np.uint8)).resize((cell, cell), Image.LANCZOS)


sheet = Image.new("RGB", (cell * 3 + 36, cell + 30), (255, 255, 255))
dr = ImageDraw.Draw(sheet)
for k, (lab, im) in enumerate([
        ("our output", big(ours)),
        ("frame %d-%d selected" % (good_lo, good_hi), big(overlay)),
        ("AF export", big(np.asarray(af, dtype=np.float32)))]):
    dr.text((k * (cell + 18) + 6, 8), lab, fill=(0, 0, 0))
    sheet.paste(im, (k * (cell + 18), 30))
p = r"B:\FocusMerge\out\selmap_%d_%d.png" % (cx, cy)
sheet.save(p)
print("wrote", p, " selected-frame stats: min %.1f median %.1f max %.1f"
      % (sel.min(), np.median(sel), sel.max()))
