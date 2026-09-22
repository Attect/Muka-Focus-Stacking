"""Ground truth for one region: render it from several raw frames.

Prints a strip of the same physical patch at successive focus positions, so the
truly sharpest frame can be read off by eye instead of inferred from a metric.
"""
import os
import sys
import numpy as np
from PIL import Image, ImageDraw

Image.MAX_IMAGE_PIXELS = None
D = os.environ.get("STACK_DIR", "stack")
AF_X, AF_Y = 1288, 488
CV_X, CV_Y = 20, 13
files = sorted(f for f in os.listdir(D) if f.upper().startswith("DSC") and f.upper().endswith(".JPG"))

# name, ref-crop x, y, size, frame list
REGION = {
    "leaf": (1400, 3040, 300, [7, 9, 11, 13, 15, 17, 19, 21]),
    "turntable": (600, 3560, 300, [7, 9, 11, 13, 15, 17, 19, 21]),
    "legs": (1250, 1900, 300, [10, 14, 18, 22, 26, 30, 34, 38]),
}
which = sys.argv[1] if len(sys.argv) > 1 else "leaf"
x, y, s, frames = REGION[which]

tiles = []
labels = []
for i in frames:
    a = np.asarray(Image.open(os.path.join(D, files[i])).convert("RGB"), dtype=np.float32)
    tiles.append(a[AF_Y + y + CV_Y:AF_Y + y + CV_Y + s, AF_X + x + CV_X:AF_X + x + CV_X + s])
    labels.append("frame %d" % i)

for name, path in [
    ("fix2 bracket", r"B:\FocusMerge\out\fix2.png"),
    ("hard pick", r"B:\FocusMerge\out\hard.png"),
    ("pyramid", r"B:\FocusMerge\out\pyr.png"),
    ("AF export", None),
]:
    if path is None:
        a = np.asarray(Image.open(os.path.join(D, "AF导出.png")).convert("RGB"), dtype=np.float32)
        tiles.append(a[y:y + s, x:x + s])
    elif os.path.exists(path):
        a = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32)
        tiles.append(a[AF_Y + y:AF_Y + y + s, AF_X + x:AF_X + x + s])
    else:
        continue
    labels.append(name)

cols = 4
rows = (len(tiles) + cols - 1) // cols
cell = s + 6
sheet = np.full((rows * (cell + 24), cols * cell, 3), 255.0, dtype=np.float32)
for k, t in enumerate(tiles):
    cy, cx = divmod(k, cols)
    sheet[cy * (cell + 24) + 24:cy * (cell + 24) + 24 + s, cx * cell:cx * cell + s] = t
out = Image.fromarray(sheet.astype(np.uint8))
dr = ImageDraw.Draw(out)
for k, lab in enumerate(labels):
    cy, cx = divmod(k, cols)
    dr.text((cx * cell + 6, cy * (cell + 24) + 6), lab, fill=(0, 0, 0))
out.save(r"B:\FocusMerge\out\strip_%s.png" % which)
print("wrote out/strip_%s.png :" % which, labels)
