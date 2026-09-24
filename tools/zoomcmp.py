"""High magnification comparison of one small area, rendered from several
sources side by side. Used to settle cases where metrics and the eye disagree."""
import os
import sys
import numpy as np
from PIL import Image, ImageDraw

# The project directory, so the paths below work on any machine.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

Image.MAX_IMAGE_PIXELS = None
D = os.environ.get("STACK_DIR", "stack")
AF_X, AF_Y = 1288, 488
CV_X, CV_Y = 20, 13
files = sorted(f for f in os.listdir(D) if f.upper().startswith("DSC") and f.upper().endswith(".JPG"))

# x, y, size, zoom, frames
x, y, s, zoom = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
frames = [int(t) for t in sys.argv[5].split(",")]

SOURCES = [
    ("AF export", None),
    ("OURS cur", os.path.join(ROOT, "out/cur.png")),
    ("OURS rev", os.path.join(ROOT, "out/rev.png")),
]
# Optional argv[6]: comma separated "label=path" entries replacing SOURCES.
# The literal label @AF stands for the reference export.
if len(sys.argv) > 6 and sys.argv[6]:
    SOURCES = []
    for entry in sys.argv[6].split(";"):
        lab, _, path = entry.partition("=")
        SOURCES.append((lab, None if path == "@AF" else path))
tiles, labels = [], []
for i in frames:
    a = np.asarray(Image.open(os.path.join(D, files[i])).convert("RGB"), dtype=np.float32)
    tiles.append(a[AF_Y + y + CV_Y:AF_Y + y + CV_Y + s, AF_X + x + CV_X:AF_X + x + CV_X + s])
    labels.append("frame %d" % i)

for name, path in SOURCES:
    if path is None:
        a = np.asarray(Image.open(os.path.join(D, "AF导出.png")).convert("RGB"), dtype=np.float32)
        tiles.append(a[y:y + s, x:x + s])
    elif os.path.exists(path):
        a = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32)
        tiles.append(a[AF_Y + y:AF_Y + y + s, AF_X + x:AF_X + x + s])
    else:
        continue
    labels.append(name)

big = []
for t in tiles:
    im = Image.fromarray(t.astype(np.uint8)).resize((s * zoom, s * zoom), Image.NEAREST)
    big.append(np.asarray(im, dtype=np.float32))

cols = 3
rows = (len(big) + cols - 1) // cols
cell = s * zoom + 8
sheet = np.full((rows * (cell + 26), cols * cell, 3), 255.0, dtype=np.float32)
for k, t in enumerate(big):
    cy, cx = divmod(k, cols)
    sheet[cy * (cell + 26) + 26:cy * (cell + 26) + 26 + t.shape[0], cx * cell:cx * cell + t.shape[1]] = t
out = Image.fromarray(sheet.astype(np.uint8))
dr = ImageDraw.Draw(out)
for k, lab in enumerate(labels):
    cy, cx = divmod(k, cols)
    dr.text((cx * cell + 6, cy * (cell + 26) + 8), lab, fill=(0, 0, 0))
p = os.path.join(ROOT, "out/zoom_%d_%d.png") % (x, y)
out.save(p)
print("wrote", p, labels)
