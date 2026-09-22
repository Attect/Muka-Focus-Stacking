"""Find the truly sharpest frame for one patch, and compare our depth against it.

Reports, for a patch centred on (x, y) in reference-crop coordinates:
  * per-frame edge acutance, ranked, so the best focus position is known
  * the depth value our run assigned there, raw and final
  * the focus prominence we measured there

Then renders the top frames next to our output and the reference, at high
magnification, so the conclusion can be checked by eye.
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
N = len(files)

cx, cy, size = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
zoom = int(sys.argv[4]) if len(sys.argv) > 4 else 2
depth_path = sys.argv[5] if len(sys.argv) > 5 else r"B:\FocusMerge\out\depth3_final.png"
raw_path = sys.argv[6] if len(sys.argv) > 6 else r"B:\FocusMerge\out\depth3.png"

x0, y0 = cx - size // 2, cy - size // 2


def acutance(a, q=0.99):
    g = a[..., 0] * 0.299 + a[..., 1] * 0.587 + a[..., 2] * 0.114
    gy, gx = np.gradient(g.astype(np.float32))
    m = np.hypot(gx, gy).ravel()
    return float(m[m >= np.quantile(m, q)].mean())


print("patch centre (%d,%d) size %d  ->  canvas (%d,%d)"
      % (cx, cy, size, AF_X + cx + CV_X, AF_Y + cy + CV_Y))

scores = np.zeros(N)
for i, f in enumerate(files):
    a = np.asarray(Image.open(os.path.join(D, f)).convert("RGB"), dtype=np.float32)
    scores[i] = acutance(a[AF_Y + y0 + CV_Y:AF_Y + y0 + CV_Y + size,
                           AF_X + x0 + CV_X:AF_X + x0 + CV_X + size])
order = np.argsort(scores)[::-1]
print("per-frame edge acutance, best first:",
      ", ".join("%d(%.1f)" % (i, scores[i]) for i in order[:8]))

if os.path.exists(depth_path):
    fin = np.asarray(Image.open(depth_path), dtype=np.float32) / 65535.0 * (N - 1)
    raw = np.asarray(Image.open(raw_path), dtype=np.float32) / 65535.0 * (N - 1)
    # depth maps are at analysis resolution (half size) over the canvas
    ax = (AF_X + cx + CV_X) // 2
    ay = (AF_Y + cy + CV_Y) // 2
    s = size // 2
    print("our depth  raw %.1f   final %.1f   (true best frame %d)"
          % (np.median(raw[ay:ay + s, ax:ax + s]), np.median(fin[ay:ay + s, ax:ax + s]), order[0]))

tiles, labels = [], []
for i in list(order[:4]):
    a = np.asarray(Image.open(os.path.join(D, files[i])).convert("RGB"), dtype=np.float32)
    tiles.append(a[AF_Y + y0 + CV_Y:AF_Y + y0 + CV_Y + size,
                   AF_X + x0 + CV_X:AF_X + x0 + CV_X + size])
    labels.append("frame %d (%.1f)" % (i, scores[i]))
for name, path in [("FINAL", r"B:\FocusMerge\out\final.png"),
                   ("AF export", None)]:
    if path is None:
        a = np.asarray(Image.open(os.path.join(D, "AF导出.png")).convert("RGB"), dtype=np.float32)
        tiles.append(a[y0:y0 + size, x0:x0 + size])
    else:
        a = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32)
        tiles.append(a[AF_Y + y0:AF_Y + y0 + size, AF_X + x0:AF_X + x0 + size])
    labels.append(name)

big = [np.asarray(Image.fromarray(t.astype(np.uint8)).resize((size * zoom, size * zoom), Image.NEAREST),
                  dtype=np.float32) for t in tiles]
cols = 3
rows = (len(big) + cols - 1) // cols
cell = size * zoom + 8
sheet = np.full((rows * (cell + 26), cols * cell, 3), 255.0, dtype=np.float32)
for k, t in enumerate(big):
    ry, rx = divmod(k, cols)
    sheet[ry * (cell + 26) + 26:ry * (cell + 26) + 26 + t.shape[0], rx * cell:rx * cell + t.shape[1]] = t
out = Image.fromarray(sheet.astype(np.uint8))
dr = ImageDraw.Draw(out)
for k, lab in enumerate(labels):
    ry, rx = divmod(k, cols)
    dr.text((rx * cell + 6, ry * (cell + 26) + 8), lab, fill=(0, 0, 0))
p = r"B:\FocusMerge\out\probe_%d_%d.png" % (cx, cy)
out.save(p)
print("wrote", p, labels)
