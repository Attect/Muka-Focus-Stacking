"""Visualise the depth field next to the image, for one region.

Shows whether a soft edge is caused by the depth field assigning the wrong frame
to a band along an object boundary.
"""
import os
import sys
import numpy as np
from PIL import Image, ImageDraw

Image.MAX_IMAGE_PIXELS = None
D = os.environ.get("STACK_DIR", "stack")
AF_X, AF_Y = 1288, 488
CV_X, CV_Y = 20, 13
S = 2  # analysis scale
files = sorted(f for f in os.listdir(D) if f.upper().startswith("DSC") and f.upper().endswith(".JPG"))
N = len(files)

cx, cy, half = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
depth_path = sys.argv[4] if len(sys.argv) > 4 else r"B:\FocusMerge\out\depth3_final.png"
raw_path = sys.argv[5] if len(sys.argv) > 5 else r"B:\FocusMerge\out\depth3.png"

x0, y0 = cx - half, cy - half
size = 2 * half


def crop(img, x, y, s):
    return np.asarray(img.crop((x, y, x + s, y + s)), dtype=np.float32)


fin = np.asarray(Image.open(depth_path), dtype=np.float32) / 65535.0 * (N - 1)
raw = np.asarray(Image.open(raw_path), dtype=np.float32) / 65535.0 * (N - 1)
ax, ay = (AF_X + x0 + CV_X) // S, (AF_Y + y0 + CV_Y) // S
h = size // S
fin_c = fin[ay:ay + h, ax:ax + h]
raw_c = raw[ay:ay + h, ax:ax + h]

ours = Image.open(r"B:\FocusMerge\out\final.png").crop(
    (AF_X + x0, AF_Y + y0, AF_X + x0 + size, AF_Y + y0 + size))
af = Image.open(os.path.join(D, "AF导出.png")).crop((x0, y0, x0 + size, y0 + size))

# depth rendered as a grey ramp, nearest-neighbour upscaled to match
def depth_img(a):
    v = np.clip(a / max(1, N - 1) * 255, 0, 255).astype(np.uint8)
    return Image.fromarray(v).resize((size, size), Image.NEAREST)


tiles = [
    ("final depth", depth_img(fin_c)),
    ("raw depth", depth_img(raw_c)),
    ("our output", ours.convert("RGB")),
    ("AF export", af.convert("RGB")),
]

zoom = 2
cell = size * zoom
sheet = Image.new("RGB", (cell * 2 + 24, (cell + 30) * 2), (255, 255, 255))
dr = ImageDraw.Draw(sheet)
for k, (lab, im) in enumerate(tiles):
    ry, rx = divmod(k, 2)
    px, py = rx * (cell + 24), ry * (cell + 30)
    dr.text((px + 6, py + 8), "%s   (x=%d..%d y=%d..%d)" % (lab, x0, x0 + size, y0, y0 + size),
            fill=(0, 0, 0))
    sheet.paste(im.resize((cell, cell), Image.LANCZOS), (px, py + 30))
p = r"B:\FocusMerge\out\depthview_%d_%d.png" % (cx, cy)
sheet.save(p)
print("wrote", p)
print("depth in region: raw min %.1f median %.1f max %.1f | final min %.1f median %.1f max %.1f"
      % (raw_c.min(), np.median(raw_c), raw_c.max(),
         fin_c.min(), np.median(fin_c), fin_c.max()))
print("(0 = nearest frame, %d = farthest; brighter = farther)" % (N - 1))
