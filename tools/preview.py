"""Build a contact sheet of sample frames plus the reference AF export."""
import os
import numpy as np
from PIL import Image, ImageDraw

# The project directory, so the paths below work on any machine.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

Image.MAX_IMAGE_PIXELS = None
D = os.environ.get("STACK_DIR", "stack")
OUT = os.path.join(ROOT, "preview")
os.makedirs(OUT, exist_ok=True)

files = sorted(f for f in os.listdir(D) if f.upper().startswith("DSC") and f.upper().endswith(".JPG"))
picks = [0, 8, 16, 24, 32, 40, 49]

TW = 640
rows = []
for i in picks:
    im = Image.open(os.path.join(D, files[i])).convert("RGB")
    h = int(im.height * TW / im.width)
    rows.append((files[i], im.resize((TW, h), Image.LANCZOS)))

ref = Image.open(os.path.join(D, "AF导出.png")).convert("RGB")
rh = int(ref.height * TW / ref.width)
ref_small = ref.resize((TW, rh), Image.LANCZOS)

cell_h = max(r[1].height for r in rows)
cols = 4
n = len(rows) + 1
rows_n = (n + cols - 1) // cols
sheet = Image.new("RGB", (cols * TW, rows_n * (cell_h + 22)), (255, 255, 255))
dr = ImageDraw.Draw(sheet)

items = rows + [("AF导出 (Affinity)", ref_small)]
for k, (name, im) in enumerate(items):
    cx = (k % cols) * TW
    cy = (k // cols) * (cell_h + 22)
    sheet.paste(im, (cx, cy + 22))
    dr.text((cx + 6, cy + 5), name, fill=(0, 0, 0))

sheet.save(os.path.join(OUT, "contact_sheet.png"))
print("saved", os.path.join(OUT, "contact_sheet.png"), sheet.size)

# full-res crop of the subject area from a mid frame and the reference
mid = Image.open(os.path.join(D, files[16])).convert("RGB")
print("mid frame", mid.size)
ref.save(os.path.join(OUT, "af_export_full.png"))
print("AF", ref.size, "aspect", ref.width / ref.height)
print("orig aspect", 7008 / 4672)
