"""Three-way visual comparison of the complained-about regions.

Row layout: best raw frame in that region | focusmerge | reference export.
Seeing the raw frame next to both tells us immediately whether the reference is
actually sharper, or merely differently processed.
"""
import os
import numpy as np
from PIL import Image, ImageDraw

Image.MAX_IMAGE_PIXELS = None
D = os.environ.get("STACK_DIR", "stack")
AF_X, AF_Y = 1288, 488
CV_X, CV_Y = 20, 13
files = sorted(f for f in os.listdir(D) if f.upper().startswith("DSC") and f.upper().endswith(".JPG"))

# (label, x, y, size) in reference-crop coordinates
REGIONS = [
    ("legs / boots", 1150, 1450, 700),
    ("yellow leaf on base", 1330, 3000, 420),
    ("turntable edge", 560, 3520, 420),
    ("face (control)", 1930, 620, 420),
]

af_im = Image.open(os.path.join(D, "AF导出.png")).convert("RGB")
ours_im = Image.open(r"B:\FocusMerge\out\fix1.png").convert("RGB")
af = np.asarray(af_im, dtype=np.float32)
ours = np.asarray(ours_im, dtype=np.float32)
oc = ours[AF_Y:AF_Y + af.shape[0], AF_X:AF_X + af.shape[1]]

# pick, per region, the raw frame with the highest *mid-frequency* energy, so
# that full resolution sensor noise does not decide the winner
from PIL import ImageFilter


def midfreq(a):
    im = Image.fromarray(a.astype(np.uint8))
    low = np.asarray(im.filter(ImageFilter.GaussianBlur(2)), dtype=np.float32)
    return np.asarray(im, dtype=np.float32) - low


rows = []
meta = []
for label, x, y, s in REGIONS:
    best, bi = -1.0, 0
    for i, f in enumerate(files):
        a = np.asarray(Image.open(os.path.join(D, f)).convert("RGB"), dtype=np.float32)
        # raw frames live in canvas coordinates: crop origin plus alignment offset
        p = a[AF_Y + y + CV_Y:AF_Y + y + CV_Y + s, AF_X + x + CV_X:AF_X + x + CV_X + s]
        v = float((midfreq(p) ** 2).mean())
        if v > best:
            best, bi = v, i
    raw = np.asarray(Image.open(os.path.join(D, files[bi])).convert("RGB"), dtype=np.float32)
    pr = raw[AF_Y + y + CV_Y:AF_Y + y + CV_Y + s, AF_X + x + CV_X:AF_X + x + CV_X + s]
    po = oc[y:y + s, x:x + s]
    pa = af[y:y + s, x:x + s]
    sep = np.full((s, 10, 3), 255.0, dtype=np.float32)
    rows.append(np.concatenate([pr, sep, po, sep, pa], axis=1))
    meta.append("%s  | raw frame %d | focusmerge | AF export" % (label, bi))
    print("%-20s best raw frame %d  midfreq %.1f" % (label, bi, best**0.5))

W = max(r.shape[1] for r in rows)
H = sum(r.shape[0] + 34 for r in rows)
sheet = np.full((H, W, 3), 255.0, dtype=np.float32)
yy = 0
for r in rows:
    sheet[yy:yy + r.shape[0], 0:r.shape[1]] = r
    yy += r.shape[0] + 34
out = Image.fromarray(sheet.astype(np.uint8))
dr = ImageDraw.Draw(out)
yy = 0
for r, m in zip(rows, meta):
    dr.text((8, yy + r.shape[0] + 10), m, fill=(0, 0, 0))
    yy += r.shape[0] + 34
out.save(r"B:\FocusMerge\out\threeway.png")
print("\nwrote out/threeway.png")
