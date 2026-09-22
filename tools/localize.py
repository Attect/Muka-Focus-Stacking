"""Locate where focusmerge is soft compared with the reference, and explain why.

For the worst regions it reports, per region:
  * our depth-field value (which frame index we blended from)
  * the frame that is actually sharpest there (raw per-frame sharpness ranking)
  * the focus-confidence percentile we measured there

If the depth value disagrees with the sharpest frame, the depth field is at
fault; if the confidence is low, the region was treated as unreliable and
inpainted from its surroundings.
"""
import os
import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
D = os.environ.get("STACK_DIR", "stack")
OURS = r"B:\FocusMerge\out\可莉-全清晰.png"
DEPTH = r"B:\FocusMerge\out\depth_final.png"
CONF = r"B:\FocusMerge\out\conf.png"

# AF crop origin inside our output, and our output origin inside the full canvas
AF_X, AF_Y = 1288, 488
CV_X, CV_Y = 20, 13
SCALE = 2  # analysis scale

files = sorted(f for f in os.listdir(D) if f.upper().startswith("DSC") and f.upper().endswith(".JPG"))


def gray(a):
    return a[..., 0] * 0.299 + a[..., 1] * 0.587 + a[..., 2] * 0.114


def ge(a):
    g = gray(a) if a.ndim == 3 else a
    gy, gx = np.gradient(g.astype(np.float32))
    return np.hypot(gx, gy)


af = np.asarray(Image.open(os.path.join(D, "AF导出.png")).convert("RGB"), dtype=np.float32)
ours = np.asarray(Image.open(OURS).convert("RGB"), dtype=np.float32)
CH, CW = af.shape[0], af.shape[1]
oc = ours[AF_Y:AF_Y + CH, AF_X:AF_X + CW]
depth = np.asarray(Image.open(DEPTH), dtype=np.float32) / 65535.0 * (len(files) - 1)
conf = np.asarray(Image.open(CONF), dtype=np.float32)
print("depth", depth.shape, "conf", conf.shape, "crop", oc.shape)

go, ga = ge(oc), ge(af)
B = 128
hh, ww = (CH // B) * B, (CW // B) * B
bo = go[:hh, :ww].reshape(hh // B, B, ww // B, B).mean((1, 3))
ba = ga[:hh, :ww].reshape(hh // B, B, ww // B, B).mean((1, 3))
ratio = bo / (ba + 1e-6)

flat = ratio.ravel()
order = np.argsort(flat)[:24]
nb = ratio.shape[1]
print("\nworst blocks (ours/reference gradient ratio):")
cand = []
for k in order:
    by, bx = divmod(int(k), nb)
    cand.append((flat[k], bx * B, by * B))

# keep the worst blocks that are reasonably far apart
picked = []
for r, x, y in cand:
    if all(abs(x - px) > 300 or abs(y - py) > 300 for _, px, py in picked):
        picked.append((r, x, y))
    if len(picked) >= 5:
        break

# load raw frames lazily for the per-frame sharpness ranking
print("\n%-16s %6s %7s %9s %9s %8s" % ("region", "ratio", "depth", "sharpest", "2nd best", "conf%"))
patch = 160
for r, x, y in picked:
    cx, cy = (AF_X + x + CV_X) // SCALE, (AF_Y + y + CV_Y) // SCALE
    cs = patch // SCALE
    d = depth[max(0, cy):cy + cs, max(0, cx):cx + cs]
    c = conf[max(0, cy):cy + cs, max(0, cx):cx + cs]
    scores = []
    for f in files:
        im = Image.open(os.path.join(D, f)).convert("RGB")
        a = np.asarray(im.crop((AF_X + x + CV_X, AF_Y + y + CV_Y,
                                AF_X + x + CV_X + patch, AF_Y + y + CV_Y + patch)),
                       dtype=np.float32)
        scores.append(float(ge(a).mean()))
    scores = np.array(scores)
    best = int(np.argmax(scores))
    second = int(np.argsort(scores)[-2])
    print("%-16s %6.2f %7.1f %9d %9d %7.0f%%   (block %d,%d)"
          % ("", r, float(d.mean()), best, second,
             100.0 * float((c > np.percentile(conf, 30)).mean()), x, y))

# visual: worst three side by side
rows = []
for r, x, y in picked[:3]:
    a = oc[y - 60:y + patch + 60, x - 60:x + patch + 60]
    b = af[y - 60:y + patch + 60, x - 60:x + patch + 60]
    sep = np.full((a.shape[0], 8, 3), 255.0, dtype=np.float32)
    rows.append(np.concatenate([a, sep, b], axis=1))
W = max(r.shape[1] for r in rows)
H = sum(r.shape[0] + 12 for r in rows)
sheet = np.full((H, W, 3), 255.0, dtype=np.float32)
yy = 0
for r in rows:
    sheet[yy:yy + r.shape[0], 0:r.shape[1]] = r
    yy += r.shape[0] + 12
Image.fromarray(sheet.astype(np.uint8)).save(r"B:\FocusMerge\out\worst_crops.png")
print("\nwrote out/worst_crops.png")
