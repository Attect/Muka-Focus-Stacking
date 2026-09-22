"""Compare a focusmerge output against the reference export produced elsewhere.

The reference (`AF导出.png`) is a 1:1 crop of a full resolution merge, so the
first job is to locate that crop inside our output. Then the two are compared
quantitatively (gradient energy in matching windows) and visually (side by side
100% crops of the regions that matter).
"""
import argparse
import os
import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

ap = argparse.ArgumentParser()
ap.add_argument("--ours", required=True)
ap.add_argument("--ref", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--label-ours", default="focusmerge")
ap.add_argument("--label-ref", default="reference")
args = ap.parse_args()

os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)


def load8(path):
    im = Image.open(path)
    if im.mode != "RGB":
        im = im.convert("RGB")
    return im


def gray(a):
    return a[..., 0] * 0.299 + a[..., 1] * 0.587 + a[..., 2] * 0.114


def grad_energy(g):
    gy, gx = np.gradient(g.astype(np.float32))
    return np.hypot(gx, gy)


ours_im = load8(args.ours)
ref_im = load8(args.ref)
ours = np.asarray(ours_im, dtype=np.float32)
ref = np.asarray(ref_im, dtype=np.float32)
print("ours", ours.shape, "ref", ref.shape)

# --- locate the reference crop inside our output -----------------------------
S = 8
od = np.asarray(ours_im.resize((ours.shape[1] // S, ours.shape[0] // S), Image.BOX), dtype=np.float32)
rd = np.asarray(ref_im.resize((ref.shape[1] // S, ref.shape[0] // S), Image.BOX), dtype=np.float32)
og = grad_energy(gray(od))
rg = grad_energy(gray(rd))
og -= og.mean()
rg -= rg.mean()

F = np.fft.rfft2(og)
G = np.fft.rfft2(rg, s=og.shape)
cc = np.fft.irfft2(F * np.conj(G), s=og.shape)
peak = np.unravel_index(np.argmax(cc), cc.shape)
dy, dx = peak
if dy > og.shape[0] // 2:
    dy -= og.shape[0]
if dx > og.shape[1] // 2:
    dx -= og.shape[1]
x0, y0 = dx * S, dy * S
print("reference crop located at", (x0, y0), "with size", (ref.shape[1], ref.shape[0]))

cw, ch = ref.shape[1], ref.shape[0]
crop = ours[y0:y0 + ch, x0:x0 + cw]
print("crop shape", crop.shape)

# --- quantitative sharpness on matched windows -------------------------------
gh, gw = grad_energy(gray(crop)), grad_energy(gray(ref))
print("\ngradient energy (higher = more detail):")
print("  %-14s %.4f" % (args.label_ours, gh.mean()))
print("  %-14s %.4f" % (args.label_ref, rg.mean() * 0 + grad_energy(gray(ref)).mean()))

# local sharpness in blocks: show where each wins
B = 256
hh = ch // B * B
ww = cw // B * B
a = gh[:hh, :ww].reshape(hh // B, B, ww // B, B).mean(axis=(1, 3))
b = grad_energy(gray(ref))[:hh, :ww].reshape(hh // B, B, ww // B, B).mean(axis=(1, 3))
ratio = a / (b + 1e-6)
print("\nblock sharpness ratio %s / %s : mean=%.3f median=%.3f  blocks_ours_sharper=%.1f%%"
      % (args.label_ours, args.label_ref, ratio.mean(), np.median(ratio),
         100.0 * (ratio > 1).mean()))

# --- visual side by side -----------------------------------------------------
def panel(regions):
    """regions: list of (title, x, y, w, h) in matched coords."""
    rows = []
    for title, x, y, w, h in regions:
        a = crop[y:y + h, x:x + w]
        b = ref[y:y + h, x:x + w]
        sep = np.full((h, 6, 3), 255.0, dtype=np.float32)
        rows.append((title, np.concatenate([a, sep, b], axis=1)))
    total_h = sum(r[1].shape[0] + 30 for r in rows)
    total_w = max(r[1].shape[1] for r in rows)
    sheet = np.full((total_h, total_w, 3), 255.0, dtype=np.float32)
    yy = 0
    from PIL import ImageDraw, ImageFont
    for title, img in rows:
        sheet[yy:yy + img.shape[0], 0:img.shape[1]] = img
        yy += img.shape[0] + 30
    out = Image.fromarray(sheet.astype(np.uint8))
    dr = ImageDraw.Draw(out)
    yy = 0
    for title, img in rows:
        dr.text((6, yy + img.shape[0] + 6), title, fill=(0, 0, 0))
        yy += img.shape[0] + 30
    return out


# pick a few interesting windows: centre of the figure, a foreground bomb,
# the background edge, and the lower base
cx, cy = cw // 2, ch // 2
regions = [
    ("A face / upper body", max(0, cx - 500), max(0, cy - 900), 700, 700),
    ("B foreground props", max(0, cx - 1600), min(ch - 700, cy + 300), 700, 700),
    ("C right side figures", min(cw - 700, cx + 300), max(0, cy - 200), 700, 700),
    ("D base / ground", max(0, cx - 900), max(0, ch - 800), 700, 700),
]
sheet = panel(regions)
sheet.save(args.out.replace(".png", "_crops.png"))
print("\nwrote", args.out.replace(".png", "_crops.png"))

# small full comparison
scale_target = 1200
a = Image.fromarray(crop.astype(np.uint8))
b = Image.fromarray(ref.astype(np.uint8))
f = scale_target / a.width
a = a.resize((int(a.width * f), int(a.height * f)), Image.LANCZOS)
b = b.resize(a.size, Image.LANCZOS)
full = Image.new("RGB", (a.width * 2 + 12, a.height), (255, 255, 255))
full.paste(a, (0, 0))
full.paste(b, (a.width + 12, 0))
full.save(args.out.replace(".png", "_full.png"))
print("wrote", args.out.replace(".png", "_full.png"))
