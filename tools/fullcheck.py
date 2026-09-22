"""Whole-frame and 100%-zone check sheets for a proposed operating point.

Grain is sub-pixel at whole-frame scale and invisible there, while a whole-frame
look is what the eye judges first — so this produces both, plus a map of *where*
two builds differ, and a table of how big the differences are.

Canvas mapping (verified): canvas = output + (12, 18) = reference + (506, 1304)

    python tools/fullcheck.py <new.png> <old.png> [<reference.png>]
"""
import os
import sys

import numpy as np
from PIL import Image, ImageDraw

Image.MAX_IMAGE_PIXELS = None
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from patchlib import REF, load  # noqa: E402

WIDE = 1600  # whole-frame panels are rendered at this width
# Zones are (canvas x0, y0) of a 500x360 crop at 1:1, chosen to cover the cases
# that have gone wrong on this stack: a smooth subject surface, the dark parts,
# a silhouette, the backdrop relief, the face, and the base.
ZONES = [
    ("boot surface (red, stitched)", 3200, 3080),
    ("hand / arm, dark prop", 2512, 1620),
    ("torso + belt", 2600, 2700),
    ("silhouette (leg vs wall)", 2700, 2830),
    ("backdrop wall relief", 2400, 1120),
    ("base flower + wall", 1800, 2360),
]
ZW, ZH = 500, 360


def canvas_crop(img, x0, y0, w, h, off):
    return np.asarray(img[y0 - off[0] : y0 - off[0] + h, x0 - off[1] : x0 - off[1] + w][..., :3],
                      dtype=np.uint8)


def gray(x):
    return (0.299 * x[..., 0] + 0.587 * x[..., 1] + 0.114 * x[..., 2]).astype(np.float64)


def main():
    new_p, old_p = sys.argv[1], sys.argv[2]
    ref_p = sys.argv[3] if len(sys.argv) > 3 else REF
    N, O, R = load(new_p), load(old_p), load(ref_p)
    off_n = (12, 18)
    off_r = (506, 1304)

    # --- whole-frame, side by side -------------------------------------------
    sheets = [("OLD", O), ("NEW", N)]
    panels = []
    for lab, im in sheets:
        t = np.asarray(im[..., :3], dtype=np.uint8)
        h = int(round(WIDE * t.shape[0] / t.shape[1]))
        panels.append((lab, Image.fromarray(t).resize((WIDE, h), Image.LANCZOS)))
    hh = panels[0][1].size[1]
    sheet = Image.new("RGB", (2 * (WIDE + 8), hh + 30), (255, 255, 255))
    dr = ImageDraw.Draw(sheet)
    for k, (lab, p) in enumerate(panels):
        sheet.paste(p, (k * (WIDE + 8), 28))
        dr.text((k * (WIDE + 8) + 4, 8), "%s  (whole frame, %.2f/px)" % (lab, 6971.0 / WIDE),
                fill=(200, 0, 0))
    sheet.save("out/full_default_cmp.png")
    print("wrote out/full_default_cmp.png", sheet.size)

    # --- where do the two differ, and by how much ----------------------------
    d = np.abs(np.asarray(N, dtype=np.int16)[:, :, :3] - np.asarray(O, dtype=np.int16)[:, :, :3])
    dm = d.max(axis=2)
    print("\ndifference NEW vs OLD over the whole output (levels 0-255):")
    for t in (1, 2, 4, 8, 16):
        print("   > %-3d : %6.3f%% of pixels" % (t, 100 * (dm > t).mean()))
    print("   max %d   mean %.3f" % (dm.max(), dm.mean()))
    # Where is it concentrated? Report by broad region (canvas coordinates, so
    # the boxes can be compared with every other measurement in this project).
    reg = [
        ("backdrop wall (upper left)", 200, 1500, 200, 2500),
        ("backdrop wall (right)", 1500, 2900, 4000, 6800),
        ("subject / figure", 2000, 4100, 1900, 4100),
        ("base / leaf (smooth)", 4100, 4600, 2500, 3900),
    ]
    for lab, cy0, cy1, cx0, cx1 in reg:
        y0, x0 = cy0 - off_n[0], cx0 - off_n[1]
        w = dm[y0 : cy1 - off_n[0], x0 : cx1 - off_n[1]]
        print("   %-28s >4: %6.3f%%   mean %.3f" % (lab, 100 * (w > 4).mean(), w.mean()))
    amp = np.clip(dm.astype(np.float64) * 4.0, 0, 255).astype(np.uint8)
    base = np.asarray(N, dtype=np.uint8)[:, :, :3].copy()
    hot = dm > 4
    base[hot] = (base[hot] * 0.35 + np.array([255, 0, 0]) * 0.65).astype(np.uint8)
    t1 = Image.fromarray(np.dstack([amp, amp, amp]))
    t2 = Image.fromarray(base)
    sheet = Image.new("RGB", (2 * (WIDE + 8), hh + 30), (255, 255, 255))
    dr = ImageDraw.Draw(sheet)
    for k, (lab, p) in enumerate([("|NEW-OLD| x4", t1), ("NEW, red where |diff|>4", t2)]):
        sheet.paste(p.resize((WIDE, hh), Image.BILINEAR), (k * (WIDE + 8), 28))
        dr.text((k * (WIDE + 8) + 4, 8), lab, fill=(200, 0, 0))
    sheet.save("out/full_diff.png")
    print("wrote out/full_diff.png", sheet.size)

    # --- 1:1 zone sheet -------------------------------------------------------
    cols = [("OLD", O, off_n), ("NEW", N, off_n), ("AF export", R, off_r)]
    cw = ZW * len(cols) + 8 * (len(cols) - 1)
    sheet = Image.new("RGB", (cw, len(ZONES) * (ZH + 30)), (255, 255, 255))
    dr = ImageDraw.Draw(sheet)
    for zi, (zlab, x0, y0) in enumerate(ZONES):
        for k, (lab, im, off) in enumerate(cols):
            t = canvas_crop(im, x0, y0, ZW, ZH, off)
            sheet.paste(Image.fromarray(t), (k * (ZW + 8), zi * (ZH + 30) + 28))
        dr.text((4, zi * (ZH + 30) + 8),
                "%s   canvas (%d,%d) 1:1     columns: OLD | NEW | AF export" % (zlab, x0, y0),
                fill=(200, 0, 0))
    sheet.save("out/zones_100pct.png")
    print("wrote out/zones_100pct.png", sheet.size)


if __name__ == "__main__":
    main()
