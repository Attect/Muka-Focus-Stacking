"""Render where the depth field is wrong, against the masked ground truth.

Blue = the depth is nearer than the wall really is, red = further. The band the
user marked shows up as a blue fringe hugging the whole silhouette: the focus
window straddles the figure's edge, the edge's far stronger response wins the
argmax, and the wall within one window radius of the figure is given the figure's
depth. It is then rendered from a frame in which that wall is defocused, which is
the bright, smooth band.

    python tools/deptherrmap.py out.png scale label=depth_final.png [...]
"""
import sys

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
sys.path.insert(0, 'tools')
from patchlib import load  # noqa: E402
from bandlib import subject_mask  # noqa: E402

CELL = 8
CANVAS_FROM_OUT = (12, 18)
BOX = (2200, 3220, 1420, 2260)
PRODUCT = 'out/可莉-全清晰.png'


def main():
    out = sys.argv[1]
    scale = float(sys.argv[2])
    items = [a.split('=', 1) for a in sys.argv[3:]]
    true = np.load('out/_true_depth_cell.npy').astype(np.float32)
    good = np.load('out/_cell_good.npy')
    H, W = true.shape
    x0, x1, y0, y1 = BOX
    oy, ox = CANVAS_FROM_OUT
    base = np.asarray(load(PRODUCT), dtype=np.float32)[y0:y1, x0:x1]
    hh, ww = base.shape[:2]
    subj, _ = subject_mask(load(PRODUCT))
    edge = subj[y0:y1, x0:x1] & ~np.roll(subj[y0:y1, x0:x1], 2, 0)

    panels = []
    for lab, path in items:
        dep = np.asarray(Image.open(path)).astype(np.float32) / 65535.0 * 49.0
        win = dep[y0 + oy:y0 + oy + H * CELL, x0 + ox:x0 + ox + W * CELL]
        tool = np.median(win.reshape(H, CELL, W, CELL), axis=(1, 3))
        D = np.clip((tool - true) / scale, -1, 1)
        big = np.repeat(np.repeat(D, CELL, 0), CELL, 1)
        gm = np.repeat(np.repeat(good, CELL, 0), CELL, 1)
        col = np.zeros((hh, ww, 3), np.float32)
        ch, cw = min(hh, big.shape[0]), min(ww, big.shape[1])
        col[:ch, :cw, 0] = np.clip(big[:ch, :cw], 0, 1) * 255
        col[:ch, :cw, 2] = np.clip(-big[:ch, :cw], 0, 1) * 255
        m = np.zeros((hh, ww, 1), np.float32)
        m[:ch, :cw, 0] = (~gm[:ch, :cw]).astype(np.float32)
        img = (base * (1 - 0.55 * (1 - m)) + col * 0.55 * (1 - m)).astype(np.uint8)
        img[edge] = [255, 255, 0]
        panels.append((lab, img))
    sheet = Image.new('RGB', (len(panels) * (ww + 6), hh + 24), (255, 255, 255))
    from PIL import ImageDraw
    dr = ImageDraw.Draw(sheet)
    for i, (lab, img) in enumerate(panels):
        px = i * (ww + 6)
        dr.text((px + 4, 6), lab, fill=(200, 0, 0))
        sheet.paste(Image.fromarray(img), (px, 20))
    sheet.save(out)
    print('wrote', out, sheet.size)


main()
