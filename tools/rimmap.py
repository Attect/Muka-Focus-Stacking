"""Map where the output differs from the per-pixel depth truth, at cell resolution.

Red = brighter than the truth, blue = darker. Anything sharing the shape of a
silhouette here is fusion light that no frame contains.

    python tools/rimmap.py out.png scale label=path [...]
"""
import sys

import numpy as np
from PIL import Image, ImageDraw

Image.MAX_IMAGE_PIXELS = None
sys.path.insert(0, 'tools')
from patchlib import gray, load  # noqa: E402

CELL = 8
CANVAS_FROM_OUT = (12, 18)
BOX = (2200, 3220, 1420, 2260)
PRODUCT = 'out/可莉-全清晰.png'


def main():
    out = sys.argv[1]
    scale = float(sys.argv[2])
    items = [a.split('=', 1) for a in sys.argv[3:]]
    ref = np.load('out/_true_lum_cell.npy')
    good = np.load('out/_cell_good.npy')
    H, W = ref.shape
    x0, x1, y0, y1 = BOX
    oy, ox = CANVAS_FROM_OUT
    base = np.asarray(load(PRODUCT), dtype=np.float32)[y0:y1, x0:x1]
    hh, ww = base.shape[:2]
    panels = []
    for lab, path in items:
        G = gray(load(path))
        cl = G[y0:y0 + H * CELL, x0:x0 + W * CELL].reshape(H, CELL, W, CELL).mean(axis=(1, 3))
        D = np.clip((cl - ref) / scale, -1, 1)
        big = np.repeat(np.repeat(D, CELL, 0), CELL, 1)
        gm = np.repeat(np.repeat(good, CELL, 0), CELL, 1)
        col = np.zeros((hh, ww, 3), np.float32)
        ch, cw = min(hh, big.shape[0]), min(ww, big.shape[1])
        col[:ch, :cw, 0] = np.clip(big[:ch, :cw], 0, 1) * 255
        col[:ch, :cw, 2] = np.clip(-big[:ch, :cw], 0, 1) * 255
        m = np.zeros((hh, ww, 1), np.float32)
        m[:ch, :cw, 0] = (~gm[:ch, :cw]).astype(np.float32)
        panels.append((lab, (base * (1 - 0.6 * (1 - m)) + col * 0.6 * (1 - m)).astype(np.uint8)))
    sheet = Image.new('RGB', (len(panels) * (ww + 6), hh + 24), (255, 255, 255))
    dr = ImageDraw.Draw(sheet)
    for i, (lab, img) in enumerate(panels):
        px = i * (ww + 6)
        dr.text((px + 4, 6), lab, fill=(200, 0, 0))
        sheet.paste(Image.fromarray(img), (px, 20))
    sheet.save(out)
    print('wrote', out, sheet.size)


main()
