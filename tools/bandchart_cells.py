"""Chart the wall band against the *masked* ground truth (see depthtruth.py).

Reads the cached cell data so the bracket does not have to be decoded again.

    python tools/bandchart_cells.py out.png label=path [...]
"""
import sys

import numpy as np
from PIL import Image, ImageDraw

Image.MAX_IMAGE_PIXELS = None
sys.path.insert(0, 'tools')
from patchlib import gray, load, locate  # noqa: E402

CELL = 8
CANVAS_FROM_OUT = (12, 18)
SHOW = (4, 8, 12, 16, 22, 30, 40, 56, 76)
COLS = [(0, 0, 255), (220, 0, 0), (0, 150, 0), (230, 130, 0), (150, 0, 200), (110, 110, 110)]


def main():
    out = sys.argv[1]
    items = [a.split('=', 1) for a in sys.argv[2:]]
    ref = np.load('out/_true_lum_cell.npy')
    d = np.load('out/_cell_d.npy')
    good = np.load('out/_cell_good.npy')
    H, W = ref.shape
    x0, x1, y0, y1 = 2200, 3220, 1420, 2260
    curves = [('ground truth (wall in focus)', ref, (0, 0, 0))]
    for i, (lab, path) in enumerate(items):
        img = load(path)
        G = gray(img)
        r = c = 0
        if img.shape[1] < 6000:      # the reference export is its own crop
            r, c, _ = locate(load('out/可莉-全清晰.png'))
        win = G[y0 - r:y0 - r + H * CELL, x0 - c:x0 - c + W * CELL]
        cellL = win.reshape(H, CELL, W, CELL).mean(axis=(1, 3))
        curves.append((lab, cellL, COLS[i % len(COLS)]))
        del G, img

    BIN = 4

    def curve(c):
        o = []
        for k in range(0, 92 // BIN):
            m = good & (d >= k * BIN) & (d < (k + 1) * BIN)
            o.append(float(c[m].mean()) if m.sum() > 8 else np.nan)
        return np.array(o)

    refc = curve(ref)
    CW, CH = 1180, 470
    im = Image.new('RGB', (CW, CH), (255, 255, 255))
    dr = ImageDraw.Draw(im)
    lo, hi = 176.0, 190.0
    W1 = 580

    def px(v):
        return 46 + v / 92.0 * (W1 - 60)

    def py(v):
        return CH - 48 - (v - lo) / (hi - lo) * (CH - 92)

    dr.rectangle([46, 24, W1 - 14, CH - 48], outline=(190, 190, 190))
    for v in range(177, 187, 2):
        dr.line([46, py(v), W1 - 14, py(v)], fill=(238, 238, 238))
        dr.text((12, py(v) - 6), str(v), fill=(90, 90, 90))
    for v in range(0, 93, 20):
        dr.line([px(v), 24, px(v), CH - 48], fill=(238, 238, 238))
        dr.text((px(v) - 6, CH - 42), str(v), fill=(90, 90, 90))
    dr.text((46, 6), 'wall luminance by distance from the silhouette (px)', fill=(0, 0, 0))
    dr.text((W1 - 90, CH - 42), 'd (px)', fill=(90, 90, 90))
    for lab, c, col in curves:
        cur = curve(c)
        pts = [(px(k * BIN + BIN / 2.0), py(v)) for k, v in enumerate(cur) if np.isfinite(v)]
        dr.line(pts, fill=col, width=2)

    X2 = W1 + 26
    lo2, hi2 = -1.0, 9.0

    def px2(v):
        return X2 + 34 + v / 92.0 * (CW - X2 - 50)

    def py2(v):
        return CH - 48 - (v - lo2) / (hi2 - lo2) * (CH - 92)

    dr.rectangle([X2 + 34, 24, CW - 14, CH - 48], outline=(190, 190, 190))
    for v in range(0, 7):
        dr.line([X2 + 34, py2(v), CW - 14, py2(v)], fill=(238, 238, 238))
        dr.text((X2 + 8, py2(v) - 6), '%+d' % v, fill=(90, 90, 90))
    dr.line([X2 + 34, py2(0), CW - 14, py2(0)], fill=(120, 120, 120))
    for v in range(0, 93, 20):
        dr.text((px2(v) - 6, CH - 42), str(v), fill=(90, 90, 90))
    dr.text((X2 + 34, 6), 'minus ground truth: the light band', fill=(0, 0, 0))
    for lab, c, col in curves:
        if lab.startswith('ground'):
            continue
        cur = curve(c) - refc
        pts = [(px2(k * BIN + BIN / 2.0), py2(v)) for k, v in enumerate(cur) if np.isfinite(v)]
        dr.line(pts, fill=col, width=2)
    for i, (lab, _, col) in enumerate(curves):
        dr.text((50 + i * 200, CH - 24), lab[:30], fill=col)
    im.save(out)
    print('wrote', out)
    for lab, c, _ in curves:
        cur = curve(c)
        print('%-32s ' % lab + '  '.join('d=%d %+.1f' % (k, cur[k // BIN] - refc[k // BIN])
                                          for k in SHOW))


main()
