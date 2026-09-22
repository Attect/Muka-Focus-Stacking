"""Read the annotation layers the user painted over a mkanno.py base image.

The base is a half-scale copy of the product with a 46 px border and a 36 px
caption strip, so a mark pixel (px, py) sits at output coordinate
    ((px - 46) * 2, (py - 82) * 2)
and canvas coordinate = output + (12, 18).

Usage:
    python tools/markanchor.py
Prints the connected components of each layer in canvas coordinates and writes
out/_mark_black.npy / out/_mark_blue.npy (masks in base-image pixels).
"""
import sys

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

PAD, TOP, S = 46, 82, 2          # x offset, y offset, base-to-output scale
CANVAS_FROM_OUT = (12, 18)


def mask_of(path, colour):
    """Alpha mask, restricted to pixels whose hue matches `colour`."""
    a = np.asarray(Image.open(path).convert('RGBA')).astype(np.int16)
    rgb, al = a[..., :3], a[..., 3]
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    if colour == 'blue':
        hue = (b - r > 40) & (b - g > 40)
    else:
        hue = (r < 90) & (g < 90) & (b < 90)
    return (al > 40) & hue, al


def components(mask, cell=12, min_px=40):
    """Connected components over an occupancy grid; returns (bbox, count) each
    in base-image pixels."""
    H, W = mask.shape
    gh, gw = (H + cell - 1) // cell, (W + cell - 1) // cell
    dens = np.zeros((gh, gw), np.int32)
    ys, xs = np.nonzero(mask)
    np.add.at(dens, (ys // cell, xs // cell), 1)
    occ = dens > 0
    lab = np.zeros((gh, gw), np.int32)
    out = []
    for i in range(gh):
        for j in range(gw):
            if not occ[i, j] or lab[i, j]:
                continue
            cur = len(out) + 1
            stack = [(i, j)]
            lab[i, j] = cur
            cells = []
            while stack:
                y, x = stack.pop()
                cells.append((y, x))
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < gh and 0 <= nx < gw and occ[ny, nx] and not lab[ny, nx]:
                            lab[ny, nx] = cur
                            stack.append((ny, nx))
            sel = lab == cur
            px = int(dens[sel].sum())
            if px < min_px:
                continue
            yy, xx = np.nonzero(sel)
            # exact extent, from the mask itself
            m = mask[yy.min() * cell:(yy.max() + 1) * cell, xx.min() * cell:(xx.max() + 1) * cell]
            sub = mask.copy()
            keep = np.zeros_like(mask)
            keep[yy.min() * cell:(yy.max() + 1) * cell, xx.min() * cell:(xx.max() + 1) * cell] = m
            keep &= mask
            ksy, ksx = np.nonzero(keep)
            out.append((px, ksx.min(), ksx.max(), ksy.min(), ksy.max(),
                        float(ksx.mean()), float(ksy.mean())))
    out.sort(reverse=True)
    return out


def to_canvas(v, axis):
    off = PAD if axis == 0 else TOP
    return int(round((v - off) * S + CANVAS_FROM_OUT[axis]))


def report(name, path, colour, cell, min_px):
    mask, al = mask_of(path, colour)
    print('== %s   (%s)  标记像素 %d' % (name, path, mask.sum()))
    comps = components(mask, cell, min_px)
    print('   %-4s %8s %-24s %-24s %s' % ('#', 'pixels', 'canvas x', 'canvas y', 'size(px)'))
    for k, (n, x0, x1, y0, y1, cx, cy) in enumerate(comps[:40]):
        print('   %-4d %8d  %6d..%-6d        %6d..%-6d        %dx%d'
              % (k, n, to_canvas(x0, 0), to_canvas(x1, 0),
                 to_canvas(y0, 1), to_canvas(y1, 1),
                 (x1 - x0 + 1) * S, (y1 - y0 + 1) * S))
    print('   共 %d 个连通块（>=%d px）' % (len(comps), min_px))
    return mask


black = report('模糊（黑）', 'out/标注_模糊处.png', 'black', cell=10, min_px=40)
np.save('out/_mark_black.npy', black)
print()
blue = report('光晕（蓝）', 'out/标注_光晕处.png', 'blue', cell=14, min_px=30)
np.save('out/_mark_blue.npy', blue)
