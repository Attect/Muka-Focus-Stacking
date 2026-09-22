"""Texture and luminance as a function of *signed* distance across a silhouette.

Negative distance is inside the figure, positive is outside, both measured on the
same rings so the two sides can be compared directly. Texture is the local std of
a 5x5 high-pass.

The point of the signed version: a fusion that takes detail bands from the wrong
frame near an edge loses the wall's relief in a band outside the figure, and the
stretch of wall that is smooth while its neighbours are textured reads as the
nearer object glowing onto the wall.

Usage:
    python tools/sideprofile.py WINDOW_X0,X1,Y0,Y1 [--full] label=path [...]
"""
import sys

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
sys.path.insert(0, 'tools')
from patchlib import gray, load, locate  # noqa: E402
from bandlib import boxsum, subject_mask, texture_of  # noqa: E402

DS = (-24, -16, -10, -6, -3, 3, 6, 10, 16, 24, 40, 70)


def main():
    args = sys.argv[1:]
    x0, x1, y0, y1 = (int(v) for v in args.pop(0).split(','))
    items = [a.split('=', 1) for a in args]
    B = load(items[0][1])
    subj, _ = subject_mask(B)
    sw = subj[y0:y1, x0:x1]
    hh, ww = sw.shape
    dmax = max(DS)
    # Two independent distance fields, then one signed field: the two have to be
    # built separately, because a single pass would let the second overwrite the
    # first wherever the two sides approach within the cap.
    def dist(mask):
        out = np.full((hh, ww), np.int16(dmax + 1), np.int16)
        prev = np.zeros((hh, ww), bool)
        for k in range(0, dmax + 1):
            c = boxsum(mask, k) > 0
            out[c & ~prev] = k
            prev |= c
        return out

    dwall, dsub = dist(~sw), dist(sw)
    # outside the figure the distance to the figure is what matters, and inside
    # it the distance to the wall
    sign = np.where(sw, -dwall, dsub).astype(np.int16)
    sign[np.minimum(dwall, dsub) > dmax] = 0
    print('window %dx%d (%d,%d)-(%d,%d)   figure %d px, wall %d px'
          % (ww, hh, x0, y0, x1, y1, sw.sum(), (~sw).sum()))
    print('%-26s' % 'signed d' + ''.join('%8d' % d for d in DS))
    for lab, path in items:
        img = load(path)
        G = gray(img)
        T = texture_of(G)
        r = c = 0
        if img.shape[1] < 6000:
            r, c, _ = locate(B)
        winG = G[y0 - r:y0 - r + hh, x0 - c:x0 - c + ww]
        winT = T[y0 - r:y0 - r + hh, x0 - c:x0 - c + ww]
        row = '%-26s' % lab
        row2 = '%-26s' % ''
        for d in DS:
            m = sign == d
            row += '%8.2f' % (float(winT[m].mean()) if m.sum() > 20 else np.nan)
            row2 += '%8.1f' % (float(winG[m].mean()) if m.sum() > 20 else np.nan)
        print(row + '   texture')
        print(row2 + '   luminance')
        del G, T, img
    print('%-26s' % 'ring size (px)' + ''.join('%8d' % (sign == d).sum() for d in DS))


main()
