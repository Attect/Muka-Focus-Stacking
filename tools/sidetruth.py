"""Both sides of a silhouette against their own ground truth.

`depthtruth.py` measures the wall: the focus response is aggregated with the
figure's pixels excluded, so the figure's edge cannot vote. This does the same on
*both* sides -- inside the figure the wall is excluded instead -- and reports the
result against the *signed* distance from the silhouette, so an excess of light on
either side shows up in the same table.

The point: a band that reads as "the nearer object glows onto the wall" can be
light gained on the object's side, light gained on the wall's side, or both.

Usage:
    python tools/sidetruth.py WINDOW_X0,X1,Y0,Y1 label=path [...]
"""
import os
import sys

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
sys.path.insert(0, 'tools')
from patchlib import gray, load, locate  # noqa: E402
from bandlib import boxsum, subject_mask  # noqa: E402

D = os.environ.get("STACK_DIR", "stack")
CANVAS_FROM_OUT = (12, 18)
CELL = 8
CB = 6
BINS = (-40, -24, -16, -12, -8, -5, -3, 3, 5, 8, 12, 16, 24, 40)


def main():
    args = sys.argv[1:]
    x0, x1, y0, y1 = (int(v) for v in args.pop(0).split(','))
    items = [a.split('=', 1) for a in args]
    B = load(items[0][1])
    subj, _ = subject_mask(B)
    sw = subj[y0:y1, x0:x1]
    # The colour rule plus the morphological opening leaves the figure about
    # three pixels small, so ring +3 would still be standing on the figure's own
    # edge - where a sharp output reads far brighter than a defocused reference.
    # Grow it back before measuring the distance, or the rim is an artefact of
    # the mask. (The hand-retouched export "shows" the same +13 rim otherwise.)
    dil = int(os.environ.get('FIG_DILATE', '4'))
    if dil:
        sw = boxsum(sw, dil) > 0
    hh, ww = sw.shape
    oy, ox = CANVAS_FROM_OUT
    cy0, cx0, cy1, cx1 = y0 + oy, x0 + ox, y1 + oy, x1 + ox
    wallw = ~sw

    def dist(mask, cap=60):
        out = np.full((hh, ww), cap + 1, np.int16)
        prev = np.zeros((hh, ww), bool)
        for k in range(0, cap + 1):
            c = boxsum(mask, k) > 0
            out[c & ~prev] = k
            prev |= c
        return out

    sign = np.where(sw, -dist(wallw), dist(sw)).astype(np.int16)

    ncy, ncx = hh // CELL, ww // CELL
    SL = (slice(0, ncy * CELL), slice(0, ncx * CELL))

    def cells(a):
        return a[SL].reshape(ncy, CELL, ncx, CELL)

    cell_sign = np.median(cells(sign), axis=(1, 3))
    cell_subj = cells(sw.astype(np.float32)).mean(axis=(1, 3))
    good = np.abs(cell_sign) < 55

    files = sorted(f for f in os.listdir(D)
                   if f.upper().startswith('DSC') and f.upper().endswith('.JPG'))
    bestw = np.full((ncy, ncx), -1.0, np.float32)
    bests = np.full((ncy, ncx), -1.0, np.float32)
    bestkw = np.zeros((ncy, ncx), np.int16)
    bestks = np.zeros((ncy, ncx), np.int16)
    for k, f in enumerate(files):
        g = gray(np.asarray(Image.open(os.path.join(D, f)).convert('RGB')
                            .crop((cx0, cy0, cx1, cy1)), dtype=np.float32))
        gy, gx = np.gradient(g)
        mag = np.hypot(gx, gy)
        for lab, mask, best, bk in (('w', wallw, bestw, bestkw), ('s', sw, bests, bestks)):
            n = boxsum(mask, CB)
            resp = boxsum(mag * mask, CB) / np.maximum(n, 1.0)
            rc = cells(resp).mean(axis=(1, 3))
            up = (n[SL].reshape(ncy, CELL, ncx, CELL).mean(axis=(1, 3)) > 0.8 * CELL * CELL) & (rc > best)
            best[up] = rc[up]
            bk[up] = k
        if (k + 1) % 10 == 0:
            print('  %d/%d' % (k + 1, len(files)), flush=True)

    usek = np.where(cell_subj > 0.5, bestks, bestkw)
    lum = np.zeros((ncy, ncx), np.float32)
    for k in np.unique(usek):
        g = gray(np.asarray(Image.open(os.path.join(D, files[int(k)])).convert('RGB')
                            .crop((cx0, cy0, cx1, cy1)), dtype=np.float32))
        gc = cells(g).mean(axis=(1, 3))
        m = usek == k
        lum[m] = gc[m]
    np.save('out/_side_ref_lum.npy', lum)
    np.save('out/_side_sign.npy', cell_sign)
    np.save('out/_side_good.npy', good)

    print()
    print('真值帧: 主体侧 %s   墙侧 %s'
          % (np.unique(bestks[good & (cell_subj > 0.5)])[:6],
             np.unique(bestkw[good & (cell_subj < 0.5)])[:6]))
    hdr = '%-24s' % 'signed d' + ''.join('%7d' % d for d in BINS)
    print(hdr)
    for lab, path in items:
        img = load(path)
        G = gray(img)
        r = c = 0
        if img.shape[1] < 6000:
            r, c, _ = locate(B)
        win = G[y0 - r:y0 - r + ncy * CELL, x0 - c:x0 - c + ncx * CELL]
        cl = cells(win).mean(axis=(1, 3))
        row = '%-24s' % lab
        for d in BINS:
            m = good & (cell_sign == d)
            row += '%7.2f' % (float((cl[m] - lum[m]).mean()) if m.sum() > 4 else np.nan)
        print(row)
        del G, img
    print('%-24s' % 'cells per bin' + ''.join('%7d' % (good & (cell_sign == d)).sum()
                                              for d in BINS))


main()
