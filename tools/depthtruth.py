"""Ground truth for the wall *without* using the tool's own depth.

An earlier version built the reference by sampling "the frame the saved depth
picks". That is circular: if the tool's depth at a pixel is wrong for the reason
being investigated, the reference is wrong in the same way and the difference
comes out near zero. (It did: +0.93, while the eye still sees the band.)

Here the true frame is measured instead: for every 8x8 cell of the region, the
focus response is computed with the *subject pixels excluded from the window*, so
the strong edge of the figure cannot vote. The frame that wins is by construction
a frame in which the wall itself is sharp.

Reports, per distance band outside the silhouette:
  * the tool's depth against that truth,
  * each variant's luminance against the reference built from it.

Usage:
    python tools/depthtruth.py --depth out/d4_final.png [--bbox x0,x1,y0,y1]
        label=path [...]
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
CB = 6            # window radius, in pixels, of the masked focus response


def main():
    args = sys.argv[1:]
    bbox = None
    if '--bbox' in args:
        i = args.index('--bbox')
        bbox = tuple(int(v) for v in args[i + 1].split(','))
        del args[i:i + 2]
    depth_path = 'out/d4_final.png'
    if '--depth' in args:
        i = args.index('--depth')
        depth_path = args[i + 1]
        del args[i:i + 2]
    items = [a.split('=', 1) for a in args]

    B = load(items[0][1])
    subj, _ = subject_mask(B)
    H, W = subj.shape
    x0, x1, y0, y1 = bbox or (0, W, 0, H)
    oy, ox = CANVAS_FROM_OUT
    cy0, cx0, cy1, cx1 = y0 + oy, x0 + ox, y1 + oy, x1 + ox
    wallw = ~subj[cy0 - oy:cy1 - oy, cx0 - ox:cx1 - ox]

    # Distance of every pixel of the window from the subject, capped.
    subjw = ~wallw                      # distance is measured from the FIGURE
    dc = np.full(wallw.shape, 999, np.int16)
    prev = np.zeros(wallw.shape, bool)
    for k in range(0, 121):
        cur = boxsum(subjw, k) > 0
        dc[cur & ~prev] = k
        prev |= cur
        if bool(prev.all()):
            break
    print('距离分布: <=8 %d, 8-24 %d, 24-72 %d, >72 %d (窗口像素)'
          % ((dc <= 8).sum(), ((dc > 8) & (dc <= 24)).sum(),
             ((dc > 24) & (dc <= 72)).sum(), (dc > 72).sum()))
    hh, ww = wallw.shape
    ncy, ncx = hh // CELL, ww // CELL
    SL = (slice(0, ncy * CELL), slice(0, ncx * CELL))

    def cells(a):
        return a[SL].reshape(ncy, CELL, ncx, CELL)

    cell_d = np.median(cells(dc), axis=(1, 3))
    cell_wall = cells(wallw.astype(np.float32)).mean(axis=(1, 3))
    good = (cell_wall > 0.9) & (cell_d < 90)
    print('窗口 %dx%d  单元 %dx%d  可用单元 %d' % (ww, hh, ncx, ncy, good.sum()))

    files = sorted(f for f in os.listdir(D)
                   if f.upper().startswith('DSC') and f.upper().endswith('.JPG'))
    n = len(files)
    best = np.full((ncy, ncx), -1.0, np.float32)
    bestk = np.zeros((ncy, ncx), np.int16)
    lum = np.zeros((ncy, ncx), np.float32)
    for k, f in enumerate(files):
        g = gray(np.asarray(Image.open(os.path.join(D, f)).convert('RGB')
                            .crop((cx0, cy0, cx1, cy1)), dtype=np.float32))
        gy, gx = np.gradient(g)
        mag = np.hypot(gx, gy)
        num = boxsum(mag * wallw, CB)
        den = np.maximum(boxsum(wallw, CB), 1.0)
        resp = num / den
        rc = cells(resp).mean(axis=(1, 3))
        up = rc > best
        best[up] = rc[up]
        bestk[up] = k
        print('  %d/%d' % (k + 1, n), flush=True)
    # luminance of the reference, from the winning frame of each cell
    for k in np.unique(bestk):
        g = gray(np.asarray(Image.open(os.path.join(D, files[int(k)])).convert('RGB')
                            .crop((cx0, cy0, cx1, cy1)), dtype=np.float32))
        gc = cells(g).mean(axis=(1, 3))
        sel = bestk == k
        lum[sel] = gc[sel]
    np.save('out/_true_depth_cell.npy', bestk)
    np.save('out/_true_lum_cell.npy', lum)
    np.save('out/_cell_d.npy', cell_d)
    np.save('out/_cell_good.npy', good)

    dep = np.asarray(Image.open(depth_path)).astype(np.float32) / 65535.0 * 49.0
    dcw = dep[cy0:cy1, cx0:cx1]
    toold = np.median(cells(dcw), axis=(1, 3))

    print()
    print('%-8s %6s %8s %8s %8s' % ('d(px)', 'cells', 'true', 'tool', 'tool-true'))
    for a, b in [(2, 8), (8, 16), (16, 24), (24, 32), (32, 48), (48, 72), (72, 100)]:
        sel = good & (cell_d >= a) & (cell_d < b)
        if sel.sum() < 4:
            continue
        print('%-8s %6d %8.1f %8.1f %+8.1f'
              % ('%d-%d' % (a, b), sel.sum(), np.median(bestk[sel]), np.median(toold[sel]),
                 np.median(toold[sel]) - np.median(bestk[sel])))

    print()
    print('%-12s %8s %8s %10s' % ('variant', '4-24', '24-72', 'd=8/d=16'))
    for lab, path in items:
        img = load(path)
        G = gray(img)
        r = c = 0
        if img.shape[0] != H:
            r, c, _ = locate(img)
        # the variant's own coordinates: output images share the output grid, the
        # reference export is its own crop
        win = G[y0 - r:y0 - r + ncy * CELL, x0 - c:x0 - c + ncx * CELL]
        cellL = cells(win).mean(axis=(1, 3))
        near = good & (cell_d >= 4) & (cell_d < 24)
        far = good & (cell_d >= 24) & (cell_d < 72)
        d8 = good & (cell_d >= 4) & (cell_d < 12)
        d16 = good & (cell_d >= 12) & (cell_d < 24)
        print('%-12s %+8.2f %+8.2f %10s'
              % (lab, float((cellL[near] - lum[near]).mean()),
                 float((cellL[far] - lum[far]).mean()),
                 '%+.2f / %+.2f' % (float((cellL[d8] - lum[d8]).mean()),
                                    float((cellL[d16] - lum[d16]).mean()))))
        del G, img


main()
