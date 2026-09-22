"""Measure the wall band just outside a silhouette, and find the frame that
should have supplied it.

The figure is strongly coloured and the wall is near-neutral grey, so the subject
mask comes from saturation. The band is the wall whose Chebyshev distance to the
subject is in [lo, hi]; the far wall is wall further than `far`. Everything is in
*output* coordinates, so no alignment step can silently measure the wrong place.
Frames and the saved depth map are sampled with the canvas origin, which is
printed by the tool itself (`[crop] output window ...`).

Usage:
    python tools/bandmeasure.py [--bbox x0,x1,y0,y1] [--frames] [--overlay]
        [--ref] label=path [label=path ...]
"""
import os
import sys

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
sys.path.insert(0, 'tools')
from patchlib import REF, gray, load, locate  # noqa: E402

D = os.environ.get("STACK_DIR", "stack")
CANVAS_FROM_OUT = (12, 18)


def boxsum(m, r):
    a = m.astype(np.float32)
    p = np.pad(a, r, mode='constant')
    c = np.pad(p.cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    H, W = m.shape
    y = np.arange(H)
    x = np.arange(W)
    return (c[np.ix_(y + 2 * r + 1, x + 2 * r + 1)] - c[np.ix_(y, x + 2 * r + 1)]
            - c[np.ix_(y + 2 * r + 1, x)] + c[np.ix_(y, x)])


def box(a, r):
    return boxsum(a, r) / float((2 * r + 1) ** 2)


def texture_of(g):
    hp = g - box(g, 2)
    return np.sqrt(np.maximum(box(hp * hp, 2), 0.0))


def main():
    args = sys.argv[1:]
    lo, hi, far = 4, 22, 64
    bbox = None
    flags = {f: (f in args) for f in ('--frames', '--ref', '--overlay')}
    for f in flags:
        if flags[f]:
            args.remove(f)
    if '--bbox' in args:
        i = args.index('--bbox')
        bbox = tuple(int(v) for v in args[i + 1].split(','))
        del args[i:i + 2]

    base = [a for a in args if a.startswith('BASE=')]
    labels = []
    for a in args:
        lab, path = a.split('=', 1)
        labels.append((lab, path))
    B = load(labels[0][1])
    sat = B.max(-1) - B.min(-1)
    g0 = gray(B)
    # The wall is near-neutral mid-grey (~184). The figure is either saturated
    # (red hat, orange hook, brown boot), much darker (the near-black glove and
    # cuff, which have almost no saturation) or much brighter (pale skin, white
    # dress). Saturation alone therefore leaks the dark parts into "wall", which
    # is what made the first band measurement read 53 levels too dark.
    subj0 = (sat > 30) | (g0 < 100) | (g0 > 215)
    # Opening: the dark seam between the wallpaper panels is only a few pixels
    # wide and is "subject" by the colour rule, so it would grow a band of its
    # own (a third of the total). Erode/dilate removes anything that thin while
    # leaving the figure untouched; the removed structures are then excluded
    # from both masks so the seam's own strong gradient cannot inflate the far
    # wall either.
    k = 3
    er = boxsum(subj0, k) >= float((2 * k + 1) ** 2)
    subj = boxsum(er, k) > 0
    thin = subj0 & ~subj
    avoid = boxsum(thin, 8) > 0
    wall = ~subj & ~avoid
    H, W = subj.shape
    if bbox:
        x0, x1, y0, y1 = bbox
    else:
        x0, x1, y0, y1 = 0, W, 0, H
    roi = np.zeros((H, W), bool)
    roi[y0:y1, x0:x1] = True

    band = (boxsum(subj, hi) > 0) & (boxsum(subj, lo) == 0) & wall & roi
    farw = (boxsum(subj, far) == 0) & wall & roi
    print('roi %dx%d   band %.1f%%   far wall %.1f%%  of roi   (subject %.1f%% overall)'
          % (x1 - x0, y1 - y0, 100 * band.sum() / roi.sum(),
             100 * farw.sum() / roi.sum(), 100 * subj.mean()))

    if flags['--overlay']:
        V = B.astype(np.uint8).copy()
        V[farw] = [0, 90, 255]
        V[band] = [0, 220, 0]
        Image.fromarray(V[y0:y1, x0:x1]).save('out/band_mask_view.png')
        print('wrote out/band_mask_view.png')

    oy, ox = CANVAS_FROM_OUT
    dep = np.asarray(Image.open('out/dcur_final.png')).astype(np.float32) / 65535.0 * 49.0
    dc = dep[oy:oy + H, ox:ox + W]
    print()
    print('%-10s %9s %9s %9s %9s %9s %9s' %
          ('variant', 'band tex', 'far tex', 'band L', 'far L', 'band-far L', 'tex ratio'))
    for lab, path in labels:
        img = load(path)
        G = gray(img)
        T = texture_of(G)
        if img.shape[0] != H:
            roy, rox, _ = locate(img)
            b = band[roy:roy + img.shape[0], rox:rox + img.shape[1]]
            f = farw[roy:roy + img.shape[0], rox:rox + img.shape[1]]
            G, T = G[:b.shape[0], :b.shape[1]], T[:b.shape[0], :b.shape[1]]
        else:
            b, f = band, farw
        bt, ft = float(T[b].mean()), float(T[f].mean())
        bl, fl = float(G[b].mean()), float(G[f].mean())
        # Local ratio: compare the band against the far wall in its own
        # neighbourhood, so a wall that is not perfectly parallel to the sensor
        # cannot masquerade as a band effect.
        w = 40
        nb = boxsum(b, w)
        nf = boxsum(f, w)
        lb = box(T * b, w) / np.maximum(nb, 1.0)
        lf = box(T * f, w) / np.maximum(nf, 1.0)
        ok = b & (nb > 800) & (nf > 800)
        ratio = float((lb[ok] / np.maximum(lf[ok], 1e-6)).mean())
        print('%-10s %9.3f %9.3f %9.2f %9.2f %9.2f %9.3f'
              % (lab, bt, ft, bl, fl, bl - fl, ratio))
        del G, T

    if flags['--ref']:
        R = load(REF)
        roy, rox, _ = locate(B)
        b = band[roy:roy + R.shape[0], rox:rox + R.shape[1]]
        f = farw[roy:roy + R.shape[0], rox:rox + R.shape[1]]
        G = gray(R)[:b.shape[0], :b.shape[1]]
        T = texture_of(G)
        w = 40
        nb = boxsum(b, w)
        nf = boxsum(f, w)
        lb = box(T * b, w) / np.maximum(nb, 1.0)
        lf = box(T * f, w) / np.maximum(nf, 1.0)
        ok = b & (nb > 800) & (nf > 800)
        print('%-10s %9.3f %9.3f %9.2f %9.2f %9.2f %9.3f'
              % ('AFref', float(T[b].mean()), float(T[f].mean()),
                 float(G[b].mean()), float(G[f].mean()),
                 float(G[b].mean()) - float(G[f].mean()),
                 float((lb[ok] / np.maximum(lf[ok], 1e-6)).mean())))
        del G, T

    print()
    print('深度（画布坐标，帧号）: band 中位 %.1f   far wall 中位 %.1f   ROI 中位 %.1f'
          % (np.median(dc[band]), np.median(dc[farw]), np.median(dc[roi])))
    print('    band 深度直方图 P10/P50/P90: %.1f / %.1f / %.1f'
          % tuple(np.percentile(dc[band], [10, 50, 90])))
    print('    far  wall P10/P50/P90: %.1f / %.1f / %.1f'
          % tuple(np.percentile(dc[farw], [10, 50, 90])))

    if flags['--frames']:
        files = sorted(f for f in os.listdir(D)
                       if f.upper().startswith('DSC') and f.upper().endswith('.JPG'))
        # Everything in canvas coordinates: output + (12, 18).
        m = 8
        cy0, cx0 = y0 + oy - m, x0 + ox - m
        cy1, cx1 = y1 + oy + m, x1 + ox + m
        ys, xs = np.nonzero(band)
        relb = np.stack([ys + oy - cy0, xs + ox - cx0], 1)
        ys, xs = np.nonzero(farw)
        relf = np.stack([ys + oy - cy0, xs + ox - cx0], 1)
        print()
        print('逐帧纹理（band / far 在同一窗口 %dx%d，画布 %d,%d..%d,%d）'
              % (cy1 - cy0, cx1 - cx0, cx0, cy0, cx1, cy1))
        texb = np.zeros(len(files))
        texf = np.zeros(len(files))
        lumb = np.zeros(len(files))
        lumf = np.zeros(len(files))
        for i, f in enumerate(files):
            a = np.asarray(Image.open(os.path.join(D, f)).convert('RGB')
                           .crop((cx0, cy0, cx1, cy1)), dtype=np.float32)
            g = gray(a)
            t = texture_of(g)
            texb[i] = t[relb[:, 0], relb[:, 1]].mean()
            texf[i] = t[relf[:, 0], relf[:, 1]].mean()
            lumb[i] = g[relb[:, 0], relb[:, 1]].mean()
            lumf[i] = g[relf[:, 0], relf[:, 1]].mean()
            if (i + 1) % 10 == 0:
                print('  %d/%d' % (i + 1, len(files)), flush=True)
        ratio = texb / np.maximum(texf, 1e-6)
        best = np.argsort(texb)[::-1]
        print('  纹理最高 10 帧: ' + '  '.join('%d:%.2f' % (b, texb[b]) for b in best[:10]))
        print('  它们的 band/far 比值: ' + '  '.join('%.3f' % ratio[b] for b in best[:6]))
        print('  该比例的中位数（全 50 帧） %.3f   最锐帧 %.3f'
              % (float(np.median(ratio)), float(ratio[best[0]])))
        print('  亮度 band-far: 最锐帧 %+.2f   中位帧 %+.2f   (band 最锐帧 %.1f)'
              % (lumb[best[0]] - lumf[best[0]],
                 float(np.median(lumb - lumf)), lumb[best[0]]))
        np.save('out/_frame_tex.npy', texb)
        np.save('out/_frame_texfar.npy', texf)
        np.save('out/_frame_lum.npy', lumb)
        np.save('out/_frame_lumfar.npy', lumf)


main()
