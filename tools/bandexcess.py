"""Is the wall just outside the figure brighter than the wall further away?

The wall has its own slow brightness gradient, so a raw band-minus-far difference
is meaningless and a windowed difference converges to zero as the window grows.
This instead measures luminance as a function of distance from the silhouette,
`L(d)` for d = 0..92 px, fits a quadratic to the far part (34..90 px) and reports
how far the near part (4..22 px) rises above that fit. The fit carries the
gradient, so what is left is the local excess: the light band.

Reference values: the *sharpest raw frame* reads about -1.3 (the wall is slightly
darker right beside the figure), the AF export +1.5.

Usage:
    python tools/bandexcess.py [--bbox x0,x1,y0,y1] [--ref] [--frames] label=path [...]
"""
import os
import sys

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
sys.path.insert(0, 'tools')
from patchlib import REF, gray, load, locate  # noqa: E402
from bandlib import boxsum, subject_mask  # noqa: E402

D = os.environ.get("STACK_DIR", "stack")
CANVAS_FROM_OUT = (12, 18)
DR = 92
NEAR = (4, 22)
FIT = (34, 90)


def rings(subj, rmax):
    """rings(d) = pixels whose Chebyshev distance to the subject is exactly d."""
    prev = np.zeros(subj.shape, bool)
    out = []
    for k in range(rmax + 1):
        cur = boxsum(subj, k) > 0
        out.append(cur & ~prev)
        prev = cur
    return out


def subj_in(img_shape, subj, origin):
    """Express the output-grid subject mask in another image's coordinates."""
    s = np.zeros(img_shape, bool)
    h = min(img_shape[0], subj.shape[0] - origin[0])
    w = min(img_shape[1], subj.shape[1] - origin[1])
    s[:h, :w] = subj[origin[0]:origin[0] + h, origin[1]:origin[1] + w]
    return s


def curve(G, rk, mask):
    return np.array([float(G[r & mask].mean()) if (r & mask).sum() > 30 else np.nan
                     for r in rk])


def excess(L):
    d = np.arange(len(L), dtype=np.float64)
    sel = np.isfinite(L) & (d >= FIT[0]) & (d <= FIT[1])
    c = np.polyfit(d[sel], L[sel], 2)
    near = np.isfinite(L) & (d >= NEAR[0]) & (d <= NEAR[1])
    return float((L[near] - np.polyval(c, d[near])).mean())


def show(lab, L):
    print('%-10s %9.2f %9.2f   %s'
          % (lab, np.nanmean(L[NEAR[0]:NEAR[1] + 1]), excess(L),
             '  '.join('%.1f' % L[k] for k in (4, 8, 14, 22, 34))))


def main():
    args = sys.argv[1:]
    bbox = None
    if '--bbox' in args:
        i = args.index('--bbox')
        bbox = tuple(int(v) for v in args[i + 1].split(','))
        del args[i:i + 2]
    want_ref = '--ref' in args
    if want_ref:
        args.remove('--ref')
    want_frames = '--frames' in args
    if want_frames:
        args.remove('--frames')
    items = [a.split('=', 1) for a in args]

    B = load(items[0][1])
    subj, _ = subject_mask(B)
    H, W = subj.shape
    mask = np.ones((H, W), bool)
    if bbox:
        x0, x1, y0, y1 = bbox
        mask = np.zeros((H, W), bool)
        mask[y0:y1, x0:x1] = True
    rk = rings(subj, DR)

    print('%-10s %9s %9s   %s' % ('variant', 'near L', 'excess', 'd=4 / 8 / 14 / 22 / 34'))
    for lab, path in items:
        img = load(path)
        if img.shape[0] == H:
            G, s, m = gray(img), subj, mask
            r = rk
        else:
            oy, ox, _ = locate(img)
            G = gray(img)
            s = subj_in(img.shape[:2], subj, (oy, ox))
            m = subj_in(img.shape[:2], mask, (oy, ox))
            r = rings(s, DR)
        show(lab, curve(G, r, m))
        del G, img
    if want_ref:
        oy, ox, _ = locate(B)
        R = load(REF)
        G = gray(R)
        s = subj_in(R.shape[:2], subj, (oy, ox))
        m = subj_in(R.shape[:2], mask, (oy, ox))
        show('AFref', curve(G, rings(s, DR), m))
        del G

    if want_frames:
        files = sorted(f for f in os.listdir(D)
                       if f.upper().startswith('DSC') and f.upper().endswith('.JPG'))
        oy, ox = CANVAS_FROM_OUT
        ys, xs = np.nonzero(mask)
        cy0, cx0 = ys.min() + oy, xs.min() + ox
        cy1, cx1 = ys.max() + oy + 1, xs.max() + ox + 1
        print()
        print('逐帧 excess（画布窗口 %d,%d..%d,%d）' % (cx0, cy0, cx1, cy1))
        exc = np.zeros(len(files))
        for i, f in enumerate(files):
            g = gray(np.asarray(Image.open(os.path.join(D, f)).convert('RGB')
                                .crop((cx0, cy0, cx1, cy1)), dtype=np.float32))
            s = subj[cy0 - oy:cy1 - oy, cx0 - ox:cx1 - ox]
            exc[i] = excess(curve(g, rings(s, DR), np.ones(g.shape, bool)))
            if (i + 1) % 10 == 0:
                print('  %d/%d' % (i + 1, len(files)), flush=True)
        order = np.argsort(exc)
        print('  excess 最小 8 帧: ' + '  '.join('%d:%+.2f' % (k, exc[k]) for k in order[:8]))
        print('  帧 32 %.2f  帧 33 %.2f  帧 34 %.2f   中位 %+.2f   最大 %+.2f'
              % (exc[32], exc[33], exc[34], np.median(exc), exc.max()))
        np.save('out/_frame_excess.npy', exc)


main()
