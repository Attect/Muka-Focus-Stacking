"""Brightening as a function of distance from the silhouette.

A light band is a broad, low-contrast feature, so it is invisible unless the
slow background variation is removed first. This measures, for each distance band
outside the subject, the mean of `G - box(G, 100)` -- how much brighter that ring
is than its own large-scale neighbourhood -- and does it for every variant with
the same subject mask, so the curves are directly comparable.

Usage:
    python tools/bandprofile.py --bbox x0,x1,y0,y1 [--ref] label=path [...]
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


def rings(subj, rmax):
    """ring[k] = subject-distance exactly k, for k = 0..rmax."""
    prev = np.zeros(subj.shape, bool)
    out = []
    for k in range(rmax + 1):
        cur = boxsum(subj, k) > 0
        out.append(cur & ~prev)
        prev = cur
    return out


def curve(G, subj, bbox, rmax=46, bgr=100):
    x0, x1, y0, y1 = bbox
    roi = np.zeros(subj.shape, bool)
    roi[y0:y1, x0:x1] = True
    hf = G - box(G, bgr)
    out = np.full(rmax + 1, np.nan)
    for k, rk in enumerate(rings(subj, rmax)):
        m = rk & roi
        if m.sum() > 20:
            out[k] = float(hf[m].mean())
    return out


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
    args = [a for a in args if a != '--frames']

    variants = []
    for a in args:
        lab, path = a.split('=', 1)
        variants.append((lab, load(path)))

    B = variants[0][1]
    g0 = gray(B)
    sat = B.max(-1) - B.min(-1)
    subj0 = (sat > 30) | (g0 < 100) | (g0 > 215)
    k = 3
    er = boxsum(subj0, k) >= float((2 * k + 1) ** 2)
    subj = boxsum(er, k) > 0

    print('%-12s' % 'distance' + ''.join('%9s' % l for l, _ in variants))
    curves = {}
    for lab, img in variants:
        G = gray(img)
        origin = None
        if img.shape[0] != B.shape[0]:
            oy, ox, _ = locate(img)
            origin = (oy, ox)
        if origin is None:
            s = subj
        else:
            s = np.zeros(img.shape[:2], bool)
            h = min(s.shape[0], subj.shape[0] - origin[0])
            w = min(s.shape[1], subj.shape[1] - origin[1])
            s[:h, :w] = subj[origin[0]:origin[0] + h, origin[1]:origin[1] + w]
        curves[lab] = curve(G, s, bbox)
        del G
    for k2 in range(0, 47):
        row = '%-12d' % k2
        for lab, _ in variants:
            v = curves[lab][k2]
            row += '%9s' % ('%.2f' % v if np.isfinite(v) else '-')
        print(row)
    with open('out/bandprofile.csv', 'w') as f:
        f.write('distance,' + ','.join(l for l, _ in variants) + '\n')
        for k2 in range(0, 47):
            f.write('%d,' % k2 + ','.join('%.4f' % curves[l][k2] for l, _ in variants) + '\n')
    print('wrote out/bandprofile.csv')

    if want_ref:
        R = load(REF)
        oy, ox, _ = locate(B)
        s = np.zeros(R.shape[:2], bool)
        h = min(s.shape[0], subj.shape[0] - oy)
        w = min(s.shape[1], subj.shape[1] - ox)
        s[:h, :w] = subj[oy:oy + h, ox:ox + w]
        c = curve(gray(R), s, bbox)
        print('AFref       ' + ' '.join('%.2f' % v if np.isfinite(v) else '-' for v in c[:48:4]))
        print('   AFref 逐距离: ' + ' '.join('%d:%.2f' % (k2, c[k2]) for k2 in range(0, 46, 4)))
    for lab, _ in variants:
        print('   %-8s 逐距离: ' % lab + ' '.join('%d:%.2f' % (k2, curves[lab][k2])
                                                   for k2 in range(0, 46, 4)))


main()
