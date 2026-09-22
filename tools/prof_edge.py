"""Perpendicular profiles across a contour the user marked in red.

The red strokes in the annotation file are taken as anchor points on the
contour; for each anchor the local tangent is estimated by PCA over nearby
stroke pixels, and luminance is sampled along the normal at offsets -R..+R.
Negative offsets are on the side the user's stroke points away from (we orient
every normal so that +offset goes from the lighter side to the darker side, so
all profiles are comparable).

Usage:
    python tools/prof_edge.py MARKED.png out.png [label=path ...] [--ref] [--split]
"""
import sys

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
sys.path.insert(0, 'tools')
from patchlib import REF, gray, load, locate  # noqa: E402


def red_mask(path):
    a = np.asarray(Image.open(path).convert('RGB')).astype(np.int16)
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    return (r > 140) & (r - g > 60) & (r - b > 60)


def anchors(mask, step=8):
    ys, xs = np.nonzero(mask)
    pts = np.stack([ys, xs], 1).astype(np.float64)
    order = np.lexsort((xs, ys))
    pts = pts[order]
    keep = []
    for i in range(0, len(pts), step):
        keep.append(pts[i])
    return np.array(keep)


def sample(img, ys, xs):
    h, w = img.shape[:2]
    ys = np.clip(ys, 0, h - 1.001)
    xs = np.clip(xs, 0, w - 1.001)
    y0 = np.floor(ys).astype(int)
    x0 = np.floor(xs).astype(int)
    fy = ys - y0
    fx = xs - x0
    g = gray(img.astype(np.float32))
    v00 = g[y0, x0]
    v01 = g[y0, x0 + 1]
    v10 = g[y0 + 1, x0]
    v11 = g[y0 + 1, x0 + 1]
    return (v00 * (1 - fy) * (1 - fx) + v01 * (1 - fy) * fx
            + v10 * fy * (1 - fx) + v11 * fy * fx)


def profiles(img, pts, mask, R=36):
    """Return array [n_anchor, 2R+1] of luminance along the normal."""
    m = mask
    n = len(pts)
    out = np.zeros((n, 2 * R + 1))
    for k, (y, x) in enumerate(pts):
        y0, y1 = int(max(0, y - 14)), int(min(m.shape[0], y + 15))
        x0, x1 = int(max(0, x - 14)), int(min(m.shape[1], x + 15))
        sub = np.argwhere(m[y0:y1, x0:x1]) + [y0, x0]
        if len(sub) < 4:
            out[k] = np.nan
            continue
        c = sub - sub.mean(0)
        cov = c.T @ c / len(c)
        w, v = np.linalg.eigh(cov)
        t = v[:, -1]
        nn = np.array([-t[1], t[0]])
        off = np.arange(-R, R + 1)
        ys = y + nn[0] * off
        xs = x + nn[1] * off
        p = sample(img, ys, xs)
        a = int(np.argmin(np.abs(off + R // 2)))
        b = int(np.argmin(np.abs(off - R // 2)))
        if p[a] > p[b]:
            p = p[::-1]
        out[k] = p
    return out


def main():
    args = [a for a in sys.argv[1:]]
    marked = args.pop(0)
    outpath = args.pop(0)
    want_ref = '--ref' in args
    if '--ref' in args:
        args.remove('--ref')
    R = 40
    m = red_mask(marked)
    pts = anchors(m, 8)
    print('锚点 %d 个' % len(pts))
    variants = []
    if want_ref:
        ref = load(REF)
        variants.append(('AF_ref', ref, (0, 0), False))
    for a in args:
        lab, path = a.split('=', 1)
        img = load(path)
        oy, ox, mad = locate(img)
        variants.append((lab, img, (oy, ox), True))
        print('%-14s crop origin in output = (y %d, x %d)  mad %.2f' % (lab, oy, ox, mad))
    res = {}
    for lab, img, (oy, ox), isout in variants:
        if isout:
            p = pts - [oy, ox]
        else:
            p = pts
        pr = profiles(img, p, m if not isout else m, R=R)
        res[lab] = pr
        mean = np.nanmean(pr, 0)
        print('%-14s  min %6.1f  edge-adjacent(-8) %6.1f  (+8) %6.1f  峰值偏移 %d'
              % (lab, np.nanmin(mean), mean[R - 8], mean[R + 8],
                 int(np.argmax(mean) - R)))
    # 输出
    off = np.arange(-R, R + 1)
    lines = ['offset,' + ','.join(res)]
    for i, o in enumerate(off):
        lines.append('%d,' % o + ','.join('%.3f' % np.nanmean(res[k][:, i]) for k in res))
    open(outpath, 'w').write('\n'.join(lines))
    print('wrote', outpath)


main()
