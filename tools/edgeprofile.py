"""Luminance and texture profiles across a contour the user marked in red.

The red strokes in the annotation file are anchors on the contour. For every
anchor the local tangent is estimated by PCA over the nearby stroke pixels, and
two quantities are sampled along the normal at offsets -R..+R:

  * luminance (Rec.601 of the RGB output)
  * texture -- local std of a 5x5 high-pass, i.e. how much fine relief is present

Profiles are oriented so that +offset is the *lighter* side, which at these
contours is the wall. Every anchor lands in the same convention, so the mean
profile is meaningful.

Usage:
    python tools/edgeprofile.py MARKED.png [--ref] [--R 44] [--step 6]
        label=output.png [label=output.png ...]

Prints the mean profile and writes out/edgeprofile.csv plus two charts.
"""
import sys

import numpy as np
from PIL import Image, ImageDraw

Image.MAX_IMAGE_PIXELS = None
sys.path.insert(0, 'tools')
from patchlib import REF, gray, load, locate  # noqa: E402


def red_mask(path):
    a = np.asarray(Image.open(path).convert('RGB')).astype(np.int16)
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    return (r > 140) & (r - g > 60) & (r - b > 60)


def anchors(mask, step):
    ys, xs = np.nonzero(mask)
    pts = np.stack([ys, xs], 1)
    order = np.lexsort((pts[:, 1], pts[:, 0]))
    return pts[order][::step].astype(np.float64)


def box(a, r):
    """Box mean over a (2r+1)^2 window, via an integral image."""
    p = np.pad(a, r + 1, mode='edge')
    c = p.cumsum(0).cumsum(1)
    c = np.pad(c, ((1, 0), (1, 0)))
    H, W = a.shape
    y0 = np.arange(H)
    x0 = np.arange(W)
    s = (c[np.ix_(y0 + 2 * r + 1, x0 + 2 * r + 1)] - c[np.ix_(y0, x0 + 2 * r + 1)]
         - c[np.ix_(y0 + 2 * r + 1, x0)] + c[np.ix_(y0, x0)])
    return s / float((2 * r + 1) ** 2)


def texture_of(g):
    hp = g - box(g, 2)
    return np.sqrt(np.maximum(box(hp * hp, 2), 0.0))


def bilinear(a, ys, xs):
    h, w = a.shape
    ys = np.clip(ys, 0, h - 1.001)
    xs = np.clip(xs, 0, w - 1.001)
    y0 = np.floor(ys).astype(int)
    x0 = np.floor(xs).astype(int)
    fy = ys - y0
    fx = xs - x0
    return (a[y0, x0] * (1 - fy) * (1 - fx) + a[y0, x0 + 1] * (1 - fy) * fx
            + a[y0 + 1, x0] * fy * (1 - fx) + a[y0 + 1, x0 + 1] * fy * fx)


def saturation(a):
    mx = a.max(-1)
    mn = a.min(-1)
    return mx - mn


def profiles(G, T, S, pts_img, pts_mask, mask, R):
    """Sample `G`/`T` at `pts_img`; estimate the tangent from `mask` at `pts_mask`.

    The two point sets differ for the reference export: the mask always lives in
    output coordinates, while the reference image needs the same physical point
    expressed in its own crop.
    """
    n = len(pts_img)
    off = np.arange(-R, R + 1)
    lum = np.full((n, len(off)), np.nan)
    tex = np.full((n, len(off)), np.nan)
    for k in range(n):
        y, x = pts_mask[k]
        gy, gx = pts_img[k]
        y0, y1 = int(max(0, y - 16)), int(min(mask.shape[0], y + 17))
        x0, x1 = int(max(0, x - 16)), int(min(mask.shape[1], x + 17))
        sub = np.argwhere(mask[y0:y1, x0:x1]) + [y0, x0]
        if len(sub) < 5:
            continue
        c = sub - sub.mean(0)
        w, v = np.linalg.eigh(c.T @ c / len(c))
        t = v[:, -1]
        nn = np.array([-t[1], t[0]])
        ys = gy + nn[0] * off
        xs = gx + nn[1] * off
        p = bilinear(G, ys, xs)
        q = bilinear(T, ys, xs)
        r = bilinear(S, ys, xs)
        # Orient by saturation, not by luminance: at these contours the wall is
        # near-neutral grey while the figure is strongly coloured, whereas the
        # figure is brighter than the wall in some places (the orange hook) and
        # darker in others (the dark boot). +offset therefore always ends up on
        # the wall side.
        a = int(np.argmin(np.abs(off + R // 2)))
        b = int(np.argmin(np.abs(off - R // 2)))
        if np.nanmean(r[:a + 1]) < np.nanmean(r[b:]):
            p, q = p[::-1], q[::-1]
        lum[k], tex[k] = p, q
    return off, lum, tex


COLS = [(0, 0, 255), (0, 150, 0), (220, 0, 220), (200, 120, 0),
        (0, 0, 0), (120, 120, 120), (0, 170, 200)]


def chart(off, curves, out, title, R, ylab):
    CW, CH = 1000, 430
    im = Image.new('RGB', (CW, CH), (255, 255, 255))
    d = ImageDraw.Draw(im)
    d.rectangle([60, 30, CW - 20, CH - 56], outline=(180, 180, 180))
    allv = np.concatenate([c for _, c in curves])
    allv = allv[np.isfinite(allv)]
    lo, hi = float(allv.min()), float(allv.max())
    pad = (hi - lo) * 0.15 + 1e-6
    lo -= pad
    hi += pad

    def px(o):
        return 60 + (o + R) / (2.0 * R) * (CW - 80)

    def py(v):
        return CH - 56 - (v - lo) / (hi - lo) * (CH - 86)

    d.text((60, 10), title, fill=(0, 0, 0))
    d.text((62, CH - 46), ylab, fill=(60, 60, 60))
    d.line([px(0), 30, px(0), CH - 56], fill=(200, 0, 0))
    d.text((px(0) + 3, CH - 46), 'marked edge', fill=(200, 0, 0))
    for i, (lab, c) in enumerate(curves):
        col = COLS[i % len(COLS)]
        pts = [(px(o), py(v)) for o, v in zip(off, c) if np.isfinite(v)]
        d.line(pts, fill=col, width=2)
        d.text((70 + i * 150, CH - 30), lab, fill=col)
    im.save(out)
    print('wrote', out)


def main():
    args = sys.argv[1:]
    marked = args.pop(0)
    R, step = 44, 6
    want_ref = '--ref' in args
    if want_ref:
        args.remove('--ref')
    if '--R' in args:
        i = args.index('--R')
        R = int(args[i + 1])
        del args[i:i + 2]
    if '--step' in args:
        i = args.index('--step')
        step = int(args[i + 1])
        del args[i:i + 2]

    mask = red_mask(marked)
    pts = anchors(mask, step)
    print('anchors: %d (step=%d)' % (len(pts), step))

    variants = []
    if want_ref:
        variants.append(('AFref', load(REF), None))
    for a in args:
        lab, path = a.split('=', 1)
        img = load(path)
        oy, ox, mad = locate(img)
        print('%-10s ref-crop origin in output (y %d, x %d)  mad %.2f' % (lab, oy, ox, mad))
        variants.append((lab, img, (oy, ox)))

    off = np.arange(-R, R + 1)
    res = {}
    for lab, img, origin in variants:
        G = gray(img)
        T = texture_of(G)
        # The mask is in output coordinates. Outputs are sampled there directly;
        # the reference export needs the crop origin subtracted.
        pts_img = pts if origin is None else pts - np.array(origin, dtype=np.float64)
        _, lum, tex = profiles(G, T, saturation(img.astype(np.float32)), pts_img, pts, mask, R)
        res[lab] = (lum, tex)
        del G, T
        ml = np.nanmean(lum, 0)
        mt = np.nanmean(tex, 0)
        wall = slice(R + 6, R + 26)
        print('%-10s  wall-side L %6.1f   wall texture %5.2f   subject tex %5.2f   tex peak at %+d'
              % (lab, np.nanmean(ml[wall]), np.nanmean(mt[wall]),
                 np.nanmean(mt[3:R - 6]), int(np.nanargmax(mt)) - R))

    print()
    print('%-8s' % 'offset' + ''.join('%13s' % l for l, _, _ in variants))
    for i, o in enumerate(off):
        if o % 4:
            continue
        row = '%-8d' % o
        for lab, _, _ in variants:
            lum, tex = res[lab]
            row += '%13s' % ('%.1f / %.2f' % (np.nanmean(lum[:, i]), np.nanmean(tex[:, i])))
        print(row + '   (L / texture)')

    with open('out/edgeprofile.csv', 'w') as f:
        f.write('offset,' + ','.join('%s_lum,%s_tex' % (l, l) for l, _, _ in variants) + '\n')
        for i, o in enumerate(off):
            f.write('%d,' % o + ','.join(
                '%.3f,%.4f' % (np.nanmean(res[l][0][:, i]), np.nanmean(res[l][1][:, i]))
                for l, _, _ in variants) + '\n')
    print('wrote out/edgeprofile.csv')
    chart(off, [(l, np.nanmean(res[l][0], 0)) for l, _, _ in variants],
          'out/edgeprofile_lum.png', 'luminance profile across the marked contour', R, 'level (0-255)')
    chart(off, [(l, np.nanmean(res[l][1], 0)) for l, _, _ in variants],
          'out/edgeprofile_tex.png', 'texture (5x5 high-pass std) across the marked contour', R, 'texture std')


main()
