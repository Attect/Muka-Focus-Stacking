"""Chart: wall luminance against distance from the silhouette.

The whole complaint lives in one curve. `L(d)` for d = 0..92 px outside the
figure, for the old and the new default, the AF export, and the per-pixel
reference (each pixel taken from the frame the saved depth picks). The right
panel subtracts the reference, which is where a light band shows up as a hump
that decays to zero by 30 px.

Usage:
    python tools/bandchart.py --depth out/d4_final.png [--bbox x0,x1,y0,y1]
        old.png=new.png ...
"""
import os
import sys

import numpy as np
from PIL import Image, ImageDraw

Image.MAX_IMAGE_PIXELS = None
sys.path.insert(0, 'tools')
from patchlib import REF, gray, load, locate  # noqa: E402
from bandlib import boxsum, subject_mask  # noqa: E402

D = os.environ.get("STACK_DIR", "stack")
CANVAS_FROM_OUT = (12, 18)
DR = 92


def rings(subj, rmax):
    prev = np.zeros(subj.shape, bool)
    out = []
    for k in range(rmax + 1):
        cur = boxsum(subj, k) > 0
        out.append(cur & ~prev)
        prev = cur
    return out


def curve(G, rk, mask):
    return np.array([float(G[r & mask].mean()) if (r & mask).sum() > 30 else np.nan
                     for r in rk])


def main():
    args = sys.argv[1:]
    bbox = None
    if '--bbox' in args:
        i = args.index('--bbox')
        bbox = tuple(int(v) for v in args[i + 1].split(','))
        del args[i:i + 2]
    depth = 'out/d4_final.png'
    if '--depth' in args:
        i = args.index('--depth')
        depth = args[i + 1]
        del args[i:i + 2]
    want_ref = '--af' in args
    if want_ref:
        args.remove('--af')
    want_frame = '--frame33' in args
    if want_frame:
        args.remove('--frame33')
    items = [a.split('=') for a in args]

    B = load(items[0][1])
    subj, _ = subject_mask(B)
    H, W = subj.shape
    x0, x1, y0, y1 = bbox or (0, W, 0, H)
    mask = np.zeros((H, W), bool)
    mask[y0:y1, x0:x1] = True
    rk = rings(subj, DR)

    oy, ox = CANVAS_FROM_OUT
    dep = np.asarray(Image.open(depth)).astype(np.float32) / 65535.0 * 49.0
    dw = dep[y0 + oy:y1 + oy, x0 + ox:x1 + ox]
    files = sorted(f for f in os.listdir(D)
                   if f.upper().startswith('DSC') and f.upper().endswith('.JPG'))
    need = sorted({int(round(v)) for v in np.unique(dw)})
    ref = np.zeros(dw.shape, np.float32)
    for k in need:
        m = np.round(dw).astype(int) == k
        if m.any():
            ref[m] = gray(np.asarray(Image.open(os.path.join(D, files[k])).convert('RGB')
                                     .crop((x0 + ox, y0 + oy, x1 + ox, y1 + oy)),
                                     dtype=np.float32))[m]
    ref_full = np.zeros((H, W), np.float32)
    ref_full[y0:y1, x0:x1] = ref
    Lref = curve(ref_full, rk, mask)

    curves = [('per-pixel reference', Lref, (0, 0, 0))]
    if want_ref:
        oy2, ox2, _ = locate(B)
        R = load(REF)
        s = np.zeros(R.shape[:2], bool)
        m2 = np.zeros(R.shape[:2], bool)
        h = min(s.shape[0], subj.shape[0] - oy2)
        w = min(s.shape[1], subj.shape[1] - ox2)
        s[:h, :w] = subj[oy2:oy2 + h, ox2:ox2 + w]
        m2[:h, :w] = mask[oy2:oy2 + h, ox2:ox2 + w]
        curves.append(('AF export', curve(gray(R), rings(s, DR), m2), (0, 150, 200)))
    if want_frame:
        g = gray(np.asarray(Image.open(os.path.join(D, files[33])).convert('RGB')
                            .crop((x0 + ox, y0 + oy, x1 + ox, y1 + oy)), dtype=np.float32))
        rk_s = rings(subj[y0:y1, x0:x1], DR)
        curves.append(('frame 33', curve(g, rk_s, np.ones(g.shape, bool)), (150, 150, 150)))
    cols = [(0, 0, 255), (220, 0, 0), (0, 160, 0), (230, 130, 0)]
    for i, (lab, path) in enumerate(items):
        G = gray(load(path))
        curves.append((lab, curve(G, rk, mask), cols[i % len(cols)]))
        del G

    CW, CH = 1180, 460
    im = Image.new('RGB', (CW, CH), (255, 255, 255))
    d = ImageDraw.Draw(im)
    Ls = np.concatenate([[c for _, c, _ in curves]])
    lo, hi = 176.0, 187.0
    W1 = 580

    def px1(v):
        return 46 + v / DR * (W1 - 60)

    def py(v):
        return CH - 46 - (v - lo) / (hi - lo) * (CH - 90)

    d.rectangle([46, 24, W1 - 14, CH - 46], outline=(190, 190, 190))
    for v in range(177, 187, 2):
        d.line([46, py(v), W1 - 14, py(v)], fill=(235, 235, 235))
        d.text((10, py(v) - 6), str(v), fill=(90, 90, 90))
    for v in range(0, DR + 1, 20):
        d.line([px1(v), 24, px1(v), CH - 46], fill=(235, 235, 235))
        d.text((px1(v) - 6, CH - 40), str(v), fill=(90, 90, 90))
    d.text((46, 6), 'wall luminance L(d), d px outside the figure', fill=(0, 0, 0))
    d.text((W1 - 120, CH - 40), 'd (px)', fill=(90, 90, 90))
    for lab, c, col in curves:
        pts = [(px1(k), py(v)) for k, v in enumerate(c) if np.isfinite(v)]
        d.line(pts, fill=col, width=2)

    X2 = W1 + 30
    lo2, hi2 = -2.0, 4.0

    def px2(v):
        return X2 + 34 + v / DR * (CW - X2 - 50)

    def py2(v):
        return CH - 46 - (v - lo2) / (hi2 - lo2) * (CH - 90)

    d.rectangle([X2 + 34, 24, CW - 14, CH - 46], outline=(190, 190, 190))
    for v in range(-2, 5):
        d.line([X2 + 34, py2(v), CW - 14, py2(v)], fill=(235, 235, 235))
        d.text((X2 + 6, py2(v) - 6), '%+d' % v, fill=(90, 90, 90))
    d.line([X2 + 34, py2(0), CW - 14, py2(0)], fill=(120, 120, 120))
    for v in range(0, DR + 1, 20):
        d.text((px2(v) - 6, CH - 40), str(v), fill=(90, 90, 90))
    d.text((X2 + 34, 6), 'minus the per-pixel reference', fill=(0, 0, 0))
    for lab, c, col in curves:
        if lab == 'per-pixel reference':
            continue
        pts = [(px2(k), py2(v - Lref[k])) for k, v in enumerate(c)
               if np.isfinite(v) and np.isfinite(Lref[k])]
        d.line(pts, fill=col, width=2)
    for i, (lab, _, col) in enumerate(curves):
        d.text((50 + i * 190, CH - 22), lab, fill=col)
    im.save('out/band_chart.png')
    print('wrote out/band_chart.png')
    for lab, c, _ in curves:
        print('%-22s  d=4 %.1f  d=8 %.1f  d=14 %.1f  d=22 %.1f  d=34 %.1f'
              % (lab, c[4], c[8], c[14], c[22], c[34]))


main()
