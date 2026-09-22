"""Compare the wall beside the silhouette against a *per-pixel* reference.

A single reference frame is not the truth over a region this wide: the wall is a
plane at an angle, so the frame that has it in focus drifts by ten frames across
the marked area. The reference built here is per pixel instead: for every pixel
the frame given by the saved depth map is sampled, which is the best single-frame
reconstruction the depth allows. That isolates what the *fusion* added from what
the *depth* chose.

Prints, per variant, the ring curve difference against that reference (positive =
brighter than the reference) and the band/far relief ratio.

Usage:
    python tools/bandtruth.py --depth out/dcur_final.png [--bbox x0,x1,y0,y1]
        label=path [...]
"""
import os
import sys

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
sys.path.insert(0, 'tools')
from patchlib import gray, load, locate  # noqa: E402
from bandlib import boxsum, subject_mask, texture_of  # noqa: E402

D = os.environ.get("STACK_DIR", "stack")
CANVAS_FROM_OUT = (12, 18)
DR = 92
SHOW = (4, 6, 8, 11, 14, 18, 22, 28, 34, 44, 60, 80)


def rings(subj, rmax):
    prev = np.zeros(subj.shape, bool)
    out = []
    for k in range(rmax + 1):
        cur = boxsum(subj, k) > 0
        out.append(cur & ~prev)
        prev = cur
    return out


def main():
    args = sys.argv[1:]
    bbox = None
    depth_path = None
    if '--bbox' in args:
        i = args.index('--bbox')
        bbox = tuple(int(v) for v in args[i + 1].split(','))
        del args[i:i + 2]
    if '--depth' in args:
        i = args.index('--depth')
        depth_path = args[i + 1]
        del args[i:i + 2]
    items = [a.split('=', 1) for a in args]

    B = load(items[0][1])
    subj, _ = subject_mask(B)
    H, W = subj.shape
    x0, x1, y0, y1 = bbox or (0, W, 0, H)
    mask = np.zeros((H, W), bool)
    mask[y0:y1, x0:x1] = True
    rk = rings(subj, DR)

    oy, ox = CANVAS_FROM_OUT
    dep = np.asarray(Image.open(depth_path)).astype(np.float32) / 65535.0 * 49.0
    dw = dep[y0 + oy:y1 + oy, x0 + ox:x1 + ox]
    print('深度窗口 %s  范围 %.0f..%.0f  中位 %.1f'
          % (dw.shape, dw.min(), dw.max(), np.median(dw)))

    files = sorted(f for f in os.listdir(D)
                   if f.upper().startswith('DSC') and f.upper().endswith('.JPG'))
    need = sorted({int(round(v)) for v in np.unique(dw)} & set(range(len(files))))
    print('参照需要的帧: %s' % need)
    win = {}
    for k in need:
        win[k] = gray(np.asarray(Image.open(os.path.join(D, files[k])).convert('RGB')
                                 .crop((x0 + ox, y0 + oy, x1 + ox, y1 + oy)), dtype=np.float32))
    idx = np.clip(np.round(dw).astype(int), min(need), max(need))
    ref = np.zeros(dw.shape, np.float32)
    for k in need:
        m = idx == k
        if m.any():
            ref[m] = win[k][m]
    del win
    ref_full = np.zeros((H, W), np.float32)
    ref_full[y0:y1, x0:x1] = ref
    rk_s = rings(subj[y0:y1, x0:x1], DR)
    m_s = np.ones(rk_s[0].shape, bool)

    def curve(G, rks, m):
        return np.array([float(G[r & m].mean()) if (r & m).sum() > 30 else np.nan
                         for r in rks])

    def texcurve(G, rks, m):
        T = texture_of(G)
        return np.array([float(T[r & m].mean()) if (r & m).sum() > 30 else np.nan
                         for r in rks])

    Lref = curve(ref, rk_s, m_s)
    Tref = texcurve(ref, rk_s, m_s)
    d = np.arange(len(Lref), dtype=np.float64)
    near = (d >= 4) & (d <= 22)
    far = (d >= 34) & (d <= 90)
    print()
    print('参照(逐像素取深度对应帧) L(d): ' + '  '.join('%d:%.1f' % (k, Lref[k]) for k in SHOW))
    print('%-12s %8s %8s %9s %9s %10s'
          % ('variant', '4-22', '34-90', 'd=8', 'd=14', 'band tex'))
    print('%-12s %8s %8s %9.2f %9.2f %10.3f'
          % ('参照', 0, 0, Lref[8], Lref[14], np.nanmean(Tref[4:23]) / np.nanmean(Tref[34:91])))
    for lab, path in items:
        img = load(path)
        G = gray(img)
        if img.shape[0] == H:
            L = curve(G, rk, mask)
        else:
            r, c, _ = locate(img)
            s2 = np.zeros(img.shape[:2], bool)
            h = min(s2.shape[0], subj.shape[0] - r)
            w = min(s2.shape[1], subj.shape[1] - c)
            s2[:h, :w] = subj[r:r + h, c:c + w]
            m2 = np.zeros(img.shape[:2], bool)
            m2[:h, :w] = mask[r:r + h, c:c + w]
            L = curve(G, rings(s2, DR), m2)
        T = texcurve(G, rk, mask)
        print('%-12s %+8.2f %+8.2f %9.2f %9.2f %10.3f'
              % (lab, float((L[near] - Lref[near]).mean()),
                 float((L[far] - Lref[far]).mean()), L[8] - Lref[8], L[14] - Lref[14],
                 np.nanmean(T[4:23]) / np.nanmean(T[34:91])))
        del G, img, T


main()
