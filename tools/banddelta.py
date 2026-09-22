"""Our wall minus the wall in the frame that has it in focus.

The ring-based excess of `bandexcess.py` is a *self-referential* measure (it
compares the near wall against a fit of the far wall), so a variant that lifts
the whole neighbourhood lifts both terms and under-reports the effect. This
compares against an outside reference instead: the frame whose depth matches the
wall, sampled through the same ring geometry. Whatever the difference is, it is
brightness that the fusion invented.

Usage:
    python tools/banddelta.py [--bbox x0,x1,y0,y1] [--frame 33]
        label=path [label=path ...]
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
    kf = 33
    if '--frame' in args:
        i = args.index('--frame')
        kf = int(args[i + 1])
        del args[i:i + 2]
    items = [a.split('=', 1) for a in args]

    B = load(items[0][1])
    subj, _ = subject_mask(B)
    H, W = subj.shape
    mask = np.ones((H, W), bool)
    if bbox:
        x0_, x1_, y0_, y1_ = bbox
        mask = np.zeros((H, W), bool)
        mask[y0_:y1_, x0_:x1_] = True
    rk = rings(subj, DR)

    files = sorted(f for f in os.listdir(D)
                   if f.upper().startswith('DSC') and f.upper().endswith('.JPG'))
    oy, ox = CANVAS_FROM_OUT
    ys, xs = np.nonzero(mask)
    cy0, cx0 = ys.min() + oy, xs.min() + ox
    cy1, cx1 = ys.max() + oy + 1, xs.max() + ox + 1
    sw = subj[cy0 - oy:cy1 - oy, cx0 - ox:cx1 - ox]
    rks = rings(sw, DR)
    ref = {}
    for k in (kf - 1, kf, kf + 1):
        g = gray(np.asarray(Image.open(os.path.join(D, files[k])).convert('RGB')
                            .crop((cx0, cy0, cx1, cy1)), dtype=np.float32))
        ref[k] = curve(g, rks, np.ones(g.shape, bool))
    Lref = ref[kf]
    print('参照帧 %d 的 L(d): ' % kf + '  '.join('%d:%.1f' % (d, Lref[d]) for d in SHOW))
    print('（相邻帧 %d / %d 在 d=8 处分别 %.1f / %.1f）'
          % (kf - 1, kf + 1, ref[kf - 1][8], ref[kf + 1][8]))
    print()
    print('%-12s %8s %8s %9s %9s' % ('variant', '4-22', '34-90', 'd=8', 'd=14'))
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
        d = np.arange(len(L), dtype=np.float64)
        near = (d >= 4) & (d <= 22) & np.isfinite(L)
        far = (d >= 34) & (d <= 90) & np.isfinite(L)
        print('%-12s %+8.2f %+8.2f %9.2f %9.2f'
              % (lab, float((L[near] - Lref[near]).mean()),
                 float((L[far] - Lref[far]).mean()), L[8] - Lref[8], L[14] - Lref[14]))
        del G, img


main()
