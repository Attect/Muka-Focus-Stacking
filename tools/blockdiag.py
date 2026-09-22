"""For each marked block: is the output clean single-frame content, or a
mixture of frames that disagree?

High-pass correlation against every frame (searched over a small shift) answers
it: a clean block correlates ~0.95+ with one frame and the best shift is
consistent; a mixted block peaks lower and the frames it "wins" on disagree.
High-pass *energy* alone cannot see this - a ghost of two offset edges has more
energy than either edge, so a doubled block scores high, not low.
"""
import os
import sys

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
sys.path.insert(0, 'tools')
from patchlib import gray, load, REF  # noqa: E402

D = os.environ['STACK_DIR']
CAN = (12, 18)
REFOFF = (506, 1304)
PADX = 34


def boxb(a, r):
    p = np.pad(a, r, mode='edge')
    c = np.pad(p.cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    H, W = a.shape
    y = np.arange(H)
    x = np.arange(W)
    return (c[np.ix_(y + 2 * r + 1, x + 2 * r + 1)] - c[np.ix_(y, x + 2 * r + 1)]
            - c[np.ix_(y + 2 * r + 1, x)] + c[np.ix_(y, x)]) / ((2 * r + 1) ** 2)


def hp(a, r=5):
    return a - boxb(a, r)


def he(a, R=2):
    h = hp(a, R)
    return float((boxb(h * h, 2) ** 0.5).mean())


def best_shift(pa, big, rng=14):
    """Max normalised high-pass correlation of `pa` near the centre of `big`.

    `pa` must already be the same crop as the centre of `big` (both taken at the
    same canvas position), so the answer should be near (0, 0) - a large offset
    means the two simply do not contain the same picture.
    """
    ha = hp(pa)
    a = ha - ha.mean()
    na = np.sqrt((a * a).sum())
    h, w = ha.shape
    best = (-9.0, 0, 0)
    for step, r in ((2, rng), (1, 2)):
        c0, c1 = best[1], best[2]
        for dy in range(c0 - r, c0 + r + 1, step):
            for dx in range(c1 - r, c1 + r + 1, step):
                y0, x0 = PADX + dy, PADX + dx
                if y0 < 0 or x0 < 0 or y0 + h > big.shape[0] or x0 + w > big.shape[1]:
                    continue
                b = hp(big[y0:y0 + h, x0:x0 + w])
                bb = b - b.mean()
                c = float((a * bb).sum() / (na * np.sqrt((bb * bb).sum()) + 1e-9))
                if c > best[0]:
                    best = (c, dy, dx)
    return best


def main():
    blocks = eval(open('out/_blocks.txt').read())
    V = gray(load('out/可莉-全清晰.png')).astype(np.float32)
    AF = gray(load(REF)).astype(np.float32)
    D_ = os.environ['STACK_DIR']
    files = sorted(f for f in os.listdir(D_)
                   if f.upper().startswith('DSC') and f.upper().endswith('.JPG'))
    # crop every block out of every frame, one decode per frame
    crops = {i: [] for i, *_ in blocks}
    for k, f in enumerate(files):
        im = Image.open(os.path.join(D_, f)).convert('L')
        for i, x0, x1, y0, y1 in blocks:
            crops[i].append(np.asarray(
                im.crop((x0 - PADX, y0 - PADX, x1 + PADX, y1 + PADX)), dtype=np.float32))
        if (k + 1) % 10 == 0:
            print('  decode %d/%d' % (k + 1, len(files)), flush=True)
    print()
    print('%-3s %7s %7s %7s %-16s %-22s' %
          ('#', 'ours/B', 'oursHE', 'bestHE', 'best frame', 'top-2 frames (corr)'))
    for i, x0, x1, y0, y1 in blocks:
        h, w = y1 - y0, x1 - x0
        A = V[y0 - CAN[0]:y0 - CAN[0] + h, x0 - CAN[1]:x0 - CAN[1] + w]
        best = 0.0
        for c in crops[i]:
            v = he(c[PADX:PADX + h, PADX:PADX + w])
            if v > best:
                best = v
        res = []
        for k, c in enumerate(crops[i]):
            cc, dy, dx = best_shift(A, c)
            res.append((cc, k, dy, dx))
        res.sort(reverse=True)
        # the AF reference, same treatment
        ah = AF[y0 - REFOFF[0]:y0 - REFOFF[0] + h, x0 - REFOFF[1]:x0 - REFOFF[1] + w]
        ac = hp(ah)
        top = '  '.join('%d:%.3f(%+d,%+d)' % (k, c, dy, dx) for c, k, dy, dx in res[:2])
        print('%-3d %7.2f %7.2f %7.2f  帧%-3d            %s'
              % (i, he(A) / max(best, 1e-6), he(A), best, res[0][1], top))
        ares = sorted((best_shift(ah, c) for c in crops[i]), reverse=True)
        print('    AF 同块: HE %5.2f  与最优帧相关度 %.3f (帧%d)   ours %.3f (帧%d)'
              % (he(ah), ares[0][0], ares[0][1], res[0][0], res[0][1]))


main()
