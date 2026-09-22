"""Does the chroma-bleed pass do what it claims, and does it cost anything?

Two questions: (a) how much of the wash outside the petal is gone, (b) did the
figure's own colour or its edges change.
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
X0, X1, Y0, Y1 = 2170, 2270, 2420, 2560


def win(path, off):
    a = np.asarray(Image.open(path).convert('RGB'), dtype=np.float32)
    return a[Y0 - off[0]:Y1 - off[0], X0 - off[1]:X1 - off[1]]


def profile(a):
    """Mean R-G (and luminance) as a function of distance outside the petal."""
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    red = (r - g > 55) & (r > 110)
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    P = []
    for x in range(a.shape[1]):
        col = np.nonzero(red[:, x])[0]
        if len(col) < 12:
            continue
        ye = col.min()
        if ye < 26:
            continue
        P.append(np.stack([lum[ye - 26:ye + 2, x], (r - g)[ye - 26:ye + 2, x]])[:, ::-1])
    return np.array(P).mean(0) if P else None


def main():
    items = [('ours (default)', 'out/可莉-全清晰.png', CAN),
             ('AF (retouched)', os.path.join(D, 'AF导出.png'), REFOFF)]
    for a in sys.argv[1:]:
        lab, path = a.split('=', 1)
        items.append((lab, path, CAN))
    print('%-18s %s' % ('R-G 剖面', ' '.join('%6d' % k for k in range(0, 17, 2))))
    for lab, path, off in items:
        p = profile(win(path, off))
        if p is None:
            print('%-18s (no petal edge in window)' % lab)
            continue
        print('%-18s %s' % (lab, ' '.join('%6.1f' % p[1, k] for k in range(0, 17, 2))))
    print()
    print('%-18s %s' % ('亮度剖面', ' '.join('%6d' % k for k in range(0, 17, 2))))
    for lab, path, off in items:
        p = profile(win(path, off))
        if p is None:
            continue
        print('%-18s %s' % (lab, ' '.join('%6.1f' % p[0, k] for k in range(0, 17, 2))))

    print()
    print('全图：彩色像素（通道极差）的统计——数值掉下去说明主体颜色被削')
    print('%-18s %8s %8s %8s %8s' % ('版本', '>40的占比', '中位(>40)', '95分位', '峰值'))
    for lab, path, _ in items:
        a = load(path)
        sp = a.max(-1) - a.min(-1)
        sel = sp > 40
        print('%-18s %8.2f%% %8.1f %8.1f %8.1f'
              % (lab, 100 * sel.mean(), np.median(sp[sel]), np.percentile(sp, 95), sp.max()))

    print()
    print('主体区域锐度（gradE）与红通道分布——看有没有削边')
    for lab, path, _ in items:
        a = load(path)
        g = gray(a)
        gy, gx = np.gradient(g)
        print('%-18s gradE %7.3f   R>G+60 的像素占比 %.3f%%'
              % (lab, float(np.hypot(gx, gy).mean()), 100 * ((a[..., 0] - a[..., 1]) > 60).mean()))


main()
