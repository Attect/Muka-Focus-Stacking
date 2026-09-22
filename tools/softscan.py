#!/usr/bin/env python3
"""全图扫描：找出我们的输出比参照成品**明显更软**的位置。

按块比较对齐后的高频能量（局部去均值后的绝对值，再块平均），
做非极大值抑制后输出最差的若干个点，同时给参照坐标和画布坐标。

这是本轮唯一有效的"找缺陷"入口 —— `tools/bloom.py` 那类"低通求差"的指标
会被"我们纹理比参照强"污染，与目视相反，别再用它定案。

用法:
    python tools/softscan.py [输出图] [参照图] [topN]

缺省：输出图 out/可莉-全清晰.png，参照图 patchlib.REF。
"""

import os
import sys

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from patchlib import REF, load, locate  # noqa: E402


def box(m, r):
    h, w = m.shape
    c = np.pad(np.cumsum(np.cumsum(m, 0), 1), ((1, 0), (1, 0)))
    ys, xs = np.arange(h), np.arange(w)
    y0, y1 = np.clip(ys - r, 0, h), np.clip(ys + r + 1, 0, h)
    x0, x1 = np.clip(xs - r, 0, w), np.clip(xs + r + 1, 0, w)
    return c[np.ix_(y1, x1)] - c[np.ix_(y0, x1)] - c[np.ix_(y1, x0)] + c[np.ix_(y0, x0)]


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "out/可莉-全清晰.png"
    refp = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] else REF
    topn = int(sys.argv[3]) if len(sys.argv) > 3 else 12

    ref = load(refp)
    a = load(out)
    oy, ox, _ = locate(a, ref)
    print("align: 输出图相对参照的偏移 = (y=%d, x=%d)" % (oy, ox))

    H, W = ref.shape[:2]
    ai = np.arange(H) + oy
    aj = np.arange(W) + ox
    vy, vx = (ai >= 0) & (ai < a.shape[0]), (aj >= 0) & (aj < a.shape[1])
    A = a[np.ix_(ai[vy], aj[vx])][..., :3].astype(np.float32)
    Rf = ref[np.ix_(np.where(vy)[0], np.where(vx)[0])][..., :3].astype(np.float32)
    print("可比区域 %dx%d" % (Rf.shape[1], Rf.shape[0]))

    def en(x, rf=2, rs=8):
        g = 0.299 * x[..., 0] + 0.587 * x[..., 1] + 0.114 * x[..., 2]
        e = np.abs(g - box(g, rf))
        return box(e, rs) / (2 * rs + 1) ** 2

    so, sr = en(A), en(Rf)
    B = 48
    so = box(so, B) / (2 * B + 1) ** 2
    sr = box(sr, B) / (2 * B + 1) ** 2
    ratio = so / np.maximum(sr, 1e-6)
    m = sr > 0.5
    r = ratio[m]
    print("全局高频能量 我方/参照: mean %.3f  median %.3f  p10 %.3f  p90 %.3f"
          % (r.mean(), np.median(r), np.percentile(r, 10), np.percentile(r, 90)))
    print("  比参照软 >20%% 的块 %.1f%%;  比参照锐 >20%% 的块 %.1f%%"
          % (100 * (r < 0.8).mean(), 100 * (r > 1.2).mean()))

    sev = np.maximum(0.0, 1.0 - ratio) * sr
    picked = []
    for y, x in np.dstack(np.unravel_index(np.argsort(-sev.ravel()), sev.shape))[0]:
        if sr[y, x] < 0.8:
            continue
        if all((y - p) ** 2 + (x - q) ** 2 > 220 ** 2 for p, q in picked):
            picked.append((int(y), int(x)))
        if len(picked) >= topn:
            break
    print()
    print("最软的 %d 处 (块 %d px)" % (len(picked), (2 * B + 1)))
    print("%6s %6s %10s %10s %8s   %s" % ("y", "x", "ours", "ref", "ratio", "canvas y,x"))
    for y, x in picked:
        print("%6d %6d %10.2f %10.2f %8.2f   %d,%d"
              % (y, x, so[y, x], sr[y, x], ratio[y, x], y + oy, x + ox))
    return 0


if __name__ == "__main__":
    sys.exit(main())
