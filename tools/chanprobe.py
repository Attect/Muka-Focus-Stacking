#!/usr/bin/env python3
"""逐通道焦点判别力探针 —— 回答"把测度拆到颜色通道上再合回来有用吗"。

对若干采样点（光滑面 / 有纹理面），逐帧算局部梯度能量，通道取
L、R、G、B、R-G、B-G，然后报告每个通道的 **峰值/中位数** 判别力和峰值所在帧。

判别力 ≈ 该通道能在多大程度上把"合焦的那一帧"从其余帧里挑出来。
如果某个通道的判别力不高于亮度通道，那么把它加进测度不会带来信息，
只会把噪声按通道数加权进来。

Klee 这组素材（2026-09-20 实测）的结论：**色度通道判别力更差**——
墙面 L 5.7~7.5x，而 R-G / B-G 只有 1.2~2.3x；皮肤 L 1.2~1.8x，色度 1.1~1.4x。
原因：这个场景里几乎全部对焦细节都是亮度的（背景是灰对灰的浮雕，
R-G<8 且 G-B<8，色度上根本不存在；主体是均匀的塑料涂装）。
另外 JPEG 色度是 4:2:2 水平降采样（Y h=2，Cb/Cr h=1），
即便有色度细节也先被砍掉一半横向分辨率。

用法:
    python tools/chanprobe.py "<照片目录>" [n_skin] [n_wall]
"""

import os
import sys

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from patchlib import REF, load  # noqa: E402

DY, DX = 501, 1308  # 参照成品(AF导出) 坐标 -> 画布坐标（本项目固定关系）
CH = ["L", "R", "G", "B", "R-G", "B-G"]


def box(m, r):
    h, w = m.shape
    c = np.pad(np.cumsum(np.cumsum(m, 0), 1), ((1, 0), (1, 0)))
    ys, xs = np.arange(h), np.arange(w)
    y0, y1 = np.clip(ys - r, 0, h), np.clip(ys + r + 1, 0, h)
    x0, x1 = np.clip(xs - r, 0, w), np.clip(xs + r + 1, 0, w)
    return c[np.ix_(y1, x1)] - c[np.ix_(y0, x1)] - c[np.ix_(y1, x0)] + c[np.ix_(y0, x0)]


def erode(mask, r):
    """mask 内距离边界超过 r 的像素（远小于 r 的窗口内全是 mask）。"""
    s = box(mask.astype(np.float32), r)
    return mask & (s > 0.97 * (2 * r + 1) ** 2)


def pick(mask, n, min_sep=300):
    """在 mask 上挑 n 个彼此相距至少 min_sep 的点。"""
    ys, xs = np.where(mask)
    out = []
    for i in np.argsort(ys * 7919 + xs * 104729):
        y, x = int(ys[i]), int(xs[i])
        if all((y - a) ** 2 + (x - b) ** 2 > min_sep ** 2 for a, b in out):
            out.append((y, x))
        if len(out) >= n:
            break
    return out


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 1
    d = sys.argv[1]
    n_skin = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    n_wall = int(sys.argv[3]) if len(sys.argv) > 3 else 4

    files = sorted(
        x for x in os.listdir(d)
        if x.upper().startswith("DSC") and x.upper().endswith(".JPG")
    )
    ref = load(REF).astype(np.float32)
    r_, g_, b_ = ref[..., 0], ref[..., 1], ref[..., 2]
    lr = 0.299 * r_ + 0.587 * g_ + 0.114 * b_
    # 肤色：R>G>B 且不是高光；墙面：近中性且亮
    skin = (r_ > g_ + 10) & (g_ > b_ + 4) & (lr > 110) & (lr < 252)
    wall = (np.abs(r_ - g_) < 8) & (np.abs(g_ - b_) < 8) & (lr > 175) & (lr < 253)

    pts = [("skin", y, x) + (DY, DX) for y, x in pick(erode(skin, 30), n_skin)]
    pts += [("wall", y, x) + (DY, DX) for y, x in pick(erode(wall, 30), n_wall)]
    cp = [(lab, y + dy, x + dx) for lab, y, x, dy, dx in pts]

    print("frames %d, window 49 px" % len(files))
    for lab, y, x in cp:
        print("point %-5s canvas(%d,%d)" % (lab, y, x))

    win = 24
    resp = {k: np.zeros((len(files), len(cp))) for k in CH}
    for i, f in enumerate(files):
        a = np.asarray(Image.open(os.path.join(d, f)).convert("RGB"), dtype=np.float32)
        ch = {
            "L": 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2],
            "R": a[..., 0], "G": a[..., 1], "B": a[..., 2],
            "R-G": a[..., 0] - a[..., 1], "B-G": a[..., 2] - a[..., 1],
        }
        for k, m in ch.items():
            for j, (_lab, y, x) in enumerate(cp):
                w = m[y - win:y + win + 1, x - win:x + win + 1]
                p = w.copy()  # 3x3 低通，跟 --align-focus-blur 3 一个量级
                p[1:-1, 1:-1] = (
                    0.25 * w[1:-1, 1:-1]
                    + 0.125 * (w[:-2, 1:-1] + w[2:, 1:-1] + w[1:-1, :-2] + w[1:-1, 2:])
                    + 0.03125 * (w[:-2, :-2] + w[:-2, 2:] + w[2:, :-2] + w[2:, 2:])
                )
                gy, gx = np.gradient(p[1:-1, 1:-1])
                resp[k][i, j] = np.hypot(gx, gy).mean()
        if i % 10 == 0:
            print("  frame %d" % i, flush=True)

    print()
    print("peak/median of the per-frame response @ peak frame")
    print("%-6s %-14s" % ("kind", "canvas y,x") + "".join("%16s" % k for k in CH))
    for j, (lab, y, x) in enumerate(cp):
        line = "%-6s %-14s" % (lab, "%d,%d" % (y, x))
        for k in CH:
            v = resp[k][:, j]
            line += "%16s" % ("%.2f@%d" % (v.max() / max(float(np.median(v)), 1e-6),
                                           int(np.argmax(v))))
        print(line)
    print()
    print("frame-index centroid of the response")
    for j, (lab, y, x) in enumerate(cp):
        line = "%-6s %-14s" % (lab, "%d,%d" % (y, x))
        for k in CH:
            v = np.maximum(resp[k][:, j] - resp[k][:, j].min(), 0)
            line += "%16s" % ("%.1f" % float((v * np.arange(len(v))).sum()
                                             / max(v.sum(), 1e-9)))
        print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
