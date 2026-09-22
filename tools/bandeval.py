"""Evaluate the wall band beside a silhouette: brightness excess and relief loss.

Reports, for every variant, the two numbers that describe the reported artefact,
both computed *locally* so that the wall's own slow brightness gradient cannot
masquerade as an effect:

  * band/far L  -- how much brighter the first 4..22 px of wall outside the
    figure is than the wall further away, measured in the same 40 px
    neighbourhood. The *sharpest raw frame* is the reference: it reads -1.3.
  * tex ratio   -- the same ratio for the fine relief (5x5 high-pass std). The
    raw frames read 1.09, i.e. the band carries as much relief as the wall
    behind it; a value below 1 is relief that the fusion lost.

Usage:
    python tools/bandeval.py [--bbox x0,x1,y0,y1] [--ref] label=path [...]
"""
import os
import sys

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
sys.path.insert(0, 'tools')
from patchlib import REF, gradient_energy, grain, gray, load, locate  # noqa: E402
from bandlib import build_masks, local_pair, texture_of  # noqa: E402

D = os.environ.get("STACK_DIR", "stack")
CANVAS_FROM_OUT = (12, 18)


def metrics(G, T, b, f):
    ok, dl, ratio = local_pair(G, T, b, f)
    return dl, ratio, float(G[b].mean()), float(T[b].mean())


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
    want_frames = '--frames' in args
    if want_frames:
        args.remove('--frames')
    items = [a.split('=', 1) for a in args]
    B = load(items[0][1])
    ref_h, ref_w = np.asarray(Image.open(REF)).shape[:2]
    band, farw = build_masks(B, bbox)
    np.save('out/_band.npy', band)
    np.save('out/_far.npy', farw)
    print('band %.4f%% of image, far %.4f%%' % (100 * band.mean(), 100 * farw.mean()))
    print('%-10s %10s %10s %9s %9s %9s %9s'
          % ('variant', 'band-far L', 'tex ratio', 'band L', 'band tex', 'gradE', 'grain'))
    for lab, path in items:
        img = load(path)
        # gradE/grain must be measured on the *same* physical area in every
        # variant, so they use the located reference-sized window rather than the
        # whole crop (whose origin moves from run to run).
        oy, ox, _ = locate(img)
        h = min(ref_h, img.shape[0] - oy)
        wd = min(ref_w, img.shape[1] - ox)
        win = img[oy:oy + h, ox:ox + wd]
        ge = gradient_energy(win)
        gr = grain(win)
        del win
        G = gray(img)
        T = texture_of(G)
        if img.shape[0] != B.shape[0]:
            b = band[oy:oy + h, ox:ox + wd]
            f = farw[oy:oy + h, ox:ox + wd]
            G, T = G[:h, :wd], T[:h, :wd]
        else:
            b, f = band, farw
        dl, ratio, bl, bt = metrics(G, T, b, f)
        print('%-10s %10.2f %10.3f %9.2f %9.3f %9.3f %9.3f'
              % (lab, dl, ratio, bl, bt, ge, gr))
        del G, T, img
    if want_ref:
        R = load(REF)
        oy, ox, _ = locate(B)
        h = min(R.shape[0], band.shape[0] - oy)
        wd = min(R.shape[1], band.shape[1] - ox)
        b = band[oy:oy + h, ox:ox + wd]
        f = farw[oy:oy + h, ox:ox + wd]
        G = gray(R)[:h, :wd]
        T = texture_of(G)
        dl, ratio, bl, bt = metrics(G, T, b, f)
        print('%-10s %10.2f %10.3f %9.2f %9.3f %9s %9s'
              % ('AFref', dl, ratio, bl, bt, '-', '-'))

    if want_frames:
        files = sorted(f for f in os.listdir(D)
                       if f.upper().startswith('DSC') and f.upper().endswith('.JPG'))
        x0, x1, y0, y1 = bbox
        cy0, cx0 = y0 + CANVAS_FROM_OUT[0], x0 + CANVAS_FROM_OUT[1]
        cy1, cx1 = y1 + CANVAS_FROM_OUT[0], x1 + CANVAS_FROM_OUT[1]
        bw = band[y0:y1, x0:x1]
        fw = farw[y0:y1, x0:x1]
        dl_s = np.zeros(len(files))
        rt_s = np.zeros(len(files))
        bl_s = np.zeros(len(files))
        for i, fp in enumerate(files):
            a = np.asarray(Image.open(os.path.join(D, fp)).convert('RGB')
                           .crop((cx0, cy0, cx1, cy1)), dtype=np.float32)
            g = gray(a)
            t = texture_of(g)
            ok, dlv, rtv = local_pair(g, t, bw, fw)
            dl_s[i] = dlv
            rt_s[i] = rtv
            bl_s[i] = float(g[bw].mean())
            if (i + 1) % 10 == 0:
                print('  帧 %d/%d' % (i + 1, len(files)), flush=True)
        print('原始帧：band-far L 中位 %+.2f (最好帧 %+.2f)   tex ratio 中位 %.3f (最好帧 %.3f)'
              % (np.median(dl_s), dl_s[np.argmax(rt_s)], np.median(rt_s), rt_s.max()))
        print('   帧 32 %.3f  帧 33 %.3f   band L 中位 %.1f'
              % (rt_s[32], rt_s[33], np.median(bl_s)))
        np.save('out/_frame_ratio.npy', rt_s)


main()
