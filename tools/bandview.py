"""Render the light band beside the silhouette: three panels, same amplification.

A band a few levels high and twenty pixels wide is invisible in a 1:1 view and
easy to miss in an ordinary zoom, so each panel shows the image with its own
large-scale background removed (`G - box(G, 60)`, amplified) next to the plain
crop. Same treatment on every panel, so any band that survives is real.

Usage:
    python tools/bandview.py x0,x1,y0,y1 out.png label=path [label=path ...]
        [--ref]
"""
import os
import sys

import numpy as np
from PIL import Image, ImageDraw

Image.MAX_IMAGE_PIXELS = None
sys.path.insert(0, 'tools')
from patchlib import REF, gray, load, locate  # noqa: E402
from bandlib import box  # noqa: E402

D = os.environ.get("STACK_DIR", "stack")
CANVAS_FROM_OUT = (12, 18)


def enh(rgb, R=60, gain=4.0):
    g = gray(rgb)
    return np.clip((g - box(g, R)) * gain + 128.0, 0, 255).astype(np.uint8)


def main():
    reg = tuple(int(v) for v in sys.argv[1].split(','))
    out = sys.argv[2]
    args = sys.argv[3:]
    want_ref = '--ref' in args
    if want_ref:
        args.remove('--ref')
    frames = [a for a in args if a.startswith('frame=')]
    args = [a for a in args if not a.startswith('frame=')]
    x0, x1, y0, y1 = reg
    Z = 3

    panels = []
    B = load('out/cur.png')
    oy, ox, _ = locate(B)
    for a in args:
        lab, path = a.split('=', 1)
        img = load(path)
        if img.shape[1] < 6000:
            # the reference export is its own crop: same physical region, its
            # own coordinates
            crop = img[y0 - oy:y1 - oy, x0 - ox:x1 - ox]
        else:
            crop = img[y0:y1, x0:x1]
        panels.append((lab, crop))
    for a in frames:
        k = int(a.split('=')[1])
        import os
        files = sorted(f for f in os.listdir(D)
                       if f.upper().startswith('DSC') and f.upper().endswith('.JPG'))
        oy, ox = CANVAS_FROM_OUT
        a2 = np.asarray(Image.open(os.path.join(D, files[k])).convert('RGB')
                        .crop((x0 + ox, y0 + oy, x1 + ox, y1 + oy)), dtype=np.float32)
        panels.append(('frame%d (wall focus)' % k, a2))

    w, h = x1 - x0, y1 - y0
    sheet = Image.new('RGB', (len(panels) * (w * Z + 8), 2 * (h * Z + 26) + 8), (255, 255, 255))
    dr = ImageDraw.Draw(sheet)
    for i, (lab, crop) in enumerate(panels):
        px = i * (w * Z + 8)
        plain = Image.fromarray(np.clip(crop, 0, 255).astype(np.uint8)).resize(
            (w * Z, h * Z), Image.NEAREST)
        sheet.paste(plain, (px, 22))
        dr.text((px + 4, 8), lab, fill=(200, 0, 0))
        e = Image.fromarray(enh(crop)).resize((w * Z, h * Z), Image.NEAREST)
        sheet.paste(e, (px, h * Z + 30 + 22))
        dr.text((px + 4, h * Z + 30 + 8), lab + '  bg removed x4', fill=(200, 0, 0))
    sheet.save(out)
    print('wrote', out, sheet.size)


main()
