import sys
import numpy as np
from PIL import Image, ImageDraw
Image.MAX_IMAGE_PIXELS=None

def panel(src, x0, x1, y0, y1, z, label, stretch=None):
    a = np.asarray(src.crop((x0, y0, x1, y1)).convert('RGB'), dtype=np.float64)
    if stretch:
        lo, hi = stretch
        a = np.clip((a - lo) * 255.0 / (hi - lo), 0, 255)
    im = Image.fromarray(a.astype(np.uint8)).resize(((x1-x0)*z, (y1-y0)*z), Image.NEAREST)
    d = ImageDraw.Draw(im)
    d.text((4, 4), label, fill=(255, 0, 0))
    return im

def sheet(items, out, cols=None):
    cols = cols or len(items)
    rows = (len(items) + cols - 1) // cols
    w = max(i.width for i in items); h = max(i.height for i in items)
    s = Image.new('RGB', (cols*(w+6), rows*(h+6)), (255, 255, 255))
    for k, im in enumerate(items):
        s.paste(im, ((k % cols)*(w+6), (k//cols)*(h+6)))
    s.save(out); print('wrote', out, s.size)
