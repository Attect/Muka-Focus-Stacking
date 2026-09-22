"""Downscaled overview with a coordinate grid labelled in reference-crop pixels,
so regions can be located precisely before zooming in."""
import numpy as np
from PIL import Image, ImageDraw

Image.MAX_IMAGE_PIXELS = None
AF_X, AF_Y = 1288, 488
im = Image.open(r"B:\FocusMerge\out\final.png").convert("RGB")
scale = 1400 / im.width
small = im.resize((1400, int(im.height * scale)), Image.LANCZOS)
dr = ImageDraw.Draw(small)
step = 250  # in reference-crop pixels
for ax in range(0, 4276, step):
    sx = (AF_X + ax) * scale
    if sx >= small.width:
        break
    dr.line([(sx, 0), (sx, small.height)], fill=(255, 0, 0), width=1)
    dr.text((sx + 3, 4), str(ax), fill=(255, 0, 0))
for ay in range(0, 4003, step):
    sy = (AF_Y + ay) * scale
    if sy >= small.height:
        break
    dr.line([(0, sy), (small.width, sy)], fill=(255, 0, 0), width=1)
    dr.text((4, sy + 3), str(ay), fill=(255, 0, 0))
small.save(r"B:\FocusMerge\out\overview_grid.png")
print("wrote out/overview_grid.png", small.size)
