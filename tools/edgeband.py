"""Texture energy as a function of distance from a silhouette.

The artifact this targets: the band of background immediately outside a
foreground object loses its texture and turns into a smooth gradient. It is
invisible to whole-image metrics (the band is narrow, and the object itself is
sharp) and it is also invisible to a bloom metric, so it needs its own reading.

Method: in the reference, find the silhouette in each row of a region as the
strongest luminance gradient; then measure high-pass energy at a fixed number of
pixels on one side of it, averaged over rows. A healthy result tracks the
reference's own curve; a textureless band shows as a plateau far below it that
extends tens of pixels from the edge.

Usage:
    edgeband.py x0:x1:y0:y1 label=path [label=path ...] [--side darker|brighter]
"""
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, "tools")
from patchlib import gray, load, locate, REF  # noqa: E402

Image.MAX_IMAGE_PIXELS = None


def highpass(img, r=5):
    a = gray(img).astype(np.float32)
    h, w = a.shape
    c = np.pad(np.cumsum(np.cumsum(a, 0), 1), ((1, 0), (1, 0)))
    ys, xs = np.arange(h), np.arange(w)
    y0, y1 = np.clip(ys - r, 0, h), np.clip(ys + r + 1, 0, h)
    x0, x1 = np.clip(xs - r, 0, w), np.clip(xs + r + 1, 0, w)
    s = c[np.ix_(y1, x1)] - c[np.ix_(y0, x1)] - c[np.ix_(y1, x0)] + c[np.ix_(y0, x0)]
    lp = s / ((y1 - y0)[:, None] * (x1 - x0)[None, :]).astype(np.float32)
    return np.abs(a - lp)


def main():
    args = sys.argv[1:]
    side = "darker"
    if "--side" in args:
        i = args.index("--side")
        side = args[i + 1]
        del args[i:i + 2]
    x0, x1, y0, y1 = (int(v) for v in args[0].split(":"))
    args = args[1:]

    ref = load(REF)
    variants = [("AF export", ref, (0, 0))]
    for a in args:
        lab, path = a.split("=", 1)
        img = load(path)
        oy, ox, _ = locate(img, ref)
        variants.append((lab, img, (oy, ox)))

    # Silhouette per row, from the reference.
    spots = []
    for y in range(y0, y1, 3):
        row = gray(ref[y, x0:x1]).astype(np.float32)
        g = np.abs(np.diff(row))
        k = int(np.argmax(g))
        xs = x0 + k
        left = row[max(k - 20, 0):k].mean()
        right = row[k + 1:k + 21].mean()
        if side == "darker" and right > left:
            continue
        if side == "brighter" and left > right:
            continue
        spots.append((y, xs))
    if not spots:
        print("no silhouette found in that box")
        return
    print("silhouette found in %d rows, x %d..%d, measuring on the %s side"
          % (len(spots), min(x for _, x in spots), max(x for _, x in spots), side))

    en = {lab: highpass(img) for lab, img, _ in variants}
    dxs = [2, 4, 6, 8, 10, 12, 16, 20, 24, 30, 36, 44, 54, 64, 80]
    print("%6s" % "dx" + "".join("%16s" % lab for lab, _, _ in variants))
    for dx in dxs:
        line = "%6d" % dx
        for lab, img, (oy, ox) in variants:
            vals = [en[lab][oy + y, ox + x - dx] for y, x in spots]
            line += "%16.2f" % float(np.mean(vals))
        print(line)


if __name__ == "__main__":
    main()
