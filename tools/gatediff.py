"""How much did the picture change, and where?

A whole-frame change statistic is the fairest way to say whether a new setting
is "free": grain is sub-pixel, but a setting that also overrides decisive
winners alters the picture globally, and that is what the eye notices at a
distance.

    python tools/gatediff.py <reference.png> <label=path> [<label=path> ...]
"""
import os
import sys

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from patchlib import REF, load  # noqa: E402

REGIONS = [
    ("backdrop wall (upper left)", 200, 1500, 200, 2500),
    ("backdrop wall (right)", 1500, 2900, 4000, 6800),
    ("subject / figure", 2000, 4100, 1900, 4100),
    ("base / leaf (smooth)", 4100, 4600, 2500, 3900),
]


def main():
    base = load(sys.argv[1]).astype(np.int16)[:, :, :3]
    print("%-14s %8s %8s %8s %8s   %s" % ("variant", ">1", ">4", ">8", "mean", "subject >4"))
    for a in sys.argv[2:]:
        lab, _, p = a.partition("=")
        im = load(p).astype(np.int16)[:, :, :3]
        d = np.abs(im - base).max(axis=2)
        sub = d[2000 - 12 : 4100 - 12, 1900 - 18 : 4100 - 18]
        print("%-14s %7.2f%% %7.2f%% %7.2f%% %8.3f   %6.2f%%"
              % (lab, 100 * (d > 1).mean(), 100 * (d > 4).mean(), 100 * (d > 8).mean(),
                 d.mean(), 100 * (sub > 4).mean()))
    print()
    print("per-region mean |diff| (canvas boxes):")
    print("%-28s" % "" + "".join("%14s" % a.partition("=")[0] for a in sys.argv[2:]))
    for rlab, cy0, cy1, cx0, cx1 in REGIONS:
        line = "%-28s" % rlab
        for a in sys.argv[2:]:
            im = load(a.partition("=")[2]).astype(np.int16)[:, :, :3]
            w = np.abs(im - base)[cy0 - 12 : cy1 - 12, cx0 - 18 : cx1 - 18]
            line += "%14.3f" % w.mean()
        print(line)


if __name__ == "__main__":
    main()
