"""How far outside the stack's per-pixel value range does the output go?

The invariant a correct fusion must satisfy is simple: every output value has to
lie between the minimum and the maximum of the frames that were fused, at that
pixel. A Laplacian reconstruction can break it, because the bright side's
overshoot and the dark side's undershoot at a high-contrast edge come from
different frames and their sum then sits outside what any frame contains.

Measured on this stack that is not academic: the dark side produced pixels
darker than every frame (bug 6) and the bright side produces a bright rim along
a silhouette — the halo. This reports both, so a clamp setting can be judged by
how much of each it removes. It needs `--save-depth`-style data, i.e. the
per-pixel frame min/max cached by `tools/evalset`. See below.

    python tools/overshoot.py <minmax.npy> "label=path" [...]
"""
import os
import sys

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from patchlib import REF, load  # noqa: E402

CO = (12, 18)  # canvas = output + CO
# the silhouette that has been the trouble spot, in canvas coordinates
BAND = (2740, 2980, 2700, 2900)


def gray(x):
    return (0.299 * x[..., 0] + 0.587 * x[..., 1] + 0.114 * x[..., 2]).astype(np.float64)


def main():
    mn, mx = np.load(sys.argv[1])
    print("frame range: %s   (per-pixel min/max over the stack, canvas grid)" % (mn.shape,))
    print("%-16s %9s %9s %9s %9s   %9s %9s" % (
        "variant", "over>4", "over>8", "over>16", "over max", "under>8", "under max"))
    for a in sys.argv[2:]:
        lab, _, p = a.partition("=")
        g = gray(load(p))
        H = min(g.shape[0] - CO[0], mn.shape[0])
        W = min(g.shape[1] - CO[1], mn.shape[1])
        G = g[CO[0] : CO[0] + H, CO[1] : CO[1] + W]
        LO = mn[:H, :W]
        HI = mx[:H, :W]
        over = G - HI
        under = LO - G
        print("%-16s %8.3f%% %8.3f%% %8.3f%% %9.1f   %8.3f%% %9.1f" % (
            lab, 100 * (over > 4).mean(), 100 * (over > 8).mean(), 100 * (over > 16).mean(),
            over.max(), 100 * (under > 8).mean(), under.max()))
    print()
    print("in the silhouette band, canvas y %d..%d x %d..%d:" % BAND)
    print("%-16s %9s %9s %9s   %9s" % ("variant", "over>4", "over>8", "over max", "under>8"))
    for a in sys.argv[2:]:
        lab, _, p = a.partition("=")
        g = gray(load(p))
        y0, y1, x0, x1 = BAND
        G = g[y0 - CO[0] : y1 - CO[0], x0 - CO[1] : x1 - CO[1]]
        LO = mn[y0:y1, x0:x1]
        HI = mx[y0:y1, x0:x1]
        over = G - HI
        under = LO - G
        print("%-16s %8.2f%% %8.2f%% %9.1f   %8.2f%%" % (
            lab, 100 * (over > 4).mean(), 100 * (over > 8).mean(), over.max(),
            100 * (under > 8).mean()))


if __name__ == "__main__":
    main()
