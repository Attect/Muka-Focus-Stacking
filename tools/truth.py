"""Independent ground truth for the depth of a named location.

For each raw frame, take a patch around the point (in reference-crop
coordinates) and measure its sharpness directly on the decoded pixels, then
report which frames are sharpest. This uses nothing from the tool's own focus
measure, so it is a genuine reference to compare a depth map against.

The patch is reduced by a small integer factor first, because at full resolution
the sharpness of a smooth surface is dominated by sensor noise; averaging 4x4
keeping the raw signal is a reasonable compromise (BOX is an average, so this is
not the same as throwing detail away).

Usage:
    truth.py [patch_px] [downsample] point:x:y ...
    python tools/truth.py 64 4 leaf:1450:3100 wall:1560:1160
"""
import os
import sys

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

D = os.environ.get("STACK_DIR", "stack")
AF_X, AF_Y = 1288, 488
CV_X, CV_Y = 20, 13

DEFAULT_POINTS = [
    ("wall", 1560, 1160),
    ("face", 2000, 700),
    ("knee-edge", 1639, 1245),
    ("leaf", 1450, 3100),
    ("boot", 1520, 1180),
]


def sharpness(patch):
    g = patch[..., 0] * 0.299 + patch[..., 1] * 0.587 + patch[..., 2] * 0.114
    gy, gx = np.gradient(g.astype(np.float32))
    return float(np.median(np.hypot(gx, gy)))


def main():
    args = sys.argv[1:]
    size, ds = 64, 4
    if args and args[0].isdigit():
        size = int(args.pop(0))
    if args and args[0].isdigit():
        ds = int(args.pop(0))
    if args:
        points = []
        for a in args:
            lab, x, y = a.split(":")
            points.append((lab, int(x), int(y)))
    else:
        points = DEFAULT_POINTS

    files = sorted(
        f for f in os.listdir(D) if f.upper().startswith("DSC") and f.upper().endswith(".JPG")
    )
    n = len(files)
    prof = {lab: np.zeros(n) for lab, _, _ in points}
    for i, f in enumerate(files):
        im = Image.open(os.path.join(D, f)).convert("RGB")
        if ds > 1:
            im = im.resize((im.width // ds, im.height // ds), Image.BOX)
        a = np.asarray(im, dtype=np.float32)
        for lab, x, y in points:
            cx, cy = (AF_X + x + CV_X) // ds, (AF_Y + y + CV_Y) // ds
            h = size // ds // 2
            prof[lab][i] = sharpness(a[cy - h:cy + h, cx - h:cx + h])
        if (i + 1) % 10 == 0:
            print("  %d/%d" % (i + 1, n), flush=True)

    print()
    print("%-12s %6s %6s %8s   %s" % ("location", "best", "2nd", "peak/med", "top 8 frames"))
    for lab, _, _ in points:
        p = prof[lab]
        order = np.argsort(p)[::-1]
        med = np.median(p)
        print(
            "%-12s %6d %6d %8.2f   %s"
            % (
                lab,
                order[0],
                order[1],
                p[order[0]] / max(med, 1e-9),
                " ".join(str(int(v)) for v in order[:8]),
            )
        )


if __name__ == "__main__":
    main()
