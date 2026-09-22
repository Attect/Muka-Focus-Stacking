"""Sample a saved depth map at named locations and report the frame index.

Coordinates are given in *reference-crop* pixels (the same frame of reference as
AF导出.png) and converted to canvas pixels with AF_X/AF_Y (where the reference
crop sits inside our output) plus CV_X/CV_Y (our output's own offset inside the
alignment canvas). The depth map itself is at analysis resolution, so it is also
divided by the analysis scale.

Usage:
    depthcheck.py <depth.png> <n_frames> <analysis_scale> label:depth.png ...

    python tools/depthcheck.py out/depth_luma_final.png 50 1
"""
import os
import sys

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

D = os.environ.get("STACK_DIR", "stack")
AF_X, AF_Y = 1288, 488
CV_X, CV_Y = 20, 13

# (label, x, y, true frame) in reference-crop coordinates.
POINTS = [
    ("wall", 1560, 1160, 33),
    ("face", 2000, 700, 16),
    ("knee-edge", 1639, 1245, 7),
    ("leaf", 1450, 3100, 12),
    ("boot", 1520, 1180, 33),
    ("backdrop", 2600, 1200, None),
]


def sample(path, n, scale, half=12):
    d = np.asarray(Image.open(path), dtype=np.float32) / 65535.0 * (n - 1)
    out = []
    for lab, x, y, truth in POINTS:
        cx = (AF_X + x + CV_X) // scale
        cy = (AF_Y + y + CV_Y) // scale
        out.append(float(np.median(d[cy - half:cy + half, cx - half:cx + half])))
    return out


def main():
    files = sys.argv[1:]
    if not files:
        print(__doc__)
        return
    n = 50
    scale = 1
    if files and files[0].isdigit():
        n = int(files.pop(0))
    if files and files[0].isdigit():
        scale = int(files.pop(0))
    rows = []
    for spec in files:
        if "=" in spec:
            lab, path = spec.split("=", 1)
        else:
            lab, path = os.path.basename(spec).replace("depth_", "").replace("_final", ""), spec
        rows.append((lab, sample(path, n, scale)))
    print("%-18s %s" % ("true frame", " ".join("%9s" % (t if t else "-") for _, _, _, t in POINTS)))
    print("%-18s %s" % ("location", " ".join("%9s" % p[0] for p in POINTS)))
    for lab, vals in rows:
        err = [abs(v - t) for v, (_, _, _, t) in zip(vals, POINTS) if t is not None]
        print("%-18s %s   | mean|err| %.1f" % (lab, " ".join("%9.1f" % v for v in vals), np.mean(err)))


if __name__ == "__main__":
    main()
