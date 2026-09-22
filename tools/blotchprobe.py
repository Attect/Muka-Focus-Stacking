"""Probe the focus measure inside the blotchy regions, offline and faithfully.

Mirrors the pipeline's own measure: a 1-px box prefilter, then SML over a window
of `--focus-radius`, computed on the *unwarped* frames (the pipeline warps the
response, not the pixels it measures on).

For each point it reports the per-frame response, the peak frame, the
peak-to-median ratio (how well the measure separates the focused frame from the
rest) and the depth the pipeline actually wrote there. That distinguishes "the
measure picked a different frame" from "the measure had nothing to pick from and
the depth pipeline moved it".

    python tools/blotchprobe.py DEPTH.png [N]

Without N it prints the list of points it would probe; with N it runs.
"""
import os
import sys

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from patchlib import load  # noqa: E402

D = os.environ.get("STACK_DIR", "stack")
NFRAMES = 49.0
CANVAS_FROM_OUT = (12, 18)
R = 4  # --focus-radius


def gray(a):
    return (0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]).astype(np.float32)


def box_blur(a, r):
    k = 2 * r + 1
    p = np.pad(a, ((r + 1, r), (r + 1, r)), mode="edge")
    c = np.cumsum(np.cumsum(p, 0, dtype=np.float64), 1, dtype=np.float64)
    return ((c[k:, k:] - c[:-k, k:] - c[k:, :-k] + c[:-k, :-k]) / float(k * k)).astype(np.float32)


def sml(a, r):
    k = 2 * r + 1
    p = np.pad(a, ((r + 1, r), (r + 1, r)), mode="edge")
    c = np.cumsum(np.cumsum(p, 0, dtype=np.float64), 1, dtype=np.float64)
    return ((c[k:, k:] - c[:-k, k:] - c[k:, :-k] + c[:-k, :-k]) / float(k * k)).astype(np.float32)


def measure(frame_gray, x, y, r=R):
    w = frame_gray[y - r - 3 : y + r + 4, x - r - 3 : x + r + 4]
    if w.shape != (2 * r + 7, 2 * r + 7):
        return None
    p = box_blur(w, 1)
    c = p[2:-2, 2:-2]
    return float(
        (np.abs(2 * c - p[2:-2, :-4] - p[2:-2, 4:]) + np.abs(2 * c - p[:-4, 2:-2] - p[4:, 2:-2])).mean()
    )


def pick_points(dz):
    """Worst blotches: pixels whose depth sits far from the local median."""
    med = box_blur(dz, 2)
    rough = np.abs(dz - med)
    pts = []
    ys, xs = np.where(rough > 12)
    order = np.argsort(-rough[ys, xs])
    for i in order:
        y, x = int(ys[i]), int(xs[i])
        if any((y - a) ** 2 + (x - b) ** 2 < 400**2 for a, b in pts):
            continue
        pts.append((y, x))
        if len(pts) >= 10:
            break
    return pts, rough


def main():
    dz = np.asarray(Image.open(sys.argv[1]), dtype=np.float32) / 65535.0 * NFRAMES
    pts, rough = pick_points(dz)
    print("probe points (canvas coords), blotchiest first:")
    for y, x in pts:
        print("   (%4d,%4d)  depth %5.1f  local median %5.1f  deviation %5.1f"
              % (y, x, dz[y, x], dz[y, x] - rough[y, x] * np.sign(1), rough[y, x]))
    if len(sys.argv) < 3:
        print("\n(dry run; pass a second argument to actually measure)")
        return

    files = sorted(x for x in os.listdir(D) if x.upper().startswith("DSC") and x.upper().endswith(".JPG"))
    resp = {p: np.zeros(len(files)) for p in pts}
    for k, f in enumerate(files):
        g = gray(np.asarray(Image.open(os.path.join(D, f)).convert("RGB")))
        for p in pts:
            v = measure(g, p[1], p[0])
            resp[p][k] = 0.0 if v is None else v
        if k % 10 == 0:
            print("  frame", k, flush=True)

    print()
    print("%-14s %8s %10s %10s %8s   %s" % ("canvas y,x", "depth", "peak frame", "peak/med", "argmax", "top3 frames"))
    for p in pts:
        v = resp[p]
        mx, mn = v.max(), np.median(v)
        order = np.argsort(-v)[:3]
        print("%-14s %8.1f %10d %10.2f %8d   %s"
              % ("%d,%d" % p, dz[p[0], p[1]], int(np.argmax(v)), mx / max(mn, 1e-9), int(np.argmax(v)),
                 ", ".join("%d:%.3f" % (int(i), v[i]) for i in order)))


if __name__ == "__main__":
    main()
