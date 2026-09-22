"""Quick reconnaissance of the focus-bracketed stack.

Measures per-frame mean brightness (exposure drift), and estimates the
translation/scale between the first frame and every other frame via
multi-scale normalised cross correlation on gradient magnitude.
"""
import os
import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
D = os.environ.get("STACK_DIR", "stack")
files = sorted(f for f in os.listdir(D) if f.upper().startswith("DSC") and f.upper().endswith(".JPG"))
print("frames:", len(files), files[0], "..", files[-1])

SCALE = 8


def load_gray(name):
    im = Image.open(os.path.join(D, name))
    im.draft("L", (im.width // SCALE, im.height // SCALE))
    g = np.asarray(im.convert("L"), dtype=np.float32)
    return g


def grad_mag(g):
    gy, gx = np.gradient(g)
    return np.hypot(gx, gy)


def best_shift(ref, mov, search=6):
    """Exhaustive search of integer shifts on 4x-downsampled gradient maps."""
    r4 = ref[::4, ::4]
    m4 = mov[::4, ::4]
    h = min(r4.shape[0], m4.shape[0]) - 2 * search
    w = min(r4.shape[1], m4.shape[1]) - 2 * search
    r = r4[search:search + h, search:search + w]
    r = r - r.mean()
    rn = np.sqrt((r * r).sum())
    best = (None, -2.0)
    for dy in range(-search, search + 1):
        for dx in range(-search, search + 1):
            m = m4[search + dy:search + dy + h, search + dx:search + dx + w]
            m = m - m.mean()
            d = rn * np.sqrt((m * m).sum())
            if d <= 0:
                continue
            c = float((r * m).sum() / d)
            if c > best[1]:
                best = ((dx, dy), c)
    (dx, dy), c = best
    return dx * 4 * SCALE, dy * 4 * SCALE, c


# Load a subset for speed: first, middle, last few + sampling every 5th
idx = list(range(0, len(files), 5))
if len(files) - 1 not in idx:
    idx.append(len(files) - 1)

grays = {}
stats = []
for i, f in enumerate(files):
    g = load_gray(f)
    grays[i] = g
    stats.append((f, g.mean(), g.std()))

print("\n%.3f %s" % (0, "frame, mean, std"))
for f, m, s in stats[::5]:
    print("  %-14s mean=%7.3f std=%7.3f" % (f, m, s))

ref = grays[0]
refg = grad_mag(ref)
print("\nalignment of frame i vs frame 0 (full-res pixels):")
for i in idx:
    dx, dy, c = best_shift(refg, grad_mag(grays[i]))
    print("  frame %3d (%s): dx=%+6d dy=%+6d  ncc=%.4f" % (i, files[i], dx, dy, c))

# vertical extent of focus: crude per-frame sharpness profile
print("\nper-frame sharpness (gradient energy) profile:")
prof = []
for i in range(len(files)):
    prof.append(float((grad_mag(grays[i]) ** 2).mean()))
for i in range(0, len(prof), 3):
    print("  %3d %s %.5f" % (i, files[i], prof[i]))
