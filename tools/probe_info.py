"""Where does the focus information actually live?

Replicates the tool's focus measure exactly (analysis resolution, prefilter,
SML with step 2, box aggregation) and reports the per-frame response profile at
selected locations, so we can see whether a pixel's focus signal really peaks at
the frame that is visually sharpest — and how strong that peak is compared with
the noise.

This is the measurement that decides whether the depth field is failing because
it lacks information, or because the information is being swamped.
"""
import os
import sys
import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
D = os.environ.get("STACK_DIR", "stack")
AF_X, AF_Y, CV_X, CV_Y, S = 1288, 488, 20, 13, 2
PREFILTER = 1
RADIUS = 8
STEP = 2
files = sorted(f for f in os.listdir(D) if f.upper().startswith("DSC") and f.upper().endswith(".JPG"))
N = len(files)

# (label, ref-crop x, y)  -- sampled over a small patch centred here
POINTS = [
    ("wall (textured)", 1500, 1130),
    ("wall 2", 1560, 1180),
    ("boundary", 1612, 1240),
    ("object interior", 1650, 1300),
    ("object interior 2", 1700, 1330),
    ("face (control)", 2000, 700),
    ("leaf (control)", 1450, 3100),
]
PATCH = 24  # half-size in reference pixels


def box_mean(a, r):
    h, w = a.shape
    c = np.pad(np.cumsum(np.cumsum(a, axis=0), axis=1), ((1, 0), (1, 0)))
    ys, xs = np.arange(h), np.arange(w)
    y0, y1 = np.clip(ys - r, 0, h), np.clip(ys + r + 1, 0, h)
    x0, x1 = np.clip(xs - r, 0, w), np.clip(xs + r + 1, 0, w)
    s = c[np.ix_(y1, x1)] - c[np.ix_(y0, x1)] - c[np.ix_(y1, x0)] + c[np.ix_(y0, x0)]
    area = ((y1 - y0)[:, None] * (x1 - x0)[None, :]).astype(np.float32)
    return s / area


def focus_map(gray):
    """Same pipeline as src/focus.rs with --focus sml."""
    g = box_mean(np.pad(gray, PREFILTER, mode="edge"),
                 0) if PREFILTER == 0 else None
    if PREFILTER > 0:
        g = box_mean(gray, PREFILTER)
    else:
        g = gray
    p = np.pad(g, STEP, mode="edge")
    c = p[STEP:-STEP, STEP:-STEP]
    ml = (np.abs(2 * c - p[STEP:-STEP, :-2 * STEP] - p[STEP:-STEP, 2 * STEP:])
          + np.abs(2 * c - p[:-2 * STEP, STEP:-STEP] - p[2 * STEP:, STEP:-STEP]))
    return box_mean(ml, RADIUS)


print("computing focus maps for %d frames ..." % N)
profiles = {lab: np.zeros(N) for lab, _, _ in POINTS}
for i, f in enumerate(files):
    im = Image.open(os.path.join(D, f)).convert("L")
    im = im.resize((im.width // S, im.height // S), Image.BOX)
    g = np.asarray(im, dtype=np.float32) * (1.0 / 255.0)
    fm = focus_map(g)
    for lab, x, y in POINTS:
        ax = (AF_X + x + CV_X) // S
        ay = (AF_Y + y + CV_Y) // S
        h = PATCH // S
        profiles[lab][i] = np.median(fm[ay - h:ay + h, ax - h:ax + h])
    if (i + 1) % 10 == 0:
        print("  %d/%d" % (i + 1, N))

print("\n%-20s %7s %7s %8s %9s   %s" % ("location", "peak", "z", "2nd", "noise", "top5 frames"))
for lab, _, _ in POINTS:
    p = profiles[lab]
    order = np.argsort(p)[::-1]
    peak = order[0]
    mu, sd = p.mean(), p.std()
    z = (p[peak] - mu) / (sd + 1e-12)
    # noise level: spread of the values away from the peak
    rest = np.delete(p, peak)
    print("%-20s %7d %7.2f %8d %9.5f   %s"
          % (lab, peak, z, order[1], rest.std(),
             ",".join(str(int(t)) for t in order[:5])))
    print("%-20s   profile: %s" % ("", np.round(np.sort(p)[::-1][:6] / (p.max() + 1e-12), 3)))
