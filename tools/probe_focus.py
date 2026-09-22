"""Decisive test for the soft regions: what does the focus measure actually say?

For each problem patch we compute the Sum of Modified Laplacian (the same metric
the tool uses) for every frame, at the same analysis resolution and window
radius, and compare the peak position with the depth value the tool ended up
using. That separates "the depth estimate was wrong" from "the depth estimate
was right but got overwritten during hole filling".
"""
import os
import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
D = os.environ.get("STACK_DIR", "stack")
AF_X, AF_Y = 1288, 488
CV_X, CV_Y = 20, 13
SCALE = 2
RADIUS = 8

files = sorted(f for f in os.listdir(D) if f.upper().startswith("DSC") and f.upper().endswith(".JPG"))
n = len(files)

depth_raw = np.asarray(Image.open(r"B:\FocusMerge\out\depth.png"), dtype=np.float32) / 65535.0 * (n - 1)
depth_fin = np.asarray(Image.open(r"B:\FocusMerge\out\depth_final.png"), dtype=np.float32) / 65535.0 * (n - 1)
conf = np.asarray(Image.open(r"B:\FocusMerge\out\conf.png"), dtype=np.float32)

# problem patches in reference-crop coordinates (block origin, size)
PATCHES = [
    ("base turntable A", 640, 3584),
    ("base turntable B", 1152, 3840),
    ("yellow leaf", 1408, 3072),
    ("base front", 1792, 3840),
    ("leaf right", 3200, 3072),
]
SIZE = 160


def to_analysis(x, y):
    return (AF_X + x + CV_X) // SCALE, (AF_Y + y + CV_Y) // SCALE


def box_mean(a, r):
    c = np.cumsum(np.cumsum(a, axis=0), axis=1)
    c = np.pad(c, ((1, 0), (1, 0)))
    h, w = a.shape
    ys = np.arange(h)
    xs = np.arange(w)
    y0 = np.clip(ys - r, 0, h)
    y1 = np.clip(ys + r + 1, 0, h)
    x0 = np.clip(xs - r, 0, w)
    x1 = np.clip(xs + r + 1, 0, w)
    s = c[np.ix_(y1, x1)] - c[np.ix_(y0, x1)] - c[np.ix_(y1, x0)] + c[np.ix_(y0, x0)]
    area = ((y1 - y0)[:, None] * (x1 - x0)[None, :]).astype(np.float32)
    return s / area


def sml(gray, r=RADIUS, step=2):
    g = gray
    c = g[1:-1, 1:-1] if False else g
    # modified laplacian with border clamping
    gp = np.pad(g, step, mode="edge")
    cc = gp[step:-step, step:-step]
    ml = (np.abs(2 * cc - gp[step:-step, :-2 * step] - gp[step:-step, 2 * step:])
          + np.abs(2 * cc - gp[:-2 * step, step:-step] - gp[2 * step:, step:-step]))
    return box_mean(ml, r)


# preload analysis-resolution grays once
print("loading %d frames at 1/%d ..." % (n, SCALE))
grays = []
for i, f in enumerate(files):
    im = Image.open(os.path.join(D, f)).convert("L")
    im = im.resize((im.width // SCALE, im.height // SCALE), Image.BOX)
    grays.append(np.asarray(im, dtype=np.float32))
    if (i + 1) % 25 == 0:
        print("  %d/%d" % (i + 1, n))

print("\n%-18s %9s %9s %9s %9s" % ("patch", "rawdepth", "findepth", "SML peak", "conf%"))
for name, bx, by in PATCHES:
    ax, ay = to_analysis(bx, by)
    s = SIZE // SCALE
    dr = depth_raw[ay:ay + s, ax:ax + s]
    df = depth_fin[ay:ay + s, ax:ax + s]
    cf = conf[ay:ay + s, ax:ax + s]
    scores = np.zeros(n, dtype=np.float32)
    for k, g in enumerate(grays):
        m = sml(g[ay:ay + s, ax:ax + s])
        scores[k] = np.median(m)
    peak = int(np.argmax(scores))
    top = np.argsort(scores)[-4:][::-1]
    print("%-18s %9.1f %9.1f %9d   top4=%s  %6.0f%%"
          % (name, np.median(dr), np.median(df), peak,
             ",".join(str(int(t)) for t in top),
             100.0 * float(cf.mean() / max(conf.max(), 1e-6))))

# the depth profile around the yellow leaf: is the smoothed field smearing across
# the leaf / grass boundary?
print("\nsanity: overall depth stats raw vs final")
print("  raw   min %.1f median %.1f max %.1f" % (depth_raw.min(), np.median(depth_raw), depth_raw.max()))
print("  final min %.1f median %.1f max %.1f" % (depth_fin.min(), np.median(depth_fin), depth_fin.max()))
d = np.abs(depth_fin - depth_raw)
print("  mean |final-raw| = %.2f frames, p95 = %.2f frames" % (d.mean(), np.percentile(d, 95)))
