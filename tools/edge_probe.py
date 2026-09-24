"""At a silhouette edge, find both the truly sharpest frame *and* the frame the
tool's own focus measure picks there.

The two answers are what separate the possible causes:
  * if the focus measure picks the right frame but the output is soft, the
    fusion is losing detail;
  * if the measure picks the wrong frame, the focus window is being dominated by
    whatever else it covers — which is the failure mode on a smooth object in
    front of a textured backdrop.

`focus_map` here replicates src/focus.rs exactly (analysis resolution, prefilter,
SML with step 2, box aggregation), so the numbers are the ones the tool sees.

Usage:  edge_probe.py <x> <y> [half_size] [zoom] [depth_path]
        x, y are in reference-crop coordinates.
"""
import os
import sys
import numpy as np
from PIL import Image, ImageDraw

# The project directory, so the paths below work on any machine.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

Image.MAX_IMAGE_PIXELS = None
D = os.environ.get("STACK_DIR", "stack")
AF_X, AF_Y, CV_X, CV_Y, S = 1288, 488, 20, 13, 2
PREFILTER, RADIUS, STEP = 1, 8, 2
files = sorted(f for f in os.listdir(D) if f.upper().startswith("DSC") and f.upper().endswith(".JPG"))
N = len(files)

cx, cy = int(sys.argv[1]), int(sys.argv[2])
half = int(sys.argv[3]) if len(sys.argv) > 3 else 24
zoom = int(sys.argv[4]) if len(sys.argv) > 4 else 4
depth_path = sys.argv[5] if len(sys.argv) > 5 else os.path.join(ROOT, "out/depth_cur_final.png")
raw_path = depth_path.replace("_final", "")

cvx, cvy = AF_X + cx + CV_X, AF_Y + cy + CV_Y
x0, y0 = cx - half, cy - half


def box_mean(a, r):
    h, w = a.shape
    c = np.pad(np.cumsum(np.cumsum(a, axis=0), axis=1), ((1, 0), (1, 0)))
    ys, xs = np.arange(h), np.arange(w)
    y0i, y1i = np.clip(ys - r, 0, h), np.clip(ys + r + 1, 0, h)
    x0i, x1i = np.clip(xs - r, 0, w), np.clip(xs + r + 1, 0, w)
    s = c[np.ix_(y1i, x1i)] - c[np.ix_(y0i, x1i)] - c[np.ix_(y1i, x0i)] + c[np.ix_(y0i, x0i)]
    area = ((y1i - y0i)[:, None] * (x1i - x0i)[None, :]).astype(np.float32)
    return s / area


def focus_map(gray):
    g = box_mean(gray, PREFILTER) if PREFILTER > 0 else gray
    p = np.pad(g, STEP, mode="edge")
    c = p[STEP:-STEP, STEP:-STEP]
    ml = (np.abs(2 * c - p[STEP:-STEP, :-2 * STEP] - p[STEP:-STEP, 2 * STEP:])
          + np.abs(2 * c - p[:-2 * STEP, STEP:-STEP] - p[2 * STEP:, STEP:-STEP]))
    return box_mean(ml, RADIUS)


def acutance(a, q=0.99):
    g = a[..., 0] * 0.299 + a[..., 1] * 0.587 + a[..., 2] * 0.114
    gy, gx = np.gradient(g.astype(np.float32))
    m = np.hypot(gx, gy).ravel()
    return float(m[m >= np.quantile(m, q)].mean())


def edge_strength(a):
    """Steepest single step in the patch: what 'is the outline crisp?' means."""
    g = a[..., 0] * 0.299 + a[..., 1] * 0.587 + a[..., 2] * 0.114
    gy, gx = np.gradient(g.astype(np.float32))
    return float(np.hypot(gx, gy).max())


print("point (%d,%d) ref-crop -> canvas (%d,%d) -> analysis (%d,%d), half=%d"
      % (cx, cy, cvx, cvy, cvx // S, cvy // S, half))

tiles = []
ac = np.zeros(N)
es = np.zeros(N)
sm = np.zeros(N)
for i, f in enumerate(files):
    im = Image.open(os.path.join(D, f)).convert("RGB")
    a = np.asarray(im, dtype=np.float32)
    patch = a[cvy - half:cvy + half, cvx - half:cvx + half]
    ac[i] = acutance(patch)
    es[i] = edge_strength(patch)
    tiles.append(patch)
    g = np.asarray(im.convert("L").resize((im.width // S, im.height // S), Image.BOX),
                   dtype=np.float32) / 255.0
    fm = focus_map(g)
    ax, ay = cvx // S, cvy // S
    h = max(1, half // S)
    sm[i] = np.median(fm[ay - h:ay + h, ax - h:ax + h])
    if (i + 1) % 10 == 0:
        print("  %d/%d" % (i + 1, N))

oa, os_ = np.argsort(ac)[::-1], np.argsort(es)[::-1]
osm = np.argsort(sm)[::-1]
print()
print("per-frame ACUTANCE  (top 8):", ", ".join("%d(%.1f)" % (i, ac[i]) for i in oa[:8]))
print("per-frame EDGE STEP (top 8):", ", ".join("%d(%.1f)" % (i, es[i]) for i in os_[:8]))
print("tool SML response   (top 8):", ", ".join("%d(%.3f)" % (i, sm[i]) for i in osm[:8]))
print()
print("rank correlation  acutance vs SML: %.3f   edge-step vs SML: %.3f"
      % (np.corrcoef(np.argsort(np.argsort(ac)), np.argsort(np.argsort(sm)))[0, 1],
         np.corrcoef(np.argsort(np.argsort(es)), np.argsort(np.argsort(sm)))[0, 1]))
print("SML argmax %d,  acutance argmax %d,  edge-step argmax %d"
      % (osm[0], oa[0], os_[0]))

if os.path.exists(depth_path):
    fin = np.asarray(Image.open(depth_path), dtype=np.float32) / 65535.0 * (N - 1)
    raw = np.asarray(Image.open(raw_path), dtype=np.float32) / 65535.0 * (N - 1)
    ax, ay, s = cvx // S, cvy // S, max(1, half // S)
    print("our depth there: raw %.1f  final %.1f" % (np.median(raw[ay - s:ay + s, ax - s:ax + s]),
                                                     np.median(fin[ay - s:ay + s, ax - s:ax + s])))

show = list(oa[:3]) + list(osm[:3])
show = list(dict.fromkeys(show))
tiles2 = [tiles[i] for i in show]
labels = ["acut %.1f sml %.2f f%d" % (ac[i], sm[i], i) for i in show]
for name, path, off in [("AF export", os.path.join(D, "AF导出.png"), False),
                        ("OURS cur", os.path.join(ROOT, "out/cur.png"), True)]:
    if not os.path.exists(path):
        continue
    a = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32)
    tiles2.append(a[y0:y0 + half * 2, x0:x0 + half * 2] if not off
                  else a[AF_Y + y0:AF_Y + y0 + half * 2, AF_X + x0:AF_X + x0 + half * 2])
    labels.append(name)

big = [np.asarray(Image.fromarray(t.clip(0, 255).astype(np.uint8)).resize(
    (half * 2 * zoom, half * 2 * zoom), Image.NEAREST), dtype=np.float32) for t in tiles2]
cols = 4
rows = (len(big) + cols - 1) // cols
cell = half * 2 * zoom + 8
sheet = np.full((rows * (cell + 26), cols * cell, 3), 255.0, dtype=np.float32)
for k, t in enumerate(big):
    ry, rx = divmod(k, cols)
    sheet[ry * (cell + 26) + 26:ry * (cell + 26) + 26 + t.shape[0],
          rx * cell:rx * cell + t.shape[1]] = t
out = Image.fromarray(sheet.astype(np.uint8))
dr = ImageDraw.Draw(out)
for k, lab in enumerate(labels):
    ry, rx = divmod(k, cols)
    dr.text((rx * cell + 6, ry * (cell + 26) + 8), lab, fill=(0, 0, 0))
p = os.path.join(ROOT, "out/edge_%d_%d.png") % (cx, cy)
out.save(p)
print("wrote", p)
print("tiles:", labels)
