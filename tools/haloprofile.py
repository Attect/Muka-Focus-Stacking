"""Cross-edge luminance profile: the halo indicator.

A halo is a *localised* excursion in brightness next to a silhouette — the
object's out-of-focus image bleeding onto the background, or a sharp copy of the
edge appearing where the depth step was placed. Both show up as a systematic
bump or dip in the profile within ~20 px of the edge, so the honest way to see
one is to align many rows on the same edge and average.

Rows are anchored on the reference's edge (the strongest gradient in the search
window), so the profiles of all images are comparable; the edge found in each
image is reported too, because a displaced edge is itself a symptom.

Canvas mapping (verified): canvas = output + (12, 18) = reference + (506, 1304)

    python tools/haloprofile.py y0 y1 x0 x1 "label=path" [...]
"""
import os
import sys

import numpy as np
from PIL import Image, ImageDraw

Image.MAX_IMAGE_PIXELS = None
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from patchlib import REF, load  # noqa: E402

CANVAS_FROM_OUT = (12, 18)
CANVAS_FROM_REF = (506, 1304)
R = 26  # profile reach, pixels


def gray(x):
    return (0.299 * x[..., 0] + 0.587 * x[..., 1] + 0.114 * x[..., 2]).astype(np.float64)


def find_edge(g, y, x0, x1):
    """Strongest |dL| in [x0, x1] on row y, as a canvas x."""
    row = g[y - CANVAS_FROM_REF[0], x0 - CANVAS_FROM_REF[1] : x1 - CANVAS_FROM_REF[1]]
    d = np.abs(np.diff(row))
    return x0 + int(np.argmax(d)) + 1, float(d.max())


def main():
    y0, y1, x0, x1 = (int(v) for v in sys.argv[1:5])
    variants = [("AF export", REF, CANVAS_FROM_REF)]
    for a in sys.argv[5:]:
        lab, _, p = a.partition("=")
        variants.append((lab, p, CANVAS_FROM_OUT))
    G = {lab: gray(load(p)) for lab, p, _ in variants}

    # anchor rows on the reference
    rows = []
    for y in range(y0, y1 + 1, 10):
        ex, strength = find_edge(G["AF export"], y, x0, x1)
        if strength < 25 or ex - R < x0 - 20 or ex + R > x1 + 20:
            continue
        rows.append((y, ex))
    if not rows:
        print("no edges found in that window")
        return
    print("anchored on %d rows, edge x %d..%d (reference)" % (len(rows), min(e for _, e in rows),
                                                             max(e for _, e in rows)))
    prof = {}
    for lab, p, off in variants:
        g = G[lab]
        acc = np.zeros(2 * R + 1)
        for y, ex in rows:
            yy, xx = y - off[0], ex - off[1]
            acc += g[yy, xx - R : xx + R + 1]
        prof[lab] = acc / len(rows)
    # where does each image's own edge sit?
    print("\nedge position found in each image (median, and spread across rows):")
    for lab, p, off in variants:
        g = G[lab]
        e = []
        for y, _ in rows:
            yy = y - off[0]
            seg = g[yy, x0 - off[1] : x1 - off[1]]
            e.append(x0 + int(np.argmax(np.abs(np.diff(seg)))) + 1)
        print("   %-16s %6.1f   spread %.1f px" % (lab, np.median(e), np.std(e)))

    ds = list(range(-R, R + 1))
    print("\nluminance relative to the value 20 px inside the background (d<0 = background side):")
    print("%6s" % "d" + "".join("%16s" % lab for lab, _, _ in variants))
    for i, d in enumerate(ds):
        if d % 2:
            continue
        print("%6d" % d + "".join("%16.1f" % (prof[lab][i] - prof[lab][R - 20]) for lab, _, _ in variants))

    print("\nhalo check — extreme excursion within +-20 px of the edge, relative to the "
          "background level 20 px away (positive = brighter bump, negative = darker dip):")
    for lab, _, _ in variants:
        v = prof[lab] - prof[lab][R - 20]
        near = v[R - 20 : R + 21]
        print("   %-16s max %+6.1f at d=%+3d   min %+6.1f at d=%+3d   peak-to-peak %.1f"
              % (lab, near.max(), ds[R - 20 + int(np.argmax(near))],
                 near.min(), ds[R - 20 + int(np.argmin(near))], near.max() - near.min()))

    # plot
    W, H, pad = 900, 420, 60
    img = Image.new("RGB", (W, H), (255, 255, 255))
    dr = ImageDraw.Draw(img)
    allv = np.concatenate([prof[lab] for lab, _, _ in variants])
    lo, hi = allv.min() - 4, allv.max() + 4
    for k in range(0, 6):
        val = lo + (hi - lo) * k / 5
        py = H - pad - int((val - lo) / (hi - lo) * (H - 2 * pad))
        dr.line([(pad, py), (W - pad, py)], fill=(230, 230, 230))
        dr.text((6, py - 6), "%.0f" % val, fill=(120, 120, 120))
    cols = [(220, 60, 60), (30, 120, 220), (40, 160, 80), (200, 140, 20), (140, 60, 200)]
    for vi, (lab, _, _) in enumerate(variants):
        pts = []
        for i, d in enumerate(ds):
            px = pad + int((d + R) / (2 * R) * (W - 2 * pad))
            py = H - pad - int((prof[lab][i] - lo) / (hi - lo) * (H - 2 * pad))
            pts.append((px, py))
        dr.line(pts, fill=cols[vi % len(cols)], width=2)
        dr.text((pad + 10, 12 + vi * 16), lab, fill=cols[vi % len(cols)])
    px0 = pad + int((0 + R) / (2 * R) * (W - 2 * pad))
    dr.line([(px0, pad - 20), (px0, H - pad)], fill=(0, 0, 0), width=1)
    dr.text((px0 + 4, pad - 18), "edge (reference)", fill=(0, 0, 0))
    dr.text((W - 200, H - 18), "d = distance from edge, px", fill=(90, 90, 90))
    img.save("out/halo_profile.png")
    print("\nwrote out/halo_profile.png")


if __name__ == "__main__":
    main()
