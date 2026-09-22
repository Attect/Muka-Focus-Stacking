"""Edge-adjacent texture and brightness profiles, averaged over *straight* rows.

Two things go wrong along a silhouette and they are different:

* a **halo** — a brightness excursion next to the edge (the object's
  out-of-focus image on the background, or a sharp copy of the edge where the
  depth step was placed);
* a **smooth strip** — the background's texture vanishing for a few pixels next
  to the edge, because that strip is rendered from a frame in which the
  background is defocused.

Both need rows aligned on the same edge, so only rows whose edge sits within a
few pixels of the median are averaged; the rest are dropped, otherwise the
average smears the edge and hides exactly what we are looking for.

Canvas mapping (verified): canvas = output + (12, 18) = reference + (506, 1304)

    python tools/halostrip.py y0 y1 x0 x1 "label=path" [...]
"""
import os
import sys

import numpy as np
from PIL import Image, ImageDraw

Image.MAX_IMAGE_PIXELS = None
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from patchlib import REF, load  # noqa: E402

CO = (12, 18)
CR = (506, 1304)
R = 30  # profile reach in px, d<0 = background side


def gray(x):
    return (0.299 * x[..., 0] + 0.587 * x[..., 1] + 0.114 * x[..., 2]).astype(np.float64)


def hp(m, k=2):
    kk = 2 * k + 1
    p = np.pad(m, ((k + 1, k), (k + 1, k)), mode="edge")
    c = np.cumsum(np.cumsum(p, 0), 1)
    return np.abs(m - (c[kk:, kk:] - c[:-kk, kk:] - c[kk:, :-kk] + c[:-kk, :-kk]) / float(kk * kk))


def main():
    y0, y1, x0, x1 = (int(v) for v in sys.argv[1:5])
    V = [("AF export", REF, CR)]
    for a in sys.argv[5:]:
        lab, _, p = a.partition("=")
        V.append((lab, p, CO))
    G = {lab: gray(load(p)) for lab, p, _ in V}
    H = {lab: hp(G[lab], 2) for lab, _, _ in V}

    # rows whose reference edge is within 4 px of the median
    cand = []
    for y in range(y0, y1 + 1, 4):
        r = G["AF export"][y - CR[0], x0 - CR[1] : x1 - CR[1]]
        d = np.abs(np.diff(r))
        i = int(np.argmax(d))
        if d[i] < 30:
            continue
        cand.append((y, x0 + i + 1, float(d[i])))
    exs = np.array([e for _, e, _ in cand])
    # The silhouette is curved and the window may contain more than one edge, so
    # take the *largest cluster* of edge positions rather than the median of all:
    # a straight run of rows is what makes the averaged profile meaningful.
    bins = {}
    for e in exs:
        bins.setdefault(int(e) // 4, []).append(e)
    best = max(bins.values(), key=len)
    med = float(np.median(best))
    rows = [(y, e) for y, e, _ in cand if abs(e - med) <= 3]
    print("edge cluster at x=%.1f (%d of %d rows) -> %d straight rows kept"
          % (med, len(best), len(cand), len(rows)))
    if len(rows) < 5:
        print("too few straight rows; widen the window or shift it")
        return

    ds = np.arange(-R, 11)
    P = {}
    E = {}
    for lab, p, off in V:
        acc = np.zeros(len(ds))
        acce = np.zeros(len(ds))
        for y, ex in rows:
            yy, xx = y - off[0], ex - off[1]
            acc += G[lab][yy, xx - R : xx + 11]
            acce += H[lab][yy, xx - R : xx + 11]
        P[lab] = acc / len(rows)
        E[lab] = acce / len(rows)

    print("\nluminance, relative to the value 30 px out in the background:")
    print("%5s" % "d" + "".join("%13s" % lab for lab, _, _ in V))
    for i, d in enumerate(ds):
        if d % 2:
            continue
        print("%5d" % d + "".join("%13.1f" % (P[lab][i] - P[lab][0]) for lab, _, _ in V))

    print("\nbackground texture (|highpass| 5x5), relative to 30 px out:")
    print("%5s" % "d" + "".join("%13s" % lab for lab, _, _ in V))
    for i, d in enumerate(ds):
        if d % 2:
            continue
        print("%5d" % d + "".join("%13.1f" % (E[lab][i] - E[lab][0]) for lab, _, _ in V))

    print("\nsummary — near the edge (5..15 px out) versus far out (25..30 px out):")
    near = (ds <= -5) & (ds >= -15)
    far = (ds <= -25)
    for lab, _, _ in V:
        print("   %-16s texture near %6.2f  far %6.2f  ratio %5.2f   luminance near-far %+5.1f"
              % (lab, E[lab][near].mean(), E[lab][far].mean(),
                 E[lab][near].mean() / max(E[lab][far].mean(), 1e-6),
                 P[lab][near].mean() - P[lab][far].mean()))

    # plot both profiles
    W, Hh, pad = 960, 460, 60
    img = Image.new("RGB", (W, Hh), (255, 255, 255))
    dr = ImageDraw.Draw(img)
    cols = [(220, 60, 60), (30, 120, 220), (40, 160, 80), (200, 140, 20), (140, 60, 200)]
    for row, (title, data, base) in enumerate([("luminance", P, True), ("texture", E, False)]):
        top = 30 + row * 220
        dr.text((10, top), title, fill=(0, 0, 0))
        allv = np.concatenate([data[lab] for lab, _, _ in V])
        lo, hi = allv.min(), allv.max()
        for vi, (lab, _, _) in enumerate(V):
            pts = []
            for i, d in enumerate(ds):
                px = pad + int((d + R) / (R + 10) * (W - 2 * pad))
                py = top + 190 - int((data[lab][i] - lo) / max(hi - lo, 1e-6) * 165)
                pts.append((px, py))
            dr.line(pts, fill=cols[vi % len(cols)], width=2)
            dr.text((pad + 10 + vi * 150, top + 4), lab, fill=cols[vi % len(cols)])
        px0 = pad + int(R / (R + 10) * (W - 2 * pad))
        dr.line([(px0, top), (px0, top + 195)], fill=(0, 0, 0), width=1)
        dr.text((px0 + 4, top + 180), "edge", fill=(0, 0, 0))
    img.save("out/halo_profiles.png")
    print("\nwrote out/halo_profiles.png")


if __name__ == "__main__":
    main()
