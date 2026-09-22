"""Shared helpers for the silhouette-band measurements (see bandeval.py).

Everything here works on planes the size of the *output* image, or on a window of
it, so no measurement has to know about crops or alignment.
"""
import sys

import numpy as np

sys.path.insert(0, 'tools')
from patchlib import gray  # noqa: E402


def boxsum(m, r):
    """Count of true pixels within a (2r+1)^2 Chebyshev window."""
    a = m.astype(np.float32)
    p = np.pad(a, r, mode='constant')
    c = np.pad(p.cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    H, W = m.shape
    y = np.arange(H)
    x = np.arange(W)
    return (c[np.ix_(y + 2 * r + 1, x + 2 * r + 1)] - c[np.ix_(y, x + 2 * r + 1)]
            - c[np.ix_(y + 2 * r + 1, x)] + c[np.ix_(y, x)])


def box(a, r):
    return boxsum(a, r) / float((2 * r + 1) ** 2)


def texture_of(g):
    """Fine relief: local std of a 5x5 high-pass."""
    hp = g - box(g, 2)
    return np.sqrt(np.maximum(box(hp * hp, 2), 0.0))


def subject_mask(rgb):
    """The figure, from colour rules only.

    The wall is near-neutral mid-grey (~184). The figure is saturated (red hat,
    orange hook, brown boot), much darker (near-black glove and cuff, which have
    almost no saturation) or much brighter (pale skin, white dress). A saturation
    test alone leaks the dark parts into "wall", which made the first band
    measurement read 53 levels too dark.

    Then a morphological opening: the dark seam between the wallpaper panels is
    only a few pixels wide and is "subject" by these rules, so it would grow a
    band of its own.
    """
    g0 = gray(rgb)
    sat = rgb.max(-1) - rgb.min(-1)
    raw = (sat > 30) | (g0 < 100) | (g0 > 215)
    k = 3
    er = boxsum(raw, k) >= float((2 * k + 1) ** 2)
    return boxsum(er, k) > 0, raw


def build_masks(rgb, bbox, lo=4, hi=22, far=44):
    """band = wall 4..22 px outside the figure; far = wall more than 64 px away."""
    subj, raw = subject_mask(rgb)
    avoid = boxsum(raw & ~subj, 8) > 0
    wall = ~subj & ~avoid
    H, W = subj.shape
    roi = np.zeros((H, W), bool)
    if bbox:
        x0, x1, y0, y1 = bbox
    else:
        x0, x1, y0, y1 = 0, W, 0, H
    roi[y0:y1, x0:x1] = True
    band = (boxsum(subj, hi) > 0) & (boxsum(subj, lo) == 0) & wall & roi
    farw = (boxsum(subj, far) == 0) & wall & roi
    return band, farw


def local_pair(G, T, b, f, w=52, min_support=600):
    """Local band-vs-far comparison, in the same `w`-radius neighbourhood."""
    nb = boxsum(b, w)
    nf = boxsum(f, w)
    ok = b & (nb > min_support) & (nf > min_support)
    lb = box(G * b, w) / np.maximum(nb, 1.0)
    lf = box(G * f, w) / np.maximum(nf, 1.0)
    tb = box(T * b, w) / np.maximum(nb, 1.0)
    tf = box(T * f, w) / np.maximum(nf, 1.0)
    return ok, float((lb[ok] - lf[ok]).mean()), float((tb[ok] / np.maximum(tf[ok], 1e-6)).mean())
