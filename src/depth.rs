//! Building the depth / frame-selection field.
//!
//! The output of this module is a *fractional* frame index per pixel: a value of
//! `13.4` means "this pixel is best described by a blend of 40% frame 14 and
//! 60% frame 13". Having a continuous field (rather than a hard per-pixel pick)
//! is what lets the pyramid fusion produce seamless results, and it is the
//! reason this tool does not show the patchwork that naive "sharpest pixel
//! wins" implementations do.

use crate::pyramid::{downsample_plane, upsample_plane_bilinear};
use crate::util::{box_blur, Plane};
use rayon::prelude::*;

/// Streaming accumulator that turns a sequence of focus maps into a depth field
/// without ever holding more than a couple of maps in memory.
pub struct DepthBuilder {
    w: usize,
    h: usize,
    /// Highest focus measure seen so far at each pixel.
    best_score: Vec<f32>,
    /// Frame index that achieved `best_score`.
    best_idx: Vec<u16>,
    /// Running sum and sum of squares of the focus measure, so the confidence
    /// can be a z-score rather than a bare peak value.
    sum_score: Vec<f32>,
    sum2_score: Vec<f32>,
    /// Focus measure aggregated over a *larger* window, used only for the
    /// reliability statistics.
    ///
    /// This separation matters: on a single pixel the peak-to-mean ratio is
    /// dominated by residual noise (a low-signal region then scores *higher*
    /// than a textured one, which is exactly backwards), while the same ratio
    /// measured on the extra-smoothed response is a clean discriminator.
    sum_conf: Vec<f32>,
    best_conf: Vec<f32>,
    /// Best *local maximum* seen so far (used for sub-index refinement).
    lm_score: Vec<f32>,
    /// Sub-index interpolated around that local maximum.
    lm_idx: Vec<f32>,
    /// Focus map of the previous / second previous frame.
    prev1: Option<Vec<f32>>,
    prev2: Option<Vec<f32>>,
    n: usize,
    /// Optional coarse prior: a depth estimate, in frames, at this resolution.
    ///
    /// When present a frame only competes at a pixel if it lies within `window`
    /// of the prior there. This is what turns the estimate from purely local —
    /// and therefore noise-limited wherever the focus response has little dynamic
    /// range — into a refinement of a measurement made over a larger area, where
    /// that same response has plenty of range.
    ///
    /// Measured on the reference stack, at a textured backdrop pixel: the
    /// per-pixel response varies by only 1.3x across the stack while the true
    /// sharpness varies by 7x, so the local argmax is arbitrary and lands at 18
    /// instead of 33. Averaged over a quarter-resolution cell the response does
    /// peak at the right frame.
    prior: Option<Vec<f32>>,
    /// Half-width, in frames, of the window around the prior.
    window: f32,
    /// Optional per-pixel version of `window`. See [`prior_window`].
    window_map: Option<Vec<f32>>,
}

impl DepthBuilder {
    pub fn new(w: usize, h: usize, n: usize) -> Self {
        Self {
            w,
            h,
            best_score: vec![f32::NEG_INFINITY; w * h],
            best_idx: vec![0u16; w * h],
            sum_score: vec![0.0f32; w * h],
            sum2_score: vec![0.0f32; w * h],
            sum_conf: vec![0.0f32; w * h],
            best_conf: vec![0.0f32; w * h],
            lm_score: vec![f32::NEG_INFINITY; w * h],
            lm_idx: vec![0.0f32; w * h],
            prev1: None,
            prev2: None,
            n,
            prior: None,
            window: f32::INFINITY,
            window_map: None,
        }
    }

    /// Restrict every pixel's search to `window` frames either side of `prior`.
    pub fn set_prior(&mut self, prior: &Plane, window: f32) {
        debug_assert_eq!(prior.w, self.w);
        debug_assert_eq!(prior.h, self.h);
        self.prior = Some(prior.d.clone());
        self.window = window;
    }

    /// Per-pixel search window around the prior, replacing the scalar one. See
    /// [`prior_window`] for why a single value is not enough.
    pub fn set_prior_window(&mut self, w: &Plane) {
        debug_assert_eq!(w.w * w.h, self.w * self.h);
        self.window_map = Some(w.d.clone());
    }

    /// Feed the focus map of frame `k` (frames must arrive in order).
    pub fn push(&mut self, k: usize, fm: &Plane, conf_fm: &Plane) {
        self.sum_conf
            .par_iter_mut()
            .zip(conf_fm.d.par_iter())
            .for_each(|(a, b)| *a += *b);
        self.best_conf
            .par_iter_mut()
            .zip(conf_fm.d.par_iter())
            .for_each(|(a, b)| *a = a.max(*b));
        debug_assert_eq!(fm.w, self.w);
        debug_assert_eq!(fm.h, self.h);
        let cur = &fm.d;
        let prior = self.prior.as_deref();
        let window = self.window;
        let wmap = self.window_map.as_deref();
        self.best_score
            .par_chunks_mut(8192)
            .zip(self.best_idx.par_chunks_mut(8192))
            .zip(self.sum_score.par_chunks_mut(8192))
            .zip(self.sum2_score.par_chunks_mut(8192))
            .zip(cur.par_chunks(8192))
            .enumerate()
            .for_each(|(ci, ((((bsc, bic), ssc), sqc), curc))| {
                let base = ci * 8192;
                for i in 0..bsc.len() {
                    if let Some(pr) = prior {
                        let lim = match wmap {
                            Some(m) => m[base + i],
                            None => window,
                        };
                        if (k as f32 - pr[base + i]).abs() > lim {
                            continue;
                        }
                    }
                    let s = curc[i];
                    ssc[i] += s;
                    sqc[i] += s * s;
                    if s > bsc[i] {
                        bsc[i] = s;
                        bic[i] = k as u16;
                    }
                }
            });

        // Parabolic refinement: when frame k-1 turns out to be a local maximum
        // we can interpolate the true peak position from its three neighbours.
        // Taking the previous maps out of `self` keeps the borrow checker happy
        // and they are put back below.
        let prev2 = self.prev2.take();
        let prev1 = self.prev1.take();
        if k >= 2 {
            let (p2, p1) = (prev2.as_ref().unwrap(), prev1.as_ref().unwrap());
            self.lm_score
                .par_chunks_mut(8192)
                .zip(self.lm_idx.par_chunks_mut(8192))
                .zip(p2.par_chunks(8192))
                .zip(p1.par_chunks(8192))
                .zip(cur.par_chunks(8192))
                .enumerate()
                .for_each(|(ci, ((((lmsc, lmic), p2c), p1c), curc))| {
                    let base = ci * 8192;
                    for i in 0..lmsc.len() {
                        if let Some(pr) = prior {
                            let lim = match wmap {
                                Some(m) => m[base + i],
                                None => window,
                            };
                            if (k as f32 - pr[base + i]).abs() > lim {
                                continue;
                            }
                        }
                        let a = p2c[i];
                        let b = p1c[i];
                        let c = curc[i];
                        if !(b >= a && b > c) || b <= lmsc[i] {
                            continue;
                        }
                        let denom = a - 2.0 * b + c;
                        let delta = if denom.abs() > 1e-12 {
                            (0.5 * (a - c) / denom).clamp(-0.5, 0.5)
                        } else {
                            0.0
                        };
                        lmsc[i] = b;
                        lmic[i] = (k - 1) as f32 + delta;
                    }
                });
        }
        self.prev2 = prev1;
        self.prev1 = Some(cur.clone());
    }

    /// Finish the sweep. Returns `(depth, confidence, known)`.
    ///
    /// The depth uses the refined sub-index where a local maximum was observed
    /// and falls back to the integer argmax elsewhere (monotonic runs at the
    /// ends of the stack).
    ///
    /// The second return value is the **peak-to-mean ratio** of the focus
    /// response over the stack: how much the response varies with focus. It is
    /// the only reliability measure tried here that separates "this pixel has a
    /// focus signal" from "this pixel is noise", and it needs an *absolute*
    /// threshold — a percentile threshold would force a fixed share of the image
    /// to be classified unreliable no matter what.
    ///
    /// Measured on the reference stack: every location with genuine focus
    /// information scores 1.20-1.52, a featureless surface scores 1.09-1.13.
    pub fn finish(mut self) -> (Plane, Plane, Plane) {
        // Frames that never had a usable local maximum: keep the argmax.
        // (a monotonic ramp towards one end of the stack.)
        let last = (self.n - 1) as f32;
        let inv_n = 1.0 / self.n as f32;
        let mut depth = vec![0.0f32; self.w * self.h];
        let mut conf = vec![0.0f32; self.w * self.h];
        let mut known = vec![0.0f32; self.w * self.h];
        depth
            .par_chunks_mut(8192)
            .zip(conf.par_chunks_mut(8192))
            .zip(known.par_chunks_mut(8192))
            .enumerate()
            .for_each(|(ci, ((dc, cc), kc))| {
                let base = ci * 8192;
                for i in 0..dc.len() {
                    let b = base + i;
                    let lm = self.lm_idx[b];
                    let bi = self.best_idx[b] as f32;
                    let peak = self.best_conf[b].max(0.0);
                    let mean = self.sum_conf[b] * inv_n;
                    // Nothing competed here — an empty window — so keep the prior
                    // rather than falling through to frame 0.
                    // The sub-index may only be preferred when it *is* the
                    // global maximum.
                    //
                    // `push` records a local maximum only under the strict test
                    // `b >= a && b > c`, so a true peak that sits on a plateau
                    // (adjacent frames scoring equal, which is common when the
                    // focus steps are small) or at one end of the stack is never
                    // recorded. Taking `lm_idx` unconditionally then let a
                    // *weaker*, earlier bump win the whole pixel: measured on the
                    // reference stack, a backdrop pixel whose response peaks at
                    // frame 34 was written as 13.9 — the neighbour's edge bump —
                    // and that is what left the smooth band along the far side of
                    // the silhouette. The two scores are the same float whenever
                    // the global peak was recorded, so `>=` is exact.
                    let lm_is_global =
                        self.lm_score[b].is_finite() && self.lm_score[b] >= self.best_score[b];
                    let v = if lm_is_global {
                        lm
                    } else if self.best_score[b].is_finite() {
                        bi
                    } else if self.lm_score[b].is_finite() {
                        lm
                    } else if let Some(pr) = &self.prior {
                        pr[b]
                    } else {
                        bi
                    };
                    dc[i] = v.clamp(0.0, last);
                    // Peak-to-mean ratio of the focus response over the stack.
                    //
                    // This is the one reliability measure that actually
                    // separates "this pixel has a focus signal" from "this pixel
                    // is noise": it asks how much the response varies with focus.
                    // Measured on the reference stack, every location with real
                    // focus information scores 1.20-1.52 while a featureless
                    // surface scores 1.09-1.13, so an absolute threshold cleanly
                    // splits them.
                    //
                    // Two earlier attempts failed for instructive reasons. The
                    // raw peak value punishes smooth low-contrast surfaces,
                    // which have a perfectly well placed but small peak. The
                    // z-score fails the other way: a real focus peak in a stack
                    // with small focus steps is broad, so its peak sits only
                    // ~2 sigma above the rest — the same as the maximum of 50
                    // noise samples, making the two indistinguishable.
                    cc[i] = (peak / (mean + 1e-12)).clamp(0.0, 10.0);
                    // Pixels carrying no focus information at all must not
                    // contribute a depth. On a smooth surface JPEG decoding
                    // produces perfectly flat runs of blocks, and over such a run
                    // the response is identical — usually exactly zero — for every
                    // frame: every frame ties, the first one wins, and that
                    // fabricated value then survives the median and is spread
                    // across an otherwise well determined neighbourhood by the
                    // guided filter.
                    //
                    // The test has to look at the *raw* response, not at the
                    // smoothed confidence above. Confidence is aggregated over a
                    // 16-pixel neighbourhood, so a pixel whose own response is
                    // identically zero still inherits a healthy ratio from the
                    // structure around it and would be declared informative — with
                    // its fabricated depth. That mistake put a smooth backdrop,
                    // truly at frame 33, at frame 0 and blurred it.
                    kc[i] = if self.best_score[b] > 1e-6 { 1.0 } else { 0.0 };
                }
            });
        // Release the big score maps before returning.
        self.best_score.clear();
        self.lm_score.clear();
        self.sum_score.clear();
        self.sum2_score.clear();
        self.sum_conf.clear();
        self.best_conf.clear();
        (
            Plane { w: self.w, h: self.h, d: depth },
            Plane { w: self.w, h: self.h, d: conf },
            Plane { w: self.w, h: self.h, d: known },
        )
    }
}

/// Multi-scale (push-pull) inpainting of `vals` wherever `valid` is false.
///
/// Unreliable samples are replaced by the nearest reliable information at
/// whatever scale it first becomes available, which is far more stable than
/// growing regions from a single threshold.
///
/// Superseded by [`geodesic_fill`] for the default path: filling from a coarse
/// level mixes values across object boundaries, which is exactly what a smooth
/// foreground object in front of a textured backdrop cannot tolerate.
#[allow(dead_code)]
pub fn fill_unreliable(vals: &Plane, valid: &Plane, levels: usize) -> Plane {
    let (w, h) = (vals.w, vals.h);
    let mut nums: Vec<Vec<f32>> = Vec::with_capacity(levels + 1);
    let mut dens: Vec<Vec<f32>> = Vec::with_capacity(levels + 1);
    let mut sizes: Vec<(usize, usize)> = Vec::with_capacity(levels + 1);

    let mut num: Vec<f32> = vals
        .d
        .par_iter()
        .zip(valid.d.par_iter())
        .map(|(v, m)| v * m)
        .collect();
    let mut den: Vec<f32> = valid.d.clone();
    let (mut cw, mut ch) = (w, h);
    nums.push(num.clone());
    dens.push(den.clone());
    sizes.push((cw, ch));

    for _ in 0..levels {
        if cw <= 4 || ch <= 4 {
            break;
        }
        let (n2, w2, h2) = downsample_plane(&num, cw, ch);
        let (d2, _, _) = downsample_plane(&den, cw, ch);
        num = n2;
        den = d2;
        cw = w2;
        ch = h2;
        nums.push(num.clone());
        dens.push(den.clone());
        sizes.push((cw, ch));
    }

    // Coarsest level: normalise, fall back to the global mean where empty.
    let last = nums.len() - 1;
    let (lw, lh) = sizes[last];
    let mut glob = 0.0f32;
    let mut glob_n = 0.0f32;
    for i in 0..lw * lh {
        if dens[last][i] > 0.0 {
            glob += nums[last][i] / dens[last][i];
            glob_n += 1.0;
        }
    }
    let glob = if glob_n > 0.0 { glob / glob_n } else { 0.0 };
    let mut cur: Vec<f32> = (0..lw * lh)
        .map(|i| {
            if dens[last][i] > 1e-6 {
                nums[last][i] / dens[last][i]
            } else {
                glob
            }
        })
        .collect();

    // Push back down to level 0.
    for l in (0..last).rev() {
        let (pw, ph) = sizes[l];
        let up = upsample_plane_bilinear(&cur, sizes[l + 1].0, sizes[l + 1].1, pw, ph);
        let mut next = vec![0.0f32; pw * ph];
        next.par_chunks_mut(4096)
            .enumerate()
            .for_each(|(ci, chunk)| {
                let base = ci * 4096;
                for (i, o) in chunk.iter_mut().enumerate() {
                    let idx = base + i;
                    if dens[l][idx] > 1e-6 {
                        *o = nums[l][idx] / dens[l][idx];
                    } else {
                        *o = up[idx];
                    }
                }
            });
        cur = next;
    }
    Plane { w, h, d: cur }
}

/// Edge-aware (geodesic) fill of the depth field.
///
/// A smooth object in front of a textured backdrop has no focus information in
/// its interior at all — the only cue is its silhouette. Recovering the right
/// frame there therefore means propagating depth inward *from that silhouette*,
/// and never across it.
///
/// This does exactly that: pixels with real focus information are "known", and
/// every other pixel takes the value of the nearest known pixel measured along a
/// path whose cost rises with the image gradient. Crossing an object boundary is
/// expensive, so the fill stays inside the region it started in, however far it
/// has to travel — which a local filter cannot do, since its window is far
/// smaller than the object.
#[allow(dead_code)]
pub fn geodesic_fill(vals: &Plane, known: &Plane, guide: &Plane, edge_gain: f32) -> Plane {
    let (w, h) = (vals.w, vals.h);
    let n = w * h;

    // Cost of stepping onto a pixel, from the guide's gradient magnitude.
    let mut edge = vec![0.0f32; n];
    edge.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
        for x in 0..w {
            let (xi, yi) = (x as i64, y as i64);
            let gx = guide.sample_clamped(xi + 1, yi) - guide.sample_clamped(xi - 1, yi);
            let gy = guide.sample_clamped(xi, yi + 1) - guide.sample_clamped(xi, yi - 1);
            row[x] = (gx * gx + gy * gy).sqrt();
        }
    });

    let mut cost = vec![f32::INFINITY; n];
    let mut val = vec![0.0f32; n];
    for i in 0..n {
        if known.d[i] > 0.5 {
            cost[i] = 0.0;
            val[i] = vals.d[i];
        }
    }

    let step_cost = |edge: &[f32], a: usize, b: usize, len: f32| -> f32 {
        // A step is cheap inside a region and expensive across an edge.
        len * (1.0 + edge_gain * 0.5 * (edge[a] + edge[b]))
    };

    // Forward pass (top-left to bottom-right).
    for y in 0..h as i64 {
        for x in 0..w as i64 {
            let i = (y * w as i64 + x) as usize;
            let mut best = cost[i];
            let mut bv = val[i];
            for (dx, dy, len) in [(-1i64, 0i64, 1.0f32), (-1, -1, 1.41421), (0, -1, 1.0), (1, -1, 1.41421)] {
                let nx = x + dx;
                let ny = y + dy;
                if nx < 0 || ny < 0 || nx >= w as i64 || ny >= h as i64 {
                    continue;
                }
                let j = (ny * w as i64 + nx) as usize;
                if !cost[j].is_finite() {
                    continue;
                }
                let c = cost[j] + step_cost(&edge, j, i, len);
                if c < best {
                    best = c;
                    bv = val[j];
                }
            }
            cost[i] = best;
            val[i] = bv;
        }
    }
    // Backward pass (bottom-right to top-left).
    for y in (0..h as i64).rev() {
        for x in (0..w as i64).rev() {
            let i = (y * w as i64 + x) as usize;
            let mut best = cost[i];
            let mut bv = val[i];
            for (dx, dy, len) in [(1i64, 0i64, 1.0f32), (1, 1, 1.41421), (0, 1, 1.0), (-1, 1, 1.41421)] {
                let nx = x + dx;
                let ny = y + dy;
                if nx < 0 || ny < 0 || nx >= w as i64 || ny >= h as i64 {
                    continue;
                }
                let j = (ny * w as i64 + nx) as usize;
                if !cost[j].is_finite() {
                    continue;
                }
                let c = cost[j] + step_cost(&edge, j, i, len);
                if c < best {
                    best = c;
                    bv = val[j];
                }
            }
            cost[i] = best;
            val[i] = bv;
        }
    }
    // Pathological case: nothing was marked known.
    if val.iter().all(|v| *v == 0.0) {
        return vals.clone();
    }
    Plane { w, h, d: val }
}

/// Single channel guided filter (He, Sun & Tang 2010).
///
/// `guide` determines where the smoothing may cross; `src` is what gets
/// smoothed. Used here to snap the depth field onto the edges of the scene so
/// that a soft, continuous field still does not bleed across object boundaries.
pub fn guided_filter(src: &Plane, guide: &Plane, radius: usize, eps: f32) -> Plane {
    let (w, h) = (src.w, src.h);
    let n = w * h;
    let mut ii = vec![0.0f32; n];
    let mut ip = vec![0.0f32; n];
    ii.par_chunks_mut(4096).enumerate().for_each(|(ci, c)| {
        let base = ci * 4096;
        for (i, o) in c.iter_mut().enumerate() {
            let g = guide.d[base + i];
            *o = g * g;
        }
    });
    ip.par_chunks_mut(4096).enumerate().for_each(|(ci, c)| {
        let base = ci * 4096;
        for (i, o) in c.iter_mut().enumerate() {
            *o = guide.d[base + i] * src.d[base + i];
        }
    });
    let mean_i = box_blur(guide, radius);
    let mean_p = box_blur(src, radius);
    let corr_i = box_blur(&Plane { w, h, d: ii }, radius);
    let corr_ip = box_blur(&Plane { w, h, d: ip }, radius);

    let mut a = Plane::new(w, h);
    let mut b = Plane::new(w, h);
    a.d.par_chunks_mut(4096).enumerate().for_each(|(ci, c)| {
        let base = ci * 4096;
        for (i, o) in c.iter_mut().enumerate() {
            let idx = base + i;
            let mi = mean_i.d[idx];
            let var = (corr_i.d[idx] - mi * mi).max(0.0);
            let cov = corr_ip.d[idx] - mi * mean_p.d[idx];
            *o = cov / (var + eps);
        }
    });
    b.d.par_chunks_mut(4096).enumerate().for_each(|(ci, c)| {
        let base = ci * 4096;
        for (i, o) in c.iter_mut().enumerate() {
            let idx = base + i;
            *o = mean_p.d[idx] - a.d[idx] * mean_i.d[idx];
        }
    });
    let mean_a = box_blur(&a, radius);
    let mean_b = box_blur(&b, radius);
    let mut out = Plane::new(w, h);
    out.d.par_chunks_mut(4096).enumerate().for_each(|(ci, c)| {
        let base = ci * 4096;
        for (i, o) in c.iter_mut().enumerate() {
            let idx = base + i;
            *o = mean_a.d[idx] * guide.d[idx] + mean_b.d[idx];
        }
    });
    out
}

/// 3x3 median filter; removes isolated depth spikes cheaply.
#[allow(dead_code)]
pub fn median3(src: &Plane) -> Plane {
    median_n(src, 1)
}

/// Median filter with an arbitrary radius.
///
/// This is the main despeckling step for the raw depth field. A mean-based
/// filter (including the guided filter) spreads a boundary over its whole
/// window; a median does not, so the depth discontinuity between a foreground
/// object and the background stays sharp while the salt-and-pepper flips inside
/// each region are removed.
pub fn median_n(src: &Plane, r: usize) -> Plane {
    if r == 0 {
        return src.clone();
    }
    let (w, h) = (src.w, src.h);
    let n = (2 * r + 1) * (2 * r + 1);
    let mut out = vec![0.0f32; w * h];
    let rows: Vec<(usize, Vec<f32>)> = (0..h)
        .into_par_iter()
        .map(|y| {
            let mut row = vec![0.0f32; w];
            let mut buf = vec![0.0f32; n];
            for x in 0..w {
                let mut k = 0;
                for dy in -(r as i64)..=(r as i64) {
                    for dx in -(r as i64)..=(r as i64) {
                        buf[k] = src.sample_clamped(x as i64 + dx, y as i64 + dy);
                        k += 1;
                    }
                }
                buf.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                row[x] = buf[n / 2];
            }
            (y, row)
        })
        .collect();
    for (y, row) in rows {
        out[y * w..y * w + w].copy_from_slice(&row);
    }
    Plane { w, h, d: out }
}

/// Mark the pixels where two surfaces at different distances meet.
///
/// This is the one reliable way to find a silhouette that does not depend on how
/// the surfaces happen to look. The image gradient cannot do it: on the reference
/// stack the gradient across the leg/backdrop silhouette peaks at 5.0 per pixel,
/// no larger than the backdrop's own relief (4.0) or the object's surface (7.0),
/// because those two are similar in tone. The *depth* contrast is unambiguous
/// instead — about 29 frames across that silhouette, against about 6 frames
/// across the whole of the object's own curvature.
///
/// The min and max are taken on a subsampled copy: at `sub` 4 each output pixel
/// costs `(2*radius+1)^2 / 16` samples instead of a full local histogram, which
/// is what makes a window of a few tens of pixels affordable.
pub fn discontinuity_mask(d: &Plane, sub: usize, radius: usize, threshold: f32) -> Plane {
    let sub = sub.max(1);
    let (sw, sh) = ((d.w + sub - 1) / sub, (d.h + sub - 1) / sub);
    let mut small = vec![0.0f32; sw * sh];
    for sy in 0..sh {
        for sx in 0..sw {
            small[sy * sw + sx] = d.d[(sy * sub).min(d.h - 1) * d.w + (sx * sub).min(d.w - 1)];
        }
    }
    let mut lo = vec![0.0f32; sw * sh];
    let mut hi = vec![0.0f32; sw * sh];
    lo.par_chunks_mut(sw)
        .zip(hi.par_chunks_mut(sw))
        .enumerate()
        .for_each(|(sy, (lr, hr))| {
            for sx in 0..sw {
                let (mut a, mut b) = (f32::INFINITY, f32::NEG_INFINITY);
                let y0 = sy.saturating_sub(radius);
                let y1 = (sy + radius + 1).min(sh);
                let x0 = sx.saturating_sub(radius);
                let x1 = (sx + radius + 1).min(sw);
                for yy in y0..y1 {
                    for xx in x0..x1 {
                        let v = small[yy * sw + xx];
                        a = a.min(v);
                        b = b.max(v);
                    }
                }
                lr[sx] = a;
                hr[sx] = b;
            }
        });
    let mut out = vec![0.0f32; d.w * d.h];
    out.par_chunks_mut(d.w).enumerate().for_each(|(y, row)| {
        let sy = (y / sub).min(sh - 1);
        for x in 0..d.w {
            let sx = (x / sub).min(sw - 1);
            row[x] = if hi[sy * sw + sx] - lo[sy * sw + sx] >= threshold { 1.0 } else { 0.0 };
        }
    });
    Plane { w: d.w, h: d.h, d: out }
}

/// Bilateral filter of a depth field, with the range term taken on the depth
/// itself.
///
/// Any linear smoother spreads a step over its window, and so does a guided
/// filter: measured on this stack the *raw* depth estimate is a 1–2 pixel step at
/// an object silhouette, while after `guided_filter` (radius 6, three passes) the
/// same edge ramps over 30. That ramp is the single cause of both artifacts seen
/// at a seam — inside it the background's texture is taken from a frame in which
/// the background is defocused (a flat grey band), and the object's defocused
/// bloom sits underneath its own sharp detail (a bright ring).
///
/// Weighting each neighbour by how similar its *depth* is keeps the average
/// working on ±2-frame speckle while leaving a 20-frame jump alone. The guide
/// image cannot do this job: on this stack the silhouette's luminance gradient is
/// no stronger than the backdrop's own relief, so the guide simply does not
/// announce where the jump is.
pub fn bilateral_depth(src: &Plane, radius: usize, sigma_r: f32) -> Plane {
    let (w, h) = (src.w, src.h);
    if radius == 0 || sigma_r <= 0.0 {
        return src.clone();
    }
    // Lookup table over |delta| / sigma, in 1/32 steps up to 8 sigma.
    const LUT_N: usize = 256;
    let inv = 32.0f32 / sigma_r;
    let lut: Vec<f32> = (0..LUT_N)
        .map(|i| {
            let t = i as f32 / 32.0;
            (-0.5 * t * t).exp()
        })
        .collect();
    let mut out = vec![0.0f32; w * h];
    out.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
        for x in 0..w {
            let c = src.d[y * w + x];
            let mut acc = 0.0f32;
            let mut wsum = 0.0f32;
            let y0 = (y as i64 - radius as i64).max(0) as usize;
            let y1 = (y + radius + 1).min(h);
            let x0 = (x as i64 - radius as i64).max(0) as usize;
            let x1 = (x + radius + 1).min(w);
            for yy in y0..y1 {
                for xx in x0..x1 {
                    let v = src.d[yy * w + xx];
                    let k = ((v - c).abs() * inv) as usize;
                    let wgt = lut[k.min(LUT_N - 1)];
                    acc += wgt * v;
                    wsum += wgt;
                }
            }
            row[x] = if wsum > 0.0 { acc / wsum } else { c };
        }
    });
    Plane { w, h, d: out }
}

/// Per-pixel search window around the coarse prior.
///
/// The prior is measured at 1/`coarse_scale`, so a coarse cell that straddles a
/// surface boundary averages the responses of *both* surfaces and its depth is
/// pulled towards the stronger one. Upsampled, that puts the prior's step tens of
/// pixels inside the far surface, and a *fixed* window then locks the fine pass
/// inside that error, so the far surface never gets the chance to recover. That
/// is what leaves a smooth band along the far side of a silhouette: measured on
/// the reference stack the backdrop beside the figure was pinned at frame 13.1
/// against a true 34, and lifting the confinement moved it to 34.4.
///
/// Widening the window in proportion to the prior's own local depth range lets
/// the fine pass escape exactly where the prior is uncertain, while keeping the
/// tight window everywhere the prior is solid.
pub fn prior_window(prior: &Plane, base: f32, gain: f32, radius: usize) -> Plane {
    let (w, h) = (prior.w, prior.h);
    let mut out = vec![base; w * h];
    let r = radius as i64;
    out.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
        for x in 0..w {
            let c = prior.d[y * w + x];
            let (mut lo, mut hi) = (c, c);
            for (dx, dy) in [(-r, 0i64), (r, 0), (0, -r), (0, r)] {
                let v = prior.sample_clamped(x as i64 + dx, y as i64 + dy);
                lo = lo.min(v);
                hi = hi.max(v);
            }
            row[x] = base + gain * (hi - lo);
        }
    });
    Plane { w, h, d: out }
}

/// Where the depth field spans a large range within a small window, i.e. where
pub fn downsample_depth(src: &Plane) -> Plane {
    let (d, w, h) = downsample_plane(&src.d, src.w, src.h);
    Plane { w, h, d }
}

/// Upsample a coarse depth field with bilinear interpolation.
pub fn upsample_depth(src: &Plane, tw: usize, th: usize) -> Plane {
    Plane {
        w: tw,
        h: th,
        d: upsample_plane_bilinear(&src.d, src.w, src.h, tw, th),
    }
}
