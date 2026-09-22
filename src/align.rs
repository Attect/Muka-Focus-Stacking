//! Frame alignment.
//!
//! Even on a solid tripod a focus-bracketed stack is not perfectly registered:
//! almost every lens changes magnification as it focuses ("focus breathing"),
//! which shows up as a slow scale drift across the stack. On the reference
//! stack used to develop this tool the outermost frames needed a 0.75% scale
//! correction — about 53 pixels at 7008 px wide, plenty to wreck a stack.
//!
//! The estimator searches a small range of candidate scales and, for each, uses
//! phase correlation to find the residual translation. The candidate whose
//! correlation is strongest wins. Working on the gradient magnitude of a
//! downsampled image keeps the cost low and makes the estimate insensitive to
//! the focus change itself.

use crate::pyramid::Grid;
use crate::util::{box_blur, Plane};
use rayon::prelude::*;
use rustfft::num_complex::Complex32;
use rustfft::FftPlanner;

/// Similarity transform applied to a frame: scale about the centre, then shift.
#[derive(Clone, Copy, Debug)]
pub struct Transform {
    pub scale: f32,
    pub dx: f32,
    pub dy: f32,
}

impl Default for Transform {
    fn default() -> Self {
        Self { scale: 1.0, dx: 0.0, dy: 0.0 }
    }
}

impl Transform {
    /// Map a destination pixel back to a source pixel (both relative to the
    /// canvas centre).
    #[inline]
    pub fn inverse_map(&self, x: f32, y: f32) -> (f32, f32) {
        ((x - self.dx) / self.scale, (y - self.dy) / self.scale)
    }
}

/// Bilinear resize of a plane to an exact target size.
pub fn resize(src: &Plane, tw: usize, th: usize) -> Plane {
    if src.w == tw && src.h == th {
        return src.clone();
    }
    let mut out = vec![0.0f32; tw * th];
    let sx_scale = src.w as f32 / tw as f32;
    let sy_scale = src.h as f32 / th as f32;
    out.par_chunks_mut(tw).enumerate().for_each(|(y, row)| {
        let sy = (y as f32 + 0.5) * sy_scale - 0.5;
        for x in 0..tw {
            let sx = (x as f32 + 0.5) * sx_scale - 0.5;
            row[x] = src.sample_bilinear(sx, sy);
        }
    });
    Plane { w: tw, h: th, d: out }
}

/// Resample about the centre with a similarity transform into a `tw x th`
/// canvas whose top-left corner sits at `(off_x, off_y)` of the original
/// canvas. `fill` is used outside the source.
///
/// `off_x`/`off_y` are what makes cropping-free alignment possible: the output
/// can be an arbitrary window of the aligned canvas without re-deriving the
/// transforms.
pub fn warp_region(
    src: &Plane,
    t: &Transform,
    tw: usize,
    th: usize,
    off_x: f32,
    off_y: f32,
    fill: f32,
) -> Plane {
    let cx = src.w as f32 * 0.5;
    let cy = src.h as f32 * 0.5;
    let mut out = vec![fill; tw * th];
    out.par_chunks_mut(tw).enumerate().for_each(|(y, row)| {
        for x in 0..tw {
            let (sx, sy) = (x as f32 + 0.5 + off_x, y as f32 + 0.5 + off_y);
            let (rx, ry) = t.inverse_map(sx - cx, sy - cy);
            // Continuous coordinate -> pixel index (pixel i is centred at i+0.5).
            let ix = rx + cx - 0.5;
            let iy = ry + cy - 0.5;
            if ix < 0.0 || iy < 0.0 || ix > src.w as f32 - 1.0 || iy > src.h as f32 - 1.0 {
                row[x] = fill;
            } else {
                row[x] = src.sample_cubic(ix, iy);
            }
        }
    });
    Plane { w: tw, h: th, d: out }
}

/// Same as [`warp_region`] but for an interleaved multi-channel grid.
pub fn warp_grid_region(
    src: &Grid,
    t: &Transform,
    tw: usize,
    th: usize,
    off_x: f32,
    off_y: f32,
    fill: f32,
) -> Grid {
    let c = src.c;
    let cx = src.w as f32 * 0.5;
    let cy = src.h as f32 * 0.5;
    let mut out = vec![fill; tw * th * c];
    out.par_chunks_mut(tw * c).enumerate().for_each(|(y, row)| {
        for x in 0..tw {
            let (sx, sy) = (x as f32 + 0.5 + off_x, y as f32 + 0.5 + off_y);
            let (rx, ry) = t.inverse_map(sx - cx, sy - cy);
            let ix = rx + cx - 0.5;
            let iy = ry + cy - 0.5;
            let dp = x * c;
            if ix < 0.0 || iy < 0.0 || ix > src.w as f32 - 1.0 || iy > src.h as f32 - 1.0 {
                for ch in 0..c {
                    row[dp + ch] = fill;
                }
            } else {
                // Catmull-Rom resampling of every channel together.
                let xb = ix.floor() as i64;
                let yb = iy.floor() as i64;
                let wx = crate::util::cubic_weights(ix - xb as f32);
                let wy = crate::util::cubic_weights(iy - yb as f32);
                let dp = x * c;
                let mut acc = [0.0f32; 4];
                for j in 0..4 {
                    let yy = (yb - 1 + j as i64).clamp(0, src.h as i64 - 1) as usize;
                    let wyj = wy[j];
                    let mut row_acc = [0.0f32; 4];
                    for i in 0..4 {
                        let xx = (xb - 1 + i as i64).clamp(0, src.w as i64 - 1) as usize;
                        let w = wx[i] * wyj;
                        let sp = (yy * src.w + xx) * c;
                        for ch in 0..c {
                            row_acc[ch] += w * src.d[sp + ch];
                        }
                    }
                    for ch in 0..c {
                        acc[ch] += row_acc[ch];
                    }
                }
                for ch in 0..c {
                    row[dp + ch] = acc[ch].clamp(0.0, 1.0);
                }
            }
        }
    });
    Grid { w: tw, h: th, c, d: out }
}

/// Convenience wrapper: full-canvas warp using [`warp_region`].
pub fn warp(src: &Plane, t: &Transform, tw: usize, th: usize, fill: f32) -> Plane {
    warp_region(src, t, tw, th, 0.0, 0.0, fill)
}

/// Centre crop to a `frac` of the original size.
fn centre_crop(src: &Plane, frac: f32) -> Plane {
    let cw = ((src.w as f32 * frac) as usize).max(8);
    let ch = ((src.h as f32 * frac) as usize).max(8);
    let x0 = (src.w - cw) / 2;
    let y0 = (src.h - ch) / 2;
    let mut d = vec![0.0f32; cw * ch];
    d.par_chunks_mut(cw).enumerate().for_each(|(y, row)| {
        let src_row = &src.d[(y0 + y) * src.w + x0..(y0 + y) * src.w + x0 + cw];
        row.copy_from_slice(src_row);
    });
    Plane { w: cw, h: ch, d }
}

/// Sobel gradient magnitude.
fn gradient_magnitude(g: &Plane) -> Plane {
    let (w, h) = (g.w, g.h);
    let mut out = vec![0.0f32; w * h];
    out.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
        for x in 0..w {
            let (xi, yi) = (x as i64, y as i64);
            let gx = -g.sample_clamped(xi - 1, yi - 1) - 2.0 * g.sample_clamped(xi - 1, yi)
                - g.sample_clamped(xi - 1, yi + 1)
                + g.sample_clamped(xi + 1, yi - 1)
                + 2.0 * g.sample_clamped(xi + 1, yi)
                + g.sample_clamped(xi + 1, yi + 1);
            let gy = -g.sample_clamped(xi - 1, yi - 1) - 2.0 * g.sample_clamped(xi, yi - 1)
                - g.sample_clamped(xi + 1, yi - 1)
                + g.sample_clamped(xi - 1, yi + 1)
                + 2.0 * g.sample_clamped(xi, yi + 1)
                + g.sample_clamped(xi + 1, yi + 1);
            row[x] = (gx * gx + gy * gy).sqrt();
        }
    });
    Plane { w, h, d: out }
}

/// Feature image used for registration: gradient magnitude with its local mean
/// removed, so the estimator keys on structure rather than the slow brightness
/// ramp of the background.
pub fn registration_features(g: &Plane, highpass: usize) -> Plane {
    let gm = gradient_magnitude(g);
    let low = box_blur(&gm, highpass);
    let mut out = gm;
    out.d.par_iter_mut()
        .zip(low.d.par_iter())
        .for_each(|(a, b)| *a -= *b);
    out
}

fn transpose_complex(src: &[Complex32], w: usize, h: usize) -> Vec<Complex32> {
    let mut out = vec![Complex32::new(0.0, 0.0); w * h];
    out.par_chunks_mut(h).enumerate().for_each(|(x, dst)| {
        for (y, v) in dst.iter_mut().enumerate() {
            *v = src[y * w + x];
        }
    });
    out
}

/// In-place 2-D FFT of a row-major `w x h` complex buffer.
fn fft2(buf: &mut Vec<Complex32>, w: usize, h: usize, planner: &mut FftPlanner<f32>, inverse: bool) {
    let fft_w = if inverse {
        planner.plan_fft_inverse(w)
    } else {
        planner.plan_fft_forward(w)
    };
    let fft_h = if inverse {
        planner.plan_fft_inverse(h)
    } else {
        planner.plan_fft_forward(h)
    };
    {
        let mut rows: Vec<&mut [Complex32]> = buf.chunks_mut(w).collect();
        rows.par_iter_mut().for_each(|r| fft_w.process(r));
    }
    let mut tr = transpose_complex(buf, w, h); // now h rows of length w
    {
        let mut rows: Vec<&mut [Complex32]> = tr.chunks_mut(h).collect();
        rows.par_iter_mut().for_each(|r| fft_h.process(r));
    }
    *buf = transpose_complex(&tr, h, w);
}

/// Phase correlation. Returns `(dx, dy)` — the shift that maps `b` onto `a` in
/// working-resolution pixels — plus a confidence ratio.
pub fn phase_correlate(a: &Plane, b: &Plane, planner: &mut FftPlanner<f32>) -> (f32, f32, f32) {
    let (w, h) = (a.w, a.h);
    debug_assert_eq!(a.w, b.w);
    debug_assert_eq!(a.h, b.h);
    let mean_a = a.d.iter().sum::<f32>() / a.len() as f32;
    let mean_b = b.d.iter().sum::<f32>() / b.len() as f32;
    let mut fa: Vec<Complex32> =
        a.d.par_iter().map(|v| Complex32::new(v - mean_a, 0.0)).collect();
    let mut fb: Vec<Complex32> =
        b.d.par_iter().map(|v| Complex32::new(v - mean_b, 0.0)).collect();
    fft2(&mut fa, w, h, planner, false);
    fft2(&mut fb, w, h, planner, false);

    let mut cross: Vec<Complex32> = fa
        .par_iter()
        .zip(fb.par_iter())
        .map(|(x, y)| {
            let c = x * y.conj();
            let m = c.norm();
            if m > 1e-12 {
                c / m
            } else {
                Complex32::new(0.0, 0.0)
            }
        })
        .collect();
    fft2(&mut cross, w, h, planner, true);

    let n = (w * h) as f32;
    let values: Vec<f32> = cross.iter().map(|c| c.re / n).collect();

    let mut peak = 0usize;
    for (i, v) in values.iter().enumerate().skip(1) {
        if *v > values[peak] {
            peak = i;
        }
    }
    let px = (peak % w) as i64;
    let py = (peak / w) as i64;
    let at = |x: i64, y: i64| -> f32 {
        let xx = x.rem_euclid(w as i64) as usize;
        let yy = y.rem_euclid(h as i64) as usize;
        values[yy * w + xx]
    };
    let sub = |m: f32, c: f32, p: f32| -> f32 {
        let d = m - 2.0 * c + p;
        if d.abs() > 1e-12 {
            (0.5 * (m - p) / d).clamp(-0.5, 0.5)
        } else {
            0.0
        }
    };
    let mut dx = px as f32 + sub(at(px - 1, py), at(px, py), at(px + 1, py));
    let mut dy = py as f32 + sub(at(px, py - 1), at(px, py), at(px, py + 1));
    if dx > w as f32 / 2.0 {
        dx -= w as f32;
    }
    if dy > h as f32 / 2.0 {
        dy -= h as f32;
    }
    let peak_val = at(px, py);
    let mean = values.iter().sum::<f32>() / n;
    let var = values.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / n;
    let ratio = if var > 1e-20 { (peak_val - mean) / var.sqrt() } else { 0.0 };
    (dx, dy, ratio)
}

/// Normalised cross correlation of `b` shifted by `(dx, dy)` against `a`.
fn ncc_shifted(a: &Plane, b: &Plane, dx: i64, dy: i64) -> f32 {
    let (w, h) = (a.w, a.h);
    let x0 = dx.max(0);
    let x1 = (w as i64 + dx.min(0)).min(w as i64);
    let y0 = dy.max(0);
    let y1 = (h as i64 + dy.min(0)).min(h as i64);
    if x1 - x0 < 16 || y1 - y0 < 16 {
        return f32::NEG_INFINITY;
    }
    let mut sa = 0.0f64;
    let mut sb = 0.0f64;
    let mut saa = 0.0f64;
    let mut sbb = 0.0f64;
    let mut sab = 0.0f64;
    let mut n = 0.0f64;
    for y in y0..y1 {
        for x in x0..x1 {
            let av = a.d[(y * w as i64 + x) as usize] as f64;
            let bv = b.d[((y - dy) * w as i64 + (x - dx)) as usize] as f64;
            sa += av;
            sb += bv;
            saa += av * av;
            sbb += bv * bv;
            sab += av * bv;
            n += 1.0;
        }
    }
    let cov = sab - sa * sb / n;
    let va = saa - sa * sa / n;
    let vb = sbb - sb * sb / n;
    let d = (va * vb).sqrt();
    if d <= 1e-12 {
        0.0
    } else {
        (cov / d) as f32
    }
}

pub struct AlignOptions {
    /// Long side of the working resolution used for the search.
    pub work_size: usize,
    /// Half-width of the coarse scale search around 1.0.
    pub scale_range: f32,
    pub scale_steps: usize,
    pub highpass: usize,
    /// Low-pass radius applied to the working-resolution images before the
    /// registration features are built.
    ///
    /// Registration has to key on geometry, but gradient magnitude is exactly
    /// what changes with focus. Left unblurred, the scale search then wins by
    /// making a frame *as blurred as the reference* rather than as
    /// geometrically aligned: on the reference stack the best correlation falls
    /// from 0.93 for frames next to the reference to 0.39 at the far end of the
    /// sweep — the signature of matching focus, not position — and those
    /// transforms displaced frames by up to 8 working pixels with no physical
    /// cause (the camera was on a tripod).
    ///
    /// Blurring past the largest focus difference removes the focus signal while
    /// leaving a 0.8% magnification, worth 3.4 px at the crop corner, plainly
    /// measurable. The same measurement then returns a smooth, monotonic
    /// 0.996 -> 1.004 with a correlation of 0.98-1.00.
    pub focus_blur: usize,
    /// Half-width of the refinement search around the coarse winner.
    pub fine_range: f32,
    pub fine_steps: usize,
}

impl Default for AlignOptions {
    fn default() -> Self {
        Self {
            work_size: 900,
            scale_range: 0.02,
            scale_steps: 16,
            highpass: 4,
            focus_blur: 3,
            fine_range: 0.0025,
            fine_steps: 10,
        }
    }
}

/// Shared context for aligning many frames against one reference, so the
/// reference-side work (resize, features) is done once.
pub struct Aligner {
    /// Full resolution size of the frames.
    pub full_w: usize,
    pub full_h: usize,
    ww: usize,
    wh: usize,
    ref_core: Plane,
    pub opts: AlignOptions,
}

impl Aligner {
    pub fn new(reference_full: &Plane, opts: AlignOptions) -> Self {
        let long = reference_full.w.max(reference_full.h) as f32;
        let f = (opts.work_size as f32 / long).min(1.0);
        let ww = ((reference_full.w as f32 * f).round() as usize).max(64);
        let wh = ((reference_full.h as f32 * f).round() as usize).max(64);
        let small = box_blur(&resize(reference_full, ww, wh), opts.focus_blur);
        let feats = registration_features(&small, opts.highpass);
        let ref_core = centre_crop(&feats, 0.96);
        Self {
            full_w: reference_full.w,
            full_h: reference_full.h,
            ww,
            wh,
            ref_core,
            opts,
        }
    }

    /// Score one candidate scale: warp, build features, and take the best
    /// correlation over the residual translation phase correlation reports.
    fn score_scale(&self, small: &Plane, s: f32, planner: &mut FftPlanner<f32>) -> (f32, f32, f32) {
        let warped =
            warp(small, &Transform { scale: s, ..Default::default() }, self.ww, self.wh, 0.0);
        let feat = registration_features(&warped, self.opts.highpass);
        let core = centre_crop(&feat, 0.96);
        let (dx, dy, _) = phase_correlate(&self.ref_core, &core, planner);
        let score = ncc_shifted(&self.ref_core, &core, dx.round() as i64, dy.round() as i64);
        (score, dx, dy)
    }

    /// Estimate the transform mapping `frame_full` onto the reference.
    ///
    /// A coarse sweep establishes roughly which scale applies, then a finer one
    /// around that winner resolves it: the whole breathing range on the reference
    /// stack is under 1%, so a single coarse grid cannot place a frame better
    /// than about a third of a percent, and the residual then shows up as a
    /// spurious translation.
    pub fn estimate(&self, frame_full: &Plane) -> (Transform, f32) {
        let small = box_blur(&resize(frame_full, self.ww, self.wh), self.opts.focus_blur);
        let mut planner = FftPlanner::<f32>::new();
        let mut best = (Transform::default(), f32::NEG_INFINITY);
        let kx = self.full_w as f32 / self.ww as f32;
        let ky = self.full_h as f32 / self.wh as f32;
        let mut consider = |s: f32, best: &mut (Transform, f32)| {
            let (score, dx, dy) = self.score_scale(&small, s, &mut planner);
            if score.is_finite() && score > best.1 {
                *best = (Transform { scale: s, dx: dx * kx, dy: dy * ky }, score);
            }
        };
        let steps = self.opts.scale_steps.max(1);
        for i in 0..=steps {
            let s = 1.0 - self.opts.scale_range
                + 2.0 * self.opts.scale_range * (i as f32 / steps as f32);
            consider(s, &mut best);
        }
        let coarse = best.0.scale;
        let fs = self.opts.fine_steps.max(1);
        for i in 0..=fs {
            let s = coarse - self.opts.fine_range
                + 2.0 * self.opts.fine_range * (i as f32 / fs as f32);
            consider(s, &mut best);
        }
        best
    }
}
