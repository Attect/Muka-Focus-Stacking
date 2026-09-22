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
///
/// `pdx` / `pdy` carry the *depth-dependent* part of the shift; see [`Parallax`].
/// Refocusing moves the entrance pupil and changes magnification, so the image of
/// a scene point moves by an amount that depends on how far away it is. One
/// similarity transform cannot register a near foreground and the background
/// behind it at the same time, and the part it gets wrong is exactly the part the
/// fusion then mixes into a ghost.
#[derive(Clone, Copy, Debug)]
pub struct Transform {
    pub scale: f32,
    pub dx: f32,
    pub dy: f32,
    pub pdx: f32,
    pub pdy: f32,
}

impl Default for Transform {
    fn default() -> Self {
        Self { scale: 1.0, dx: 0.0, dy: 0.0, pdx: 0.0, pdy: 0.0 }
    }
}

impl Transform {
    /// Map a destination pixel back to a source pixel (both relative to the
    /// canvas centre).
    #[inline]
    pub fn inverse_map(&self, x: f32, y: f32) -> (f32, f32) {
        ((x - self.dx) / self.scale, (y - self.dy) / self.scale)
    }

    /// Depth-dependent shift at nearness `u` (0 = farthest, 1 = nearest).
    #[inline]
    pub fn parallax_at(&self, u: f32) -> (f32, f32) {
        (self.pdx * u, self.pdy * u)
    }
}

/// Depth-dependent residual shift: the parallax a focus sweep drags along.
///
/// On the reference stack block matching finds that the feet need about ten
/// pixels more shift than the face between the ends of the sweep, with the
/// residual changing sign in between. Whatever is left over after the global
/// similarity fit is therefore not noise: it is a systematic function of
/// distance, and it is worst where the subject comes closest to the lens, which
/// is also where the sharp frames of the sweep sit.
///
/// The residual is modelled as a shift proportional to a nearness parameter `u`
/// (1 at `near`, 0 at `far`), sampled from the depth field so that the
/// correction follows the scene. `d` holds that field at `scale` canvas pixels
/// per sample, which keeps the extra cost at one bilinear lookup per pixel.
#[derive(Clone)]
pub struct Parallax {
    /// Nearness per sample, already dilated towards the foreground and smoothed.
    pub u: Vec<f32>,
    pub w: usize,
    pub h: usize,
    pub scale: f32,
}

impl Parallax {
    /// Build the nearness field from a depth field (frame numbers).
    ///
    /// Two details matter here and both were learned the hard way. First the
    /// nearness is **dilated towards the foreground** before it is used: at an
    /// occlusion boundary the depth field legitimately jumps from the near
    /// surface to whatever is behind it, but the pixels *on* the boundary carry
    /// the near surface's content, and giving them the background's shift tears
    /// the edge apart — measured as a serrated sawtooth along the base rim, 6
    /// levels of difference against the previous version. Taking the nearest
    /// depth within a small radius is what a visible-surface map means.
    ///
    /// Then it is smoothed, because the shift has to vary at most sub-pixel per
    /// pixel or the resampling turns the residual steps of the depth field into
    /// visible geometry.
    pub fn new(depth: &[f32], w: usize, h: usize, scale: f32, near: f32, far: f32, dilate: usize) -> Self {
        let span = (far - near).max(1e-6);
        let mut u: Vec<f32> = depth
            .iter()
            .map(|d| ((far - d) / span).clamp(0.0, 1.0))
            .collect();
        if dilate > 0 {
            // Separable max filter: foreground wins.
            let mut tmp = vec![0.0f32; w * h];
            for y in 0..h {
                for x in 0..w {
                    let x0 = x.saturating_sub(dilate);
                    let x1 = (x + dilate).min(w - 1);
                    let mut m = 0.0f32;
                    for k in x0..=x1 {
                        let v = u[y * w + k];
                        if v > m {
                            m = v;
                        }
                    }
                    tmp[y * w + x] = m;
                }
            }
            for y in 0..h {
                let y0 = y.saturating_sub(dilate);
                let y1 = (y + dilate).min(h - 1);
                for x in 0..w {
                    let mut m = 0.0f32;
                    for k in y0..=y1 {
                        let v = tmp[k * w + x];
                        if v > m {
                            m = v;
                        }
                    }
                    u[y * w + x] = m;
                }
            }
        }
        let plane = Plane { w, h, d: u };
        let plane = box_blur(&plane, 1);
        Parallax { u: plane.d, w, h, scale }
    }

    /// Nearness at a canvas position, bilinear.
    #[inline]
    pub fn u_at(&self, canvas_x: f32, canvas_y: f32) -> f32 {
        let x = canvas_x / self.scale - 0.5;
        let y = canvas_y / self.scale - 0.5;
        let x0 = x.floor();
        let y0 = y.floor();
        let fx = x - x0;
        let fy = y - y0;
        let (w, h) = (self.w as i64, self.h as i64);
        let g = |xi: f32, yi: f32| -> f32 {
            let xi = (xi as i64).clamp(0, w - 1) as usize;
            let yi = (yi as i64).clamp(0, h - 1) as usize;
            self.u[yi * self.w + xi]
        };
        g(x0, y0) * (1.0 - fx) * (1.0 - fy)
            + g(x0 + 1.0, y0) * fx * (1.0 - fy)
            + g(x0, y0 + 1.0) * (1.0 - fx) * fy
            + g(x0 + 1.0, y0 + 1.0) * fx * fy
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
    warp_region_par(src, t, tw, th, off_x, off_y, fill, None)
}

/// [`warp_region`] with the depth-dependent shift applied. `par` is sampled in
/// canvas coordinates, which is what the caller passes as `off_x` / `off_y`.
pub fn warp_region_par(
    src: &Plane,
    t: &Transform,
    tw: usize,
    th: usize,
    off_x: f32,
    off_y: f32,
    fill: f32,
    par: Option<&Parallax>,
) -> Plane {
    let cx = src.w as f32 * 0.5;
    let cy = src.h as f32 * 0.5;
    let mut out = vec![fill; tw * th];
    out.par_chunks_mut(tw).enumerate().for_each(|(y, row)| {
        for x in 0..tw {
            let (sx, sy) = (x as f32 + 0.5 + off_x, y as f32 + 0.5 + off_y);
            // Sample where the scene point that lands on this canvas pixel
            // actually is in this frame: the global transform plus the
            // distance-dependent residual.
            let (px, py) = match par {
                Some(p) => t.parallax_at(p.u_at(sx, sy)),
                None => (0.0, 0.0),
            };
            let (rx, ry) = t.inverse_map(sx - cx + px, sy - cy + py);
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
#[allow(dead_code)]
pub fn warp_grid_region(
    src: &Grid,
    t: &Transform,
    tw: usize,
    th: usize,
    off_x: f32,
    off_y: f32,
    fill: f32,
) -> Grid {
    warp_grid_region_par(src, t, tw, th, off_x, off_y, fill, None)
}

/// [`warp_grid_region`] with the depth-dependent shift applied.
pub fn warp_grid_region_par(
    src: &Grid,
    t: &Transform,
    tw: usize,
    th: usize,
    off_x: f32,
    off_y: f32,
    fill: f32,
    par: Option<&Parallax>,
) -> Grid {
    let c = src.c;
    let cx = src.w as f32 * 0.5;
    let cy = src.h as f32 * 0.5;
    let mut out = vec![fill; tw * th * c];
    out.par_chunks_mut(tw * c).enumerate().for_each(|(y, row)| {
        for x in 0..tw {
            let (sx, sy) = (x as f32 + 0.5 + off_x, y as f32 + 0.5 + off_y);
            let (px, py) = match par {
                Some(p) => t.parallax_at(p.u_at(sx, sy)),
                None => (0.0, 0.0),
            };
            let (rx, ry) = t.inverse_map(sx - cx + px, sy - cy + py);
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

/// Options for the depth-dependent residual estimation.
#[derive(Clone, Copy, Debug)]
pub struct ParallaxOptions {
    /// Half-range of the shift search, in analysis-scale pixels.
    pub search: usize,
    /// Blocks per axis used to sample the residual field.
    pub blocks: usize,
}

impl Default for ParallaxOptions {
    fn default() -> Self {
        Self { search: 24, blocks: 10 }
    }
}

/// Residual shift as a function of nearness, measured by block matching.
#[derive(Clone, Copy, Debug, Default)]
pub struct ParallaxFit {
    /// Shift the model asks for at nearness 0 (the far end).
    pub dx_far: f32,
    pub dy_far: f32,
    /// Shift the model asks for at nearness 1 (the near end).
    pub dx_near: f32,
    pub dy_near: f32,
    /// 0..1: mean correlation of the blocks the fit rests on, times how many of
    /// the sampled blocks survived.
    pub quality: f32,
    pub blocks: usize,
    /// Spread of the nearness values the fit rests on. A small spread means the
    /// slope is extrapolation, and the caller should not trust it.
    #[allow(dead_code)]
    pub u_spread: f32,
}

/// Normalised cross-correlation of two equal-length patches.
#[inline]
fn ncc(pa: &[f32], pb: &[f32]) -> f32 {
    let n = pa.len() as f32;
    let ma = pa.iter().sum::<f32>() / n;
    let mb = pb.iter().sum::<f32>() / n;
    let mut num = 0.0f32;
    let mut va = 0.0f32;
    let mut vb = 0.0f32;
    for i in 0..pa.len() {
        let a = pa[i] - ma;
        let b = pb[i] - mb;
        num += a * b;
        va += a * a;
        vb += b * b;
    }
    num / (va * vb).sqrt().max(1e-9)
}

/// Copy a patch, subsampled by `step`.
fn patch(src: &Plane, x0: usize, y0: usize, bw: usize, bh: usize, step: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity((bw / step + 1) * (bh / step + 1));
    let mut y = 0;
    while y < bh {
        let mut x = 0;
        while x < bw {
            out.push(src.d[(y0 + y) * src.w + x0 + x]);
            x += step;
        }
        y += step;
    }
    out
}

/// Mean gradient magnitude of a patch: blocks with nothing to register on are
/// dropped rather than allowed to vote.
fn patch_texture(src: &Plane, x0: usize, y0: usize, bw: usize, bh: usize, step: usize) -> f32 {
    let mut acc = 0.0f32;
    let mut n = 0.0f32;
    let mut y = 1;
    while y + 1 < bh {
        let mut x = 1;
        while x + 1 < bw {
            let i = (y0 + y) * src.w + x0 + x;
            let gx = src.d[i + 1] - src.d[i - 1];
            let gy = src.d[i + src.w] - src.d[i - src.w];
            acc += (gx * gx + gy * gy).sqrt();
            n += 1.0;
            x += step;
        }
        y += step;
    }
    if n > 0.0 {
        acc / n
    } else {
        0.0
    }
}

/// The shift of `b` that best matches the patch of `a` at (x0, y0).
///
/// Hierarchical: quarter scale carries the bulk of the search range, half scale
/// and then full scale settle the last pixels. Returns the correlation of the
/// winning position so the caller can weight or reject the sample.
fn block_shift(
    a: &Plane,
    b: &Plane,
    a2: &Plane,
    b2: &Plane,
    a4: &Plane,
    b4: &Plane,
    x0: usize,
    y0: usize,
    bw: usize,
    bh: usize,
    rng: usize,
) -> (i32, i32, f32) {
    let r4 = ((rng + 3) / 4) as i32;
    let w4 = (bw / 4).max(4);
    let h4 = (bh / 4).max(4);
    let pa4 = patch(a4, x0 / 4, y0 / 4, w4, h4, 2);
    let mut best = (-9.0f32, 0i32, 0i32);
    for dy in -r4..=r4 {
        for dx in -r4..=r4 {
            let xx = x0 as i32 / 4 + dx;
            let yy = y0 as i32 / 4 + dy;
            if xx < 0 || yy < 0 || xx as usize + w4 >= a4.w || yy as usize + h4 >= a4.h {
                continue;
            }
            let c = ncc(&pa4, &patch(b4, xx as usize, yy as usize, w4, h4, 2));
            if c > best.0 {
                best = (c, dx, dy);
            }
        }
    }
    best.1 *= 4;
    best.2 *= 4;
    for step in [2usize, 1usize] {
        let (ac, bc) = if step == 2 { (a2, b2) } else { (a, b) };
        let w = (bw / step).max(4);
        let h = (bh / step).max(4);
        let bx = x0 as i32 / step as i32 + best.1 / step as i32;
        let by = y0 as i32 / step as i32 + best.2 / step as i32;
        let pa = patch(ac, bx as usize, by as usize, w, h, 1);
        let r = if step == 2 { 3 } else { 2 };
        let mut bst = (best.0, bx, by);
        for dy in -r..=r {
            for dx in -r..=r {
                let xx = bx + dx;
                let yy = by + dy;
                if xx < 0 || yy < 0 || xx as usize + w >= ac.w || yy as usize + h >= ac.h {
                    continue;
                }
                let c = ncc(&pa, &patch(bc, xx as usize, yy as usize, w, h, 1));
                if c > bst.0 {
                    bst = (c, xx, yy);
                }
            }
        }
        best.0 = bst.0;
        best.1 = (bst.1 - x0 as i32 / step as i32) * step as i32;
        best.2 = (bst.2 - y0 as i32 / step as i32) * step as i32;
    }
    (best.1, best.2, best.0)
}

/// Weighted least squares of value against nearness, with correlation weights.
/// Returns (intercept_x, slope_x, intercept_y, slope_y).
fn weighted_line(pts: &[(f32, f32, f32, f32)]) -> (f32, f32, f32, f32) {
    let mut sw = 0.0f32;
    let mut su = 0.0f32;
    for p in pts {
        let w = p.3.max(0.0).powi(2);
        sw += w;
        su += w * p.0;
    }
    let mu = su / sw.max(1e-9);
    let mut svv = 0.0f32;
    let mut svx = 0.0f32;
    let mut svy = 0.0f32;
    let mut sx = 0.0f32;
    let mut sy = 0.0f32;
    for p in pts {
        let w = p.3.max(0.0).powi(2);
        let v = p.0 - mu;
        svv += w * v * v;
        svx += w * v * p.1;
        svy += w * v * p.2;
        sx += w * p.1;
        sy += w * p.2;
    }
    let svv = svv.max(1e-9);
    let slope_x = svx / svv;
    let slope_y = svy / svv;
    let mean_x = sx / sw.max(1e-9);
    let mean_y = sy / sw.max(1e-9);
    (mean_x - slope_x * mu, slope_x, mean_y - slope_y * mu, slope_y)
}

/// How far the frame's image moves between the far and the near end of the
/// stack, as a function of nearness `u`.
///
/// `reference` and `frame` must both already be warped onto the canvas with the
/// global transform, and `depth` must be the canvas depth field at the same
/// scale. Both images are low-passed first: what differs between them is defocus,
/// and a blurred edge is still an edge in the same place, so blurring costs the
/// registration nothing and removes the temptation to match focus instead of
/// position.
pub fn estimate_parallax(
    reference: &Plane,
    frame: &Plane,
    depth: &Plane,
    near: f32,
    far: f32,
    opts: &ParallaxOptions,
) -> ParallaxFit {
    const BLK: usize = 96;
    let (w, h) = (reference.w.min(frame.w), reference.h.min(frame.h));
    let a = box_blur(reference, 2);
    let b = box_blur(frame, 2);
    let (a2, b2) = (resize(&a, w / 2, h / 2), resize(&b, w / 2, h / 2));
    let (a4, b4) = (resize(&a2, w / 4, h / 4), resize(&b2, w / 4, h / 4));
    let nb = opts.blocks.max(2);
    let step = (h / nb).max(BLK);

    let blk = BLK.min(w / 2).min(h / 2);
    if blk < 16 {
        return ParallaxFit::default();
    }
    let mut samples: Vec<(f32, f32, f32, f32, f32)> = Vec::new();
    let mut y0 = step / 2;
    while y0 + blk < h {
        let mut x0 = step / 2;
        while x0 + blk < w {
            // The block has to be dominated by one surface, otherwise its
            // nearness does not describe the content it matched.
            let mut ds: Vec<f32> = Vec::new();
            let mut y = 0;
            while y < blk {
                let mut x = 0;
                while x < blk {
                    let yy = (y0 + y).min(depth.h - 1);
                    let xx = (x0 + x).min(depth.w - 1);
                    ds.push(depth.d[yy * depth.w + xx]);
                    x += 8;
                }
                y += 8;
            }
            ds.sort_by(|p, q| p.partial_cmp(q).unwrap_or(std::cmp::Ordering::Equal));
            let med = ds[ds.len() / 2];
            let spread = ds[ds.len() * 9 / 10] - ds[ds.len() / 10];
            let tex = patch_texture(&a, x0, y0, blk, blk, 2);
            if tex > 0.0 && spread < 8.0 {
                let (dx, dy, c) =
                    block_shift(&a, &b, &a2, &b2, &a4, &b4, x0, y0, blk, blk, opts.search);
                if c > 0.25 {
                    let u = if far > near {
                        ((far - med) / (far - near)).clamp(0.0, 1.0)
                    } else {
                        0.0
                    };
                    samples.push((u, dx as f32, dy as f32, c, tex));
                }
            }
            x0 += step;
        }
        y0 += step;
    }
    if samples.len() < 6 {
        return ParallaxFit::default();
    }
    // Keep the more textured half: a block sitting on flat paint matches
    // anything and would drag the line.
    let mut texs: Vec<f32> = samples.iter().map(|s| s.4).collect();
    texs.sort_by(|p, q| p.partial_cmp(q).unwrap_or(std::cmp::Ordering::Equal));
    let tex_min = texs[texs.len() / 2];
    let mut pts: Vec<(f32, f32, f32, f32)> = samples
        .iter()
        .filter(|s| s.4 >= tex_min)
        .map(|s| (s.0, s.1, s.2, s.3))
        .collect();
    if pts.len() < 6 {
        return ParallaxFit::default();
    }
    let mut fit = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
    for round in 0..3 {
        fit = weighted_line(&pts);
        if round == 2 {
            break;
        }
        let before = pts.len();
        pts.retain(|p| {
            let rx = p.1 - (fit.0 + fit.1 * p.0);
            let ry = p.2 - (fit.2 + fit.3 * p.0);
            (rx * rx + ry * ry).sqrt() < 2.0
        });
        if pts.len() < 6 {
            return ParallaxFit::default();
        }
        if pts.len() == before {
            break;
        }
    }
    let mut us: Vec<f32> = pts.iter().map(|p| p.0).collect();
    us.sort_by(|p, q| p.partial_cmp(q).unwrap_or(std::cmp::Ordering::Equal));
    let u_spread = us[us.len() - 1] - us[0];
    if u_spread < 0.25 {
        return ParallaxFit::default();
    }
    let mean_c = pts.iter().map(|p| p.3).sum::<f32>() / pts.len() as f32;
    ParallaxFit {
        dx_far: fit.0,
        dy_far: fit.2,
        dx_near: fit.0 + fit.1,
        dy_near: fit.2 + fit.3,
        quality: mean_c * pts.len() as f32 / (pts.len() + 4) as f32,
        blocks: pts.len(),
        u_spread,
    }
}


/// Smooth the per-frame fits across the stack.
///
/// The parallax is a property of the lens, so as the focus setting walks from
/// one end of the sweep to the other it has to vary *smoothly*. Fitting each
/// frame on its own does not give that: neighbouring frames come out with
/// residuals that differ by more than the residual itself, and then correcting
/// each frame by its own noisy estimate makes the frames the fusion *mixes*
/// disagree more than before — measured as a sawtooth along the high-contrast
/// rim of the base, and as an increase of 1.9 in the high-frequency energy of
/// that patch, which is the opposite of what a registration fix should do.
///
/// So the individual measurements are treated as samples of a smooth curve: a
/// weighted quadratic in the frame index is fitted through all of them, and the
/// curve is evaluated per frame. Frames whose measurement failed simply do not
/// contribute. The curve is then shifted so that it is exactly zero at the
/// reference frame, which is what defines the canvas and therefore must not move.
pub fn smooth_fits(fits: &[ParallaxFit], reference: usize) -> Vec<ParallaxFit> {
    let n = fits.len();
    if n < 4 {
        return fits.to_vec();
    }
    let x = |k: usize| (k as f32 - reference as f32) / (n as f32 * 0.5);
    // Basis: 1, x, x^2. Weighted normal equations per output channel.
    let mut ata = [[0.0f64; 3]; 3];
    let mut atb = [[0.0f64; 4]; 3];
    for (k, f) in fits.iter().enumerate() {
        if f.blocks < 6 {
            continue;
        }
        let w = (f.quality as f64).max(1e-3);
        let v = [1.0f64, x(k) as f64, (x(k) * x(k)) as f64];
        for i in 0..3 {
            for j in 0..3 {
                ata[i][j] += w * v[i] * v[j];
            }
            atb[i][0] += w * v[i] * f.dx_far as f64;
            atb[i][1] += w * v[i] * f.dy_far as f64;
            atb[i][2] += w * v[i] * f.dx_near as f64;
            atb[i][3] += w * v[i] * f.dy_near as f64;
        }
    }
    let coef = solve3(&ata, &atb);
    let Some(coef) = coef else {
        return fits.to_vec();
    };
    let eval = |k: usize, ch: usize| -> f32 {
        let v = [1.0f64, x(k) as f64, (x(k) * x(k)) as f64];
        (v[0] * coef[0][ch] + v[1] * coef[1][ch] + v[2] * coef[2][ch]) as f32
    };
    let base = [eval(reference, 0), eval(reference, 1), eval(reference, 2), eval(reference, 3)];
    (0..n)
        .map(|k| {
            let f = fits[k];
            if f.blocks < 6 {
                // A frame that could not be measured gets the curve, which is
                // exactly what the curve is for.
                return ParallaxFit {
                    dx_far: eval(k, 0) - base[0],
                    dy_far: eval(k, 1) - base[1],
                    dx_near: eval(k, 2) - base[2],
                    dy_near: eval(k, 3) - base[3],
                    quality: 0.5,
                    blocks: 0,
                    u_spread: 0.0,
                };
            }
            ParallaxFit {
                dx_far: eval(k, 0) - base[0],
                dy_far: eval(k, 1) - base[1],
                dx_near: eval(k, 2) - base[2],
                dy_near: eval(k, 3) - base[3],
                ..f
            }
        })
        .collect()
}

/// Solve a 3x3 system with 4 right-hand sides by Gaussian elimination.
fn solve3(a: &[[f64; 3]; 3], b: &[[f64; 4]; 3]) -> Option<[[f64; 4]; 3]> {
    let mut m = [[0.0f64; 7]; 3];
    for i in 0..3 {
        for j in 0..3 {
            m[i][j] = a[i][j];
        }
        for j in 0..4 {
            m[i][3 + j] = b[i][j];
        }
    }
    for col in 0..3 {
        let mut piv = col;
        for r in col..3 {
            if m[r][col].abs() > m[piv][col].abs() {
                piv = r;
            }
        }
        if m[piv][col].abs() < 1e-12 {
            return None;
        }
        m.swap(col, piv);
        let d = m[col][col];
        for j in 0..7 {
            m[col][j] /= d;
        }
        for r in 0..3 {
            if r == col {
                continue;
            }
            let f = m[r][col];
            for j in 0..7 {
                m[r][j] -= f * m[col][j];
            }
        }
    }
    Some([[m[0][3], m[0][4], m[0][5], m[0][6]],
          [m[1][3], m[1][4], m[1][5], m[1][6]],
          [m[2][3], m[2][4], m[2][5], m[2][6]]])
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
                *best = (Transform { scale: s, dx: dx * kx, dy: dy * ky, ..Default::default() }, score);
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
