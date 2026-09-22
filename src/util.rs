//! Basic image containers and separable filtering helpers.
//!
//! Everything downstream works on `f32` planes normalised to `[0, 1]`.
//! Colour images are stored interleaved (RGBRGB...) because the fusion stage
//! touches all three channels of the same pixel together.

use rayon::prelude::*;

/// Single channel floating point image.
#[derive(Clone, Debug)]
pub struct Plane {
    pub w: usize,
    pub h: usize,
    pub d: Vec<f32>,
}

impl Plane {
    #[allow(dead_code)]
    pub fn new(w: usize, h: usize) -> Self {
        Self { w, h, d: vec![0.0; w * h] }
    }

    #[allow(dead_code)]
    pub fn filled(w: usize, h: usize, v: f32) -> Self {
        Self { w, h, d: vec![v; w * h] }
    }

    #[inline]
    #[allow(dead_code)]
    pub fn at(&self, x: usize, y: usize) -> f32 {
        self.d[y * self.w + x]
    }

    #[inline]
    #[allow(dead_code)]
    pub fn set(&mut self, x: usize, y: usize, v: f32) {
        self.d[y * self.w + x] = v;
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.w * self.h
    }

    #[allow(dead_code)]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Clamped sampler (edge extend).
    #[inline]
    pub fn sample_clamped(&self, x: i64, y: i64) -> f32 {
        let xi = x.clamp(0, self.w as i64 - 1) as usize;
        let yi = y.clamp(0, self.h as i64 - 1) as usize;
        self.d[yi * self.w + xi]
    }

    /// Bilinear sampler with edge clamping. Coordinates are in pixel units.
    #[inline]
    pub fn sample_bilinear(&self, x: f32, y: f32) -> f32 {
        let x0 = x.floor();
        let y0 = y.floor();
        let fx = x - x0;
        let fy = y - y0;
        let x0 = x0 as i64;
        let y0 = y0 as i64;
        let v00 = self.sample_clamped(x0, y0);
        let v10 = self.sample_clamped(x0 + 1, y0);
        let v01 = self.sample_clamped(x0, y0 + 1);
        let v11 = self.sample_clamped(x0 + 1, y0 + 1);
        let a = v00 + (v10 - v00) * fx;
        let b = v01 + (v11 - v01) * fx;
        a + (b - a) * fy
    }

    /// Catmull-Rom cubic sampler with edge clamping.
    #[inline]
    pub fn sample_cubic(&self, x: f32, y: f32) -> f32 {
        let xb = x.floor();
        let yb = y.floor();
        let wx = cubic_weights(x - xb);
        let wy = cubic_weights(y - yb);
        let xb = xb as i64;
        let yb = yb as i64;
        let mut acc = 0.0f32;
        for j in 0..4 {
            let yy = (yb - 1 + j as i64).clamp(0, self.h as i64 - 1) as usize;
            let mut row = 0.0f32;
            for i in 0..4 {
                let xx = (xb - 1 + i as i64).clamp(0, self.w as i64 - 1) as usize;
                row += wx[i] * self.d[yy * self.w + xx];
            }
            acc += wy[j] * row;
        }
        acc
    }

    #[allow(dead_code)]
    pub fn min_max(&self) -> (f32, f32) {
        let mut lo = f32::INFINITY;
        let mut hi = f32::NEG_INFINITY;
        for &v in &self.d {
            if v < lo {
                lo = v;
            }
            if v > hi {
                hi = v;
            }
        }
        (lo, hi)
    }
}

/// Interleaved RGB floating point image, values in `[0, 1]`.
#[derive(Clone, Debug)]
pub struct RgbImage {
    pub w: usize,
    pub h: usize,
    /// len = w * h * 3
    pub d: Vec<f32>,
}

impl RgbImage {
    #[allow(dead_code)]
    pub fn new(w: usize, h: usize) -> Self {
        Self { w, h, d: vec![0.0; w * h * 3] }
    }

    #[inline]
    pub fn len_pixels(&self) -> usize {
        self.w * self.h
    }

    /// Rec.709 luma on the (gamma encoded) values; good enough as a guide /
    /// focus-measure source because it is only used for relative comparisons.
    #[allow(dead_code)]
    pub fn to_luma(&self) -> Plane {
        let n = self.len_pixels();
        let mut out = vec![0.0f32; n];
        out.par_chunks_mut(4096)
            .enumerate()
            .for_each(|(ci, chunk)| {
                let base = ci * 4096 * 3;
                for (i, o) in chunk.iter_mut().enumerate() {
                    let p = base + i * 3;
                    *o = 0.2126 * self.d[p] + 0.7152 * self.d[p + 1] + 0.0722 * self.d[p + 2];
                }
            });
        Plane { w: self.w, h: self.h, d: out }
    }

    /// Grayscale plane produced by the first principal component of the RGB
    /// values, as in `focus-stack`. Using the dominant component preserves more
    /// detail than a fixed luma weighting, which helps the focus measure on
    /// strongly coloured subjects.
    /// Fixed Rec.601 luma, identical for every frame of a stack.
    ///
    /// A focus measure may only compare frames if the *only* thing that changes
    /// between them is focus. Any per-frame conversion breaks that: this stack
    /// drifts in exposure and, more subtly, in colour covariance as the detail
    /// content changes with focus, so a per-frame principal-component projection
    /// applies a slightly different gain to each frame — and the measure, which
    /// is sensitive to amplitude, then ranks frames partly on that gain. Keeping
    /// the weights fixed removes the whole class of error; `to_grayscale_pca`
    /// remains available via `--gray pca` for comparison.
    #[allow(dead_code)]
    pub fn to_grayscale_luma(&self) -> Plane {
        let n = self.len_pixels();
        let mut out = vec![0.0f32; n];
        out.par_chunks_mut(4096)
            .enumerate()
            .for_each(|(ci, chunk)| {
                let base = ci * 4096 * 3;
                for (i, o) in chunk.iter_mut().enumerate() {
                    let p = base + i * 3;
                    *o = 0.299 * self.d[p] + 0.587 * self.d[p + 1] + 0.114 * self.d[p + 2];
                }
            });
        Plane { w: self.w, h: self.h, d: out }
    }

    pub fn to_grayscale_pca(&self) -> Plane {
        let n = self.len_pixels();
        let mut mean = [0.0f64; 3];
        for i in 0..n {
            for c in 0..3 {
                mean[c] += self.d[i * 3 + c] as f64;
            }
        }
        for c in 0..3 {
            mean[c] /= n as f64;
        }
        let mut cov = [[0.0f64; 3]; 3];
        for i in 0..n {
            let v = [
                self.d[i * 3] as f64 - mean[0],
                self.d[i * 3 + 1] as f64 - mean[1],
                self.d[i * 3 + 2] as f64 - mean[2],
            ];
            for a in 0..3 {
                for b in a..3 {
                    cov[a][b] += v[a] * v[b];
                }
            }
        }
        for a in 0..3 {
            for b in 0..a {
                cov[a][b] = cov[b][a];
            }
        }
        // Power iteration on the 3x3 covariance: cheap and sufficient here.
        let mut v = [1.0f64, 1.0f64, 1.0f64];
        for _ in 0..64 {
            let mut nv = [0.0f64; 3];
            for a in 0..3 {
                for b in 0..3 {
                    nv[a] += cov[a][b] * v[b];
                }
            }
            let norm = (nv[0] * nv[0] + nv[1] * nv[1] + nv[2] * nv[2]).sqrt();
            if norm < 1e-12 {
                break;
            }
            for a in 0..3 {
                v[a] = nv[a] / norm;
            }
        }
        // Keep the sign such that the projection correlates positively with luma.
        if v[0] * 0.2126 + v[1] * 0.7152 + v[2] * 0.0722 < 0.0 {
            for a in 0..3 {
                v[a] = -v[a];
            }
        }
        let w = [v[0] as f32, v[1] as f32, v[2] as f32];
        let mut out = vec![0.0f32; n];
        out.par_chunks_mut(4096).enumerate().for_each(|(ci, chunk)| {
            let base = ci * 4096 * 3;
            for (i, o) in chunk.iter_mut().enumerate() {
                let p = base + i * 3;
                *o = w[0] * self.d[p] + w[1] * self.d[p + 1] + w[2] * self.d[p + 2];
            }
        });
        Plane { w: self.w, h: self.h, d: out }
    }

    /// Encode to 8 or 16 bit interleaved output buffer.
    pub fn to_u16(&self) -> Vec<u16> {
        let mut out = vec![0u16; self.d.len()];
        out.par_chunks_mut(4096 * 3)
            .enumerate()
            .for_each(|(ci, chunk)| {
                let base = ci * 4096 * 3;
                for (i, o) in chunk.iter_mut().enumerate() {
                    let v = self.d[base + i];
                    *o = (v.clamp(0.0, 1.0) * 65535.0 + 0.5) as u16;
                }
            });
        out
    }
}

/// Transpose a row-major `w x h` buffer into a row-major `h x w` one.
fn transpose(src: &[f32], w: usize, h: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; w * h];
    // Each destination row corresponds to one source column.
    out.par_chunks_mut(h).enumerate().for_each(|(x, dst)| {
        for (y, v) in dst.iter_mut().enumerate() {
            *v = src[y * w + x];
        }
    });
    out
}

/// One-dimensional box filter over the rows of a row-major `w x h` buffer.
fn box_rows(src: &[f32], w: usize, h: usize, r: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; w * h];
    let win = (2 * r + 1) as f64;
    out.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
        let base = y * w;
        let mut acc = 0.0f64;
        for k in 0..=r {
            acc += src[base + k.min(w - 1)] as f64;
        }
        acc += src[base] as f64 * r as f64; // replicate the left border
        for x in 0..w {
            row[x] = (acc / win) as f32;
            let a = (x as i64 - r as i64).clamp(0, w as i64 - 1) as usize;
            let b = (x as i64 + r as i64 + 1).clamp(0, w as i64 - 1) as usize;
            acc += src[base + b] as f64 - src[base + a] as f64;
        }
    });
    out
}

/// Catmull-Rom cubic interpolation weights for a fractional offset `f` in
/// `[0, 1)`, for the four source samples at `-1, 0, 1, 2`.
///
/// Used instead of bilinear when resampling frames: at the sub-pixel shifts the
/// alignment produces, bilinear behaves like a 2-tap box filter and costs a
/// surprising amount of high frequency detail.
#[inline]
pub fn cubic_weights(f: f32) -> [f32; 4] {
    let f2 = f * f;
    let f3 = f2 * f;
    [
        -0.5 * f3 + f2 - 0.5 * f,
        1.5 * f3 - 2.5 * f2 + 1.0,
        -1.5 * f3 + 2.0 * f2 + 0.5 * f,
        0.5 * f3 - 0.5 * f2,
    ]
}

/// Separable box blur with radius `r` (window = 2r+1), edge clamped.
pub fn box_blur(src: &Plane, r: usize) -> Plane {
    if r == 0 {
        return src.clone();
    }
    let (w, h) = (src.w, src.h);
    let tmp = box_rows(&src.d, w, h, r);
    let tr = transpose(&tmp, w, h);
    let tr2 = box_rows(&tr, h, w, r);
    let out = transpose(&tr2, h, w);
    Plane { w, h, d: out }
}

/// Median of a slice, used for robust statistics.
#[allow(dead_code)]
pub fn median(v: &mut [f32]) -> f32 {
    if v.is_empty() {
        return 0.0;
    }
    let n = v.len();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    if n % 2 == 1 {
        v[n / 2]
    } else {
        0.5 * (v[n / 2 - 1] + v[n / 2])
    }
}

/// Median absolute deviation.
#[allow(dead_code)]
pub fn mad(v: &[f32], med: f32) -> f32 {
    let mut dev: Vec<f32> = v.iter().map(|x| (x - med).abs()).collect();
    median(&mut dev)
}

/// Percentile of a slice (input is copied and sorted). `q` in `[0, 1]`.
#[allow(dead_code)]
pub fn percentile(v: &[f32], q: f32) -> f32 {
    if v.is_empty() {
        return 0.0;
    }
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let idx = (q.clamp(0.0, 1.0) * (s.len() - 1) as f32).round() as usize;
    s[idx]
}
