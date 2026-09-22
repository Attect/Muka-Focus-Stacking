//! Gaussian / Laplacian pyramid construction and collapse.
//!
//! The fusion stage works in a Laplacian pyramid domain so that low frequency
//! (colour / brightness) and high frequency (detail) information can be blended
//! with different spatial support. That is what removes the visible seams a
//! plain per-pixel "pick the sharpest" approach leaves behind.

use rayon::prelude::*;

/// 5-tap binomial kernel used for both reduction and expansion.
const K: [f32; 5] = [0.0625, 0.25, 0.375, 0.25, 0.0625];

/// Border reflection used by OpenCV (`BORDER_REFLECT_101`).
#[inline]
fn reflect101(i: i64, n: usize) -> usize {
    if n == 1 {
        return 0;
    }
    let n = n as i64;
    let mut i = i;
    loop {
        if i < 0 {
            i = -i;
        } else if i >= n {
            i = 2 * (n - 1) - i;
        } else {
            return i as usize;
        }
    }
}

/// Row-major image with `c` interleaved channels.
#[derive(Clone, Debug)]
pub struct Grid {
    pub w: usize,
    pub h: usize,
    pub c: usize,
    pub d: Vec<f32>,
}

impl Grid {
    pub fn new(w: usize, h: usize, c: usize) -> Self {
        Self { w, h, c, d: vec![0.0; w * h * c] }
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.w * self.h * self.c
    }

    /// Reduce by a factor of two (blur + decimate).
    pub fn pyr_down(&self) -> Grid {
        let (w, h, c) = (self.w, self.h, self.c);
        let w2 = (w + 1) / 2;
        let h2 = (h + 1) / 2;
        let mut out = vec![0.0f32; w2 * h2 * c];
        out.par_chunks_mut(w2 * c).enumerate().for_each(|(j, row)| {
            let sy_base = 2 * j as i64 - 2;
            for i in 0..w2 {
                let sx_base = 2 * i as i64 - 2;
                let mut acc = [0.0f32; 4];
                for ky in 0..5 {
                    let sy = reflect101(sy_base + ky as i64, h);
                    let wr = K[ky];
                    for kx in 0..5 {
                        let sx = reflect101(sx_base + kx as i64, w);
                        let wc = wr * K[kx];
                        let sp = (sy * w + sx) * c;
                        for ch in 0..c {
                            acc[ch] += wc * self.d[sp + ch];
                        }
                    }
                }
                for ch in 0..c {
                    row[i * c + ch] = acc[ch];
                }
            }
        });
        Grid { w: w2, h: h2, c, d: out }
    }

    /// Expand by a factor of two and crop to `(tw, th)`.
    pub fn pyr_up(&self, tw: usize, th: usize) -> Grid {
        let (w, h, c) = (self.w, self.h, self.c);
        let (w2, h2) = (2 * w, 2 * h);
        let mut buf = vec![0.0f32; w2 * h2 * c];
        // Zero-stuffing: even output samples take the source value.
        buf.par_chunks_mut(w2 * c).enumerate().for_each(|(yy, row)| {
            if yy % 2 == 0 {
                let sy = yy / 2;
                for xx in (0..w2).step_by(2) {
                    let sp = (sy * w + xx / 2) * c;
                    row[xx * c..xx * c + c].copy_from_slice(&self.d[sp..sp + c]);
                }
            }
        });
        // Convolve with 4 * K, then crop.
        let mut out = vec![0.0f32; tw * th * c];
        out.par_chunks_mut(tw * c).enumerate().for_each(|(y, row)| {
            let sy_base = y as i64 - 2;
            for x in 0..tw {
                let sx_base = x as i64 - 2;
                let mut acc = [0.0f32; 4];
                for ky in 0..5 {
                    let sy = reflect101(sy_base + ky as i64, h2);
                    let wr = 4.0 * K[ky];
                    for kx in 0..5 {
                        let sx = reflect101(sx_base + kx as i64, w2);
                        let wc = wr * K[kx];
                        let sp = (sy * w2 + sx) * c;
                        for ch in 0..c {
                            acc[ch] += wc * buf[sp + ch];
                        }
                    }
                }
                for ch in 0..c {
                    row[x * c + ch] = acc[ch];
                }
            }
        });
        Grid { w: tw, h: th, c, d: out }
    }
}

/// Sizes of each pyramid level for a `w x h` image with `levels` levels.
pub fn pyramid_sizes(w: usize, h: usize, levels: usize) -> Vec<(usize, usize)> {
    let mut v = Vec::with_capacity(levels);
    let (mut cw, mut ch) = (w, h);
    for _ in 0..levels {
        v.push((cw, ch));
        cw = (cw + 1) / 2;
        ch = (ch + 1) / 2;
    }
    v
}

/// How many levels are usable before the image becomes degenerate.
pub fn max_levels(w: usize, h: usize) -> usize {
    let mut n = 0;
    let (mut cw, mut ch) = (w, h);
    while cw > 8 && ch > 8 && n < 16 {
        cw = (cw + 1) / 2;
        ch = (ch + 1) / 2;
        n += 1;
    }
    n.max(1)
}

/// Full Gaussian pyramid. Level 0 is the input.
pub fn gaussian_pyramid(g: &Grid, levels: usize) -> Vec<Grid> {
    let mut v = Vec::with_capacity(levels);
    v.push(g.clone());
    for i in 1..levels {
        let prev = &v[i - 1];
        if prev.w <= 2 || prev.h <= 2 {
            break;
        }
        v.push(prev.pyr_down());
    }
    v
}

/// Add all Laplacian levels back together.
pub fn collapse(lp: &[Grid]) -> Grid {
    let n = lp.len();
    let mut acc = lp[n - 1].clone();
    for i in (0..n - 1).rev() {
        let up = acc.pyr_up(lp[i].w, lp[i].h);
        let mut d = vec![0.0f32; lp[i].len()];
        d.par_chunks_mut(4096)
            .enumerate()
            .for_each(|(ci, chunk)| {
                let base = ci * 4096;
                for (k, o) in chunk.iter_mut().enumerate() {
                    *o = lp[i].d[base + k] + up.d[base + k];
                }
            });
        acc = Grid { w: lp[i].w, h: lp[i].h, c: lp[i].c, d };
    }
    acc
}

/// Downsample a single channel `f32` buffer by two using the same kernel.
pub fn downsample_plane(src: &[f32], w: usize, h: usize) -> (Vec<f32>, usize, usize) {
    let g = Grid { w, h, c: 1, d: src.to_vec() };
    let out = g.pyr_down();
    (out.d, out.w, out.h)
}

/// Bilinear upsample of a single channel buffer to exactly `(tw, th)`.
pub fn upsample_plane_bilinear(
    src: &[f32],
    w: usize,
    h: usize,
    tw: usize,
    th: usize,
) -> Vec<f32> {
    let mut out = vec![0.0f32; tw * th];
    let sx_scale = w as f32 / tw as f32;
    let sy_scale = h as f32 / th as f32;
    out.par_chunks_mut(tw).enumerate().for_each(|(y, row)| {
        let sy = (y as f32 + 0.5) * sy_scale - 0.5;
        let y0 = sy.floor();
        let fy = sy - y0;
        let y0i = (y0 as i64).clamp(0, h as i64 - 1) as usize;
        let y1i = (y0 as i64 + 1).clamp(0, h as i64 - 1) as usize;
        for x in 0..tw {
            let sx = (x as f32 + 0.5) * sx_scale - 0.5;
            let x0 = sx.floor();
            let fx = sx - x0;
            let x0i = (x0 as i64).clamp(0, w as i64 - 1) as usize;
            let x1i = (x0 as i64 + 1).clamp(0, w as i64 - 1) as usize;
            let a = src[y0i * w + x0i] + (src[y0i * w + x1i] - src[y0i * w + x0i]) * fx;
            let b = src[y1i * w + x0i] + (src[y1i * w + x1i] - src[y1i * w + x0i]) * fx;
            row[x] = a + (b - a) * fy;
        }
    });
    out
}
