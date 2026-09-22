//! Focus / sharpness measures.
//!
//! A focus measure must be: (a) maximal when the region is in focus, (b)
//! monotonic as defocus increases, and (c) comparable across the whole stack —
//! the last point rules out any per-image normalisation, since the images would
//! then no longer be ranked on a common scale.

use crate::util::{box_blur, Plane};
use rayon::prelude::*;

#[derive(Copy, Clone, Debug, PartialEq, Eq, clap::ValueEnum)]
pub enum FocusMeasure {
    /// Sum of Modified Laplacian (Nayar & Nakagawa 1994). The classic choice for
    /// focus stacking: robust to illumination gradients and cheap to compute.
    Sml,
    /// Tenengrad: squared Sobel gradient magnitude, box-summed.
    Tenengrad,
    /// Local variance (standard deviation inside a window).
    Variance,
}

/// Step size used by the modified Laplacian.
const ML_STEP: i64 = 2;

fn modified_laplacian(g: &Plane) -> Plane {
    let (w, h) = (g.w, g.h);
    let mut out = vec![0.0f32; w * h];
    out.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
        for x in 0..w {
            let c = g.d[y * w + x];
            let l = g.sample_clamped(x as i64 - ML_STEP, y as i64);
            let r = g.sample_clamped(x as i64 + ML_STEP, y as i64);
            let u = g.sample_clamped(x as i64, y as i64 - ML_STEP);
            let d = g.sample_clamped(x as i64, y as i64 + ML_STEP);
            row[x] = (2.0 * c - l - r).abs() + (2.0 * c - u - d).abs();
        }
    });
    Plane { w, h, d: out }
}

fn tenengrad(g: &Plane) -> Plane {
    let (w, h) = (g.w, g.h);
    let mut out = vec![0.0f32; w * h];
    out.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
        for x in 0..w {
            let xi = x as i64;
            let yi = y as i64;
            let gx = -g.sample_clamped(xi - 1, yi - 1)
                - 2.0 * g.sample_clamped(xi - 1, yi)
                - g.sample_clamped(xi - 1, yi + 1)
                + g.sample_clamped(xi + 1, yi - 1)
                + 2.0 * g.sample_clamped(xi + 1, yi)
                + g.sample_clamped(xi + 1, yi + 1);
            let gy = -g.sample_clamped(xi - 1, yi - 1)
                - 2.0 * g.sample_clamped(xi, yi - 1)
                - g.sample_clamped(xi + 1, yi - 1)
                + g.sample_clamped(xi - 1, yi + 1)
                + 2.0 * g.sample_clamped(xi, yi + 1)
                + g.sample_clamped(xi + 1, yi + 1);
            row[x] = gx * gx + gy * gy;
        }
    });
    Plane { w, h, d: out }
}

/// Compute the focus measure map of one grayscale image.
///
/// `radius` is the window radius over which the per-pixel response is summed.
/// A larger window gives a more stable depth estimate at the cost of spatial
/// resolution; the guided filter in `depth.rs` restores the edge alignment.
pub fn focus_map(gray: &Plane, kind: FocusMeasure, radius: usize) -> Plane {
    let raw = match kind {
        FocusMeasure::Sml => modified_laplacian(gray),
        FocusMeasure::Tenengrad => tenengrad(gray),
        FocusMeasure::Variance => {
            // E[x^2] - E[x]^2 over the window, computed with two box blurs.
            let mean = box_blur(gray, radius);
            let mut sq = gray.clone();
            sq.d.par_iter_mut().for_each(|v| *v *= *v);
            let mean_sq = box_blur(&sq, radius);
            let mut v = mean_sq;
            v.d.par_iter_mut()
                .zip(mean.d.par_iter())
                .for_each(|(a, b)| *a = (*a - b * b).max(0.0));
            return v;
        }
    };
    box_blur(&raw, radius)
}
