//! Fusion: turn the aligned frames plus the depth field into one all-in-focus
//! image.
//!
//! Four strategies are provided.
//!
//! * `pyramid` (default) — the frames are decomposed into Laplacian pyramids and
//!   each spatial frequency band is blended between the two frames that bracket
//!   the local depth value. Blending in the pyramid domain means low frequencies
//!   (colour, brightness) transition over a wide region while high frequencies
//!   (detail) follow the depth edge, which is what avoids both seams and halos.
//! * `max` — per band, keep the coefficient of largest magnitude. This is the
//!   classic approach and preserves micro contrast well, but it amplifies noise
//!   and can produce halos.
//! * `pixel` — cross fade between the bracketing frames in the pixel domain.
//!   Cheap and soft; useful as a fast preview.
//! * `hard` — take a single frame per pixel. Fastest, useful for checking what
//!   the depth map decided.

use crate::align::{warp_grid_region_par, Parallax, Transform};
use crate::pyramid::{collapse, gaussian_pyramid, Grid};
use crate::util::{box_blur, Plane, RgbImage};
use rayon::prelude::*;

#[derive(Copy, Clone, Debug, PartialEq, Eq, clap::ValueEnum)]
pub enum FusionMode {
    /// Laplacian bands, blended between the two frames bracketing the local
    /// depth, with per-band arbitration by coefficient energy. Default.
    Bracket,
    /// Same bands, but the blend weight comes purely from the depth value.
    /// Smoothest result, visibly softer than `bracket`.
    Pyramid,
    /// Per band, keep the coefficient of largest magnitude across the whole
    /// stack. Sharpest, but band choices are unconstrained so halos and noise
    /// can appear.
    Max,
    /// Cross fade between the bracketing frames in the pixel domain.
    Pixel,
    /// Take a single frame per pixel.
    Hard,
}

/// A rectangle in the aligned canvas.
#[derive(Copy, Clone, Debug)]
pub struct Rect {
    pub x: usize,
    pub y: usize,
    pub w: usize,
    pub h: usize,
}

impl Rect {
    pub fn full(w: usize, h: usize) -> Self {
        Self { x: 0, y: 0, w, h }
    }
}

/// Largest rectangle of the canvas that every transformed frame still covers.
pub fn valid_rect(transforms: &[Transform], w: usize, h: usize, margin: usize) -> Rect {
    let (mut x0, mut x1) = (f32::NEG_INFINITY, f32::INFINITY);
    let (mut y0, mut y1) = (f32::NEG_INFINITY, f32::INFINITY);
    let (fw, fh) = (w as f32, h as f32);
    for t in transforms {
        let s = t.scale;
        let lo_x = t.dx - s * fw * 0.5 + fw * 0.5 - 0.5;
        let hi_x = t.dx + s * (fw * 0.5 - 1.0) + fw * 0.5 - 0.5;
        let lo_y = t.dy - s * fh * 0.5 + fh * 0.5 - 0.5;
        let hi_y = t.dy + s * (fh * 0.5 - 1.0) + fh * 0.5 - 0.5;
        x0 = x0.max(lo_x);
        x1 = x1.min(hi_x);
        y0 = y0.max(lo_y);
        y1 = y1.min(hi_y);
    }
    let m = margin as f32;
    let x0 = (x0 + m).ceil().max(0.0);
    let y0 = (y0 + m).ceil().max(0.0);
    let x1 = (x1 - m).floor().min(w as f32 - 1.0);
    let y1 = (y1 - m).floor().min(h as f32 - 1.0);
    let ww = ((x1 - x0 + 1.0).max(1.0)) as usize;
    let hh = ((y1 - y0 + 1.0).max(1.0)) as usize;
    let x0 = (x0 as usize).min(w.saturating_sub(1));
    let y0 = (y0 as usize).min(h.saturating_sub(1));
    Rect { x: x0, y: y0, w: ww.min(w - x0), h: hh.min(h - y0) }
}

/// Crop a plane to a rectangle.
pub fn crop_plane(src: &Plane, r: Rect) -> Plane {
    let mut d = vec![0.0f32; r.w * r.h];
    d.par_chunks_mut(r.w).enumerate().for_each(|(y, row)| {
        let sy = r.y + y;
        row.copy_from_slice(&src.d[sy * src.w + r.x..sy * src.w + r.x + r.w]);
    });
    Plane { w: r.w, h: r.h, d }
}

/// Source of frames; implemented over the file list so that only one decoded
/// image is resident at a time.
pub trait FrameSource: Sync {
    fn count(&self) -> usize;
    fn load(&self, i: usize) -> anyhow::Result<RgbImage>;
}

pub struct FuseOptions {
    pub mode: FusionMode,
    pub levels: usize,
    pub rect: Rect,
    /// Per frame, per channel additive offset applied before fusion.
    pub offsets: Vec<[f32; 3]>,
    /// How strongly `bracket` mode prefers the per-band winner over the depth
    /// interpolation, in `[0, 1]`.
    pub bracket_strength: f32,
    /// How many frames either side of the depth value may be considered as the
    /// per-band winner. 0 restricts the choice to the two bracketing frames.
    pub bracket_radius: usize,

    /// Candidate-window radius used for the coarsest `coarse_levels` detail
    /// levels. See the note at the call site.
    pub bracket_radius_coarse: usize,

    /// How many of the coarsest detail levels use `bracket_radius_coarse` instead
    /// of `bracket_radius`.
    pub coarse_levels: usize,
    /// Depth-dependent registration, sampled per pixel. See [`Parallax`].
    pub parallax: Option<Parallax>,
    /// Soft-threshold strength applied to the fused detail bands, in units of
    /// the estimated per-band noise sigma. 0 disables denoising.
    pub denoise: f32,
    /// Whether the coarsest level — the low-frequency residual carrying colour
    /// and brightness — may also take the per-band winner, and how strongly.
    ///
    /// Off by default (0). Letting it follow the depth value instead is what
    /// normally keeps a wrong depth from producing a halo: a local DC shift is
    /// far more visible than a detail band taken from the wrong frame.
    ///
    /// Kept because it is the natural next suspect, but **measured neutral on the
    /// reference stack**: 0.5 and 1.0 moved the backdrop's acutance by 0.01 and
    /// the global gradient energy by 0.01. The reason is that the coarsest level
    /// holds only very low frequencies, and defocus barely changes those — so it
    /// does not matter whether that level came from frame 18 or frame 33.
    pub base_winner: f32,
    /// Per-pixel uncertainty of the depth field, in `[0, 1]`, at full resolution.
    ///
    /// Where the per-pixel estimate agreed with the coarse prior the depth is
    /// trustworthy and the candidate window stays at `bracket_radius`; where the
    /// two disagreed it is not, and the window opens by up to
    /// `adaptive_window` more frames so the band's own energy decides. Narrow
    /// windows are what keep a halo from forming at a depth boundary, so paying
    /// for width only where the depth is actually in doubt is strictly better
    /// than paying for it everywhere.
    pub slack: Option<Plane>,
    /// How many extra frames the window may open by where `slack` is 1.
    pub adaptive_window: f32,
    /// How sharply the candidate window closes where the depth field changes
    /// fast, in frames per pixel. 0 disables it (the window keeps its full width
    /// everywhere).
    ///
    /// This is what stops the seam halo. At an object boundary the depth field
    /// is a short ramp rather than a step, so pixels inside the ramp sit at an
    /// intermediate depth: their *base* band (colour and brightness) then comes
    /// from a frame in which neither surface is sharp, while their *detail*
    /// bands may still be won by a frame up to `bracket_radius` away, in which
    /// one surface is very sharp. The two halves therefore disagree — the
    /// defocused object's bloom underneath its own sharp detail — and the eye
    /// reads it as a ring of the object smeared out past its edge.
    ///
    /// Closing the window where the depth changes fast restores agreement: at a
    /// seam the detail bands come from the same two frames as the base, which is
    /// exactly what `--bracket-radius 0` does everywhere. Away from seams the
    /// window stays fully open, so the detail that the wide window buys is kept.
    pub seam_window: f32,
    /// Exponent of the energy weighting used to combine the candidates at a
    /// band. 0 keeps the plain maximum; larger values concentrate on the winner.
    /// See [`power_mean`].
    pub band_power: f32,
    /// Radius of the box aggregation applied to a band's energy *before* the
    /// winner is chosen. 0 decides the winner on the single-pixel energy, as
    /// before.
    ///
    /// The winner is a per-pixel maximum, so where a band holds no signal the
    /// argmax follows the candidate frames' noise and flips from frame to frame
    /// across neighbouring pixels. Assembling a band out of different frames per
    /// pixel puts several independent noise realisations side by side, which is
    /// the grain visible on a smooth subject surface. Aggregating the energy over
    /// a neighbourhood first makes the choice coherent over that neighbourhood —
    /// the band then comes from one frame locally, and carries that frame's own
    /// micro-texture instead of a mixture. Where a signal does exist it survives
    /// the aggregation (the sharp frame wins over an area, not at a pixel), so
    /// detail is unaffected.
    pub band_energy_smooth: usize,
    /// Runner-up/best ratio above which a pixel's winner is treated as a coin
    /// toss and handed to the neighbourhood-coherent winner. See
    /// [`resolve_ambiguous`].
    pub band_energy_gate: f32,
    /// Clamp the fused image back into the per-pixel range spanned by the frames
    /// that were actually fused. See [`clamp_to_range`].
    pub clamp_range: bool,
    /// Which side of that range to enforce, and how much slack to leave.
    /// See [`clamp_to_range`] for why the two sides are not equivalent.
    pub clamp_lo: bool,
    pub clamp_hi: bool,
    pub clamp_slack: f32,
    /// Radius by which the *upper* bound is dilated before clamping. The
    /// per-pixel maximum is noisy at the finest scale, so clamping to it raw
    /// cuts into real relief; dilating asks "did any frame, anywhere nearby,
    /// reach this value" instead. 0 uses the raw bound. See [`clamp_to_range`].
    pub clamp_hi_dilate: usize,

    /// Radius, in pixels, over which the colour a nearer surface spills onto the
    /// surface behind it is pulled back to that surface's own colour. 0 disables.
    /// See [`suppress_depth_spill`].
    pub spill: usize,
    /// Write the per-pixel frame range that the clamp uses, for diagnosis. The
    /// bound is built from the *warped* frames, so it is the only authority on
    /// whether a value the fusion produced is one that no frame contains.
    pub save_range: Option<String>,
    /// Per-pixel focus reliability (peak-to-mean of the response over the stack).
    /// Where it falls below `conf_lo` the band falls back to the depth
    /// interpolation, because the per-band winner there is a coin flip between
    /// frames and the mixture of their noise is what shows up as mottle on a
    /// smooth surface. See [`mix_bands_gated`].
    pub conf: Option<Plane>,
    pub conf_lo: f32,
    pub conf_hi: f32,
}

/// Produce the merged image.
pub fn fuse(
    source: &dyn FrameSource,
    transforms: &[Transform],
    depth: &Plane,
    opts: &FuseOptions,
    progress: &(dyn Fn(usize, usize) + Sync),
) -> anyhow::Result<RgbImage> {
    let n = source.count();
    let (ow, oh) = (opts.rect.w, opts.rect.h);
    let levels = opts
        .levels
        .max(1)
        .min(crate::pyramid::max_levels(ow, oh));

    // Depth field restricted to the output window, plus its per-level versions.
    let sizes = crate::pyramid::pyramid_sizes(ow, oh, levels);
    let mut dlev: Vec<Vec<f32>> = Vec::with_capacity(levels);
    let mut cur = crop_plane(depth, opts.rect);
    for l in 0..levels {
        dlev.push(cur.d.clone());
        if l + 1 < levels {
            cur = crate::depth::downsample_depth(&cur);
        }
    }

    // Per-level focus reliability, for the gated band combination.
    let mut clev: Vec<Vec<f32>> = Vec::with_capacity(levels);
    if opts.conf.is_some() {
        let mut ccur = crop_plane(opts.conf.as_ref().unwrap(), opts.rect);
        for l in 0..levels {
            clev.push(ccur.d.clone());
            if l + 1 < levels {
                ccur = crate::depth::downsample_depth(&ccur);
            }
        }
    }

    // Same treatment for the per-pixel depth uncertainty.
    let mut slev: Vec<Vec<f32>> = Vec::with_capacity(levels);
    let mut scur = match &opts.slack {
        Some(p) => crop_plane(p, opts.rect),
        None => Plane { w: opts.rect.w, h: opts.rect.h, d: vec![0.0f32; opts.rect.w * opts.rect.h] },
    };
    for l in 0..levels {
        slev.push(scur.d.clone());
        if l + 1 < levels {
            scur = crate::depth::downsample_depth(&scur);
        }
    }

    // Per-level measure of how fast the depth changes, used to close the
    // candidate window at object boundaries.
    let selev: Vec<Vec<f32>> = if opts.seam_window > 0.0 {
        (0..levels)
            .map(|l| {
                let (w, h) = sizes[l];
                debug_assert_eq!(w * h, dlev[l].len());
                seam_measure(&Plane { w, h, d: dlev[l].clone() })
            })
            .collect()
    } else {
        Vec::new()
    };

    let warp_all = |k: usize| -> anyhow::Result<Grid> {
        let mut rgb = source.load(k)?;
        apply_offset(&mut rgb, &opts.offsets[k]);
        let g = Grid { w: rgb.w, h: rgb.h, c: 3, d: rgb.d };
        Ok(warp_grid_region_par(
            &g,
            &transforms[k],
            ow,
            oh,
            opts.rect.x as f32,
            opts.rect.y as f32,
            0.0,
            opts.parallax.as_ref(),
        ))
    };

    if n == 1 {
        let w = warp_all(0)?;
        return Ok(RgbImage { w: w.w, h: w.h, d: w.d });
    }

    // Per-pixel range spanned by the frames that go into the merge. A Laplacian
    // reconstruction can leave it: at a high-contrast edge the bright side's
    // overshoot and the dark side's undershoot are drawn from different frames,
    // and their sum then sits outside what any frame contains. Measured on the
    // reference stack, that produced pixels darker than every frame in the stack
    // (0.86% of pixels below L=5 against a hard floor of 0.32%) — visible as a
    // black fringe along the dark side of bright/dark boundaries. Clamping back
    // into the observed range makes "no value the inputs do not contain" a hard
    // guarantee rather than a hope.
    let mut range_lo = if opts.clamp_range { vec![f32::INFINITY; ow * oh * 3] } else { Vec::new() };
    let mut range_hi = if opts.clamp_range { vec![f32::NEG_INFINITY; ow * oh * 3] } else { Vec::new() };

    match opts.mode {
        FusionMode::Bracket | FusionMode::Pyramid => {
            let strength = if opts.mode == FusionMode::Bracket {
                opts.bracket_strength.clamp(0.0, 1.0)
            } else {
                0.0
            };
            let radius = opts.bracket_radius;
            let coarse_levels = opts.coarse_levels;
            let last = sizes.len() - 1;
            // Depth interpolation accumulator, per-band window winner, the energy
            // that chose it, and — for the soft (power-mean) candidate — the
            // energy-weighted numerator and denominator.
            let mut acc_d: Vec<Grid> =
                sizes.iter().map(|(w, h)| Grid::new(*w, *h, 3)).collect();
            let mut acc_m: Vec<Grid> =
                sizes.iter().map(|(w, h)| Grid::new(*w, *h, 3)).collect();
            let mut acc_e: Vec<Vec<f32>> = sizes
                .iter()
                .map(|(w, h)| vec![f32::NEG_INFINITY; w * h])
                .collect();
            let soft = opts.band_power > 0.0;
            let mut acc_num: Vec<Grid> = if soft {
                sizes.iter().map(|(w, h)| Grid::new(*w, *h, 3)).collect()
            } else {
                Vec::new()
            };
            let mut acc_den: Vec<Vec<f32>> = if soft {
                sizes.iter().map(|(w, h)| vec![0.0f32; w * h]).collect()
            } else {
                Vec::new()
            };
            // For the neighbourhood-coherent winner: the runner-up energy (to
            // tell a decisive winner from a coin toss), and the winner of the
            // *aggregated* energy with its coefficients. Only allocated when the
            // aggregation is on, so the default `--band-energy-smooth 0` path
            // keeps its exact cost and output.
            let sm = opts.band_energy_smooth > 0;
            let mut acc_e2: Vec<Vec<f32>> = if sm {
                sizes.iter().map(|(w, h)| vec![0.0f32; w * h]).collect()
            } else {
                Vec::new()
            };
            let mut acc_eg: Vec<Vec<f32>> = if sm {
                sizes.iter().map(|(w, h)| vec![f32::NEG_INFINITY; w * h]).collect()
            } else {
                Vec::new()
            };
            let mut acc_mg: Vec<Grid> = if sm {
                sizes.iter().map(|(w, h)| Grid::new(*w, *h, 3)).collect()
            } else {
                Vec::new()
            };
            for k in 0..n {
                let warped = warp_all(k)?;
                note_range(&mut range_lo, &mut range_hi, &warped);
                let mut gp = gaussian_pyramid(&warped, levels);
                let nl = gp.len();
                for i in (0..nl).rev() {
                    let li = if i == nl - 1 {
                        gp[i].clone()
                    } else {
                        laplacian_level(&gp[i], &gp[i + 1])
                    };
                    let num = acc_num.get_mut(i);
                    let den = acc_den.get_mut(i);
                    let seam = selev.get(i).map(|v| v.as_slice());
                    // Energy aggregated over a neighbourhood, so the winner is
                    // decided over an area rather than at a pixel. See
                    // `FuseOptions::band_energy_smooth`.
                    let eplane = if opts.band_energy_smooth > 0 {
                        let (lw, lh) = sizes[i];
                        let mut e = vec![0.0f32; lw * lh];
                        for (px, ch) in li.d.chunks(3).enumerate() {
                            e[px] = ch.iter().map(|v| v.abs()).sum::<f32>();
                        }
                        Some(crate::util::box_blur(
                            &Plane { w: lw, h: lh, d: e },
                            opts.band_energy_smooth,
                        ))
                    } else {
                        None
                    };
                    accumulate_bands(
                        &mut acc_d[i],
                        &mut acc_m[i],
                        &mut acc_e[i],
                        if sm { Some(&mut acc_e2[i]) } else { None },
                        if sm { Some(&mut acc_eg[i]) } else { None },
                        if sm { Some(&mut acc_mg[i]) } else { None },
                        num,
                        den,
                        opts.band_power,
                        &li,
                        eplane.as_ref(),
                        k,
                        &dlev[i],
                        n,
                        // The coarsest detail levels are reconstructed from the
                        // narrow window. The wide one is there to rescue *fine*
                        // detail whose depth was misread; at these scales it
                        // instead hands the pixel to a frame whose bloom from the
                        // neighbouring surface is what the maximum is picking up,
                        // and a bloom is a coarse feature. Measured on the marked
                        // silhouette: the wall band came out 4.6 levels brighter
                        // than the frame that has the wall in focus, 6.8 right at
                        // the contour; narrowing only these levels keeps the
                        // backdrop's fine relief (which the wide window earns)
                        // while removing the band. See `banddelta.py`.
                        if coarse_levels > 0 && i + coarse_levels >= last {
                            opts.bracket_radius_coarse
                        } else {
                            radius
                        },
                        i == last,
                        opts.base_winner,
                        &slev[i],
                        opts.adaptive_window,
                        seam,
                        opts.seam_window,
                    );
                    // gp[i+1] has been consumed by level i; drop it so peak
                    // memory stays near one pyramid instead of two.
                    if i + 1 < nl {
                        gp[i + 1] = Grid::new(0, 0, 1);
                    }
                }
                progress(k + 1, n);
            }
            if let Some(prefix) = &opts.save_range {
                save_range(prefix, &range_lo, &range_hi, ow, oh);
            }
            // Where the per-pixel winner was not decisive, hand the pixel to the
            // neighbourhood-coherent winner instead. See `resolve_ambiguous`.
            if sm {
                for i in 0..sizes.len() {
                    resolve_ambiguous(
                        &mut acc_m[i],
                        &acc_mg[i],
                        &acc_e[i],
                        &acc_e2[i],
                        opts.band_energy_gate,
                    );
                }
            }
            let gated = !clev.is_empty();
            let mut lp: Vec<Grid> = (0..sizes.len())
                .map(|i| {
                    let s = if i == last { strength * opts.base_winner } else { strength };
                    if s == 0.0 {
                        acc_d[i].clone()
                    } else if soft {
                        let win = power_mean(&acc_num[i], &acc_den[i], &acc_m[i]);
                        mix_bands(&acc_d[i], &win, s)
                    } else if gated {
                        mix_bands_gated(
                            &acc_d[i],
                            &acc_m[i],
                            s,
                            &clev[i],
                            opts.conf_lo,
                            opts.conf_hi,
                        )
                    } else {
                        mix_bands(&acc_d[i], &acc_m[i], s)
                    }
                })
                .collect();
            denoise_bands(&mut lp, opts.denoise);
            let mut out = collapse(&lp);
            clamp_to_range(&mut out.d, &range_lo, &range_hi, ow, oh, 3, opts);
            suppress_depth_spill(&mut out.d, ow, oh, &dlev[0], opts.spill);
            Ok(RgbImage { w: out.w, h: out.h, d: out.d })
        }
        FusionMode::Max => {
            let mut acc: Vec<Grid> =
                sizes.iter().map(|(w, h)| Grid::new(*w, *h, 3)).collect();
            let mut energy: Vec<Vec<f32>> = sizes
                .iter()
                .map(|(w, h)| vec![f32::NEG_INFINITY; w * h])
                .collect();
            for k in 0..n {
                let warped = warp_all(k)?;
                note_range(&mut range_lo, &mut range_hi, &warped);
                let mut gp = gaussian_pyramid(&warped, levels);
                let nl = gp.len();
                for i in (0..nl).rev() {
                    let li = if i == nl - 1 {
                        gp[i].clone()
                    } else {
                        laplacian_level(&gp[i], &gp[i + 1])
                    };
                    accumulate_max(&mut acc[i], &mut energy[i], &li);
                    if i + 1 < nl {
                        gp[i + 1] = Grid::new(0, 0, 1);
                    }
                }
                progress(k + 1, n);
            }
            let mut lp = acc;
            denoise_bands(&mut lp, opts.denoise);
            let mut out = collapse(&lp);
            clamp_to_range(&mut out.d, &range_lo, &range_hi, ow, oh, 3, opts);
            suppress_depth_spill(&mut out.d, ow, oh, &dlev[0], opts.spill);
            Ok(RgbImage { w: out.w, h: out.h, d: out.d })
        }
        FusionMode::Pixel | FusionMode::Hard => {
            let mut out = vec![0.0f32; ow * oh * 3];
            for k in 0..n {
                let warped = warp_all(k)?;
                note_range(&mut range_lo, &mut range_hi, &warped);
                accumulate_pixel(&mut out, &warped, k, &dlev[0], n, opts.mode);
                progress(k + 1, n);
            }
            let mut img = RgbImage { w: ow, h: oh, d: out };
            clamp_to_range(&mut img.d, &range_lo, &range_hi, ow, oh, 3, opts);
            suppress_depth_spill(&mut img.d, ow, oh, &dlev[0], opts.spill);
            Ok(img)
        }
    }
}

/// Separable min or max filter over a single-channel plane.
fn sep_extreme(src: &[f32], w: usize, h: usize, r: usize, want_min: bool) -> Vec<f32> {
    let mut tmp = vec![0.0f32; src.len()];
    tmp.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
        for x in 0..w {
            let a = x.saturating_sub(r);
            let b = (x + r).min(w - 1);
            let mut v = src[y * w + a];
            for k in a + 1..=b {
                let s = src[y * w + k];
                if (want_min && s < v) || (!want_min && s > v) {
                    v = s;
                }
            }
            row[x] = v;
        }
    });
    let mut out = vec![0.0f32; src.len()];
    out.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
        let a = y.saturating_sub(r);
        let b = (y + r).min(h - 1);
        for x in 0..w {
            let mut v = tmp[a * w + x];
            for k in a + 1..=b {
                let s = tmp[k * w + x];
                if (want_min && s < v) || (!want_min && s > v) {
                    v = s;
                }
            }
            row[x] = v;
        }
    });
    out
}

/// Remove the colour that a nearer surface spills onto whatever stands behind it.
///
/// Defocus does not respect silhouettes: a coloured object in front of a wall
/// spreads its light onto that wall, so just outside the outline the wall carries
/// a wash of the object's hue, 15-20 px wide on the reference stack. It is
/// physically real — *every* frame of that stack carries more of it than the
/// fusion does (36-52 units of R-G against the output's 24) — but it reads as the
/// object glowing onto the wall, and the hand-retouched reference has all but
/// removed it.
///
/// The decision is made on the **depth field alone**, never on colour, so the
/// background may be a grey wall, a green cloth or a wooden table: the wash can
/// only appear on the *far* side of a depth step, because that is the only place
/// a nearer surface can spill onto. Two things follow from that.
///
/// * The pixels to correct are the ones within `radius` of a place where the
///   depth field jumps by more than a few frames.
/// * The colour to use instead comes from the **same depth slice**, sampled well
///   away from any jump, so it is the colour of the background itself at that
///   distance and not a mixture of background and object.
///
/// Only the low-frequency colour is changed: the correction is a per-slice colour
/// offset applied to a block average, so texture and detail come through
/// untouched. `radius == 0` disables it.
pub fn suppress_depth_spill(
    d: &mut [f32],
    w: usize,
    h: usize,
    depth: &[f32],
    radius: usize,
) {
    const S: usize = 4; // work at 1/4 resolution: this is a colour correction
    const STEP: f32 = 3.0; // frames of depth jump that count as a silhouette
    /// Share of the reference window that must hold usable samples before the
    /// correction is applied at full strength.
    const TRUST: f32 = 0.05;
    if radius == 0 || w < 8 * S || h < 8 * S || depth.len() < w * h {
        return;
    }
    let (sw, sh) = (w / S, h / S);
    let n4 = sw * sh;
    let inv = 1.0 / (S * S) as f32;

    // ---- colour and depth at 1/4 scale -------------------------------------
    let mut col = vec![0.0f32; n4 * 3];
    col.par_chunks_mut(3).enumerate().for_each(|(i, px)| {
        let (bx, by) = (i % sw, i / sw);
        let mut acc = [0.0f32; 3];
        for y in 0..S {
            for x in 0..S {
                let si = ((by * S + y) * w + bx * S + x) * 3;
                acc[0] += d[si];
                acc[1] += d[si + 1];
                acc[2] += d[si + 2];
            }
        }
        px[0] = acc[0] * inv;
        px[1] = acc[1] * inv;
        px[2] = acc[2] * inv;
    });
    let mut dep = vec![0.0f32; n4];
    dep.par_iter_mut().enumerate().for_each(|(i, v)| {
        let (bx, by) = (i % sw, i / sw);
        let mut acc = 0.0f32;
        for y in 0..S {
            for x in 0..S {
                acc += depth[(by * S + y) * w + bx * S + x];
            }
        }
        *v = acc * inv;
    });

    // ---- where a silhouette is, and how far each pixel is from one ---------
    //
    // The silhouette is the depth *gradient*, not the local depth span. A span
    // test with a window as wide as the reference radius marks a band tens of
    // pixels thick, which leaves no "far from a silhouette" pixel inside the
    // reference window - and then the pass silently does nothing where it is
    // needed. A gradient gives a line a pixel or two wide, so the distance field
    // and the reference regions both mean what they say.
    let rw = (radius / S).max(2);
    let smooth = box_blur(&Plane { w: sw, h: sh, d: dep.clone() }, 2);
    let mut cur = vec![0.0f32; n4];
    cur.par_iter_mut().enumerate().for_each(|(i, v)| {
        let (bx, by) = (i % sw, i / sw);
        let gx = if bx + 1 < sw { smooth.d[i + 1] - smooth.d[i] } else { 0.0 };
        let gy = if by + 1 < sh { smooth.d[i + sw] - smooth.d[i] } else { 0.0 };
        *v = if (gx * gx + gy * gy).sqrt() > STEP { 1.0 } else { 0.0 };
    });
    let cap = radius / S + 1;
    let mut dist = vec![(cap + 1) as f32; n4];
    for k in 0..=cap {
        for i in 0..n4 {
            if cur[i] > 0.0 && dist[i] > cap as f32 {
                dist[i] = k as f32;
            }
        }
        if k == cap {
            break;
        }
        let p = Plane { w: sw, h: sh, d: cur.clone() };
        let b = box_blur(&p, 1);
        cur.par_iter_mut().enumerate().for_each(|(i, v)| {
            *v = if b.d[i] > 0.0 { 1.0 } else { 0.0 };
        });
    }

    // ---- reference colour: the surface behind, sampled clear of the figure ---
    //
    // A pixel standing on the surface behind should be coloured like that surface.
    // The candidates are the pixels that are *both* clear of the silhouette - the
    // wash hugs it, so they are the contaminated ones - and on its far side, since
    // the near side is the figure itself. Averaging them over a window several
    // times the correction radius gives the surface's own colour.
    //
    // An earlier attempt took the replacement from "the same depth slice" instead.
    // That fails, because the wash is *on* the surface behind: the replacement and
    // the thing it replaces are in the same slice, so excluding the wash leaves
    // nothing to sample from and including it pulls the answer back towards the
    // wash. Distance from the silhouette is the discriminator that works.
    let dmax = sep_extreme(&dep, sw, sh, rw, false);
    let mut mask = Plane::new(sw, sh);
    mask.d.par_iter_mut().enumerate().for_each(|(i, v)| {
        *v = if dist[i] > cap as f32 && dmax[i] - dep[i] < STEP {
            1.0
        } else {
            0.0
        };
    });
    let rwm = rw * 4;
    let wsum = box_blur(&mask, rwm);
    let mut refc = vec![0.0f32; n4 * 3];
    for ch in 0..3 {
        let mut p = Plane::new(sw, sh);
        p.d.par_iter_mut().enumerate().for_each(|(i, v)| {
            *v = col[i * 3 + ch] * mask.d[i];
        });
        let sums = box_blur(&p, rwm);
        refc.par_chunks_mut(3).enumerate().for_each(|(i, px)| {
            px[ch] = sums.d[i] / wsum.d[i].max(1e-3);
        });
    }

    // ---- apply the offset, leaving the detail in place ---------------------
    let mut off = vec![0.0f32; n4 * 3];
    let mut alpha = vec![0.0f32; n4];
    for i in 0..n4 {
        if dmax[i] - dep[i] >= STEP {
            continue; // this pixel is on the figure, not on the surface behind
        }
        // How much of the window held a usable reference. A hard test here would
        // throw away the pixels nearest the silhouette - exactly the ones that
        // need the correction - because the sample count is naturally lowest
        // there; a soft ramp keeps them, at a weight that says how much to trust.
        let rel = (wsum.d[i] / TRUST).clamp(0.0, 1.0);
        let a = (1.0 - dist[i] / (cap as f32 + 1.0)).clamp(0.0, 1.0) * rel;
        if a <= 0.0 {
            continue;
        }
        alpha[i] = a;
        for ch in 0..3 {
            off[i * 3 + ch] = (refc[i * 3 + ch] - col[i * 3 + ch]) * a;
        }
    }
    // Bilinear, *not* nearest. The correction lives on a 1/S grid; replicating
    // those cells prints a 1/S-pixel checkerboard of colour steps over everything
    // it touches, which shows up as a large rise in gradient energy on a flat
    // backdrop (measured: 27.8 to 42.7) - the pass is supposed to change hue, not
    // to add edges.
    d.par_chunks_mut(3).enumerate().for_each(|(i, px)| {
        let (bx, by) = (i % w, i / w);
        let fx = bx as f32 / S as f32 - 0.5;
        let fy = by as f32 / S as f32 - 0.5;
        let (x0, y0) = (fx.floor(), fy.floor());
        let (tx, ty) = (fx - x0, fy - y0);
        let (x0c, x1c) = (
            (x0 as i64).clamp(0, sw as i64 - 1) as usize,
            (x0 as i64 + 1).clamp(0, sw as i64 - 1) as usize,
        );
        let (y0c, y1c) = (
            (y0 as i64).clamp(0, sh as i64 - 1) as usize,
            (y0 as i64 + 1).clamp(0, sh as i64 - 1) as usize,
        );
        let (a00, a01) = ((1.0 - tx) * (1.0 - ty), tx * (1.0 - ty));
        let (a10, a11) = ((1.0 - tx) * ty, tx * ty);
        let (i00, i01) = (y0c * sw + x0c, y0c * sw + x1c);
        let (i10, i11) = (y1c * sw + x0c, y1c * sw + x1c);
        for ch in 0..3 {
            px[ch] += off[i00 * 3 + ch] * a00
                + off[i01 * 3 + ch] * a01
                + off[i10 * 3 + ch] * a10
                + off[i11 * 3 + ch] * a11;
        }
    });
}

/// Fold one warped frame into the per-pixel min/max range.

fn note_range(lo: &mut [f32], hi: &mut [f32], g: &Grid) {
    if lo.is_empty() {
        return;
    }
    lo.par_chunks_mut(4096)
        .zip(hi.par_chunks_mut(4096))
        .zip(g.d.par_chunks(4096))
        .for_each(|((l, h), s)| {
            for i in 0..l.len() {
                let v = s[i];
                if v < l[i] {
                    l[i] = v;
                }
                if v > h[i] {
                    h[i] = v;
                }
            }
        });
}

/// Clamp every channel back into the range the frames spanned at that pixel.
fn clamp_to_range(
    d: &mut [f32],
    lo: &[f32],
    hi: &[f32],
    w: usize,
    h: usize,
    c: usize,
    opts: &FuseOptions,
) {
    if lo.is_empty() || !(opts.clamp_lo || opts.clamp_hi) {
        return;
    }
    let (lo_side, hi_side, slack) = (opts.clamp_lo, opts.clamp_hi, opts.clamp_slack);
    // The upper bound is dilated before use.
    //
    // `hi` is a per-pixel maximum over the frames, so at the finest scale it is
    // noisy: one pixel's peak can sit ten levels above its neighbours'. Clamping
    // to it directly therefore cuts into the relief of a textured surface
    // (measured: the silhouette band's texture fell to 5.4 against 8.2 for the
    // reference and 8.2 for the raw winner). Dilating it replaces "the peak at
    // this pixel" with "the peak anywhere nearby", which is what a bound on
    // *content* should be: the bound only has to exclude what no frame contains,
    // and a bright rim that no frame reaches within several pixels is exactly
    // that.
    let hi_d = if hi_side && opts.clamp_hi_dilate > 0 {
        dilate_max(hi, w, h, c, opts.clamp_hi_dilate)
    } else {
        Vec::new()
    };
    let hi = if hi_d.is_empty() { hi } else { &hi_d[..] };
    // The two sides are not symmetric. Below `lo` sits the undershoot that made
    // the dark side of a bright/dark boundary darker than every frame (bug 6);
    // above `hi` sits the reconstruction that carries the far surface's relief
    // next to a silhouette — clamping that flattens the very texture we are
    // trying to keep. Keep the switches separate so each can be chosen on
    // evidence.
    d.par_chunks_mut(4096).enumerate().for_each(|(ci, chunk)| {
        let base = ci * 4096;
        for (j, v) in chunk.iter_mut().enumerate() {
            let i = base + j;
            if lo[i] > hi[i] {
                continue;
            }
            if lo_side && *v < lo[i] - slack {
                *v = lo[i] - slack;
            }
            if hi_side && *v > hi[i] + slack {
                *v = hi[i] + slack;
            }
        }
    });
}

/// Write the clamp's per-pixel bounds as two 16-bit PNGs (value x256).
fn save_range(prefix: &str, lo: &[f32], hi: &[f32], w: usize, h: usize) {
    if lo.is_empty() {
        return;
    }
    for (suffix, src) in [("lo", lo), ("hi", hi)] {
        let mut data = vec![0u16; w * h];
        for (i, v) in data.iter_mut().enumerate() {
            let s = i * 3;
            let m = src[s].max(src[s + 1]).max(src[s + 2]);
            *v = (m.clamp(0.0, 255.999) * 256.0) as u16;
        }
        if let Some(img) =
            image::ImageBuffer::<image::Luma<u16>, Vec<u16>>::from_raw(w as u32, h as u32, data)
        {
            let _ = img.save(format!("{}_{}.png", prefix, suffix));
        }
    }
}

/// Separable max filter, used to dilate the upper clamp bound. See
/// [`clamp_to_range`].
fn dilate_max(src: &[f32], w: usize, h: usize, c: usize, r: usize) -> Vec<f32> {
    let mut tmp = vec![0.0f32; src.len()];
    tmp.par_chunks_mut(w * c).enumerate().for_each(|(y, row)| {
        for x in 0..w {
            let x0 = x.saturating_sub(r);
            let x1 = (x + r).min(w - 1);
            for ch in 0..c {
                let mut m = f32::NEG_INFINITY;
                for k in x0..=x1 {
                    let v = src[(y * w + k) * c + ch];
                    if v > m {
                        m = v;
                    }
                }
                row[x * c + ch] = m;
            }
        }
    });
    let mut out = vec![0.0f32; src.len()];
    out.par_chunks_mut(w * c).enumerate().for_each(|(y, row)| {
        let y0 = y.saturating_sub(r);
        let y1 = (y + r).min(h - 1);
        for x in 0..w {
            for ch in 0..c {
                let mut m = f32::NEG_INFINITY;
                for k in y0..=y1 {
                    let v = tmp[(k * w + x) * c + ch];
                    if v > m {
                        m = v;
                    }
                }
                row[x * c + ch] = m;
            }
        }
    });
    out
}

/// `level - upsample(next)`, i.e. one Laplacian band.
fn laplacian_level(level: &Grid, next: &Grid) -> Grid {
    let up = next.pyr_up(level.w, level.h);
    let mut d = vec![0.0f32; level.len()];
    d.par_chunks_mut(4096).enumerate().for_each(|(ci, chunk)| {
        let b = ci * 4096;
        for (j, o) in chunk.iter_mut().enumerate() {
            *o = level.d[b + j] - up.d[b + j];
        }
    });
    Grid { w: level.w, h: level.h, c: level.c, d }
}

fn apply_offset(img: &mut RgbImage, off: &[f32; 3]) {
    if off[0] == 0.0 && off[1] == 0.0 && off[2] == 0.0 {
        return;
    }
    img.d.par_chunks_mut(4096).enumerate().for_each(|(ci, chunk)| {
        let base = ci * 4096;
        for (i, v) in chunk.iter_mut().enumerate() {
            *v = (*v + off[(base + i) % 3]).clamp(0.0, 1.0);
        }
    });
}

/// Magnitude of the depth field's gradient, smoothed with a small box so that no
/// single noisy pixel can close the candidate window on its own.
fn seam_measure(d: &Plane) -> Vec<f32> {
    let mut g = Plane::new(d.w, d.h);
    g.d.par_chunks_mut(d.w).enumerate().for_each(|(y, row)| {
        for x in 0..d.w {
            let (xi, yi) = (x as i64, y as i64);
            let gx = d.sample_clamped(xi + 1, yi) - d.sample_clamped(xi - 1, yi);
            let gy = d.sample_clamped(xi, yi + 1) - d.sample_clamped(xi, yi - 1);
            // Central difference spans two pixels.
            row[x] = 0.5 * (gx * gx + gy * gy).sqrt();
        }
    });
    crate::util::box_blur(&g, 4).d
}

/// Factor in `(0, 1]` by which the candidate window is scaled at a pixel.
#[inline]
fn seam_window_scale(seam: f32, g0: f32) -> f32 {
    if g0 <= 0.0 {
        return 1.0;
    }
    let t = seam / g0;
    1.0 / (1.0 + t * t)
}

/// Weight of frame `k` for a pixel whose (fractional) best frame is `d`.
#[inline]
fn blend_weight(k: usize, d: f32, n: usize, mode: FusionMode) -> f32 {
    match mode {
        FusionMode::Hard => {
            if (d - k as f32).abs() <= 0.5 {
                1.0
            } else {
                0.0
            }
        }
        _ => {
            let last = (n - 2) as f32;
            let k0 = d.floor().clamp(0.0, last);
            let frac = (d - k0).clamp(0.0, 1.0);
            let kf = k as f32;
            if (kf - k0).abs() < 1e-6 {
                1.0 - frac
            } else if (kf - (k0 + 1.0)).abs() < 1e-6 {
                frac
            } else {
                0.0
            }
        }
    }
}

/// Route one band of one frame into the two accumulators.
///
/// * `acc_d` receives the depth-weighted contribution of the two bracketing
///   frames, i.e. the smooth, depth-driven composite.
/// * `acc_m` receives the coefficient of whichever frame within
///   `[floor(d) - radius, floor(d) + 1 + radius]` carries the most energy at
///   this band, together with that energy in `acc_e`; `acc_e2` receives the
///   runner-up energy.
/// * with `energy_smooth` set, `acc_mg`/`acc_eg` receive the same thing but
///   chosen on energy aggregated over a neighbourhood.
///
/// The final band is a mix of the two, which is what keeps fine detail crisp
/// without letting the selection wander far from the depth field.
#[allow(clippy::too_many_arguments)]
fn accumulate_bands(
    acc_d: &mut Grid,
    acc_m: &mut Grid,
    acc_e: &mut [f32],
    acc_e2: Option<&mut [f32]>,
    acc_eg: Option<&mut [f32]>,
    acc_mg: Option<&mut Grid>,
    acc_num: Option<&mut Grid>,
    acc_den: Option<&mut Vec<f32>>,
    power: f32,
    li: &Grid,
    energy_smooth: Option<&Plane>,
    k: usize,
    dlev: &[f32],
    n: usize,
    radius: usize,
    is_base_level: bool,
    base_winner: f32,
    slack: &[f32],
    adaptive: f32,
    seam: Option<&[f32]>,
    seam_g0: f32,
) {
    let (w, c) = (acc_d.w, acc_d.c);
    let last = (n - 2) as f32;
    let kf = k as f32;

    // One pixel: route the depth-weighted part, and either hard-max or
    // energy-weighted-average the candidate window.
    let row = |y: usize,
               rd: &mut [f32],
               rm: &mut [f32],
               re: &mut [f32],
               re2: Option<&mut [f32]>,
               reg: Option<&mut [f32]>,
               rmg: Option<&mut [f32]>,
               rn: Option<&mut [f32]>,
               rw: Option<&mut [f32]>| {
        let (mut re2, mut reg, mut rmg, mut rn, mut rw) = (re2, reg, rmg, rn, rw);
        for x in 0..w {
            let p = y * w + x;
            let sp = p * c;
            let dp = x * c;
            let d = dlev[p];
            let k0 = d.floor().clamp(0.0, last);
            let frac = (d - k0).clamp(0.0, 1.0);
            let wgt = if (kf - k0).abs() < 1e-6 {
                1.0 - frac
            } else if (kf - (k0 + 1.0)).abs() < 1e-6 {
                frac
            } else {
                0.0
            };
            if wgt != 0.0 {
                for ch in 0..c {
                    rd[dp + ch] += wgt * li.d[sp + ch];
                }
            }
            if is_base_level && base_winner <= 0.0 {
                continue;
            }
            // The candidate window, opened by the local depth uncertainty:
            // where the per-pixel estimate disagreed with the coarse prior the
            // depth cannot be trusted and the band's own energy has to decide,
            // where it agreed the window stays narrow — which is what keeps a
            // halo from forming across a depth boundary.
            let r = radius as f32 * seam.map_or(1.0, |s| seam_window_scale(s[p], seam_g0))
                + adaptive * slack[p];
            if kf < k0 - r || kf > k0 + 1.0 + r {
                continue;
            }
            let e = {
                let mut acc = 0.0f32;
                for ch in 0..c {
                    acc += li.d[sp + ch].abs();
                }
                acc
            };
            // Running raw maximum, with the runner-up so a later pass can tell a
            // decisive winner from a coin toss.
            if e > re[x] {
                if let Some(d2) = re2.as_deref_mut() {
                    d2[x] = re[x];
                }
                re[x] = e;
                for ch in 0..c {
                    rm[dp + ch] = li.d[sp + ch];
                }
            } else if let Some(d2) = re2.as_deref_mut() {
                if e > d2[x] {
                    d2[x] = e;
                }
            }
            // The same, but chosen on energy aggregated over a neighbourhood.
            if let Some(ea) = energy_smooth.map(|ep| ep.d[p]) {
                if let (Some(eg), Some(mg)) = (reg.as_deref_mut(), rmg.as_deref_mut()) {
                    if ea > eg[x] {
                        eg[x] = ea;
                        for ch in 0..c {
                            mg[dp + ch] = li.d[sp + ch];
                        }
                    }
                }
            }
            if let (Some(rn), Some(rw)) = (rn.as_deref_mut(), rw.as_deref_mut()) {
                let weight = (e + 1e-12).powf(power);
                rw[x] += weight;
                for ch in 0..c {
                    rn[dp + ch] += weight * li.d[sp + ch];
                }
            }
        }
    };

    match (acc_num, acc_den, acc_e2, acc_eg, acc_mg) {
        (Some(num), Some(den), Some(e2), Some(eg), Some(mg)) => {
            acc_d
                .d
                .par_chunks_mut(w * c)
                .zip(acc_m.d.par_chunks_mut(w * c))
                .zip(acc_e.par_chunks_mut(w))
                .zip(e2.par_chunks_mut(w))
                .zip(eg.par_chunks_mut(w))
                .zip(mg.d.par_chunks_mut(w * c))
                .zip(num.d.par_chunks_mut(w * c))
                .zip(den.par_chunks_mut(w))
                .enumerate()
                .for_each(|(y, (((((((rd, rm), re), e2), eg), mg), rn), rw))| {
                    row(
                        y,
                        rd,
                        rm,
                        re,
                        Some(e2),
                        Some(eg),
                        Some(mg),
                        Some(rn),
                        Some(rw),
                    )
                });
        }
        (Some(num), Some(den), _, _, _) => {
            acc_d
                .d
                .par_chunks_mut(w * c)
                .zip(acc_m.d.par_chunks_mut(w * c))
                .zip(acc_e.par_chunks_mut(w))
                .zip(num.d.par_chunks_mut(w * c))
                .zip(den.par_chunks_mut(w))
                .enumerate()
                .for_each(|(y, ((((rd, rm), re), rn), rw))| {
                    row(y, rd, rm, re, None, None, None, Some(rn), Some(rw))
                });
        }
        (_, _, Some(e2), Some(eg), Some(mg)) => {
            acc_d
                .d
                .par_chunks_mut(w * c)
                .zip(acc_m.d.par_chunks_mut(w * c))
                .zip(acc_e.par_chunks_mut(w))
                .zip(e2.par_chunks_mut(w))
                .zip(eg.par_chunks_mut(w))
                .zip(mg.d.par_chunks_mut(w * c))
                .enumerate()
                .for_each(|(y, (((((rd, rm), re), e2), eg), mg))| {
                    row(y, rd, rm, re, Some(e2), Some(eg), Some(mg), None, None)
                });
        }
        _ => {
            acc_d
                .d
                .par_chunks_mut(w * c)
                .zip(acc_m.d.par_chunks_mut(w * c))
                .zip(acc_e.par_chunks_mut(w))
                .enumerate()
                .for_each(|(y, ((rd, rm), re))| {
                    row(y, rd, rm, re, None, None, None, None, None)
                });
        }
    }
}

/// Hand a pixel to the neighbourhood-coherent winner where the per-pixel winner
/// was not decisive.
///
/// The winner is a per-pixel maximum, so where a band holds no signal the argmax
/// follows the candidates' noise and flips between frames from one pixel to the
/// next — the grain on a smooth subject surface. Aggregating the energy over a
/// neighbourhood first makes the choice coherent, but doing that *everywhere*
/// also overrides pixels where the raw winner was decisive and right, which
/// softens real texture (measured: the backdrop relief lost about 15% of its
/// high-frequency energy).
///
/// The runner-up energy tells the two cases apart without any absolute
/// threshold: a resolved edge or texture has one frame clearly ahead, while a
/// no-signal band has every candidate at the same level, so `runner_up / best`
/// approaches 1. `gate` is where the hand-over starts, with a fixed ramp of
/// ±0.15 around it.
fn resolve_ambiguous(acc_m: &mut Grid, acc_mg: &Grid, e1: &[f32], e2: &[f32], gate: f32) {
    let (w, c) = (acc_m.w, acc_m.c);
    let lo = gate - 0.15;
    acc_m.d.par_chunks_mut(w * c).enumerate().for_each(|(y, rm)| {
        for x in 0..w {
            let p = y * w + x;
            let a = e1[p];
            if !(a.is_finite() && a > 0.0) {
                continue;
            }
            let amb = (e2[p] / a).clamp(0.0, 1.0);
            // `gate <= 0` means "never discriminate": aggregate everywhere, which
            // is what the flag did before the gate existed.
            let t = if gate <= 0.0 {
                1.0
            } else {
                ((amb - lo) / 0.3).clamp(0.0, 1.0)
            };
            if t > 0.0 {
                let dp = x * c;
                for ch in 0..c {
                    rm[dp + ch] = rm[dp + ch] * (1.0 - t) + acc_mg.d[p * c + ch] * t;
                }
            }
        }
    });
}

/// Energy-weighted mean of the candidate band coefficients.
///
/// A hard per-pixel maximum is the natural way to keep the sharpest frame's
/// detail, but it is also an extreme-value statistic: where every candidate is
/// noise it returns `max(noise)`, which is systematically larger than any single
/// frame's noise. That is the grain this tool carries in flat areas.
///
/// Weighting each candidate by `energy^power` and renormalising is a continuous
/// family between the two behaviours. Where one frame genuinely dominates — a
/// real edge or a resolved texture, typically several times the runner-up — the
/// largest weight is so much larger that the mean is practically that frame
/// alone, so detail is untouched. Where the candidates are all comparable, which
/// is exactly the no-signal case, the mean is an average of independent noise
/// and shrinks it by roughly the square root of the number of candidates.
/// The adaptation is therefore automatic and needs no threshold.
fn power_mean(num: &Grid, den: &[f32], fallback: &Grid) -> Grid {
    let c = num.c;
    let mut out = vec![0.0f32; num.len()];
    // `den` is per pixel while `out`/`num` are interleaved, so iterate pixels.
    out.par_chunks_mut(4096 * c).enumerate().for_each(|(ci, chunk)| {
        let base = ci * 4096;
        for k in 0..chunk.len() / c {
            let p = base + k;
            let w = den[p];
            let sp = k * c;
            for ch in 0..c {
                let src = p * c + ch;
                chunk[sp + ch] = if w > 0.0 { num.d[src] / w } else { fallback.d[src] };
            }
        }
    });
    Grid { w: num.w, h: num.h, c, d: out }
}

/// Linear mix of the depth-driven band and the per-band winner.
/// As [`mix_bands`], but the per-band winner only contributes where the pixel
/// has a focus signal worth trusting.
///
/// `conf` is the peak-to-mean ratio of the response over the stack. Below
/// `lo` the band is taken entirely from the depth interpolation — on a smooth
/// surface that is a blend of the two frames the depth sits between, both near
/// focus, so no detail is invented and no grain is mixed in. Above `hi` the
/// winner is used in full. In between it fades, so a surface with a weak but
/// real signal keeps most of its texture.
fn mix_bands_gated(
    acc_d: &Grid,
    acc_m: &Grid,
    strength: f32,
    conf: &[f32],
    lo: f32,
    hi: f32,
) -> Grid {
    let inv = 1.0 / (hi - lo).max(1e-6);
    let c = acc_d.c;                    // `acc_d` is interleaved by channel; `conf` is not.
    let mut out = vec![0.0f32; acc_d.len()];
    out.par_chunks_mut(4096).enumerate().for_each(|(ci, chunk)| {
        let base = ci * 4096;
        for (i, o) in chunk.iter_mut().enumerate() {
            let p = (base + i) / c;
            let t = ((conf[p] - lo) * inv).clamp(0.0, 1.0);
            let a = acc_d.d[p];
            *o = a + strength * t * (acc_m.d[p] - a);
        }
    });
    Grid { w: acc_d.w, h: acc_d.h, c: acc_d.c, d: out }
}

fn mix_bands(acc_d: &Grid, acc_m: &Grid, strength: f32) -> Grid {
    let mut out = vec![0.0f32; acc_d.len()];
    out.par_chunks_mut(4096).enumerate().for_each(|(ci, chunk)| {
        let base = ci * 4096;
        for (i, o) in chunk.iter_mut().enumerate() {
            let a = acc_d.d[base + i];
            *o = a + strength * (acc_m.d[base + i] - a);
        }
    });
    Grid { w: acc_d.w, h: acc_d.h, c: acc_d.c, d: out }
}

/// Robust per-band noise estimate from the median absolute deviation.
///
/// Picking a winner per pixel is a maximum statistic: even from frames that hold
/// no real signal, the largest of N responses is systematically larger than any
/// single one. On the reference stack that inflated a flat backdrop from a
/// per-frame sigma of ~1.6 to ~13 once the per-pixel maximum was taken, so the
/// fused bands need shrinking back to the noise floor.
fn band_sigma(v: &[f32]) -> f32 {
    let mut s: Vec<f32> = v.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let med = s[s.len() / 2];
    let mut dev: Vec<f32> = v.iter().map(|x| (x - med).abs()).collect();
    dev.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    1.4826 * dev[dev.len() / 2]
}

/// Soft threshold (wavelet shrinkage) of every detail band.
///
/// Keeps large coefficients — real edges and texture — untouched while pulling
/// the small ones, which is where the selection noise lives, towards zero.
pub fn denoise_bands(lp: &mut [Grid], strength: f32) {
    if strength <= 0.0 || lp.len() < 2 {
        return;
    }
    let last = lp.len() - 1;
    for band in lp.iter_mut().take(last) {
        let sigma = band_sigma(&band.d);
        let tau = strength * sigma;
        if tau <= 0.0 {
            continue;
        }
        band.d.par_iter_mut().for_each(|c| {
            let a = c.abs() - tau;
            *c = if a > 0.0 { a.copysign(*c) } else { 0.0 };
        });
    }
}

/// Per-band "largest coefficient wins".
fn accumulate_max(acc: &mut Grid, energy: &mut [f32], li: &Grid) {
    let (w, c) = (acc.w, acc.c);
    acc.d
        .par_chunks_mut(w * c)
        .zip(energy.par_chunks_mut(w))
        .enumerate()
        .for_each(|(y, (acc_row, e_row))| {
            for x in 0..w {
                let p = y * w + x;
                let sp = p * c;
                let dp = x * c;
                let mut e = 0.0f32;
                for ch in 0..c {
                    e += li.d[sp + ch].abs();
                }
                if e > e_row[x] {
                    e_row[x] = e;
                    for ch in 0..c {
                        acc_row[dp + ch] = li.d[sp + ch];
                    }
                }
            }
        });
}

/// Pixel-domain blend of the two bracketing frames, or a hard pick.
fn accumulate_pixel(out: &mut [f32], img: &Grid, k: usize, dlev: &[f32], n: usize, mode: FusionMode) {
    let (w, c) = (img.w, img.c);
    out.par_chunks_mut(w * c)
        .enumerate()
        .for_each(|(y, row)| {
            for x in 0..w {
                let p = y * w + x;
                let wgt = blend_weight(k, dlev[p], n, mode);
                if wgt == 0.0 {
                    continue;
                }
                let sp = p * c;
                let dp = x * c;
                for ch in 0..c {
                    row[dp + ch] += wgt * img.d[sp + ch];
                }
            }
        });
}
