//! Muka-Focus-Stacking (command: `mukastack`) — merge a focus-bracketed photo
//! sequence into one all-in-focus image.
//!
//! Pipeline
//! --------
//! 1. **Decode + cache.** Each frame is decoded once, converted to grayscale
//!    and cached at a reduced analysis resolution (8-bit, so 50 frames of a
//!    33 MP stack cost a few hundred megabytes rather than tens of gigabytes).
//! 2. **Align.** Every frame is matched against the middle frame with a
//!    scale + translation search (phase correlation over a range of scales),
//!    which also handles the focus-breathing magnification drift.
//! 3. **Depth.** A focus measure (Sum of Modified Laplacian by default) is
//!    computed for each frame, the per-pixel winner is tracked with sub-index
//!    parabolic refinement, unreliable pixels are inpainted and the resulting
//!    fractional depth field is snapped to image edges with a guided filter.
//! 4. **Fuse.** Frames are re-decoded one at a time, aligned, decomposed into
//!    Laplacian pyramids, and each band is blended between the two frames
//!    bracketing the local depth value. Only one decoded frame plus the
//!    accumulators are ever resident.

mod align;
mod depth;
mod focus;
mod fuse;
mod pyramid;
mod util;

use anyhow::{bail, Context, Result};
use clap::Parser;
use rayon::prelude::*;
use std::path::{Path, PathBuf};
use std::time::Instant;

use align::{AlignOptions, Aligner, Transform};
use depth::{fill_unreliable, guided_filter, median_n, upsample_depth, DepthBuilder};
use focus::{focus_map, FocusMeasure};
use fuse::{valid_rect, FuseOptions, FusionMode, Rect};
use util::{Plane, RgbImage};

#[derive(Parser, Debug)]
#[command(
    name = "mukastack",
    about = "Merge a focus-bracketed sequence into a single all-in-focus image",
    version
)]
struct Cli {
    /// Input files, directories or a mix. Directories are scanned for images.
    #[arg(required = true)]
    inputs: Vec<PathBuf>,

    /// Output image path. Extension decides the format (.png = 16-bit).
    #[arg(short, long, default_value = "merged.png")]
    output: PathBuf,

    /// Fusion strategy.
    #[arg(short, long, value_enum, default_value_t = FusionMode::Bracket)]
    mode: FusionMode,

    /// How strongly each frequency band prefers the locally sharpest frame over
    /// the depth interpolation, in `[0, 1]`. 0 is identical to `--mode pyramid`.
    #[arg(long, default_value_t = 1.0)]
    bracket_strength: f32,

    /// Whether the low-frequency residual (colour and brightness) may also take
    /// the per-band winner instead of following the depth value. 0 keeps the old
    /// behaviour. Raising it recovers the low frequencies of a surface whose
    /// depth was misread, at the risk of a visible halo at depth boundaries.
    #[arg(long, default_value_t = 0.0)]
    base_winner: f32,

    /// How many frames either side of the depth value may win a band. 0 means
    /// only the two frames that bracket the depth are candidates.
    ///
    /// Wider than it looks like it needs to be, on purpose. The depth field is
    /// the weak link at low-contrast detail — a backdrop whose relief sits at the
    /// pixel scale is read as frame 18 when its true frame is 33 — and this window
    /// is what rescues it: with a radius that reaches frame 33 the band's own
    /// energy picks it out and the texture comes back, with no visible halo, since
    /// only the detail bands may wander while colour and brightness always follow
    /// the depth. On the reference stack widening this from 3 to 16 took the
    /// backdrop's edge acutance from 12.9 to 16.6 and the flat-backdrop detail
    /// from 29.1 to 31.9 (the reference export scores 19.1 and 28.3), and the
    /// flat-region noise did not move at all — the noise that a wide window used
    /// to add was an artefact of the depth field being wrong, not of the window.
    #[arg(long, default_value_t = 16)]
    bracket_radius: usize,

    /// Half-range, in full-resolution pixels, of the search for the
    /// depth-dependent residual shift; 0 (the default) turns it off.
    ///
    /// Refocusing moves the entrance pupil and changes magnification, so the
    /// image of a scene point moves by an amount that depends on how far away it
    /// is, and one similarity transform cannot satisfy every distance at once.
    /// That the effect exists is easy to see on the raw frames: between the ends
    /// of the sweep the feet need about 10 px more shift than the face. Almost
    /// all of it, though, is the global magnification change that the alignment
    /// above already removes. Measured *after* it, the depth-dependent part is
    /// only 1-2 px, and correcting it moves the fine detail the fusion holds by
    /// less than the run-to-run noise (bottom band keeps 0.985 of the per-frame
    /// best either way). It is therefore off by default, and worth trying when
    /// the residual really is large: high magnification, or a lens whose pupil
    /// travels a long way between the near and the far end.
    #[arg(long, default_value_t = 0)]
    parallax: usize,

    /// Blocks per axis used to sample the residual shift field.
    #[arg(long, default_value_t = 16)]
    parallax_blocks: usize,

    /// Candidate-window radius used for the coarsest detail levels. The wide
    /// window exists to rescue fine detail whose depth was misread; at the
    /// coarsest detail scales it does the opposite, because that is where a
    /// neighbouring surface's defocused bloom lives and the maximum there picks
    /// the bloom. Below a silhouette that is a band of wall brighter than any
    /// wall the frames contain. See `coarse_levels`.
    #[arg(long, default_value_t = 0)]
    bracket_radius_coarse: usize,

    /// How many of the coarsest detail levels use `--bracket-radius-coarse`.
    ///
    /// The band outside a silhouette has two causes and this is one of them.
    /// Measured against a ground truth that takes the wall's own focus frame with
    /// the figure excluded from the measuring window (`tools/depthtruth.py`), the
    /// wall 4..24 px outside the figure came out +2.36 levels brighter than that
    /// truth (at 8 px, +4.4). A candidate window that is wide at *the coarsest*
    /// levels is what puts the neighbour's defocused bloom into the maximum; how
    /// much the depth field itself contributes is `--edge-aware`'s business.
    /// Neither alone is enough: with only `--edge-aware` the band is +3.27 and
    /// with only this +2.36, against +0.22 with both. At 6 the flat backdrop
    /// loses a tenth of its acutance, so 4.
    ///
    /// The rescue the wide window earns lives in the *fine* levels: they keep
    /// `--bracket-radius` in full.
    #[arg(long, default_value_t = 4)]
    coarse_levels: usize,

    /// Soft-threshold strength for the fused detail bands, in units of the
    /// estimated per-band noise sigma. Off by default: per-pixel winner
    /// selection is a maximum statistic and does inflate noise in flat areas,
    /// but any threshold large enough to fix that also erases the genuine micro
    /// texture of smooth surfaces (polished resin, painted plastic), which reads
    /// as blur. Raise it for a noisy stack; the tool measures the trade-off.
    #[arg(long, default_value_t = 0.0)]
    denoise: f32,

    /// Focus measure used to build the depth field.
    #[arg(long, value_enum, default_value_t = FocusMeasure::Sml)]
    focus: FocusMeasure,

    /// How RGB is reduced to the single channel the measure runs on.
    ///
    /// Keep this fixed across the stack unless you have a reason not to: the
    /// measure only ranks frames correctly if the conversion is the same for
    /// every one of them.
    #[arg(long, value_enum, default_value_t = GrayMode::Luma)]
    gray: GrayMode,

    /// Pyramid levels; 0 = choose automatically from the image size.
    #[arg(short, long, default_value_t = 0)]
    levels: usize,

    /// Downsample factor for the depth analysis (1, 2 or 4). 1 is slowest and
    /// most precise, 4 is fastest.
    ///
    /// The default is 1, i.e. the depth is analysed at full resolution, and that
    /// is not a luxury. Defocus shows up first in the finest detail present, and
    /// on this stack a good deal of it — the relief of the backdrop, the grain of
    /// a resin surface — sits at the pixel scale. Downsampling by two averages
    /// exactly that away before the focus measure ever sees it, and the measure
    /// then has almost no dynamic range left: measured on one backdrop pixel the
    /// true sharpness changes by a factor of 7 across the stack while the
    /// half-resolution response changes by 1.3, with its peak at the wrong frame.
    /// Keeping the cache at 16 bits costs about 65 MB per 33 MP frame, so a
    /// 50-frame stack holds roughly 3 GB of grayscale in addition to the fusion
    /// buffers.
    #[arg(long, default_value_t = 1)]
    analysis_scale: usize,

    /// Window radius of the focus measure, in *full resolution* pixels (it is
    /// divided by the analysis scale internally, so the window keeps its physical
    /// size if you change that).
    ///
    /// **8, not 16.** Where a focus-measure window straddles a hard edge the
    /// edge's response dominates the sum, so the depth step lands about one
    /// radius *inside* the far surface: a pixel of the far surface that close to
    /// the object gets the object's depth and is then rendered from a frame in
    /// which the far surface is defocused, which is exactly the smooth band that
    /// appears along the far side of a silhouette. The displacement shrinks with
    /// the radius — measured on the reference stack, the relief recovered inside
    /// that band went 5.35 at radius 16, 7.34 at 8 and 7.73 at 4, against 10.84
    /// for the reference export — while the cost is negligible (global gradient
    /// energy 6.438 / 6.428 / 6.423). Larger radii were measured earlier and
    /// found neutral; smaller ones had simply never been tried.
    #[arg(long, default_value_t = 4)]
    focus_radius: usize,

    /// Divisor for the coarse prior of the depth search. 4 measures the prior at
    /// a quarter of the analysis resolution, where each sample averages 16 pixels
    /// of the full-resolution response and therefore has the dynamic range the
    /// per-pixel response lacks. 1 disables the coarse stage.
    #[arg(long, default_value_t = 4)]
    coarse_scale: usize,

    /// Focus window radius, in full-resolution pixels, for an optional second
    /// depth pass.
    ///
    /// The window the focus measure is summed over sets how wide the depth
    /// transition is at an object boundary: at radius 16 a pixel just inside a
    /// silhouette still has the backdrop inside its window, so its response peaks
    /// between the two surfaces and the depth ends up as a ramp tens of pixels
    /// wide. Inside that ramp the fusion's base band — colour and brightness —
    /// comes from a frame in which neither surface is sharp, and that is the
    /// seam halo: the object's defocused bloom showing outside its own detail.
    ///
    /// A small window tracks the silhouette closely, but on its own it is far too
    /// noisy on low-contrast surfaces. Constrained to a narrow band around the
    /// first pass (`--refine-window`) it only has to place the transition, not
    /// find the surface. 0 disables the second pass.
    #[arg(long, default_value_t = 0)]
    refine_radius: usize,

    /// Depth range, in frames, across a local window that marks a pixel as
    /// sitting on a silhouette between two surfaces. 0 disables the replacement
    /// and leaves the second pass confined to the prior band (see
    /// `--refine-radius`). Measured on this stack: about 29 frames across the
    /// leg/backdrop silhouette against about 6 across the object's own curvature.
    #[arg(long, default_value_t = 0.0)]
    edge_refine: f32,

    /// Subsampling factor used when scanning for silhouettes; see
    /// `depth::discontinuity_mask`.
    #[arg(long, default_value_t = 4)]
    edge_sub: usize,

    /// Radius, in analysis pixels, of the window the silhouette scan looks over.
    #[arg(long, default_value_t = 6)]
    edge_radius: usize,

    /// How much the search window around the coarse prior opens per frame of the
    /// prior's own local depth range. 0 restores the fixed window. See
    /// `depth::prior_window`: a fixed window cannot recover where the coarse
    /// prior's boundary was contaminated by the edge it straddles, which is what
    /// leaves a smooth band along the far side of a silhouette.
    #[arg(long, default_value_t = 1.0)]
    prior_window_gain: f32,

    /// Radius, in analysis pixels, over which that local range is measured.
    #[arg(long, default_value_t = 8)]
    prior_window_radius: usize,

    /// Half-width, in frames, of the per-pixel search around the prior. With
    /// `--coarse-scale` it bounds the search around the coarse prior; with
    /// `--refine-radius` it bounds the small-window second pass around the first
    /// pass's result. 0 disables both restrictions.
    #[arg(long, default_value_t = 8.0)]
    refine_window: f32,

    /// Extra frames the fusion's candidate window may open by, per unit of local
    /// depth uncertainty (see `FuseOptions::slack`). 0 gives every pixel the same
    /// window. Needs `--refine-window` > 0, which is what defines the uncertainty.
    #[arg(long, default_value_t = 0.0)]
    adaptive_window: f32,

    /// How fast the candidate window closes where the depth field changes,
    /// in frames per pixel. 0 disables it. See `FuseOptions::seam_window`: this
    /// is what removes the halo at an object boundary, where the depth is a
    /// short ramp rather than a step.
    #[arg(long, default_value_t = 0.0)]
    seam_window: f32,

    /// Radius of the edge-aware aggregation applied to the focus response before
    /// the argmax. 0 disables it. This is the standard cost-volume filtering step
    /// (Rhemann et al., Fast Cost-Volume Filtering, 2013): the response is a
    /// cost volume over frames, and aggregating it with weights taken from a
    /// fixed guide stops a window from averaging across a surface boundary.
    /// See the comment at the call site for why the guide is the winner image.
    ///
    /// Default 16 (2026-09-22). This is the fix for the band along the far side of
    /// a silhouette; it had been implemented and left off. Measured against a
    /// ground truth that excludes the figure from the focus window: the depth
    /// 2-8 px outside the silhouette was 5.0 frames too near without it and 0.6
    /// with it, and the wall there came out +2.36 levels too bright, against
    /// +0.22 with it (+4.4 at 8 px against +0.9). Five of the six reference
    /// regions get sharper; the flat backdrop loses 0.9%. It does not replace
    /// `--coarse-levels`: with the depth correct but the window wide, the band
    /// comes back at +3.27.
    #[arg(long, default_value_t = 16)]
    edge_aware: usize,

    /// Regularisation of that aggregation; larger smooths more within a surface.
    #[arg(long, default_value_t = 1e-4)]
    edge_eps: f32,

    /// Do not clamp the fused result back into the per-pixel range spanned by
    /// the frames. Only useful for reproducing the pre-fix behaviour, where a
    /// Laplacian overshoot at a high-contrast edge left pixels darker than every
    /// frame in the stack.
    #[arg(long, default_value_t = false)]
    no_clamp_range: bool,

    /// Gate the per-band winner by how *smooth the depth is locally*, so that a
    /// band ignores the winner where the depth is wandering. Per-band maximum
    /// leaves mottle on a surface with no focus signal (each band picks a
    /// different frame); falling back to the depth interpolation there removes it
    /// without touching a silhouette, where the depth is a clean step.
    ///
    /// The focus response's own peak-to-mean was tried first and does not work:
    /// on the plane the fusion consumes, the silhouette band scores 1.67 and the
    /// subject's smooth plastic 2.34 — backwards. See the region dump below.
    #[arg(long, default_value_t = false)]
    conf_band: bool,

    /// Depth roughness `1/(1+|d - median5x5(d)|)` below which a band ignores the
    /// winner entirely.
    #[arg(long, default_value_t = 0.2)]
    conf_lo: f32,

    /// Reliability above which a band uses the winner in full.
    #[arg(long, default_value_t = 0.5)]
    conf_hi: f32,

    /// Skip the *lower* side of that clamp. Not recommended: that side is what
    /// removes the dark fringes of bug 6.
    #[arg(long, default_value_t = false)]
    no_clamp_lo: bool,

    /// Skip clamping the *upper* side of that range. The upper side is on by
    /// default: without it the reconstruction lands up to +15 levels above what
    /// any frame holds at that pixel, which is the bright rim outside a
    /// silhouette (bug 9). Note the bound is itself built from the warped frames
    /// and so inherits their defects, which is why clamping removes the bright
    /// band but also cuts relief; see `--bracket-radius-coarse` for the fix that
    /// removes the cause instead.
    #[arg(long = "no-clamp-hi", default_value_t = false)]
    no_clamp_hi: bool,

    /// Slack, in levels, left on both sides of the clamp.
    #[arg(long, default_value_t = 0.0)]
    clamp_slack: f32,

    /// Radius by which the upper clamp bound is dilated before clamping. The
    /// per-pixel maximum over the frames is noisy at the finest scale, so
    /// clamping to it raw cuts into real relief (measured: the silhouette
    /// band's texture fell to 5.4, against 8.2 with the winner used raw and 7.2
    /// for the reference). Dilating asks "did any frame, anywhere within this
    /// radius, reach this value" — which is the right question for a bound on
    /// content, and still excludes a bright rim that no frame comes near.
    #[arg(long, default_value_t = 0)]
    clamp_hi_dilate: usize,

    /// Radius, in pixels, over which the colour a nearer surface spills onto
    /// whatever stands behind it is pulled back to that surface's own colour.
    /// 0 (the default) leaves it alone.
    ///
    /// Defocus ignores silhouettes: a coloured object in front of a wall spreads
    /// its light onto that wall, so the wall just outside the outline carries a
    /// wash of the object's hue, 15-20 px wide. It is real - every frame of the
    /// reference stack carries more of it than the fusion does - but it reads as
    /// the object glowing onto the wall, and the hand-retouched reference has
    /// removed it.
    ///
    /// The judgement is made on the depth field, never on colour, so the surface
    /// behind may be anything. Only low-frequency colour is touched.
    #[arg(long, default_value_t = 0)]
    spill: usize,

    /// Write the per-pixel frame range the clamp enforces, as `<prefix>_lo.png`
    /// and `<prefix>_hi.png` (16-bit, value x256). This is the only authority on
    /// whether a value in the output is one that no frame contains.
    #[arg(long)]
    save_range: Option<String>,

    /// Exponent of the energy weighting used to combine the candidate frames at
    /// each frequency band. 0 = plain maximum (the sharpest candidate wins,
    /// which also takes the maximum of the noise in flat areas). 3-5 keeps the
    /// winner wherever one candidate genuinely dominates — a real edge or a
    /// resolved texture — while averaging comparable candidates, which is the
    /// no-signal case, shrinking its noise. See `fuse::power_mean`.
    #[arg(long, default_value_t = 0.0)]
    band_power: f32,

    /// Radius over which a band's energy is aggregated *before* its winner is
    /// chosen. 0 decides the winner per pixel, as before.
    ///
    /// The winner is a per-pixel maximum, so in a band that holds no signal the
    /// argmax follows the candidates' noise and flips between frames from one
    /// pixel to the next. A band assembled from different frames per pixel places
    /// several independent noise realisations side by side — the grain visible on
    /// a smooth subject surface, which `--band-power` removes only by giving up
    /// detail. Aggregating the energy first makes the choice coherent over the
    /// neighbourhood, so the band comes from one frame locally and carries that
    /// frame's own micro-texture; where a real signal exists it survives the
    /// aggregation, because the sharp frame wins over an area and not at a pixel.
    /// A different trade-off axis from `--band-power`, and measured better on
    /// this stack: see README bug 8.
    #[arg(long, default_value_t = 2)]
    band_energy_smooth: usize,

    /// Runner-up/best energy ratio above which the per-pixel winner counts as a
    /// coin toss and the neighbourhood-coherent winner takes over, with a ramp
    /// of +-0.15 around it. Aggregating everywhere also overrides pixels where
    /// the winner was decisive and right, which softens real texture — measured
    /// on the reference stack the backdrop relief lost about 15% of its
    /// high-frequency energy. This keeps the aggregation to the pixels that
    /// actually need it. 0 disables the discrimination (aggregate everywhere).
    #[arg(long, default_value_t = 0.7)]
    band_energy_gate: f32,

    /// Blur the analysis image by this radius before measuring focus. Sensor
    /// noise lives at the single-pixel scale while defocus blur does not, so a
    /// small prefilter makes the focus measure usable on smooth low-contrast
    /// surfaces (polished plastic, painted resin) that would otherwise be
    /// noise-dominated. 0 disables it.
    #[arg(long, default_value_t = 1)]
    focus_prefilter: usize,

    /// Skip geometric alignment (use when the stack is already registered).
    #[arg(long)]
    no_align: bool,

    /// Long side of the working resolution used by the alignment search.
    #[arg(long, default_value_t = 900)]
    align_work_size: usize,

    /// Half-width of the scale search, e.g. 0.02 searches 0.98 .. 1.02.
    #[arg(long, default_value_t = 0.02)]
    align_scale_range: f32,

    /// Number of scale candidates tested.
    #[arg(long, default_value_t = 16)]
    align_scale_steps: usize,

    /// Low-pass radius applied to the alignment working images. Registration
    /// must key on geometry, but its features (gradient magnitude) also change
    /// with focus, so without this the scale search matches blur instead of
    /// position — see the note on `AlignOptions::focus_blur`. 0 restores the old
    /// behaviour and is only useful for demonstrating the difference.
    #[arg(long, default_value_t = 3)]
    align_focus_blur: usize,

    /// Do not crop the result to the area covered by every frame.
    #[arg(long)]
    no_crop: bool,

    /// Extra pixels trimmed from the auto-computed crop.
    #[arg(long, default_value_t = 2)]
    margin: usize,

    /// Explicit crop in full resolution pixels as x,y,w,h.
    #[arg(long)]
    crop: Option<String>,

    /// Spatial radius of the depth-similarity (bilateral) smoother. 0 uses the
    /// guided filter instead. See `depth::bilateral_depth`.
    #[arg(long, default_value_t = 6)]
    depth_bilateral: usize,

    /// Range sigma of that smoother, in frames. Neighbours further than a few
    /// frames away in depth do not contribute, which is what stops it from
    /// spreading a depth step.
    #[arg(long, default_value_t = 1.5)]
    depth_sigma: f32,

    /// Radius of the median despeckling applied to the raw depth field. A mean
    /// filter spreads an object boundary over its whole window, which makes the
    /// blend band around a foreground edge pick intermediate frames that are
    /// sharp for neither side; a median does not.
    #[arg(long, default_value_t = 2)]
    depth_median: usize,

    /// Guided-filter radius used to snap the depth field to image edges. Kept
    /// small on purpose: the median already removed the speckle, so this only
    /// has to align boundaries, and a large radius would soften them again.
    #[arg(long, default_value_t = 6)]
    guided_radius: usize,

    /// Guided-filter regularisation; smaller keeps more edges.
    #[arg(long, default_value_t = 0.001)]
    guided_eps: f32,

    /// Number of guided-filter passes over the depth field.
    #[arg(long, default_value_t = 2)]
    depth_smooth: usize,

    /// Extra aggregation radius used only for judging whether a pixel carries
    /// focus information. Reliability is a peak-to-mean ratio, and on a single
    /// pixel that ratio is dominated by residual noise — so much so that a
    /// featureless region scores *higher* than a textured one. Aggregating the
    /// response further, without touching the map the argmax uses, turns the
    /// ratio into a clean discriminator.
    ///
    /// In full resolution pixels; divided by the analysis scale internally.
    #[arg(long, default_value_t = 32)]
    conf_smooth: usize,


    /// Lower bound on how much the per-pixel depth estimate is trusted, in
    /// `[0, 1]`. 1 keeps every per-pixel estimate and uses the smoothed field
    /// only where the median could not resolve anything. Raise it when smooth
    /// surfaces come out soft: a weak but correctly placed focus peak still
    /// points at the right frame, and blending it away with a neighbourhood
    /// sitting at a different distance is exactly what blurs those surfaces.
    #[arg(long, default_value_t = 0.0)]
    depth_trust: f32,

    /// Force the frame chosen inside a rectangle, given as `x,y,w,h,frame` in
    /// full resolution output coordinates. Repeatable.
    ///
    /// A large smooth object in front of a strongly textured backdrop is the one
    /// case this tool cannot solve on its own: the object carries no focus
    /// information in its interior, and the only cue — its silhouette — has its
    /// focus measure swamped by the backdrop texture sitting right next to it,
    /// so the wrong frame is selected along the edge and inherited inward.
    /// Professional tools expose depth-map retouching for exactly this reason;
    /// this is the same escape hatch.
    #[arg(long = "depth-fix", value_name = "X,Y,W,H,FRAME")]
    depth_fix: Vec<String>,

    /// Skip the per-frame brightness matching.
    #[arg(long)]
    no_match_exposure: bool,

    /// Write an extra downscaled JPEG for quick inspection.
    #[arg(long)]
    preview: Option<PathBuf>,

    /// Long side of the preview image.
    #[arg(long, default_value_t = 1600)]
    preview_size: usize,

    /// Worker threads (0 = all cores).
    #[arg(long, default_value_t = 0)]
    threads: usize,

    /// Do not write the 8-bit companions of the output.
    #[arg(long)]
    quiet: bool,

    /// Dump the computed depth field as a 16-bit PNG (analysis resolution).
    /// Useful for diagnosing localised softness: the value is the frame index,
    /// scaled to the full 16-bit range.
    #[arg(long)]
    save_depth: Option<PathBuf>,

    /// Dump the focus-confidence map as an 8-bit PNG (analysis resolution).
    #[arg(long)]
    save_conf: Option<PathBuf>,

    /// Debug: print the per-frame focus response at one pixel, both for the
    /// frame in its own coordinates and for the relocated response field.
    /// Coordinates are given in aligned-canvas pixels. Used to tell "the
    /// per-frame values are not comparable" apart from "the transform is wrong".
    #[arg(long)]
    debug_point: Option<String>,

    /// Debug: print the depth profile along one canvas row at every stage of the
    /// post-processing chain, to find which stage moves a depth step away from
    /// the image edge. Format `Y,X0,X1` in aligned-canvas pixels, sampled every
    /// 4 columns.
    #[arg(long)]
    depth_trace: Option<String>,

    /// Keep frames whose pixel dimensions differ from the majority. Useful only
    /// for deliberately mixed stacks; off by default because a folder often
    /// also holds exports/retouches that must not be treated as frames.
    #[arg(long)]
    all_sizes: bool,
}

const IMAGE_EXTS: [&str; 6] = ["jpg", "jpeg", "png", "tif", "tiff", "bmp"];

fn is_image(p: &Path) -> bool {
    p.extension()
        .and_then(|e| e.to_str())
        .map(|e| IMAGE_EXTS.contains(&e.to_ascii_lowercase().as_str()))
        .unwrap_or(false)
}

/// Natural ordering so that `DSC9.JPG` sorts before `DSC10.JPG`.
fn natural_key(p: &Path) -> Vec<(u8, u64, String)> {
    let name = p
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("")
        .to_string();
    let mut key = Vec::new();
    let mut num = String::new();
    let mut txt = String::new();
    for ch in name.chars() {
        if ch.is_ascii_digit() {
            if !txt.is_empty() {
                key.push((1u8, 0u64, std::mem::take(&mut txt)));
            }
            num.push(ch);
        } else {
            if !num.is_empty() {
                key.push((0u8, num.parse::<u64>().unwrap_or(0), String::new()));
                num.clear();
            }
            txt.push(ch);
        }
    }
    if !num.is_empty() {
        key.push((0u8, num.parse::<u64>().unwrap_or(0), String::new()));
    }
    if !txt.is_empty() {
        key.push((1u8, 0u64, txt));
    }
    key
}

fn collect_inputs(inputs: &[PathBuf]) -> Result<Vec<PathBuf>> {
    let mut files = Vec::new();
    for inp in inputs {
        if inp.is_dir() {
            let mut here: Vec<PathBuf> = std::fs::read_dir(inp)
                .with_context(|| format!("cannot read directory {}", inp.display()))?
                .filter_map(|e| e.ok())
                .map(|e| e.path())
                .filter(|p| p.is_file() && is_image(p))
                .collect();
            here.sort_by_key(|p| natural_key(p));
            files.extend(here);
        } else if inp.is_file() {
            files.push(inp.clone());
        } else {
            bail!("input does not exist: {}", inp.display());
        }
    }
    if files.is_empty() {
        bail!("no input images found");
    }
    files.sort_by_key(|p| natural_key(p));
    files.dedup();
    Ok(files)
}

/// Drop frames whose pixel size differs from the majority.
///
/// A stack folder very often also contains a finished export, a retouched
/// version, or a downscaled web copy. Silently treating those as bracketing
/// frames would either fail (mismatched sizes) or quietly poison the result,
/// so the odd ones out are removed and reported.
fn filter_by_size(files: Vec<PathBuf>, exclude: Option<&Path>, quiet: bool) -> Result<Vec<PathBuf>> {
    use std::collections::HashMap;
    let mut sized: Vec<(PathBuf, (u32, u32))> = Vec::with_capacity(files.len());
    for p in files {
        if let Some(ex) = exclude {
            if let (Ok(a), Ok(b)) = (std::fs::canonicalize(&p), std::fs::canonicalize(ex)) {
                if a == b {
                    continue;
                }
            }
        }
        match image::image_dimensions(&p) {
            Ok(d) => sized.push((p, d)),
            Err(_) => {
                if !quiet {
                    eprintln!("  skipping unreadable {}", p.display());
                }
            }
        }
    }
    if sized.is_empty() {
        bail!("no readable images found");
    }
    let mut counts: HashMap<(u32, u32), usize> = HashMap::new();
    for (_, d) in &sized {
        *counts.entry(*d).or_insert(0) += 1;
    }
    let majority = counts
        .iter()
        .max_by_key(|(_, c)| **c)
        .map(|(d, _)| *d)
        .unwrap();
    let mut kept = Vec::new();
    for (p, d) in sized {
        if d == majority {
            kept.push(p);
        } else if !quiet {
            eprintln!(
                "  ignoring {} ({}x{}, majority is {}x{})",
                p.file_name().and_then(|s| s.to_str()).unwrap_or("?"),
                d.0,
                d.1,
                majority.0,
                majority.1
            );
        }
    }
    Ok(kept)
}

/// Decode a file to floating point RGB in `[0, 1]`.
fn load_rgb(path: &Path) -> Result<RgbImage> {
    let img = image::open(path).with_context(|| format!("cannot open {}", path.display()))?;
    let rgb = img.to_rgb8();
    let (w, h) = (rgb.width() as usize, rgb.height() as usize);
    let mut d = vec![0.0f32; w * h * 3];
    d.par_chunks_mut(4096 * 3)
        .enumerate()
        .for_each(|(ci, chunk)| {
            let base = ci * 4096 * 3;
            let src = rgb.as_raw();
            for (i, o) in chunk.iter_mut().enumerate() {
                *o = src[base + i] as f32 * (1.0 / 255.0);
            }
        });
    Ok(RgbImage { w, h, d })
}

/// How a frame is reduced to the single channel the focus measure works on.
#[derive(Copy, Clone, Debug, PartialEq, Eq, clap::ValueEnum)]
enum GrayMode {
    /// Fixed Rec.601 weights, identical for every frame.
    Luma,
    /// Per-frame principal-component projection of the RGB values. Looks
    /// attractive (it maximises contrast) but is a per-frame normalisation, and
    /// the measure then ranks frames partly on the gain it applied — see
    /// [`crate::util::RgbImage::to_grayscale_luma`].
    Pca,
}

/// Cached, reduced resolution grayscale copy of every frame.
///
/// Stored as 16-bit: `f32` planes would need 6.5 GB for a 33 MP stack at
/// analysis scale 1, but plain 8-bit quantisation is *not* enough — the focus
/// measure on a smooth surface produces per-pixel responses of the same order
/// as an 8-bit step, so the depth estimate degrades measurably. 16-bit keeps
/// the precision at a quarter of the `f32` cost.
struct GrayCache {
    frames: Vec<Vec<u16>>,
    full_w: usize,
    full_h: usize,
    cache_w: usize,
    cache_h: usize,
}

impl GrayCache {
    fn build(
        paths: &[PathBuf],
        factor: usize,
        gray_mode: GrayMode,
        quiet: bool,
    ) -> Result<(Self, Vec<[f32; 3]>)> {
        let mut frames = Vec::with_capacity(paths.len());
        let mut means = Vec::with_capacity(paths.len());
        let mut dims = (0usize, 0usize);
        for (i, p) in paths.iter().enumerate() {
            let rgb = load_rgb(p)?;
            let gray = match gray_mode {
                GrayMode::Luma => rgb.to_grayscale_luma(),
                GrayMode::Pca => rgb.to_grayscale_pca(),
            };
            let mut m = [0.0f32; 3];
            let px = rgb.len_pixels() as f32;
            for v in rgb.d.chunks_exact(3) {
                m[0] += v[0];
                m[1] += v[1];
                m[2] += v[2];
            }
            for c in m.iter_mut() {
                *c /= px;
            }
            means.push(m);
            let (fw, fh) = (gray.w, gray.h);
            let small = if factor <= 1 {
                gray
            } else {
                align::resize(&gray, fw / factor, fh / factor)
            };
            dims = (fw, fh);
            if !quiet {
                eprintln!(
                    "  read {}/{} {} ({}x{})",
                    i + 1,
                    paths.len(),
                    p.file_name().and_then(|s| s.to_str()).unwrap_or("?"),
                    small.w,
                    small.h
                );
            }
            frames.push(
                small.d.iter().map(|v| (v.clamp(0.0, 1.0) * 65535.0) as u16).collect(),
            );
        }
        let (cache_w, cache_h) = (dims.0 / factor.max(1), dims.1 / factor.max(1));
        Ok((
            Self { frames, full_w: dims.0, full_h: dims.1, cache_w, cache_h },
            means,
        ))
    }

    /// Materialise one cached frame as a `[0, 1]` plane of the cached size.
    fn plane(&self, i: usize) -> Plane {
        Plane {
            w: self.cache_w,
            h: self.cache_h,
            d: self.frames[i].iter().map(|v| *v as f32 * (1.0 / 65535.0)).collect(),
        }
    }
}

fn parse_crop(s: &str) -> Option<Rect> {
    let parts: Vec<usize> = s
        .split(',')
        .filter_map(|t| t.trim().parse::<usize>().ok())
        .collect();
    if parts.len() == 4 {
        Some(Rect { x: parts[0], y: parts[1], w: parts[2], h: parts[3] })
    } else {
        None
    }
}

fn main() -> Result<()> {
    let args = Cli::parse();
    if args.threads > 0 {
        rayon::ThreadPoolBuilder::new()
            .num_threads(args.threads)
            .build_global()
            .ok();
    }
    let t_total = Instant::now();
    let quiet = args.quiet;
    let say = |s: &str| {
        if !quiet {
            eprintln!("{s}");
        }
    };

    let files = collect_inputs(&args.inputs)?;
    let files = if args.all_sizes {
        files
    } else {
        filter_by_size(files, Some(&args.output), quiet)?
    };
    if files.len() < 2 {
        bail!("need at least two frames, got {}", files.len());
    }
    say(&format!("mukastack: {} frames", files.len()));

    // ---------------------------------------------------------------- stage 1
    let t = Instant::now();
    let factor = args.analysis_scale.max(1);
    let (cache, means) = GrayCache::build(&files, factor, args.gray, quiet)?;
    say(&format!(
        "[1/4] decoded and cached grayscale at 1/{} ({}x{}) in {:.1}s",
        factor,
        cache.cache_w,
        cache.cache_h,
        t.elapsed().as_secs_f32()
    ));

    // ---------------------------------------------------------------- stage 2
    let t = Instant::now();
    let mid = cache.frames.len() / 2;
    let mut transforms_full: Vec<Transform> = if args.no_align {
        vec![Transform::default(); files.len()]
    } else {
        let opts = AlignOptions {
            work_size: args.align_work_size,
            scale_range: args.align_scale_range,
            scale_steps: args.align_scale_steps,
            highpass: 4,
            focus_blur: args.align_focus_blur,
            fine_range: 2.0 * args.align_scale_range / args.align_scale_steps.max(1) as f32,
            fine_steps: 10,
        };
        let aligner = Aligner::new(&cache.plane(mid), opts);
        let res: Vec<(Transform, f32)> = (0..files.len())
            .into_par_iter()
            .map(|i| {
                if i == mid {
                    (Transform::default(), 1.0)
                } else {
                    aligner.estimate(&cache.plane(i))
                }
            })
            .collect();
        if !quiet {
            for (i, (tr, score)) in res.iter().enumerate() {
                if i == mid || i % 5 == 0 {
                    eprintln!(
                        "  frame {:3} scale={:.4} dx={:+.2} dy={:+.2} score={:.4}",
                        i, tr.scale, tr.dx, tr.dy, score
                    );
                }
            }
        }
        // Transforms are in cached-resolution pixels; convert to full resolution.
        res.iter()
            .map(|(tr, _)| Transform {
                scale: tr.scale,
                dx: tr.dx * factor as f32,
                dy: tr.dy * factor as f32,
                ..Default::default()
            })
            .collect()
    };
    say(&format!("[2/4] alignment done in {:.1}s", t.elapsed().as_secs_f32()));

    // The output window is needed early because `--depth-fix` coordinates are
    // given in output pixels.
    let rect = if let Some(c) = &args.crop {
        parse_crop(c).context("--crop must be x,y,w,h")?
    } else if args.no_crop {
        Rect::full(cache.full_w, cache.full_h)
    } else {
        valid_rect(&transforms_full, cache.full_w, cache.full_h, args.margin)
    };

    say(&format!(
        "[crop] output window x={} y={} w={} h={}  (canvas {}x{})",
        rect.x, rect.y, rect.w, rect.h, cache.cache_w, cache.cache_h
    ));

    // ---------------------------------------------------------------- stage 3
    let t = Instant::now();
    let (aw, ah) = (cache.cache_w, cache.cache_h);
    let dbg_pt: Option<(usize, usize)> = match &args.debug_point {
        Some(s) => {
            let (a, b) = s.split_once(',').context("--debug-point must be x,y")?;
            Some((
                a.trim().parse::<usize>().context("debug x")? / factor,
                b.trim().parse::<usize>().context("debug y")? / factor,
            ))
        }
        None => None,
    };
    let transforms_cache: Vec<Transform> = transforms_full
        .iter()
        .map(|t| Transform { scale: t.scale, dx: t.dx / factor as f32, dy: t.dy / factor as f32, ..Default::default() })
        .collect();

    // Window sizes are given in full-resolution pixels so that changing the
    // analysis scale cannot silently change the physical size of the window.
    let focus_radius = (args.focus_radius / factor).max(1);
    let conf_smooth =
        if args.conf_smooth == 0 { 0 } else { (args.conf_smooth / factor).max(1) };

    let mut builder = DepthBuilder::new(aw, ah, files.len());

    // Coarse prior.
    //
    // The per-pixel focus response has almost no dynamic range where the content
    // is low-contrast, so there the argmax is decided by noise: on the reference
    // stack a textured backdrop pixel's response varies by 1.3x across the whole
    // stack while its true sharpness varies by 7x, which is why its depth came
    // out at 18 instead of 33. The very same response, averaged over a
    // quarter-resolution cell, does peak at the right frame. So the depth is
    // measured twice: coarse first, then per pixel but confined to a window
    // around the coarse answer, which keeps the precision of the per-pixel
    // estimate wherever it is meaningful without letting it invent a depth where
    // it is not.
    let prior: Option<Plane> = if args.refine_window > 0.0 && args.coarse_scale > 1 {
        let cs = args.coarse_scale;
        let (cw, ch) = ((aw + cs - 1) / cs, (ah + cs - 1) / cs);
        let tr_c: Vec<Transform> = transforms_cache
            .iter()
            .map(|t| Transform { scale: t.scale, dx: t.dx / cs as f32, dy: t.dy / cs as f32, ..Default::default() })
            .collect();
        let mut cb = DepthBuilder::new(cw, ch, files.len());
        for k in 0..files.len() {
            let g = cache.plane(k);
            let small = align::resize(&g, cw, ch);
            let pre = if args.focus_prefilter > 0 {
                util::box_blur(&small, (args.focus_prefilter + cs - 1) / cs)
            } else {
                small
            };
            let fmf = focus_map(&pre, args.focus, (focus_radius + cs - 1) / cs);
            let fm = align::warp(&fmf, &tr_c[k], cw, ch, 0.0);
            let fmc = fm.clone();
            cb.push(k, &fm, &fmc);
        }
        let (d, _, _) = cb.finish();
        let p = upsample_depth(&d, aw, ah);
        say(&format!("  coarse prior at 1/{} ({}x{})", cs, cw, ch));
        Some(p)
    } else {
        None
    };
    if let Some(p) = &prior {
        builder.set_prior(p, args.refine_window);
        if args.prior_window_gain > 0.0 {
            let win = depth::prior_window(
                p,
                args.refine_window,
                args.prior_window_gain,
                args.prior_window_radius,
            );
            builder.set_prior_window(&win);
        }
    }
    let mut guide = vec![0.0f32; aw * ah];
    // Guide for the edge-aware cost aggregation below: per pixel, the value from
    // whichever frame currently has the strongest response. It has sharp edges
    // *everywhere*, which the stack mean does not — at a depth discontinuity each
    // frame has that boundary either sharp or blurred, so the mean is soft, and a
    // guided filter can only follow edges its guide actually has.
    let mut best_so_far = vec![f32::NEG_INFINITY; aw * ah];
    let mut winner_img = vec![0.0f32; aw * ah];
    // One frame resident at a time: at analysis scale 1 holding all 50 warped
    // planes would cost 6.5 GB.
    for k in 0..files.len() {
        let g = cache.plane(k);
        let wg = align::warp(&g, &transforms_cache[k], aw, ah, 0.0);
        // Sensor noise sits at the single-pixel scale; defocus blur does not.
        // A tiny prefilter therefore removes the noise floor from the focus
        // measure without touching the focus signal, which is what makes smooth
        // low-contrast surfaces (polished resin, painted plastic) solvable.
        //
        // The measure is taken on the *unwarped* frame and the resulting response
        // field is then moved into the canvas; the images themselves are still
        // warped for the guide and for the fusion.
        //
        // This ordering is essential, not cosmetic. Warping is a resampling whose
        // smoothing depends on that frame's transform — this stack breathes by
        // 0.75% across the focus sweep, so each frame carries a slightly
        // different scale — and a Laplacian-based measure is acutely sensitive to
        // that. Measured after warping, the per-frame values are no longer
        // comparable: whichever frames' transforms happened to sit closest to
        // identity score highest *whatever their actual focus was*. On this stack
        // that biased every low-signal pixel towards one end of the sweep, by up
        // to 30 frames — a face whose true frame is 16 was read as 32, a leaf at
        // 12 as 35 — which is exactly the "smooth surfaces come out blurry, the
        // textured backdrop stays sharp" failure. The response field is smooth
        // (it is box-summed over a radius-8 window), so relocating it costs
        // nothing, while the numbers it contains are now taken from unresampled
        // pixels and mean the same thing for every frame.
        let (gw, gh) = (g.w, g.h);
        let pre = if args.focus_prefilter > 0 {
            util::box_blur(&g, args.focus_prefilter)
        } else {
            g
        };
        let fm_frame = focus_map(&pre, args.focus, focus_radius);
        let fm = align::warp(&fm_frame, &transforms_cache[k], aw, ah, 0.0);
        if let Some((px, py)) = dbg_pt {
            if px < aw && py < ah {
                let t = &transforms_cache[k];
                // The frame-local map lives in the frame's own coordinates, so
                // the canvas point has to be mapped back to compare like with
                // like. Sampling it at the canvas coordinate instead would read a
                // different physical place in every frame.
                let (rx, ry) = t.inverse_map(
                    px as f32 + 0.5 - aw as f32 * 0.5,
                    py as f32 + 0.5 - ah as f32 * 0.5,
                );
                let (sx, sy) = (rx + gw as f32 * 0.5, ry + gh as f32 * 0.5);
                let local = if sx < 0.0 || sy < 0.0 || sx > gw as f32 - 1.0 || sy > gh as f32 - 1.0
                {
                    f32::NAN
                } else {
                    fm_frame.sample_bilinear(sx, sy)
                };
                eprintln!(
                    "DP f{:02} unwarped {:.7} relocated {:.7}  scale {:.4} dx {:+.3} dy {:+.3}  at {:+.2},{:+.2}",
                    k, local, fm.d[py * aw + px], t.scale, t.dx, t.dy, sx, sy
                );
            }
        }
        // Reliability is judged on a larger aggregation window than the one the
        // argmax uses: see DepthBuilder::sum_conf.
        let fm_conf = if conf_smooth > 0 {
            util::box_blur(&fm, conf_smooth)
        } else {
            fm.clone()
        };
        builder.push(k, &fm, &fm_conf);
        best_so_far
            .par_iter_mut()
            .zip(winner_img.par_iter_mut())
            .zip(fm.d.par_iter())
            .zip(wg.d.par_iter())
            .for_each(|(((bs, wi), f), w)| {
                if *f > *bs {
                    *bs = *f;
                    *wi = *w;
                }
            });
        guide.par_iter_mut()
            .zip(wg.d.par_iter())
            .for_each(|(a, b)| *a += *b);
    }
    let n = files.len() as f32;
    guide.par_iter_mut().for_each(|v| *v /= n);
    let (raw_depth, conf, known) = builder.finish();

    // Edge-aware cost aggregation.
    //
    // The window that produced `fm` is a plain box, so wherever it straddled a
    // hard edge the edge's far stronger response dominated the sum and the depth
    // step ended up about one window radius *inside* the far surface. A pixel of
    // the far surface that close to the object then takes the object's depth and
    // is rendered from a frame in which that surface is defocused — the smooth
    // band along the far side of a silhouette.
    //
    // Re-running the argmax on a response aggregated with weights taken from a
    // fixed guide cannot mix the two surfaces, so the step lands on the edge
    // itself. The weights depend only on the guide and never on the frame, which
    // is what keeps the invariant the rest of the pipeline rests on: nothing in
    // the measure may vary from frame to frame (see bug 4 in the README).
    let (raw_depth, conf, known) = if args.edge_aware > 0 {
        let t_ea = Instant::now();
        let guide_ea = Plane { w: aw, h: ah, d: winner_img };
        let r = (args.edge_aware / factor).max(1);
        let mut be = DepthBuilder::new(aw, ah, files.len());
        for k in 0..files.len() {
            let g = cache.plane(k);
            let pre = if args.focus_prefilter > 0 {
                util::box_blur(&g, args.focus_prefilter)
            } else {
                g
            };
            let fmf = focus_map(&pre, args.focus, focus_radius);
            let fm = align::warp(&fmf, &transforms_cache[k], aw, ah, 0.0);
            let agg = guided_filter(&fm, &guide_ea, r, args.edge_eps);
            let aggc = if conf_smooth > 0 {
                util::box_blur(&agg, conf_smooth)
            } else {
                agg.clone()
            };
            be.push(k, &agg, &aggc);
        }
        let (d, c, kn) = be.finish();
        say(&format!(
            "  edge-aware cost aggregation, radius {r}, in {:.1}s",
            t_ea.elapsed().as_secs_f32()
        ));
        (d, c, kn)
    } else {
        (raw_depth, conf, known)
    };

    // A pixel with no focus information gets its depth from the nearest pixel
    // that has one.
    //
    // On a smooth surface JPEG decoding yields perfectly flat runs of blocks, and
    // over such a run the focus response is identical — usually exactly zero —
    // for every frame. Every frame ties, the first one wins, and that fabricated
    // value then survives the median and is spread across an otherwise well
    // determined neighbourhood by the guided filter. It is worth repairing rather
    // than merely smoothing, because the fabricated value is not noise around the
    // truth but a systematic pull towards one end of the sweep.
    let share = |k: &Plane| k.d.iter().filter(|v| **v < 0.5).count() as f32 / k.d.len() as f32;

    // Optional second pass with a smaller focus window, confined to a narrow band
    // around the first pass's result. See `--refine-radius`.
    let (raw_depth, conf, _known, unknown_share) = if args.refine_radius > 0 {
        let t2 = Instant::now();
        let first = if share(&known) > 0.0 {
            fill_unreliable(&raw_depth, &known, 12)
        } else {
            raw_depth
        };
        let r2 = (args.refine_radius / factor).max(1);
        let mut b2 = DepthBuilder::new(aw, ah, files.len());
        // Either confine the second pass to a band around the first result, or
        // let it search freely where the depth is discontinuous. The confined
        // form cannot sharpen a transition (the prior is itself the ramp, so the
        // true depth of either surface lies outside the band); the free form can,
        // but only at a silhouette, where the measure has strong contrast — hence
        // the mask.
        if args.edge_refine <= 0.0 {
            b2.set_prior(&first, args.refine_window);
        }
        let mask = if args.edge_refine > 0.0 {
            Some(depth::discontinuity_mask(
                &first,
                args.edge_sub,
                args.edge_radius,
                args.edge_refine,
            ))
        } else {
            None
        };
        for k in 0..files.len() {
            let g = cache.plane(k);
            let pre = if args.focus_prefilter > 0 {
                util::box_blur(&g, args.focus_prefilter)
            } else {
                g
            };
            let fmf = focus_map(&pre, args.focus, r2);
            let fm = align::warp(&fmf, &transforms_cache[k], aw, ah, 0.0);
            let fmc = if conf_smooth > 0 {
                util::box_blur(&fm, conf_smooth)
            } else {
                fm.clone()
            };
            b2.push(k, &fm, &fmc);
        }
        let (d2, c2, k2) = b2.finish();
        // A pixel counts as informed if either pass saw a response: the large
        // window is the more robust of the two.
        let mut known2 = k2;
        known2
            .d
            .iter_mut()
            .zip(known.d.iter())
            .for_each(|(a, b)| *a = a.max(*b));
        // Where two surfaces meet, the second pass's estimate replaces the
        // first's; everywhere else the first pass stands. This is what keeps the
        // depth transition as narrow as the small window instead of as wide as
        // the large one.
        let d2 = match &mask {
            Some(m) => {
                let mut merged = d2;
                merged.d.iter_mut().zip(m.d.iter()).zip(first.d.iter()).for_each(
                    |((v, mk), f)| {
                        if *mk < 0.5 {
                            *v = *f;
                        }
                    },
                );
                merged
            }
            None => d2,
        };
        let u = share(&known2);
        let filled = if u > 0.0 { fill_unreliable(&d2, &known2, 12) } else { d2 };
        say(&format!(
            "  refined with a radius-{r2} window in {:.1}s",
            t2.elapsed().as_secs_f32()
        ));
        (filled, c2, known2, u)
    } else {
        let u = share(&known);
        let filled = if u > 0.0 {
            fill_unreliable(&raw_depth, &known, 12)
        } else {
            raw_depth
        };
        (filled, conf, known, u)
    };

    // Stage-by-stage trace of the depth post-processing chain. The focus measure
    // was verified to return the right frame at the problem location, so if the
    // saved depth is wrong the damage happens in this chain — find which stage.
    let trace_row = |label: &str, pl: &Plane| {
        if let Some(t) = args.depth_trace.as_deref() {
            let v: Vec<usize> = t.split(',').filter_map(|q| q.trim().parse().ok()).collect();
            if v.len() == 3 && v[0] < ah && v[2] < aw {
                let mut line = String::new();
                let mut x = v[1];
                while x <= v[2] {
                    line.push_str(&format!(" {:5.1}", pl.d[v[0] * aw + x]));
                    x += 4;
                }
                eprintln!("TRACE {label:<12}{line}");
            }
        }
    };
    trace_row("1 raw", &raw_depth);

    // Reliability ramp.
    //
    // The earlier design masked every pixel below a fixed confidence percentile
    // and inpainted the result. That guarantees a fixed *share* of the image
    // (30% by default) gets its depth replaced whether or not it was actually
    // unusable, and on this stack it moved 5% of all pixels by more than 11
    // frames — a smooth resin leaf whose true depth was 29.8 came out at 8.0.
    //
    // Instead the confidence now only controls how much a pixel is pulled
    // towards the smoothed depth field: weak but valid estimates survive.

    if let Some(p) = &args.save_conf {
        let (lo, hi) = conf.min_max();
        let span = (hi - lo).max(1e-9);
        let bytes: Vec<u8> = conf
            .d
            .iter()
            .map(|v| (((v - lo) / span) * 255.0) as u8)
            .collect();
        image::GrayImage::from_raw(aw as u32, ah as u32, bytes)
            .context("confidence buffer")?
            .save(p)
            .context("saving confidence map")?;
        say(&format!("wrote {}", p.display()));
    }
    if let Some(p) = &args.save_depth {
        let scale = (files.len() - 1).max(1) as f32;
        let data: Vec<u16> = raw_depth
            .d
            .iter()
            .map(|v| ((v / scale).clamp(0.0, 1.0) * 65535.0) as u16)
            .collect();
        image::ImageBuffer::<image::Luma<u16>, Vec<u16>>::from_raw(aw as u32, ah as u32, data)
            .context("depth buffer")?
            .save(p)
            .context("saving depth map")?;
        say(&format!("wrote {} (raw per-pixel estimate)", p.display()));
    }

    let base = if args.depth_median > 0 {
        median_n(&raw_depth, args.depth_median)
    } else {
        raw_depth.clone()
    };
    trace_row("2 median", &base);
    // Depth finalisation.
    //
    // An edge-aware (geodesic) fill was implemented here on the theory that a
    // large smooth object should inherit the depth of its own silhouette. It is
    // kept in `depth.rs` for further work but is not used: the whole scheme
    // depends on knowing which pixels carry real focus information, and no
    // reliability measure that could be validated on this stack was found.
    //
    // * raw peak value — punishes smooth low-contrast surfaces, whose focus peak
    //   is small but perfectly well placed.
    // * peak minus mean, z-score — the maximum of 50 noise samples also clears
    //   those; a real focus peak in a stack with small focus steps is broad, so
    //   it scores about the same.
    // * peak-to-mean ratio, evaluated per pixel — measured *higher* on a
    //   featureless surface (1.24) than on a textured face (1.12), i.e. exactly
    //   backwards, because in a low-signal region the noise dominates the ratio.
    //   Evaluating the ratio on a much larger aggregation window did not fix it.
    //
    // What does work, and what the pipeline uses, is to smooth the per-pixel
    // estimates themselves: a median (which removes speckle without spreading
    // boundaries) followed by a small-radius guided filter. Measured against the
    // reference export that combination reaches 1.11-1.23x on edge acutance.
    let guide_plane = Plane { w: aw, h: ah, d: guide };
    let mut dfield = base;
    if args.depth_bilateral > 0 {
        // Depth-similarity weighting, so the step at a silhouette survives the
        // smoothing that removes speckle on smooth surfaces. See
        // `depth::bilateral_depth`.
        for _ in 0..args.depth_smooth.max(1) {
            dfield = depth::bilateral_depth(&dfield, args.depth_bilateral, args.depth_sigma);
        }
        // The bilateral stops a step from being spread, but it cannot *move* one
        // either — and a step sitting one focus-window radius inside the far
        // surface is exactly what leaves the smooth band there. The guided filter
        // can pull it back, because its output follows the guide's edges; it
        // needs a guide in which that edge is sharp. Run sequentially rather than
        // either/or: they fix different things.
        dfield = guided_filter(&dfield, &guide_plane, args.guided_radius, args.guided_eps);
    } else {
        for _ in 0..args.depth_smooth.max(1) {
            dfield = guided_filter(&dfield, &guide_plane, args.guided_radius, args.guided_eps);
        }
    }
    trace_row("3 bilateral", &dfield);
    // One light pass to remove isolated flips, small enough not to drag the
    // depth across object boundaries.
    if args.depth_smooth > 0 {
        dfield = guided_filter(
            &dfield,
            &guide_plane,
            (args.guided_radius / 3).max(2),
            args.guided_eps,
        );
    }

    trace_row("4 final", &dfield);

    // Manual depth overrides, applied after everything else so that nothing can
    // undo them.
    for spec in &args.depth_fix {
        let parts: Vec<f32> = spec
            .split(',')
            .filter_map(|t| t.trim().parse::<f32>().ok())
            .collect();
        if parts.len() != 5 {
            bail!("--depth-fix expects x,y,w,h,frame — got '{spec}'");
        }
        let (rx, ry, rw, rh, frame) = (
            parts[0], parts[1], parts[2], parts[3], parts[4],
        );
        // The override is given in output coordinates (which are the canvas
        // shifted by the crop origin), the depth field is in canvas coordinates
        // at analysis resolution.
        let ax0 = ((rect.x as f32 + rx) / factor as f32).floor().max(0.0) as usize;
        let ay0 = ((rect.y as f32 + ry) / factor as f32).floor().max(0.0) as usize;
        let ax1 = ((rect.x as f32 + rx + rw) / factor as f32).ceil() as usize;
        let ay1 = ((rect.y as f32 + ry + rh) / factor as f32).ceil() as usize;
        let ax1 = ax1.min(aw);
        let ay1 = ay1.min(ah);
        let v = frame.clamp(0.0, (files.len() - 1) as f32);
        for y in ay0..ay1 {
            for x in ax0..ax1 {
                dfield.d[y * aw + x] = v;
            }
        }
        say(&format!(
            "  depth override {rw:.0}x{rh:.0} at ({rx:.0}, {ry:.0}) -> frame {v:.0}"
        ));
    }
    if let Some(p) = &args.save_depth {
        let scale = (files.len() - 1).max(1) as f32;
        let data: Vec<u16> = dfield
            .d
            .iter()
            .map(|v| ((v / scale).clamp(0.0, 1.0) * 65535.0) as u16)
            .collect();
        let fin = p.with_file_name(format!(
            "{}_final.{}",
            p.file_stem().and_then(|s| s.to_str()).unwrap_or("depth"),
            p.extension().and_then(|s| s.to_str()).unwrap_or("png")
        ));
        image::ImageBuffer::<image::Luma<u16>, Vec<u16>>::from_raw(aw as u32, ah as u32, data)
            .context("depth buffer")?
            .save(&fin)
            .context("saving final depth map")?;
        say(&format!("wrote {} (final)", fin.display()));
    }
    let drift: f32 = dfield
        .d
        .iter()
        .zip(raw_depth.d.iter())
        .map(|(a, b)| (a - b).abs())
        .sum::<f32>()
        / raw_depth.d.len() as f32;
    say(&format!(
        "[3/4] depth field ({}x{}, {:.0}% without focus information, mean drift {:.2} frames) in {:.1}s",
        aw,
        ah,
        100.0 * unknown_share,
        drift,
        t.elapsed().as_secs_f32()
    ));

    // ------------------------------------------- depth-dependent registration
    //
    // Everything up to here registered the frames with one similarity transform
    // each, which by construction satisfies a single distance. Refocusing also
    // moves the entrance pupil and changes magnification, so the image of a scene
    // point moves by an amount that depends on how far away the point is, and
    // what is left over after the global fit is not noise but a systematic
    // function of depth. On the reference stack the near foreground needs about
    // 10 px more shift than the face, and the frames that carry the foreground
    // sharply are exactly the ones that are furthest out of register there — so
    // the fusion, which mixes neighbouring frames, mixes misregistered content
    // and the eye reads the result as soft.
    //
    // Measure that residual against the depth field and let every warp follow it.
    let parallax: Option<align::Parallax> = if args.parallax > 0 {
        let tp = Instant::now();
        let mut ds: Vec<f32> = dfield.d.clone();
        ds.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let near = ds[((ds.len() as f32 - 1.0) * 0.02) as usize];
        let far = ds[((ds.len() as f32 - 1.0) * 0.98) as usize];
        let popts = align::ParallaxOptions {
            search: (args.parallax / factor).max(4),
            blocks: args.parallax_blocks.max(2),
        };
        let reference = align::warp(&cache.plane(mid), &transforms_cache[mid], aw, ah, 0.0);
        let mut fits: Vec<align::ParallaxFit> = Vec::with_capacity(files.len());
        let mut worst = 0.0f32;
        let mut qsum = 0.0f32;
        let mut used = 0;
        for k in 0..files.len() {
            let wg = align::warp(&cache.plane(k), &transforms_cache[k], aw, ah, 0.0);
            let f = align::estimate_parallax(&reference, &wg, &dfield, near, far, &popts);
            worst = worst.max(f.dx_near.abs().max(f.dy_near.abs()));
            if f.blocks > 0 {
                qsum += f.quality;
                used += 1;
            }
            fits.push(f);
        }
        let raw = fits.clone();
        fits = align::smooth_fits(&fits, mid);
        for (k, f) in fits.iter().enumerate() {
            transforms_full[k].dx += f.dx_far * factor as f32;
            transforms_full[k].dy += f.dy_far * factor as f32;
            transforms_full[k].pdx = (f.dx_near - f.dx_far) * factor as f32;
            transforms_full[k].pdy = (f.dy_near - f.dy_far) * factor as f32;
        }
        for k in (0..files.len()).step_by((files.len() / 8).max(1)) {
            let (f, r) = (fits[k], raw[k]);
            say(&format!(
                "  frame {:>2}: far ({:+.1},{:+.1})  near ({:+.1},{:+.1})   raw far ({:+.1},{:+.1}) near ({:+.1},{:+.1}) blocks {:>2} q {:.2}",
                k, f.dx_far, f.dy_far, f.dx_near, f.dy_near,
                r.dx_far, r.dy_far, r.dx_near, r.dy_near, r.blocks, r.quality
            ));
        }
        say(&format!(
            "  parallax: max |shift| {:.1} px at analysis scale, {}/{} frames fitted (mean quality {:.2}) in {:.1}s",
            worst,
            used,
            files.len(),
            qsum / used.max(1) as f32,
            tp.elapsed().as_secs_f32()
        ));
        let dilate = (3 / factor).max(1);
        Some(align::Parallax::new(
            &dfield.d,
            aw,
            ah,
            factor as f32,
            near,
            far,
            dilate,
        ))
    } else {
        None
    };

    // ---------------------------------------------------------------- stage 4
    let t = Instant::now();

    // Per-pixel depth uncertainty: how far the final estimate had to be dragged
    // from the coarse prior, in units of the search window. 0 means the per-pixel
    // evidence agreed with the coarse measurement, 1 that it ran to the edge of
    // the window and is therefore not to be trusted.
    let slack: Option<Plane> = prior.as_ref().map(|p| {
        let win = args.refine_window.max(1.0);
        let mut s = Plane::new(aw, ah);
        s.d.par_iter_mut().enumerate().for_each(|(i, o)| {
            *o = ((dfield.d[i] - p.d[i]).abs() / win).clamp(0.0, 1.0);
        });
        s
    });
    let slack_full = slack.map(|s| {
        if factor > 1 {
            upsample_depth(&s, cache.full_w, cache.full_h)
        } else {
            s
        }
    });

    let depth_full = if factor > 1 {
        upsample_depth(&dfield, cache.full_w, cache.full_h)
    } else {
        dfield
    };

    let rect = rect;
    say(&format!(
        "  output window {}x{} at ({}, {})",
        rect.w, rect.h, rect.x, rect.y
    ));

    // Per-frame brightness offsets relative to the middle frame.
    let offsets: Vec<[f32; 3]> = if args.no_match_exposure {
        vec![[0.0; 3]; files.len()]
    } else {
        let refm = means[mid];
        means
            .iter()
            .map(|m| {
                let mut o = [0.0f32; 3];
                for c in 0..3 {
                    o[c] = (refm[c] - m[c]).clamp(-8.0 / 255.0, 8.0 / 255.0);
                }
                o
            })
            .collect()
    };

    let levels = if args.levels > 0 {
        args.levels
    } else {
        // Use two thirds of the usable levels: the coarsest bands carry colour
        // only and gain nothing from going all the way down.
        let m = pyramid::max_levels(rect.w, rect.h);
        ((m * 2 + 2) / 3).max(3).min(m)
    };
    say(&format!("  fusion mode {:?}, {} pyramid levels", args.mode, levels));

    struct Files(Vec<PathBuf>);
    impl fuse::FrameSource for Files {
        fn count(&self) -> usize {
            self.0.len()
        }
        fn load(&self, i: usize) -> Result<RgbImage> {
            load_rgb(&self.0[i])
        }
    }
    let source = Files(files.clone());
    let fopts = FuseOptions {
        mode: args.mode,
        levels,
        rect,
        offsets,
        bracket_strength: args.bracket_strength,
        bracket_radius: args.bracket_radius,
        bracket_radius_coarse: args.bracket_radius_coarse,
        parallax: parallax.clone(),
        coarse_levels: args.coarse_levels,
        denoise: args.denoise,
        base_winner: args.base_winner.clamp(0.0, 1.0),
        slack: slack_full,
        adaptive_window: args.adaptive_window.max(0.0),
        seam_window: args.seam_window.max(0.0),
        band_power: args.band_power.max(0.0),
        band_energy_smooth: args.band_energy_smooth,
        band_energy_gate: args.band_energy_gate,
        clamp_range: !args.no_clamp_range,
        clamp_lo: !args.no_clamp_lo,
        clamp_hi: !args.no_clamp_hi,
        clamp_slack: args.clamp_slack,
        clamp_hi_dilate: args.clamp_hi_dilate,
        spill: args.spill,
        save_range: args.save_range.clone(),
        conf: if args.conf_band {
            // Gate on how *locally smooth the depth itself* is.
            //
            // The reliability of the focus response was tried first and does not
            // work here: on the plane the fusion actually consumes, the
            // silhouette band (which must be kept) scores 1.67 while the subject's
            // smooth plastic (which should be gated) scores 2.34 — the wrong
            // order. That is the same failure the comment in `depth.rs` records
            // for every other reliability measure tried on this stack.
            //
            // The depth's own roughness does order them the way we need, and not
            // by accident: it *is* the defect. A step at a silhouette is locally
            // smooth, while the wandering that per-band maximum turns into mottle
            // is rough, so gating on it keeps the band and calms the subject.
            let med = depth::median_n(&raw_depth, 2);
            let g = Plane {
                w: raw_depth.w,
                h: raw_depth.h,
                d: raw_depth
                    .d
                    .iter()
                    .zip(med.d.iter())
                    .map(|(a, b)| 1.0 / (1.0 + (a - b).abs()))
                    .collect(),
            };
            let d: Vec<f32> = if factor > 1 {
                upsample_depth(&g, cache.full_w, cache.full_h).d
            } else {
                g.d
            };
            {
                let mut v: Vec<f32> = d.clone();
                v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                let q = |t: f32| v[((v.len() - 1) as f32 * t) as usize];
                say(&format!(
                    "  reliability gate: p1 {:.2} p10 {:.2} p50 {:.2} p90 {:.2} p99 {:.2}   (lo {:.2} hi {:.2})",
                    q(0.01), q(0.10), q(0.50), q(0.90), q(0.99), args.conf_lo, args.conf_hi
                ));
                // Named regions, in canvas coordinates, so the thresholds can be
                // set against the plane the fusion actually consumes rather than
                // against an offline reconstruction of it.
                for (name, cy, cx, s) in [
                    ("silhouette-band", 2874usize, 2795usize, 40usize),
                    ("wall-relief", 2874, 2670, 120),
                    ("flat-backdrop", 2412, 1218, 120),
                    ("subject-plastic", 3062, 3168, 160),
                    ("leather-band", 2962, 3218, 120),
                    ("face-patch", 2506, 2004, 120),
                    ("leaf-patch", 1412, 3046, 180),
                ] {
                    if cy + s <= ah && cx + s <= aw {
                        let mut w: Vec<f32> = Vec::with_capacity(s * s);
                        for y in cy..cy + s {
                            for x in cx..cx + s {
                                w.push(d[y * aw + x]);
                            }
                        }
                        w.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                        say(&format!(
                            "    {:<18} canvas({},{}) {}px  median {:.2}  p10 {:.2}  p90 {:.2}",
                            name, cy, cx, s, w[w.len() / 2], w[w.len() / 10], w[w.len() * 9 / 10]
                        ));
                    }
                }
            }
            Some(Plane { w: aw, h: ah, d })
        } else {
            None
        },
        conf_lo: args.conf_lo,
        conf_hi: args.conf_hi,
    };
    let merged = fuse::fuse(
        &source,
        &transforms_full,
        &depth_full,
        &fopts,
        &|k, total| {
            if k == total || k % 5 == 0 {
                eprint!("\r  fusing {}/{}   ", k, total);
                if k == total {
                    eprintln!();
                }
            }
        },
    )?;
    say(&format!(
        "[4/4] fused {}x{} in {:.1}s",
        merged.w,
        merged.h,
        t.elapsed().as_secs_f32()
    ));

    // ------------------------------------------------------------------ output
    write_output(&merged, &args.output)?;
    say(&format!("wrote {}", args.output.display()));
    if let Some(p) = &args.preview {
        let long = merged.w.max(merged.h) as f32;
        let f = (args.preview_size as f32 / long).min(1.0);
        let pw = (merged.w as f32 * f) as usize;
        let ph = (merged.h as f32 * f) as usize;
        let small = downsample_rgb(&merged, pw, ph);
        let buf = image::RgbImage::from_raw(
            pw as u32,
            ph as u32,
            small
                .d
                .iter()
                .map(|v| (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8)
                .collect(),
        )
        .context("preview buffer")?;
        buf.save(p).context("saving preview")?;
        say(&format!("wrote {}", p.display()));
    }
    say(&format!("total {:.1}s", t_total.elapsed().as_secs_f32()));
    Ok(())
}

fn downsample_rgb(src: &RgbImage, tw: usize, th: usize) -> RgbImage {
    let mut d = vec![0.0f32; tw * th * 3];
    let kx = src.w as f32 / tw as f32;
    let ky = src.h as f32 / th as f32;
    d.par_chunks_mut(tw * 3).enumerate().for_each(|(y, row)| {
        let sy = (y as f32 + 0.5) * ky - 0.5;
        for x in 0..tw {
            let sx = (x as f32 + 0.5) * kx - 0.5;
            let x0 = sx.floor().max(0.0) as usize;
            let y0 = sy.floor().max(0.0) as usize;
            let x1 = (x0 + 1).min(src.w - 1);
            let y1 = (y0 + 1).min(src.h - 1);
            let x0 = x0.min(src.w - 1);
            let y0 = y0.min(src.h - 1);
            let fx = sx - sx.floor();
            let fy = sy - sy.floor();
            for c in 0..3 {
                let a = src.d[(y0 * src.w + x0) * 3 + c];
                let b = src.d[(y0 * src.w + x1) * 3 + c];
                let cc = src.d[(y1 * src.w + x0) * 3 + c];
                let dd = src.d[(y1 * src.w + x1) * 3 + c];
                let top = a + (b - a) * fx;
                let bot = cc + (dd - cc) * fx;
                row[x * 3 + c] = top + (bot - top) * fy;
            }
        }
    });
    RgbImage { w: tw, h: th, d }
}

fn write_output(img: &RgbImage, path: &Path) -> Result<()> {
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("png")
        .to_ascii_lowercase();
    if ext == "jpg" || ext == "jpeg" {
        let buf = image::RgbImage::from_raw(
            img.w as u32,
            img.h as u32,
            img.d
                .iter()
                .map(|v| (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8)
                .collect(),
        )
        .context("building output buffer")?;
        let mut out = std::fs::File::create(path)?;
        let mut enc = image::codecs::jpeg::JpegEncoder::new_with_quality(&mut out, 95);
        enc.encode_image(&buf)?;
    } else {
        // 16 bit PNG keeps the blending smooth, avoiding banding in gradients.
        let data = img.to_u16();
        let buf = image::ImageBuffer::<image::Rgb<u16>, Vec<u16>>::from_raw(
            img.w as u32,
            img.h as u32,
            data,
        )
        .context("building 16-bit output buffer")?;
        buf.save(path)
            .with_context(|| format!("saving {}", path.display()))?;
    }
    Ok(())
}
