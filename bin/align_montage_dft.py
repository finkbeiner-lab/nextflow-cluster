#!/usr/bin/env python
"""Montage alignment across timepoints (whole-well rigid translation).

Revised version of ``align_montage_dft.py``. Same inputs, outputs and
database updates, but with a registration method that is robust to the
artefacts present in background-subtracted IXM montages.

Why the original failed
-----------------------
* Montages are stitched from tiles that were background-subtracted
  independently and clipped at 0, so each tile is a flat block with a
  different offset and large all-zero regions. A global, unfiltered phase
  correlation locks onto those block edges (tile seams, image borders) and
  returns false shifts of 1,000-2,000 px (often ~one tile width, with one
  axis exactly 0).
* Shifts are chained (T(n) vs T(n-1)) and summed with no per-step limit and
  no quality check, so one false step pushes every later timepoint off by
  the same amount. On B02 the cumulative shift exceeded the image size and
  T22 + T31-T48 were written as entirely black frames.

What this version does, per well
--------------------------------
1. Pre-processes the registration channel so cells, not tile blocks, drive
   the match: per-tile median removal, 4x binning, high-pass filter
   (subtract a Gaussian blur), clip to [0, p99.5], Hann window.
2. Phase-correlates each timepoint against the last *confidently* registered
   timepoint, searching only shifts up to ``--max_step`` px per timepoint
   gap (a seam peak 2,048 px away cannot win). Sub-pixel parabolic peak
   refinement.
3. Scores each match by peak strength (peak / std of the correlation
   surface). A weak match is not trusted: the frame keeps the previous
   cumulative shift and is flagged, and the next frame is registered
   against the last good frame instead, so a single bad frame (focus
   failure, empty image) cannot corrupt later timepoints.
4. Applies the cumulative shift to every selected channel (including the
   registration channel) by integer translation with zero fill, which keeps
   raw pixel intensities unchanged (no interpolation) and preserves dtype.
5. Writes ``<well>_shifts.csv`` (step, cumulative shift, peak strength,
   status) next to the aligned images, plus the pickled shift dict, and
   logs a warning for every flagged step.

Aligned images are saved under ``AlignedMontages/<well>/`` and the
``alignedmontagepath`` column of ``tiledata`` is updated, exactly as before.
"""

import argparse
import csv
import datetime
import logging
import math
import os
import pickle
from typing import Dict, List, Optional, Tuple

import cv2
import imageio
import numpy as np
from scipy.ndimage import gaussian_filter
from tqdm import tqdm

from db_util import Ops
from sql import Database

# Module-level logger — writes to ./finkbeiner_logs/AlignMontageDFT-log_<ts>.log
# and (via attach_experiment_log below) to <output_path>/pipeline_debug.log.
logger = logging.getLogger("AlignMontageDFT")
logger.setLevel(logging.INFO)
_now = datetime.datetime.now()
_TIMESTAMP = '%d%02d%02d%02d%02d' % (_now.year, _now.month, _now.day, _now.hour, _now.minute)
_fink_log_dir = './finkbeiner_logs'
os.makedirs(_fink_log_dir, exist_ok=True)
logger.addHandler(logging.FileHandler(os.path.join(_fink_log_dir, f'AlignMontageDFT-log_{_TIMESTAMP}.log')))
# Also mirror this logger to the experiment-scoped debug log
# (<params.output_path>/pipeline_debug.log) shared across all parallel wells.
from experiment_logger import attach_experiment_log  # noqa: E402
attach_experiment_log(logger, os.environ.get('NEXTFLOW_OUTPUT_PATH', ''), 'ALIGN_MONTAGE_DFT')


# ---------------------------------------------------------------------------
# Registration primitives (pure functions, no DB access)
# ---------------------------------------------------------------------------

def remove_tile_background(img: np.ndarray, tile_px: Optional[Tuple[int, int]]) -> np.ndarray:
    """Return a float32 copy with each stitched tile's median subtracted.

    Each tile in the montage was background-corrected independently, so tiles
    sit at different offsets. Removing the per-tile median stops the tile
    blocks/seams from being the dominant structure in the image.
    """
    im = img.astype(np.float32, copy=True)
    if tile_px is not None:
        h, w = im.shape
        th, tw = tile_px
        for y in range(0, h, th):
            for x in range(0, w, tw):
                blk = im[y:y + th, x:x + tw]
                blk -= np.median(blk)
    return im


def prep_for_registration(img: np.ndarray, tile_px: Optional[Tuple[int, int]],
                          bin_factor: int, highpass_sigma: float = 12.0) -> np.ndarray:
    """Turn a montage into a registration-friendly image and return its FFT.

    Args:
        img: 2-D montage image (any dtype).
        tile_px: ``(tile_h, tile_w)`` of the stitched tiles inside the
            montage, or ``None`` to skip per-tile background removal.
        bin_factor: Downsampling factor (block mean).
        highpass_sigma: Gaussian sigma (in binned px) of the background
            estimate subtracted for high-pass filtering.

    Returns:
        Complex 2-D FFT of the pre-processed, windowed, binned image.
    """
    # 1. Per-tile background removal.
    im = remove_tile_background(img, tile_px)
    h, w = im.shape

    # 2. Bin (block mean) — faster FFTs and suppresses pixel noise.
    b = max(1, int(bin_factor))
    h2, w2 = (h // b) * b, (w // b) * b
    im = im[:h2, :w2].reshape(h2 // b, b, w2 // b, b).mean(axis=(1, 3))

    # 3. High-pass: keep cell-sized structure, drop slow background and
    #    the residual step at tile edges.
    im = im - gaussian_filter(im, highpass_sigma)

    # 4. Keep only bright structure (cells); clip hot pixels/debris.
    hi = np.percentile(im, 99.5)
    im = np.clip(im, 0, hi if hi > 0 else None)
    im -= im.mean()

    # 5. Hann window: tapers image borders so the montage edge and the
    #    zero-filled regions don't create a spurious peak at zero shift.
    win = np.hanning(im.shape[0])[:, None] * np.hanning(im.shape[1])[None, :]
    return np.fft.fft2(im * win)


def phase_correlate(f_moving: np.ndarray, f_ref: np.ndarray, bin_factor: int,
                    max_shift: float) -> Tuple[float, float, float]:
    """Constrained phase correlation between two pre-processed FFTs.

    Args:
        f_moving: FFT of the moving image (from :func:`prep_for_registration`).
        f_ref: FFT of the reference image.
        bin_factor: Binning used in pre-processing (to rescale the shift).
        max_shift: Largest allowed |dy| or |dx| in full-resolution px. Peaks
            outside this window are ignored.

    Returns:
        ``(dy, dx, strength)`` where ``(dy, dx)`` is the displacement of the
        moving image's content relative to the reference in full-resolution
        px (shift the moving image by ``(-dy, -dx)`` to align it), and
        *strength* is the peak height divided by the std of the correlation
        surface (higher = more reliable).
    """
    r_spec = f_moving * np.conj(f_ref)
    r_spec /= np.abs(r_spec) + 1e-9
    r = np.fft.ifft2(r_spec).real
    ny, nx = r.shape

    # Restrict the search to plausible shifts.
    lim = max_shift / bin_factor
    fy = np.fft.fftfreq(ny) * ny
    fx = np.fft.fftfreq(nx) * nx
    r_masked = r.copy()
    r_masked[np.abs(fy) > lim, :] = r.min()
    r_masked[:, np.abs(fx) > lim] = r.min()

    iy, ix = np.unravel_index(np.argmax(r_masked), r.shape)
    std = r.std()
    strength = float(r[iy, ix] / std) if std > 0 else 0.0

    def _subpixel(line: np.ndarray, m: int, n: int) -> float:
        left, centre, right = line[(m - 1) % n], line[m], line[(m + 1) % n]
        denom = left - 2 * centre + right
        return 0.0 if denom == 0 else 0.5 * (left - right) / denom

    dy = iy + _subpixel(r[:, ix], iy, ny)
    dx = ix + _subpixel(r[iy, :], ix, nx)
    if dy > ny / 2:
        dy -= ny
    if dx > nx / 2:
        dx -= nx
    return float(dy * bin_factor), float(dx * bin_factor), strength


def translate_image(img: np.ndarray, dy: float, dx: float) -> np.ndarray:
    """Integer translation with zero fill; keeps dtype and raw intensities.

    Positive ``dy``/``dx`` move content down/right.
    """
    dy, dx = int(round(dy)), int(round(dx))
    out = np.zeros_like(img)
    h, w = img.shape[:2]
    if abs(dy) >= h or abs(dx) >= w:
        return out
    ys, yd = (slice(0, h - dy), slice(dy, h)) if dy >= 0 else (slice(-dy, h), slice(0, h + dy))
    xs, xd = (slice(0, w - dx), slice(dx, w)) if dx >= 0 else (slice(-dx, w), slice(0, w + dx))
    out[yd, xd] = img[ys, xs]
    return out


def infer_tile_px(shape: Tuple[int, int], n_tiles: int) -> Optional[Tuple[int, int]]:
    """Infer the stitched tile size from montage shape and tile count.

    Returns ``None`` if the tile layout is not a square grid.
    """
    n = int(round(math.sqrt(n_tiles))) if n_tiles > 0 else 0
    if n < 1 or n * n != n_tiles:
        return None
    return shape[0] // n, shape[1] // n


def compute_well_shifts(frames: List[Tuple[int, str]], tile_px_hint: int, n_tiles: int,
                        bin_factor: int, max_step: float, min_peak: float
                        ) -> List[Dict]:
    """Compute cumulative drift for one well, relative to the first timepoint.

    Each frame is registered against the last confidently-registered frame.
    Weak matches keep the previous drift and are flagged.

    Args:
        frames: ``[(timepoint, montage_path), ...]`` sorted by timepoint.
        tile_px_hint: Tile size in px inside the montage; 0 = infer from
            montage shape and ``n_tiles``.
        n_tiles: Number of tiles stitched into the montage.
        bin_factor: Downsampling factor for registration.
        max_step: Max allowed shift (px) per one-timepoint gap.
        min_peak: Peak strength below which a match is rejected.

    Returns:
        One dict per readable frame with keys ``timepoint``, ``path``,
        ``ref_timepoint``, ``step_dy``, ``step_dx``, ``drift_dy``,
        ``drift_dx``, ``peak``, ``status``.
    """
    rows: List[Dict] = []
    ref_fft = None
    ref_tp = None
    ref_idx = None
    ref_drift = np.zeros(2)
    ref_shape = None
    tile_px = None

    for idx, (tp, path) in enumerate(frames):
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            logger.warning(f'cv2.imread returned None for {path}; skipping T{tp}.')
            print(f'⚠️ cv2.imread returned None for {path}; skipping T{tp}.')
            continue
        if img.ndim == 3:
            img = img[..., 0]

        if ref_fft is None:
            # First readable frame is the anchor (drift = 0).
            ref_shape = img.shape
            tile_px = ((tile_px_hint, tile_px_hint) if tile_px_hint > 0
                       else infer_tile_px(img.shape, n_tiles))
            ref_fft = prep_for_registration(img, tile_px, bin_factor)
            ref_tp, ref_idx = tp, idx
            rows.append(dict(timepoint=tp, path=path, ref_timepoint=tp, step_dy=0.0, step_dx=0.0,
                             drift_dy=0.0, drift_dx=0.0, peak=float('nan'), status='anchor'))
            continue

        if img.shape != ref_shape:
            # Mismatched montage size: can't correlate; keep previous drift.
            logger.warning(f'T{tp} shape {img.shape} != {ref_shape}; keeping previous drift.')
            rows.append(dict(timepoint=tp, path=path, ref_timepoint=ref_tp, step_dy=0.0, step_dx=0.0,
                             drift_dy=ref_drift[0], drift_dx=ref_drift[1], peak=float('nan'),
                             status='shape_mismatch'))
            continue

        f_cur = prep_for_registration(img, tile_px, bin_factor)
        gap = idx - ref_idx  # allow more drift if we are bridging a bad frame
        dy, dx, peak = phase_correlate(f_cur, ref_fft, bin_factor, max_step * gap)

        if peak >= min_peak:
            drift = ref_drift + np.array([dy, dx])
            rows.append(dict(timepoint=tp, path=path, ref_timepoint=ref_tp,
                             step_dy=round(dy, 2), step_dx=round(dx, 2),
                             drift_dy=round(drift[0], 2), drift_dx=round(drift[1], 2),
                             peak=round(peak, 2), status='ok'))
            # This frame becomes the new reference.
            ref_fft, ref_tp, ref_idx, ref_drift = f_cur, tp, idx, drift
        else:
            # Unreliable match: don't move this frame relative to the last good
            # one, and keep registering later frames against the last good one.
            rows.append(dict(timepoint=tp, path=path, ref_timepoint=ref_tp,
                             step_dy=round(dy, 2), step_dx=round(dx, 2),
                             drift_dy=round(ref_drift[0], 2), drift_dx=round(ref_drift[1], 2),
                             peak=round(peak, 2), status='low_confidence'))

    return rows


def write_shift_csv(path: str, well: str, rows: List[Dict]) -> None:
    """Write the per-step shift table for one well."""
    fields = ['well', 'timepoint', 'ref_timepoint', 'step_dy', 'step_dx',
              'drift_dy', 'drift_dx', 'peak', 'status']
    with open(path, 'w', newline='') as fh:
        wr = csv.DictWriter(fh, fieldnames=fields, extrasaction='ignore')
        wr.writeheader()
        for r in rows:
            wr.writerow({**r, 'well': well})


def load_shift_dict(path: str) -> Optional[Dict[Tuple[str, int], np.ndarray]]:
    """Load a precomputed ``{(well, timepoint): [dy, dx]}`` pickle, if given."""
    if not path or path.lower() in ('null', 'none') or not os.path.isfile(path):
        return None
    with open(path, 'rb') as f:
        d = pickle.load(f)
    return {k: np.asarray(v, dtype=float) for k, v in d.items()}


# ---------------------------------------------------------------------------
# Pipeline entry point
# ---------------------------------------------------------------------------

def align_tiles(opt: argparse.Namespace) -> None:
    """Compute shifts on the registration channel and apply them to all channels.

    Step 1 (per well): compute cumulative drift per timepoint on
    ``opt.align_channel`` (defaults to ``opt.morphology_channel``), unless a
    precomputed ``--shift_dict`` pickle is supplied.

    Step 2: translate every selected channel montage (including the
    registration channel) by the negative drift, save to
    ``AlignedMontages/<well>/`` and update ``tiledata.alignedmontagepath``.

    Args:
        opt: Parsed command-line arguments.
    """
    ops = Ops(opt)
    df = ops.get_df_for_training(['channeldata'])

    if 'timepoint' not in df.columns:
        raise KeyError("tiledata must contain 'timepoint' column")

    df = df[df['newimagemontage'].str.endswith('.tif', na=False)]

    db = Database()
    _, analysisdir = ops.get_raw_and_analysis_dir()
    print("analysisdir", analysisdir)
    out_root = os.path.join(analysisdir, 'AlignedMontages')
    os.makedirs(out_root, exist_ok=True)

    align_channel = opt.align_channel or opt.morphology_channel
    if align_channel not in set(df['channel']):
        logger.warning(f'align_channel {align_channel} not found; falling back to '
                       f'{opt.morphology_channel}.')
        align_channel = opt.morphology_channel

    # ---- Step 1: shifts per (well, timepoint) ----
    shifts: Dict[Tuple[str, int], np.ndarray] = {}
    precomputed = load_shift_dict(opt.shift_dict)
    if precomputed is not None:
        print(f'Using precomputed shifts from {opt.shift_dict}')
        logger.info(f'Using precomputed shifts from {opt.shift_dict}')
        shifts = precomputed
    else:
        reg = df[df['channel'] == align_channel]
        for well, wgroup in reg.groupby('well'):
            reps = (wgroup.sort_values('timepoint')
                          .drop_duplicates(subset=['timepoint']))
            frames = [(int(r.timepoint), r.newimagemontage) for r in reps.itertuples()]
            n_tiles = wgroup['tile'].nunique() if 'tile' in wgroup.columns else 0
            rows = compute_well_shifts(frames, opt.montage_tile_px, n_tiles,
                                       opt.bin, opt.max_step, opt.min_peak)
            for r in rows:
                shifts[(well, r['timepoint'])] = np.array([r['drift_dy'], r['drift_dx']])

            out_dir = os.path.join(out_root, well)
            os.makedirs(out_dir, exist_ok=True)
            csv_path = os.path.join(out_dir, f'{well}_shifts.csv')
            write_shift_csv(csv_path, well, rows)

            flagged = [r for r in rows if r['status'] not in ('ok', 'anchor')]
            max_drift = max((math.hypot(r['drift_dy'], r['drift_dx']) for r in rows), default=0.0)
            msg = (f'{well}: {len(rows)} timepoints on {align_channel}, max drift {max_drift:.0f} px, '
                   f'{len(flagged)} flagged step(s). Shifts: {csv_path}')
            print(msg)
            logger.info(msg)
            for r in flagged:
                w = (f'{well} T{r["timepoint"]}: {r["status"]} (peak={r["peak"]}, '
                     f'step=({r["step_dy"]},{r["step_dx"]}) vs T{r["ref_timepoint"]}); '
                     f'kept previous drift — please inspect.')
                print('⚠️ ' + w)
                logger.warning(w)

        ts = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
        shift_path = os.path.join(out_root, f'calculated_shift_{ts}.dict')
        with open(shift_path, 'wb') as f:
            pickle.dump({k: v.tolist() for k, v in shifts.items()}, f)
        logger.info(f'Saved shift dict to {shift_path}')

    # ---- Step 2: apply shifts to every selected channel ----
    if opt.chosen_channels.lower() != 'all':
        wanted = {c.strip() for c in opt.chosen_channels.split(',')}
        wanted.add(opt.morphology_channel)
        wanted.add(align_channel)
        df = df[df['channel'].isin(wanted)]

    grouped = df.groupby(['well', 'timepoint', 'channel', 'newimagemontage'])
    for (well, timepoint, channel, montage_path), group in tqdm(
            grouped, desc='Applying shifts'):
        key = (well, int(timepoint))
        if key not in shifts:
            logger.warning(f'No shift for {well} T{timepoint}; skipping {channel}.')
            continue
        dy, dx = shifts[key]
        img = cv2.imread(montage_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f'⚠️ cv2.imread returned None for {montage_path}; '
                  f'skipping {well} T{timepoint} {channel}.')
            continue
        aligned = translate_image(img, -dy, -dx)
        out_dir = os.path.join(out_root, well)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, os.path.basename(montage_path).replace('.tif', '_ALIGNED.tif'))
        imageio.imwrite(out_path, aligned)
        for tdid in group['tiledata_id'].tolist():
            db.update('tiledata', {'alignedmontagepath': out_path}, {'id': tdid})

    print('Montage alignment complete.')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Align montages across timepoints (robust phase correlation)')
    p.add_argument('--experiment', required=True)
    p.add_argument('--morphology_channel', required=True)
    p.add_argument('--wells_toggle', default='include')
    p.add_argument('--timepoints_toggle', default='include')
    p.add_argument('--channels_toggle', default='include')
    p.add_argument('--chosen_wells', default='all')
    p.add_argument('--chosen_timepoints', default='all')
    p.add_argument('--chosen_channels', default='all',
                   help="Comma-separated channels or 'all' to apply shifts to every channel.")
    p.add_argument('--tile', type=int, default=0)
    p.add_argument('--shift_dict', type=str, default='',
                   help="Optional pickle {(well, timepoint): [dy, dx]} to apply instead of computing.")
    p.add_argument('--img_norm_name', choices=['division', 'subtraction', 'identity'], default='subtraction')
    # New, all optional (defaults work for 3x3 IXM montages of 2048-px tiles):
    p.add_argument('--align_channel', type=str, default='',
                   help="Channel used to measure shifts; '' = morphology_channel.")
    p.add_argument('--montage_tile_px', type=int, default=0,
                   help='Tile size (px) inside the montage; 0 = infer from montage size and tile count.')
    p.add_argument('--bin', type=int, default=4, help='Downsampling factor for registration.')
    p.add_argument('--max_step', type=float, default=800,
                   help='Max allowed shift (px) between consecutive timepoints.')
    p.add_argument('--min_peak', type=float, default=8.0,
                   help='Correlation peak strength below which a step is rejected and flagged.')
    args = p.parse_args()
    align_tiles(args)
