#!/usr/bin/env python
"""Montage alignment via DFT-based image registration.

Computes inter-frame translational shifts on the morphology channel using
phase cross-correlation (with an ``imreg_dft`` fallback when the shift
exceeds a plausibility threshold), then applies those same shifts to every
other selected channel so that all channels are spatially co-registered
across timepoints.

Aligned images are saved as 16-bit TIFFs under an ``AlignedMontages/``
directory, and the ``alignedmontagepath`` column in the ``tiledata``
database table is updated accordingly.
"""

import argparse
import datetime
import logging
import os
import pickle
from typing import Tuple

import cv2
import imageio
import numpy as np
from skimage import transform
from skimage.registration import phase_cross_correlation
import imreg_dft as ird
from tqdm import tqdm

from db_util import Ops
from sql import Database

# Module-level logger — writes to ./finkbeiner_logs/AlignMontageDFT-log_<ts>.log
# and (via attach_experiment_log below) to <output_path>/pipeline_debug.log.
logger = logging.getLogger("AlignMontageDFT")
_now = datetime.datetime.now()
_TIMESTAMP = '%d%02d%02d%02d%02d' % (_now.year, _now.month, _now.day, _now.hour, _now.minute)
_fink_log_dir = './finkbeiner_logs'
os.makedirs(_fink_log_dir, exist_ok=True)
logger.addHandler(logging.FileHandler(os.path.join(_fink_log_dir, f'AlignMontageDFT-log_{_TIMESTAMP}.log')))
# Also mirror this logger to the experiment-scoped debug log
# (<params.output_path>/pipeline_debug.log) shared across all parallel wells.
from experiment_logger import attach_experiment_log  # noqa: E402
attach_experiment_log(logger, os.environ.get('NEXTFLOW_OUTPUT_PATH', ''), 'ALIGN_MONTAGE_DFT')


def cross_correlation_dft_combo(
    ref_img: np.ndarray, img: np.ndarray
) -> Tuple[np.ndarray, str]:
    """Measure the translational shift between two images.

    Tries scikit-image phase cross-correlation first.  If the resulting
    shift exceeds 1/9 of the image dimensions (likely an artefact), falls
    back to ``imreg_dft`` translation estimation.

    Args:
        ref_img: Reference (fixed) image as a 2-D array.
        img: Moving image to be registered against *ref_img*.

    Returns:
        A tuple of ``(shift, method)`` where *shift* is a length-2 numpy
        array ``[dy, dx]`` and *method* is one of ``'phase'``,
        ``'phase_error'``, ``'dft'``, or ``'dft_error'``.
    """
    fix_by_dft = False
    # Plausibility thresholds: shifts larger than ~11% of each axis are suspect
    y_thresh = ref_img.shape[0] / 9
    x_thresh = ref_img.shape[1] / 9

    while True:
        if not fix_by_dft:
            try:
                shift, _, _ = phase_cross_correlation(ref_img, img)
                method = 'phase'
            except Exception:
                shift = np.zeros(2)
                method = 'phase_error'
        else:
            try:
                result = ird.translation(ref_img, img)
                shift = np.array(result['tvec'])
                method = 'dft'
            except Exception:
                shift = np.zeros(2)
                method = 'dft_error'

        # If shift exceeds plausibility threshold, retry with DFT method
        if (abs(shift[0]) >= y_thresh or abs(shift[1]) >= x_thresh) and not fix_by_dft:
            fix_by_dft = True
            continue
        break

    return shift, method


def save_image_uint16(path: str, img: np.ndarray) -> None:
    """Save an image array as a 16-bit unsigned integer TIFF.

    Float images in the [0, 1] range are scaled to the full uint16 range.
    Float images with values > 1 are clipped to uint16 max.  Integer
    images are cast directly.

    Args:
        path: Destination file path (should end in ``.tif``).
        img: Image array to save.
    """
    if np.issubdtype(img.dtype, np.floating):
        max_val = img.max()
        if max_val <= 1.0:
            out = (img * np.iinfo(np.uint16).max).astype(np.uint16)
        else:
            out = np.clip(img, 0, np.iinfo(np.uint16).max).astype(np.uint16)
    else:
        out = img.astype(np.uint16)
    imageio.imwrite(path, out)


def align_tiles(opt: argparse.Namespace) -> None:
    """Compute and apply translational alignment to montaged tile images.

    Alignment is performed in two passes:

    1. **Morphology channel**: For each (well, tile) group sorted by
       timepoint, consecutive-frame shifts are accumulated into a running
       offset.  The aligned morphology images are saved and the database
       is updated.
    2. **Other channels**: The pre-computed shifts from step 1 are applied
       to every remaining selected channel so that all channels share the
       same spatial registration.

    The computed shift dictionary is persisted as a pickle file under the
    ``AlignedMontages/`` output directory for later inspection.

    Args:
        opt: Namespace with experiment parameters including
            ``experiment``, ``morphology_channel``, ``chosen_channels``,
            well/timepoint/channel selection toggles, and
            ``img_norm_name``.
    """
    ops = Ops(opt)
    df = ops.get_df_for_training(['channeldata'])

    if 'timepoint' not in df.columns:
        raise KeyError("tiledata must contain 'timepoint' column")

    df = df[df['newimagemontage'].str.endswith('.tif', na=False)]

    db = Database()
    _, analysisdir = ops.get_raw_and_analysis_dir()
    print("analysisdir",analysisdir)
    out_root = os.path.join(analysisdir, 'AlignedMontages')
    os.makedirs(out_root, exist_ok=True)

    # Determine which non-morphology channels to align; None means all
    if opt.chosen_channels.lower() == 'all':
        other_channels = None
    else:
        other_channels = [c.strip() for c in opt.chosen_channels.split(',')]

    # Step 1: compute morphology shifts, once per (well, timepoint).
    # Every tiledata row for a given well/timepoint/channel shares the SAME
    # montage file (montage.py sets newimagemontage with a filter that omits
    # tile), so the previous group-by ['well','tile'] recomputed the identical
    # DFT shift + warp + write once per original tile and clobbered the same
    # output path. Here we compute the shift once per well/timepoint and then
    # update every tiledata row that shares it.
    morph = df[df['channel'] == opt.morphology_channel]
    new_shifts = {}
    for well, wgroup in morph.groupby('well'):
        # One representative montage per timepoint (dedup across tiles).
        reps = (wgroup.sort_values('timepoint')
                      .drop_duplicates(subset=['timepoint', 'newimagemontage']))
        prev = None
        running = np.zeros(2)
        for row in reps.itertuples():
            img = cv2.imread(row.newimagemontage, cv2.IMREAD_UNCHANGED)
            if img is None:
                print(f'⚠️ cv2.imread returned None for {row.newimagemontage}; '
                      f'skipping {well} T{row.timepoint}.')
                continue
            if prev is None:
                running = np.zeros(2)
            else:
                shift, method = cross_correlation_dft_combo(prev, img)
                running += shift
            prev = img.copy()
            new_shifts[(well, row.timepoint)] = running.copy()

            # save morphology aligned image once for this well+timepoint
            out_dir = os.path.join(out_root, well)
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, os.path.basename(row.newimagemontage).replace('.tif','_ALIGNED.tif'))
            # Negate and swap (dy,dx)->(tx,ty) to convert shift to inverse warp
            tform = transform.SimilarityTransform(translation=(-running[1], -running[0]))
            aligned = transform.warp(img, tform, preserve_range=True)
            save_image_uint16(out_path, aligned)

            # Update every tiledata row for this well/timepoint (morphology).
            id_matches = wgroup[wgroup['timepoint'] == row.timepoint]['tiledata_id'].tolist()
            for tdid in id_matches:
                db.update('tiledata', {'alignedmontagepath': out_path}, {'id': tdid})

    print(new_shifts)
    # save new shift dict
    ts = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    shift_path = os.path.join(out_root, f'calculated_shift_{ts}.dict')
    with open(shift_path,'wb') as f:
        pickle.dump(new_shifts,f)

    # Step 2: apply shifts to other channels, once per (well, timepoint,
    # channel) montage instead of once per original tile. Shifts depend only
    # on well+timepoint (derived from morphology), so all tiles of a channel
    # montage warp identically — dedup and update all sharing tiledata rows.
    others = df[df['channel'] != opt.morphology_channel]
    if other_channels is not None:
        others = others[others['channel'].isin(other_channels)]

    grouped = others.groupby(['well', 'timepoint', 'channel', 'newimagemontage'])
    for (well, timepoint, channel, montage_path), ogroup in tqdm(
            grouped, desc='Applying shifts to other channels'):
        key = (well, timepoint)
        if key not in new_shifts:
            continue
        running = new_shifts[key]
        img = cv2.imread(montage_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f'⚠️ cv2.imread returned None for {montage_path}; '
                  f'skipping {well} T{timepoint} {channel}.')
            continue
        tform = transform.SimilarityTransform(translation=(-running[1], -running[0]))
        aligned = transform.warp(img, tform, preserve_range=True)
        out_dir = os.path.join(out_root, well)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, os.path.basename(montage_path).replace('.tif','_ALIGNED.tif'))
        save_image_uint16(out_path, aligned)
        for tdid in ogroup['tiledata_id'].tolist():
            db.update('tiledata', {'alignedmontagepath': out_path}, {'id': tdid})

    print('Tile alignment complete.')

if __name__=='__main__':
    p = argparse.ArgumentParser(description='Align tiles using morphology shifts')
    p.add_argument('--experiment', required=True)
    p.add_argument('--morphology_channel', required=True)
    p.add_argument('--wells_toggle', default='include')
    p.add_argument('--timepoints_toggle', default='include')
    p.add_argument('--channels_toggle', default='include')
    p.add_argument('--chosen_wells', default='all')
    p.add_argument('--chosen_timepoints', default='all')
    p.add_argument('--chosen_channels', default='all',
                   help="Comma-separated channels or 'all' to apply shifts to every non-morphology channel.")
    p.add_argument('--tile', type=int, default=0)
    p.add_argument('--shift_dict', type=str, default='')
    p.add_argument('--img_norm_name', choices=['division','subtraction','identity'], default='subtraction')
    args = p.parse_args()
    align_tiles(args)
