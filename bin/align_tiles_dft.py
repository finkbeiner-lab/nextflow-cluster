#!/opt/conda/bin/python
"""
Aligns tile images using phase cross-correlation and DFT fallback.
Computes shifts only on the morphology channel; applies those shifts to other channels as defined by params.chosen_channels (or all if "all").
Updates alignedtilepath in the database.
"""

import os
import cv2
import argparse
import numpy as np
from skimage import transform
from skimage.registration import phase_cross_correlation
import imreg_dft as ird
from sql import Database
from tqdm import tqdm
import imageio
import pickle
from db_util import Ops
import datetime
from normalization import Normalize


def cross_correlation_dft_combo(ref_img, img):
    """
    Measure shift between ref_img and img using phase correlation first, fallback to DFT.
    Returns: (shift_vector, method_string).
    """
    fix_by_dft = False
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

        # Retry if shift too large
        if (abs(shift[0]) >= y_thresh or abs(shift[1]) >= x_thresh) and not fix_by_dft:
            fix_by_dft = True
            continue
        break

    return shift, method


def save_image_uint16(path, img):
    """
    Save img as uint16, scaling float images appropriately.
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


def align_tiles(opt):
    ops = Ops(opt)
    df = ops.get_df_for_training(['channeldata'])
    Norm = Normalize(opt)

    if 'timepoint' not in df.columns:
        raise KeyError("tiledata must contain 'timepoint' column")

    df = df[df['filename'].str.endswith('.tif')]

    # precompute backgrounds per well/timepoint
    for (well, timepoint), sub in df.groupby(['well','timepoint']):
        Norm.get_background_image(sub, well, timepoint)

    db = Database()
    _, analysisdir = ops.get_raw_and_analysis_dir()
    out_root = os.path.join(analysisdir, 'AlignedTiles')
    os.makedirs(out_root, exist_ok=True)

    # Determine channels to align beyond morphology
    if opt.chosen_channels.lower() == 'all':
        other_channels = None  # means all except morphology
    else:
        other_channels = [c.strip() for c in opt.chosen_channels.split(',')]

    # Step 1: compute morphology shifts
    morph = df[df['channel'] == opt.morphology_channel]
    new_shifts = {}
    for (well, tile), group in morph.groupby(['well','tile']):
        group = group.sort_values('timepoint')
        prev = None
        running = np.zeros(2)
        for idx, row in enumerate(group.itertuples()):
            raw = cv2.imread(row.filename, cv2.IMREAD_UNCHANGED)
            img = Norm.image_bg_correction[opt.img_norm_name](raw, row.well, row.timepoint)
            if idx == 0:
                running = np.zeros(2)
                prev = img.copy()
            else:
                shift, method = cross_correlation_dft_combo(prev, img)
                running += shift
                prev = img.copy()
            new_shifts[(well, tile, row.timepoint)] = running.copy()

            # save morphology aligned image
            out_dir = os.path.join(out_root, well)
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, os.path.basename(row.filename).replace('.tif','_ALIGNED.tif'))
            tform = transform.SimilarityTransform(translation=(-running[1], -running[0]))
            aligned = transform.warp(img, tform, preserve_range=True)
            save_image_uint16(out_path, aligned)
            db.update('tiledata', {'alignedtilepath': out_path}, {'id': row.tiledata_id})

    # save new shift dict
    ts = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    shift_path = os.path.join(out_root, f'calculated_shift_{ts}.dict')
    with open(shift_path,'wb') as f:
        pickle.dump(new_shifts,f)

    # Step 2: apply shifts to other channels if selected
    others = df[df['channel'] != opt.morphology_channel]
    if other_channels is not None:
        others = others[others['channel'].isin(other_channels)]

    for row in tqdm(others.itertuples(), desc='Applying shifts to other channels'):
        key = (row.well, row.tile, row.timepoint)
        if key not in new_shifts:
            continue
        running = new_shifts[key]
        raw = cv2.imread(row.filename, cv2.IMREAD_UNCHANGED)
        img = Norm.image_bg_correction[opt.img_norm_name](raw, row.well, row.timepoint)
        tform = transform.SimilarityTransform(translation=(-running[1], -running[0]))
        aligned = transform.warp(img, tform, preserve_range=True)
        out_dir = os.path.join(out_root, row.well)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, os.path.basename(row.filename).replace('.tif','_ALIGNED.tif'))
        save_image_uint16(out_path, aligned)
        db.update('tiledata', {'alignedtilepath': out_path}, {'id': row.tiledata_id})

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
