#!/usr/bin/env python3
"""Ultrack-based cell tracking on montaged segmentation, as a robust alternative to
the greedy proximity/overlap tracker (which drops ~34% of motile fibroblasts).

Ultrack (royerlab/ultrack) solves tracking as an ILP over segmentation hypotheses,
handling appear/disappear/division and crowded/ambiguous linking. It runs in a
dedicated container (see ultrack.def) — NOT the main SIF.

Per well: stack the Cellpose seg-mask montages (alignedmontagemaskpath, morphology
channel) into a (T, Y, X) label array, derive foreground+contours, solve, and write
(1) per-timepoint tracked-label montages ``*_TRACKED.tif`` (pixel value = track_id;
the MASKTRACKED image Jeremy's GEDI2 measurement wants) and (2) an ultrack_tracks CSV
(well, track_id, timepoint, y, x, parent_id).

Solver: solver_name='' auto-selects Gurobi if the WLS license checks out, else the
bundled CBC — so it degrades gracefully on compute nodes without WLS network.

Usage:
    tracking_ultrack.py --experiment hevo-pmsG-1 --wells C6,F6 \
        --max_distance 150 --min_area 500 --max_area 40000 --solver '' \
        --work_dir /gladstone/finkbeiner/home/aholub/GXYTMPS/ULTRACK_WORK
"""
import argparse
import csv
import os
import shutil
from typing import List

import numpy as np
import imageio.v3 as iio
import tifffile
from sqlalchemy import create_engine, text

from ultrack import MainConfig, Tracker
from ultrack.utils import labels_to_contours


def get_engine():
    pw = None
    with open('/gladstone/finkbeiner/lab/GALAXY_INFO/pass.csv') as f:
        for r in csv.DictReader(f):
            if 'pw' in r:
                pw = r['pw'].strip(); break
    return create_engine(
        f'postgresql://postgres:{pw}@fb-postgres01.gladstone.internal:5432/galaxy')


def tracked_path(mask_path: str) -> str:
    """Derive the _TRACKED.tif path from a seg-mask montage path (matches the
    convention tracking_montage.py uses)."""
    for a in ('_MONTAGE_ALIGNED_ENCODED.tif', '_MONTAGE_ENCODED.tif',
              '_MONTAGE_ALIGNED.tif', '_MONTAGE.tif'):
        if a in mask_path:
            return mask_path.replace(a, '_TRACKED.tif')
    return mask_path.replace('.tif', '_TRACKED.tif')


def track_well(eng, exp_uuid: str, well: str, ch: str, opt) -> int:
    """Track one well with Ultrack; write _TRACKED.tif per timepoint + rows list."""
    with eng.connect() as c:
        rows = c.execute(text("""
            SELECT t.timepoint, t.alignedmontagemaskpath
            FROM tiledata t JOIN channeldata ch ON t.channeldata_id=ch.id
            JOIN welldata w ON t.welldata_id=w.id
            WHERE w.well=:w AND ch.channel=:c AND t.experimentdata_id=:x
              AND t.alignedmontagemaskpath IS NOT NULL
            ORDER BY t.timepoint"""), {'w': well, 'c': ch, 'x': exp_uuid}).fetchall()
    rows = [(int(tp), mp) for tp, mp in rows if mp and os.path.exists(mp)]
    if len(rows) < 2:
        print(f'  {well}: <2 timepoints with masks — skipping', flush=True)
        return 0
    tps = [tp for tp, _ in rows]
    masks = [iio.imread(mp).astype(np.int32) for _, mp in rows]
    H, W = masks[0].shape
    labels = np.stack(masks, axis=0)  # (T, Y, X)
    print(f'  {well}: {len(tps)} timepoints, montage {W}x{H}', flush=True)

    foreground, contours = labels_to_contours(labels, sigma=opt.sigma)

    wd = os.path.join(opt.work_dir, well)
    if os.path.exists(wd):
        shutil.rmtree(wd)
    os.makedirs(wd, exist_ok=True)
    config = MainConfig()
    config.data_config.working_dir = wd
    config.segmentation_config.min_area = opt.min_area
    config.segmentation_config.max_area = opt.max_area
    config.linking_config.max_distance = opt.max_distance
    config.linking_config.max_neighbors = opt.max_neighbors
    config.tracking_config.solver_name = opt.solver  # '' = auto (Gurobi if licensed, else CBC)

    tracker = Tracker(config)
    tracker.track(foreground=foreground, contours=contours, overwrite='all')
    tracks_df, _ = tracker.to_tracks_layer(include_parents=True)
    # relabeled tracked segmentation (T, Y, X); pixel value = track_id
    segments = np.asarray(tracker.to_zarr(tracks_df=tracks_df, overwrite=True))

    n_written = 0
    for ti, (tp, mp) in enumerate(rows):
        out = tracked_path(mp)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        tifffile.imwrite(out, segments[ti].astype(np.uint16), compression=None)
        n_written += 1
    # per-well tracks rows (map ultrack frame index t -> real timepoint)
    tcol = 't' if 't' in tracks_df.columns else 'time'
    idx2tp = {i: tp for i, tp in enumerate(tps)}
    tracks_df = tracks_df.copy()
    tracks_df['timepoint'] = tracks_df[tcol].map(idx2tp)
    tracks_df['well'] = well
    n_tracks = tracks_df['track_id'].nunique()
    print(f'  {well}: {n_tracks} tracks, wrote {n_written} _TRACKED.tif', flush=True)
    shutil.rmtree(wd, ignore_errors=True)
    return tracks_df


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--experiment', required=True)
    p.add_argument('--wells', default='all')
    p.add_argument('--morphology_channel', default='Epi-GFP16')
    p.add_argument('--max_distance', type=float, default=150.0,
                   help='Max displacement (px) between segments across frames.')
    p.add_argument('--max_neighbors', type=int, default=5)
    p.add_argument('--min_area', type=int, default=500)
    p.add_argument('--max_area', type=int, default=40000)
    p.add_argument('--sigma', type=float, default=1.0, help='labels_to_contours edge smoothing.')
    p.add_argument('--solver', default='', help="'' (auto: Gurobi if licensed else CBC) | 'GUROBI' | 'CBC'.")
    p.add_argument('--work_dir', default='/gladstone/finkbeiner/home/aholub/GXYTMPS/ULTRACK_WORK')
    p.add_argument('--out_csv', default='')
    args = p.parse_args()

    eng = get_engine()
    with eng.connect() as c:
        exp_uuid = c.execute(text('SELECT id FROM experimentdata WHERE experiment=:e'),
                             {'e': args.experiment}).scalar()
        adir = c.execute(text('SELECT analysisdir FROM experimentdata WHERE id=:x'),
                         {'x': exp_uuid}).scalar()
        if args.wells == 'all':
            ws = [r[0] for r in c.execute(text(
                'SELECT DISTINCT well FROM welldata WHERE experimentdata_id=:x ORDER BY 1'),
                {'x': exp_uuid}).fetchall()]
        else:
            ws = args.wells.split(',')
    os.makedirs(args.work_dir, exist_ok=True)
    import pandas as pd
    all_tracks = []
    for well in ws:
        r = track_well(eng, exp_uuid, well, args.morphology_channel, args)
        if isinstance(r, pd.DataFrame):
            all_tracks.append(r)
    if all_tracks:
        df = pd.concat(all_tracks, ignore_index=True)
        out_csv = args.out_csv or os.path.join(adir, f'{args.experiment}_ultrack_tracks.csv')
        df.to_csv(out_csv, index=False)
        print(f'wrote {len(df)} rows to {out_csv}', flush=True)
    print('ultrack tracking done.', flush=True)


if __name__ == '__main__':
    main()
