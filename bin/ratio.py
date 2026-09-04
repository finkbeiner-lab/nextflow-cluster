#!/usr/bin/env python
"""RGEDI ratio stage -> MRID ratio_output.csv (replaces MRID's ratio R stage).

Computes per-cell RFP/GFP ratio and live/dead labels, emitting MRID's exact
ratio_output.csv schema so the (R) logodds survival stage consumes it unchanged.

Two input modes:
  --input  percell_gated.csv  (our reworked pipeline: one row/cell with fitc_mean,
           rfp_mean, egfp_pos_*; EGFP+ transfected cells) -- the default path.
  --legacy cell_data.csv      (legacy Galaxy long format: MeasurementTag FITC/RFP,
           PixelIntensityMean) -- faithfully reproduces MRID's pairing for back-compat
           / validation (dedup by largest BlobArea; inner-merge FITC+RFP on
           timepoint_well_neuron; ratio = RFP mean / GFP mean).

Live/dead (MRID rule): live_guesses = 1 if ratio<live_thr, 0 if ratio>dead_thr,
cells between the thresholds dropped; classifier.score.live/dead are hard 0/1 on
live_thr. Thresholds come from ratio_threshold (auto per exp x tp, or manual).
"""
import argparse
import os
import numpy as np
import pandas as pd

import ratio_threshold as rt


def _wellkey(w):
    return str(w)


def load_gated(path, gate_col):
    df = pd.read_csv(path)
    df = df[df.get(gate_col, True)].copy() if gate_col in df.columns else df.copy()
    df['ratio'] = df['rfp_mean'] / (df['fitc_mean'] + 1e-9)
    df['Sci_WellID'] = df['well'].map(_wellkey)
    df['Timepoint'] = df['tp'].str.lstrip('Tt').astype(int)
    df['ObjectLabelsFound'] = df['label']
    df['filenames'] = (df['exp'].astype(str) + '_T' + df['Timepoint'].astype(str)
                       + '_' + df['Sci_WellID'] + '_' + df['label'].astype(str))
    keep = ['exp', 'tp', 'Timepoint', 'Sci_WellID', 'ObjectLabelsFound', 'filenames', 'ratio']
    for c in ('genotype', 'line'):          # carry through for a self-derived Sci_SampleID
        if c in df.columns:
            keep.append(c)
    return df[keep]


def load_legacy(path, exp_name):
    d = pd.read_csv(path)
    # dedup duplicate objects, keep largest BlobArea (MRID deduplicate_by_small_deletion)
    d['_k'] = (d.Sci_PlateID.astype(str) + '_' + d.Sci_WellID.astype(str) + '_'
               + d.ObjectLabelsFound.astype(str) + '_' + d.Timepoint.astype(str)
               + '_' + d.MeasurementTag.astype(str))
    d = d.sort_values('BlobArea', ascending=False).drop_duplicates('_k')
    d['twn'] = (d.Timepoint.astype(str) + '_' + d.Sci_WellID.astype(str) + '_'
                + d.ObjectLabelsFound.astype(str))
    fitc = d[d.MeasurementTag.str.contains('FITC|GFP', case=False, regex=True)][
        ['twn', 'PixelIntensityMean']].rename(columns={'PixelIntensityMean': 'GFP'})
    rfp = d[d.MeasurementTag.str.contains('RFP', case=False, regex=True)].copy()
    m = rfp.merge(fitc, on='twn', how='inner')
    m['ratio'] = m['PixelIntensityMean'] / (m['GFP'] + 1e-9)
    m['exp'] = exp_name
    m['tp'] = 'T' + m['Timepoint'].astype(str)
    m['filenames'] = ('PID_' + exp_name + '_T' + m.Timepoint.astype(str) + '_'
                      + m.Sci_WellID.astype(str) + '_' + m.ObjectLabelsFound.astype(str))
    return m[['exp', 'tp', 'Timepoint', 'Sci_WellID', 'ObjectLabelsFound', 'filenames', 'ratio']]


def assign_livedead(df, gates):
    df = df.copy()
    lv = df.apply(lambda r: gates[(r.exp, r.tp)][0], axis=1)
    dd = df.apply(lambda r: gates[(r.exp, r.tp)][1], axis=1)
    df['classifier.score.live'] = (df.ratio <= lv).astype(float)
    df['classifier.score.dead'] = (df.ratio > lv).astype(float)
    df['live_guesses'] = np.where(df.ratio < lv, 1.0, np.where(df.ratio > dd, 0.0, np.nan))
    return df.dropna(subset=['live_guesses'])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', help='percell_gated.csv (our pipeline)')
    ap.add_argument('--legacy', help='legacy cell_data.csv')
    ap.add_argument('--plate-layout', default=None,
                    help='Sci_WellID,Sci_SampleID,Drug csv; if omitted, Sci_SampleID is derived '
                         'as <genotype>_<line> from the per-cell data (MRID compare_class then '
                         'matches the genotype substring), Drug = "No drug"')
    ap.add_argument('--expt-name', default='EXPT')
    ap.add_argument('--gate-col', default='egfp_pos_ksd')
    ap.add_argument('--output-path', default='.')
    ap.add_argument('--method', default='auto', choices=['auto', 'manual'])
    ap.add_argument('--anchor-exp', default='XDP5')
    ap.add_argument('--anchor', type=float, default=0.07)
    ap.add_argument('--live', type=float, default=None)
    ap.add_argument('--dead', type=float, default=None)
    ap.add_argument('--ambiguous-margin', type=float, default=0.0)
    ap.add_argument('--hours-per-tp', type=float, default=24.0,
                    help='uniform hours between timepoints for timepoint.csv (if no explicit csv)')
    ap.add_argument('--timepoint-hours-csv', default=None,
                    help='explicit Timepoint,Hour csv (overrides --hours-per-tp)')
    a = ap.parse_args()
    os.makedirs(a.output_path, exist_ok=True)

    cells = load_gated(a.input, a.gate_col) if a.input else load_legacy(a.legacy, a.expt_name)
    if a.plate_layout:
        layout = pd.read_csv(a.plate_layout)
        layout.columns = [c.strip().lstrip('﻿') for c in layout.columns]
        if 'Drug' not in layout.columns:
            layout['Drug'] = 'No drug'
        cells = cells.merge(layout[['Sci_WellID', 'Sci_SampleID', 'Drug']], on='Sci_WellID', how='left')
    else:   # derive Sci_SampleID = <genotype>_<line> so MRID's class matcher finds the genotype
        geno = cells['genotype'].astype(str) if 'genotype' in cells.columns else ''
        line = cells['line'].astype(str) if 'line' in cells.columns else ''
        cells['Sci_SampleID'] = (geno + '_' + line).str.strip('_')
        cells['Drug'] = 'No drug'

    gates, k = rt.calibrate(cells, a.method, a.anchor_exp, a.anchor, a.live, a.dead, a.ambiguous_margin)
    pd.DataFrame([dict(exp=e, tp=t, live_threshold=round(lv, 6), dead_threshold=round(dd, 6), reliable=rel)
                  for (e, t), (lv, dd, rel) in gates.items()]).to_csv(
        os.path.join(a.output_path, 'livedead_thresholds.csv'), index=False)

    out = assign_livedead(cells, gates)
    cols = ['filenames', 'Sci_WellID', 'Sci_SampleID', 'Drug', 'Timepoint', 'ratio',
            'live_guesses', 'classifier.score.live', 'classifier.score.dead']
    out[cols].to_csv(os.path.join(a.output_path, f'{a.expt_name}_ratio_output.csv'), index=True)

    # cell counts (per condition, per well with empty-well NA backfill)
    cc = cells.groupby(['Sci_SampleID', 'Drug', 'Timepoint']).size().reset_index(name='n')
    cc.to_csv(os.path.join(a.output_path, 'cell_count.csv'), index=False)
    ccw = cells.groupby(['Sci_WellID', 'Timepoint']).size().reset_index(name='n')
    all_wells = cells.Sci_WellID.unique()
    full = pd.MultiIndex.from_product([all_wells, sorted(cells.Timepoint.unique())],
                                      names=['Sci_WellID', 'Timepoint']).to_frame(index=False)
    ccw = full.merge(ccw, on=['Sci_WellID', 'Timepoint'], how='left')
    ccw.to_csv(os.path.join(a.output_path, 'cell_count_well.csv'), index=False)

    # timepoint.csv (Timepoint, Hour) consumed by the logodds stage
    tps = sorted(cells.Timepoint.unique())
    if a.timepoint_hours_csv:
        th = pd.read_csv(a.timepoint_hours_csv)
    else:
        th = pd.DataFrame({'Timepoint': tps, 'Hour': [(t - tps[0]) * a.hours_per_tp for t in tps]})
    th.to_csv(os.path.join(a.output_path, 'timepoint.csv'), index=False)

    print(f"{len(out)} live/dead-called cells (of {len(cells)} paired). "
          f"method={a.method}, wrote {a.expt_name}_ratio_output.csv + cell counts to {a.output_path}")


if __name__ == '__main__':
    main()
