#!/usr/bin/env python
"""Cross-timepoint tracking of (gated) Cellpose cells within a well (no GPU).

Links each cell to itself across timepoints so RGEDI intensity can be followed
per cell over time (what OG's tracking step did, rebuilt for Cellpose masks).
Per (exp, well): order timepoints, estimate a robust global drift between
consecutive frames, then Hungarian-match cell centroids within max_dist,
allowing births, deaths, and short gap-closing.

Input: a per-cell CSV with columns exp, well, tp, cy, cx (+ any readouts carried
through, e.g. fitc_mean, rfp_mean, line, genotype). Use the EGFP+ cells
(filter egfp_pos_ksd upstream) so tracks are transfected cells.
Output: <in>_tracked.csv (input rows + track_id) and tracks_summary.csv
(one row per track: span + readout trajectory).

Usage: python track_cells.py percell_gated_egfppos.csv [--max-dist 40] [--max-gap 1] [--no-drift]
"""
import argparse
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment


def tp_int(tp):
    return int(str(tp).lstrip('Tt'))


def estimate_drift(prev_xy, cur_xy, radius):
    """Robust global shift (cur - prev) from mutual nearest neighbours."""
    if len(prev_xy) == 0 or len(cur_xy) == 0:
        return np.zeros(2)
    disp = []
    for p in prev_xy:
        d = np.linalg.norm(cur_xy - p, axis=1)
        j = np.argmin(d)
        if d[j] <= radius:
            disp.append(cur_xy[j] - p)
    return np.median(disp, axis=0) if disp else np.zeros(2)


def track_well(df, max_dist, max_gap, drift):
    """Assign track_id to each cell row of one well across timepoints."""
    df = df.copy()
    df['track_id'] = -1
    tps = sorted(df.tp_i.unique())
    # active tracks: {tid: (last_tp, last_xy)}
    active, next_id = {}, 0
    for ti in tps:
        cur = df[df.tp_i == ti]
        cur_idx = cur.index.to_numpy()
        cur_xy = cur[['cy', 'cx']].to_numpy(dtype=float)
        # candidate tracks still alive within the gap window
        cand = [(tid, xy) for tid, (lt, xy) in active.items() if ti - lt <= max_gap + 1]
        if cand and len(cur_xy):
            tids = [c[0] for c in cand]
            prev_xy = np.array([c[1] for c in cand], dtype=float)
            sh = estimate_drift(prev_xy, cur_xy, max_dist * 2) if drift else np.zeros(2)
            cost = np.linalg.norm((prev_xy[:, None, :] + sh) - cur_xy[None, :, :], axis=2)
            big = max_dist + 1
            cost_c = np.where(cost <= max_dist, cost, big)
            ri, ci = linear_sum_assignment(cost_c)
            matched_cur = set()
            for r, c in zip(ri, ci):
                if cost_c[r, c] <= max_dist:
                    tid = tids[r]
                    df.at[cur_idx[c], 'track_id'] = tid
                    active[tid] = (ti, cur_xy[c])
                    matched_cur.add(c)
            new_cols = [c for c in range(len(cur_xy)) if c not in matched_cur]
        else:
            new_cols = list(range(len(cur_xy)))
        for c in new_cols:
            df.at[cur_idx[c], 'track_id'] = next_id
            active[next_id] = (ti, cur_xy[c])
            next_id += 1
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('percell')
    ap.add_argument('--max-dist', type=float, default=40.0, help='max centroid move between frames (px)')
    ap.add_argument('--max-gap', type=int, default=1, help='frames a cell may vanish and still relink')
    ap.add_argument('--no-drift', action='store_true', help='disable global drift correction')
    a = ap.parse_args()

    df = pd.read_csv(a.percell)
    df['tp_i'] = df['tp'].map(tp_int)
    out = []
    for (exp, well), g in df.groupby(['exp', 'well']):
        t = track_well(g, a.max_dist, a.max_gap, not a.no_drift)
        # make track_id globally unique across wells
        t['track_id'] = exp + '_' + well + '_' + t['track_id'].astype(str)
        out.append(t)
    tracked = pd.concat(out, ignore_index=True)
    base = a.percell.rsplit('.', 1)[0]
    tracked.to_csv(f"{base}_tracked.csv", index=False)

    # per-track summary
    readouts = [c for c in ['rfp_mean', 'fitc_mean'] if c in tracked.columns]
    agg = {'tp_i': ['min', 'max', 'count']}
    summ = tracked.groupby(['exp', 'well', 'track_id']).agg(
        first_tp=('tp_i', 'min'), last_tp=('tp_i', 'max'), n_tp=('tp_i', 'count')).reset_index()
    for r in readouts:
        wide = tracked.pivot_table(index='track_id', columns='tp_i', values=r, aggfunc='mean')
        wide.columns = [f"{r}_T{c}" for c in wide.columns]
        summ = summ.merge(wide.reset_index(), on='track_id', how='left')
    summ.to_csv("tracks_summary.csv", index=False)

    n = tracked.track_id.nunique()
    multi = (tracked.groupby('track_id').size() >= 2).sum()
    print(f"cells: {len(tracked)}   tracks: {n}   multi-timepoint tracks: {multi} "
          f"({100*multi/max(n,1):.0f}%)")
    print(f"wrote {base}_tracked.csv (per-cell + track_id) and tracks_summary.csv (per-track trajectory)")
    print("NOTE: real tracking needs a CONTIGUOUS timepoint series per well (T1..Tn); "
          "the go/no-go eval's T2+T9 is a 2-frame sanity case only.")


if __name__ == '__main__':
    main()
