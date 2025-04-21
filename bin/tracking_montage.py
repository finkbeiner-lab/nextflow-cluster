#!/opt/conda/bin/python

import cv2
import numpy as np
import logging
import argparse
import os
from sql import Database
import pandas as pd
import collections

logger = logging.getLogger("TrackingDB")
logging.basicConfig(level=logging.INFO)

class Cell:
    """A class that makes cells from contours or masks."""
    def __init__(self, cnt):
        self.cnt = cnt

    def __repr__(self):
        center, _ = self.get_circle()
        return f"Cell instance at {center}"

    def get_circle(self):
        center, radius = cv2.minEnclosingCircle(self.cnt)
        return center, radius

    def evaluate_overlap(self, circle2):
        c2, r2 = circle2
        c1, r1 = self.get_circle()
        dist = np.hypot(c1[0] - c2[0], c1[1] - c2[1])
        return dist < (r1 + r2) * 0.8

    def evaluate_dist(self, circle2):
        c2, _ = circle2
        c1, _ = self.get_circle()
        return np.hypot(c1[0] - c2[0], c1[1] - c2[1])


def sort_cell_info_by_index(time_dict, time_list):
    for tp in time_list:
        time_dict[tp] = sorted(
            [(int(idx), obj) for idx, obj in time_dict[tp]],
            key=lambda x: x[0]
        )
    return time_dict


def populate_cell_ind_overlap(time_dict, time_list):
    first = time_list[0]
    for i, rec in enumerate(time_dict[first], 1): rec[0] = i
    counter = len(time_dict[first]) + 1
    for i in range(1, len(time_list)):
        curr, prev = time_list[i], time_list[i-1]
        for rec in time_dict[curr]:
            circ = rec[1].get_circle(); matched = False
            for p in time_dict[prev]:
                if p[1].evaluate_overlap(circ): rec[0] = p[0]; matched = True; break
            if not matched: rec[0] = counter; counter += 1
    return time_dict


def populate_cell_ind_closest(time_dict, time_list, max_dist=100):
    first = time_list[0]
    for i, rec in enumerate(time_dict[first],1): rec[0] = i
    counter = len(time_dict[first]) + 1
    for i in range(1, len(time_list)):
        curr, prev = time_list[i], time_list[i-1]
        for rec in time_dict[curr]:
            circ = rec[1].get_circle(); best = float('inf')
            for p in time_dict[prev]:
                d = p[1].evaluate_dist(circ)
                if d < best: best, rec[0] = d, p[0]
            if best > max_dist: rec[0] = counter; counter += 1
    return time_dict

class MontageDBTracker:
    def __init__(self, experiment, track_type, max_dist):
        self.Db = Database()
        self.experiment = experiment
        self.track_type = track_type
        self.max_dist = max_dist
        logger.info(f"Initialized MontageDBTracker for experiment {experiment}")

    def gather_encoded_from_db(self, wells, channel_marker="_ENCODED_MONTAGE"):
        """
        Fetch montage paths from the `welldata` table via Ops.get_welldata_df().
        Returns: dict of {well: OrderedDict(timepoint: [[label, Cell], ...])}
        """
        from db_util import Ops
        import argparse

        opt_inner = argparse.Namespace(
            experiment=self.experiment,
            chosen_wells=','.join(wells),
            wells_toggle='include',
            chosen_timepoints='',
            timepoints_toggle='include',
            chosen_channels='all',
            channels_toggle='include',
            tile=0
        )
        
        op = Ops(opt_inner)
        # Pull tiledata (including your newmaskmontage column)
        tiledata_df = op.get_tiledata_df()
        # 1. Filter to only your newmaskmontage rows in the wells you care about
        df = tiledata_df[
            tiledata_df['well'].isin(wells) &
            tiledata_df['newmaskmontage'].str.contains(channel_marker, na=False)
        ]

        # 2. Collapse to one mask per (well, timepoint) — take the first path in each group
        df = (
            df
            .groupby(['well','timepoint'], as_index=False)
            .agg({'newmaskmontage':'first'})
        )
        # # Only rows for our wells with the encoded‐montage suffix
        # df = tiledata_df[
        #     tiledata_df['well'].isin(wells) &
        #     tiledata_df['newmaskmontage'].str.contains(channel_marker, na=False)
        # ]
#Original
        # welldata_df = op.get_welldata_df()

        # df = welldata_df[
        #     welldata_df.well.isin(wells) &
        #     welldata_df.maskmontage.str.contains(channel_marker, na=False)
        # ]

        results = {}
        for well in wells:
            df_w = df[df['well'] == well]
            if df_w.empty:
                logger.warning(f"No encoded masks for well {well}")
                continue
            time_dict = collections.OrderedDict()
            for _, row in df_w.iterrows():
                mask_path = row['newmaskmontage']
                tp_label = os.path.basename(mask_path).split('_')[2]
                tp = int(tp_label.lstrip('T')) if tp_label.startswith('T') else int(tp_label)
                entries = time_dict.setdefault(tp, [])
                mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
                labels = np.unique(mask); labels = labels[labels > 0]
                for lbl in labels:
                    bin_mask = (mask == lbl).astype(np.uint8) * 255
                    cnts, _ = cv2.findContours(bin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if cnts:
                        cnt = max(cnts, key=cv2.contourArea)
                        entries.append([lbl, Cell(cnt)])
            results[well] = time_dict
        return results

    def run(self, wells):
        exp_id = self.Db.get_table_uuid('experimentdata', {'experiment': self.experiment})
        all_wells = self.gather_encoded_from_db(wells)
        for well, time_dict in all_wells.items():
            welldata_id = self.Db.get_table_uuid(
                'welldata', {'experimentdata_id': exp_id, 'well': well}
            )
            tps = sorted(time_dict.keys())
            tracked = (
                populate_cell_ind_overlap(time_dict, tps)
                if self.track_type == 'overlap'
                else populate_cell_ind_closest(time_dict, tps, max_dist=self.max_dist)
            )
            sorted_td = sort_cell_info_by_index(tracked, tps)
                        # ─── INSERT SUMMARY HERE ───
            labels_per_tp = { tp: {idx for idx,_ in sorted_td[tp]} for tp in tps }
            total    = len(labels_per_tp[tps[0]])
            in_all   = set(labels_per_tp[tps[0]])
            for tp in tps[1:]:
                in_all &= labels_per_tp[tp]
            count_all     = len(in_all)
            count_missing = total - count_all
            pct           = (count_all/total)*100 if total else 0.0

            print(f"Summary for {well}:")
            print(f"  total unique cells: {total}")
            print(f"  cells in all {len(tps)} TPs: {count_all}")
            print(f"  cells missing ≥1 TP: {count_missing}")
            print(f"  % tracked in all {len(tps)} TPs: {pct:.1f}%")
            # ────────────────────────────

            # Update DB with new cellids per timepoint
            for tp, recs in sorted_td.items():
                # find corresponding tiledata_id for this timepoint
                tiledata_id = self.Db.get_table_uuid(
                    'tiledata',
                    {'experimentdata_id': exp_id,
                     'welldata_id': welldata_id,
                     'timepoint': tp}
                )
                for new_id, _ in recs:
                    self.Db.update(
                        'celldata',
                        update_dct={'cellid': int(new_id)},
                        kwargs={
                            'tiledata_id': tiledata_id,
                            'randomcellid': int(new_id)
                        }
                    )

            print(f"Well {well}:")
            for tp, recs in sorted_td.items():
                ids = [idx for idx, _ in recs]
                print(f"  T{tp}: {len(ids)} cells, IDs = {ids}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Track montage masks stored in DB")
    parser.add_argument('--experiment', required=True, help='Experiment name')
    parser.add_argument('--track_type', choices=['overlap','proximity'], default='overlap', help='Tracking method')
    parser.add_argument('--max_dist', type=int, default=100, help='Max distance for proximity')
    parser.add_argument('--wells', required=True, help='Comma-separated list of wells, e.g. A1,B1')
    args = parser.parse_args()
    wells = [w.strip() for w in args.wells.split(',')]
    tracker = MontageDBTracker(args.experiment, args.track_type, args.max_dist)
    tracker.run(wells)
