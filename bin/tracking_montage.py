#!/opt/conda/bin/python

import cv2
import numpy as np
import logging
import argparse
import os
from db_util import Ops
from sql import Database
import pandas as pd
import collections
import ast  # Safe way to evaluate a string representation of a list


logger = logging.getLogger("TrackingDB")
logging.basicConfig(level=logging.INFO)

class Cell:
    """A class that makes cells from contours or masks."""
    def __init__(self, cnt, randomcellid_montage=None):
        self.cnt = cnt
        self.randomcellid_montage = randomcellid_montage

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
    for i, rec in enumerate(time_dict[first], 1):
        rec[0] = i
    counter = len(time_dict[first]) + 1

    for i in range(1, len(time_list)):
        curr, prev = time_list[i], time_list[i - 1]
        for rec in time_dict[curr]:
            circ = rec[1].get_circle()
            matched = False
            for p in time_dict[prev]:
                if p[1].evaluate_overlap(circ):
                    rec[0] = p[0]
                    matched = True
                    break
            if not matched:
                rec[0] = counter
                counter += 1

    return time_dict

def populate_cell_ind_closest(time_dict, time_list, max_dist=100):
    first = time_list[0]
    for i, rec in enumerate(time_dict[first], 1):
        rec[0] = i
    counter = len(time_dict[first]) + 1

    for i in range(1, len(time_list)):
        curr, prev = time_list[i], time_list[i - 1]
        for rec in time_dict[curr]:
            circ = rec[1].get_circle()
            best = float('inf')
            for p in time_dict[prev]:
                d = p[1].evaluate_dist(circ)
                if d < best:
                    best, rec[0] = d, p[0]
            if best > max_dist:
                rec[0] = counter
                counter += 1

    return time_dict

class MontageDBTracker:
    def __init__(self, experiment, track_type, max_dist, target_channel):
        self.Db = Database()
        self.experiment = experiment
        self.track_type = track_type
        self.max_dist = max_dist
        self.target_channel = target_channel.split(',')



        self.analysisdir = ""
        logger.info(f"Initialized MontageDBTracker for experiment {experiment}")

    def gather_encoded_from_db(self, wells, channel_marker="_MONTAGE_ALIGNED_ENCODED"):
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
        tiledata_df = op.get_tiledata_df()

        df = (
            tiledata_df
            [tiledata_df['well'].isin(wells)]
            [tiledata_df['alignedmontagemaskpath'].str.contains(channel_marker, na=False)]
        )

        df = df.groupby(['well', 'timepoint'], as_index=False).agg({'alignedmontagemaskpath': 'first'})

        results = {}
        for well in wells:
            logger.info(f"Tracking well {well}")
            df_w = df[df['well'] == well]
            if df_w.empty:
                logger.warning(f"No encoded masks for well {well}")
                continue

            time_dict = collections.OrderedDict()
            for _, row in df_w.iterrows():
                mask_path = row['alignedmontagemaskpath']
                tp_label = os.path.basename(mask_path).split('_')[2]
                tp = int(tp_label.lstrip('T')) if tp_label.startswith('T') else int(tp_label)

                mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
                labels = np.unique(mask)
                labels = labels[labels > 0]

                entries = time_dict.setdefault(tp, [])
                for lbl in labels:
                    bin_mask = (mask == lbl).astype(np.uint8) * 255
                    cnts, _ = cv2.findContours(bin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if cnts:
                        cnt = max(cnts, key=cv2.contourArea)
                        entries.append([None, Cell(cnt, randomcellid_montage=lbl)])

            if time_dict:
                time_dict = collections.OrderedDict(sorted(time_dict.items()))
                results[well] = time_dict

        return results, df, tiledata_df

    def get_cell_props(self, mask, image_stack, label):
        bin_mask = (mask == label).astype(np.uint8)
        M = cv2.moments(bin_mask)
        if M["m00"] == 0:
            return None
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        area = cv2.countNonZero(bin_mask)

        intensities = {}
        for chan, img in image_stack.items():
            intensities[chan] = float(np.mean(img[bin_mask > 0]))

        return cx, cy, area, intensities

    def run(self, wells):
        exp_id = self.Db.get_table_uuid('experimentdata', {'experiment': self.experiment})
        all_wells, df, tiledata_df = self.gather_encoded_from_db(wells)

        out_records = []

        for well, time_dict in all_wells.items():
            welldata_id = self.Db.get_table_uuid('welldata', {'experimentdata_id': exp_id, 'well': well})
            tps = sorted(time_dict.keys())
            tracked = (
                populate_cell_ind_overlap(time_dict, tps)
                if self.track_type == 'overlap'
                else populate_cell_ind_closest(time_dict, tps, max_dist=self.max_dist)
            )
            sorted_td = sort_cell_info_by_index(tracked, tps)

            for tp, recs in sorted_td.items():
                df_wtp = df[(df['well'] == well) & (df['timepoint'] == tp)]
                if df_wtp.empty:
                    continue
                mask_path = df_wtp['alignedmontagemaskpath'].iloc[0]
              
                self.analysisdir = os.path.dirname(mask_path.split('CellMasksMontage')[0])
                mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
                base_path = mask_path.replace('_MONTAGE_ALIGNED_ENCODED.tif', '')
                
                tracked_mask = np.zeros_like(mask, dtype=np.uint16)
                for new_id, cell in recs:
                    if cell.randomcellid_montage is None:
                        continue
                    tracked_mask[mask == cell.randomcellid_montage] = new_id

                # Construct the output path for the tracked TIFF
            
                tracked_folder = os.path.join(self.analysisdir, 'TrackedCellMasksMontage', well)
               
                os.makedirs(tracked_folder, exist_ok=True)

                tracked_filename = os.path.basename(mask_path).replace(
                    '_MONTAGE_ALIGNED_ENCODED.tif', '_TRACKED.tif'
                )
                tracked_path = os.path.join(tracked_folder, tracked_filename)

                # Save the tracked mask
                cv2.imwrite(tracked_path, tracked_mask)
                logger.info(f"Saved tracked mask: {tracked_path}")
                channel_imgs = {}
                aligned_img_dir = f"/gladstone/finkbeiner/steve/WeiyiLiu/GXYTMP/{self.experiment}/AlignedMontages/{well}"

                # Target channenl only for calculate the intensities, not tracking

                # for ch in self.target_channel :
                #     aligned_path = mask_path.replace('/CellMasksMontage/', '/AlignedMontages/')
    
                #     # Step 2: Replace "_MONTAGE_ALIGNED_ENCODED.tif" with "_MONTAGE_ALIGNED.tif"
                #     aligned_path = aligned_path.replace('_MONTAGE_ALIGNED_ENCODED.tif', '_MONTAGE_ALIGNED.tif')

                #     img_path = os.path.join(aligned_img_dir, aligned_path)
                    
                #     if os.path.exists(img_path):
                #         channel_imgs[ch] = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
                #     else:
                #         logger.warning(f"[WARN] Missing aligned image for {ch} at: {img_path}")

                # import pdb
                # pdb.set_trace()
                # Split comma-separated string into a list
                
                for ch in self.target_channel:
                    print("Intensity calcuation ", ch)
                  
                    aligned_path = mask_path.replace('/CellMasksMontage/', '/AlignedMontages/')

                    # Ensure the correct channel name is in the filename
                    # Extract the filename part
                    filename = os.path.basename(aligned_path)
                    dirname = os.path.dirname(aligned_path)

                    # Replace the channel name in the filename (the part before "_MONTAGE_ALIGNED_ENCODED.tif")
                    parts = filename.split('_')
                    # The channel is usually at index -5; replace it safely
                    for i, part in enumerate(parts):
                        if part.startswith('Epi-') or part.startswith('DAPI') or part.startswith('Cy'):
                            parts[i] = ch
                            break
                    filename = '_'.join(parts)

                    # Rebuild the aligned_path with corrected channel
                    aligned_path = os.path.join(dirname, filename)

                    # Then convert "_MONTAGE_ALIGNED_ENCODED.tif" to "_MONTAGE_ALIGNED.tif"
                    aligned_path = aligned_path.replace('_MONTAGE_ALIGNED_ENCODED.tif', '_MONTAGE_ALIGNED.tif')

                    img_path = os.path.join(aligned_img_dir, aligned_path)

                    if os.path.exists(img_path):
                        channel_imgs[ch] = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
                    else:
                        logger.warning(f"[WARN] Missing aligned image for {ch} at: {img_path}")



                for new_id, cell in recs:
                    props = self.get_cell_props(mask, channel_imgs, cell.randomcellid_montage)
                    if props is None:
                        continue
                    cx, cy, area, intensities = props
                    row = {
                        'experiment': self.experiment,
                        'well': well,
                        'timepoint': tp,
                        'tracked_id': new_id,
                        'centroid_x': cx,
                        'centroid_y': cy,
                        'area': area
                    }
                    row.update({f"intensity_mean_{k}": v for k, v in intensities.items()})
                    out_records.append(row)

        out_df = pd.DataFrame(out_records)
        outfile = os.path.join(self.analysisdir, f"{self.experiment}_tracked_montage_summary.csv")
        out_df.to_csv(outfile, index=False)
        logger.info(f"Wrote tracked data to {outfile}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Track montage masks stored in DB")
    parser.add_argument('--experiment', required=True, help='Experiment name')
    parser.add_argument('--track_type', choices=['overlap','proximity'], default='overlap', help='Tracking method')
    parser.add_argument('--max_dist', type=int, default=100, help='Max distance for proximity')
    parser.add_argument('--wells', required=True, help='Comma-separated list of wells, e.g. A1,B1')
    parser.add_argument("--target_channel",  type=str, default='Cy5',
                        dest="target_channel",
                        help="Get intensity of this channel.")
    args = parser.parse_args()

    wells = [w.strip() for w in args.wells.split(',')]
    tracker = MontageDBTracker(args.experiment, args.track_type, args.max_dist, args.target_channel)
    tracker.run(wells)
