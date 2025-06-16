import collections
import datetime
import shutil
import argparse
import cv2
import numpy as np
import os
import scipy.stats as stat
import utils
from segmentation import filter_contours, find_cells


class Cell:
    """A class that makes cells from contours."""
    def __init__(self, cnt, ch_images=None):
        self.cnt = cnt
        self.all_ch_int_stats = None
        if ch_images:
            self.collect_all_ch_intensities(ch_images)

    def __repr__(self):
        return f"Cell instance ({self.get_circle()[0]} center)"

    def get_circle(self):
        center, radius = cv2.minEnclosingCircle(self.cnt)
        return center, radius

    def evaluate_overlap(self, circle2):
        center2, radius2 = circle2
        center1, radius1 = self.get_circle()
        dist = np.hypot(center1[0]-center2[0], center1[1]-center2[1])
        return dist < (radius1 + radius2) * 0.8

    def evaluate_dist(self, circle2):
        center2, _ = circle2
        center1, _ = self.get_circle()
        return np.hypot(center1[0]-center2[0], center1[1]-center2[1])

    def find_cnt_int_dist(self, img):
        mask = np.zeros(img.shape[:2], np.uint8)
        cv2.drawContours(mask, [self.cnt], 0, 256, -1)
        vals = img[mask>0]
        stats = {
            'min': vals.min(), 'max': vals.max(), 'mean': float(vals.mean()),
            'std': float(vals.std()), 'sum': float(vals.sum())
        }
        # percentiles and skew/kurtosis can be added as needed
        return stats

    def collect_all_ch_intensities(self, ch_images):
        self.all_ch_int_stats = {}
        for color, frames in ch_images.items():
            self.all_ch_int_stats[color] = {
                frame: self.find_cnt_int_dist(img)
                for frame, img in frames.items()
            }


def sort_cell_info_by_index(time_dict, time_list):
    for tp in time_list:
        pairs = [(int(idx), obj) for idx, obj in time_dict[tp]]
        time_dict[tp] = sorted(pairs, key=lambda x: x[0])
    return time_dict


def populate_cell_ind_overlap(time_dict, time_list):
    # assign indices to first timepoint
    first = time_list[0]
    for i, entry in enumerate(time_dict[first], 1):
        entry[0] = i
    counter = len(time_dict[first]) + 1
    # propagate overlaps
    for i in range(1, len(time_list)):
        curr, prev = time_list[i], time_list[i-1]
        for entry in time_dict[curr]:
            cell = entry[1]; circ = cell.get_circle()
            matched = False
            for p in time_dict[prev]:
                if p[1].evaluate_overlap(circ):
                    entry[0] = p[0]; matched = True; break
            if not matched:
                entry[0] = counter; counter += 1
    return time_dict


def populate_cell_ind_closest(time_dict, time_list, max_dist=100):
    first = time_list[0]
    for i, entry in enumerate(time_dict[first], 1): entry[0] = i
    counter = len(time_dict[first]) + 1
    for i in range(1, len(time_list)):
        curr, prev = time_list[i], time_list[i-1]
        for entry in time_dict[curr]:
            circ = entry[1].get_circle(); best = float('inf')
            for p in time_dict[prev]:
                d = p[1].evaluate_dist(circ)
                if d < best:
                    best, entry[0] = d, p[0]
            if best > max_dist:
                entry[0] = counter; counter += 1
    return time_dict


def make_encoded_mask(sorted_dict, filelist, time_list, out_dir):
    mask_type = np.uint16
    for tp in time_list:
        fp = [f for f in filelist if f"_{tp}_" in f][0]
        img = cv2.imread(fp, 0)
        mask = np.zeros(img.shape, mask_type)
        well = os.path.basename(fp).split('_')[4]
        out_path = os.path.join(out_dir, f"{well}_{tp}_ENCODED.tif")
        for idx, cell in sorted_dict[tp]:
            cv2.drawContours(mask, [cell.cnt], -1, int(idx), -1)
        cv2.imwrite(out_path, mask)


def tracking(var_dict, path_to_masks, out_dir):
    for well in var_dict['Wells']:
        sel = utils.make_selector(well=well, channel=var_dict['MorphologyChannel'])
        files = [f for f in utils.make_filelist_wells(path_to_masks, sel) if 'ENCODED' not in f]
        if not files:
            print(f"No masks for well {well}"); continue
        tps = utils.get_timepoints(files)
        td = collections.OrderedDict()
        for tp in tps:
            img = cv2.imread([f for f in files if tp in f][0], 0)
            cnts = filter_contours(find_cells(img, img_is_mask=True),
                                    small=var_dict['MinCellSize'], large=var_dict['MaxCellSize'])
            td[tp] = [['n', Cell(cnt)] for cnt in cnts]
        if var_dict['TrackType']=='overlap': td = populate_cell_ind_overlap(td, tps)
        else: td = populate_cell_ind_closest(td, tps, max_dist=var_dict['MaxDistance'])
        sorted_td = sort_cell_info_by_index(td, tps)
        make_encoded_mask(sorted_td, files, tps, out_dir)
    print("Tracking complete. Encoded masks in", out_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Track cells from mask images without dictionary files.")
    parser.add_argument('--input_image_path', required=True, help="Folder of mask images (CellMasks)")
    parser.add_argument('--track_type', choices=['overlap','proximity'], default='overlap', help="Tracking method")
    parser.add_argument('--min_cell', type=int, required=True, help="Minimum cell size (pixels)")
    parser.add_argument('--max_cell', type=int, required=True, help="Maximum cell size (pixels)")
    parser.add_argument('--max_dist', type=int, default=100, help="Max movement distance for proximity (pixels)")
    parser.add_argument('--wells', required=True,
                        help="Comma-separated list of wells, e.g. A1,B1,C1")
    parser.add_argument('--morph_channel', required=True,
                        help="Morphology channel name used in file naming")
    args = parser.parse_args()

    var_dict = {
        'MinCellSize': args.min_cell,
        'MaxCellSize': args.max_cell,
        'MaxDistance': args.max_dist,
        'TrackType': args.track_type,
        'MorphologyChannel': args.morph_channel,
        'Wells': args.wells.split(',')
    }
    # Run tracking
    tracking(var_dict, args.input_image_path, args.input_image_path)

    # Example invocation:
    # python tracking_no_dict.py \
    #     --input_image_path ./CellMasks \
    #     --track_type overlap \
    #     --min_cell 100 \
    #     --max_cell 10000 \
    #     --max_dist 50 \
    #     --wells A1,B1,C1 \
    #     --morph_channel GFP
