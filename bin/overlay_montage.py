#!/usr/bin/env python
"""Overlay cell IDs on aligned montage images based on tracked CSV and database experiment info."""

import imageio
import os
from PIL import Image, ImageFont, ImageDraw
import numpy as np
import pandas as pd
import argparse
from sql import Database
import pdb


class OverlayBatch:
    def __init__(self, opt):
        self.opt = opt
        self.Db = Database()
        self.experiment_name = opt.experiment_name

        # Fetch analysisdir from experimentdata table
        result = self.Db.get_table_value(
            tablename='experimentdata',
            column='analysisdir',
            kwargs=dict(experiment=self.experiment_name)
        )

        if not result or not result[0][0]:
            raise ValueError(f"[ERROR] analysisdir not found for experiment '{self.experiment_name}'")

        self.experiment_root = result[0][0].rstrip('/')
        print(f"[INFO] Found experiment root: {self.experiment_root}")
        
        aligned_path = os.path.join(self.experiment_root, "AlignedMontages")
        montaged_path = os.path.join(self.experiment_root, "MontagedImages")

        if os.path.isdir(aligned_path) and os.listdir(aligned_path):
            self.montage_root = aligned_path
            self.selected = True
        else:
            self.montage_root = montaged_path
            self.selected = False

        self.overlay_root = os.path.join(self.experiment_root, "Overlay_Montages")

        self.font = ImageFont.truetype('/usr/share/fonts/dejavu/DejaVuSansMono.ttf', 50)

        summary_path = os.path.join(
            self.experiment_root,
            f"{self.experiment_name}_tracked_montage_summary.csv"
        )
        if not os.path.exists(summary_path):
            raise FileNotFoundError(f"Tracking CSV not found: {summary_path}")
        self.df = pd.read_csv(summary_path)
        self.df['timepoint'] = self.df['timepoint'].astype(int)
        print(f"[INFO] Loaded tracking summary: {summary_path}")

    def run(self):
        
        for well in os.listdir(self.montage_root):
            well_path = os.path.join(self.montage_root, well)
            if not os.path.isdir(well_path):
                continue

            # Filter wells
            if self.opt.wells_toggle == "include" and self.opt.chosen_wells != "all":
                if well not in self.opt.chosen_wells.split(","):
                    continue
            elif self.opt.wells_toggle == "exclude":
                if well in self.opt.chosen_wells.split(","):
                    continue

            for fname in os.listdir(well_path):
                

                if self.selected:
                    if "_MONTAGE_ALIGNED.tif" not in fname:
                        continue
                else:
                    if "_MONTAGE.tif" not in fname:
                        continue
                
                aligned_path = os.path.join(well_path, fname)
                timepoint = self.extract_timepoint(fname)

                # Filter timepoints
                if self.opt.timepoints_toggle == "include" and self.opt.chosen_timepoints != "all":
                    tp_range = self.parse_timepoints(self.opt.chosen_timepoints)
                    if timepoint not in tp_range:
                        continue
                elif self.opt.timepoints_toggle == "exclude":
                    tp_range = self.parse_timepoints(self.opt.chosen_timepoints)
                    if timepoint in tp_range:
                        continue

                self.overlay_image(
                    aligned_path=aligned_path,
                    well=well,
                    timepoint=timepoint
                )

    def extract_timepoint(self, filename):
        parts = filename.split('_')
        for part in parts:
            if part.startswith('T') and part[1:].isdigit():
                return int(part[1:])
        raise ValueError(f"Could not find timepoint in filename: {filename}")

    def parse_timepoints(self, tp_string):
        if tp_string == "all":
            return list(range(100))  # Assume max 100 timepoints
        elif "-" in tp_string:
            start, end = map(int, tp_string.replace("T", "").split("-"))
            return list(range(start, end + 1))
        else:
            return [int(tp_string.replace("T", ""))]

    def overlay_image(self, aligned_path, well, timepoint):
        df_filtered = self.df[
            (self.df['experiment'] == self.experiment_name) &
            (self.df['well'] == well) &
            (self.df['timepoint'] == timepoint)
        ]

        if df_filtered.empty:
            print(f"[WARN] No tracked cells found for: {aligned_path}")
            return

        aligned_img = imageio.imread(aligned_path)
        img = aligned_img - np.min(aligned_img)
        img = np.float32((img / np.max(img) * 128 * self.opt.contrast))
        img[img > 128] = 128
        img = np.uint8(img)

        text_img = np.zeros_like(img)
        text_img = Image.fromarray(text_img)
        draw = ImageDraw.Draw(text_img)

        for _, row in df_filtered.iterrows():
            cellid = int(row['tracked_id'])
            x = int(min(max(row['centroid_x'] + self.opt.shift, 0), img.shape[1] - 20))
            y = int(min(max(row['centroid_y'] + self.opt.shift, 0), img.shape[0] - 20))
            draw.text((x, y), str(cellid), (127), self.font)

        text_img = np.array(text_img)
        overlay_img = np.dstack([img, img + text_img, img])

        overlaydir = os.path.join(self.overlay_root, well)
        os.makedirs(overlaydir, exist_ok=True)

        outname = os.path.basename(aligned_path).replace('.tif', '_OVERLAY.png')
        outpath = os.path.join(overlaydir, outname)
        print(f"[INFO] Saving overlay: {outpath}")
        imageio.v3.imwrite(outpath, overlay_img)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_name', required=True, help="Experiment name")
    parser.add_argument('--target_channel', required=True, help="Channel to use (e.g., Epi-GFP16)")
    parser.add_argument('--chosen_wells', default="all", help="Comma-separated wells (e.g., D3,E4) or 'all'")
    parser.add_argument('--chosen_timepoints', default="all", help="e.g. 'T0', 'T0-T2', or 'all'")
    parser.add_argument('--wells_toggle', default="include", choices=["include", "exclude"])
    parser.add_argument('--timepoints_toggle', default="include", choices=["include", "exclude"])
    parser.add_argument('--channels_toggle', default="include", choices=["include", "exclude"])
    parser.add_argument('--shift', default=20, type=int, help="Pixel shift for label placement")
    parser.add_argument('--contrast', default=1.3, type=float, help="Adjust contrast")
    args = parser.parse_args()

    Ovr = OverlayBatch(args)
    Ovr.run()
