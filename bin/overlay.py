"""Overlay mask and numbers on image in 8-bit"""

import imageio
import os
from PIL import Image, ImageFont, ImageDraw
import numpy as np
import pandas as pd
import os
import argparse
from db_util import Ops
from sql import Database
from threading import Thread


class Overlay:
    def __init__(self, opt):
        self.opt = opt
        self.Dbops = Ops(opt)
        self.Db = Database()
        self.imagedir, self.analysisdir = self.Dbops.get_raw_and_analysis_dir()
        self.overlaydir = os.path.join(self.analysisdir, 'Overlays')
        self.font = ImageFont.truetype('/usr/share/fonts/dejavu/DejaVuSansMono.ttf', 20)
        # try:
            # self.font = ImageFont.truetype('/usr/share/fonts/dejavu/DejaVuSansMono.ttf', 3)
        # except OSError:
            # self.font = ImageFont.load_default()
        self.thread_lim = 4

    def run(self):
        celldata = self.Dbops.get_celldata_df()  # get crops from morphology channel
        groups = celldata.groupby('tiledata_id')
        jobs = []
        for tiledata_id, df in groups:
            thread = Thread(target=self.thread_overlay, args=(df,))
            jobs.append(thread)
            if len(jobs) >= self.thread_lim:
                for j in jobs:
                    j.start()
                for j in jobs:
                    j.join()
                jobs = []
            if len(jobs) > 0:
                for j in jobs:
                    j.start()
                for j in jobs:
                    j.join()
            jobs = []
        print('Done')


    def thread_overlay(self, df):
        if len(df):
            well = df.well.iloc[0]
                # get channeldata id
            target_channeldata_id = self.Db.get_table_uuid('channeldata', dict(welldata_id=df.welldata_id.iloc[0],
                                                                            channel=self.opt.target_channel))

            filename = self.Db.get_table_value('tiledata', 'filename',
                                                   dict(welldata_id=df.welldata_id.iloc[0],
                                                        channeldata_id=target_channeldata_id,
                                                        timepoint = int(df.timepoint.iloc[0]),
                                                        tile=int(df.tile.iloc[0])))
            filename = filename[0][0]
            print(f'Running {well} for tile {df.tile.iloc[0]} for channel {self.opt.target_channel}')
            print(f'Running {filename}')
            img = imageio.imread(filename)

            sh = img.shape
            maskfile = df.trackedmaskpath.iloc[0]
            if maskfile is None:
                raise Exception('Tracked mask path is none, tracking not found.')
            mask = imageio.imread(maskfile)
            img = img -np.min(img)
            img = np.float32((img / np.max(img) * 128 * self.opt.contrast))
            img[img > 128] = 128
            img = np.uint8(img)
            mask = np.uint8((mask > 0) * 50)
            text_img = np.zeros_like(img)
            text_img = Image.fromarray(text_img)
            draw = ImageDraw.Draw(text_img)
            for i, row in df.iterrows():
                if row.cellid is not None and row.cellid > 0:
                    cellid = int(row.cellid)
                else:
                    raise Exception('Cellid not found, run tracking.')
                    # centroids from morphology channel
                x, y = self.get_coord(sh, row)
                draw.text((x,y), str(cellid), (127), self.font)

            text_img = np.array(text_img)
            overlay_img = np.dstack([img+mask , img + text_img, img])
            name = os.path.basename(filename)
            name = name.split('.t')[0]
            name += f'_OVERLAY.png'
            welldir = os.path.join(self.overlaydir, well)
            if not os.path.exists(welldir):
                os.makedirs(welldir)
            overlayname = os.path.join(welldir, name)
            print('overlayname', overlayname)
            imageio.v3.imwrite(overlayname, overlay_img)
        return

    def get_coord(self, sh, row):
        x = int(row.centroid_x + self.opt.shift)
        y = int(row.centroid_y + self.opt.shift)
        x = x if x >= 0 else 0
        y = y if y >= 0 else 0
        x = x if x < sh[1] else sh[1]-20
        y = y if y < sh[0] else sh[0]-20
        return int(x), int(y)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input_dict',
        help='path to pickle',
        default=f'/gladstone/finkbeiner/linsley/josh/GALAXY/YD-Transdiff-XDP-Survival1-102822/GXYTMP/tmp.pkl'
    )
    parser.add_argument(
        '--outfile',
        help='Text status',
        default=f'/gladstone/finkbeiner/linsley/josh/GALAXY/YD-Transdiff-XDP-Survival1-102822/GXYTMP/tmp_output.txt'
    )
    parser.add_argument('--experiment', type=str)
    parser.add_argument('--img_norm_name', choices=['division', 'subtraction', 'identity'], type=str,
                        help='Image normalization method using flatfield image.')
    parser.add_argument('--shift', default=20, type=int, help="Side length of square")
    parser.add_argument('--contrast', default=1.3, type=float, help="Side length of square")

    parser.add_argument("--wells_toggle",
                        help="Chose whether to include or exclude specified wells.")
    parser.add_argument("--timepoints_toggle",
                        help="Chose whether to include or exclude specified timepoints.")
    parser.add_argument("--channels_toggle", default='include',
                        help="Chose whether to include or exclude specified channels.")
    parser.add_argument("--chosen_wells", "-cw",
                        dest="chosen_wells", default='',
                        help="Specify wells to include or exclude")
    parser.add_argument("--chosen_timepoints", "-ct",
                        dest="chosen_timepoints", default='',
                        help="Specify timepoints to include or exclude.")
    parser.add_argument("--chosen_channels", "-cc",
                        dest="chosen_channels",
                        help="Morphology Channel")
    parser.add_argument("--target_channel",
                        dest="target_channel",
                        help="Get intensity of this channel.")
    parser.add_argument('--tile', default=0, type=int, help="Select single tile to segment. Default is to segment all tiles.")
    args = parser.parse_args()
    print(args)
    Ovr = Overlay(args)
    Ovr.run()
