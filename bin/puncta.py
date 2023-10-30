import numpy as np
import imageio
import matplotlib.pyplot as plt
from skimage import (
    color, feature, filters, measure, morphology, segmentation, util
)
from sql import Database
from db_util import Ops
import pandas as pd
import time
import logging
import os
import datetime
import argparse
import cv2
from puncta_helper import update_punctadata_and_intensitypunctadata

logger = logging.getLogger("Puncta")
# logger.propagate = False
now = datetime.datetime.now()
TIMESTAMP = '%d%02d%02d%02d%02d' % (now.year, now.month, now.day, now.hour, now.minute)
print('Timestamp', TIMESTAMP)
fink_log_dir = './finkbeiner_logs'
if not os.path.exists(fink_log_dir):
    os.makedirs(fink_log_dir)
logname = os.path.join(fink_log_dir, f'Puncta-log_{TIMESTAMP}.log')
fh = logging.FileHandler(logname)
# fh.setLevel(logging.DEBUG)
logger.addHandler(fh)
logger.warning('Registering experiment with database.')


class Puncta:
    def __init__(self, opt):
        self.opt = opt
        self.threshold_func = dict(minimum=filters.threshold_minimum,
                                   yen=filters.threshold_yen,
                                   local=filters.threshold_local,
                                   li=filters.threshold_li,
                                   isodata=filters.threshold_isodata,
                                   mean=filters.threshold_mean,
                                   otsu=filters.threshold_otsu,
                                   sauvola=filters.threshold_sauvola,
                                   triangle=filters.threshold_triangle,
                                   manual=None,
                                   tryall=filters.try_all_threshold)
        self.thresh_func = self.threshold_func[self.opt.segmentation_method]
        self.Op = Ops(self.opt)
        _, self.analysisdir = self.Op.get_raw_and_analysis_dir()
        self.puncta_folder_name = 'Puncta'

    def run(self):
        # get mask and image per tile
        Db = Database()
        celldata_df = self.Op.get_celldata_df()
        print(f'Number of cells to start: {len(celldata_df)}')
        groups = celldata_df.groupby(by=['welldata_id', 'tile', 'timepoint'])
        print(f'Number of tiles to start: {len(groups)}')
        for (welldata_id, tile, timepoint), df in groups:
            logger.warning(f'Getting puncta for tile: {tile} at T{timepoint} with uuid: {welldata_id}')
            target_channel_uuid = Db.get_table_uuid('channeldata', dict(channel=self.opt.target_channel,
                                                                        welldata_id=welldata_id))

            trackedmaskpath = df.trackedmaskpath.iloc[0]
            if trackedmaskpath is None:
                trackedmaskpath = df.maskpath.iloc[0]
            # filename of image with target channel
            filename = Db.get_table_value('tiledata', column='filename', kwargs=dict(welldata_id=welldata_id,
                                                                                     channeldata_id=target_channel_uuid,
                                                                                     tile=int(tile),
                                                                                     timepoint=int(timepoint)
                                                                                     ))
            if filename is None:
                raise Exception(f'Filename for channel {self.opt.target_channel} is not found.')
            if trackedmaskpath is None:
                raise Exception(f'Path to mask is not found.')

            # TODO: add normalization
            self.thresh_puncta_per_mask(df, Db, filename[0][0], trackedmaskpath,
                                        sigma1=self.opt.sigma1,
                                        sigma2=self.opt.sigma2)

        # loop through uuids, not cellids or randomcellids because tracking may not have been run
        # use tracked mask if available, otherwise mask.
        # cell must exist in mask

    def thresh_puncta_per_mask(self, df, Db, filename, trackedmaskpath,
                               sigma1=2, sigma2=4, save_puncta_image_bool=True):
        puncta_in_tile = 0
        labelled_mask = imageio.v3.imread(trackedmaskpath)
        target = imageio.v3.imread(filename)
        logger.warning(f'Running difference of gaussians for puncta with sigma1: {sigma1} and sigma2: {sigma2}')
        # Threshold
        smoothed_im = self.difference_of_gaussian(target, sigma2=sigma2, sigma1=sigma1)
        if self.opt.segmentation_method == 'manual':
            thresh = self.opt.manual_thresh
        elif self.opt.segmentation_method == 'tryall':
            fig, ax = self.thresh_func(smoothed_im, figsize=(12,12))
            savedir = os.path.join(self.analysisdir, self.puncta_folder_name, df.well.iloc[0])
            print(f'Saved {self.opt.segmentation_method} segmentation mask to {savedir}')
            fig.tight_layout()
            fig.savefig(os.path.join(savedir, f'try_all_{df.well.iloc[0]}.png'))   # save the figure to file
            return
        else:
            try:
                thresh = self.thresh_func(smoothed_im)
            except:
                thresh = np.zeros_like(smoothed_im)
        regions = (smoothed_im > thresh) * 255
        if save_puncta_image_bool:
            savedir = os.path.join(self.analysisdir, self.puncta_folder_name, df.well.iloc[0])
            puncta_mask_path = self.save_puncta_mask(regions, filename, savedir)
            logger.warning(f'Saved puncta mask to {puncta_mask_path}')

        # TODO: is it faster to get contours of cells and see if puncta region props are inside contours?
        # cv2.findContours
        # cv2.pointPolygonTest
        # need to keep track which contour belongs to which cell

        # Loop through cellids / randomcellids
        for i, row in df.iterrows():
            cellid = row.cellid
            if cellid is None:
                cellid = row.randomcellid
            cellmask = regions * (labelled_mask == cellid)
            if np.any(cellmask > 0):
                labels = measure.label(cellmask)
                puncta_props = measure.regionprops_table(labels, intensity_image=target,
                                                         properties=('label', 'area', 'centroid_weighted',
                                                                     'orientation',
                                                                     'solidity',
                                                                     'extent',
                                                                     'perimeter',
                                                                     'eccentricity',
                                                                     'intensity_max',
                                                                     'intensity_mean',
                                                                     'intensity_min',
                                                                     'axis_major_length',
                                                                     'axis_minor_length',
                                                                     )
                                                         )
                props_df = pd.DataFrame(puncta_props)
                puncta_in_tile += len(props_df)
                update_punctadata_and_intensitypunctadata(row, props_df, Db)

        # to target puncta, make a circle with puncta area at its location
        logger.warning(f'Found {puncta_in_tile} puncta in in cells in tile.')

    def difference_of_gaussian(self, img, sigma1=2, sigma2=4):
        assert sigma2 > sigma1
        smooth2 = filters.gaussian(img, sigma=sigma2)
        smooth1 = filters.gaussian(img, sigma=sigma1)
        smooth = np.abs(smooth2 - smooth1)
        return smooth

    def save_puncta_mask(self, mask, image_file, savedir):
        name = os.path.basename(image_file)
        name = name.split('.t')[0]  # split by tiff suffix
        if not os.path.exists(savedir):
            os.makedirs(savedir)
        savepath = os.path.join(savedir, name + '_PUNCTA.tif')
        cv2.imwrite(savepath, mask)
        return savepath


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input_dict',
        help='path to pickle, used to link modules in galaxy',
        default=f'/gladstone/finkbeiner/linsley/josh/GALAXY/YD-Transdiff-XDP-Survival1-102822/GXYTMP/tmp.pkl'
    )
    parser.add_argument(
        '--outfile',
        help='Tiff image of last tile',
        default=f'/gladstone/finkbeiner/linsley/josh/GALAXY/YD-Transdiff-XDP-Survival1-102822/GXYTMP/tmp_output.tif'
    )
    parser.add_argument('--experiment', type=str)
    parser.add_argument('--segmentation_method', choices=['minimum', 'yen', 'local', 'li', 'isodata', 'mean',
                                                          'otsu', 'sauvola', 'triangle', 'manual','tryall'], type=str,
                        help='Auto segmentation method.')
    parser.add_argument('--area_thresh', default=1000, type=int, help="Lowerbound for cell area. Remove cells with area less than this value.")
    parser.add_argument('--manual_thresh', default=0, type=int, help="Threshold if using manual threshold method.")
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
    parser.add_argument("--sigma1", dest="sigma1", type=float, help='Lesser gaussian blur for difference of gaussians')
    parser.add_argument("--sigma2", dest="sigma2", type=float, help='Greater gaussian blur for difference of gaussians')
    parser.add_argument('--tile', default=0, type=int, help="Select single tile to segment. Default is to segment all tiles.")
    args = parser.parse_args()
    print(args)
    Pun = Puncta(args)
    Pun.run()
