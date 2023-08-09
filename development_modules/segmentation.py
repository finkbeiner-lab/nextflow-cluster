"""Cell segmentation with various methods"""
import imageio
import argparse
from sql import Database
from skimage import (
    color, feature, filters, measure, morphology, segmentation, util
)
import uuid
import os
import numpy as np
import cv2
import pandas as pd
from db_util import Ops
from normalization import Normalize
import logging
import datetime
from segmentation_helper import save_mask, update_celldata_and_intensitycelldata

logger = logging.getLogger("Segmentation")
# logger.propagate = False
now = datetime.datetime.now()
TIMESTAMP = '%d%02d%02d%02d%02d' % (now.year, now.month, now.day, now.hour, now.minute)
print('Timestamp', TIMESTAMP)
fink_log_dir = '/finkbeiner/imaging/work/metal3/galaxy/finkbeiner_logs'
if not os.path.exists(fink_log_dir):
    os.makedirs(fink_log_dir)
logname = os.path.join(fink_log_dir, f'Segmentation-log_{TIMESTAMP}.log')
fh = logging.FileHandler(logname)
# fh.setLevel(logging.DEBUG)
logger.addHandler(fh)
logger.warn('Running Segmentation from Database.')


class Segmentation:
    def __init__(self, opt):
        self.opt = opt
        self.segmentation_method = opt.segmentation_method
        logger.warn(f'Segmentation Method: {self.segmentation_method}')
        self.mask_folder_name = 'CellMasks'
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
        self.thresh_func = self.threshold_func[self.segmentation_method]
        self.Norm = Normalize(self.opt)
        _, self.analysisdir = self.Norm.get_raw_and_analysis_dir()
        logger.warn(f'Save directory: {self.analysisdir}')

    def run(self):
        tiledata_df = self.Norm.get_flatfields()
        self.run_threshold(tiledata_df)
        logger.warn('Completed threshold')

    def run_threshold(self, df):
        Db = Database()
        # TODO: thread
        img, thresh, regions, masks = None, None, None, None
        for i, row in df.iterrows():
            print('row', row)
            img = imageio.imread(row.filename)  # TODO: is opencv faster/ more memory efficient?
            cleaned_im = self.Norm.image_correction[self.opt.img_norm_name](img, row.tile)
            smoothed_im = self.Norm.gaussian_filter(cleaned_im)
            if self.segmentation_method=='manual':
                thresh = self.opt.manual_thresh
            elif self.segmentation_method=='tryall':
                fig, ax = self.thresh_func(smoothed_im)
                savedir = os.path.join(self.analysisdir, self.mask_folder_name, row.well)
                if not os.path.exists(savedir):
                    os.makedirs(savedir)
                print(f'Saved {self.segmentation_method} segmentation mask to {savedir}')
                fig.savefig(os.path.join(savedir, f'try_all_{row.well}.png'))   # save the figure to file
                return
            else:
                try:
                    thresh = self.thresh_func(smoothed_im)
                except ValueError:
                    thresh = np.ones_like(img) * 65535
            regions = (smoothed_im > thresh) * 255
            masks = measure.label(regions)
            props = measure.regionprops_table(masks, intensity_image=img,
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
            props_df = pd.DataFrame(props)
            props_df, masks = self.filter_by_area(props_df, masks)

            # props['intensity_max'] *= 65535/255
            # props['intensity_mean'] *= 65535/255
            # props['intensity_min'] *= 65535/255
            savedir = os.path.join(self.analysisdir, self.mask_folder_name, row.well)
            maskpath = save_mask(masks, row.filename, savedir)
            print(f'Saved {self.segmentation_method} segmentation mask to {maskpath}')

            # update tiledata with maskpath (not tracked)
            Db.update('tiledata', dict(maskpath=maskpath), kwargs=dict(experimentdata_id=row.experimentdata_id,
                                                                       welldata_id=row.welldata_id,
                                                                       channeldata_id=row.channeldata_id,
                                                                       tile=row.tile,
                                                                       timepoint=row.timepoint))
            update_celldata_and_intensitycelldata(row, props_df, Db)
        if masks is not None:
            print(f'shapes {np.shape(img)}  {np.shape(regions)}')

            last_tile = np.hstack((img, regions))
            imageio.imwrite(self.opt.outfile, last_tile)
        print('Done.')
        
    def filter_by_area(self, props_df:pd.DataFrame, labelled_mask):
        to_delete = []
        # filter small areas
        for area, lbl in zip(props_df.area.tolist(), props_df.label.tolist()):
            if (area < self.opt.area_thresh) or (area > 600 ** 2):
                to_delete.append(lbl)
        print('num masks before filters', len(np.unique(labelled_mask)))
        props_df = props_df[~props_df.label.isin(to_delete)]
        for lbl in to_delete:
            labelled_mask[labelled_mask == lbl] = 0
        print('num masks after filters', len(np.unique(labelled_mask)))
        return props_df, labelled_mask

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input_dict',
        help='path to pickle',
        default=f'/gladstone/finkbeiner/linsley/josh/GALAXY/YD-Transdiff-XDP-Survival1-102822/GXYTMP/tmp.pkl'
    )
    parser.add_argument(
        '--outfile',
        help='Tiff image of last tile',
        default=f'/gladstone/finkbeiner/linsley/josh/GALAXY/YD-Transdiff-XDP-Survival1-102822/GXYTMP/tmp_output.tif'
    )
    parser.add_argument('--experiment', type=str)

    parser.add_argument('--segmentation_method', choices=['minimum', 'yen', 'local', 'li', 'isodata', 'mean',
                                                          'otsu', 'sauvola', 'triangle', 'manual', 'tryall'], type=str,
                        help='Auto segmentation method.')
    parser.add_argument('--img_norm_name', choices=['division', 'subtraction', 'identity'], type=str,
                        help='Image normalization method using flatfield image.')
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
                        help="Specify channels to include or exclude.")
    parser.add_argument('--tile', default=0, type=int, help="Select single tile to segment. Default is to segment all tiles.")
    args = parser.parse_args()
    print(args)
    Seg = Segmentation(args)
    Seg.run()
