#!/opt/conda/bin/python
"""Cell segmentation with various methods"""
import imageio
import argparse
from sql import Database
from skimage import (
    color, feature, filters, measure, morphology, segmentation, util
)
import os
import numpy as np
import pandas as pd
from db_util import Ops
from normalization import Normalize
import logging
import datetime
from time import time
from segmentation_helper import save_mask, update_celldata_and_intensitycelldata
from threading import Thread  # todo: dispy.SharedJobCluster library? distribute jobs across nodes


logger = logging.getLogger("Segmentation")
# logger.propagate = False
now = datetime.datetime.now()
TIMESTAMP = '%d%02d%02d%02d%02d' % (now.year, now.month, now.day, now.hour, now.minute)
print('Timestamp', TIMESTAMP)
fink_log_dir = './finkbeiner_logs'
if not os.path.exists(fink_log_dir):
    os.makedirs(fink_log_dir)
logname = os.path.join(fink_log_dir, f'Segmentation-log_{TIMESTAMP}.log')
fh = logging.FileHandler(logname)
# fh.setLevel(logging.DEBUG)
logger.addHandler(fh)
logger.warning('Running Segmentation from Database.')


class Segmentation:
    def __init__(self, opt):
        self.opt = opt
        self.thread_lim = 4
        self.segmentation_method = opt.segmentation_method
        assert len(self.opt.chosen_channels) > 0, 'Must select a channel for segmentation'
        logger.warning(f'Segmentation Method: {self.segmentation_method}')
        self.mask_folder_name = 'CellMasks'
        self.threshold_func = dict(sd_from_mean=self.sd_from_mean,
                                   minimum=filters.threshold_minimum,
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
        logger.warning(f'Save directory: {self.analysisdir}')

    def run(self):
        self.run_threshold()
        logger.warning('Completed threshold')

    def sd_from_mean(self, img):
        thresh = int(np.mean(img) + np.std(img) * self.opt.sd_scale_factor)
        return thresh

    def run_threshold(self):
        Db = Database()
        tiledata_df = self.Norm.get_tiledata_df()
        g = tiledata_df.groupby(['well', 'timepoint'])
        
        jobs = []
        for (well, timepoint), df in g:
            # self.thresh_single(Db, df, well, timepoint)

            thread = Thread(target=self.thresh_single, args=(Db, df, well, timepoint))
            jobs.append(thread)
            if len(jobs) > self.thread_lim:
                for j in jobs:
                    j.start()
                for j in jobs:
                    j.join()
                jobs = []
        if len(jobs):
            for j in jobs:
                j.start()
            for j in jobs:
                j.join()
        jobs = []
        print('Done.')

    def thresh_single(self, Db, df, well, timepoint):
        strt = time()
        df.sort_values(by='tile', inplace=True)
        img, thresh, regions, masks = None, None, None, None
        print(f'Thresholding well {well} at timepoint {timepoint}')
        self.Norm.get_background_image(df, well, timepoint)
        for i, row in df.iterrows():
            tile_strt = time()
            print('row', row)
            img = imageio.imread(row.filename)  # TODO: is opencv faster/ more memory efficient?

            smoothed_im = self.Norm.image_bg_correction[self.opt.img_norm_name](img, well, timepoint)
            smoothed_im = self.Norm.gaussian_filter(smoothed_im)
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
            regions = np.uint8((smoothed_im > thresh) * 255)
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
                                                                        timepoint=row.timepoint,
                                                                        ))  # TODO: add segmentationmethod=f'{self.segmentation_method}' with analysis version
            update_celldata_and_intensitycelldata(row, props_df, Db)
            print(f'Finished tile {row.tile} for well + timepoint in {time() - tile_strt:.2f}')

        if masks is not None:
            print(f'shapes {np.shape(img)}  {np.shape(regions)}')

            last_tile = np.hstack((img, regions))
            imageio.imwrite(self.opt.outfile, last_tile)
        del self.Norm.backgrounds[well][timepoint]
        print(f'Finished well + timepoint in {time() - strt:.2f}')
        
    def filter_by_area(self, props_df:pd.DataFrame, labelled_mask):
        props_df = props_df[(self.opt.upper_area_thresh > props_df.area) & (props_df.area > self.opt.lower_area_thresh) ]
        filtered_labels = props_df.label.tolist()
        all_labels = np.unique(labelled_mask)
        print('num masks before area filters', len(all_labels))
        # filtered_mask = np.zeros_like(labelled_mask)
        # rows, cols = np.shape(labelled_mask)
        # for i in range(rows):
        #     for j in range(cols):
        #         filtered_mask[i,j] = labelled_mask[i,j] if labelled_mask[i,j] in filtered_labels else 0

        to_delete = list(set(all_labels) - set(props_df.label.tolist()))
        props_df = props_df[~props_df.label.isin(to_delete)]
        for lbl in to_delete:
            labelled_mask[labelled_mask == lbl] = 0
        print('num masks after area filters', len(np.unique(labelled_mask)))
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

    parser.add_argument('--experiment',default='0907-FB-1-JL-gedi-test', type=str)

    parser.add_argument('--segmentation_method', default='sd_from_mean', choices=['sd_from_mean', 'minimum', 'yen', 'local', 'li', 'isodata', 'mean',
                                                          'otsu', 'sauvola', 'triangle', 'manual', 'tryall'], type=str,
                        help='Auto segmentation method.')
    parser.add_argument('--img_norm_name', default='subtraction', choices=['division', 'subtraction', 'identity'], type=str,
                        help='Image normalization method using flatfield image.')
    parser.add_argument('--lower_area_thresh', default=50, type=int, help="Lowerbound for cell area. Remove cells with area less than this value.")
    parser.add_argument('--upper_area_thresh', default=36000, type=int, help="Upperbound for cell area. Remove cells with area greater than this value.")
    parser.add_argument('--sd_scale_factor', default=3.5, type=float, help="Standard Deviation (SD) scale factor if using sd_from_mean threshold.")
    parser.add_argument('--manual_thresh', default=0, type=int, help="Threshold if using manual threshold method.")
    parser.add_argument("--wells_toggle", default='include',
                        help="Chose whether to include or exclude specified wells.")
    parser.add_argument("--timepoints_toggle", default='include',
                        help="Chose whether to include or exclude specified timepoints.")
    parser.add_argument("--channels_toggle", default='include',
                        help="Chose whether to include or exclude specified channels.")
    parser.add_argument("--chosen_wells", "-cw",
                        dest="chosen_wells", default='E4',
                        help="Specify wells to include or exclude")
    parser.add_argument("--chosen_timepoints", "-ct",
                        dest="chosen_timepoints", default='all',
                        help="Specify timepoints to include or exclude.")
    parser.add_argument("--chosen_channels", "-cc",
                        dest="chosen_channels", default='GFP-DMD1',
                        help="Morphology channel.")
    parser.add_argument('--tile', default=0, type=int, help="Select single tile to segment. Default is to segment all tiles.")
    args = parser.parse_args()
    print(args)
    Seg = Segmentation(args)
    Seg.run()
