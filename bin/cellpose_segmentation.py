#!/opt/conda/bin/python

import argparse
import os
from cellpose import models
from skimage import (
    color, feature, filters, measure, morphology, segmentation, util
)
from segmentation_helper import save_mask, update_celldata_and_intensitycelldata
from normalization import Normalize
import pandas as pd
import imageio
from sql import Database
from string import ascii_uppercase
import logging
import datetime

logger = logging.getLogger("CellposeSegmentation")
# logger.propagate = False
now = datetime.datetime.now()
TIMESTAMP = '%d%02d%02d%02d%02d' % (now.year, now.month, now.day, now.hour, now.minute)
print('Timestamp', TIMESTAMP)
fink_log_dir = './finkbeiner_logs'
if not os.path.exists(fink_log_dir):
    os.makedirs(fink_log_dir)
logname = os.path.join(fink_log_dir, f'CellposeSegmentation-log_{TIMESTAMP}.log')
fh = logging.FileHandler(logname)
# fh.setLevel(logging.DEBUG)
logger.addHandler(fh)
logger.warning('Running Segmentation from Database.')


class CellposeSegmentation:
    def __init__(self, opt):
        self.opt = opt
        self.opt.img_norm_name = 'identity'
        self.experiment = opt.experiment
        assert len(self.opt.chosen_channels) > 0, 'Channel must be selected'
        self.Norm = Normalize(self.opt)
        _, self.analysisdir = self.Norm.get_raw_and_analysis_dir()
        logger.warning(f'Save directory: {self.analysisdir}')
        self.mask_folder_name = 'CellMasks'

    def run(self):
        tiledata_df = self.Norm.get_flatfields()
        self.thresh_with_cellpose(tiledata_df)
        logger.warning('Completed threshold')

    def thresh_with_cellpose(self, df):
        """
        https://cellpose.readthedocs.io/en/latest/notebook.html
        """
        # model_type='cyto' or 'nuclei' or 'cyto2'
        Db = Database()
        logger.warning(f'running cellpose {self.opt.model_type}')
        model = models.Cellpose(gpu=True, model_type=self.opt.model_type)

        # define CHANNELS to run segementation on
        # grayscale=0, R=1, G=2, B=3
        # channels = [cytoplasm, nucleus]
        # if NUCLEUS channel does not exist, set the second channel to 0
        chan = [[0, 0]]
        # IF ALL YOUR image_files ARE THE SAME TYPE, you can give a list with 2 elements
        # channels = [0,0] # IF YOU HAVE GRAYSCALE
        # channels = [2,3] # IF YOU HAVE G=cytoplasm and B=nucleus
        # channels = [2,1] # IF YOU HAVE G=cytoplasm and R=nucleus
        logger.warning('Starting eval with cellpose.')
        # if plot_bool:
        #     plt.figure()
        #     plt.imshow(img)
        #     plt.show()
        for i, row in df.iterrows():
            img = imageio.imread(row.filename)
            # cleaned_im = self.Norm.image_correction[self.opt.img_norm_name](img, row.tile)
            # smoothed_im = self.Norm.gaussian_filter(cleaned_im)
            masks, props_df = self.cellpose_single_image(model, chan, img)

            savedir = os.path.join(self.analysisdir, self.mask_folder_name, row.well)
            maskpath = save_mask(masks, row.filename, savedir)
            print(f'Saved Cellpose segmentation mask to {maskpath}')
            logger.warning(f'Saved Cellpose segmentation mask to {maskpath}')
            print('props_df', props_df)

            # Add to database
            Db.update('tiledata', dict(maskpath=maskpath, segmentationmethod='cellpose'), kwargs=dict(experimentdata_id=row.experimentdata_id,
                                                                       welldata_id=row.welldata_id,
                                                                       channeldata_id=row.channeldata_id,
                                                                       tile=row.tile,
                                                                       timepoint=row.timepoint
                                                                       ))
            update_celldata_and_intensitycelldata(row, props_df, Db)
            logger.warning(f'Updated celldata and intensitycelldata for well {row.well} tile {row.tile}')

            # if plot_bool:
            #     fig = plt.figure(figsize=(12, 5))
            #     # plt.imshow(masks)
            #     plot.show_segmentation(fig, img, masks, flows[0], channels=chan)
            #     plt.tight_layout()
            #     plt.show()
        return

    def cellpose_single_image(self, model, chan, img):
        masks, flows, styles, diams = model.eval(img, batch_size=self.opt.batch_size, diameter=self.opt.cell_diameter, channels=chan,
                                                 flow_threshold=self.opt.flow_threshold, cellprob_threshold=self.opt.cell_probability)
        print('Labelled masks with cellpose.')
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
        return masks, props_df

    @staticmethod
    def _get_timepoint(f):
        tp = f.split('/')[-1].split('_')[2]
        return int(tp[1:])

    @staticmethod
    def _get_timestamp(f):
        pid = f.split('/')[-1].split('_')[0]
        return int(pid[3:])

    @staticmethod
    def _generate_wells():
        wells = []
        for i in range(16):
            for j in range(24):
                wells.append(ascii_uppercase[i] + str(j))
        return wells


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input_dict',
        help='path to pickle',
        default=f'/gladstone/finkbeiner/linsley/josh/GALAXY/YD-Transdiff-XDP-Survival1-102822/GXYTMP/tmp.pkl'
    )
    parser.add_argument(
        '--outfile',
        help='path to save pickle file',
        default=f'/gladstone/finkbeiner/linsley/josh/GALAXY/YD-Transdiff-XDP-Survival1-102822/GXYTMP/tmp_output.pkl'
    )
    parser.add_argument('--experiment', default='20230928-MsNeu-RGEDItau1', type=str)
    parser.add_argument('--batch_size',default=1, type=int)
    parser.add_argument('--cell_diameter', default=50, type=int)
    parser.add_argument('--flow_threshold', default=.4, type=float)
    parser.add_argument('--cell_probability',default=0.,  type=float)
    parser.add_argument('--model_type',default='cyto2', type=str)
    parser.add_argument("--wells_toggle", default='include',
                        help="Chose whether to include or exclude specified wells.")
    parser.add_argument("--timepoints_toggle", default='include',
                        help="Chose whether to include or exclude specified timepoints.")
    parser.add_argument("--channels_toggle", default='include',
                        help="Chose whether to include or exclude specified channels.")
    parser.add_argument("--chosen_wells", "-cw",
                        dest="chosen_wells", default='A1',
                        help="Specify wells to include or exclude")
    parser.add_argument("--chosen_timepoints", "-ct",
                        dest="chosen_timepoints", default='T0',
                        help="Specify timepoints to include or exclude.")
    parser.add_argument("--chosen_channels", "-cc", default='Confocal-GFP16',
                        dest="chosen_channels",
                        help="Specify channels to include or exclude.")
    parser.add_argument('--tile', default=0, type=int, help="Select single tile to segment. Default is to segment all tiles.")

    args = parser.parse_args()
    print(args)
    Seg = CellposeSegmentation(args)
    Seg.run()
