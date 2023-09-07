#!/opt/conda/bin/python
"""Montage images or masks"""
from db_util import Ops
from normalization import Normalize
from sql import Database
import argparse
import imageio
import numpy as np
import os
import logging
import datetime

logger = logging.getLogger("Intensity")
# logger.propagate = False
now = datetime.datetime.now()
TIMESTAMP = '%d%02d%02d%02d%02d' % (now.year, now.month, now.day, now.hour, now.minute)
print('Timestamp', TIMESTAMP)
fink_log_dir = './finkbeiner_logs'
if not os.path.exists(fink_log_dir):
    os.makedirs(fink_log_dir)
logname = os.path.join(fink_log_dir, f'Intensity-log_{TIMESTAMP}.log')
fh = logging.FileHandler(logname)
# fh.setLevel(logging.DEBUG)
logger.addHandler(fh)
logger.warn('Running Montage from Database.')

class Intensity:
    def __init__(self, opt):
        self.opt = opt
        self.Norm = Normalize(self.opt)
        self.Op = Ops(self.opt)

    def run(self):
        self.Norm.get_flatfields()
        Db = Database()
        celldata_df = self.Op.get_celldata_df()
        print(f'Number of cells to start: {len(celldata_df)}')
        logger.warn(f'Number of cells to start: {len(celldata_df)}')
        celldata_df = celldata_df.sort_values(by=['tile', 'timepoint'])
        groups = celldata_df.groupby(by=['welldata_id', 'tile', 'timepoint'])
        for (welldata_id, tile, timepoint), df in groups:
            target_channel_uuid = Db.get_table_uuid('channeldata', dict(channel=self.opt.target_channel, welldata_id=welldata_id))
            print(f'tile: {tile} timepoint {timepoint}')
            logger.warn(f'tile: {tile} timepoint {timepoint}')
            # trackedmaskpath = df.trackedmaskpath.iloc[0]
            maskpath = df.maskpath.iloc[0]

            print(f'maskpath {maskpath}')
            labelled_mask = imageio.v3.imread(maskpath)
            # labelled_mask = imageio.v3.imread(trackedmaskpath)

            filename = Db.get_table_value('tiledata', column='filename', kwargs=dict(welldata_id=welldata_id,
                                                                          channeldata_id=target_channel_uuid,
                                                                          tile=int(tile),
                                                                          timepoint=int(timepoint)
                                                                          ))
            if filename is None: raise Exception(f'Filename for channel {self.opt.target_channel} is not found.')
            print(f'filename {filename}')
            img = imageio.v3.imread(filename[0][0])
            img = self.Norm.image_correction[self.opt.img_norm_name](img, tile)

            intensitycelldata_dcts = []
            check_celldata_dct = dict(tiledata_id=df.tiledata_id.iloc[0],  # morphology tiledata_id
                                        channeldata_id=target_channel_uuid)
            print(f'Number of cells: {len(df)}')
            logger.warn(f'Number of cells: {len(df)}')
            print('cellids')
            z = set(df.randomcellid).symmetric_difference(np.unique(labelled_mask))
            print('extra cellids', z)
            for i, row in df.iterrows():
                # offset channel?
                # print('row', row)
                # print('randomcellid', row.randomcellid)
                if row.randomcellid not in labelled_mask:
                    raise Exception(f'Cellid not in mask {row.cellid} in {maskpath}')
                logger.warn(f'randomcellid {row.randomcellid}')
                cell = img[labelled_mask == row.randomcellid]
                intensity_max = float(np.max(cell))
                intensity_mean = float(np.sum(cell) / np.count_nonzero(cell))
                nonzero_arr = cell[cell!=0]
                intensity_min = float(np.min(nonzero_arr)) if len(nonzero_arr) else 0
                # intensity_std = float(np.std(nonzero_arr)) if len(nonzero_arr) else 0
                intensitycelldata_dcts.append(dict(experimentdata_id=row.experimentdata_id,
                                                   welldata_id=row.welldata_id,
                                                   tiledata_id=row.tiledata_id,
                                                   celldata_id=row.id,
                                                   channeldata_id=target_channel_uuid,
                                                   intensity_max=intensity_max,
                                                   intensity_mean=intensity_mean,
                                                   intensity_min=intensity_min))
            # Clears intensity for cell in the tile with the selected channel
            Db.delete_based_on_duplicate_name(tablename='intensitycelldata', kwargs=check_celldata_dct)
            Db.add_row(tablename='intensitycelldata', dct=intensitycelldata_dcts)

        with open(self.opt.outfile, 'w') as f:
            f.write(f'Updated intensity {self.opt.chosen_channels} in intensitycelldata database.')
        print('Done.')        # read data

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
    parser.add_argument('--experiment', default = '20230828-2-msneuron-cry2', type=str)
    parser.add_argument('--img_norm_name', default='subtraction', choices=['division', 'subtraction', 'identity'], type=str,
                        help='Image normalization method using flatfield image.')
    parser.add_argument("--wells_toggle", default='include', 
                        help="Chose whether to include or exclude specified wells.")
    parser.add_argument("--timepoints_toggle", default='include',
                        help="Chose whether to include or exclude specified timepoints.")
    parser.add_argument("--channels_toggle", default='include',
                        help="Chose whether to include or exclude specified channels.")
    parser.add_argument("--chosen_wells", "-cw",
                        dest="chosen_wells", default='E7',
                        help="Specify wells to include or exclude")
    parser.add_argument("--chosen_timepoints", "-ct", default='T0',
                        dest="chosen_timepoints", 
                        help="Specify timepoints to include or exclude.")
    parser.add_argument("--chosen_channels", "-cc", default='RFP1',
                        dest="chosen_channels",
                        help="Morphology Channel")
    parser.add_argument("--target_channel", default='YFP1',
                        dest="target_channel",
                        help="Get intensity of this channel.")
    parser.add_argument('--tile', default=0, type=int, help="Select single tile to segment. Default is to segment all tiles.")
    args = parser.parse_args()
    print(args)
    I = Intensity(args)
    I.run()
