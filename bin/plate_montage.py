#!/opt/conda/bin/python
import argparse
import cv2
import os
import numpy as np
from sql import Database
from db_util import Ops
from montage import Montage
import imageio
import logging
import datetime

logger = logging.getLogger("PlateMontage")
# logger.propagate = False
now = datetime.datetime.now()
TIMESTAMP = '%d%02d%02d%02d%02d' % (now.year, now.month, now.day, now.hour, now.minute)
print('Timestamp', TIMESTAMP)
fink_log_dir = './finkbeiner_logs'
if not os.path.exists(fink_log_dir):
    os.makedirs(fink_log_dir)
logname = os.path.join(fink_log_dir, f'PlateMontage-log_{TIMESTAMP}.log')
fh = logging.FileHandler(logname)
# fh.setLevel(logging.DEBUG)
logger.addHandler(fh)
logger.warning('Running Plate Montage from Database.')

class PlateMontage(Montage):
    def __init__(self, opt):
        super().__init__(opt)
        self.opt = opt
        self.opt.img_size = int(self.opt.img_size)
        self.Dbops = Ops(self.opt)
        self.Db = Database()
        self.imagedir, self.analysisdir = self.Dbops.get_raw_and_analysis_dir()
        logger.warning(f'Analysis dir : {self.analysisdir}')

    def run(self):
        tiledata_df = self.Norm.get_flatfields_for_training(['channeldata'])
        tiledata_df = tiledata_df.sort_values(by=['timepoint', 'well', 'tile'])
        channels = tiledata_df.channel.unique()
        timepoints = tiledata_df.timepoint.unique()
        groups = tiledata_df.groupby(by='timepoint')
        logger.warning(f'Tiledata length: {len(tiledata_df)}')
        wells = tiledata_df.well.unique()
        rows = sorted(list(np.unique([w[0] for w in wells])))
        cols = sorted(list(np.unique([int(w[1:]) for w in wells])))
        plate_montage = np.zeros((len(rows) * self.opt.img_size, len(cols) * self.opt.img_size), dtype='uint8')
        for timepoint in timepoints:
            for channel in channels:
                logger.warning(f'Running tp {timepoint}')
                savepath = os.path.join(self.montagedir, f'{self.opt.experiment}_{channel}_plate-montage-T{timepoint}.png')

                tp_df = tiledata_df[(tiledata_df.timepoint == timepoint) & (tiledata_df.channel==channel)]
                groups = tp_df.groupby('well')
                for well, df in groups:
                    logger.warning(f'running {well}')
                    well_montage = self.single_montage(df, savebool=False)

                    row = well[0]
                    col = int(well[1:])
                    ridx = rows.index(row)
                    cidx = cols.index(col)
                    logger.warning(f'Well montage: {np.size(well_montage)}')
                    img = cv2.resize(well_montage, dsize=(self.opt.img_size, self.opt.img_size), interpolation=cv2.INTER_AREA)
                    img = np.clip(img / self.opt.norm_intensity * 255, 0, 255)
                    img = img.astype('uint8')

                    plate_montage[ridx * self.opt.img_size: (ridx + 1) * self.opt.img_size,
                                cidx * self.opt.img_size: (cidx + 1) * self.opt.img_size] = img
                imageio.v3.imwrite(savepath, plate_montage)
        with open(self.opt.outfile, 'w') as f:
            f.write(f'Plate Montaged Images.')
        logger.warning(f"Saved to {self.opt.outfile}")
        print('Done.')


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
    parser.add_argument('--experiment', default='20230928-MsNeu-RGEDItau1', type=str)
    parser.add_argument('--img_size',default=200, type=int, help='Side length of well montage.')
    parser.add_argument('--norm_intensity', default=2000, type=int, help='Value to normalize: (img / norm) * 255.')
    parser.add_argument('--tiletype', default='filename', choices=['filename', 'maskpath', 'trackedmaskpath'], type=str,
                        help='Montage image, binary mask, or tracked mask.')
    parser.add_argument('--img_norm_name', default='identity', choices=['division', 'subtraction', 'identity'], type=str,
                        help='Image normalization method using flatfield image.')
    parser.add_argument('--montage_pattern', default='standard', choices=['standard', 'legacy'], help="Montage snaking with 3 2 1 4 5 6 9 8 7 pattern.")
    parser.add_argument("--wells_toggle", default='include', 
                        help="Chose whether to include or exclude specified wells.")
    parser.add_argument("--timepoints_toggle", default='include', 
                        help="Chose whether to include or exclude specified timepoints.")
    parser.add_argument("--channels_toggle", default='include',
                        help="Chose whether to include or exclude specified channels.")
    parser.add_argument("--chosen_wells", "-cw",
                        dest="chosen_wells", default='all',
                        help="Specify wells to include or exclude")
    parser.add_argument("--chosen_timepoints", "-ct",
                        dest="chosen_timepoints", default='T0',
                        help="Specify timepoints to include or exclude.")
    parser.add_argument("--chosen_channels", "-cc",
                        dest="chosen_channels", default='all',
                        help="Specify channels to include or exclude.")
    parser.add_argument('--tile', default=0, type=int, help="Select single tile to segment. Default is to segment all tiles.")
    args = parser.parse_args()
    print(args)
    Mt = PlateMontage(args)
    Mt.run()
