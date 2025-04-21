#!/opt/conda/bin/python
"""Montage images or masks"""
import argparse
from normalization import Normalize
import datetime
import logging
import numpy as np
import os
import imageio
from sql import Database
import pdb

logger = logging.getLogger("Montage")
# logger.propagate = False
now = datetime.datetime.now()
TIMESTAMP = '%d%02d%02d%02d%02d' % (now.year, now.month, now.day, now.hour, now.minute)
print('Timestamp', TIMESTAMP)
fink_log_dir = './finkbeiner_logs'
if not os.path.exists(fink_log_dir):
    os.makedirs(fink_log_dir)
logname = os.path.join(fink_log_dir, f'Montage-log_{TIMESTAMP}.log')
fh = logging.FileHandler(logname)
# fh.setLevel(logging.DEBUG)
logger.addHandler(fh)
logger.warning('Running Montage from Database.')


class Montage:
    def __init__(self, opt):
        self.opt = opt
        self.Db = Database()  # 🔹 Initialize database connection
        logger.warning('Montage class initialized.')
        self.Norm = Normalize(self.opt)
        _, self.analysisdir = self.Norm.get_raw_and_analysis_dir()
        self.montage_folder_name = 'MontagedImages'
        self.montagedir = os.path.join(self.analysisdir, self.montage_folder_name)
        if self.opt.tiletype!='filename':
            self.opt.img_norm_name = 'identity'

    def run(self, savebool=True):
        print(f"[DEBUG] Running montage with tiletype: {self.opt.tiletype}")
        print(f"[DEBUG] Chosen wells: {self.opt.chosen_wells}")
        print(f"[DEBUG] Chosen channels: {self.opt.chosen_channels}")

        tiledata_df = self.Norm.get_df_for_training(['channeldata'])
        # tiledata_df = self.Norm.get_flatfields()
        tiledata_df = tiledata_df.sort_values(by=['timepoint', 'well', 'tile',])

        groups = tiledata_df.groupby(by=['timepoint', 'well', 'channel'])
        for name, df in groups:
            self.single_montage(df)
        with open(self.opt.outfile, 'w') as f:
            f.write(f'Montaged Images.')
        print('Done.')

    def single_montage(self, df, savebool=True):
        """
                # Get Robo0/3/4 montage order indexes relative to
        # regular left to right and top to bottom order.
        # like
        # 3 2 1                 1 2 3
        # 4 5 6     relative to 4 5 6
        # 9 8 7                 7 8 9
        # But here we start from 0
        """
        images = []
        savepath = None
        mont = None
        overlap = 0  # TODO: use overlap?
        logger.warning(f'Length of df: {len(df)} and max tile: {df.tile.max()}')
        print(f'Length of df: {len(df)} and max tile: {df.tile.max()}')
        df = df.sort_values('tile')
        if len(df) == df.tile.max() and not df[self.opt.tiletype].isna().any():
            well= df.well.iloc[0]
            timepoint = int(df.timepoint.iloc[0])
            print(f'Well {well}, Timepoint {timepoint}')
            if self.opt.img_norm_name != 'identity' and self.opt.tiletype=='filename':
                self.Norm.get_background_image(df, well, timepoint)#kaushik edit
            
            
            
            for i, row in df.iterrows():
                f = row[self.opt.tiletype]
                print("\n\n\n-------------------", f)
                # overlap = row.overlap
                if not savepath:
                    name = os.path.basename(f)
                    name = name.split('.t')[0] + '_MONTAGE.tif'
                    welldir = os.path.join(self.montagedir, row.well)
                    if not os.path.exists(welldir):
                        os.makedirs(welldir)
                    savepath = os.path.join(welldir, name)
                img = imageio.v3.imread(f)
                
                cleaned_im = self.Norm.image_bg_correction[self.opt.img_norm_name](img, row.well, row.timepoint)
                images.append(cleaned_im)
            if well in self.Norm.backgrounds and timepoint in self.Norm.backgrounds[well]:
                del self.Norm.backgrounds[well][timepoint]
        num_tiles = len(images)
        print(f'Number of tiles processed: {num_tiles}')
        logger.warning(f'Num tiles: {num_tiles}')
        side = int(np.sqrt(num_tiles))
        if num_tiles:
            h, w = np.shape(images[0])
            mont = np.zeros((int(h * side), int(w * side)), dtype=np.uint16)
            for i in range(side):
                for j in range(side):
                    #TODO: map montages for legacy montage, new montages, and ixm montages
                    if self.opt.montage_pattern == 'legacy':
                        if i%2==0:
                            k = side - (j+1)
                        else:
                            k = j
                    else:
                        k = j
                    mont[i * h:(i + 1) * h, j * w:(j + 1) * w] = images[i * side + k]
            if savebool:
                print(f'saved to {savepath}')
                imageio.v3.imwrite(savepath, mont)
            
            # if os.path.exists(savepath):
            if os.path.exists(savepath):
                # Get IDs
                experimentdata_id = self.Db.get_table_uuid(
                    'experimentdata',
                    dict(experiment=self.opt.experiment)
                )
                welldata_id = self.Db.get_table_uuid(
                    'welldata',
                    dict(experimentdata_id=experimentdata_id, well=well)
                )
                channel = df.channel.iloc[0]
                channeldata_id = self.Db.get_table_uuid(
                    'channeldata',
                    dict(
                        experimentdata_id=experimentdata_id,
                        welldata_id=welldata_id,
                        channel=channel
                    )
                )

                # Choose your new tiledata column
                if self.opt.tiletype == 'filename':
                    update_field = 'newimagemontage'
                elif self.opt.tiletype == 'maskpath':
                    update_field = 'newmaskmontage'
                else:
                    update_field = 'newtrackedmontage'

                # Write into tiledata keyed by timepoint
                self.Db.update(
                    'tiledata',
                    update_dct={update_field: savepath},
                    kwargs={
                        'experimentdata_id': experimentdata_id,
                        'welldata_id':       welldata_id,
                        'channeldata_id':    channeldata_id,
                        'timepoint':         int(timepoint)
                    }
                )
                logger.warning(
                    f'Updated {update_field} in tiledata for {well} T{timepoint}'
                )


            #Original
            #     experimentdata_id = self.Db.get_table_uuid('experimentdata', dict(experiment=self.opt.experiment))
            #     welldata_id = self.Db.get_table_uuid('welldata', dict(experimentdata_id=experimentdata_id, well=well))
                
            #     # Choose the appropriate column based on tiletype
            #     if self.opt.tiletype == 'filename':
            #         update_field = 'imagemontage'
            #     elif self.opt.tiletype == 'maskpath':
            #         update_field = 'maskmontage'
            #     else:
            #         update_field = 'imagemontage'  # Default fallback; adjust if needed for trackedmaskpath
                
            #     #     # write into channeldata (one row per well+timepoint+channel)
            #     # channeldata_id = self.Db.get_table_uuid(
            #     #     'channeldata',
            #     #     dict(
            #     #         welldata_id=welldata_id,
            #     #         timepoint=timepoint,
            #     #         channel=df.channel.iloc[0]
            #     #     )
            #     # )
            #     # self.Db.update(
            #     #     'channeldata',
            #     #     update_dct={update_field: savepath},
            #     #     kwargs={'id': channeldata_id}
            #     # )
            #     self.Db.update(
            #         'welldata',
            #         update_dct={update_field: savepath},  # Store montage image path in the correct column
            #         kwargs={'id': welldata_id}  # Ensure correct well ID
            #     )
                
            #     logger.warning(f'Updated {update_field} in welldata for well {well}')
            #     logger.warning(f'  → {update_field}: {savepath}')
            # else:
            #     logger.warning(f'Failed to save montage for well {well}')
            #     logger.warning(f'  → Expected file: {savepath}')

            # Ensure montage files are correctly saved before updating the database Original
            # if os.path.exists(savepath):
            #     # Get database IDs
            #         experimentdata_id = self.Db.get_table_uuid('experimentdata', dict(experiment=self.opt.experiment))
            #         welldata_id = self.Db.get_table_uuid('welldata', dict(experimentdata_id=experimentdata_id, well=well))

            #         # Update `imagemontage` in `welldata`
            #         self.Db.update(
            #             'welldata',
            #             update_dct={'imagemontage': savepath},  # Store montage image path
            #             kwargs={'id': welldata_id}  # Ensure correct well ID
            #         )

            #         logger.warning(f'Updated imagemontage in welldata for well {well}')
            #         logger.warning(f'  → imagemontage: {savepath}')
            # else:
            #         logger.warning(f'Failed to save montage image for well {well}')
            #         logger.warning(f'  → Expected file: {savepath}')
            #         ## original end
        return mont


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
    print("chosenchanneltest")
    parser.add_argument('--experiment', default='JAK-COR7508012023-GEDI', type=str)
    parser.add_argument('--tiletype', default='maskpath', choices=['filename', 'maskpath', 'trackedmaskpath'], type=str,
                        help='Montage image, binary mask, or tracked mask.')
    parser.add_argument('--img_norm_name', default='subtraction', choices=['division', 'subtraction', 'identity'], type=str,
                        help='Image normalization method using flatfield image.')
    parser.add_argument('--montage_pattern',default='standard', choices=['standard', 'legacy'], help="Montage snaking with 3 2 1 4 5 6 9 8 7 pattern.")
    parser.add_argument("--wells_toggle", default='include',
                        help="Chose whether to include or exclude specified wells.")
    parser.add_argument("--timepoints_toggle", default='include',
                        help="Chose whether to include or exclude specified timepoints.")
    parser.add_argument("--channels_toggle", default='include',
                        help="Chose whether to include or exclude specified channels.")
    parser.add_argument("--chosen_wells", "-cw", 
                        dest="chosen_wells", default='C15',
                        help="Specify wells to include or exclude")
    parser.add_argument("--chosen_timepoints", "-ct",
                        dest="chosen_timepoints", default='',
                        help="Specify timepoints to include or exclude.")
    parser.add_argument("--chosen_channels", "-cc",
                        dest="chosen_channels", default='all',
                        help="Specify channels to include or exclude.")
    parser.add_argument("--image_overlap", "-io",
                        dest="image_overlap", default='all',
                        help="Specify amount of overlap")
    parser.add_argument('--tile', default=0, type=int, help="Select single tile to segment. Default is to segment all tiles.")
    args = parser.parse_args()
    print(args)
    Mt = Montage(args)
    Mt.run()
