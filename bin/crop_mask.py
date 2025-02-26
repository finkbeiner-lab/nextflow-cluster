#!/opt/conda/bin/python
"""Get mask crops"""

import imageio
import os
import pandas as pd
import os
import argparse
from db_util import Ops
from sql import Database
import uuid
from threading import Thread


class Crop:
    def __init__(self, opt):
        self.opt = opt
        self.Dbops = Ops(opt)
        self.Db = Database()
        self.imagedir, self.analysisdir = self.Dbops.get_raw_and_analysis_dir()
        # self.cropdir = os.path.join(self.analysisdir, 'CroppedImages') #original
        # #KS edit check
        # if self.opt.mask_crop:
        #     self.cropdir = os.path.join(self.analysisdir, 'MaskCroppedImages')  
        # else:
        #     self.cropdir = os.path.join(self.analysisdir, 'CroppedImages')

        self.cropdir = os.path.join(self.analysisdir, 'MaskCroppedImages') #KSedit
        self.experimentdata_id = self.Db.get_table_uuid('experimentdata',
                                                        dict(experiment=self.opt.experiment))
        self.thread_lim = 2

    def run(self):
        celldata = self.Dbops.get_celldata_df()  # get crops from morphology channel
        groups = celldata.groupby('tiledata_id')
        jobs = []
        for tiledata_id, df in groups:
            thread = Thread(target=self.thread_crop, args=(df,))
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
        for tiledata_id, df in groups:
            # open image
            self.thread_crop(df)
        print('Done')

    def thread_crop(self, df):
        if len(df):
            well = df.well.iloc[0]
                # get target channeldata id
            target_channeldata_id = self.Db.get_table_uuid('channeldata', dict(welldata_id=df.welldata_id.iloc[0],
                                                                                   channel=self.opt.target_channel))



            # if self.opt.mask_crop:
            #     filename = self.Db.get_table_value('tiledata', 'maskpath',
            #                            dict(welldata_id=df.welldata_id.iloc[0],
            #                                 channeldata_id=target_channeldata_id,
            #                                 timepoint=int(df.timepoint.iloc[0]),
            #                                 tile=int(df.tile.iloc[0])))  
            # else:
            #     filename = self.Db.get_table_value('tiledata', 'filename',
            #                            dict(welldata_id=df.welldata_id.iloc[0],
            #                                 channeldata_id=target_channeldata_id,
            #                                 timepoint=int(df.timepoint.iloc[0]),
            #                                 tile=int(df.tile.iloc[0])))

            # filename = self.Db.get_table_value('tiledata', 'filename',
            #                                        dict(welldata_id=df.welldata_id.iloc[0],
            #                                             channeldata_id=target_channeldata_id,
            #                                             timepoint=int(df.timepoint.iloc[0]),
            #                                             tile=int(df.tile.iloc[0])))  ##original
            filename = self.Db.get_table_value('tiledata', 'trackedmaskpath',
                                                   dict(welldata_id=df.welldata_id.iloc[0],
                                                        channeldata_id=target_channeldata_id,
                                                        timepoint=int(df.timepoint.iloc[0]),
                                                        tile=int(df.tile.iloc[0])))  ##KSedit
            if filename is None:
                return
            filename = filename[0][0]
            print(f'Running {well} for tile {df.tile.iloc[0]} for channel {self.opt.target_channel}')
            print(f'Running {filename}')
            img = imageio.v3.imread(filename)
            sh = img.shape
            cropdcts = []
            check_cropdcts = []
            for i, row in df.iterrows():
                if row.cellid is not None and row.cellid > 0:
                    cellid = row.cellid
                else:
                    cellid = row.randomcellid
                    # centroids from morphology channel
                xi, xf, yi, yf = self.get_coords(sh, row)
                crop = img[yi:yf, xi:xf]
                name = os.path.basename(filename)
                name = name.split('.t')[0]
                name += f'_CROP-{cellid}.png'
                welldir = os.path.join(self.cropdir, well)
                if not os.path.exists(welldir):
                    os.makedirs(welldir)
                croppath = os.path.join(welldir, name)
                imageio.v3.imwrite(croppath, crop)
                check_cropdct = dict(experimentdata_id=self.experimentdata_id,
                                         welldata_id=df.welldata_id.iloc[0],
                                         channeldata_id=target_channeldata_id,
                                         celldata_id=row.id)
                cropdcts.append(dict(id=uuid.uuid4(),
                                         experimentdata_id=self.experimentdata_id,
                                         welldata_id=df.welldata_id.iloc[0],
                                         channeldata_id=target_channeldata_id,
                                         celldata_id=row.id,
                                         croppath=croppath))
                self.Db.delete_based_on_duplicate_name(tablename='cropdata', kwargs=check_cropdct)

            self.Db.add_row('cropdata', cropdcts)

    def get_coords(self, sh, row):
        xi = int(row.centroid_x - self.opt.crop_size // 2)
        xf = xi + self.opt.crop_size
        yi = int(row.centroid_y - self.opt.crop_size // 2)
        yf = yi + self.opt.crop_size
        xi = xi if xi >= 0 else 0
        yi = yi if yi >= 0 else 0
        if xi == 0:
            xf = self.opt.crop_size
        if yi == 0:
            yf = self.opt.crop_size
        xf = xf if xf < sh[1] else sh[1] - 1
        yf = yf if yf < sh[0] else sh[0] - 1
        if xf == sh[1] - 1 :  
            xi = sh[1] - (self.opt.crop_size + 1)
        if yf == sh[0] - 1 :
            yi = sh[0] - (self.opt.crop_size + 1)
        return xi, xf, yi, yf


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
    parser.add_argument('--experiment', default='20230807-KS1-neuron-optocrispr', type=str)
    parser.add_argument('--crop_size', default=300, type=int, help="Side length of square")
    parser.add_argument("--wells_toggle", default='include',
                        help="Chose whether to include or exclude specified wells.")
    parser.add_argument("--timepoints_toggle", default='include',
                        help="Chose whether to include or exclude specified timepoints.")
    parser.add_argument("--channels_toggle", default='include',
                        help="Chose whether to include or exclude specified channels.")
    parser.add_argument("--chosen_wells", "-cw",
                        dest="chosen_wells", default='F2',
                        help="Specify wells to include or exclude")
    parser.add_argument("--chosen_timepoints", "-ct",
                        dest="chosen_timepoints", default='T0',
                        help="Specify timepoints to include or exclude.")
    parser.add_argument("--chosen_channels", "-cc",
                        dest="chosen_channels", default='GFP-DMD1',
                        help="Morphology Channel")
    parser.add_argument("--target_channel",
                        dest="target_channel", default='GFP-DMD1',
                        help="Get intensity of this channel.")
    parser.add_argument('--tile', default=0, type=int, help="Select single tile to segment. Default is to segment all tiles.")
    args = parser.parse_args()
    print(args)
    Crp = Crop(args)
    Crp.run()

## KS edits- To Do use batch for it
# """Get cell crops"""

# import imageio
# import os
# import pandas as pd
# import argparse
# from db_util import Ops
# from sql import Database
# import uuid
# import gc

# class Crop:
#     def __init__(self, opt):
#         self.opt = opt
#         self.Dbops = Ops(opt)
#         self.Db = Database()
#         self.imagedir, self.analysisdir = self.Dbops.get_raw_and_analysis_dir()
#         self.cropdir = os.path.join(self.analysisdir, 'CroppedImages')
#         self.experimentdata_id = self.Db.get_table_uuid('experimentdata',
#                                                         dict(experiment=self.opt.experiment))
#         self.batch_size = 20  # Process 20 crops at a time

#     def run(self):
#         celldata = self.Dbops.get_celldata_df()  # get crops from morphology channel
#         groups = celldata.groupby('tiledata_id')

#         for tiledata_id, df in groups:
#             print(f"Processing tile {tiledata_id}...")

#             # Process in batches
#             df_batches = [df.iloc[i:i + self.batch_size] for i in range(0, len(df), self.batch_size)]
#             for batch in df_batches:
#                 self.process_batch(batch)
                
#             gc.collect()  # Trigger garbage collection after processing a tile
#         print('Done')

#     def process_batch(self, df):
#         if len(df):
#             well = df.well.iloc[0]

#             # Get target channeldata id
#             target_channeldata_id = self.Db.get_table_uuid('channeldata', dict(welldata_id=df.welldata_id.iloc[0],
#                                                                                channel=self.opt.target_channel))

#             filename = self.Db.get_table_value('tiledata', 'filename',
#                                                dict(welldata_id=df.welldata_id.iloc[0],
#                                                     channeldata_id=target_channeldata_id,
#                                                     timepoint=int(df.timepoint.iloc[0]),
#                                                     tile=int(df.tile.iloc[0])))
#             if filename is None:
#                 return
#             filename = filename[0][0]

#             print(f'Processing file: {filename}')

#             # Load the image
#             img = imageio.v3.imread(filename)
#             sh = img.shape
#             cropdcts = []

#             for i, row in df.iterrows():
#                 if row.cellid is not None and row.cellid > 0:
#                     cellid = row.cellid
#                 else:
#                     cellid = row.randomcellid

#                 # Get crop coordinates
#                 xi, xf, yi, yf = self.get_coords(sh, row)
#                 crop = img[yi:yf, xi:xf]

#                 # Save the crop
#                 name = os.path.basename(filename).split('.t')[0] + f'_CROP-{cellid}.png'
#                 welldir = os.path.join(self.cropdir, well)
#                 if not os.path.exists(welldir):
#                     os.makedirs(welldir)
#                 croppath = os.path.join(welldir, name)
#                 imageio.v3.imwrite(croppath, crop)

#                 # Prepare database entry
#                 cropdcts.append(dict(id=uuid.uuid4(),
#                                      experimentdata_id=self.experimentdata_id,
#                                      welldata_id=df.welldata_id.iloc[0],
#                                      channeldata_id=target_channeldata_id,
#                                      celldata_id=row.id,
#                                      croppath=croppath))

#             # Add to database
#             self.Db.add_row('cropdata', cropdcts)

#             del img  # Free memory for the image
#             gc.collect()  # Trigger garbage collection

#     def get_coords(self, sh, row):
#         xi = int(row.centroid_x - self.opt.crop_size // 2)
#         xf = xi + self.opt.crop_size
#         yi = int(row.centroid_y - self.opt.crop_size // 2)
#         yf = yi + self.opt.crop_size
#         xi = xi if xi >= 0 else 0
#         yi = yi if yi >= 0 else 0
#         if xi == 0:
#             xf = self.opt.crop_size
#         if yi == 0:
#             yf = self.opt.crop_size
#         xf = xf if xf < sh[1] else sh[1] - 1
#         yf = yf if yf < sh[0] else sh[0] - 1
#         if xf == sh[1] - 1:
#             xi = sh[1] - (self.opt.crop_size + 1)
#         if yf == sh[0] - 1:
#             yi = sh[0] - (self.opt.crop_size + 1)
#         return xi, xf, yi, yf

# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--experiment', default='20230807-KS1-neuron-optocrispr', type=str)
#     parser.add_argument('--crop_size', default=300, type=int, help="Side length of square")
#     parser.add_argument("--wells_toggle", default='include',
#                         help="Chose whether to include or exclude specified wells.")
#     parser.add_argument("--timepoints_toggle", default='include',
#                         help="Chose whether to include or exclude specified timepoints.")
#     parser.add_argument("--channels_toggle", default='include',
#                         help="Chose whether to include or exclude specified channels.")
#     parser.add_argument("--chosen_wells", "-cw",
#                         dest="chosen_wells", default='F2',
#                         help="Specify wells to include or exclude")
#     parser.add_argument("--chosen_timepoints", "-ct",
#                         dest="chosen_timepoints", default='T0',
#                         help="Specify timepoints to include or exclude.")
#     parser.add_argument("--chosen_channels", "-cc",
#                         dest="chosen_channels", default='GFP-DMD1',
#                         help="Morphology Channel")
#     parser.add_argument("--target_channel",
#                         dest="target_channel", default='GFP-DMD1',
#                         help="Get intensity of this channel.")
#     parser.add_argument('--tile', default=0, type=int, help="Select single tile to segment. Default is to segment all tiles.")
#     args = parser.parse_args()
#     print(args)
#     Crp = Crop(args)
#     Crp.run()