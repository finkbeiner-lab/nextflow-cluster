#!/opt/conda/bin/python
"""Copy _MASK.tif to _MASKTRACKED.tif for T0 only and update database trackedmaskpath column"""

import os
import shutil
import argparse
from db_util import Ops
from sql import Database


class MaskToTracked:
    def __init__(self, opt):
        self.opt = opt
        self.Dbops = Ops(opt)
        self.Db = Database()
        _, self.analysisdir = self.Dbops.get_raw_and_analysis_dir()
        self.experimentdata_id = self.Db.get_table_uuid('experimentdata', dict(experiment=self.opt.experiment))

    def run(self):
        tiledata = self.Dbops.get_tiledata_df()
        tiledata = tiledata[tiledata['timepoint'] == 0]

        for idx, row in tiledata.iterrows():
            maskpath = row.maskpath
            print(f"Checking existence of: {maskpath}")
            if maskpath and '_T0_' in maskpath and '_MASK.tif' in maskpath and os.path.exists(maskpath):
                masktracked_path = maskpath.replace('_MASK.tif', '_MASKTRACKED.tif')

                shutil.copyfile(maskpath, masktracked_path)
                print(f"Copied {maskpath} to {masktracked_path}")

                self.Db.update(
                    tablename='tiledata',
                    update_dct={'trackedmaskpath': masktracked_path},
                    kwargs=dict(id=row.id)
                )
                



        print("All T0 MASK files processed and database updated successfully!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Copy MASK.tif to MASKTRACKED.tif and update database.")
    parser.add_argument('--experiment', required=True, type=str)
    parser.add_argument("--chosen_wells", default='all', type=str)
    parser.add_argument("--chosen_timepoints", default='T0', type=str)
    parser.add_argument("--chosen_channels", default='all', type=str)
    parser.add_argument("--wells_toggle", default='include', type=str)
    parser.add_argument("--timepoints_toggle", default='include', type=str)
    parser.add_argument("--channels_toggle", default='include', type=str)
    parser.add_argument('--tile', default=0, type=int, help="Select single tile. Default=0 for all tiles.")

    opt = parser.parse_args()

    masker = MaskToTracked(opt)
    masker.run()