"""Copy experiment in one db to another db"""
from sql import Database
import os
import pandas as pd
import argparse

class CopyCSVStoDB:
    def __init__(self, opt):
        self.opt = opt
        self.tablenames = ['experimentdata', 'welldata', 'channeldata', 'dosagedata', 'tiledata', 'celldata', 'intensitycelldata', 'punctadata', 'intensitypunctadata', 'modeldata', 'modelcropdata']

    def copytables(self):
        Db = Database()
        for tablename in self.tablenames:
            f = os.path.join(self.opt.csvdir, f'{tablename}.csv')
            if os.path.exists(f):
                df = pd.read_csv(f)
                df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
                for i, row in df.iterrows():
                    if tablename == 'experimentdata' and row.experiment != self.opt.experiment: continue
                    row_dct = row.to_dict()
                    Db.add_row(tablename, row_dct)

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
    args = parser.parse_args()
    print(args)
    TODB = CopyCSVStoDB(args)
    TODB.copytables()
