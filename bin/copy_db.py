#!/opt/conda/bin/python
"""Copy experiment in one db to another db"""
from sql import Database
import os
import pandas as pd
import datetime
import argparse
import numpy as np
from tqdm import tqdm

class CopyCSVStoDB:
    def __init__(self, opt):
        self.opt = opt
        self.tablenames = ['experimentdata', 'welldata', 'channeldata', 'dosagedata', 'tiledata', 'celldata', 'intensitycelldata', 'punctadata', 'intensitypunctadata', 'modeldata', 'modelcropdata']
        self.nasdir = '/gladstone/finkbeiner/robodata/ThinkingMicroscope'

    def copytables(self):
        Db = Database()
        tiledata_df = pd.read_csv(os.path.join(self.opt.csvdir, f'tiledata.csv'))
        celldata_df = pd.read_csv(os.path.join(self.opt.csvdir, f'celldata.csv'))
        for tablename in self.tablenames:
            f = os.path.join(self.opt.csvdir, f'{tablename}.csv')
            if os.path.exists(f):
                print(f'Copying {tablename}')
                df = pd.read_csv(f)
                df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
                df = df.replace({np.nan: None})
                if tablename=='intensitycelldata':
                    df = df[df.tiledata_id.isin(tiledata_df.id) & df.celldata_id.isin(celldata_df.id)]
                dcts = []
                if not df.empty:
                    for i, row in tqdm(df.iterrows()):
                        if tablename == 'experimentdata':
                            if row.experiment != self.opt.experiment: continue
                            row['imagedir'] = os.path.join(f'/gladstone/finkbeiner/robodata/ThinkingMicroscope/{self.opt.experiment}')
                            row['analysisdir'] = os.path.join(f'/gladstone/finkbeiner/linsley/TM_analysis/GXYTMP-{self.opt.experiment}')
                            row['microscope'] = 'TM'
                            now = datetime.datetime.now()
                            analysisdate = f'{now.year}-{now.month:02}-{now.day:02}'
                            row['analysisdate'] = analysisdate
                        elif tablename=='tiledata':
                               # /gladstone/finkbeiner/robodata/ThinkingMicroscope/20230828-2-msneuron-cry2/F8
                            for key in ['filename', 'maskpath', 'trackedmaskpath']:
                                if row[key] is not None:
                                    parts = row[key].split('\\')
                                    filepath = '/'.join(parts)
                                    filepath = filepath.replace('D:/Images', '/gladstone/finkbeiner/robodata/ThinkingMicroscope')
                                    row[key] = filepath
                        
                        dcts.append(row.to_dict())
                        if len(dcts) > 1000:
                            Db.add_row(tablename, dcts)
                            dcts = []
                        # Db.add_row(tablename, row.to_dict())

                    Db.add_row(tablename, dcts)
        print('Done')

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
    parser.add_argument('--experiment',default='20230828-2-msneuron-cry2', type=str)
    parser.add_argument('--csvdir', default='/gladstone/finkbeiner/robodata/ThinkingMicroscope-DB/GXYTMP_20230828-2-msneuron-cry2/CSVS', type=str)
    args = parser.parse_args()
    print(args)
    TODB = CopyCSVStoDB(args)
    TODB.copytables()
