#!/opt/conda/bin/python

"""Update path on database after moving from TM
"""
from sql import Database
import os
import datetime
import argparse


class UpdatePath:
    def __init__(self, opt):
        self.opt = opt
        self.Db = Database()
        self.robofolder = dict(TM='ThinkingMicroscope', Robo4='Robo4Images')
        self.origfolder = dict(TM='D:/Images', Robo4='E:/Images')

    def run(self):
        # Add to database
        now = datetime.datetime.now()
        analysisdate = f'{now.year}-{now.month:02}-{now.day:02}'
        microscope = self.Db.get_table_value(
            tablename='experimentdata', column='microscope', kwargs=dict(experiment=self.opt.experiment))
        author = self.Db.get_table_value(
            tablename='experimentdata', column='researcher', kwargs=dict(experiment=self.opt.experiment))
        exp_uuid = self.Db.get_table_uuid(
            tablename='experimentdata', kwargs=dict(experiment=self.opt.experiment))
        print(microscope)
        assert microscope[0][0] in ['TM', 'Robo4'], 'Experiment is not from Thinking Microscope or Robo4'
        analysisdir = os.path.join(
            f'/gladstone/finkbeiner/linsley/{author}/GXYTMPS/GXYTMP-{self.opt.experiment}')
        imagedir = os.path.join(
            f'/gladstone/finkbeiner/robodata/{self.robofolder[microscope]}/{self.opt.experiment}')
        if not os.path.exists(analysisdir):
            os.makedirs(analysisdir)
        self.Db.update(tablename='experimentdata',
                       update_dct=dict(imagedir=imagedir,
                                       analysisdir=analysisdir,
                                       analysisdate=analysisdate),
                       kwargs=dict(experiment=self.opt.experiment))

        # /gladstone/finkbeiner/robodata/ThinkingMicroscope/20230828-2-msneuron-cry2/F8
        self.Db.update_slashes('tiledata', exp_uuid=exp_uuid)
        self.Db.update_prefix_path('tiledata', exp_uuid=exp_uuid, old_string=self.origfolder[microscope],
                                   new_string=f'/gladstone/finkbeiner/robodata/{self.robofolder[microscope]}')
        print(f'Done: Replaced {self.origfolder[microscope]} with /gladstone/finkbeiner/robodata/{self.robofolder[microscope]}')


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
    parser.add_argument(
        '--experiment', default='112023-TH-GEDI-DSMs', type=str)
    args = parser.parse_args()
    print(args)
    Up = UpdatePath(args)
    Up.run()
