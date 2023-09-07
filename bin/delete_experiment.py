#!/opt/conda/bin/python

from sql import Database
import argparse

def delete_experiment(opt):
    Db = Database()
    exp_dct = dict(experiment=opt.experiment)
    Db.delete_based_on_duplicate_name('experimentdata', exp_dct)
    print(f'Deleted experiment {opt.experiment}')


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
    parser.add_argument('--experiment', default='20230828-2-msneuron-cry2', type=str)
    args = parser.parse_args()
    print(args)
    delete_experiment(args)
