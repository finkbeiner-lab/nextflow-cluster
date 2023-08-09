"""Takes curated csv and tracking result and scores."""
from db_util import Ops
import argparse
import pandas as pd
import os

class ScoreTracking:
    def __init__(self, opt):
        self.opt = opt
        self.Op = Ops(self.opt)
        self.tolerance = 3
        self.results = dict(total=[], correct=[], wrong=[], acc=[], well=[],tile=[], experiment=[])
        self.imagedir, self.analysisdir = self.Op.get_raw_and_analysis_dir()

    def run(self):
        celldata_df = self.Op.get_celldata_df()
        celldata_df.sort_values(['cellid','timepoint'], inplace=True)
        celldata_df['gtcellid'] = -1
        wells = celldata_df.well.unique()
        tiles = celldata_df.tile.unique()
        for well in wells:
            for tile in tiles:
                gt_csv = os.path.join(self.imagedir, well, f'labels_{self.opt.experiment}_{well}.csv')
                if not os.path.exists(gt_csv):  # if there's not csv, don't run iteration
                    continue 
                gt = pd.read_csv(gt_csv)
                df = celldata_df.loc[(celldata_df.well==well) & (celldata_df.tile==tile)]
                # match up values within tolerance
                track_df = self.compare(df, gt)
                total, correct, wrong = self.score(track_df)
                self.results['total'].append(total)
                self.results['correct'].append(correct)
                self.results['wrong'].append(wrong)
                self.results['acc'].append(correct / total)
                self.results['well'].append(well)
                self.results['tile'].append(tile)
                self.results['experiment'].append(self.opt.experiment)
        pd.DataFrame(self.results).to_csv(os.path.join(self.analysisdir, f'{self.opt.experiment}_tracking_scores.csv'))



    def compare(self, track:pd.DataFrame, gt:pd.DataFrame):
        """Compare tracked dataframe with ground truth"""
        # align with position, may be a little off from segmentation

        timepoints = track.timepoint.unique()
    
        for timepoint in timepoints:
            _gt = gt[gt.timepoint==timepoint]
            _track = track[track.timepoint==timepoint]
            for i, row in _gt.iterrows():
                _track.loc[(abs(track.centroid_x - row.x) < self.tolerance) & (abs(track.centroid_y - row.y) < self.tolerance),
                          'gtcellid'] = row.cellid

            track.loc[_track.index] = _track

        return track
    
    def score(self, df):
        """Score, check when gtcellid and cellid do not agree"""
        correct_df = df.groupby('gtcellid').filter(lambda g: g['cellid'].nunique()==1)
        wrong_df = df.groupby('gtcellid').filter(lambda g: g['cellid'].nunique() > 1)
        well = correct_df.well.iloc[0]
        correct_df.to_csv(os.path.join(self.analysisdir, f'correct_tracks_{well}.csv'))
        wrong_df.to_csv(os.path.join(self.analysisdir, f'wrong_tracks_{well}.csv'))
        groups = df.groupby('gtcellid')
        total = len(groups)
        correct = len(correct_df.groupby('gtcellid'))
        wrong = len(wrong_df.groupby('gtcellid'))
        return total, correct, wrong
        
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
    parser.add_argument("--wells_toggle",
                        help="Chose whether to include or exclude specified wells.")
    parser.add_argument("--timepoints_toggle",
                        help="Chose whether to include or exclude specified timepoints.")
    parser.add_argument("--channels_toggle", default='include',
                        help="Chose whether to include or exclude specified channels.")
    parser.add_argument("--chosen_wells", "-cw",
                        dest="chosen_wells", default='',
                        help="Specify wells to include or exclude")
    parser.add_argument("--chosen_timepoints", "-ct",
                        dest="chosen_timepoints", default='',
                        help="Specify timepoints to include or exclude.")
    parser.add_argument("--chosen_channels", "-cc",
                        dest="chosen_channels",
                        help="Specify channels to include or exclude.")
    parser.add_argument('--tile', default=0, type=int, help="Select single tile to segment. Default is to segment all tiles.")
    args = parser.parse_args()
    print(args)
    Score = ScoreTracking(args)
    Score.run()

