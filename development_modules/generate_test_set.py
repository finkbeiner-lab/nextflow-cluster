"""Generates test images, for tracking at the moment."""
import numpy as np
import imageio
import pandas as pd
import os
import argparse


class TestSet:
    def __init__(self, opt):
        self.opt = opt
        self.well = None
        self.tile = 1
        self.sh = (1000, 1000)
        self.track = dict(cellid=[], timepoint=[], x=[], y=[])

    # create images
    # create directory structure for experiment
    # save images
    # create xlsx template

    def run(self):
        """Just a tile for well A1"""
        for i, cell in enumerate(range(4, 16)):
            self.well = f'B{i+1}'
            coord_dct = self.generate_positions(cell**2)
            for tp, coords in coord_dct.items():
                for cellid, x, y in coords:
                    self.track['cellid'].append(cellid)
                    self.track['x'].append(int(x))
                    self.track['y'].append(int(y))
                    self.track['timepoint'].append(tp)

            for timepoint, coords in coord_dct.items():
                im = np.zeros(self.sh, np.uint16)
                for cellid, x, y in coords:
                    # Add cells
                    im = self.make_cell(im, int(x), int(y))
                welldir = os.path.join(self.opt.image_dir, self.well)
                if not os.path.exists(welldir):
                    os.makedirs(welldir)
                # PID20230608_20230608-KS-neuron-gedi-minisog_T0_0-0_B1_1_RFP2_0.0_0_1.tif
                filename = f'PID2023_{self.opt.experiment}_T{timepoint}_0-0_{self.well}_{self.tile}_RFP_0_0_1.tif'
                savepath = os.path.join(welldir, filename)
                imageio.imwrite(savepath, im)
                print(f'Saved test image to {savepath}')
            df = pd.DataFrame(self.track)
            df.to_csv(os.path.join(self.opt.image_dir, self.well, f'labels_{self.opt.experiment}_{self.well}.csv'))

    def make_cell(self, im, x, y, side=10):
        im[y - side // 2:y + side // 2, x - side // 2:x + side // 2] = 50000
        return im

    def generate_positions(self, num_cells):
        cells = num_cells  # must be a square
        timepoints = 5
        current_cell_id = 1
        # grid
        coords = {i: [] for i in range(timepoints)}
        for timepoint in range(timepoints):
            if timepoint == 0:
                xs = np.linspace(100, self.sh[0] - 100, int(np.sqrt(cells)))
                ys = np.linspace(100, self.sh[1] - 100, int(np.sqrt(cells)))
                for x in xs:
                    for y in ys:
                        coords[timepoint].append((current_cell_id, int(x), int(y)))
                        current_cell_id += 1

            elif timepoint == 1:
                # random perturbation
                delta = 50
                for cellid, x, y in coords[timepoint - 1]:
                    dx = np.random.random() - .5
                    dy = np.random.random() - .5
                    coords[timepoint].append((cellid, x + delta * dx, y + delta * dy))
                assert len(coords[timepoint]) == len(coords[timepoint - 1])
            elif timepoint == 2:
                # split cell
                delta = 20
                for cellid, x, y in coords[timepoint - 1]:
                    dx = np.random.random() + .2
                    dy = np.random.random() + .2
                    coords[timepoint].append((cellid, x + delta * dx, y + delta * dy))
                    if np.random.random() > 0.7:
                        # add new cell
                        coords[timepoint].append((current_cell_id, x - 2 * delta * dx, y - 2 * delta * dy))
                        current_cell_id += 1
            elif timepoint == 3:
                # stronger random perturbation
                delta = 100
                for cellid, x, y in coords[timepoint - 1]:
                    dx = np.random.random() - .5
                    dy = np.random.random() - .5
                    coords[timepoint].append((cellid, x + delta * dx, y + delta * dy))
            elif timepoint == 4:
                # drop cell
                delta = 20
                for cellid, x, y in coords[timepoint - 1]:
                    if np.random.random() < 0.8:
                        dx = np.random.random() - .5
                        dy = np.random.random() - .5
                        coords[timepoint].append((cellid, x + delta * dx, y + delta * dy))
        return coords


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--outfile',
        help='Tiff image of last tile',
        default=f'/gladstone/finkbeiner/linsley/josh/GALAXY/YD-Transdiff-XDP-Survival1-102822/GXYTMP/tmp_output.tif'
    )
    parser.add_argument('--experiment', type=str)
    parser.add_argument('--image_dir', type=str)
    # parser.add_argument('--well', type=str)
    # parser.add_argument('--cells', type=str)

    args = parser.parse_args()
    print(args)
    Test = TestSet(args)
    Test.run()
