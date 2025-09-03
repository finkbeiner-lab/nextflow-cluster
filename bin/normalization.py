#!/opt/conda/bin/python

"""Normalization file. Divide or subtract by flatfield image. Get flatfield by median of N images."""
import numpy as np
from db_util import Ops
import random
import imageio
import os
from time import time
from skimage import filters, restoration, transform
import argparse
import cv2


class Normalize(Ops):
    def __init__(self, opt):
        super().__init__(opt)
        self.flatfields = {}
        self.backgrounds = {}
        # self.image_correction = dict(division=self.division_flatfield,
        #                              subtraction=self.subtract_flatfield,
        #                              identity=self.identity,
        #                              rollingball=self.rolling_ball)
        self.image_bg_correction = dict(division=self.division_bg,
                                        subtraction=self.subtract_bg,
                                        identity=self.identity_bg)

                
    def test(self):
        """Save background corrected images for viewing"""
        _, analysisdir = self.get_raw_and_analysis_dir()
        savedir = os.path.join(analysisdir, 'NormalizedImages')
        if not os.path.exists(savedir):
            os.makedirs(savedir)
        tiledata_df = self.get_df_for_training(['channeldata'])
        tiledata_df.rename(columns={'id': 'tiledata_id'}, inplace=True)
        g = tiledata_df.groupby(['well', 'timepoint', 'channel'])
        for (well, timepoint, channel), df in g:
            self.get_background_image(df, well, timepoint)
            
            for i, row in df.iterrows():
                # print('row', row)
                
                img = imageio.v3.imread(row.filename)
                img = np.uint16(self.image_bg_correction[self.opt.img_norm_name](img, well, timepoint))
                normpath = self.save_norm(img, row.filename, savedir, well)
                print('normpath', normpath)
            del self.backgrounds[well][timepoint]

    def to_eight_bit(self, target):
        return np.uint8(target / np.max(target) * 255)

    # def identity(self, img, tile):
    #     return img

    # def division_flatfield(self, img, tile):
    #     im = img / self.flatfields[tile]
    #     im = im / np.max(im) * 50000
    #     return im

    # def rolling_ball(self, img, tile):
    #     # img = np.uint16(self.gaussian_filter(img))
    #     img = transform.rescale(img, 1 / 8, anti_aliasing=True)
    #     background = restoration.rolling_ball(img, radius=100)

    #     im = img - background
    #     im = np.uint16(transform.rescale(im, 8,))
    #     return im

    # def subtract_flatfield(self, img, tile):
    #     im = img - self.flatfields[tile]
    #     print('flatfield', np.min(self.flatfields[tile]), np.max(self.flatfields[tile]))
    #     print('im', np.min(im), np.max(im))
    #     im[im < 0] = 0
    #     return im

    def identity_bg(self, img, well, timepoint):
        return img

    def division_bg(self, img, well, timepoint):
        im = img / self.backgrounds[well][timepoint]
        im = im / np.max(im) * 50000
        return im

    def subtract_bg(self, img, well, timepoint):
        im = img - self.backgrounds[well][timepoint]
        im[im < 0] = 0
        return im

    def z_whitening(self, img):
        return (img - np.mean(img)) / np.std(img)

    def gaussian_filter(self, target, sigma=3):
        smooth = filters.gaussian(target, sigma=sigma)
        return smooth

    def get_flatfields(self):
        df = self.get_tiledata_df()
        if self.opt.img_norm_name not in ['division', 'subtraction']:
            return df
        tiles = sorted(list(df.tile.unique()))
        for i in tiles:
            img_lst = []
            # Different timepoint, same tile
            filenames = df.loc[df.tile == i, 'filename'].tolist()
            random.seed(121)
            if not len(filenames):
                print(filenames)
                print('ONLY ONE FILE: TOO FEW FOR BACKGROUND SUBTRACTION/DIVISION')
            filenames = random.sample(filenames, min(20, len(filenames)))
            for f in filenames:
                img = imageio.v3.imread(f)
                img_lst.append(img)
            flat = np.median(img_lst, axis=0)
            flat[flat < 1] = 1
            self.flatfields[i] = flat
        return df

    def get_flatfields_for_training(self, tablenames: list):
        df = self.get_df_for_training(tablenames)
        if self.opt.img_norm_name not in ['division', 'subtraction']:
            return df
        tiles = sorted(list(df.tile.unique()))
        for i in tiles:
            img_lst = []
            filenames = df.loc[df.tile == i, 'filename'].tolist()
            random.seed(121)
            if len(filenames):
                pass  # Skip verbose message
            filenames = random.sample(filenames, min(20, len(filenames)))
            for f in filenames:
                img = imageio.imread(f)
                img_lst.append(img)
            flat = np.median(img_lst, axis=0)
            flat[flat < 1] = 1
            self.flatfields[i] = flat
        return df

    @staticmethod
    def in_well(df, x):
        return df.well == x

    @staticmethod
    def around_well(df, x):
        return df.well == x

    def get_background_image(self, df, well, timepoint):
        if well in self.backgrounds and timepoint in self.backgrounds[well]:
            return
        else:
            self.collect_images(df, well, timepoint)
        return

    def collect_images(self, df, well, timepoint):
        strt = time()
        img_lst = []
        filenames = df.filename.tolist()
        for f in filenames:
            img = imageio.v3.imread(f)
            img_lst.append(img)
        bg = np.median(img_lst, axis=0)
        bg[bg < 1] = 1
        if well not in self.backgrounds:
            self.backgrounds[well] = {}
        self.backgrounds[well][timepoint] = bg

    def collect_images_by_timepoint(self, df):
        tiles = sorted(list(df.tile.unique()))
        for i in tiles:
            img_lst = []
            filenames = df.loc[df.tile == i, 'filename'].tolist()
            random.seed(121)
            if len(filenames):
                print('ONLY ONE FILE: TOO FEW FOR BACKGROUND SUBTRACTION/DIVISION')
            filenames = random.sample(filenames, min(20, len(filenames)))
            for f in filenames:
                img = imageio.imread(f)
                img_lst.append(img)
            flat = np.median(img_lst, axis=0)
            flat[flat < 1] = 1
            self.flatfields[i] = flat

    def collect_images_by_tile_position(self, df):
        tiles = sorted(list(df.tile.unique()))
        for i in tiles:
            img_lst = []
            filenames = df.loc[df.tile == i, 'filename'].tolist()
            random.seed(121)
            if len(filenames):
                print('ONLY ONE FILE: TOO FEW FOR BACKGROUND SUBTRACTION/DIVISION')
            filenames = random.sample(filenames, min(20, len(filenames)))
            for f in filenames:
                img = imageio.imread(f)
                img_lst.append(img)
            flat = np.median(img_lst, axis=0)
            flat[flat < 1] = 1
            self.flatfields[i] = flat

    def save_norm(self, normed_image, image_file, saveparentdir, well):
        name = os.path.basename(image_file)
        name = name.split('.t')[0]  # split by tiff suffix
        savedir = os.path.join(saveparentdir, well)
        if not os.path.exists(savedir):
            os.makedirs(savedir)
        savepath = os.path.join(savedir, name + '_' + self.opt.img_norm_name + '-NORM.tif')
        cv2.imwrite(savepath, normed_image)
        return savepath


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--outfile',
        help='Tiff image of last tile',
        default=f'/gladstone/finkbeiner/linsley/josh/GALAXY/YD-Transdiff-XDP-Survival1-102822/GXYTMP/tmp_output.tif'
    )
    parser.add_argument('--experiment', default='JAK-COR7508012023-GEDI', type=str)
    parser.add_argument('--img_norm_name', default='subtraction', choices=['division', 'subtraction', 'identity', 'rollingball'], type=str,
                        help='Image normalization method using flatfield image.')
    parser.add_argument("--wells_toggle", default='include',
                        help="Chose whether to include or exclude specified wells.")
    parser.add_argument("--timepoints_toggle", default='include', 
                        help="Chose whether to include or exclude specified timepoints.")
    parser.add_argument("--channels_toggle", default='include',
                        help="Chose whether to include or exclude specified channels.")
    parser.add_argument("--chosen_wells", "-cw", 
                        dest="chosen_wells", default='B03',
                        help="Specify wells to include or exclude")
    parser.add_argument("--chosen_timepoints", "-ct",
                        dest="chosen_timepoints", default='T1',
                        help="Specify timepoints to include or exclude.")
    parser.add_argument("--chosen_channels", "-cc", default='all',
                        dest="chosen_channels",
                        help="Specify channels to include or exclude.")
    parser.add_argument('--tile', default=0, type=int, help="Select single tile to segment. Default is to segment all tiles.")
    args = parser.parse_args()
    Norm = Normalize(args)
    Norm.test()
