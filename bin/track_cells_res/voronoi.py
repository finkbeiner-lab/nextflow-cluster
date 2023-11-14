import time

import numpy as np
import cv2
# from scipy.stats import mode
# from matplotlib import pyplot as plt
import imageio
# import os
# import sys
from scipy.spatial import Voronoi
# import pandas as pd
import warnings
from threading import Thread
import logging
logger = logging.getLogger("Tracking")


class VoronoiClass:
    def __init__(self, dataframe, image_shape, debug=False):
        self.vor = None
        self.centroids = None
        self.lowcentroids = None
        self.dataframe = dataframe
        self.sh = image_shape

        self.debug = debug
        self.num_makeline = 30000
        self.pixel_locations = []
        self.width = self.sh[1]
        self.height = self.sh[0]
        logger.info(f'image shape {self.sh}')
        self.corner_centroids = [[0, 0],
                                 [self.sh[0] - 1, 1],
                                 [0, self.sh[1] - 1],
                                 [self.sh[0] - 1, self.sh[1] - 1]]
        self.low_flag = None
        self.ctr_idx_dict = {}
        # self.centroid_loc_dict = {}
        # self.centroid_mask_dict = {}
        self.centroid_contour_dict = {}
        self.centroid_vorarea_dict = {}
        self.centroid_vorcom_dict = {}
        self.randomcellid_str = 'randomcellid'
        self.centroid_x_str = 'centroid_x'
        self.centroid_y_str = 'centroid_y'

    def apply_voronoi(self):
        if len(self.centroids) < 4:
            self.lowcentroids = np.concatenate((self.corner_centroids, self.centroids))
            self.low_flag = True
        else:
            self.lowcentroids = None
            self.low_flag = False
        # logger.info('centroids', self.centroids)
        if len(self.centroids) > 0:
            # Voronoi needs at least 4 points
            # Centroids are all in pixel space. They are centroids in an image. But they're floats.
            if not self.low_flag:
                vor = Voronoi(self.centroids)
            else:
                vor = Voronoi(self.lowcentroids)
        # if self.low_flag:
        #     self.lowcentroids = np.uint64(np.floor(vorcentroids))
        #     self.corner_centroids = self.lowcentroids[:4]
        #
        # else:
        #     self.lowcentroids = self.lowcentroids
        #     self.corner_centroids = None
        self.vor = vor

    def vor2mask(self):
        """
        Main function, sets and labels voronoi mask -
        Returns:
        """
        start = time.time()
        border = np.zeros(self.sh, dtype=np.uint8)  # boolean?
        mask = np.ones(self.sh, dtype=np.uint8)
        border_locations = self.handle_vor_at_infinity(verbose=False)
        logger.info(f'border locations {time.time() - start}')

        border = self.set_border(border, border_locations)
        logger.info(f'set border {time.time() - start}')

        mask = mask - border
        # INITIAL EROSION OF BINARY OR ENCODED MASK
        mask = cv2.erode(mask, np.ones((5, 5)), iterations=1)
        if self.debug and 0:
            plt.figure()
            plt.imshow(mask)
            for cc in self.centroids:
                plt.scatter(cc[0], cc[1])
            plt.title('eroded mask')
        labels, labelled_mask, stats, region_centroids = cv2.connectedComponentsWithStats(mask)
        logger.info(f'connected components {time.time() - start}')

        if self.debug and 0:
            fig = plt.figure()
            plt.imshow(labelled_mask)
            for cc in self.centroids:
                plt.scatter(cc[0], cc[1])
            plt.title('CV2 Labels')
        else:
            fig = None
        if self.debug and 0:
            plt.show()

        return labelled_mask

    def contour_to_idx_dict(self):
        """Returns dict with pandas dataframe index as key"""
        idx_contour_dict = {}
        for ctr, mask in self.centroid_contour_dict.items():
            idx = self.ctr_idx_dict[ctr]  # centroid idx
            idx_contour_dict[idx] = mask
        return idx_contour_dict

    def vorarea_to_idx_dict(self):
        """Returns dict with pandas dataframe index as key"""
        idx_vorarea_dict = {}
        for ctr, mask in self.centroid_vorarea_dict.items():
            idx = self.ctr_idx_dict[ctr]  # centroid idx
            idx_vorarea_dict[idx] = mask
        return idx_vorarea_dict

    def vorcom_to_idx_dict(self):
        """Returns dict with pandas dataframe index as key"""
        idx_vorcom_dict = {}
        for ctr, com in self.centroid_vorcom_dict.items():
            idx = self.ctr_idx_dict[ctr]  # centroid idx
            idx_vorcom_dict[idx] = com
        return idx_vorcom_dict

    # def locs_to_idx_dict(self):
    #     idx_loc_dict = {}
    #     for ctr, locs in self.centroid_loc_dict.items():
    #         idx = self.ctr_idx_dict[ctr]
    #         idx_loc_dict[idx] = locs
    #     return idx_loc_dict

    # def locs_to_df(self, target_df):
    #     """
    #     Per well per timepoint updates celldata dataframe with string of coordinates
    #     :param target_df: df to update
    #     :param centroid_loc_dict:
    #     :param ctr_idx_dict:
    #     :return:
    #     """
    #     for ctr, locs in self.centroid_loc_dict.items():
    #         idx = self.ctr_idx_dict[ctr]
    #         locstr = self.locs_to_str(locs)  # separated coords as string 'x_y-x2_y2-x3_y3'
    #         target_df.loc[idx, 'voronoi_region'] = locstr
    #     return target_df

    @staticmethod
    def locs_to_str(locs):
        """Convert region coords to str to store in pandas"""
        res = ''
        for pair in locs:
            x = pair[0]  # todo: might be switched, check csv on a well
            y = pair[1]
            res += f'{x}_{y}'
            res += '-'
        return res[:-1]

    def get_centroids(self, index_name='cellid'):
        indices = self.dataframe[index_name].values
        centroid_x = np.uint32(self.dataframe[self.centroid_x_str].values)
        centroid_y = np.uint32(self.dataframe[self.centroid_y_str].values)
        self.centroids = np.vstack((centroid_x, centroid_y)).T
        for ctr, idx in zip(self.centroids, indices):
            self.ctr_idx_dict[(ctr[1], ctr[0])] = idx
        return 0

    #
    # def str_locs_to_coords(self, strlocs):
    #     coords = strlocs.split('-')
    #     return coords

    def set_labels_on_mask(self, mask, label_name='cellid'):
        """Use dataframe and centroids to reset labels on image"""
        mask = np.float32(mask)
        lbl_on_edge = []  # if lbl not found in this set, send to minflow
        for i, row in self.dataframe.iterrows():
            lbl = row[label_name]
            random_lbl = mask[int(row[self.centroid_y_str]), int(row[self.centroid_x_str])]
            if not random_lbl:
                lbl_on_edge.append(lbl)
            else:
                mask[mask == random_lbl] = lbl + .1
        # set any regions that haven't changed to zero
        filter = np.abs((mask - np.uint16(mask)) * 10)
        res = np.uint16(mask * filter)
        # assert np.all(np.unique(res)[1:] == self.dataframe[label_name].values), 'mask does not match labels'
        return res, lbl_on_edge

    def handle_vor_at_infinity(self, verbose=False):
        if not self.low_flag:
            _centroids = np.float64(self.centroids)
        else:
            _centroids = np.float64(self.lowcentroids)
        border_locs = []
        delta = 10
        # xmin = np.min(self.centroids[:, 0]) - delta
        # ymin = np.min(self.centroids[:, 1]) - delta
        # xmax = np.max(self.centroids[:, 0]) + delta
        # ymax = np.max(self.centroids[:, 1]) + delta
        # if self.debug:
        #     plt.figure(5)
        #     plt.plot(_centroids[:, 0], _centroids[:, 1], 'o')
        #     plt.plot(self.vor.vertices[:, 0], self.vor.vertices[:, 1], '*')
        #     plt.xlim(0, self.width)
        #     plt.ylim(0, self.height)
        for simplex in self.vor.ridge_vertices:
            simplex = np.asarray(simplex)
            if np.all(simplex >= 0):
                _lin = self.makeline_by_points(self.vor.vertices[simplex, 0], self.vor.vertices[simplex, 1],
                                               N=self.num_makeline)
                border_locs.append(_lin)  # positions to set to 1 in mask
                # logger.info('border_locs', border_locs)
                if self.debug and 0:
                    plt.plot(self.vor.vertices[simplex, 0], self.vor.vertices[simplex, 1], 'b-')

        # Handle points at infinity
        center = _centroids.mean(axis=0)
        for pointidx, simplex in zip(self.vor.ridge_points, self.vor.ridge_vertices):
            simplex = np.asarray(simplex)
            if np.any(simplex < 0):  # ridge vertex < 0 means there's a point at infinity
                i = simplex[simplex >= 0][0]  # finite end Voronoi vertex

                t = _centroids[pointidx[1]] - _centroids[pointidx[0]]  # tangent, vector subtraction
                if verbose:
                    logger.info(f'pointidx {pointidx}')
                    logger.info(f'centroids 1 {_centroids[pointidx[1]]}')
                    logger.info(f'centroids 0 {_centroids[pointidx[0]]}')
                    logger.info(f'vor vertex 0 {self.vor.vertices[i, 0]}')
                    logger.info(f'vor vertex 1 {self.vor.vertices[i, 1]}')
                    logger.info(f't {t}')
                t = t / np.linalg.norm(t)
                n = np.array([-t[1], t[0]])  # normal, (normal to tangent)
                midpoint = _centroids[pointidx].mean(axis=0)
                far_point = self.vor.vertices[i] + np.sign(np.dot(midpoint - center, n)) * n * self.num_makeline
                if verbose:
                    logger.info(f'far point {far_point}')
                _lin_infinity = self.makeline(x1=self.vor.vertices[i, 0], x2=far_point[0],
                                              y1=self.vor.vertices[i, 1],
                                              y2=far_point[1],
                                              N=self.num_makeline)
                border_locs.append(_lin_infinity)
                # logger.info('border locs size', sys.getsizeof(border_locs))
                # if self.debug:
                #     plt.plot([self.vorin.vertices[i, 0], far_point[0]],
                #              [self.vorin.vertices[i, 1], far_point[1]], 'r--')
                #     plt.title('Vor')
                #     # plt.axis('square')
                #     plt.show()
        return border_locs

    def set_border(self, border, border_locations):
        """
        Marks border with one
        :param border: numpy zero image
        :param border_locations: border locations from voronoi
        :return: border
        """
        for b in border_locations:
            _idx = list(np.int64(np.round(np.transpose(b))))
            use_idx = np.array(
                [i for i in _idx if i[0] >= 0 and i[0] < self.width and i[1] >= 0 and i[1] < self.height],
                dtype=np.int64)
            # logger.info('idx shape', use_idx.shape)
            if use_idx.shape[0] > 0:
                for u in use_idx:
                    border[u[1], u[0]] = 1
        return border

    def makeline_by_points(self, p1, p2, N):
        x1 = p1[0]
        x2 = p1[1]
        y1 = p2[0]
        y2 = p2[1]
        xsize = np.abs(x1 - x2)
        xsign = np.sign(x2 - x1)
        ysign = np.sign(y2 - y1)
        ysize = np.abs(y1 - y2)

        slope = float(y2 - y1) / (x2 - x1)
        # logger.info('x1, x2', x1, x2)

        if xsize >= ysize:
            x = np.arange(x1, x2, xsign * 0.3)
            y = slope * (x - x1) + y1
        else:
            y = np.arange(y1, y2, ysign * 0.3)
            x = ((y - y1) / slope) + x1
        # x = np.linspace(x1, x2, N)
        # y = np.linspace(y1, y2, N)
        x = np.int64(np.round(x))
        y = np.int64(np.round(y))
        argx = np.argwhere((x < 0) | (x > self.width))
        x = np.delete(x, argx)
        y = np.delete(y, argx)
        argy = np.argwhere((y < 0) | (y > self.height))
        x = np.delete(x, argy)
        y = np.delete(y, argy)
        lin = np.vstack((x, y))
        return lin.tolist()

    def makeline(self, x1, x2, y1, y2, N):
        """
        y - y1 = m ( x - x1)
        Makes a line. Linspace then stack.
        Args:
            x1: point 1, x
            x2: point 2, x
            y1: point 1, y
            y2: point 2, y
            N: Number of points in linspace
        Returns:
        """
        # logger.info('makeline')
        xsize = np.abs(x1 - x2)
        ysize = np.abs(y1 - y2)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            slope = (y2 - y1) / (x2 - x1)
        # logger.info('slope', slope)
        xsign = np.sign(x2 - x1)
        ysign = np.sign(y2 - y1)
        # on zero division warning, xsize < ysize because xsize is zero
        if xsize >= ysize:
            x = np.arange(x1, x2, xsign * 0.3, dtype=np.float32)
            y = slope * (x - x1) + y1
        else:
            y = np.arange(y1, y2, ysign * 0.3, dtype=np.float32)
            x = ((y - y1) / slope) + x1  # dividing by infinite slope cancels the first term

        x = np.int64(np.round(x))
        y = np.int64(np.round(y))
        x = np.int64(np.round(x))
        y = np.int64(np.round(y))
        argx = np.argwhere((x < 0) | (x > self.width))
        x = np.delete(x, argx)
        y = np.delete(y, argx)
        argy = np.argwhere((y < 0) | (y > self.height))
        x = np.delete(x, argy)
        y = np.delete(y, argy)
        # x = np.linspace(x1, x2, N)
        # y = np.linspace(y1, y2, N)
        lin = np.vstack((x, y))
        return lin.tolist()


def voronoi_intersection(loc1, loc2):
    """
    :param loc1: str, encoded 'sh[0]_sh[1]_0_cnt_1_cnt_0_cnt...'
    :param loc2: str, encoded
    :return:
    """
    # loc1 = self.decode_mask(loc1)
    # loc2 = self.decode_mask(loc2)
    loc_mult = loc1 * loc2
    # loc_sum = loc1 + loc2
    intersection = np.count_nonzero(loc_mult)
    if intersection > 0:
        # union = np.count_nonzero(loc_sum)
        intersection_over_node1 = intersection / np.count_nonzero(loc1)
        intersection_over_node2 = intersection / np.count_nonzero(loc2)
    else:
        # union = 1
        intersection_over_node1 = 0
        intersection_over_node2 = 0
    # set1 = set(tuple(x) for x in loc1)  # set1 is l_node, previous timepoint
    # set2 = set(tuple(x) for x in loc2)  # set2 is r_node, current timepoint
    # intersection = set1.intersection(set2)
    # union = set1.union(set2)
    # iou = intersection / union
    return intersection_over_node1, intersection_over_node2


def plot_matched(mask_arr, img_arr=None, save_file=None, add_spots=True, erode_bool=True, verbose=False,
                 plot_bool=False, use16=True,
                 save_bool=True):
    if verbose:
        logger.info('Plot matched')
    sh = np.shape(mask_arr)
    # logger.info('sh', sh)
    num = sh[0]
    x = sh[1]
    y = sh[2]
    montage = np.zeros((x, y * num))
    for i, mask in enumerate(mask_arr):
        # logger.info('i', i)
        montage[:, i * y: (i + 1) * y] = mask

    if img_arr is not None:
        im_montage = np.zeros((x, y * num))
        for i, img in enumerate(img_arr):
            # logger.info('i', i)
            im_montage[:, i * y: (i + 1) * y] = img
        # plt.figure(num=2, figsize=(12, 8))
        # plt.imshow(im_montage)
        if add_spots:
            filt_montage = (
                    np.ones_like(im_montage) - im_montage / np.max(im_montage))  # max value of binary neuron labels
            # Commented - was throwing error on reverse non-encoded.
            # assert len(np.unique(filt_montage)) == 2, '{}'.format(len(np.unique(filt_montage)))
            montage = montage * filt_montage
    if erode_bool:
        cv2.erode(montage, kernel=np.ones((5, 5), dtype=np.uint8))
    if plot_bool:
        pass
        # plt.figure(figsize=(22,))
        # plt.imshow(montage)
        # plt.show()
    if save_bool:
        if use16:
            _montage = np.uint16(montage)
        else:
            _montage = np.uint8(montage)
        imageio.imwrite(save_file, _montage)
        logger.info('Saved to:', save_file)


def run_voronoi(celldata1, celldata2, img1, img2, debug=False):

    VR1 = VoronoiClass(celldata1, img1, debug)
    VR1.get_centroids()
    VR1.apply_voronoi()
    mask1 = VR1.vor2mask()
    logger.info('labelling mask 1')
    mask1, cellids_for_minflow = VR1.set_labels_on_mask(mask1, label_name='cellid')

    VR2 = VoronoiClass(celldata2, img2, debug)
    VR2.get_centroids()  # todo: do i need to get the centroids in a dict?
    VR2.apply_voronoi()
    mask2 = VR2.vor2mask()
    logger.info('labelling mask 2')
    mask2, randomcellids_for_minflow = VR2.set_labels_on_mask(mask2, label_name='randomcellid')

    cellids = np.unique(mask1)[1:]
    randomcellids = np.unique(mask2)[1:]

    edges_to_keep = []
    logger.info(f'cellids for minflow {cellids_for_minflow}')
    logger.info(f'randomcellids for minflow {randomcellids_for_minflow}')  # todo: this is high because the rows are duplicated on the database
    for cellid in cellids:
        for randomcellid in randomcellids_for_minflow:
            edges_to_keep.append((f'L{cellid}', f'R{randomcellid}'))
    for randomcellid in randomcellids:
        for cellid in cellids_for_minflow:
            edges_to_keep.append((f'L{cellid}', f'R{randomcellid}'))
    jobs = []
    cnt = 0
    for cellid in cellids:
        #todo: To plot with matplotlib, function needs to be in main thread
        # thread_intersection(mask1, mask2, cellid, edges_to_keep,'forward')
        thread = Thread(target=thread_intersection, args=(mask1, mask2, cellid, edges_to_keep,'forward'))
        jobs.append(thread)
        if len(jobs) >= 12:
            for j in jobs:
                j.start()
            for j in jobs:
                j.join()
            jobs = []
            cnt += 1
    if len(jobs) > 0:
        for j in jobs:
            j.start()
        for j in jobs:
            j.join()
    jobs = []

        # thread_intersection(mask1, mask2, cellid, edges_to_keep, direction='forward')
        # filter = mask1 == cellid
        # # area = np.count_nonzero(filter)
        # cellid_on_mask2 = filter * mask2
        # bins = np.bincount(cellid_on_mask2.flatten())
        # uni = np.nonzero(bins)
        # cnts = bins[uni]
        # # uni, cnts = np.unique(cellid_on_mask2, return_counts=True)
        # for u, cnt in zip(uni[1:], cnts[1:]):  # exclude zero
        #     # percentage = cnt / area
        #     edges_to_keep.append((f'L{cellid}', f'R{u}'))
    cnt = 0
    for randomcellid in randomcellids:
        thread = Thread(target=thread_intersection, args=(mask2, mask1, randomcellid, edges_to_keep, 'reverse'))
        jobs.append(thread)
        if len(jobs) >= 12:
            # init_thread_time = time.time()
            for j in jobs:
                j.start()
            for j in jobs:
                j.join()
            jobs = []
            cnt += 1
            # logger.info('count', cnt)
            # logger.info('thread time:', time.time() - init_thread_time)
    if len(jobs) > 0:
        for j in jobs:
            j.start()
        for j in jobs:
            j.join()
    jobs = []
        # thread_intersection(mask2, mask1, randomcellid, edges_to_keep, direction='forward')

        # filter = mask2 == randomcellid
        # # area = np.count_nonzero(filter)
        # randomcellid_on_mask1 = filter * mask1
        # bins = np.bincount(randomcellid_on_mask1.flatten())
        # uni = np.nonzero(bins)
        # cnts = bins[uni]
        # # uni, cnts = np.unique(randomcellid_on_mask1, return_counts=True)
        # for u, cnt in zip(uni[1:], cnts[1:]):  # exclude zero
        #     # percentage = cnt / area
        #     edges_to_keep.append((f'L{u}', f'R{randomcellid}'))

    return edges_to_keep

def thread_intersection(mask_ident, mask_base, cell_ident, edges_to_keep, direction, debug=False):
    filter = mask_ident == cell_ident
    # area = np.count_nonzero(filter)
    cell_ident_on_mask = filter * mask_base
    if debug and 0:
        plt.figure()
        plt.imshow(filter * 255)
        plt.title(f'filter intersection {cell_ident}')
        plt.figure()
        plt.imshow(cell_ident_on_mask)
        plt.title('Filter Projection')
        plt.show()
    bins = np.bincount(cell_ident_on_mask.flatten())
    uni = np.nonzero(bins)[0]
    cnts = bins[uni]
    # uni, cnts = np.unique(cellid_on_mask2, return_counts=True)
    for u, cnt in zip(uni[1:], cnts[1:]):  # exclude zero
        # percentage = cnt / area
        if direction=='forward':
            edges_to_keep.append((f'L{cell_ident}', f'R{u}'))
        else:
            edges_to_keep.append((f'L{u}', f'R{cell_ident}'))


if __name__ == '__main__':
    from main.Database.sql import Database

    Db = Database()
    exp = 'tracking2'
    well = 'D3'

    celldata = Db.get_df_from_query(column_name='experiment', identifier=exp)
    celldata1 = celldata.loc[(celldata.channel == 'GFP_DMD') & (celldata.timepoint == 0)]
    celldata2 = celldata.loc[(celldata.channel == 'GFP_DMD') & (celldata.timepoint == 1)]
    celldata2 = celldata2.drop_duplicates('randomcellid')
    f1 = rf'D:\Images\tracking2\D3\PID20230214_tracking2_T0_0-0_D3_1_GFP_DMD_0.0_0_1.tif'
    f2 = rf'D:\Images\tracking2\D3\PID20230214_tracking2_T1_0-0_D3_1_GFP_DMD_0.0_0_1.tif'
    img1 = imageio.v3.imread(f1)
    img2 = imageio.v3.imread(f2)
    logger.info(celldata.columns)

    DEBUG = False
    SAVEBOOL = True
    start = time.time()
    edges_to_keep = run_voronoi(celldata1, celldata2, img1, img2, DEBUG)
    logger.info(f'elapsed: {time.time() - start}')