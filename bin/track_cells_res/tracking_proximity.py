import collections
import datetime
import pickle
import pprint
import shutil

import argparse
import cv2
import numpy as np
import os
import scipy.stats as stat
import utils

from segmentation import filter_contours, find_cells


class Cell(object):
    '''
    A class that makes cells from contours.
    '''

    def __init__(self, cnt, ch_images=None):
        self.cnt = cnt
        self.all_ch_int_stats = None

        if ch_images:
            self.collect_all_ch_intensities(ch_images)

    def __repr__(self):
        return "Cell instance (%s center)" % str(self.get_circle()[0])

    # ----Contours-------------------------------
    def get_circle(self):
        '''Returns centroid of contour.'''
        center, radius = cv2.minEnclosingCircle(self.cnt)
        return center, radius

    def intersects_contour(self, centroid, verbose=False):
        '''
        Determines if a given centroid overlaps with cell's contour.
        '''
        if verbose:
            print("\t\tContour", self.cnt, "Centroid", centroid)

        overlap_value = cv2.pointPolygonTest(
            self.cnt, centroid, False)
        return overlap_value

    def evaluate_overlap(self, circle2):
        '''
        Calculates distance between centroids.
        Evaluates if distance is within 80'%' of both radii.

        @Usage
        circle2 is passed in as (center, radius) tuple
        center is an (x,y) tuple
        '''
        center1, radius1 = self.get_circle()
        center2, radius2 = circle2

        distance = (
                           (center1[0] - center2[0]) ** 2 +
                           (center1[1] - center2[1]) ** 2) ** 0.5

        return distance < (radius1 + radius2) * 0.8

    def evaluate_dist(self, circle2):
        '''
        Calculates distance between centroids.
        Evaluates if distance is smaller than previous.

        @Usage
        circle2 is passed in as (center, radius) tuple
        center is an (x,y) tuple
        '''
        center1, radius1 = self.get_circle()
        center2, radius2 = circle2

        distance = (
                           (center1[0] - center2[0]) ** 2 +
                           (center1[1] - center2[1]) ** 2) ** 0.5

        return distance

    def calculate_cnt_parameters(self):
        '''Extracts all cell-relevant parameters.'''
        cell_params = {}
        area_cnt = cv2.contourArea(self.cnt)
        cell_params['BlobArea'] = area_cnt

        perimeter = cv2.arcLength(self.cnt, True)
        cell_params['BlobPerimeter'] = perimeter

        center, radius = cv2.minEnclosingCircle(self.cnt)
        cell_params['Radius'] = radius

        (x, y), (MA, ma), angle = cv2.fitEllipse(self.cnt)
        cell_params['BlobCentroidX'] = x
        cell_params['BlobCentroidY'] = y

        # reference channel intensity-weighted centroid
        M = cv2.moments(self.cnt)
        if M['m00'] != 0:
            cell_params['BlobCentroidX_RefIntWeighted'] = M['m10'] / M['m00']
            cell_params['BlobCentroidY_RefIntWeighted'] = M['m01'] / M['m00']

        ecc = np.sqrt(1 - ((MA) ** 2 / (ma) ** 2))
        cell_params['BlobCircularity'] = ecc

        hull = cv2.convexHull(self.cnt)
        hull_area = cv2.contourArea(hull)
        solidity = float(area_cnt) / hull_area
        cell_params['Spread'] = solidity

        convexity = cv2.isContourConvex(self.cnt)
        cell_params['Convexity'] = convexity

        return cell_params

    # ----Intensities----------------------------
    def find_cnt_int_dist(self, img):
        '''
        Finds pixels associated with contour.
        Returns intensity parameters.
        This is one of the required parameters to instnatiate a Cell_obj.
        '''

        mask = np.zeros(img.shape[:2], np.uint8)
        cv2.drawContours(mask, [self.cnt], 0, 256, -1)
        cnt_ints = img[np.nonzero(mask)]

        cell_int_stats = {}
        # Intensity params
        cell_int_stats['PixelIntensityMinimum'] = cnt_ints.min()
        cell_int_stats['PixelIntensityMaximum'] = cnt_ints.max()
        cell_int_stats['PixelIntensityMean'] = cnt_ints.mean()
        cell_int_stats['PixelIntensityStdDev'] = cnt_ints.std()
        cell_int_stats['PixelIntensityVariance'] = cnt_ints.var()
        cell_int_stats['PixelIntensityTotal'] = cnt_ints.sum()

        (q1, q5, q10, q25, q50, q75, q90, q95, q99) = np.percentile(
            cnt_ints, [1, 5, 10, 25, 50, 75, 90, 95, 99])
        cell_int_stats['PixelIntensity1Percentile'] = q1
        cell_int_stats['PixelIntensity5Percentile'] = q5
        cell_int_stats['PixelIntensity10Percentile'] = q10
        cell_int_stats['PixelIntensity25Percentile'] = q25
        cell_int_stats['PixelIntensity50Percentile'] = q50
        cell_int_stats['PixelIntensity75Percentile'] = q75
        cell_int_stats['PixelIntensity90Percentile'] = q90
        cell_int_stats['PixelIntensity95Percentile'] = q95
        cell_int_stats['PixelIntensity99Percentile'] = q99
        cell_int_stats['PixelIntensityInterquartileRange'] = q75 - q25

        cell_int_stats['PixelIntensitySkewness'] = stat.skew(cnt_ints)
        cell_int_stats['PixelIntensityKurtosis'] = stat.kurtosis(cnt_ints)

        return cell_int_stats

    def collect_all_ch_intensities(self, ch_images, verbose=False):
        '''
        Takes list of already opened images (per timepoint, well, and frame).
        Calculates intensity statistics based on morphology contour.
        '''

        if verbose:
            print('Get collect_all_ch_intensities:')
            pprint.pprint(ch_images)

        self.all_ch_int_stats = {}
        for color in ch_images.keys():
            # if var_dict['MorphologyChannel'] not in ch_images.keys():
            #     continue
            self.all_ch_int_stats[color] = {}

            # Read image-holding dictionary back in and collect all intensities/image
            for frame, ch_image in ch_images[color].items():
                self.all_ch_int_stats[color][frame] = self.find_cnt_int_dist(ch_image)

        if verbose:
            pprint.pprint(self.all_ch_int_stats)


def sort_cell_info_by_index(time_dictionary, time_list):
    '''
    Takes arrays for each timepoint (key) and sorts on index of tuple.
    '''
    for timepoint in time_list:
        cell_inds = [int(cell[0]) for cell in time_dictionary[timepoint]]
        cell_objs = [cell[1] for cell in time_dictionary[timepoint]]
        inds_and_objs = zip(cell_inds, cell_objs)

        sorted_cell_objs = sorted(inds_and_objs, key=lambda pair: pair[0])
        time_dictionary[timepoint] = sorted_cell_objs

    return time_dictionary


# ----The main tracking function-----------------
def populate_cell_ind_overlap(time_dictionary, time_list, verbose=False):
    '''
    Updates cell_ind from 'n' to value.
    Value is determined from match with previous time point based on overlap.
    Each cell record is ['n', CellObj]
    '''
    print(time_list)
    if verbose:
        print('--time_dictionary before--')
        pprint.pprint(time_dictionary.items())

    assert len(time_dictionary.keys()) > 0, 'No time point data given.'
    first_entry_time = time_list[0]
    # Ordering first entry
    for ind, cell_record in enumerate(time_dictionary[first_entry_time], 1):
        cell_record[0] = ind

    # Numbering the rest
    print('Initial number of cells:', len(time_dictionary[first_entry_time]))

    num_cell = len(time_dictionary[first_entry_time]) + 1


    for time_ind in range(1, len(time_list)):
        t_curr = time_list[time_ind]
        t_prev = time_list[time_ind - 1]

        # Definition: cell_record = (cell_ind, cell_obj)
        for cell_record_c in time_dictionary[t_curr]:
            cell_curr = cell_record_c[1]
            circle_curr = cell_curr.get_circle()
            if verbose:
                print('Current cell id:---------', cell_record_c)

            found = False
            # Sweep previous cells and look for intersection.
            for cell_record_p in time_dictionary[t_prev]:
                cell_prev = cell_record_p[1]
                overlap = cell_prev.evaluate_overlap(circle_curr)

                if overlap == True:
                    found = True
                    cell_record_c[0] = cell_record_p[0]
                    if verbose:
                        print('-----------Found overlap:', cell_record_c)
                    break

            if not found:
                cell_record_c[0] = num_cell
                if verbose:
                    print('----No overlap, new cell:', cell_record_c)
                num_cell += 1

    print('Final number of cells:', num_cell)
    if verbose:
        print('--time_dictionary after--')
        pprint.pprint(time_dictionary.items())

    # Make sure all 'n' were registered.
    for time, cell_records in time_dictionary.items():
        for cell_record in cell_records:
            index = cell_record[0]
            assert index != 'n', cell_records

    return time_dictionary


def populate_cell_ind_closest(time_dictionary, time_list, max_dist=100, verbose=False):
    '''
    Updates cell_ind from 'n' to value.
    Value is determined from match with previous time point based on proximity.
    Each cell record is ['n', CellObj]
    '''

    if verbose:
        print('Maximum distance is:', max_dist)
        print('--time_dictionary before--')
        pprint.pprint(time_dictionary.items())

    assert len(time_dictionary.keys()) > 0, 'No time point data given.'
    first_entry_time = time_list[0]
    # Ordering first entry
    for ind, cell_record in enumerate(time_dictionary[first_entry_time], 1):
        cell_record[0] = ind

    # Numbering the rest
    print('Initial number of cells:', len(time_dictionary[first_entry_time]))

    num_cell = len(time_dictionary[first_entry_time]) + 1

    for time_ind in range(1, len(time_list)):
        t_curr = time_list[time_ind]
        t_prev = time_list[time_ind - 1]

        # Definition: cell_record = (cell_ind, cell_obj)
        for cell_record_c in time_dictionary[t_curr]:
            cell_curr = cell_record_c[1]
            circle_curr = cell_curr.get_circle()

            # Rounded number of pixels on image side
            dist_found = 4000
            # Sweep previous cells and look for intersection.
            for cell_record_p in time_dictionary[t_prev]:
                cell_prev = cell_record_p[1]
                dist_delta = cell_prev.evaluate_dist(circle_curr)
                overlap = dist_delta < dist_found

                if overlap == True:
                    # Update the distance to the cell-cell distance
                    dist_found = dist_delta
                    cell_record_c[0] = cell_record_p[0]

            if dist_found > max_dist:
                cell_record_c[0] = num_cell
                num_cell += 1

    print('Final number of cells:', num_cell)
    if verbose:
        print('--time_dictionary after--')
        pprint.pprint(time_dictionary.items())

    # Make sure all 'n' were registered.
    for time, cell_records in time_dictionary.items():
        for cell_record in cell_records:
            index = cell_record[0]
            assert index != 'n', cell_records

    return time_dictionary


# ----Handling dictionary structure--------------
def time_all_cell_dict(well_filelist, time_list, resolution, var_dict):
    time_dictionary = collections.OrderedDict()
    for time_id in time_list:
        img_pointer = [string for string in well_filelist if time_id + '_' in string]
        img = cv2.imread(img_pointer[0], resolution)
        time_id_cell_dict(img, time_id, time_dictionary, var_dict)

    return time_dictionary


def time_id_cell_dict(img, time_id, time_dictionary, var_dict):
    '''
    Finds contours and filters for cells.
    Then adds cells to time_dictionary.
    '''
    kept_contours = find_cells(img, img_is_mask=True)
    kept_contours = filter_contours(kept_contours,
                                    small=var_dict["MinCellSize"], large=var_dict["MaxCellSize"])

    time_dictionary[time_id] = []
    for cnt in kept_contours:
        time_dictionary[time_id].append(
            ['n', Cell(cnt)])

    return time_dictionary


# ----Encoding dictionary structure--------------
def make_encoded_mask(sorted_time_dict, well_filelist, time_list, write_path, resolution):
    '''
    Draw kept cells onto image with intensity value corresponding to cell number.
    '''
    # Should be 16-bit encode > 256 objects
    d_type = np.uint16

    for time_id in time_list:
        img_pointer = [string for string in well_filelist if time_id + '_' in string]
        img_pointer = img_pointer[0]

        if 'MASK.tif' not in img_pointer and 'MASKS.tif' not in img_pointer:
            continue

        mask_shape = cv2.imread(img_pointer, resolution).shape[0:2]
        mask = np.zeros(mask_shape, dtype=d_type)
        orig_name = utils.extract_file_name(img_pointer)
        img_name = utils.make_file_name(write_path, orig_name + '_ENCODED')
        well_name = os.path.basename(img_pointer).split('_')[4]
        img_name = utils.reroute_imgpntr_to_wells(img_name, well_name)
        time_name = os.path.basename(img_pointer).split('_')[2]
        assert time_id == time_name, 'Error: Dictionary timepoint and file not matching.'

        # Loop through Cells in dictionary and encode into mask
        for cnt_ind, cell_obj in sorted_time_dict[time_id]:
            cv2.drawContours(mask, [cell_obj.cnt], 0, cnt_ind, -1)
            cv2.drawContours(mask, [cell_obj.cnt], 0, cnt_ind, 5)

        cv2.imwrite(img_name, mask)
        # cv2.imwrite(img_pointer, mask)


def tracking(var_dict, path_to_masks, write_path):
    '''
    Main point of entry.
    '''
    resolution = 0

    morph_channel = var_dict["MorphologyChannel"]
    var_dict['TrackedCells'] = {}
    for well in var_dict['Wells']:
        selector = utils.make_selector(well=well, channel=morph_channel)
        # well_filelist = utils.make_filelist(path_to_masks, selector)
        well_filelist = utils.make_filelist_wells(path_to_masks, selector)
        well_filelist = [imp for imp in well_filelist if 'ENCODED' not in imp]
        if len(well_filelist) == 0:
            # print 'No files associated with morphology channel.'
            print('Confirm that CellMasks folder contains files.')
            continue

        time_list = utils.get_timepoints(well_filelist)
        print('Time points that have a morphology image:')
        print('Well', well, time_list)
        time_dictionary = time_all_cell_dict(
            well_filelist, time_list, resolution, var_dict)
        # pprint.pprint(time_dictionary.items())
        if var_dict["TrackType"] == 'overlap':
            time_dictionary = populate_cell_ind_overlap(
                time_dictionary, time_list)
        elif var_dict["TrackType"] == 'proximity':
            time_dictionary = populate_cell_ind_closest(
                time_dictionary, time_list, max_dist=int(var_dict["MaxDistance"]))
        else:
            track_type = var_dict["TrackType"]
            assert track_type == 'overlap' or track_type == 'proximity', 'No track type given.'
        sorted_time_dict = sort_cell_info_by_index(time_dictionary, time_list)
        # pprint.pprint(time_dictionary.items())
        var_dict['TrackedCells'][well] = sorted_time_dict
        # print 'sorted_time_dict'
        # pprint.pprint(sorted_time_dict.items())
        make_encoded_mask(
            sorted_time_dict, well_filelist, time_list, write_path, resolution)

    # For select_analysis_module input, set var_dict['OutputPath']
    var_dict["OutputPath"] = write_path


if __name__ == '__main__':

    # ----Parser-----------------------
    parser = argparse.ArgumentParser(
        description="Track cells from cell masks.")
    parser.add_argument("input_dict",
                        help="Load input variable dictionary")
    parser.add_argument("--input_image_path",
                        help="Folder path to input data.", default='')
    parser.add_argument("track_type",
                        help="Overlap or proximity.")
    parser.add_argument("output_dict",
                        help="Write variable dictionary.")
    parser.add_argument("--min_cell",
                        dest="min_cell", type=int,
                        help="Minimum feature size considered as cell.")
    parser.add_argument("--max_cell",
                        dest="max_cell", type=int,
                        help="Maximum feature size considered as cell.")
    parser.add_argument("--max_dist",
                        dest="max_dist", type=int,
                        help="Maximum distance a cell can travel between time points in pixels.")
    args = parser.parse_args()

    # ----Load path dict-------------------------
    infile = args.input_dict
    var_dict = pickle.load(open(infile, 'rb'))
    var_dict["MaxDistance"] = args.max_dist
    var_dict["TrackType"] = args.track_type
    print('var_dict["TrackType"]', var_dict["TrackType"])

    # Test if min max values for cell size were previously set
    try:
        min_cell = var_dict["MinCellSize"]
        max_cell = var_dict["MaxCellSize"]
    except KeyError:
        print("Using updated min/max size object parameters.")
        try:
            var_dict["MinCellSize"] = int(args.min_cell)
            var_dict["MaxCellSize"] = int(args.max_cell)
            assert var_dict["MinCellSize"] < var_dict[
                "MaxCellSize"], 'Minimum size should be smaller than maximum size.'
        except TypeError:
            assert type(args.min_cell) == int, 'Please provide numerical minimum value.'
            assert type(args.max_cell) == int, 'Please provide numerical maximum value.'

    print("Minimum object size set to:", var_dict["MinCellSize"])
    print("Maximum object size set to:", var_dict["MaxCellSize"])

    # ----Initialize parameters------------------
    path_to_masks = utils.get_path(args.input_image_path, var_dict['GalaxyOutputPath'], 'CellMasks')
    print('Input path: %s' % path_to_masks)
    write_path = path_to_masks
    outfile = args.output_dict
    # resolution = 0#var_dict['Resolution']

    # ----Confirm given folders exist--
    assert os.path.exists(path_to_masks), 'Confirm the path for data exists (%s)' % path_to_masks

    # ----Run tracking---------------------------
    start_time = datetime.datetime.utcnow()

    tracking(var_dict, path_to_masks, write_path)

    end_time = datetime.datetime.utcnow()
    print('Tracking run time:', end_time - start_time)

    # ----Output for user and save dict----------
    print('Cells were tracked for each time point.')
    print('Output from this step is an encoded mask written to:')
    print(write_path)
    pickle.dump(var_dict, open('var_dict.p', 'wb'))
    # outfile = os.rename('var_dict.p', outfile)
    outfile = shutil.move('var_dict.p', outfile)
    timestamp = utils.update_timestring()
    utils.save_user_args_to_csv(args, write_path, 'tracking' + '_' + timestamp)