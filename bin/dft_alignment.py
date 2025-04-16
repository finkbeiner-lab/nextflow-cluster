#!/opt/conda/bin/python
import argparse
import datetime
import math
import multiprocessing
import os
import pickle
import time
import cv2
import imreg_dft as ird
#from pylibtiff import TIFF
from skimage import transform
#from skimage.feature import register_translation
from skimage.registration import phase_cross_correlation
import utils
from db_util import Ops
from sql import Database
import logging
import imageio

LOG_INFO = {}

logger = logging.getLogger("Alignment")
# logger.propagate = False
now = datetime.datetime.now()
TIMESTAMP = '%d%02d%02d%02d%02d' % (now.year, now.month, now.day, now.hour, now.minute)
print('Timestamp', TIMESTAMP)
fink_log_dir = './finkbeiner_logs'
if not os.path.exists(fink_log_dir):
    os.makedirs(fink_log_dir)
logname = os.path.join(fink_log_dir, f'Alignment-log_{TIMESTAMP}.log')
fh = logging.FileHandler(logname)
# fh.setLevel(logging.DEBUG)
logger.addHandler(fh)
logger.warning('Running Alignment from Database.')




class dft_Alignment:
    def __init__(self, opt):
        self.opt = opt
        self.Db = Database()
        self.analysisdir = self.Db.get_table_analysisdir('experimentdata', dict(experiment=opt.experiment))
        print(self.analysisdir)
        self.montage_folder_name = 'MontagedImages'
        self.montagedir = os.path.join(self.analysisdir, self.montage_folder_name)
        self.alignment_folder_name = 'AlignedImages'
        self.alignmentdir =  os.path.join(self.analysisdir, self.alignment_folder_name)
        self.robo_num = int(args.robo_num)
        self.CHANNEL_SET = set()
        self.NUMBER_OF_PROCESSORS = math.floor(multiprocessing.cpu_count()/2)
        assert os.path.isdir(self.montagedir), 'Confirm the path for input data exists (%s)' % self.montagedir
        os.makedirs(self.alignmentdir, exist_ok=True)
        assert os.path.isdir(self.alignmentdir), 'Confirm the path for output data exists (%s)' % self.alignmentdir
        print(self.opt)
        if self.opt.shift_dict!="null":
            print('Previous calculated shift file path: %s' % self.opt.shift_dict.strip())
            with open(os.path.join(self.alignmentdir, self.opt.shift_dict.strip()), 'rb') as f: 
                unpickler = pickle.Unpickler(f)
                self.shift = unpickler.load()
                print(self.shift)
        else:
            self.shift = {}


    def get_image_tokens_list(self):
        ''' Get image file token list
        Args:
        input_montaged_dir: Input dir. each image file is Montaged time point separated.
        robo_num: Which Robo microscope
        imaging_mode: Confocal or epi

        Time separated image name examples(4 naming types):
        Robo3:
        PID20150217_BioP7asynA_T0_0_A1_1_RFP-DFTrCy5_BG_MONTAGE_ALIGNED_CROPPED.tif
        PID20150904_PGPSTest_T1_8_A7_MONTAGE_RFP-DFTrCy5.tif
        PID20150217_BioP7asynA_T1_12-3_A10_1_RFP-DFTrCy5_MN.tif
        Robo4 epi:
        PID20160706_MerckMitoPlate23_T0_12_A11_1_Cyan_DAPI-FITC_525_1_0.0_ANDORZYLA120XELWD.tif
        Robo4 confocal:
        PID20160706_MerckMitoPlate23_T0_12_A11_1_488_561_Empty_525_1_0.0_AndorZyla120XELWD.tif
        Robo4 latest:
        PID20160706_MerckMitoPlate23_T0_0-0_A11_1_Epi-DAPI_0.0_0_1.0.tif

        '''
        stack_dict = {}

        # use os.walk() to recursively iterate through a directory and all its subdirectories
        # image_paths = [os.path.join(root, name) for root, dirs, files in os.walk(input_montaged_dir) for name in files if name.endswith('.tif')]
        image_paths = ''
        if self.opt.dir_structure == 'root_dir':
            image_paths = [os.path.join(self.montagedir, name) for name in os.listdir(self.montagedir) if name.endswith('.tif') and '_FIDUCIARY_' not in name]
        elif self.opt.dir_structure == 'sub_dir':
            # Only traverse root and immediate subdirectories for images
            relevant_dirs = [self.montagedir] + [os.path.join(self.montagedir, name) for name in os.listdir(self.montagedir) if os.path.isdir(os.path.join(self.montagedir, name))]
            image_paths = [os.path.join(relevant_dir, name) for relevant_dir in relevant_dirs for name in os.listdir(relevant_dir) if name.endswith('.tif') and '_FIDUCIARY_' not in name]
        else:
            raise Exception('Unknown Directory Structure!')

        # Robo3 naming
        # Example: PID20150217_BioP7asynA_T1_12-3_A10_1_RFP-DFTrCy5_MN.tif
        if self.robo_num == 3:
            for image_path in image_paths:
                image_name = os.path.basename(image_path)
                name_tokens = image_name.split('_')
                pid_token = name_tokens[0]
                experiment_name_token = name_tokens[1]
                timepoint_token = name_tokens[2]
                if timepoint_token not in self.opt.chosen_timepoints:
                    continue
                # Check burst
                burst_idx_token = None
                if '-' in name_tokens[3]:
                    numofhours_token, burst_idx_token = name_tokens[3].split('-')
                    numofhours_token = float(numofhours_token)
                    burst_idx_token = int(burst_idx_token)
                else:
                    numofhours_token = int(name_tokens[3])

                    burst_idx_token = None
                well_id_token = name_tokens[4]
                if well_id_token not in self.opt.chosen_wells:
                    continue
                channel_token = name_tokens[6].replace('.tif', '')
                self.CHANNEL_SET.add(channel_token)
                z_idx_token = None

                # Split well id token to make sorting easier
                # Well ID example: H12
                experiment_well_key = (experiment_name_token, well_id_token[0], int(well_id_token[1:]))

                if experiment_well_key in stack_dict:
                    stack_dict[experiment_well_key].append([image_path, pid_token, experiment_name_token, well_id_token, channel_token, timepoint_token, numofhours_token, z_idx_token, burst_idx_token])
                else:
                    stack_dict[experiment_well_key] = [[image_path, pid_token, experiment_name_token, well_id_token, channel_token, timepoint_token, numofhours_token, z_idx_token, burst_idx_token]]
        # Robo4 epi naming
        # Example: PID20160706_MerckMitoPlate23_T0_12_A11_1_Cyan_DAPI-FITC_525_1_0.0_ANDORZYLA120XELWD.tif
        elif self.robo_num == 4 and self.opt.imaging_mode == 'epi':
            for image_path in image_paths:
                image_name = os.path.basename(image_path)
                name_tokens = image_name.split('_')
                pid_token = name_tokens[0]
                experiment_name_token = name_tokens[1]
                timepoint_token = name_tokens[2]
                if timepoint_token not in self.opt.chosen_timepoints:
                    continue
                # Check burst
                burst_idx_token = None
                if '-' in name_tokens[3]:
                    numofhours_token, burst_idx_token = name_tokens[3].split('-')
                    numofhours_token = float(numofhours_token)
                    burst_idx_token = int(burst_idx_token)
                else:
                    numofhours_token = int(name_tokens[3])
                    burst_idx_token = None
                well_id_token = name_tokens[4]
                if well_id_token not in self.opt.chosen_wells:
                    continue
                channel_token = name_tokens[6]
                self.CHANNEL_SET.add(channel_token)
                z_idx_token = int(name_tokens[9])

                # Split well id token to make sorting easier
                # Well ID example: H12
                experiment_well_key = (experiment_name_token, well_id_token[0], int(well_id_token[1:]))

                if experiment_well_key in stack_dict:
                    stack_dict[experiment_well_key].append([image_path, pid_token, experiment_name_token, well_id_token, channel_token, timepoint_token, numofhours_token, z_idx_token, burst_idx_token])
                else:
                    stack_dict[experiment_well_key] = [[image_path, pid_token, experiment_name_token, well_id_token, channel_token, timepoint_token, numofhours_token, z_idx_token, burst_idx_token]]
        # Robo4 confocal naming
        # Example: PID20160706_MerckMitoPlate23_T0_12_A11_1_488_561_Empty_525_1_0.0_AndorZyla120XELWD.tif
        elif self.robo_num == 4 and self.opt.imaging_mode == 'confocal':
            for i in range(len(image_paths)):
                image_path = image_paths[i]
                image_name = os.path.basename(image_path)
                name_tokens = image_name.split('_')
                pid_token = name_tokens[0]
                experiment_name_token = name_tokens[1]
                timepoint_token = name_tokens[2]
                if timepoint_token not in self.opt.chosen_timepoints:
                    continue
                # Check burst
                burst_idx_token = None
                if '-' in name_tokens[3]:
                    numofhours_token, burst_idx_token = name_tokens[3].split('-')
                    numofhours_token = float(numofhours_token)
                    burst_idx_token = int(burst_idx_token)
                else:
                    numofhours_token = int(name_tokens[3])
                    burst_idx_token = None
                well_id_token = name_tokens[4]
                if well_id_token not in self.opt.chosen_wells:
                    continue
                # Find the Z-step marker position
                z_step_pos = None
                if i == 0:
                    for idx, e in reversed(list(enumerate(name_tokens))):
                        if name_tokens[idx].isdigit():
                            continue
                        else:
                            try:
                                float(name_tokens[idx])
                                z_step_pos = idx
                            except ValueError:
                                continue

                channel_token = name_tokens[z_step_pos-2]
                self.CHANNEL_SET.add(channel_token)
                z_idx_token = int(name_tokens[z_step_pos-1])

                # Split well id token to make sorting easier
                # Well ID example: H12
                experiment_well_key = (experiment_name_token, well_id_token[0], int(well_id_token[1:]))

                if experiment_well_key in stack_dict:
                    stack_dict[experiment_well_key].append([image_path, pid_token, experiment_name_token, well_id_token, channel_token, timepoint_token, numofhours_token, z_idx_token, burst_idx_token])
                else:
                    stack_dict[experiment_well_key] = [[image_path, pid_token, experiment_name_token, well_id_token, channel_token, timepoint_token, numofhours_token, z_idx_token, burst_idx_token]]
        # Robo4 latest naming Robo0
        # Example: PID20160706_MerckMitoPlate23_T0_0-0_A11_1_Epi-DAPI_0.0_0_1.0.tif
        elif self.robo_num == 0:
            for image_path in image_paths:
                image_name = os.path.basename(image_path)
                name_tokens = image_name.split('_')
                pid_token = name_tokens[0]
                experiment_name_token = name_tokens[1]
                timepoint_token = name_tokens[2]
                if timepoint_token not in self.opt.chosen_timepoints:
                    continue
                # Check burst
                burst_idx_token = None
                if '-' in name_tokens[3]:
                    numofhours_token, burst_idx_token = name_tokens[3].split('-')
                    numofhours_token = float(numofhours_token)
                    burst_idx_token = int(burst_idx_token)
                else:
                    numofhours_token = int(name_tokens[3])
                    burst_idx_token = None
                well_id_token = name_tokens[4]
                if well_id_token not in self.opt.chosen_wells:
                    continue
                channel_token = name_tokens[6]
                self.CHANNEL_SET.add(channel_token)
                z_idx_token = int(name_tokens[8])

                # Split well id token to make sorting easier
                # Well ID example: H12
                experiment_well_key = (experiment_name_token, well_id_token[0], int(well_id_token[1:]))

                if experiment_well_key in stack_dict:
                    stack_dict[experiment_well_key].append([image_path, pid_token, experiment_name_token, well_id_token, channel_token, timepoint_token, numofhours_token, z_idx_token, burst_idx_token])
                else:
                    stack_dict[experiment_well_key] = [[image_path, pid_token, experiment_name_token, well_id_token, channel_token, timepoint_token, numofhours_token, z_idx_token, burst_idx_token]]
        else:
            raise Exception('Unknowed RoboNumber!')

        return [stack_dict[ewkey] for ewkey in sorted(stack_dict)]




    def register_stack(self, image_stack_experiment_well):
        ''' Worker process for single well
        args:
        image_stack_experiment_well:  a list of time series images tokens for the one experiment-well, including possible multiple channels

        image_stack format example: [[image_path, pid_token, experiment_name_token, well_id_token, channel_token, timepoint_token, numofhours_token, z_idx_token, burst_idx_token], ...]

        '''
        shift_logs = []
        suspicious_misalignment_logs = []
        asymmetric_missing_image_logs = []
        # Dictionary key by channel
        channel_dict = {}
        for tks in image_stack_experiment_well:
            if tks[4] in channel_dict:
                channel_dict[tks[4]].append(tks)
            else:
                channel_dict[tks[4]] = [tks]

        # Dictionary key by timepoint
        for ch in channel_dict:
            timepoint_dict = {}
            tks_list_in_channel = channel_dict[ch]
            for tks in tks_list_in_channel:
                if tks[5] in timepoint_dict:
                    timepoint_dict[tks[5]].append(tks)
                else:
                    timepoint_dict[tks[5]] = [tks]
            # Sort timepoint_dict by z_idx_token and burst_idx_token
            for t in timepoint_dict:
                # int(value or 0) will use 0 in the case when you provide any value that Python considers False, such as None, 0, [], "",
                timepoint_dict[t] = sorted(timepoint_dict[t], key=lambda x: (int(x[7] or 0), int(x[8] or 0)))
            channel_dict[ch] = timepoint_dict



        # Process morphology channel first, then use the calculated shift to apply to other channels
        morphology_timepoint_dict = channel_dict[self.opt.morphology_channel]
        num_of_timepoints = len(morphology_timepoint_dict)

        processing_log = "Processing [experiment: %s, well: %s, channel: %s]" % (image_stack_experiment_well[0][2], image_stack_experiment_well[0][3], self.opt.morphology_channel)
        print(processing_log)
        # current_experiment_well_log.append(processing_log)

        fixed_image = None
        moving_image = None

        # Load previous calculated shifts if exist
        pre_calculated_shift_dict = {}
        if len(self.shift)>0 and (image_stack_experiment_well[0][2], image_stack_experiment_well[0][3]) in self.shift:
            print("shift needs to be applied!")
            pre_calculated_shift_dict = self.shift[(image_stack_experiment_well[0][2], image_stack_experiment_well[0][3])]
            print("pre_calculated_shift_dict", pre_calculated_shift_dict)
            # Sort Tx (e.g. T8) in order and loop
            sorted_morphology_timepoint_keys = sorted(morphology_timepoint_dict, key=lambda x: int(x[1:]))
            # Not loop last item to avoid idx+1 index overflow
            for idx, t in enumerate(sorted_morphology_timepoint_keys[:-1]):
                # For all the z, bursts
                for ix, item in enumerate(morphology_timepoint_dict[t]):
                    # Only calc shift at first image in current timepoint, then propogate to other images
                    if ix == 0:
                        # Write first time point image as fixed image
                        if idx == 0:
                            shift_for_cur_timepoint = [0, 0]
                            # shift_dict[t] = shift_for_cur_timepoint
                            # fixed_image = TIFF.open(item[0], mode='r')
                            # fixed_image = fixed_image.read_image()


                            # Python: cv2.imread(filename[, flags])
                            # <0 Return the loaded image as is (with alpha channel).
                            fixed_image = cv2.imread(item[0], -1)
                            fixed_image_filename = os.path.basename(item[0])
                            if fixed_image is None:
                                raise Exception('%s is corrupted, please run Corrupted Images Detector to find out all the corrupted images.' % (fixed_image_filename))

                            output_img_location = ''
                            if self.opt.dir_structure == 'root_dir':
                                output_img_location = os.path.join(self.alignmentdir, fixed_image_filename.replace('.tif', '_ALIGNED.tif'))
                            elif self.opt.dir_structure == 'sub_dir':
                                output_img_location = utils.reroute_imgpntr_to_wells(os.path.join(self.alignmentdir, fixed_image_filename.replace('.tif', '_ALIGNED.tif')), image_stack_experiment_well[0][3])
                            else:
                                raise Exception('Unknown Directory Structure!')
                            #tif_output = TIFF.open(output_img_location, mode='w')
                            #tif_output.write_image(fixed_image, compression='lzw')
                            # Open the output image in write mode
                            #with imageio.get_writer(output_img_location, format='TIFF', compress_level=9) as tif_output:
                            #    tif_output.append_data(fixed_image)
                            imageio.v3.imwrite(output_img_location, fixed_image, plugin='tifffile', compression='lzw')
                            if os.path.exists(output_img_location):
                                well = image_stack_experiment_well[0][3]
                                experimentdata_id = self.Db.get_table_uuid('experimentdata', dict(experiment=self.opt.experiment))
                                welldata_id = self.Db.get_table_uuid('welldata', dict(experimentdata_id=experimentdata_id, well=well))
                                if self.opt.tiletype == 'maskpath':
                                    print("mask path is updated!")
                                    update_field = 'maskAligned'
                                else:
                                    update_field = 'imageAligned'
                                self.Db.update(
                                    'welldata',
                                    update_dct={update_field: output_img_location},  # Store montage image path in the correct column
                                    kwargs={'id': welldata_id}  # Ensure correct well ID
                                )
                                
                            #del tif_output


                        # moving_image = TIFF.open(morphology_timepoint_dict[sorted_morphology_timepoint_keys[idx+1]][ix][0], mode='r')
                        # moving_image = moving_image.read_image()
                        moving_image = cv2.imread(morphology_timepoint_dict[sorted_morphology_timepoint_keys[idx+1]][ix][0], -1)
                        moving_image_filename = os.path.basename(morphology_timepoint_dict[sorted_morphology_timepoint_keys[idx+1]][ix][0])
                        if moving_image is None:
                            raise Exception('%s is corrupted, please run Corrupted Images Detector to find out all the corrupted images.' % (moving_image_filename))
                        bit_depth = moving_image.dtype
                        # print fixed_image_filename
                        # print moving_image_filename



                        if sorted_morphology_timepoint_keys[idx+1] not in pre_calculated_shift_dict:
                            raise Exception('Previous calculated shifts for %s, %s, %s does not exists' % (image_stack_experiment_well[0][2], image_stack_experiment_well[0][3], sorted_morphology_timepoint_keys[idx+1]))

                        shift_for_cur_timepoint = pre_calculated_shift_dict[sorted_morphology_timepoint_keys[idx+1]]
                        # Shift back to fixed. Note the transform usage is opposite to normal. For example this shift from T1 to T0(fixed target) is [x, y],
                        # parameter in transform.warp should be reversed as [-x, -y]
                        tform = transform.SimilarityTransform(translation=(-shift_for_cur_timepoint[1], -shift_for_cur_timepoint[0]))
                        # With preserve_range=True, the original range of the data will be preserved, even though the output is a float image
                        # with the original pixel value preserved. Otherwise default pixel value is [0, 1] for float
                        corrected_image = transform.warp(moving_image, tform, preserve_range=True)

                        # print "before", type(corrected_image), corrected_image.dtype, corrected_image.shape
                        # print "before", corrected_image.max(), corrected_image.min()

                        # transform.warp default returns double float64 ndarray, have to convert back to original bit depth
                        corrected_image = corrected_image.astype(bit_depth, copy=False)

                        # print "after:", type(corrected_image), corrected_image.dtype, corrected_image.shape
                        # print "after", corrected_image.max(), corrected_image.min()


                        # Output the corrected images to file
                        output_img_location = ''
                        if self.opt.dir_structure == 'root_dir':
                            output_img_location = os.path.join(self.alignmentdir, moving_image_filename.replace('.tif', '_ALIGNED.tif'))
                        elif self.opt.dir_structure == 'sub_dir':
                            output_img_location = utils.reroute_imgpntr_to_wells(os.path.join(self.alignmentdir, moving_image_filename.replace('.tif', '_ALIGNED.tif')), image_stack_experiment_well[0][3])
                        else:
                            raise Exception('Unknown Directory Structure!')
                        #tif_output = TIFF.open(output_img_location, mode='w')
                        #tif_output.write_image(corrected_image, compression='lzw')
                        # Open the output image in write mode
                        #with imageio.get_writer(output_img_location, format='TIFF', compress_level=9) as tif_output:
                        #    tif_output.append_data(corrected_image)
                        imageio.v3.imwrite(output_img_location, corrected_image, plugin='tifffile', compression='lzw')
                        if os.path.exists(output_img_location):
                            well = image_stack_experiment_well[0][3]
                            experimentdata_id = self.Db.get_table_uuid('experimentdata', dict(experiment=self.opt.experiment))
                            welldata_id = self.Db.get_table_uuid('welldata', dict(experimentdata_id=experimentdata_id, well=well))
                            if self.opt.tiletype == 'maskpath':
                                update_field = 'maskAligned'
                            else:
                                update_field = 'imageAligned'
                            self.Db.update(
                                'welldata',
                                update_dct={update_field: output_img_location},  # Store montage image path in the correct column
                                kwargs={'id': welldata_id}  # Ensure correct well ID
                            )
                        
                        #del tif_output

                        # Move to next slice
                        fixed_image = moving_image
                        fixed_image_filename = moving_image_filename
                    # Apply the shift to the other iamges(zs, bursts) in current timepoint
                    else:
                        # other_zb_image = TIFF.open(item[0], mode='r')
                        # other_zb_image = other_zb_image.read_image()
                        other_zb_image = cv2.imread(item[0], -1)
                        other_zb_image_filename = os.path.basename(item[0])
                        if other_zb_image is None:
                            raise Exception('%s is corrupted, please run Corrupted Images Detector to find out all the corrupted images.' % (other_zb_image_filename))
                        if idx != 0:
                            bit_depth =other_zb_image.dtype
                            tform = transform.SimilarityTransform(translation=(-pre_calculated_shift_dict[t][1], -pre_calculated_shift_dict[t][0]))
                            other_zb_image = transform.warp(other_zb_image, tform, preserve_range=True)
                            other_zb_image = other_zb_image.astype(bit_depth, copy=False)

                        output_img_location = ''
                        if self.opt.dir_structure == 'root_dir':
                            output_img_location = os.path.join(self.alignmentdir, other_zb_image_filename.replace('.tif', '_ALIGNED.tif'))
                        elif self.opt.dir_structure == 'sub_dir':
                            output_img_location = utils.reroute_imgpntr_to_wells(os.path.join(self.alignmentdir, other_zb_image_filename.replace('.tif', '_ALIGNED.tif')), image_stack_experiment_well[0][3])
                        else:
                            raise Exception('Unknown Directory Structure!')
                        #tif_output = TIFF.open(output_img_location, mode='w')
                        #tif_output.write_image(other_zb_image, compression='lzw')
                        # Open the output image in write mode
                        #with imageio.get_writer(output_img_location, format='TIFF', compress_level=9) as tif_output:
                        #    tif_output.append_data(other_zb_image)
                        imageio.v3.imwrite(output_img_location, other_zb_image, plugin='tifffile', compression='lzw')
                        if os.path.exists(output_img_location):
                            well = image_stack_experiment_well[0][3]
                            experimentdata_id = self.Db.get_table_uuid('experimentdata', dict(experiment=self.opt.experiment))
                            welldata_id = self.Db.get_table_uuid('welldata', dict(experimentdata_id=experimentdata_id, well=well))
                            if self.opt.tiletype == 'maskpath':
                                update_field = 'maskAligned'
                            else:
                                update_field = 'imageAligned'
                            self.Db.update(
                                'welldata',
                                update_dct={update_field: output_img_location},  # Store montage image path in the correct column
                                kwargs={'id': welldata_id}  # Ensure correct well ID
                            )
                        
                        #del tif_output

            # Reduce memory consumption. Maybe help garbage collection
            fixed_image = None
            moving_image = None

            # Apply the same shift to the other channels(Assuming the Microscope is done with position first imaging method)
            for chl in channel_dict:
                if chl != self.opt.morphology_channel:
                    other_channel_timepoint_dict = channel_dict[chl]
                    other_channel_log = "Applying shift to other channels [experiment: %s, well: %s, channel: %s]" % (image_stack_experiment_well[0][2], image_stack_experiment_well[0][3], chl)
                    print(other_channel_log)
                    # current_experiment_well_log.append(other_channel_log)

                    # Sort Tx (e.g. T8) in order and loop
                    sorted_other_channel_timepoint_keys = sorted(other_channel_timepoint_dict, key=lambda x: int(x[1:]))
                    # No idx+1, so enumerate all timepoints
                    for idx, t in enumerate(sorted_other_channel_timepoint_keys):
                        # Check if current image has related morphology shift calculated, in case of asymmetric images
                        if t in pre_calculated_shift_dict:
                            # For all the z, bursts
                            for ix, item in enumerate(other_channel_timepoint_dict[t]):
                                # other_channel_image = TIFF.open(item[0], mode='r')
                                # other_channel_image = other_channel_image.read_image()
                                other_channel_image = cv2.imread(item[0], -1)
                                other_channel_image_filename = os.path.basename(item[0])
                                if other_channel_image is None:
                                    raise Exception('%s is corrupted, please run Corrupted Images Detector to find out all the corrupted images.' % (other_channel_image_filename))
                                if idx != 0:
                                    bit_depth =other_channel_image.dtype
                                    tform = transform.SimilarityTransform(translation=(-pre_calculated_shift_dict[t][1], -pre_calculated_shift_dict[t][0]))
                                    other_channel_image = transform.warp(other_channel_image, tform, preserve_range=True)
                                    other_channel_image = other_channel_image.astype(bit_depth, copy=False)

                                output_img_location = ''
                                if self.opt.dir_structure == 'root_dir':
                                    output_img_location = os.path.join(self.alignmentdir, other_channel_image_filename.replace('.tif', '_ALIGNED.tif'))
                                elif self.opt.dir_structure == 'sub_dir':
                                    output_img_location = utils.reroute_imgpntr_to_wells(os.path.join(self.alignmentdir, other_channel_image_filename.replace('.tif', '_ALIGNED.tif')), image_stack_experiment_well[0][3])
                                else:
                                    raise Exception('Unknown Directory Structure!')
                                #tif_output = TIFF.open(output_img_location, mode='w')
                                #tif_output.write_image(other_channel_image, compression='lzw')
                                # Open the output image in write mode
                                #with imageio.get_writer(output_img_location, format='TIFF', compress_level=9) as tif_output:
                                #    tif_output.append_data(other_channel_image)
                                imageio.v3.imwrite(output_img_location, other_channel_image, plugin='tifffile', compression='lzw')
                                if os.path.exists(output_img_location):
                                    well = image_stack_experiment_well[0][3]
                                    experimentdata_id = self.Db.get_table_uuid('experimentdata', dict(experiment=self.opt.experiment))
                                    welldata_id = self.Db.get_table_uuid('welldata', dict(experimentdata_id=experimentdata_id, well=well))
                                    if self.opt.tiletype == 'maskpath':
                                        update_field = 'maskAligned'
                                    else:
                                        update_field = 'imageAligned'
                                    self.Db.update(
                                        'welldata',
                                        update_dct={update_field: output_img_location},  # Store montage image path in the correct column
                                        kwargs={'id': welldata_id}  # Ensure correct well ID
                                    )
                                #del tif_output

                        else:
                            print('!!----------- Warning ----------!!')
                            asymmetric_missing_image_log = 'Related morphology channel image does not exist for [experiment: %s, well: %s, channel: %s, timepoint: %s], can not aligned!!\n' %(image_stack_experiment_well[0][2], image_stack_experiment_well[0][3], chl, t)
                            print(asymmetric_missing_image_log)
                            # asymmetric_missing_image_logs.append(asymmetric_missing_image_log)

            # Return dict of current well log
            # current_experiment_well_log.extend(suspicious_misalignments_log)
            #results.append([{(image_stack_experiment_well[0][2], image_stack_experiment_well[0][3][0], int(image_stack_experiment_well[0][3][1:])): (shift_logs, suspicious_misalignment_logs, asymmetric_missing_image_logs)}, {(image_stack_experiment_well[0][2], image_stack_experiment_well[0][3]): pre_calculated_shift_dict}])
            #result_queue.put(results)
            LOG_INFO.update({(image_stack_experiment_well[0][2], image_stack_experiment_well[0][3][0], int(image_stack_experiment_well[0][3][1:])): (shift_logs, suspicious_misalignment_logs, asymmetric_missing_image_logs)})
            self.shift.update({(image_stack_experiment_well[0][2], image_stack_experiment_well[0][3]): pre_calculated_shift_dict})
            return [{(image_stack_experiment_well[0][2], image_stack_experiment_well[0][3][0], int(image_stack_experiment_well[0][3][1:])): (shift_logs, suspicious_misalignment_logs, asymmetric_missing_image_logs)}, {(image_stack_experiment_well[0][2], image_stack_experiment_well[0][3]): pre_calculated_shift_dict}]
        else:
            if len(self.shift)>0:
                raise Exception('Previous calculated shifts for %s, %s does not exists' % (image_stack_experiment_well[0][2], image_stack_experiment_well[0][3]))
            else:
                shift_dict = {}
                # Sort Tx (e.g. T8) in order and loop
                sorted_morphology_timepoint_keys = sorted(morphology_timepoint_dict, key=lambda x: int(x[1:]))
                print("sorted_morphology_timepoint_keys", sorted_morphology_timepoint_keys[:-1])
                # Not loop last item to avoid idx+1 index overflow
                for idx, t in enumerate(sorted_morphology_timepoint_keys[:-1]):
                    # For all the z, bursts
                    print(idx, t, morphology_timepoint_dict[t])
                    for ix, item in enumerate(morphology_timepoint_dict[t]):
                        # Only calc shift at first image in current timepoint, then propogate to other images
                        if ix == 0:
                            # Write first time point image as fixed image
                            if idx == 0:
                                shift_for_cur_timepoint = [0, 0]
                                shift_dict[t] = shift_for_cur_timepoint
                                # fixed_image = TIFF.open(item[0], mode='r')
                                # fixed_image = fixed_image.read_image()


                                # Python: cv2.imread(filename[, flags])
                                # <0 Return the loaded image as is (with alpha channel).
                                fixed_image = cv2.imread(item[0], -1)
                                fixed_image_filename = os.path.basename(item[0])
                                if fixed_image is None:
                                    raise Exception('%s is corrupted, please run Corrupted Images Detector to find out all the corrupted images.' % (fixed_image_filename))

                                output_img_location = ''
                                if self.opt.dir_structure == 'root_dir':
                                    output_img_location = os.path.join(self.alignmentdir, fixed_image_filename.replace('.tif', '_ALIGNED.tif'))
                                elif self.opt.dir_structure == 'sub_dir':
                                    output_img_location = utils.reroute_imgpntr_to_wells(os.path.join(self.alignmentdir, fixed_image_filename.replace('.tif', '_ALIGNED.tif')), image_stack_experiment_well[0][3])
                                    print(output_img_location)
                                else:
                                    raise Exception('Unknown Directory Structure!')
                                #tif_output = TIFF.open(output_img_location, mode='w')
                                #tif_output.write_image(fixed_image, compression='lzw')
                                #with imageio.get_writer(output_img_location, format='TIFF', compress_level=9) as tif_output:
                                #    tif_output.append_data(fixed_image)
                                imageio.v3.imwrite(output_img_location, fixed_image, plugin='tifffile', compression='lzw')
                                if os.path.exists(output_img_location):
                                    well = image_stack_experiment_well[0][3]
                                    experimentdata_id = self.Db.get_table_uuid('experimentdata', dict(experiment=self.opt.experiment))
                                    welldata_id = self.Db.get_table_uuid('welldata', dict(experimentdata_id=experimentdata_id, well=well))
                                    if self.opt.tiletype == 'maskpath':
                                        update_field = 'maskAligned'
                                    else:
                                        update_field = 'imageAligned'
                                    self.Db.update(
                                        'welldata',
                                        update_dct={update_field: output_img_location},  # Store montage image path in the correct column
                                        kwargs={'id': welldata_id}  # Ensure correct well ID
                                    )
                                
                                
                                #del tif_output


                            # moving_image = TIFF.open(morphology_timepoint_dict[sorted_morphology_timepoint_keys[idx+1]][ix][0], mode='r')
                            # moving_image = moving_image.read_image()
                            moving_image = cv2.imread(morphology_timepoint_dict[sorted_morphology_timepoint_keys[idx+1]][ix][0], -1)
                            moving_image_filename = os.path.basename(morphology_timepoint_dict[sorted_morphology_timepoint_keys[idx+1]][ix][0])
                            if moving_image is None:
                                raise Exception('%s is corrupted, please run Corrupted Images Detector to find out all the corrupted images.' % (moving_image_filename))
                            bit_depth = moving_image.dtype
                            # print fixed_image_filename
                            # print moving_image_filename

                            fix_by_dft = False
                            while True:
                                # Calulate shift
                                shift = None

                                if self.opt.alignment_algorithm == 'cross_correlation' or (self.opt.alignment_algorithm == 'cross_correlation_dft_combo' and not fix_by_dft):
                                    # pixel precision. Subpixel precision does not help while pixel prcision misalign, and it increases computation time
                                    shift, error, phasediff = phase_cross_correlation(fixed_image, moving_image)
                                elif self.opt.alignment_algorithm == 'dft' or (self.opt.alignment_algorithm == 'cross_correlation_dft_combo' and fix_by_dft):
                                    # by DFT algorithm
                                    dft_result_dict = ird.translation(fixed_image, moving_image)
                                    shift = dft_result_dict['tvec']
                                    success_number = dft_result_dict['success']
                                else:
                                    raise Exception('error type of ALIGNMENT_ALGORITHM.')


                                # Shift sum for current timepoint to first timepoint (Tx --> T0)
                                shift_for_cur_timepoint = [ y+x for y, x in zip(shift_dict[sorted_morphology_timepoint_keys[idx]], shift)]
                                shift_dict[sorted_morphology_timepoint_keys[idx+1]] = shift_for_cur_timepoint

                                shift_log = ''
                                if self.opt.alignment_algorithm == 'cross_correlation' or (self.opt.alignment_algorithm == 'cross_correlation_dft_combo' and not fix_by_dft):
                                    shift_log = "Detected subpixel offset[%s, %s, %s][%s --> %s: (x: %s, y: %s)] [%s --> %s: (x: %s, y:%s)] [error:%s; phasediff:%s]" %(morphology_timepoint_dict[t][ix][2], morphology_timepoint_dict[t][ix][3], morphology_timepoint_dict[t][ix][4], sorted_morphology_timepoint_keys[idx+1], t, shift[1], shift[0],  sorted_morphology_timepoint_keys[idx+1], sorted_morphology_timepoint_keys[0], shift_for_cur_timepoint[1], shift_for_cur_timepoint[0], error, phasediff)
                                    print(shift_log)
                                    shift_logs.append(shift_log)
                                elif self.opt.alignment_algorithm == 'dft' or (self.opt.alignment_algorithm == 'cross_correlation_dft_combo' and fix_by_dft):
                                    shift_log = "Detected subpixel offset[%s, %s, %s][%s --> %s: (x: %s, y: %s)] [%s --> %s: (x: %s, y:%s)] [success number:%s]" %(morphology_timepoint_dict[t][ix][2], morphology_timepoint_dict[t][ix][3], morphology_timepoint_dict[t][ix][4], sorted_morphology_timepoint_keys[idx+1], t, shift[1], shift[0],  sorted_morphology_timepoint_keys[idx+1], sorted_morphology_timepoint_keys[0], shift_for_cur_timepoint[1], shift_for_cur_timepoint[0], success_number)
                                    if self.opt.alignment_algorithm == 'cross_correlation_dft_combo':
                                        shift_log = '[Switched to DFT]' + shift_log
                                        print(shift_log)
                                        shift_logs[-1] = shift_logs[-1] + '\n' + shift_log
                                        shift_logs.append(shift_log)
                                    else:
                                        print(shift_log)
                                        shift_logs.append(shift_log)

                                else:
                                    print('error type of ALIGNMENT_ALGORITHM.')

                                # If the shift is dramatic, add to suspicious misalignmenet list
                                y_threshold_shift = fixed_image.shape[0]/9
                                x_threshold_shift = fixed_image.shape[1]/9
                                if (abs(shift[0]) >= y_threshold_shift and abs(shift_for_cur_timepoint[0]) >= y_threshold_shift) or (abs(shift[1]) >= x_threshold_shift and abs(shift_for_cur_timepoint[1]) >= x_threshold_shift):
                                    suspicious_misalignment_log = ''
                                    if self.opt.alignment_algorithm == 'cross_correlation' or (self.opt.alignment_algorithm == 'cross_correlation_dft_combo' and not fix_by_dft):
                                        suspicious_misalignment_log = "Suspicious Misalignment: [%s, %s, %s][%s --> %s: (x: %s, y: %s)] [%s --> %s: (x: %s, y:%s)] [error:%s; phasediff:%s]" %(morphology_timepoint_dict[t][ix][2], morphology_timepoint_dict[t][ix][3], morphology_timepoint_dict[t][ix][4], sorted_morphology_timepoint_keys[idx+1], t, shift[1], shift[0],  sorted_morphology_timepoint_keys[idx+1], sorted_morphology_timepoint_keys[0], shift_for_cur_timepoint[1], shift_for_cur_timepoint[0], error, phasediff)
                                        if self.opt.alignment_algorithm == 'cross_correlation_dft_combo':
                                            fix_by_dft = True
                                        print(suspicious_misalignment_log)
                                        suspicious_misalignment_logs.append(suspicious_misalignment_log)

                                    elif self.opt.alignment_algorithm == 'dft' or (self.opt.alignment_algorithm == 'cross_correlation_dft_combo' and fix_by_dft):
                                        suspicious_misalignment_log = "Suspicious Misalignment: [%s, %s, %s][%s --> %s: (x: %s, y: %s)] [%s --> %s: (x: %s, y:%s)] [success number:%s]" %(morphology_timepoint_dict[t][ix][2], morphology_timepoint_dict[t][ix][3], morphology_timepoint_dict[t][ix][4], sorted_morphology_timepoint_keys[idx+1], t, shift[1], shift[0],  sorted_morphology_timepoint_keys[idx+1], sorted_morphology_timepoint_keys[0], shift_for_cur_timepoint[1], shift_for_cur_timepoint[0], success_number)
                                        if self.opt.alignment_algorithm == 'cross_correlation_dft_combo':
                                            fix_by_dft = False
                                            suspicious_misalignment_log = '[Switched to DFT]' + suspicious_misalignment_log
                                            print(suspicious_misalignment_log)
                                            suspicious_misalignment_logs[-1] = suspicious_misalignment_logs[-1] + '\n' + suspicious_misalignment_log
                                        else:
                                            print(suspicious_misalignment_log)
                                            suspicious_misalignment_logs.append(suspicious_misalignment_log)
                                    else:
                                        print('error type of ALIGNMENT_ALGORITHM.')



                                # If Not suspicious misaligned, break the loop
                                else:
                                    break
                                # If already fixed by DFT once, break
                                if not fix_by_dft:
                                    break

                            # Shift back to fixed. Note the transform usage is opposite to normal. For example this shift from T1 to T0(fixed target) is [x, y],
                            # parameter in transform.warp should be reversed as [-x, -y]
                            tform = transform.SimilarityTransform(translation=(-shift_for_cur_timepoint[1], -shift_for_cur_timepoint[0]))
                            # With preserve_range=True, the original range of the data will be preserved, even though the output is a float image
                            # with the original pixel value preserved. Otherwise default pixel value is [0, 1] for float
                            corrected_image = transform.warp(moving_image, tform, preserve_range=True)

                            # print "before", type(corrected_image), corrected_image.dtype, corrected_image.shape
                            # print "before", corrected_image.max(), corrected_image.min()

                            # transform.warp default returns double float64 ndarray, have to convert back to original bit depth
                            corrected_image = corrected_image.astype(bit_depth, copy=False)

                            # print "after:", type(corrected_image), corrected_image.dtype, corrected_image.shape
                            # print "after", corrected_image.max(), corrected_image.min()


                            # Output the corrected images to file
                            output_img_location = ''
                            if self.opt.dir_structure == 'root_dir':
                                output_img_location = os.path.join(self.alignmentdir, moving_image_filename.replace('.tif', '_ALIGNED.tif'))
                            elif self.opt.dir_structure == 'sub_dir':
                                output_img_location = utils.reroute_imgpntr_to_wells(os.path.join(self.alignmentdir, moving_image_filename.replace('.tif', '_ALIGNED.tif')), image_stack_experiment_well[0][3])
                            else:
                                raise Exception('Unknown Directory Structure!')
                            #tif_output = TIFF.open(output_img_location, mode='w')
                            #tif_output.write_image(corrected_image, compression='lzw')
                            # Open the output image in write mode
                            #with imageio.get_writer(output_img_location, format='TIFF', compress_level=9) as tif_output:
                            #    tif_output.append_data(corrected_image)
                            imageio.v3.imwrite(output_img_location, corrected_image, plugin='tifffile', compression='lzw')
                            #del tif_output

                            # Move to next slice
                            fixed_image = moving_image
                            fixed_image_filename = moving_image_filename
                        # Apply the shift to the other iamges(zs, bursts) in current timepoint
                        else:
                            # other_zb_image = TIFF.open(item[0], mode='r')
                            # other_zb_image = other_zb_image.read_image()
                            other_zb_image = cv2.imread(item[0], -1)
                            other_zb_image_filename = os.path.basename(item[0])
                            if other_zb_image is None:
                                raise Exception('%s is corrupted, please run Corrupted Images Detector to find out all the corrupted images.' % (other_zb_image_filename))
                            if idx != 0:
                                bit_depth =other_zb_image.dtype
                                tform = transform.SimilarityTransform(translation=(-shift_dict[t][1], -shift_dict[t][0]))
                                other_zb_image = transform.warp(other_zb_image, tform, preserve_range=True)
                                other_zb_image = other_zb_image.astype(bit_depth, copy=False)

                            output_img_location = ''
                            if self.opt.dir_structure == 'root_dir':
                                output_img_location = os.path.join(self.alignmentdir, other_zb_image_filename.replace('.tif', '_ALIGNED.tif'))
                            elif self.opt.dir_structure == 'sub_dir':
                                output_img_location = utils.reroute_imgpntr_to_wells(os.path.join(self.alignmentdir, other_zb_image_filename.replace('.tif', '_ALIGNED.tif')), image_stack_experiment_well[0][3])
                            else:
                                raise Exception('Unknown Directory Structure!')
                            #tif_output = TIFF.open(output_img_location, mode='w')
                            #tif_output.write_image(other_zb_image, compression='lzw')
                            # Open the output image in write mode
                            #with imageio.get_writer(output_img_location, format='TIFF', compress_level=9) as tif_output:
                            #    tif_output.append_data(other_zb_image)
                            imageio.v3.imwrite(output_img_location, other_zb_image, plugin='tifffile', compression='lzw')
                            if os.path.exists(output_img_location):
                                well = image_stack_experiment_well[0][3]
                                experimentdata_id = self.Db.get_table_uuid('experimentdata', dict(experiment=self.opt.experiment))
                                welldata_id = self.Db.get_table_uuid('welldata', dict(experimentdata_id=experimentdata_id, well=well))
                                if self.opt.tiletype == 'maskpath':
                                    update_field = 'maskAligned'
                                else:
                                    update_field = 'imageAligned'
                                self.Db.update(
                                    'welldata',
                                    update_dct={update_field: output_img_location},  # Store montage image path in the correct column
                                    kwargs={'id': welldata_id}  # Ensure correct well ID
                                )
                            #del tif_output

                # Reduce memory consumption. Maybe help garbage collection
                fixed_image = None
                moving_image = None
                print("channel_dict", channel_dict)
                # Apply the same shift to the other channels(Assuming the Microscope is done with position first imaging method)
                for chl in channel_dict:
                    if chl != self.opt.morphology_channel:
                        other_channel_timepoint_dict = channel_dict[chl]
                        other_channel_log = "Applying shift to other channels [experiment: %s, well: %s, channel: %s]" % (image_stack_experiment_well[0][2], image_stack_experiment_well[0][3], chl)
                        print("other_channel_log",other_channel_log)
                        # current_experiment_well_log.append(other_channel_log)

                        # Sort Tx (e.g. T8) in order and loop
                        sorted_other_channel_timepoint_keys = sorted(other_channel_timepoint_dict, key=lambda x: int(x[1:]))
                        print("sorted_other_channel_timepoint_keys", sorted_other_channel_timepoint_keys)
                        # No idx+1, so enumerate all timepoints
                        for idx, t in enumerate(sorted_other_channel_timepoint_keys):
                            # Check if current image has related morphology shift calculated, in case of asymmetric images
                            if t in shift_dict:
                                # For all the z, bursts
                                for ix, item in enumerate(other_channel_timepoint_dict[t]):
                                    # other_channel_image = TIFF.open(item[0], mode='r')
                                    # other_channel_image = other_channel_image.read_image()
                                    other_channel_image = cv2.imread(item[0], -1)
                                    other_channel_image_filename = os.path.basename(item[0])
                                    if other_channel_image is None:
                                        raise Exception('%s is corrupted, please run Corrupted Images Detector to find out all the corrupted images.' % (other_channel_image_filename))
                                    if idx != 0:
                                        bit_depth =other_channel_image.dtype
                                        tform = transform.SimilarityTransform(translation=(-shift_dict[t][1], -shift_dict[t][0]))
                                        other_channel_image = transform.warp(other_channel_image, tform, preserve_range=True)
                                        other_channel_image = other_channel_image.astype(bit_depth, copy=False)

                                    output_img_location = ''
                                    if self.opt.dir_structure == 'root_dir':
                                        output_img_location = os.path.join(self.alignmentdir, other_channel_image_filename.replace('.tif', '_ALIGNED.tif'))
                                    elif self.opt.dir_structure == 'sub_dir':
                                        output_img_location = utils.reroute_imgpntr_to_wells(os.path.join(self.alignmentdir, other_channel_image_filename.replace('.tif', '_ALIGNED.tif')), image_stack_experiment_well[0][3])
                                    else:
                                        raise Exception('Unknown Directory Structure!')
                                    #tif_output = TIFF.open(output_img_location, mode='w')
                                    #tif_output.write_image(other_channel_image, compression='lzw')
                                    #with imageio.get_writer(output_img_location, format='TIFF', compress_level=9) as tif_output:
                                    #    tif_output.append_data(other_channel_image)
                                    imageio.v3.imwrite(output_img_location, other_channel_image, plugin='tifffile', compression='lzw')
                                    if os.path.exists(output_img_location):
                                        well = image_stack_experiment_well[0][3]
                                        experimentdata_id = self.Db.get_table_uuid('experimentdata', dict(experiment=self.opt.experiment))
                                        welldata_id = self.Db.get_table_uuid('welldata', dict(experimentdata_id=experimentdata_id, well=well))
                                        if self.opt.tiletype == 'maskpath':
                                            update_field = 'maskAligned'
                                        else:
                                            update_field = 'imageAligned'
                                        self.Db.update(
                                            'welldata',
                                            update_dct={update_field: output_img_location},  # Store montage image path in the correct column
                                            kwargs={'id': welldata_id}  # Ensure correct well ID
                                        )
                                    #del tif_output
                            else:
                                print('!!----------- Warning ----------!!')
                                asymmetric_missing_image_log = 'Related morphology channel image does not exist for [experiment: %s, well: %s, channel: %s, timepoint: %s], can not aligned!!\n' %(image_stack_experiment_well[0][2], image_stack_experiment_well[0][3], chl, t)
                                print(asymmetric_missing_image_log)
                                asymmetric_missing_image_logs.append(asymmetric_missing_image_log)

                # Return dict of current well log
                # current_experiment_well_log.extend(suspicious_misalignments_log)
                LOG_INFO.update({(image_stack_experiment_well[0][2], image_stack_experiment_well[0][3][0], int(image_stack_experiment_well[0][3][1:])): (shift_logs, suspicious_misalignment_logs, asymmetric_missing_image_logs)})
                self.shift.update({(image_stack_experiment_well[0][2], image_stack_experiment_well[0][3]): shift_dict})
                return [{(image_stack_experiment_well[0][2], image_stack_experiment_well[0][3][0], int(image_stack_experiment_well[0][3][1:])): (shift_logs, suspicious_misalignment_logs, asymmetric_missing_image_logs)}, {(image_stack_experiment_well[0][2], image_stack_experiment_well[0][3]): shift_dict}]

    def check_mask_present_and_separate(self):
        image_list = []
        mask_list = []
        input_image_stack_list = self.get_image_tokens_list()
        print(len(input_image_stack_list))
        for i in input_image_stack_list[0]:
            print(i)
            if i[0].find("ENCODED")!=-1:
                mask_list.append(i)
            else:
                image_list.append(i)
        if len(mask_list)>0:
            mask_present=True
        else:
            mask_present=False
        return mask_present, image_list, mask_list

        
                
        


    def variety_alignments(self, input_image_stack_list, shift={}):
        #input_image_stack_list = self.get_image_tokens_list()
        #print("input_image_stack_list", input_image_stack_list)
        if len(shift)==0:
            results = []
            # Single instance test
            for i in range(len(input_image_stack_list)):
                results.append(self.register_stack(input_image_stack_list[i]))
            for r in results:
                LOG_INFO.update(r[0])
                self.shift.update(r[1])
            return self.shift
        else:
            self.shift = shift
            self.opt.tiletype="maskpath"
            results = []
            # Single instance test
            for i in range(len(input_image_stack_list)):
                results.append(self.register_stack(input_image_stack_list[i]))
            #for r in results:
            #    LOG_INFO.update(r[0])
            #    self.shift.update(r[1])
            return None
            
        
        
    def save_logs(self):
        # Save dict to file
        # If no previous calculated shifts
        if len(self.shift)>0:
            # Output console log info to file
            with open(os.path.join(self.alignmentdir, '%s_ResultLog.txt' % self.opt.alignment_algorithm), 'wb') as logfile:
                #logfile.write('Alignmet Algorithm %s Result:\n\n' % self.opt.alignment_algorithm)
                log_values = [LOG_INFO[ewkey] for ewkey in sorted(LOG_INFO)]
                print(log_values)

                # Write the shift log order by timepoint
                for idx in range(len(self.opt.chosen_timepoints)-1):
                    for log_tuple in log_values:
                        if log_tuple[0] != [] and idx < len(log_tuple[0]):
                            logfile.write((log_tuple[0][idx]+'\n').encode('utf-8'))
                # Write suspicious misalignment order by well
                for log_tuple in log_values:
                    for suspicious_misalignment in log_tuple[1]:
                        logfile.write( (suspicious_misalignment+'\n').encode('utf-8'))
                # Write asymmetic missing images order by well
                for log_tuple in log_values:
                    for asymmetric_missing_image in log_tuple[2]:
                        logfile.write( (asymmetric_missing_image+'\n').encode('utf-8'))
            # Save the shift to dict file
            print(self.shift)
            with open(os.path.join(self.alignmentdir, 'calculated_shift_%s.dict' % time.strftime('%Y%m%d-%H%M%S')), 'wb') as shiftfile:
                pickle.dump(self.shift, shiftfile)
        

if __name__ == '__main__':
    # --- For Galaxy run ---
    start_time = datetime.datetime.utcnow()
    # Parser
    parser = argparse.ArgumentParser(
        description="Variety Alignments")
    parser.add_argument("--experiment",
        help="--experiment id")
    parser.add_argument("--chosen_wells",
        help="well list")
    parser.add_argument("--chosen_timepoints",
        help="timepoints list")
    parser.add_argument("--morphology_channel",
        help="morphology channel")
    parser.add_argument("--alignment_algorithm",default="cross_correlation_dft_combo",
        help="Algorithm for alignment.- cross_correlation, dft, cross_correlation_dft_combo ")
    parser.add_argument("--robo_num",
        help="robo number", default = None)
    parser.add_argument("--dir_structure",
        help="The image directory -  root_dir, sub_dir", default = 'sub_dir')
    parser.add_argument("--imaging_mode",
        help="IMAGING_MODE - epi, confocal; specification not needed for ROBO 3", default = '')
    parser.add_argument('--tiletype', default='filename', choices=['filename', 'maskpath'], type=str,
        help='Montage image or mask')
    parser.add_argument("--shift_dict",
        help="Previous calculated shifts, please provide the image one for mask", default=None)
    
    args = parser.parse_args()

    # Make sure more than two timepoints to align
    assert len(args.chosen_timepoints) > 1, 'Less than two time points, no need to use alignment module.'
    args.chosen_timepoints =  args.chosen_timepoints.split("-")
    args.chosen_wells =  args.chosen_wells.split("-")
    # Run alignment
    alignment = dft_Alignment(args)
    
    mask_present, image_list, mask_list = alignment.check_mask_present_and_separate()
    
    print("mask_present", mask_present)
    
    print("image_list", len(image_list))
    
    print("mask_list", len(mask_list))
    
    shift = alignment.variety_alignments([image_list], {})
    
    print("shift", shift)
    
    alignment.save_logs()
    
    if mask_present==True:
        _ = alignment.variety_alignments([mask_list], shift)
    
    
    
    # Print Total process time
    end_time = datetime.datetime.utcnow()
    print('Alignment correction run time:', end_time - start_time)
    # Output for user
    print('Check out %s_ResultLog.txt for detail log.' % args.alignment_algorithm)

