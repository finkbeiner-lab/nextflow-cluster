import datetime
import os
import pandas as pd
import logging
from microscope_config import Configure

__author__ = "Josh Lamstein"
__copyright__ = "Copyright 2023"

logger = logging.getLogger("TM")


class TemplateClass:
    def __init__(self, template):
        """Track current filename of image
        Ex. PID20220819_20220817-eos-gfp_T0_0-0_A3_6_Transmission-GFP_0.0_0_1"""
        self.template = template
        self.plate = None
        self.exp = None
        self.timepoint = None
        self.microscope = None
        self.platemap = None
        self.current_well = None
        self.current_tp = None
        self.current_hour = '0-0'  # todo: calculate hour imaged
        self.current_montage_image = 1  # todo: set this from run main
        self.current_channel = None
        self.current_objective = None
        now = datetime.datetime.now()
        self.timestamp = '%d%02d%02d' % (now.year, now.month, now.day)
        self.current_pid = f'PID{self.timestamp}'
        self.time_imaged = None

        self.parent_dir = None
        self.experiment_directory = None
        self.well_spacing = None
        self.current_experiment = None
        self.current_microscope = None
        self.light_source_has_variable_intensity = None
        self.use_epi = None
        self.use_confocal = None
        self.use_dmd = None
        self.use_database = None
        self.use_tracking = None
        self.use_cellpose = None
        self.rowsign = None
        self.colsign = None
        self.fid_stepsize = None
        self.arm_dll_file = None
        self.arm_param_file = None
        self.arm_teach_file = None
        self.arm_seq_file = None
        self.x_arm_coord = None
        self.y_arm_coord = None
        self.stage_has_lock = None
        self.stim_channel = None
        self.stim_turretfilter = None
        self.use_fiducial=None
        self.fid_zheight = None
        self.fiducial_dir = None

        self.EpiLightSourceName = None
        self.LightSourceHasVariableIntensity = None
        self.ConfocalLightSourceName = None
        self.ConfocalLightSourceNameSecondary = None
        self.ConfocalChannelNameInConfig = None
        self.DMDChannelNameInConfig = None
        self.ConfocalChannelNameInConfigSecondary = None
        self.ZDriveNameInConfig = None
        self.ObjectiveDeviceNameInConfig = None
        self.ObjectiveGroupNameInConfig = None
        self.PFSDeviceNameInConfig = None
        self.pfs_prop_name = None
        self.spreadsheet_tab = None

        self.read_template()

    def construct_filename(self):
        # well = self.current_well if self.current_well is not None else 'current-position' # for CURRENT_POSITION debugging
        f = f'{self.current_pid}_{self.current_experiment}_T{self.current_tp}_' \
            f'{self.current_hour}_{self.current_well}_{self.current_montage_image}_' \
            f'{self.current_channel}_0.0_0_1.tif'
        well_dir = os.path.join(self.parent_dir, self.current_experiment, self.current_well)
        if not os.path.exists(well_dir):
            os.makedirs(well_dir)
        savename = os.path.join(well_dir, f)
        return savename

    def construct_fiducial_dir(self):
        fid_dir = os.path.join(self.parent_dir, self.current_experiment, 'Fiducials')
        return fid_dir

    def construct_mask_filename(self):
        f = f'{self.current_pid}_{self.current_experiment}_T{self.current_tp}_' \
            f'{self.current_hour}_{self.current_well}_{self.current_montage_image}_' \
            f'{self.current_channel}_0.0_0_1_'
        mask_dir = os.path.join(self.parent_dir, self.current_experiment, self.current_well, 'MASK')
        if not os.path.exists(mask_dir):
            os.makedirs(mask_dir)
        savename = os.path.join(mask_dir, f + 'MASK.tif')
        savename_relabelled = os.path.join(mask_dir, f + 'MASKTRACKED.tif')
        return savename, savename_relabelled

    def update_well(self, well):
        self.current_well = well
        logger.info(f'File: current well: {self.current_well}')

    def update_tp(self, tp):
        self.current_tp = tp
        logger.info(f'File: current timepoint: {self.current_tp}')

    def update_channel(self, channel):
        self.current_channel = channel
        logger.info(f'File: current channel: {self.current_channel}')

    def update_tracking_channel(self, tracking_channel):
        self.current_tracking_channel = tracking_channel
        logger.info(f'File: current tracking channel: {self.current_tracking_channel}')

    def update_montage_image(self, cnt):
        self.current_montage_image = cnt
        logger.info(f'File: current montage image (montage pos): {self.current_montage_image}')

    def update_objective(self, objective):
        self.current_objective = objective
        logger.info(f'File: current objective {self.current_objective}')

    def update_time_imaged(self):
        now = datetime.datetime.now()
        self.time_imaged = '%d%02d%02d%02d%02d%02d' % (now.year, now.month, now.day, now.hour, now.minute, now.second)
        logger.info(f'File: time_imaged {self.time_imaged}')

    def update_stim(self, stim_channel, stim_filter):
        self.stim_channel = stim_channel
        self.stim_turretfilter = stim_filter
        logger.info(f'File: stim channel for well {self.stim_channel}')
        logger.info(f'File: stim filter for well {self.stim_turretfilter}')

    def read_template(self):
        if self.template is not None:
            self.exp = pd.read_excel(self.template, sheet_name='experiment')
            self.plate = pd.read_excel(self.template, sheet_name='plate')
            self.timepoint = pd.read_excel(self.template, sheet_name='timepoint')
            self.microscope = pd.read_excel(self.template, sheet_name='microscope')
            try:
                self.platemap = pd.read_excel(self.template, sheet_name='platemap')
            except:
                print('Warning: In excel template, platemap sheet does not exist.')

            self.parent_dir = self.exp.ImageFolder.iloc[0]
            self.current_experiment = self.exp.ExperimentName.iloc[0]
            self.experiment_directory = os.path.join(self.parent_dir, self.current_experiment)
            num_wells = int(self.exp.WellCount.iloc[0])
            self.current_microscope = self.microscope.Microscope.iloc[0]
            self.use_epi = self.microscope.UseEpi.iloc[0]
            self.use_confocal = self.microscope.UseConfocal.iloc[0]
            self.use_dmd = self.microscope.UseDMD.iloc[0]
            self.use_database = self.microscope.UseDatabase.iloc[0]
            self.use_tracking = self.microscope.UseTracking.iloc[0]  # todo: turn voronoi on/off
            self.use_cellpose = self.microscope.UseCellpose.iloc[0]
            self.use_fiducial = self.microscope.UseFiducial.iloc[0]
            self.fid_stepsize = self.microscope.Fid_Stepsize.iloc[0]
            self.track_puncta = self.microscope.TrackPuncta.iloc[0]
            c = Configure(self.current_microscope, num_wells)  # load microscope specific params
            self.rowsign = c.rowsign
            self.colsign = c.colsign
            self.x_arm_coord = c.x_arm_coord
            self.y_arm_coord = c.y_arm_coord
            self.arm_dll_file = c.arm_dll_file
            self.arm_param_file = c.arm_param_file
            self.arm_teach_file = c.arm_teach_file
            self.arm_seq_file = c.arm_seq_file
            self.stage_has_lock = c.stage_has_lock
            self.fid_zheight = c.fid_zheight
            self.well_spacing = c.well_spacing
            self.spreadsheet_tab = c.spreadsheet_tab

            self.EpiLightSourceName = c.EpiLightSourceName
            self.LightSourceHasVariableIntensity = c.LightSourceHasVariableIntensity
            self.ConfocalLightSourceName = c.ConfocalLightSourceName
            self.ConfocalLightSourceNameSecondary = c.ConfocalLightSourceNameSecondary
            self.ConfocalChannelNameInConfig = c.ConfocalChannelNameInConfig
            self.DMDChannelNameInConfig = c.DMDChannelNameInConfig
            self.ConfocalChannelNameInConfigSecondary = c.ConfocalChannelNameInConfigSecondary
            self.ZDriveNameInConfig = c.ZDriveNameInConfig
            self.ObjectiveDeviceNameInConfig = c.ObjectiveDeviceNameInConfig
            self.ObjectiveGroupNameInConfig = c.ObjectiveGroupNameInConfig
            self.PFSDeviceNameInConfig = c.PFSDeviceNameInConfig
            self.pfs_prop_name = c.pfs_prop_name



            self.fiducial_dir = self.construct_fiducial_dir()
        logger.info(f"File: Parent Directory: {self.parent_dir}")
        logger.info(f"File: Current Experiment: {self.current_experiment}")
        logger.info(f"File: Current Microscope: {self.current_microscope}")
        logger.info(f"File: Use Epi: {self.use_epi}")
        logger.info(f"File: Use DMD: {self.use_dmd}")
        logger.info(f"File: Use Database: {self.use_database}")
        logger.info(f"File: Use Tracking: {self.use_tracking}")
        logger.info(f"File: Light source has variable intensity: {self.light_source_has_variable_intensity}")
        return self.exp, self.plate, self.timepoint