#!/opt/conda/bin/python
"""Create folders and initialize parameters based on filenames

Write to
analysis param table
tiledata table
platemap table
imaging template table (optional)

celldata table is filled at segmentation
"""

import os
import sys

print('working dir', os.getcwd())
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

sys.path.append(os.path.dirname(SCRIPT_DIR))

from glob import glob
import pandas as pd
from sql import Database
import pickle
import re
import argparse
import datetime
from glob import glob
import utils
import logging
from template_pycro import TemplateClass
from template_QC import TemplateQC
from ixm2galaxy import ConvertTemplate
from robo2galaxy import RoboConvertTemplate
import uuid
from time import time

logger = logging.getLogger("Folder2Db")
# logger.propagate = False
now = datetime.datetime.now()
TIMESTAMP = '%d%02d%02d%02d%02d' % (now.year, now.month, now.day, now.hour, now.minute)
print('Timestamp', TIMESTAMP)
fink_log_dir = './finkbeiner_logs'
if not os.path.exists(fink_log_dir):
    os.makedirs(fink_log_dir)
logname = os.path.join(fink_log_dir, f'Folder2Db-log_{TIMESTAMP}.log')
fh = logging.FileHandler(logname)
# fh.setLevel(logging.DEBUG)
logger.addHandler(fh)
logger.warning('Registering experiment with database.')

class Intro:
    def __init__(self,):
        pass

    def parse_tokens_to_df(self, filenames, robo_num):
        """
        Creates data frame of all files in directory with tokens parsed into separate columns
        """
        # /PID20220721_AALS-Set12-07192022-GEDI-JAK_T1_0-1_N16_8_FITC_0_1_0.tif
        files_df = pd.DataFrame(filenames, columns=['filepath'])
        files_df['filename'] = files_df['filepath']
        print(files_df.filename.iloc[0])
        logger.warning(f'first file: {files_df.filename.iloc[0]}')
        fname_df = files_df['filepath'].str.split('/', expand=True)
        num_cols = len(fname_df.columns)
        if robo_num == 0 or robo_num == 4:

            files_df[['pid', 'experiment', 'timepoint', 'hours', 'well', 'tile', 'channel', 'burstinterval',
                      'zstep', 'zstep_size']] = fname_df[num_cols-1].str.split('_', expand=True)
            files_df[['hours', 'burstindex']] = files_df['hours'].str.split('-', expand=True)
            files_df['zstep_size'] = files_df['zstep_size'].str.replace('.tif', '')
            files_df['timepoint'] = files_df['timepoint'].str.replace('T', '')
            print(files_df)
            print(files_df['timepoint']=='s8')
            print(files_df['tile']=='s8')
            print(files_df['burstindex']=='s8')
            print(files_df['zstep']=='s8')

            files_df[['timepoint', 'tile', 'burstindex', 'zstep']] = files_df[['timepoint', 'tile', 'burstindex', 'zstep']].astype(int)
            files_df.sort_index(level=['well', 'timepoint', 'channel', 'tile', 'burstindex', 'zstep'], inplace=True)
        elif robo_num == 3:
            files_df[['pid', 'experiment', 'timepoint', 'hours', 'well', 'tile', 'channel']] = fname_df[num_cols-1].str.split('_', expand=True)
            files_df['timepoint'] = files_df['timepoint'].str.replace('T', '')
            files_df[['timepoint', 'tile']] = files_df[['timepoint', 'tile']].astype(int)
            files_df.sort_index(level=['well', 'timepoint', 'tile', 'channel'], inplace=True)
        return files_df

    @staticmethod
    def make_results_folders(input_path, output_path):
        '''Generate folder hierarchy for each output step.'''

        bg_corrected_path = os.path.join(output_path, 'BackgroundCorrected')  # background_removal
        montaged_path = os.path.join(output_path, 'MontagedImages')  # montage
        aligned_path = os.path.join(output_path, 'AlignedImages')  # alignment
        cropped_path = os.path.join(output_path, 'CroppedImages')  # shift_crop
        results = os.path.join(output_path, 'OverlaysTablesResults')  # overlay_tracks and extract_cell_info
        cell_masks = os.path.join(output_path, 'CellMasks')  # segmentation
        qc_path = os.path.join(output_path, 'QualityControl')  # segmentation visualization
        stacking_scratch_path = os.path.join(output_path, 'StackingTemp')

        output_dir_names = ['BackgroundCorrected', 'MontagedImages',
                            'AlignedImages', 'CroppedImages', 'QualityControl',
                            'OverlaysTablesResults', 'CellMasks', 'StackingTemp']
        output_subdirs = [bg_corrected_path, montaged_path,
                          aligned_path, cropped_path, qc_path, results, cell_masks]

        utils.create_folder_hierarchy(output_subdirs, output_path)

    def main(self):
        '''Point of entry.'''

        # Argument parsing
        parser = argparse.ArgumentParser(description="Process cell data.")
        parser.add_argument("--input_path",default='/gladstone/finkbeiner/robodata/Robo4Images/20230920-MsDS-GFP',
                            help="Folder path to input data.")
        parser.add_argument("--output_path", default='/gladstone/finkbeiner/linsley/Shijie/NewGalaxy/GXYTMP-20230920-MsDS-GFP',
                            help="Folder path to ouput results.")
        parser.add_argument("--template_path",
                            help="Path to xlsx template. It should be in ~/Microscope/Templates_sent on the computer you ran the job on.")
        parser.add_argument("--platemap_path", default= '/gladstone/finkbeiner/robodata/Robo4Images/20230920-MsDS-GFP/20230920-MsDS-GFP-platemap.csv', help="Path to csv platemap.")
        parser.add_argument("--ixm_hts_file", help="Path to IXM HTS Template File.")
        parser.add_argument("--robo_file", default='/gladstone/finkbeiner/robodata/Robo4Images/20230920-MsDS-GFP/20230920-MsDS-GFP.csv', help="Path to CSV Template File (Legacy template).")
        parser.add_argument("--illumination_file", default=r'/gladstone/finkbeiner/robodata/IXM Documents/illumination-setting-2023-06-16.ILS',
                            help="Path to IXM Illumination file. On metaxpres -> Control -> Devices -> Configure Illumination -> Backup")
        parser.add_argument("--overwrite_experiment", default=0, choices=[0,1, '0', '1'], help='If 1, overwrite experiment.')
        parser.add_argument("--robo_num", default=0, 
                            type=int,
                            help="Robo number")
        parser.add_argument("--wells_toggle",
                            help="Chose whether to include or exclude specified wells.")
        parser.add_argument("--timepoints_toggle",
                            help="Chose whether to include or exclude specified timepoints.")
        parser.add_argument("--channels_toggle",
                            help="Chose whether to include or exclude specified channels.")
        parser.add_argument("--chosen_wells", "-cw",
                            dest="chosen_wells", default='all',
                            help="Specify wells to include or exclude")
        parser.add_argument("--chosen_timepoints", "-ct",
                            dest="chosen_timepoints", default='',
                            help="Specify timepoints to include or exclude.")
        parser.add_argument("--chosen_channels", "-cc",
                            dest="chosen_channels", default='',
                            help="Specify channels to include or exclude.")
        parser.add_argument(
            '--outfile',
            help='path to save pickle file',
            default=f'/gladstone/finkbeiner/linsley/josh/GALAXY/YD-Transdiff-XDP-Survival1-102822/GXYTMP/tmp_output.txt'
        )
        args = parser.parse_args()
        print('args', args)
        logger.warning(f'args {args}')
        # Set up I/O parameters
        input_path = str.strip(args.input_path)
        output_path = str.strip(args.output_path)
        if not os.path.exists(output_path):
            os.makedirs(output_path)
            logger.warning(f'Created Directory {output_path}')
        assert not (args.ixm_hts_file is not None and len(args.ixm_hts_file) > 1 and args.robo_file is not None and len(args.robo_file) > 1), 'Only IXM template or legacy roboscope template may be entered at one time.'
        if args.ixm_hts_file is not None and len(args.ixm_hts_file) > 1:
            print('IXM Template File: ', args.ixm_hts_file)
            new_template_path = os.path.join(output_path, 'ixm_template.xlsx')
            Conv = ConvertTemplate(new_template_path)
            Conv.convert_template(args.ixm_hts_file, args.illumination_file)
            args.template_path = new_template_path
            logger.warning(f'Converted template from IXM to XLSX.')
            logger.warning(f'Template path set to {args.template_path}')
        
        if args.robo_file is not None and len(args.robo_file) > 1:
            print(f'Robo file (old template): {args.robo_file}')
            new_template_path = os.path.join(output_path, 'robo_template.xlsx')
            Conv = RoboConvertTemplate(new_template_path)
            Conv.convert_template(args.robo_file)
            args.template_path = new_template_path
            logger.info(f'Converted template from CSV to XLSX.')
            logger.info(f'Template path set to {args.template_path}')
  
        # If platemap is given as an arg
        if args.platemap_path is not None:
            platemap_path = str.strip(args.platemap_path)
            mapdf = pd.read_csv(platemap_path)
        else:
            mapdf = None

        if args.template_path is not None:
            template_path = str.strip(args.template_path)
            # TODO: check template, but not micromanager core
            QC = TemplateQC(template_path)
            QC.check_experiment_template()
            QC.check_plate_template()
            # QC.check_timepoint_template()
            QC.check_microscope_template()
            File = TemplateClass(template_path)  # read template
            if File.platemap is not None:
                mapdf = File.platemap
                QC.check_platemap_template()
        else:
            File = None
            
        robo_num = args.robo_num
        outfile = args.outfile

        # Confirm given folders exist
        assert os.path.exists(input_path), 'Confirm that the input folder (%s) exists.' % input_path
        assert 'GXYTMP' in output_path, 'Output folder must contain the string "GXYTMP" (case sensitive)'
        assert re.match('^[a-zA-Z0-9_-]+$', os.path.split(output_path)[1]), 'Confirm that the output folder name (%s) does not contain special characters.' % os.path.split(output_path)[1]
        assert '/gladstone/finkbeiner/' in output_path, 'Output folder must be in the new server'


        filepaths = glob(os.path.join(input_path, '*[0-9]', '*.tif'))
        # TODO: obviously like to avoid relying on filename strings to get experiment info, but in case there's no template or platemap...
        logger.warning(f'first filepath: {filepaths[0]}')
        files_df = self.parse_tokens_to_df(filepaths, robo_num)
        print('files df', files_df.iloc[0])
        # Set up dictionary parameters
        utils.create_dir(output_path)
        self.make_results_folders(input_path, output_path)

        all_files = utils.get_all_files_all_subdir(input_path)
        assert len(all_files) > 0, 'No files to process.'
        start_time = datetime.datetime.utcnow()

        # Handle processing specified wells
        # TODO: filter from file dataframe or add to table
        user_chosen_wells = args.chosen_wells.strip()
        files_df = self.filter_df_by_col(args.wells_toggle, files_df, 'well', user_chosen_wells)
        # Handle processing specified timepoints
        user_chosen_timepoints = args.chosen_timepoints.strip()
        files_df = self.filter_df_by_col(args.timepoints_toggle, files_df, 'timepoint', user_chosen_timepoints)

        user_chosen_channels = args.chosen_channels.strip()
        files_df = self.filter_df_by_col(args.channels_toggle, files_df, 'channel', user_chosen_channels)
        current_experiment = files_df.experiment.iloc[0]
        # Dataframe filtered by user. Now add to database.
        Db = Database()
        if int(args.overwrite_experiment):
            Db.delete_based_on_duplicate_name('experimentdata', dict(experiment=File.current_experiment))
            
        self.write_to_experimentdata(input_path, output_path, File, Db)
        exp_uuid = Db.get_table_uuid('experimentdata', dict(experiment=current_experiment))
        
        logger.warning(f'Experiment uuid: {exp_uuid}')

        wells = pd.unique(files_df.well).tolist()
        if File is not None:
            File.plate = File.plate[File.plate.Well.isin(wells)]
            files_df = files_df[files_df.well.isin(File.plate.Well.tolist())]
            if File.current_experiment != files_df.experiment.iloc[0]:
                raise Exception(f'Template experiment and Filename experiment do not match. \n{File.current_experiment}\n{files_df.experiment.iloc[0]}')
            overlap = File.plate['Overlap'].iloc[0]
        else:
            overlap = 0

        self.write_to_welldata(mapdf, Db, exp_uuid, wells)

        self.write_to_channeldata(File, Db, exp_uuid, wells)

        self.write_to_tiledata_from_files(files_df, overlap, Db, exp_uuid, wells)

        # ----Output for user and save dict----------
        print('Input path:', input_path)
        print('Results output path:', output_path)

        # pickle.dump({'tmp':0}, open('var_dict.p', 'wb'))
        # outfile = os.rename('var_dict.p', outfile)
        # outfile = shutil.move('var_dict.p', outfile)
        timestamp = utils.update_timestring()
        utils.save_user_args_to_csv(args, output_path, 'create_folders' + '_' + timestamp)

        end_time = datetime.datetime.utcnow()
        print('Module run time:', end_time - start_time)
        with open(outfile, 'w') as f:
            f.write(f'Registered {current_experiment} to database.')
        print('Done.')


    def write_to_experimentdata(self, input_path, output_path, File, Db):
        now = datetime.datetime.now()
        analysisdate = f'{now.year}-{now.month:02}-{now.day:02}'
        exp_dct = dict(experiment=File.current_experiment,
                       microscope=File.current_microscope,
                       researcher=File.exp.Author.iloc[0],
                       description=File.exp.Description.iloc[0],
                       project=File.exp.Description.iloc[0],  
                       platetype=File.exp.Plate.iloc[0],
                       wellcount=int(File.exp.WellCount.iloc[0]),
                       analysisdate = analysisdate,
                       imagedir=input_path,
                       analysisdir=output_path
                       )
        # check_exp = dict(experiment=File.current_experiment,
        #                microscope=File.current_microscope,
        #                researcher=File.exp.Author.iloc[0],
        #                description=File.exp.Description.iloc[0],
        #                project=File.exp.Description.iloc[0],  
        #                platetype=File.exp.Plate.iloc[0],
        #                wellcount=int(File.exp.WellCount.iloc[0]),
        #                imagedir=input_path,
        #                analysisdir=output_path
        #                )
        logger.warning(f'exp dct {exp_dct}')
        # Db.delete_based_on_duplicate_name('experimentdata', check_exp)
        Db.add_row('experimentdata', exp_dct)

    def write_to_tiledata_from_files(self, files_df, overlap, Db, exp_uuid, wells):
        tile_dcts = []
        check_tile_dcts = []
        exp = files_df.experiment.iloc[0]
        print('exp_uuid', exp_uuid)
        channel_uuid_dict ={}

        for well in wells:
            strt = time()
            well_uuid = Db.get_table_uuid('welldata', dict(experimentdata_id=exp_uuid, well=well))
            logger.warning(f'Well uuid for tiledata {well_uuid}')
            print(f'Well uuid for tiledata {well_uuid}')
            well_files_df = files_df.loc[files_df.well == well]
            print(f'Filtered df {time() - strt:.2f}')
            for i, row in well_files_df.iterrows():
                logger.warning(f'Tiledata row {row}')
                if (well_uuid, row.channel) in channel_uuid_dict:
                    channel_uuid = channel_uuid_dict[(well_uuid, row.channel)]
                else:
                    channel_uuid = Db.get_table_uuid('channeldata', dict(welldata_id=well_uuid, channel=row.channel))
                    channel_uuid_dict[(well_uuid, row.channel)] = channel_uuid
                # print('channel', row.channel)
                # print(f'Channel uuid for tiledata {channel_uuid}')
                # logger.warning(f'Channel uuid for tiledata {channel_uuid}')
                check_tile_dcts.append(dict(experimentdata_id=exp_uuid,
                                                    welldata_id=well_uuid,
                                                    channeldata_id=channel_uuid,
                                                    tile=int(row.tile),
                                                    timepoint=int(row.timepoint),
                                                    ))
                tile_dcts.append(dict(id=uuid.uuid4(),
                                      experimentdata_id=exp_uuid,
                                      welldata_id=well_uuid,
                                      channeldata_id=channel_uuid,
                                      tile=int(row.tile),
                                      pid=row.pid,
                                      hours=float(row.hours),
                                      timepoint=int(row.timepoint),
                                      overlap=float(overlap),
                                      zstep=int(row.zstep),
                                      zstep_size=float(row.zstep_size),
                                      filename=row.filename,
                                      )
                                 )
            print(f'Looped through filepaths {time() - strt:.2f}')
        
        for tile_dct in check_tile_dcts:
            Db.delete_based_on_duplicate_name('tiledata', tile_dct)
        Db.add_row('tiledata', tile_dcts)

    def write_to_channeldata(self, File, Db, exp_uuid, wells):
        # TODO: check template channels match filename channels
        if File is None:
            return
        channel_dcts = []
        # Get available wavelengths
        epi_wavelengths = File.microscope.AvailableWavelengths.iloc[0].split(';')
        dmd_wavelengths = File.microscope.AvailableWavelengths.iloc[0].split(';')
        confocal_wavelengths = File.microscope.AvailableConfocalWavelengths.iloc[0].split(';') if not pd.isna(File.microscope.AvailableConfocalWavelengths.iloc[0]) else []
        # Filter wells
        template_df = File.plate[File.plate.Well.isin(wells)]
        template_df.DMD_Exposure = template_df.DMD_Exposure.astype(str)
        template_df.Exposure = template_df.Exposure.astype(str)
        template_df.Confocal_Exposure = template_df.Confocal_Exposure.astype(str)
        # Loop through template
        for i, row in template_df.iterrows():
            well_uuid = Db.get_table_uuid('welldata', dict(experimentdata_id=exp_uuid, well=row.Well))
            print('row ord', row['Ordering'])
            logger.warning(f'row ord {row.Ordering}')
            ordering = list(set(row['Ordering'].split(';') if not pd.isna(row['Ordering']) else []))  # taking a set in case there are duplicates

            channels = row['Channel'].split(';')  if not pd.isna(row['Channel']) else []
            intensities = row['ExcitationIntensity'].split(':') if not pd.isna(row['ExcitationIntensity']) else []
            exposures = row['Exposure'].split(';') if not pd.isna(row['Exposure']) else []
            logger.warning(f'channels {channels}')
            print(f'channels {channels}')
            logger.warning(f'epi exposures {exposures}')
            print(f'epi exposures {exposures}')

            dmd_channels = row['DMD_Channel'].split(';') if not pd.isna(row['DMD_Channel']) else []
            dmd_intensities = row['DMD_ExcitationIntensity'].split(':') if not pd.isna(row['DMD_ExcitationIntensity']) else []
            dmd_exposures = row['DMD_Exposure'].split(';') if not pd.isna(row['DMD_Exposure']) else []
            logger.warning(f'dmd channels {dmd_channels}')
            print(f'dmd channels {dmd_channels}')
            logger.warning(f'dmd exposures {dmd_exposures}')
            print(f'dmd exposures {dmd_exposures}')

            confocal_channels = row['Confocal_Channel'].split(';') if not pd.isna(row['Confocal_Channel']) else []
            confocal_intensities = row['Confocal_ExcitationIntensity'].split(':') if not pd.isna(row['Confocal_ExcitationIntensity']) else []
            confocal_exposures = row['Confocal_Exposure'].split(';') if not pd.isna(row['Confocal_Exposure']) else []
            logger.warning(f'confocal channels {confocal_channels}')
            print(f'confocal channels {confocal_channels}')
            logger.warning(f'confocal exposures {confocal_exposures}')
            print(f'confocal exposures {confocal_exposures}')

            cobolt_channels = row['Cobolt_Channel'] if not pd.isna(row['Cobolt_Channel']) else '' # single wavelength
            cobolt_intensities = row['Cobolt_ExcitationIntensity'] if not pd.isna(row['Cobolt_ExcitationIntensity']) else 0
            cobolt_exposures = row['Cobolt_Exposure'] if not pd.isna(row['Cobolt_Exposure']) else 0
            logger.warning(f'cobolt channels {cobolt_channels}')
            print(f'cobolt channels {cobolt_channels}')
            logger.warning(f'cobolt exposures {cobolt_exposures}')
            print(f'cobolt exposures {cobolt_exposures}')

            intensities_dct = {wave.lower(): float(intensity) for wave, intensity in zip(epi_wavelengths, intensities)}
            for wave, intensity in zip(dmd_wavelengths, dmd_intensities):
                intensities_dct[wave.lower()] = max(intensities_dct[wave.lower()], float(intensity)) # TODO: need a new intensity dict for dmd
            for wave, intensity in zip(confocal_wavelengths, confocal_intensities):
                intensities_dct[wave.lower()] = float(intensity)
            intensities_dct['cobolt_intensity'] = float(cobolt_intensities)
            intensities_dct['welldata_id'] = well_uuid
            intensities_dct['experimentdata_id'] = exp_uuid
            for ord in ordering:
                intensities_dct['channel'] = ord
                intensities_dct['objective'] = row['Objective']
                if ord in channels:
                    intensities_dct['exposure'] = exposures[channels.index(ord)]
                elif ord in dmd_channels:
                    intensities_dct['exposure'] = dmd_exposures[dmd_channels.index(ord)]
                elif ord in confocal_channels:
                    intensities_dct['exposure'] = confocal_exposures[confocal_channels.index(ord)]
                elif ord in cobolt_channels:
                    intensities_dct['exposure'] = cobolt_exposures
                else:
                    raise Exception(f'Channel not in ordering: {ord}, {ordering}')
                channel_dcts.append(intensities_dct.copy())

            del intensities_dct
        for channel_dct in channel_dcts:
            print('channel dct', channel_dct)
            Db.delete_based_on_duplicate_name('channeldata', channel_dct)

        Db.add_row('channeldata', channel_dcts)

    def write_to_welldata(self, mapdf, Db, exp_uuid, wells):
        logger.warning('Writing to well data: {wells}')
        well_dcts = []
        dosage_dcts = []

        if mapdf is not None:

            for well in wells:
                print('Well for platemap', well)
                ident = uuid.uuid4()

                df = mapdf[mapdf.well == well]
                celltype = df['celltype'].iloc[0] if not pd.isna(df['celltype'].iloc[0]) else None
                condition = df['condition'].iloc[0] if not pd.isna(df['condition'].iloc[0]) else None
                print('condition', condition)
                well_dcts.append(dict(id = ident, 
                                      experimentdata_id=exp_uuid,
                                      well=well,
                                      celltype=celltype,
                                      condition=condition
                                      ))
                for i, row in df.iterrows():
                    print('row', row)
                    dosage_dcts.append(dict(experimentdata_id=exp_uuid,
                                            welldata_id=ident,
                                            name =row['name'],
                                            dosage=row.dosage,
                                            kind = row.kind))
        else:
            for well in wells:
                well_dcts.append(dict(experimentdata_id=exp_uuid,
                                      well=well))
        for well_dct in well_dcts:
            Db.delete_based_on_duplicate_name('welldata', well_dct)
        print('dosage dcts', dosage_dcts)
        Db.add_row('welldata', well_dcts)
        Db.add_row('dosagedata', dosage_dcts)

    @ staticmethod
    def filter_df_by_col(toggle: str, df: pd.DataFrame, column_name: str, user_chosen_str: str):
        user_chosen_lst = []
        if user_chosen_str != '' and user_chosen_str != 'all':
            user_chosen_lst = utils.get_iter_from_user(user_chosen_str, column_name)
            if toggle == 'exclude':
                df = df[~df[column_name].isin(user_chosen_lst)]
            elif toggle == 'include':
                df = df[df[column_name].isin(user_chosen_lst)]
        uni = pd.unique(df[column_name])
        if not len(df):
            logger.warning(f'Empty df by doing {toggle} for {column_name}s {user_chosen_lst} on dataframe with {uni}')
            raise Exception(f'Empty df by doing {toggle} for {column_name}s {user_chosen_lst} on dataframe with  {uni}')

        logger.warning(f'Selected {column_name} {uni}')
        return df


if __name__ == "__main__":
    I = Intro()
    I.main()
