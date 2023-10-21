"""Templates include
Plate: channel and well info, montage pattern, pfs height, excitation intensity, dmd intensity, overlap
Timepoint: timepoints, duration, date, current time
Experiment: name, platetype, microscope, save folder, incubator, incubator location"""

import pandas as pd
import numpy as np
import logging
from string import ascii_uppercase
import datetime
from microscope_config import Configure

logger = logging.getLogger("TM")
# todo: check template for stim_dmd and stim_filterturret
# todo: add a check for using the correct wavelengths per microscope

class TemplateQC:

    def __init__(self,template):
        self.template = template
        _df_experiment = pd.read_excel(self.template, sheet_name='experiment')
        _df_microscope = pd.read_excel(self.template, sheet_name='microscope')
        assert _df_microscope.Microscope.iloc[0] in ['TM', 'IXM', 'ROBO3', 'ROBO4', 'ROBO5', 'ROBO6']
        assert _df_experiment.WellCount.iloc[0] in [96, 384]
        self.c = Configure(microscope=_df_microscope.Microscope.iloc[0],
                           num_wells=_df_experiment.WellCount.iloc[0])

    def check_microscope_template(self):
        df = pd.read_excel(self.template, sheet_name='microscope')
        assert df.Microscope.iloc[0] in ['TM', 'IXM', 'ROBO3', 'ROBO4', 'ROBO5', 'ROBO6']
        assert ~df.AvailableWavelengths.str.contains(':').any()
        if df.UseEpi.iloc[0]:
            assert ~pd.isna(df.AvailableWavelengths.iloc[
                                0]), f'LightSourceHasVariableIntensity is true, but no AvailableWavelengths is None.'
        incubator_com_port = df.IncubatorCOMPort.iloc[0]
        if not pd.isna(incubator_com_port):
            assert 'COM' in incubator_com_port, f'IncubatorCOMPort does not have COM in string: {incubator_com_port}'
            assert df.Shelf.iloc[0] >= 0
            assert df.Stack.iloc[0] >= 0
        a1_coordinate = (df.A1_x.iloc[0], df.A1_y.iloc[0])
        assert isinstance(a1_coordinate[0], (int, float, np.int64))
        assert isinstance(a1_coordinate[1], (int, float, np.int64))
        assert isinstance(df.Fid_Stepsize.iloc[0], (int, float, np.int64))
        assert df.UseEpi.iloc[0] in [False, True], f'UseEpi {df.UseEpi.iloc[0]} should be True or False'
        assert df.UseDMD.iloc[0] in [False, True], f'UseDMD {df.UseDMD.iloc[0]} should be True or False'
        assert df.UseFiducial.iloc[0] in [False, True], f'UseFiducial {df.UseFiducial.iloc[0]} should be True or False'
        assert df.UseDatabase.iloc[0] in [False, True], f'UseDatabase {df.UseDatabase.iloc[0]} should be True or False'
        assert df.UseTracking.iloc[0] in [False, True], f'UseTracking {df.UseTracking.iloc[0]} should be True or False'
        assert df.TrackPuncta.iloc[0] in [False, True], f'UseTracking {df.UseTracking.iloc[0]} should be True or False'
        if df.UseTracking.iloc[0]:
            assert df.UseDatabase.iloc[
                       0] == True, f'UseDatabase {df.UseDatabase.iloc[0]} should be True is UseTracking is True.'
        if df.TrackPuncta.iloc[0]:
            assert df.UseTracking.iloc[
                       0] == True, f'UseTracking {df.UseTracking.iloc[0]} should be True is TrackPuncta is True.'

    def check_timepoint_template(self):
        df = pd.read_excel(self.template, sheet_name='timepoint')
        now = datetime.datetime.now()
        # assert df.Timepoint.iloc[0] == 0
        assert ~pd.isna(df.Date.iloc[0])
        # todo: QC for time, instance and reasonable values
        timepoints = [-1]
        for i, row in df.iterrows():
            _date = row.Date
            assert _date.year == now.year
            _time = row.Time
            tp = row.Timepoint
            assert tp not in timepoints, 'timepoints must not repeat'
            assert np.max(timepoints) < tp, f'timepoint {tp} must be greater than the previous timepoints'
            timepoints.append(tp)
            hour = row.Hour
            # estimatedDuration = row.estimatedDuration

    def check_platemap_template(self):
        df = pd.read_excel(self.template, sheet_name='platemap')
        well_lst = df.well.tolist()
        plate_df = pd.read_excel(self.template, sheet_name='plate')
        for well in plate_df.Well:
            assert well in well_lst, f'Well not in platemap {well}'


    def check_experiment_template(self):
        df = pd.read_excel(self.template, sheet_name='experiment')
        assert ~pd.isna(df.ExperimentName.iloc[0])
        assert '_' not in df.ExperimentName.iloc[0]
        assert ~pd.isna(df.ImageFolder.iloc[0])
        assert df.WellCount.iloc[0] in [96, 384]
        assert df.ImagingPattern.iloc[0] in ['epi_only', 'epi_dmd_per_well', 'dmd_only', 'epi_and_confocal']
        assert ~pd.isna(df.Plate.iloc[0])

    def check_plate_template(self):
        df = pd.read_excel(self.template, sheet_name='plate')
        assert ~pd.isna(df.Channel.iloc[0])
        assert 100 > df.Montage.iloc[0] >= 0

        possible_channels = []
        orders = []
        wells = self.generate_wells() + ['CURRENT_POSITION']
        # Orders and possible channels must have same channels

        well_lst = df.Well.to_list()
        _, well_cnts=np.unique(well_lst, return_counts=True)
        if np.any(well_cnts > 1): raise Exception('Well is duplicated in template.')

        for i, row in df.iterrows():

            if not pd.isna(row.Exposure):
                if isinstance(row.Exposure, str):
                    assert ':' not in row.Exposure
                    exposure_vals = row.Exposure.split(';')
                    for exposure in exposure_vals:
                        assert exposure.isnumeric(), f'exposure {exposure} is not numeric'
                else:
                    assert 0 < row.Exposure < 20000, f'check Exposure in template, {row.Exposure}'

            if isinstance(row.Show_Image, str):
                assert ':' not in row.Show_Image
                show_vals = row.Show_Image.split(';')
                for sho in show_vals:
                    assert sho in [0, 1, '0', '1'], f'show image should be 0 or 1, {sho}'
            else:
                assert row.Show_Image == 0 or row.Show_Image == 1 or pd.isna(row.Show_Image), f'check Show_Image in template, {row.Show_Image}'

            if isinstance(row.DMD_Generate, str):
                assert ':' not in row.DMD_Generate
                dmd_gens = row.DMD_Generate.split(';')
                for dmd_gen in dmd_gens:
                    assert dmd_gen in [0, 1, '0', '1'], f'DMD_Generate should be 0 or 1, {dmd_gen}'
            else:
                assert row.DMD_Generate == 0 or row.DMD_Generate == 1 or pd.isna(
                    row.DMD_Generate), f'check DMD_Generate in template, {row.DMD_Generate}'

            assert row.Well in wells, f'{row.Well} not in {wells}'

            if row.Well == 'CURRENT_POSITION':
                assert row.Montage == 1, 'montage must equal 1 for imaging current position'
            row_orders = row.Ordering.split(';') if not pd.isna(row.Ordering) else []

            if not pd.isna(row.DMD_Function):
                dmd_functions = row.DMD_Function.split(';') if not pd.isna(row.DMD_Function) else ''
                for dmd_function in dmd_functions:
                    assert dmd_function in ['all_on', 'half_on', 'corners_on',
                                            'checkerboard'], f'dmd_function is {dmd_function}'
            else:
                dmd_functions = [''] * len(row_orders)

            possible_channels += row.DMD_Channel.split(';') if not pd.isna(row.DMD_Channel) else ''

            if not pd.isna(row.Channel):
                possible_channels += row.Channel.split(';') if not pd.isna(row.Channel) else ''
            if not pd.isna(row.Confocal_Channel):
                possible_channels += row.Confocal_Channel.split(';') if not pd.isna(row.Confocal_Channel) else ''
            if not pd.isna(row.Cobolt_Channel):
                possible_channels += row.Cobolt_Channel.split(';') if not pd.isna(row.Cobolt_Channel) else ''

            if not pd.isna(row.Show_Image):
                if isinstance(row.Show_Image, str):
                    assert len(row.Show_Image.split(';')) == len(row.Ordering.split(';'))
                else:
                    assert len(row.Ordering.split(';')) == 1

            if not pd.isna(row.DMD_Generate):
                if isinstance(row.DMD_Generate, str):
                    assert len(row.DMD_Generate.split(';')) == len(
                        row.Ordering.split(';')), 'DMD generate is not the same length as ordering'
                else:
                    assert len(row.Ordering.split(';')) == 1, 'DMD Generate is the same length as ordering'

            orders += row.Ordering.split(';')
            # todo: handle accidental white spaces in xlsx template

            if isinstance(row.Exposure, str):
                exposures = row.Exposure.split(';')
            else:
                exposures = [row.Exposure]
            for e in exposures:
                if isinstance(e, str):
                    try:
                        assert int(e) >= 0, f"{e} is not greater than or equal to 0."
                    except ValueError as err:
                        logger.error(f'Value {e} cannot be converted to int\n{err}')
                        print(f'Value {e} cannot be converted to int\n{err}')
                        raise err

            if not pd.isna(row.ExcitationIntensity):
                excitations = df.ExcitationIntensity.iloc[0].split(':')
                for e in excitations:
                    assert 1000 >= int(e) >= 0

            dmd_generations = row.DMD_Generate.split(';') if not pd.isna(row.DMD_Generate) else [0 for _ in
                                                                                                 range(len(row_orders))]
            showimages = row.Show_Image.split(';') if not pd.isna(row.Show_Image) else [0 for _ in range(len(row_orders))]
            if isinstance(row.PFSHeight, str):  # todo: clean this up, standardize col types
                pfsoffsets = row.PFSHeight.split(';') if not pd.isna(row.PFSHeight) else []
            else:
                pfsoffsets = [row.PFSHeight] * len(row_orders)
            assert len(row_orders) == len(dmd_generations) == len(showimages) == len(dmd_functions) == len(pfsoffsets), f'{row.Well}'


        possible_channels = np.unique(possible_channels)
        orders = np.unique(orders)
        for ord in orders:
            assert '_' not in ord, f'Underscore cannot be in channel: {ord}'
            assert ord in possible_channels, f'Ordering {ord} not found in {possible_channels}'
        for ch in possible_channels:
            assert ch in orders, f'Channel {ch} not found in ordering {orders}'


    def generate_wells(self):
        wells = []
        for i in range(16):
            for j in range(24):
                wells += [ascii_uppercase[i] + str(j)]
                wells += [f'{ascii_uppercase[i]}{j:02d}']
        return wells
