"""Read IXM template to galaxy xlsx format."""
import pandas as pd
import shutil
import os
from string import ascii_uppercase
import numpy as np
from collections import OrderedDict


class ConvertTemplate:
    def __init__(self, savepath, description='IXM Experiment'):
        self.savepath = savepath
        self.channel_dct = {}
        self.num_wavelengths = None
        self.exp = dict(ExperimentName=None,
                        Barcode=None,
                        Author=None,
                        Description=description,  
                        Plate=None,
                        WellCount=96, # changes if 384
                        ImageFolder='/gladstone/finkbeiner/robodata/IXM4Galaxy',  # appends author name later
                        PFSPerTile=1,
                        ImagingPattern='epi_only',
                        Email=None)
        self.plate = dict(Well=[],
                          Montage=None,
                          PFSHeight=None,
                          Channel=[],
                          Exposure=[],
                          ExcitationIntensity=[],
                          Objective=None,
                          Overlap=None,
                          Ordering=[],

                          # IXM will only use items above
                          DMD_Channel=None,
                          DMD_Exposure=None,
                          DMD_ExcitationIntensity=None,
                          Confocal_Channel=None,
                          Confocal_Exposure=None,
                          Confocal_ExcitationIntensity=None,
                          Cobolt_Channel=None,
                          Cobolt_Exposure=None,
                          Cobolt_ExcitationIntensity=None,
                          WellHeight=None,
                          DMD_Generate=None,
                          DMD_Function=None,
                          Experiment_Target=None,
                          Track_Channel=None,
                          Stim_Channel=None,
                          Stim_Filter=None,
                          DMD_Paint=None,
                          Holdout_N_Track=None,
                          Show_Image=None,
                          )
        self.timepoint = dict(Date=None,
                              Time=None,
                              Timepoint=None,
                              Hour=None,
                              EstimatedDuration=None
                              )
        self.microscope = dict(Microscope='IXM',
                               A1_x=0,
                               A1_y=0,
                               IncubatorCOMPort=None,
                               Stack=None,
                               Shelf=None,
                               UseArm=True,
                               AvailableWavelengths=None,
                               AvailableConfocalWavelengths=None,
                               UseEpi=True,
                               UseConfocal=False,
                               UseDMD=False,
                               UseDatabase=False,
                               UseTracking=False,
                               UseCellpose=False,
                               TrackPuncta=False,
                               UseFiducial=False,
                               Fid_Stepsize=400,
                               )

    def line_to_dict(self, line):
        # Experiment
        if 'stExperimentSet' in line:
            self.exp['Author'] = line.split(',')[1]
            self.exp['ImageFolder'] = os.path.join(self.exp['ImageFolder'], self.exp['Author'])

        if 'stDataFile' in line:
            self.exp['ExperimentName'] = line.split(',')[1]
        if 'stPlateType' in line:  # or stPlateCustomName?
            self.exp['Plate'] = line.split(',')[1]

        if 'bWells' in line:
            tokens = line.split(',')
            print(tokens)
            if int(tokens[1]) > 12 or int(tokens[2]) > 8:
                self.exp['WellCount'] = 384
            if tokens[-1] == 'TRUE':
                letter = ascii_uppercase[int(tokens[2])-1]
                num = int(tokens[1])
                well = f'{letter}{num:02d}'
                self.plate['Well'].append(well)

        # Plate
        if 'nWavelengths' in line:
            self.num_wavelengths = int(line.split(',')[-1])
        if 'gstMagnification' in line:
            self.plate['Objective'] = line.split(',')[1]
        if 'iSiteCount' in line:
            self.plate['Montage'] = int(np.sqrt(int(line.split(',')[1])))
        if 'stSetOfIllumination' in line:
            cnt = int(line.split(',')[1])
            if cnt <= self.num_wavelengths:
                channel_name = line.split(',')[-1]
                channels = self.channel_dct[channel_name]
                channel_lst = []
                intensity_lst = []
                for ch, intensity in channels.items():
                    channel_lst.append(ch)
                    intensity_lst.append(str(intensity))
                excitation_intensity = ':'.join(intensity_lst)
                available_wavelengths = ';'.join(channel_lst)
                self.plate['Channel'].append(channel_name)
                self.plate['Ordering'].append(channel_name)
                self.plate['ExcitationIntensity'].append(excitation_intensity)
                self.microscope['AvailableWavelengths']=available_wavelengths
        if 'dbSetOfExposures' in line:
            cnt = int(line.split(',')[1])
            if cnt <= self.num_wavelengths:
                self.plate['Exposure'].append((line.split(',')[-1]))

    def set_up_plate(self):
        intensity_mat = None
        for intensity in self.plate['ExcitationIntensity']:
            intensities = intensity.split(':')
            intensities = np.array([int(i) for i in intensities])
            if intensity_mat is None:
                intensity_mat = intensities
            else:
                intensity_mat = np.vstack((intensity_mat, intensities))
        intensity_maxes = list(intensity_mat.max(axis=0))
        intensity_maxes = [str(i) for i in intensity_maxes]
        intensity_str = ':'.join(intensity_maxes)
        # todo: table to keep track of channels instead of list of values?

        self.plate['ExcitationIntensity'] = intensity_str
        self.plate['Channel'] = ';'.join(self.plate['Channel'])
        self.plate['Exposure'] = ';'.join(self.plate['Exposure'])
        self.plate['Ordering'] = ';'.join(self.plate['Ordering'])
        length = len(self.plate['Well'])
        for key in self.plate.keys():
            if key!='Well':
                print(key)
                self.plate[key] = [self.plate[key]] * length
        print()



    def save_xlsx(self):
        df_exp = pd.DataFrame(self.exp, index=[0])
        df_plate = pd.DataFrame(self.plate)
        df_timepoint = pd.DataFrame(self.timepoint, index=[0])
        df_microscope = pd.DataFrame(self.microscope, index=[0])
        with pd.ExcelWriter(self.savepath) as writer:

            df_exp.to_excel(writer, sheet_name='experiment', index=False)
            df_plate.to_excel(writer, sheet_name='plate', index=False)
            df_timepoint.to_excel(writer, sheet_name='timepoint', index=False)
            df_microscope.to_excel(writer, sheet_name='microscope', index=False)

    def convert_template(self, hts, illumination):
        count = 0
        self.get_channels(illumination)
        with open(hts) as fp:
            while True:
                count += 1
                line = fp.readline()

                if not line:
                    break
                line = line.replace(' ', '')
                line = line.replace('"', '')
                line = line.replace('\n', '')
                # print("Line{}: {}".format(count, line.replace(' ', '')))
                self.line_to_dict(line)
        self.set_up_plate()
        self.save_xlsx()

    def get_channels(self, illumination):
        with open(illumination) as fp:
            while True:
                line = fp.readline()
                if not line:
                    break
                print(line)
                if len(line) > 500:
                    line = line.replace(' ', '')
                    line = line.replace('"', '')
                    line = line.replace('', '')
                    line = line.replace('\n', '')
                    tokens = line.split('*')
                    if 'Setting' in tokens[0]:
                        channel_name = tokens[1].split('^')[1]
                        self.channel_dct[channel_name] = OrderedDict()
                        for token in tokens:
                            if 'ComponentName' in token:
                                parts = token.split('^')
                                if 'Intensity' in parts[1]:
                                    wavelength_name = parts[1].split(',')[0]
                                    wavelength_name = wavelength_name.replace('Lumencor', '')
                                    wavelength_name = wavelength_name.replace('Intensity', '')
                                    channel_intensity = int(parts[-1])
                                    self.channel_dct[channel_name][wavelength_name] = channel_intensity


if __name__ == '__main__':
    hts = '/home/jlamstein/Downloads/AH-carDIFF6-chemDNDA-ICC-06082023.HTS'
    illumination = r'/gladstone/finkbeiner/robodata/IXM Documents/illumination-setting-2023-06-16.ILS'
    savepath = '/gladstone/finkbeiner/robodata/IXM4Galaxy/Austin/AH-carDIFF6-chemDNDA-ICC-06082023/AH-carDIFF6-chemDNDA-ICC-06082023.xlsx'
    # dst = os.path.join(os.path.dirname(hts), os.path.basename(hts) + '.csv')
    # shutil.copyfile(hts, dst)
    print('running main file of ixm2galaxy')
    Conv = ConvertTemplate(savepath)
    Conv.convert_template(hts, illumination)
