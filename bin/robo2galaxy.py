"""Read Robo template to galaxy xlsx format."""
import pandas as pd
import shutil
import os
from string import ascii_uppercase
import numpy as np
from collections import OrderedDict


class RoboConvertTemplate:
    def __init__(self, savepath, description='Roboscope Experiment'):
        self.savepath = savepath
        self.channel_dct = {}
        self.num_wavelengths = None
        self.exp = dict(ExperimentName=None,
                        Barcode=None,
                        Author=None,
                        Description=description,  
                        Plate=None,
                        WellCount=96, # changes if 384
                        ImageFolder='/gladstone/finkbeiner/robodata',  # appends roboscope and experiment name later
                        PFSPerTile=1,
                        ImagingPattern='epi_and_confocal',
                        Email=None)
        self.plate = dict(Well=[],
                          Montage=[],
                          PFSHeight=[],
                          Channel=[],
                          Exposure=[],
                          ExcitationIntensity=[],
                          Objective=[],
                          Overlap=[],
                          Ordering=[],

                          # IXM will only use items above
                          DMD_Channel=None,
                          DMD_Exposure=None,
                          DMD_ExcitationIntensity=None,
                          Confocal_Channel=[],
                          Confocal_Exposure=[],
                          Confocal_ExcitationIntensity=[],
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
                               UseConfocal=True,
                               UseDMD=False,
                               UseDatabase=False,
                               UseTracking=False,
                               UseCellpose=False,
                               TrackPuncta=False,
                               UseFiducial=False,
                               Fid_Stepsize=400,
                               )
                
    def line_to_dict(self, line, current_table):
        tokens = line.split(',')
        print(tokens)

        if current_table==1:
            # experiment table
            self.exp['ExperimentName'] = tokens[0]
            self.exp['Barcode'] = tokens[1]
            self.microscope['Stack'] = int(tokens[2])
            self.microscope['Shelf'] = int(tokens[3])
            self.exp['Author'] = tokens[4]
            if not isinstance(tokens[5], str):
                self.exp['Description'] = ''
            self.exp['Plate'] = tokens[6]
            self.exp['WellCount'] = int(tokens[7])
            self.microscope['Microscope'] = tokens[8].upper()
            self.exp['ImageFolder'] = os.path.join(self.exp['ImageFolder'], self.microscope['Microscope'].capitalize() + 'Images', self.exp['ExperimentName'])
            self.exp['PFSPerTile'] = bool(tokens[11])
            if self.microscope['Microscope'].lower()=='robo3':
                epi = 'Violet;Blue;Cyan;Teal;Green;Red'
                confocal = '405nm-5;447nm-6;488nm-7;516nm-2;561nm-4;642nm-3'
            elif self.microscope['Microscope'].lower()=='robo4':
                epi = 'Violet;Blue;Cyan;Teal;Green;Red'
                confocal = '405nm-5;447nm-6;488nm-7;516nm-2;561nm-4;642nm-3'
            elif self.microscope['Microscope'].lower()=='robo5':
                epi = 'Violet;Blue;Cyan;Teal;Green;Red'
                confocal = '405nm-5;447nm-6;488nm-7;516nm-2;561nm-4;642nm-3'
            self.microscope['AvailableWavelengths'] = epi
            self.microscope['AvailableConfocalWavelengths'] = confocal
        if current_table==2:
            pass
        if current_table==3:
            pass
        if current_table==4:
            self.plate['Well'].append(tokens[0])
            self.plate['Montage'].append(int(tokens[2][0]))  # 3x3, 4x4, assume square montage
            self.plate['PFSHeight'].append(tokens[3])
            self.plate['ExcitationIntensity'].append(tokens[6])  # unchanged from template, includes powers from confocal wavelengths but not used in epi, the config is still set up this way
            self.plate['Objective'].append(tokens[9])
            self.plate['Overlap'].append(float(tokens[10]))
            tokens[4] = tokens[4].replace('_', '-')
            self.plate['Ordering'].append(tokens[4])
            channels = tokens[4].split(';')
            
            exposures = tokens[5].split(';')
            exposures = tokens[5].split(';')
            excitations = tokens[6].split(':')
            confocal_excitations = excitations[-6:]
            confocal_channels = []
            confocal_exposures = []
            indices = []
            for i, (ch, exposure) in enumerate(zip(channels, exposures)):
                if 'confocal' in ch.lower():
                    confocal_channels.append(ch)
                    confocal_exposures.append(exposure)
                    indices.append(i)
            for i in reversed(indices):
                del channels[i]
                del exposures[i]
            self.plate['Channel'].append(';'.join(channels))
            self.plate['Exposure'].append(';'.join(exposures))
            self.plate['Confocal_Channel'].append(';'.join(confocal_channels))
            self.plate['Confocal_Exposure'].append(';'.join(confocal_exposures))
            self.plate['Confocal_ExcitationIntensity'].append(':'.join(confocal_excitations))
            # Cobolt isn't running on old template
            

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
            
    def convert_template(self, robo):
        count = 0
        no_line = 1
        current_table = 0
        with open(robo) as fp:
            while True:
                count += 1
                line = fp.readline()
                line = line.replace(' ', '')
                line = line.replace('"', '')
                line = line.replace('\n', '')
                print(line)
                is_line = False
                if line:
                    for t in line.split(','):
                        if len(t)>0: 
                            is_line = True
                if not is_line:
                    print('no line')
                    no_line += 1
                    if no_line > 3: break
                    continue
                elif no_line > 0:
                    # white space before this line, new table
                    table_cols = line.split(',')
                    current_table += 1
                    no_line = 0
                else:
                    self.line_to_dict(line, current_table)
        self.save_xlsx()

    def _convert_template(self, hts, illumination):
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
    robo_template = '/gladstone/finkbeiner/robodata/Robo4Images/20230920-MsDS-GFP/20230920-MsDS-GFP.csv'
    savepath = '/gladstone/finkbeiner/robodata/Robo4Images/20230920-MsDS-GFP/20230920-MsDS-GFP.xlsx'
    # dst = os.path.join(os.path.dirname(hts), os.path.basename(hts) + '.csv')
    # shutil.copyfile(hts, dst)
    print('running main file of robo2galaxy')
    Conv = RoboConvertTemplate(savepath)
    Conv.convert_template(robo_template)
