"""File for config between microscope."""


class Configure:
    def __init__(self, microscope: str, num_wells=96):
        self.rowsign = None
        self.colsign = None
        self.x_arm_coord = None
        self.y_arm_coord = None
        self.arm_dll_file = None
        self.arm_param_file = None
        self.arm_teach_file = None
        self.arm_seq_file = None
        self.a1_coordinate = None  # A1 in manual and automated runs will be overriden by experiment template.
        self.PFSDeviceNameInConfig = 'PFS'
        self.pfs_prop_name = 'FocusMaintenance'
        self.stage_has_lock = False
        self.incubator_com_port = None
        self.well_spacing = None
        self.fid_zheight = None

        self.EpiLightSourceName = None
        self.LightSourceHasVariableIntensity = True
        self.ConfocalLightSourceName = 'LMM5-Hub'
        self.ConfocalLightSourceNameSecondary = 'Cobolt'
        self.ConfocalChannelNameInConfig = 'Confocal'
        self.EpiChannelNameInConfig = 'Channels'
        self.DMDChannelNameInConfig = 'DMD'
        self.ConfocalChannelNameInConfigSecondary = 'ConfocalCobolt'
        self.ZDriveNameInConfig = 'ZDrive'
        self.ObjectiveDeviceNameInConfig = 'Nosepiece'
        self.ObjectiveGroupNameInConfig = 'Objective'
        self.spreadsheet_tab = None

        self.num_wells = num_wells
        if num_wells == 96:
            self.well_spacing = 9000
        elif num_wells == 384:
            self.well_spacing = 4500

        if microscope.lower() == 'tm':
            self.rowsign = -1
            self.colsign = 1
            self.a1_coordinate = (-49635, 31104)  # TM
            self.PFSDeviceNameInConfig = 'PFS'
            self.pfs_prop_name = 'FocusMaintenance'
            self.EpiLightSourceName = 'LightEngine_epi'


        elif microscope.lower() == 'robo3':
            self.rowsign = -1  # Sign for zeroed at top right
            self.colsign = -1
            self.x_arm_coord = -500000
            self.y_arm_coord = -300000
            self.arm_dll_file = r'C:\Users\finkbeinerlab\Documents\Peak Robotics, Inc\Peak KiNEDx Robot Control DLL V3\KiNEDxRobotControl.dll'
            self.arm_param_file = "C:\\ProgramData\\Peak Robotics, Inc\\Peak KiNEDx Robot Control DLL V3\\ParametersXXXX.ini"
            self.arm_teach_file = "C:\\ProgramData\\Peak Robotics, Inc\\Peak KiNEDx Robot Control DLL V3\\TeachPoints.ini"
            self.arm_seq_file = "C:\\ProgramData\\Peak Robotics, Inc\\Peak KiNEDx Robot Control DLL V3\\Sequences.ini"
            self.a1_coordinate = (173225, -17775)  # ROBO3 zeroed at up right
            self.fid_zheight = 7300
            self.PFSDeviceNameInConfig = 'TIPFSStatus'
            self.pfs_prop_name = 'Status'
            self.incubator_com_port = 'COM6'
            self.LightSourceHasVariableIntensity = False
            self.spreadsheet_tab = 'Robo III'



        elif microscope.lower() == 'robo4':
            self.rowsign = 1  # Sign for zeroed at top right
            self.colsign = 1
            self.x_arm_coord = 500000
            self.y_arm_coord = -300000
            self.arm_dll_file = r'C:\Users\finkbeinerlab\Desktop\Peak KiNEDx Robot Control DLL V3\KiNEDxRobotControl.dll'
            self.arm_param_file = r'C:\ProgramData\Peak Robotics, Inc\Peak KiNEDx Robot Control DLL V3\ParametersXXXX.ini'
            self.arm_teach_file = r'C:\ProgramData\Peak Robotics, Inc\Peak KiNEDx Robot Control DLL V3\TeachPoints.ini'
            self.arm_seq_file = r'C:\ProgramData\Peak Robotics, Inc\Peak KiNEDx Robot Control DLL V3\Sequences.ini'
            self.a1_coordinate = (15581, -70685)  # ROBO4 top right origin
            # self.a1_coordinate = (106460, 8268)  # ROBO4 bottom left origin
            self.fid_zheight = 7300
            self.PFSDeviceNameInConfig = 'TIPFSStatus'
            self.pfs_prop_name = 'Status'
            self.incubator_com_port = 'COM6'
            self.stage_has_lock = True

            self.EpiLightSourceName = 'Spectra'
            self.ConfocalLightSourceName = 'LMM5-Hub'
            self.ConfocalLightSourceNameSecondary = 'Cobolt'
            self.spreadsheet_tab = 'Robo IV'

        elif microscope.lower()=='ixm':
            self.rowsign = 1  # Sign for zeroed at top right
            self.colsign = 1
            self.x_arm_coord = 500000
            self.y_arm_coord = -300000
            self.arm_dll_file = None
            self.arm_param_file = None
            self.arm_teach_file = None
            self.arm_seq_file = None
            self.a1_coordinate = (0,0)
            self.fid_zheight = 7300
            self.PFSDeviceNameInConfig = None
            self.pfs_prop_name = None
            self.incubator_com_port = None
            self.stage_has_lock = True

            self.EpiLightSourceName = 'Spectra'
            self.ConfocalLightSourceName = 'LMM5-Hub'
            self.ConfocalLightSourceNameSecondary = 'Cobolt'
            self.spreadsheet_tab = 'IXM'


        else:
            raise Exception(f'Microscope {microscope} is not valid')
            # todo: kinedx files should live in the same directory for all computers