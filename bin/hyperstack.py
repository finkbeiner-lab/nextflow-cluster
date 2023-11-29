'''
Creates ImageJ-readable hyperstack of channels and overlays
'''

import os, pickle, argparse, shutil, tifffile, cv2, struct
import numpy as np
import utils
from sql import Database
from db_util import Ops
from montage import Montage
from normalization import Normalize

def get_display_metadata(colormap_list):
    """
    Return IJMetadata and IJMetadataByteCounts tags from metadata dict.
    The tags can be passed to the TiffWriter.save function as extratags.
    """
    gray = np.tile(np.arange(256, dtype='uint8'), (3, 1))
    green = np.zeros((3, 256), dtype='uint8')
    green[1] = np.arange(256, dtype='uint8')
    red = np.zeros((3, 256), dtype='uint8')
    red[0] = np.arange(256, dtype='uint8')
    blue = np.zeros((3, 256), dtype='uint8')
    blue[2] = np.arange(256, dtype='uint8')

    colormap_arrays = []
    for c in colormap_list:
        if 'gray' in c:
            colormap_arrays.append(gray)
        elif 'green' in c:
            colormap_arrays.append(green)
        elif 'red' in c:
            colormap_arrays.append(red)
        elif 'blue' in c:
            colormap_arrays.append(blue)

    metadata = {'LUTs': colormap_arrays}

    byteorder = '>'
    header = [{'>': b'IJIJ', '<': b'JIJI'}[byteorder]]
    bytecounts = [0]
    body = []

    def writestring(data, byteorder):
        return data.encode('utf-16' + {'>': 'be', '<': 'le'}[byteorder])

    def writedoubles(data, byteorder):
        return struct.pack(byteorder+('d' * len(data)), *data)

    def writebytes(data, byteorder):
        return data.tobytes()

    metadata_types = (
        ('Info', b'info', 1, writestring),
        ('Labels', b'labl', None, writestring),
        ('Ranges', b'rang', 1, writedoubles),
        ('LUTs', b'luts', None, writebytes),
        ('Plot', b'plot', 1, writebytes),
        ('ROI', b'roi ', 1, writebytes),
        ('Overlays', b'over', None, writebytes))

    for key, mtype, count, func in metadata_types:
        if key not in metadata:
            continue
        if byteorder == '<':
            mtype = mtype[::-1]
        values = metadata[key]
        if count is None:
            count = len(values)
        else:
            values = [values]
        header.append(mtype + struct.pack(byteorder+'I', count))
        for value in values:
            data = func(value, byteorder)
            body.append(data)
            bytecounts.append(len(data))

    body = b''.join(body)
    header = b''.join(header)
    data = header + body
    bytecounts[0] = len(header)
    bytecounts = struct.pack(byteorder+('I' * len(bytecounts)), *bytecounts)
    return ((50839, 'B', len(data), data, True),
            (50838, 'I', len(bytecounts)//4, bytecounts, True))


def get_dataset_maximum(var_dict, img_paths, norm_intensity):
    '''get dataset max intensity for normalization'''
    for p in img_paths:
        img = cv2.imread(p[0], -1)
        img_max = np.amax(img)
        if img_max > norm_intensity:
            norm_intensity = img_max
        img = None
    return int(norm_intensity)

class Hyperstack:
    def __init__(self, opt):
        self.opt = opt
        self.Db = Database()
        self.Dbops = Ops()
        self.Norm = Normalize(self.opt)
        self.imagedir, self.analysisdir = self.Dbops.get_raw_and_analysis_dir()
        self.hyperstackdir = os.path.join(self.analysisdir, 'Hyperstacks')
        if not os.path.exists(self.hyperstackdir):
            os.makedirs(self.hyperstackdir)
        
        
    def run(args):
        
        tiledata_df = self.Norm.get_flatfields_for_training(['channeldata'])
        tiledata_df = tiledata_df.sort_values(by=['timepoint', 'well', 'tile'])
        available_channels = tiledata_df.channel.unique()
        timepoints = tiledata_df.timepoint.unique()
        wells = tiledata_df.well.unique()
        channels = args.channels.replace(' ', '').split(',')
        channels = [utils.get_ref_channel(c, available_channels) for c in channels]

        available_colors = ['green', 'red', 'blue', 'gray']
        colors = args.colors.replace(' ', '').lower().split(',')
        assert len(channels) == len(colors), 'Number of channels must equal the number of colors'
        assert len([c for c in colors if c not in available_colors]) == 0, 'Colors must be from the list of available colors (%s)' % str(available_colors)

        for i, ch in enumerate(channels):
            print ('Channel #%i: %s -- %s' % (i+1, ch, colors[i]))

        # scaling factor
        scaling_factor = args.scaling_factor
        print('Scaling factor:', str(scaling_factor))

        # parse normalization parameters
        norm_intensities = args.norm_intensities.replace(' ', '')
        if norm_intensities == '':
            normalize = False
            print('No normalization')
        else:
            normalize = True
            norm_intensities = norm_intensities.split(',')
            assert len(norm_intensities) == len(channels), 'Number of normalization intensities must match the number of channels (%i).' % len(channels)
            norm_vals = []
            for i, norm_val in enumerate(norm_intensities):
                if int(norm_val) == 0:
                    # determine the max intensity across all images of the current channel
                    if channels[i] == 'overlay':
                        norm_vals.append(255)
                    else:
                        img_paths = []
                        for well in var_dict['Wells']:
                            for tp in var_dict['TimePoints']:
                                print(image_tokens)
                                print(well)
                                print(tp)
                                print(channels[i])
                                print(var_dict['RoboNumber'])
                                img_paths.append(utils.get_filename(image_tokens, well, tp, channels[i], var_dict['RoboNumber']))
                        norm_vals.append(get_dataset_maximum(var_dict, img_paths, float(norm_val)))
                else:
                    # use the user-specified intensity for this channel
                    norm_vals.append(int(norm_val))
            print('Normalization intensities:', str(norm_vals))

        # loop through wells
        for well in wells:

            # initialize list for image channel stack
            c_stack_list = []

            # loop through timepoints
            for timepoint in timepoints:

                # initialize list for image channel stack paths
                c_stack_paths = []
                ch_colors_selected = []

                # loop through channels
                overlay_idx = []
                for idx, channel in enumerate(channels):
                    montage_file = tiledata_df.loc[(tiledata_df.well==well) & (tiledata_df.timepoint==timepoint) & (tiledata_df.channel==channel), 'imagemontage']
                    # get image file path for current well, timepoint, channel
                    c_stack_paths.append(montage_file)
                    # get channel colormap
                    ch_colors_selected.append(colors[idx])

                # read in images
                c_stack = [cv2.imread(i, -1) for i in c_stack_paths]

                # downscale
                if scaling_factor != 1:
                    c_stack = [cv2.resize(i, (0,0), fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA) for i in c_stack]

                # normalize
                if normalize:
                    c_stack_normed = []
                    for idx, c in enumerate(c_stack):
                        c_normed = np.rint(np.clip(c.astype(np.float) / norm_vals[idx], 0, 1) * 65535).astype(np.uint16)
                        c_stack_normed.append(c_normed)
                    c_stack = c_stack_normed

                # stack channels for current timepoint along 0th dimension
                c_stack = np.stack(c_stack, axis=0)

                # add channel stack to list of channel stacks
                c_stack_list.append(c_stack)

            # combine channel stacks for all timepoints along 0th axis
            t_stack = np.stack(c_stack_list, axis=0)

            # create hyperstack filename
            
            savename = os.path.basename(montage_file)
            savename = savename.split('.t')
            savename = savename + '_STACK.tif'
            savepath = os.path.join(self.hyperstackdir, savename)
            
            # generate colormap metadata
            display_metadata = get_display_metadata(ch_colors_selected)

            # save hyperstack
            with tifffile.TiffWriter(savepath, bigtiff=False, byteorder='>', imagej=True) as tif:
                for i in range(t_stack.shape[0]):
                    tif.save(t_stack[i], metadata={'Composite mode': 'composite'}, extratags=display_metadata)


    

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Generate hyperstacks.')
    parser.add_argument('input_dict',
        help='Load input variable dictionary')
    parser.add_argument('--experiment', default='20230928-MsNeu-RGEDItau1', type=str)
    parser.add_argument('channels',
        help='First channel of hyperstack.')
    parser.add_argument('colors',
        help='ImageJ color for first channel.')
    parser.add_argument('scaling_factor',
        help='Scaling factor', type=float, default=1)
    parser.add_argument('--norm_intensities',
        help='Option to normalize intensities', default='')
    args = parser.parse_args()
    print('Args', args)
    Hyp = Hyperstack(args)
    Hyp.run()