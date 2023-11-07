'''
Creates ImageJ-readable hyperstack of channels and overlays
'''

import os, pickle, argparse, shutil, tifffile, cv2, struct
import numpy as np
import utils

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


def main():
    ''' Main point of entry '''

    # get arguments
    parser = argparse.ArgumentParser(description='Generate hyperstacks.')
    parser.add_argument('input_dict',
        help='Load input variable dictionary')
    parser.add_argument('channels',
        help='First channel of hyperstack.')
    parser.add_argument('colors',
        help='ImageJ color for first channel.')
    parser.add_argument('scaling_factor',
        help='Scaling factor', type=float, default=1)
    parser.add_argument('output_dict',
        help='Write variable dictionary.')
    parser.add_argument('--norm_intensities',
        help='Option to normalize intensities', default='')
    parser.add_argument('--images_path',
        help='Path to images folder', default='')
    parser.add_argument('--overlays_path',
        help='Path to overlay images folder', default='')
    parser.add_argument('--output_path',
        help='Path to save hyperstacks', default='')
    args = parser.parse_args()

    # assign arguments to variables
    var_dict = pickle.load(open(args.input_dict, 'rb'))

    # parse channels and corresponding pseudocolors
    channels = args.channels.replace(' ', '').split(',')
    available_channels = var_dict['Channels']
    available_channels.append('overlay')
    channels = [utils.get_ref_channel(c, available_channels) for c in channels]

    available_colors = ['green', 'red', 'blue', 'gray']
    colors = args.colors.replace(' ', '').lower().split(',')
    assert len(channels) == len(colors), 'Number of channels must equal the number of colors'
    assert len([c for c in colors if c not in available_colors]) == 0, 'Colors must be from the list of available colors (%s)' % str(available_colors)

    for i, ch in enumerate(channels):
        print ('Channel #%i: %s -- %s' % (i+1, ch, colors[i]))

    # get input/output paths
    images_path = utils.get_path(args.images_path, var_dict['GalaxyOutputPath'], 'AlignedImages')
    assert os.path.exists(images_path), 'Confirm path for images exists (%s)' % args.images_path
    print ('Images path:', images_path)

    overlays_path = utils.get_path(args.overlays_path, var_dict['GalaxyOutputPath'], 'OverlaysTablesResults')

    output_path = utils.get_path(args.output_path, var_dict['GalaxyOutputPath'], 'Hyperstacks')
    utils.create_dir(output_path)
    assert os.path.exists(os.path.split(output_path)[0]), 'Confirm that the output path parent folder (%s) exists.' % os.path.split(output_path)[0]
    print('Output path:' , output_path)

    if any('overlay' in x for x in channels):
        assert os.path.exists(overlays_path), 'Confirm path for overlays exists (%s)' % overlays_path
        print('Overlays path:', overlays_path)
        overlay_paths = ''
        if var_dict['DirStructure'] == 'root_dir':
            overlay_paths = [os.path.join(overlays_path, name) for name in os.listdir(overlays_path) if name.endswith('.tif') and '_FIDUCIARY_' not in name]
        elif var_dict['DirStructure'] == 'sub_dir':
            # Only traverse root and immediate subdirectories for images
            relevant_dirs = [overlays_path] + [os.path.join(overlays_path, name) for name in os.listdir(overlays_path) if os.path.isdir(os.path.join(overlays_path, name))]
            overlay_paths = [os.path.join(relevant_dir, name) for relevant_dir in relevant_dirs for name in os.listdir(relevant_dir) if name.endswith('.tif') and '_FIDUCIARY_' not in name]
        else:
            raise Exception('Unknown Directory Structure!')
        overlay_tokens = utils.tokenize_files(overlay_paths)

    # scaling factor
    scaling_factor = args.scaling_factor
    print('Scaling factor:', str(scaling_factor))

    # get image paths
    image_paths = ''
    if var_dict['DirStructure'] == 'root_dir':
        image_paths = [os.path.join(images_path, name) for name in os.listdir(images_path) if name.endswith('.tif') and '_FIDUCIARY_' not in name]
    elif var_dict['DirStructure'] == 'sub_dir':
        # Only traverse root and immediate subdirectories for images
        relevant_dirs = [images_path] + [os.path.join(images_path, name) for name in os.listdir(images_path) if os.path.isdir(os.path.join(images_path, name))]
        image_paths = [os.path.join(relevant_dir, name) for relevant_dir in relevant_dirs for name in os.listdir(relevant_dir) if name.endswith('.tif') and '_FIDUCIARY_' not in name]
    else:
        raise Exception('Unknown Directory Structure!')
    image_tokens = utils.tokenize_files(image_paths)

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
    for well in var_dict['Wells']:

        # initialize list for image channel stack
        c_stack_list = []

        # loop through timepoints
        for timepoint in var_dict['TimePoints']:

            # initialize list for image channel stack paths
            c_stack_paths = []
            ch_colors_selected = []

            # loop through channels
            overlay_idx = []
            for idx, channel in enumerate(channels):
                # get image file path for current well, timepoint, channel
                if channel == 'overlay':
                    overlay_path = utils.get_filename(overlay_tokens, well, timepoint, var_dict['MorphologyChannel'], var_dict['RoboNumber'])
                    assert len(overlay_path) > 0, 'No overlay image found for well %s at %s in %s' % (well, timepoint, overlays_path)
                    assert len(overlay_path) == 1, 'More than one overlay image found for well %s at %s in %s' % (well, timepoint, overlays_path)
                    c_stack_paths.append(overlay_path[0])
                    overlay_idx.append(idx)
                else:
                    img_path = utils.get_filename(image_tokens, well, timepoint, channel, var_dict['RoboNumber'])
                    assert len(img_path) > 0, 'No image found for well %s at %s in %s' % (well, timepoint, images_path)
                    assert len(img_path) == 1, 'More than one image found for well %s at %s in %s' % (well, timepoint, images_path)
                    c_stack_paths.append(img_path[0])

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
        orig_name = utils.extract_file_name(c_stack_paths[0])
        new_name = utils.make_file_name(output_path, orig_name + '_STACK')

        # generate colormap metadata
        display_metadata = get_display_metadata(ch_colors_selected)

        # save hyperstack
        with tifffile.TiffWriter(new_name, bigtiff=False, byteorder='>', imagej=True) as tif:
            for i in range(t_stack.shape[0]):
                tif.save(t_stack[i], metadata={'Composite mode': 'composite'}, extratags=display_metadata)

    # write out dictionary
    outfile = args.output_dict
    pickle.dump(var_dict, open('var_dict.p', 'wb'))
    outfile = shutil.move('var_dict.p', outfile)
    utils.save_user_args_to_csv(args, output_path, 'hyperstack')

if __name__ == '__main__':

     main()