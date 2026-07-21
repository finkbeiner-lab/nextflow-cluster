"""Shared utility functions for the Finkbeiner Lab image analysis pipeline.

Provides file I/O helpers, natural sorting, image manipulation, filename
tokenization, ImageMagick wrappers, and user-input parsing used across
multiple pipeline modules.
"""

import re
import os
import glob
import sys
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import cv2
import pickle
import pprint
import subprocess
import datetime
import shutil

# Pre-compiled pattern for splitting strings on digit sequences
numbers = re.compile(r'(\d+)')


def natural_sort(l: List[str]) -> List[str]:
    """Sort a list of strings in natural (human-friendly) order.

    Numeric substrings are compared by value rather than lexicographically,
    so ``['img2', 'img10', 'img1']`` becomes ``['img1', 'img2', 'img10']``.

    Args:
        l: List of strings to sort.

    Returns:
        A new list sorted in natural order.
    """
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)


def numericalSort(value: str) -> List[Union[str, int]]:
    """Convert a string into a mixed list of strings and ints for sorting.

    Splits the input on digit sequences and converts the numeric parts to
    integers.  Useful as a ``key`` function for ``sorted()``.

    Example:
        ``'ho-1-22-333'`` becomes ``['ho-', 1, '-', 22, '-', 333]``

    Args:
        value: The string to split and convert.

    Returns:
        A list of alternating string and integer elements.
    """
    # Split on digit groups: 'ho-1-22-333' -> ['ho-', '1', '-', '22', '-', '333']
    parts = numbers.split(value)
    # Convert every other element (the digit groups) to int for numeric comparison
    parts[1::2] = map(int, parts[1::2])
    return parts


def create_dir(path: str) -> None:
    """Create a directory (and parents) if it does not already exist.

    Args:
        path: Filesystem path of the directory to create.
    """
    if not os.path.exists(path):
        os.makedirs(path)


def get_path(user_path: str, default_path: str, default_folder: str) -> str:
    """Return user-supplied path if non-empty, otherwise a default.

    Args:
        user_path: Path entered by the user (may be blank/whitespace).
        default_path: Base directory for the fallback path.
        default_folder: Folder name appended to *default_path* when
            *user_path* is blank.

    Returns:
        The resolved filesystem path.
    """
    if str.strip(user_path) != '':
        path = str.strip(user_path)
    else:
        path = os.path.join(default_path, default_folder)

    return path


def make_filelist(path: str, identifier: str, verbose: bool = False) -> List[str]:
    """Glob files in *path* matching *identifier* and return them sorted.

    Args:
        path: Directory to search.
        identifier: Substring that filenames must contain.
        verbose: If True, print the glob pattern and matching filenames.

    Returns:
        Naturally-sorted list of absolute file paths.
    """
    filelist = sorted(
        glob.glob(os.path.join(path, '*' + identifier + '*')), key=numericalSort)

    if verbose:
        print("Selection filter:")
        print(os.path.join(path, '*' + identifier + '*'))
        pprint.pprint([os.path.basename(fel) for fel in filelist])

    return filelist


def make_filelist_wells(path: str, identifier: str, verbose: bool = False) -> List[str]:
    """Glob files matching *identifier* in *path* and all its subdirectories.

    Args:
        path: Root directory to walk.
        identifier: Substring that filenames must contain.
        verbose: If True, print discovery details.

    Returns:
        Flat, naturally-sorted list of absolute file paths from all
        subdirectories.
    """
    folders = [x[0] for x in os.walk(path)]
    filelist = []
    ffilelist = []
    for folder in folders:
        if verbose:
            print('Checking for files in:', os.path.basename(folder))

        well_files = sorted(glob.glob(os.path.join(folder, '*' + identifier + '*')), key=numericalSort)
        if verbose:
            print('Found the following files:')
            pprint.pprint([os.path.basename(wf) for wf in well_files])
        filelist.append(well_files)
        ffilelist = [item for sublist in filelist for item in sublist]
    if verbose:
        print("Selection filter:")
        print(os.path.join(path, '*' + identifier + '*'))
        pprint.pprint([os.path.basename(fel) for fel in ffilelist])
        pprint.pprint([fel for fel in ffilelist])
    return ffilelist


def reroute_imgpntr_to_wells(img_pointer: str, well: str, verbose: bool = False) -> str:
    """Insert a well subdirectory into an image path if not already present.

    Example::

        Input:  /data/CellMasks/my_image.tif
        Output: /data/CellMasks/A1/my_image.tif

    If the well subdirectory already exists in the path, the pointer is
    returned unchanged.

    Args:
        img_pointer: Full path to the image file.
        well: Well identifier (e.g. ``'A1'``).
        verbose: If True, print before/after paths.

    Returns:
        Updated image path containing the well subdirectory.
    """
    tif_path, tif_name = os.path.split(img_pointer)
    assert well in tif_name, 'The well is not part of the filename.'
    assert tif_name.split('_')[4] == well, 'Well token is in unexpected location.'
    # Check if a well directory is already present
    if well in os.path.basename(tif_path):
        tif_path_with_well = tif_path
    else:
        tif_path_with_well = os.path.join(tif_path, well)
    if not os.path.exists(tif_path_with_well):
        os.makedirs(tif_path_with_well)
    updated_img_pointer = os.path.join(tif_path_with_well, tif_name)
    if verbose:
        print('Initial destination:', img_pointer)
        print('Updated destination:', updated_img_pointer)
    return updated_img_pointer


def find_stack_size(path: str, identifier: str) -> int:
    """Count files in *path* whose names contain *identifier*.

    Args:
        path: Directory to search.
        identifier: Substring to match in filenames.

    Returns:
        Number of matching files.
    """
    num_images = len(glob.glob(os.path.join(path, '*' + identifier + '*')))
    print('Number of images:', num_images)
    return num_images


def find_max_dimensions(filelist: List[str]) -> Tuple[int, int]:
    """Find the maximum height and width across all images in *filelist*.

    Args:
        filelist: List of image file paths.

    Returns:
        Tuple of ``(max_rows, max_cols)``.
    """
    shapes = []
    for filename in filelist:
        img = cv2.imread(filename, 0)
        if img is None:
            raise ValueError(
                f'cv2.imread returned None (missing or corrupt image): {filename}')
        shapes.append(img.shape)
    max_rows = max([shape[0] for shape in shapes])
    max_cols = max([shape[1] for shape in shapes])
    return (max_rows, max_cols)


def find_image_canvas(aligned_path: str, unprocessed_data_path: str,
                      identifier: str, label: str) -> Tuple[int, int]:
    """Load or compute the maximum image dimensions for the experiment.

    If a cached dimensions file exists in *aligned_path*, it is loaded.
    Otherwise, dimensions are computed from channel-1 images and saved.

    Args:
        aligned_path: Directory where cached dimension files live.
        unprocessed_data_path: Directory of raw images used to compute dims.
        identifier: Substring used to locate the cached file (e.g. ``'dim'``).
        label: Prefix for the saved dimensions filename.

    Returns:
        Tuple of ``(max_height, max_width)``.
    """
    max_dim_entry = make_filelist(aligned_path, identifier)
    if len(max_dim_entry) > 0:
        max_dim = load_obj(
            aligned_path,
            extract_file_name(max_dim_entry[0]))
    else:
        a_channel_list = make_filelist(unprocessed_data_path, 'c1')
        max_dim = find_max_dimensions(a_channel_list)
        print(max_dim)
        save_obj(max_dim, aligned_path, label + '_max_dim')

    print('Found max image dimensions (height, width):', max_dim)
    return max_dim


def make_file_name_from_tokens(img_pointer: str, step_related_suffix: str,
                               specific_token: Union[str, int] = 'N') -> str:
    """Build a new filename by appending a suffix to an existing tokenized name.

    For Robo0 naming conventions.  Optionally overrides a specific token
    position with ``'1'`` before joining.

    Args:
        img_pointer: Full path to the reference image file.
        step_related_suffix: Suffix to append (e.g. ``'MN'``).
        specific_token: Token index to override with ``'1'``, or ``'N'``
            to leave all tokens unchanged.

    Returns:
        New image file path with the suffix appended.
    """
    img_path = os.path.dirname(img_pointer)
    img_name_tokens = os.path.splitext(os.path.basename(img_pointer))[0].split('_')

    if specific_token != 'N':
        img_name_tokens[specific_token] = str(1)
    new_img_pointer = '_'.join(['_'.join(img_name_tokens[0:len(img_name_tokens)]), step_related_suffix + '.tif'])
    new_img_pointer = os.path.join(img_path, new_img_pointer)

    return new_img_pointer


def extract_file_name(filename_path: str) -> str:
    """Extract the filename stem (no directory, no extension) from a path.

    Args:
        filename_path: Full or relative file path.

    Returns:
        Filename without extension.
    """
    img_file_name = os.path.basename(filename_path)
    img_name = os.path.splitext(img_file_name)
    return img_name[0]


def make_file_name(path: str, image_name: str, ext: str = '.tif') -> str:
    """Join a directory, image name, and extension into a full path.

    Args:
        path: Directory path.
        image_name: Base name without extension.
        ext: File extension including the dot.

    Returns:
        Full file path.
    """
    return os.path.join(path, image_name + ext)


def modify_file_name(filename_path: str, identifier: str) -> str:
    """Append *identifier* to a filename (before the ``.tif`` extension).

    Args:
        filename_path: Original file path.
        identifier: String to append to the base name.

    Returns:
        Modified file path ending in ``.tif``.
    """
    path = os.path.dirname(filename_path)
    original_name = extract_file_name(filename_path)
    new_file_name = os.path.join(path, original_name + identifier + '.tif')
    return new_file_name


def cnum_to_tnum_renamer(filename_cnum: str, character_string: str) -> str:
    """Replace the channel prefix in a filename with *character_string*.

    Used when renaming channel-numbered files (e.g. ``c1``) to
    threshold-numbered files (e.g. ``t1``).

    Args:
        filename_cnum: File path with channel-style naming.
        character_string: Replacement prefix (e.g. ``'t'``).

    Returns:
        New filename stem with the substituted prefix.
    """
    original_name = extract_file_name(filename_cnum)
    filename_tnum = ''.join((
        original_name[0:len(original_name) - 2],
        character_string,
        original_name[len(original_name) - 1]))
    return filename_tnum


def channel_free_name_extractor(filename_cnum: str) -> str:
    """Strip the trailing channel token from a filename.

    Removes the last 3 characters of the stem (e.g. ``_c1``).

    Args:
        filename_cnum: File path with a channel suffix.

    Returns:
        Filename stem without the channel token.
    """
    original_name = extract_file_name(filename_cnum)
    channel_free_name = original_name[0:len(original_name) - 3]
    return channel_free_name


def get_filelists(path: str, channel_list: List[str]) -> Dict[str, List[str]]:
    """Build a dictionary mapping channel identifiers to their file lists.

    Args:
        path: Directory to search.
        channel_list: Channel identifier strings
            (e.g. ``['c1', 'c2', 'c3', 'c4']``).

    Returns:
        Dictionary keyed by channel name, with values being sorted file
        lists. Channels with no matching files are omitted.
    """
    channel_dictionary = {}
    for channel_name in channel_list:
        filelist = make_filelist(path, channel_name)
        if len(filelist) == 0:
            continue
        channel_dictionary[channel_name] = filelist
    return channel_dictionary


# --------------------stuck on this---------------------------------------
def compare_filename_lists(channel_dictionary: Dict[str, List[str]]) -> None:
    """Assert that every channel has the same number of files with matching names.

    Args:
        channel_dictionary: Mapping of channel name to file list (as
            returned by :func:`get_filelists`).

    Raises:
        AssertionError: If any channel's file list differs from the first.
    """
    # create a reference
    first_filelist = channel_dictionary.values()[0]
    ref_filelist = [channel_free_name_extractor(first_filelist[ind])
                    for ind in range(len(first_filelist))]
    # check all against reference
    for chnl, filelist in channel_dictionary.items():

        if len(filelist) != len(first_filelist):
            print('Channel', chnl, '/ c1', 'has ',
                  len(filelist), '/', len(first_filelist), 'files.')
            print('Check length of ' + chnl + '.')
            print('Below are the files that do not match.')
            print('This list is short if a file is deleted.')
            print('This list is long if slides were imaged \
                with different number of channels.')
            print('\n'.join(set(filelist) ^ set(first_filelist)))
        comp_filelist = [channel_free_name_extractor(filelist[ind])
                         for ind in range(len(first_filelist))]
        assert comp_filelist == ref_filelist, 'Check that ' + chnl + ' files match.'


# ------------------------------------------------------------------------

def save_obj(object_name: Any, destination_path: str, obj_save_name: str) -> None:
    """Pickle an object to disk.

    Args:
        object_name: Python object to serialize.
        destination_path: Directory in which to save the file.
        obj_save_name: Filename stem (a ``.p`` extension is appended).
    """
    pickle.dump(object_name, open(os.path.join(
        destination_path, obj_save_name + '.p'), 'wb'))


def load_obj(source_path: str, obj_save_name: str) -> Any:
    """Unpickle an object from disk.

    Args:
        source_path: Directory containing the pickle file.
        obj_save_name: Filename stem (a ``.p`` extension is appended).

    Returns:
        The deserialized Python object.
    """
    return pickle.load(open(os.path.join(
        source_path, obj_save_name + '.p'), 'rb'))


def resizer(img: np.ndarray, factor: float = 0.25) -> np.ndarray:
    """Resize an image by a constant scale factor.

    Args:
        img: 2-D grayscale image array.
        factor: Scale factor (e.g. 0.25 for quarter size).

    Returns:
        Resized image.
    """
    height, width = img.shape
    small_width, small_height = int(factor * width), int(factor * height)
    small_img = cv2.resize(img, (
        small_width, small_height), interpolation=cv2.INTER_CUBIC)
    #
    return small_img


def width_resizer(img: np.ndarray, target_width: int = 100) -> np.ndarray:
    """Resize an image to a target width, preserving aspect ratio.

    Args:
        img: 2-D or 3-D image array.
        target_width: Desired width in pixels.

    Returns:
        Resized image.
    """
    height, width = img.shape[0:2]
    factor = float(target_width) / width
    small_width, small_height = int(factor * width), int(factor * height)
    small_img = cv2.resize(img, (
        small_width, small_height), interpolation=cv2.INTER_CUBIC)

    return small_img


def height_resizer(img: np.ndarray, target_height: int = 100) -> np.ndarray:
    """Resize an image to a target height, preserving aspect ratio.

    Args:
        img: 2-D or 3-D image array.
        target_height: Desired height in pixels.

    Returns:
        Resized image.
    """
    height, width = img.shape[0:2]
    factor = float(target_height) / height
    small_width, small_height = int(factor * width), int(factor * height)
    small_img = cv2.resize(img, (
        small_width, small_height), interpolation=cv2.INTER_CUBIC)

    return small_img


def show_wait_close(display_string: str, image: np.ndarray) -> None:
    """Display an equalized, resized image and wait for a keypress to close.

    Args:
        display_string: Window title.
        image: Grayscale image to display.
    """
    cv2.imshow(display_string, resizer(cv2.equalizeHist(image), 0.1))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def show_stack(path: str, identifier: str) -> None:
    """Sequentially display all images in *path* matching *identifier*.

    Each image is shown with its filename overlaid; press any key to
    advance.

    Args:
        path: Directory containing images.
        identifier: Substring to filter filenames.
    """
    filelist = make_filelist(path, identifier)
    for ind in range(len(filelist)):
        image = cv2.imread(filelist[ind], 0)
        font = cv2.FONT_HERSHEY_PLAIN
        details = extract_file_name(filelist[ind])
        cv2.putText(image, details, (10, 20), font, 1, 200, 2, cv2.LINE_AA)
        cv2.imshow(extract_file_name(
            filelist[ind]), resizer(cv2.equalizeHist(image), 0.1))
        cv2.waitKey(0)
    cv2.destroyAllWindows()


def find_ind_of_filename(image_path: str, image_filename: str) -> Union[int, bool]:
    """Find the index of a specific file within its channel's sorted file list.

    Args:
        image_path: Directory containing the images.
        image_filename: Basename of the image (e.g. ``'Shh_slide1_slice1_c1.tif'``).

    Returns:
        Integer index of the file in the sorted list, or ``False`` if not found.
    """
    image_name = extract_file_name(image_filename)
    print(image_name)
    ch_id = image_name[len(image_name) - 2:]
    one_channel_filelist = make_filelist(image_path, ch_id)
    if os.path.join(image_path, image_filename) not in one_channel_filelist:
        print('Image', image_filename, 'is not in this experiment.')
        print(' Please correct options.py.')
        ind = False

    if os.path.join(image_path, image_filename) in one_channel_filelist:
        ind = one_channel_filelist.index(os.path.join(image_path, image_filename))
    return ind


def return_image_indices(bad_image_list: List[str], image_path: str) -> List[int]:
    """Look up sorted-list indices for a list of image filenames.

    Args:
        bad_image_list: Basenames of images to locate.
        image_path: Directory containing the images.

    Returns:
        List of integer indices (files not found are silently excluded).
    """
    bad_img_indices = []
    for image in bad_image_list:
        ind = find_ind_of_filename(image_path, image)
        if type(ind) == int:
            bad_img_indices.append(ind)
    return bad_img_indices


def assign_or_make_dir_path(base_path: str, directory: str) -> None:
    """Create *directory* inside *base_path* if it does not exist.

    Args:
        base_path: Parent directory (concatenated directly, not joined).
        directory: Subdirectory name to create.
    """
    if not os.path.exists(base_path + directory):
        os.makedirs(base_path + directory)


def create_folder_hierarchy(output_subdirs: List[str], base_path: str) -> None:
    """Create each directory in *output_subdirs* if it does not exist.

    Existing non-empty directories are skipped silently.

    Args:
        output_subdirs: List of directory paths to create.
        base_path: Unused (kept for backward compatibility).
    """
    for directory in output_subdirs:
        if not os.path.exists(directory):
            os.makedirs(directory)
            continue
        if len(os.listdir(directory)) == 0:
            continue


def get_image_parameters(renamed_unaligned_path: str, aligned_path: str,
                         label: str) -> Tuple[int, Tuple[int, int], int, float, Tuple[int, int]]:
    """Collect image-stack metadata: counts, dimensions, and scale factor.

    Args:
        renamed_unaligned_path: Directory with renamed, unaligned images.
        aligned_path: Directory for cached dimension data.
        label: Prefix for cached dimension files.

    Returns:
        Tuple of ``(num_images, max_dim, num_channels, scale_factor,
        scaled_max_dim)``.
    """
    num_images = find_stack_size(renamed_unaligned_path, 'c3')
    max_dim = find_image_canvas(
        aligned_path, renamed_unaligned_path, 'dim', label)
    num_channels = len(get_filelists(
        renamed_unaligned_path, ['c1', 'c2', 'c3', 'c4']).keys())
    # HD video, scale high resolution to write video
    scale_factor = 1800. / max_dim[1]
    sc_max_dim = (int(max_dim[0] * scale_factor), int(max_dim[1] * scale_factor))
    return num_images, max_dim, num_channels, scale_factor, sc_max_dim


def draw_1mm_scale_bar(image: np.ndarray, num_pixels_in_1mm: int) -> None:
    """Draw a 1 mm scale bar on *image* (in-place, bottom-left corner).

    Args:
        image: Image array to annotate (modified in place).
        num_pixels_in_1mm: Number of pixels corresponding to 1 mm.
    """
    num_rows, num_cols = image.shape[0:2]  # (y,x)
    cv2.line(image, (int(num_cols * .05), int(num_rows * .95)), (
        int(num_cols * .05) + int(num_pixels_in_1mm), int(num_rows * .95)), (
                 255, 255, 255), int(num_cols * .005))
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(image, '1 mm', (int(num_cols * .05), int(
        num_rows * .93)), font, 1, (255, 255, 255), 2, cv2.LINE_AA)


def align_renamer(well: str, path_aligned_images: str,
                  path_montaged_images: str) -> None:
    """Rename ImageJ alignment output files to match montage naming.

    ImageJ stack alignment does not preserve filenames, so this function
    pairs aligned outputs with their montage counterparts and renames
    them with an ``_ALIGNED`` suffix.

    Args:
        well: Well identifier used to filter file lists.
        path_aligned_images: Directory containing aligned output files.
        path_montaged_images: Directory containing reference montage files.
    """
    rename_list = make_filelist(path_aligned_images, well)
    previous_list = make_filelist(path_montaged_images, well)
    prev_green = [fname for fname in previous_list if 'FITC' in fname]
    prev_red = [fname for fname in previous_list if 'RFP' in fname]
    rename_green = [fname for fname in rename_list if 'C2' in fname]
    rename_red = [fname for fname in rename_list if 'C1' in fname]
    for rname, pname in zip(rename_green, prev_green):
        rname_new = os.path.join(
            path_aligned_images, extract_file_name(
                pname) + '_ALIGNED.tif')
        os.rename(rname, rname_new)
    for rname, pname in zip(rename_red, prev_red):
        rname_new = os.path.join(
            path_aligned_images, extract_file_name(
                pname) + '_ALIGNED.tif')
        os.rename(rname, rname_new)


def zerone_normalizer(image: np.ndarray) -> np.ndarray:
    """Normalize pixel values to a fixed range [0, 240].

    Applies a linear rescale similar to ``cv2.equalizeHist`` but with
    configurable output bounds.

    Args:
        image: Input image array.

    Returns:
        Rescaled copy of the image.
    """
    copy_image = image.copy()
    # set scale
    new_img_min, new_img_max = 0, 240
    img_min, img_max = image.min(), image.max()
    # guard against divide-by-zero on a flat (constant) image
    if img_max == img_min:
        return np.full(image.shape, new_img_min, dtype=image.dtype)
    zero_one_norm = (copy_image - img_min) * (
            (new_img_max - new_img_min) / (img_max - img_min)) + new_img_min
    return zero_one_norm


def give_error_exit(error_string: str) -> None:
    """Print an error message and terminate the process.

    Args:
        error_string: Message to print and pass to ``sys.exit``.
    """
    print('---------------')
    print(error_string)
    print('---------------')
    sys.exit(error_string)


def collapse_stack_to_image(file_list: List[str],
                           collapse_type: str = 'max_proj') -> Optional[np.ndarray]:
    """Collapse an image stack to a single image via projection.

    Args:
        file_list: Paths to the images in the stack.
        collapse_type: ``'max_proj'`` for maximum intensity projection or
            ``'avg_proj'`` for average projection.

    Returns:
        Projected image array, or ``None`` if *collapse_type* is unrecognized.
    """
    start_time = datetime.datetime.utcnow()
    if collapse_type == 'max_proj':
        img_list = []
        for img_pointer in file_list:
            img = cv2.imread(img_pointer, -1)
            if img is None:
                raise ValueError(
                    f'cv2.imread returned None (missing or corrupt image): {img_pointer}')
            img_list.append(img)
        max_img = np.max(np.array(img_list), axis=0)

        end_time = datetime.datetime.utcnow()
        print('Numpy collapse run time:', end_time - start_time)
        return max_img

    if collapse_type == 'avg_proj':
        img_list = []
        for img_pointer in file_list:
            img = cv2.imread(img_pointer, -1)
            if img is None:
                raise ValueError(
                    f'cv2.imread returned None (missing or corrupt image): {img_pointer}')
            img_list.append(img)
        avg_img = np.average(np.array(img_list), axis=0)

        end_time = datetime.datetime.utcnow()
        print('Numpy collapse run time:', end_time - start_time)
        return avg_img


# ----Magick---------------------------
def collapse_stack_magically(output_file_name: str, selector: str,
                             collapse_type: str = 'max_proj',
                             verbose: bool = False) -> None:
    """Collapse an image stack using ImageMagick (faster than NumPy).

    Args:
        output_file_name: Path for the output image.
        selector: Glob pattern selecting the input images.
        collapse_type: ``'max_proj'`` or ``'avg_proj'``.
        verbose: If True, print timing and file info.
    """
    start_time = datetime.datetime.utcnow()
    if collapse_type == 'max_proj':
        magic_command = ['convert', '-maximum', selector, output_file_name]

    if collapse_type == 'avg_proj':
        magic_command = ['convert', '-average', selector, output_file_name]

    p = subprocess.Popen(magic_command, stderr=subprocess.PIPE)
    p.wait()
    if p.returncode != 0:
        stderr = p.stderr.read().decode(errors='replace') if p.stderr else ''
        raise RuntimeError(
            f'Command {magic_command} failed with return code '
            f'{p.returncode}: {stderr}')
    end_time = datetime.datetime.utcnow()

    if verbose:
        print("Output from: collapse_stack_magically")
        print("Selector", selector)
        print("Output name", output_file_name)
        print('Magic collapse run time:', end_time - start_time)


def split_stack_magically(stack_selector: str, output_filenames: str,
                          verbose: bool = False) -> None:
    """Split an image stack into individual files using ImageMagick.

    Args:
        stack_selector: Glob pattern or path selecting the stack file(s).
        output_filenames: Output path template with ``%d`` for numbering
            (e.g. ``'single%d.tif'``).
        verbose: If True, print timing and file info.
    """
    start_time = datetime.datetime.utcnow()

    magic_command = ['convert', stack_selector, output_filenames]
    p = subprocess.Popen(magic_command, stderr=subprocess.PIPE)
    p.wait()
    if p.returncode != 0:
        stderr = p.stderr.read().decode(errors='replace') if p.stderr else ''
        raise RuntimeError(
            f'Command {magic_command} failed with return code '
            f'{p.returncode}: {stderr}')

    end_time = datetime.datetime.utcnow()

    if verbose:
        print("Selector", stack_selector)
        print("Output name", output_filenames)
        print('Magic stack images run time:', end_time - start_time)


def make_stack_magically(selector: str, output_filename: str,
                         verbose: bool = False) -> None:
    """Combine individual images into a stack using ImageMagick.

    Args:
        selector: Glob pattern selecting the input images.
        output_filename: Path for the output stack file.
        verbose: If True, print timing and file info.
    """
    start_time = datetime.datetime.utcnow()

    magic_command = ['convert', selector, output_filename]
    p = subprocess.Popen(magic_command, stderr=subprocess.PIPE)
    p.wait()
    if p.returncode != 0:
        stderr = p.stderr.read().decode(errors='replace') if p.stderr else ''
        raise RuntimeError(
            f'Command {magic_command} failed with return code '
            f'{p.returncode}: {stderr}')

    end_time = datetime.datetime.utcnow()
    if verbose:
        print("Selector", selector)
        print("Output name", output_filename)
        print('Magic stack images run time:', end_time - start_time)


def make_unstacking_folder(path: str) -> None:
    """Create an ``Unstacked_Collector`` subdirectory in *path*.

    Args:
        path: Parent directory.
    """
    directory = os.path.join(path, 'Unstacked_Collector')
    if not os.path.exists(directory):
        os.makedirs(directory)


def unstack_stacks(stack_pointer: str) -> None:
    """Split a stack image into individual files in the ``Unstacked_Collector`` folder.

    Args:
        stack_pointer: Path to the stack image file.
    """
    path = os.path.dirname(stack_pointer)
    directory = os.path.join(path, 'Unstacked_Collector')
    output_filenames = os.path.join(directory, 'unstacked-%d.tif')
    split_stack_magically(stack_pointer, output_filenames)


def kill_unstacking_folder(path: str) -> None:
    """Remove the ``Unstacked_Collector`` subdirectory and its contents.

    Args:
        path: Parent directory containing the folder to remove.
    """
    directory = os.path.join(path, 'Unstacked_Collector')
    if os.path.exists(directory):
        shutil.rmtree(directory)


def create_cleanup_unstacked(path: str) -> None:
    """Unstack all PID stack files in *path*, then remove the temp folder.

    Iterates over stack files, splits each into singles in a temporary
    directory, then removes the temporary directory.

    Args:
        path: Directory containing stack files.
    """
    stack_files = make_filelist(path, 'PID')
    make_unstacking_folder(path)
    for stack in stack_files:
        unstack_stacks(stack)
    kill_unstacking_folder(path)


def get_frame_files(selected_image_list: List[str], frame: Union[str, int],
                    token_num: int) -> List[str]:
    """Filter a file list to those matching a specific frame token value.

    Args:
        selected_image_list: Full paths to image files.
        frame: Frame identifier (burst or depth value) to match.
        token_num: Underscore-delimited token index where the frame value
            appears in the filename.

    Returns:
        Sorted list of matching file paths.
    """
    path_images = os.path.dirname(selected_image_list[0])

    # Get all the files and tokenize them
    selected_image_list = [os.path.basename(fname) for fname in selected_image_list]
    selected_image_list = [fname.split('_') for fname in selected_image_list]
    # Filter the files and re-join the tokens
    frame_selected_files = [tok_fname for tok_fname in selected_image_list if str(frame) == tok_fname[token_num]]
    frame_selected_files = ['_'.join(tok_fname) for tok_fname in frame_selected_files]
    frame_selected_files = [os.path.join(path_images, fname) for fname in frame_selected_files]
    frame_selected_files.sort(key=numericalSort)

    return frame_selected_files


def get_selected_files(path_images: str, well: str, timepoint: str,
                       channel: str, robonum: int = 0) -> List[str]:
    """Filter image files by well, timepoint, and channel.

    Args:
        path_images: Directory containing ``.tif`` images.
        well: Well identifier (token index 4).
        timepoint: Timepoint identifier (token index 2).
        channel: Channel identifier.
        robonum: Robot number controlling channel token position.

    Returns:
        Sorted list of matching file paths.
    """
    ch_token_pos = get_channel_token(robonum)
    # Get all the files and tokenize them
    all_files = [fname for fname in os.listdir(path_images) if '.tif' in fname]
    all_files = [fname.split('_') for fname in all_files]
    # Filter the files and re-join the tokens
    selected_files = [tok_fname for tok_fname in all_files if
                      well == tok_fname[4] and timepoint == tok_fname[2] and channel == tok_fname[ch_token_pos]]
    selected_files = ['_'.join(tok_fname) for tok_fname in selected_files]
    selected_files = [os.path.join(path_images, fname) for fname in selected_files]
    selected_files.sort(key=numericalSort)

    return selected_files


def get_filename(tokenized_files: List[List[str]], well: str, timepoint: str,
                 channel: str, robonum: int = 0) -> List[str]:
    """Filter pre-tokenized file records by well, timepoint, and channel.

    Unlike :func:`get_selected_files`, this works with pre-tokenized input
    (from :func:`tokenize_files`) and supports files in subdirectories.

    Args:
        tokenized_files: Nested list where each element is
            ``[full_path, token0, token1, ...]``.
        well: Well identifier.
        timepoint: Timepoint identifier.
        channel: Channel identifier.
        robonum: Robot number controlling channel token position.

    Returns:
        List of full file paths matching the criteria.
    """
    # Get channel token standard
    ch_token_pos = get_channel_token(robonum)
    # Filter the files (0 position is the full file path)
    selected_files = [img for img in tokenized_files if
                      well == img[5] and timepoint == img[3] and channel == img[ch_token_pos + 1]]
    # Return full file paths
    return [fname[0] for fname in selected_files]


def tokenize_files(paths: List[str]) -> List[List[str]]:
    """Split each ``.tif`` filename on underscores, prepending the full path.

    Args:
        paths: List of file paths.

    Returns:
        Nested list where each element is ``[full_path, token0, token1, ...]``.
    """
    tokenized_files = []
    for p in paths:
        if '.tif' in os.path.basename(p):
            tokens = [p] + os.path.basename(p).split('_')
            tokenized_files.append(tokens)

    return tokenized_files


def make_selector_from_tokens(robonum: int, well: str = '*', time: str = '*',
                              channel: str = '*', panel: str = '*') -> str:
    """Build a glob selector string from filename token values.

    Args:
        robonum: Robot number controlling channel token position.
        well: Well identifier or ``'*'`` for all.
        time: Timepoint or ``'*'`` for all.
        channel: Channel or ``'*'`` for all.
        panel: Panel index or ``'*'`` for all.

    Returns:
        Glob-style selector string (e.g. ``'*_T0_*_*_A1_1_GFP_*_*_*.tif'``).
    """
    ch_token_pos = get_channel_token(robonum)
    stokens = ['*'] * 10
    stokens[2] = time
    stokens[ch_token_pos] = channel
    stokens[4] = well
    stokens[5] = str(panel)
    stokens[9] = '*.tif'
    selector = '_'.join(stokens)
    return selector


def get_channel_token(robonum: int, light_path: str = 'epi',
                      verbose: bool = False) -> int:
    """Return the underscore-delimited token index of the channel field.

    The position depends on the robot and light-path configuration.

    Args:
        robonum: Robot number (0, 3, or 4).
        light_path: ``'epi'`` or ``'confocal'`` (only relevant for Robo 4).
        verbose: If True, print the resolved position.

    Returns:
        Token index for the channel identifier.

    Raises:
        AssertionError: If *robonum* or *light_path* is unsupported.
    """
    robonum = int(robonum)
    if robonum == 3 or robonum == 0:
        ch_pos = 6
    elif robonum == 4:
        if light_path == 'epi':
            ch_pos = 6
        elif light_path == 'confocal':
            ch_pos = -4
        else:
            assert light_path == 'epi' or light_path == 'confocal', 'Token position for robo4 light path not known.'
    else:
        assert robonum == 3 or robonum == 4 or robonum == 0, 'Token position for this robo not known.'

    if verbose:
        print('Channel position:', ch_pos)
    return ch_pos


def make_selector(iterator: str = '', well: str = '', timepoint: str = '',
                  channel: str = '', frame: str = '', verbose: bool = False,
                  robo0: bool = True) -> str:
    """Construct a glob selector string for matching image files.

    Args:
        iterator: Iterator type (``'TimeBursts'``, ``'ZDepths'``, or empty).
        well: Well identifier.
        timepoint: Timepoint identifier.
        channel: Channel identifier.
        frame: Frame value for bursts or depths.
        verbose: If True, print debug details.
        robo0: Whether to use Robo0 naming convention.

    Returns:
        Glob-style selector string.
    """

    burst = ''
    depth = ''

    if iterator == 'TimeBursts':
        burst = frame
    elif iterator == 'ZDepths':
        depth = frame
    else:
        burst = ''
        depth = ''

    selector = timepoint + '_*' + burst + '_*' + well + '_' + '*' + '_' + channel + '*' + str(depth)

    if verbose == True:
        print('Depth:', depth, 'Burst:', burst, 'Frame:', frame)
        print('Set selector:', selector)
    return selector


def set_iterator(var_dict: Dict[str, Any]) -> Tuple[Optional[str], list]:
    """Determine whether the experiment uses burst or depth iterations.

    Args:
        var_dict: Experiment variable dictionary with ``'Bursts'``,
            ``'BurstIDs'``, and ``'Depths'`` keys.

    Returns:
        Tuple of ``(iterator_name, iter_list)`` where *iterator_name* is
        ``'TimeBursts'``, ``'ZDepths'``, or ``None``.
    """
    if len(var_dict['Bursts']) > 1:
        iter_list = var_dict['BurstIDs']
        iterator = 'TimeBursts'
    elif len(var_dict["Depths"]) > 1:
        iter_list = var_dict["Depths"]
        iterator = 'ZDepths'
    else:
        iter_list = []
        iterator = None

    # iter_list.sort(key=numericalSort)
    return iterator, iter_list


def order_wells_correctly(value: str) -> Tuple[str, int]:
    """Key function for sorting well IDs in natural order (A1, A2, ... A11).

    Args:
        value: Well identifier string (e.g. ``'A1'``, ``'B12'``).

    Returns:
        Tuple of ``(letter, number)`` for comparison.
    """
    return value[0], int(value[1:])


def get_all_files(input_path: str, verbose: bool = False) -> List[str]:
    """List all PID ``.tif`` images in *input_path*, excluding fiduciary files.

    Args:
        input_path: Directory to search.
        verbose: If True, print the count of files found.

    Returns:
        Sorted list of matching file paths.
    """
    all_files = make_filelist(input_path, 'PID')
    all_files = [afile for afile in all_files if 'FIDUCIARY' not in afile]
    all_files = [afile for afile in all_files if '.tif' in afile]
    if verbose:
        print('Number of image files:', len(all_files))
    return all_files


def get_all_files_all_subdir(input_path: str, verbose: bool = False) -> List[str]:
    """Like :func:`get_all_files` but also searches subdirectories.

    Args:
        input_path: Root directory to walk.
        verbose: If True, print discovery details.

    Returns:
        Flat list of matching PID ``.tif`` file paths.
    """
    all_files = make_filelist_wells(input_path, 'PID', verbose=verbose)
    all_files = [afile for afile in all_files if 'FIDUCIARY' not in afile]
    all_files = [afile for afile in all_files if '.tif' in afile]
    if verbose:
        print('Number of image files:', len(all_files))
    return all_files


def get_wells(all_files: List[str], verbose: bool = False) -> List[str]:
    """Extract the unique, naturally-sorted well identifiers from filenames.

    Args:
        all_files: List of image file paths.
        verbose: If True, print the discovered wells.

    Returns:
        Sorted list of unique well strings.
    """
    wells = []
    for one_file in all_files:
        if os.path.basename(one_file).split('_')[4] == '0':
            print(os.path.basename(one_file))
        well = os.path.basename(one_file).split('_')[4]
        wells.append(well)
    wells = list(set(wells))
    wells.sort(key=order_wells_correctly)
    if verbose:
        print('Wells:', wells)
    return wells


def get_well_panel(all_files: List[str], verbose: bool = False) -> List[int]:
    """Extract the unique, sorted well-panel indices from filenames.

    Args:
        all_files: List of image file paths.
        verbose: If True, print the discovered panels.

    Returns:
        Sorted list of unique panel indices (as integers).
    """
    well_panels = []
    for one_file in all_files:
        panel_index = os.path.basename(one_file).split('_')[5]
        well_panels.append(panel_index)
    well_panels = list(set([int(panel) for panel in well_panels]))
    well_panels.sort()
    if verbose:
        print('Well panels:', well_panels)
    return well_panels


def get_timepoints(all_files: List[str], verbose: bool = False) -> List[str]:
    """Extract the unique, naturally-sorted timepoint identifiers from filenames.

    Args:
        all_files: List of image file paths.
        verbose: If True, print the discovered timepoints.

    Returns:
        Sorted list of unique timepoint strings.
    """
    timepoints = []
    for one_file in all_files:
        time = os.path.basename(one_file).split('_')[2]
        timepoints.append(time)
    timepoints = sorted(list(set(timepoints)))
    timepoints.sort(key=order_wells_correctly)
    if verbose:
        print('Timepoints:', timepoints)
    return timepoints


def get_channels(all_files: List[str], robonum: int, light_path: str = 'epi',
                 verbose: bool = False) -> List[str]:
    """Extract unique channel identifiers from filenames using robot-specific parsing.

    Args:
        all_files: List of image file paths.
        robonum: Robot number (0, 3, or 4).
        light_path: ``'epi'`` or ``'confocal'`` (Robo 4 only).
        verbose: If True, print the discovered channels.

    Returns:
        List of unique channel strings.
    """
    robonum = int(robonum)
    channels = []
    for one_file in all_files:
        if robonum == 3 or robonum == 0:
            channel = os.path.basename(one_file).split('_')[6].split('.')[0]
        elif robonum == 4:
            if light_path == 'epi':
                channel = os.path.basename(one_file).split('_')[6].split('.')[0]
            elif light_path == 'confocal':
                channel = os.path.basename(one_file).split('_')[-4].split('.')[0]
            else:
                assert light_path == 'epi' or light_path == 'confocal', 'No treatment for this light path yet.'
        else:
            assert robonum == 3 or robonum == 4 or robonum == 0, 'No treatment for this robo yet'

        channels.append(channel)

    channels = list(set(channels))
    if verbose:
        print('Channels:', channels)
    return channels


def get_channels_from_user(all_files: List[str], channel_token: int,
                           verbose: bool = False) -> List[str]:
    """Extract unique channel identifiers using a user-specified token index.

    Brightfield channels are excluded from the result.

    Args:
        all_files: List of image file paths.
        channel_token: Underscore-delimited token index for the channel.
        verbose: If True, print the discovered channels.

    Returns:
        List of unique non-Brightfield channel strings.
    """

    channel_token = int(channel_token)
    channels = []
    for one_file in all_files:
        channel = os.path.basename(one_file).split('_')[channel_token].split('.')[0]
        if 'Brightfield' not in channel:
            channels.append(channel)

    channels = list(set(channels))
    if verbose:
        print('Channels:', channels)
    return channels


def get_ref_channel(morph_channel: str, channels: List[str],
                    verbose: bool = False) -> str:
    """Find the full channel name matching a case-insensitive substring.

    Args:
        morph_channel: Substring to search for (e.g. ``'gfp'``).
        channels: Available channel names.
        verbose: If True, print the matched channel.

    Returns:
        The unique matching channel name.

    Raises:
        AssertionError: If zero or multiple channels match.
    """
    morphology_channel = [ch for ch in channels if morph_channel.lower() in ch.lower()]
    assert len(morphology_channel) > 0, 'Your channel string (%s) was not found.' % morph_channel
    assert len(morphology_channel) == 1, 'Your channel string (%s) was not unique.' % morph_channel
    morphology_channel = morphology_channel[0]
    if verbose:
        print('Morphology channel:', morphology_channel)
    return morphology_channel


def get_plate_id(all_files: List[str], verbose: bool = False) -> str:
    """Extract the plate ID from the first file's name (tokens 0 and 1).

    Args:
        all_files: List of image file paths.
        verbose: If True, print the plate ID.

    Returns:
        Plate ID string (e.g. ``'PID_12345'``).
    """
    first_data_file = os.path.basename(all_files[0])
    plateID_tokens = first_data_file.split('_')[0:2]
    plateID = '_'.join(plateID_tokens)
    if verbose:
        print('PlateID:', plateID)
    return plateID


def get_bursts(all_files: List[str], verbose: bool = False) -> List[str]:
    """Extract unique, sorted burst identifiers from filenames (token 3).

    Args:
        all_files: List of image file paths.
        verbose: If True, print the burst IDs.

    Returns:
        Naturally-sorted list of unique burst ID strings.
    """
    burstIDs = list(set(
        [os.path.basename(fname).split('_')[3] for fname in all_files]))
    burstIDs.sort(key=numericalSort)
    if verbose:
        print('Burst IDs:', burstIDs)
    return burstIDs


def get_burst_iter(all_files: List[str], verbose: bool = False) -> List[str]:
    """Extract burst frame sub-identifiers (the part after the dash).

    Args:
        all_files: List of image file paths.
        verbose: If True, print the burst frames.

    Returns:
        Sorted list of unique burst frame strings (e.g. ``['-1', '-2']``).
    """
    burstIDs = get_bursts(all_files, verbose=False)
    burst_frames = []
    for burst in burstIDs:
        try:
            burst_frames.append('-' + burst.split('-')[1])
        except IndexError:
            continue
    burst_frames = list(set(burst_frames))
    burst_frames.sort(key=numericalSort)
    if verbose:
        print('Burst frames:', burst_frames)
    return burst_frames


def get_depths(all_files: List[str], robonum: int,
               verbose: bool = False) -> list:
    """Extract unique Z-depth values from filenames using robot-specific parsing.

    Args:
        all_files: List of image file paths.
        robonum: Robot number (0, 3, or 4).
        verbose: If True, print the depth values.

    Returns:
        Sorted list of depth values (empty if only one depth exists).
    """
    # Depths are  stored as stacks for robo4

    robonum = int(robonum)
    if robonum == 3:
        depths = []
    elif robonum == 4:
        depths = list(set([int(
            os.path.basename(fname).split('_')[-3]) for fname in all_files]))
    elif robonum == 0:
        try:
            depths = list(set([
                os.path.basename(fname).split('_')[8] for fname in all_files]))
        except IndexError:
            depths = []

    else:
        assert robonum == 3 or robonum == 4 or robonum == 0, 'Unknown Robo number.'

    if len(set(depths)) <= 1:
        depths = []
    else:
        depths.sort()

    if verbose == True:
        print('Depths:', depths)

    return depths


def get_iter_from_user(comma_list_without_spaces: str, iter_value: str = '',
                       verbose: bool = False) -> List[str]:
    """Parse a user-supplied comma-separated list, expanding any ranges.

    Supports individual items (``'A1,A3'``) and alphanumeric ranges
    (``'A1-B3'``).  Ranges expand across both letters and numbers:
    ``'A1-B3'`` yields ``A1, A2, A3, B1, B2, B3``.  Leading zeros in
    numeric parts are preserved for single-digit values.

    Args:
        comma_list_without_spaces: Comma-separated string of items and/or
            ranges (e.g. ``'A1,A3-B2'``).
        iter_value: Label for verbose output (e.g. ``'wells'``).
        verbose: If True, print the resolved list.

    Returns:
        Deduplicated, naturally-sorted list of identifiers.
    """
    # Normalize: strip spaces, uppercase everything
    comma_list_without_spaces = comma_list_without_spaces.replace(" ", "").upper()
    user_chosen_iter = comma_list_without_spaces.split(',')

    # Identify range entries (items containing a dash, e.g. 'A1-B3')
    user_range = [user_iter for user_iter in user_chosen_iter if len(user_iter.split('-')) > 1]

    # Build the full alphabet for letter-range expansion
    all_letters = sorted(map(chr, range(65, 91)))

    if len(user_range) > 0:
        for iter_range in user_range:
            # Remove the range token; we will expand it into individual items
            user_chosen_iter.remove(iter_range)

            # Split range into start and end (e.g. 'A1' and 'B3')
            end_values = iter_range.split('-')

            # Determine the letter span (e.g. A..B)
            start_letter = all_letters.index(end_values[0][0])
            end_letter = all_letters.index(end_values[1][0])
            letter_range = all_letters[start_letter:end_letter + 1]

            # Determine the numeric span (e.g. 1..3)
            number_start = end_values[0][1:]
            number_end = end_values[1][1:]

            # Detect leading-zero formatting (e.g. '01')
            leading_zero = re.match(r'0\d+', str(number_start))
            num_range = range(int(number_start), int(number_end) + 1)

            # Expand the full Cartesian product of letters x numbers
            complete_range = []
            for letter in letter_range:
                for num in num_range:
                    # Preserve leading zero for single-digit numbers if original had one
                    if leading_zero and num < 10:
                        num = '0' + str(num)
                    complete_range.append(letter + str(num))

            # Include both endpoints and the expanded range
            user_chosen_iter.append(end_values[0])
            user_chosen_iter.extend(complete_range)
            user_chosen_iter.append(end_values[1])

    # Deduplicate and sort naturally
    user_chosen_iter = list(set(user_chosen_iter))
    user_chosen_iter.sort(key=numericalSort)
    if verbose:
        print('Your selected ' + iter_value + ':', user_chosen_iter)
    return user_chosen_iter


def overwrite_io_paths(var_dict: Dict[str, Any], output_path: str,
                       verbose: bool = False) -> Dict[str, Any]:
    """Chain pipeline steps by redirecting input/output paths in *var_dict*.

    Sets ``InputPath`` to the previous step's ``OutputPath``, then updates
    ``OutputPath`` to the new *output_path*.

    Args:
        var_dict: Mutable experiment variable dictionary.
        output_path: New output directory for the current step.
        verbose: If True, print before/after paths.

    Returns:
        The updated *var_dict* (modified in place).
    """

    if verbose:
        print('Initial input (passed var_dict)', var_dict['InputPath'])
        print('Initial output (passed var_dict)', var_dict["OutputPath"])

    input_path = var_dict["OutputPath"]
    var_dict['InputPath'] = input_path
    var_dict["OutputPath"] = output_path

    if verbose:
        print('Final input (new var_dict)', var_dict['InputPath'])
        print('Final output (new var_dict)', var_dict["OutputPath"])

    return var_dict


def save_user_args_to_csv(arg_parser: Any, save_path: str, module_string: str,
                          verbose: bool = True) -> None:
    """Serialize argparse arguments to a CSV file for reproducibility.

    Args:
        arg_parser: Parsed ``argparse.Namespace`` object.
        save_path: Directory to write the CSV into.
        module_string: Module name used as the CSV filename stem.
        verbose: If True, print the output path.
    """
    with open(os.path.join(save_path, module_string + '.csv'), 'w') as params_txt:
        for arg, user_val in vars(arg_parser).items():
            if arg != 'input_dict' and arg != 'output_dict' and arg != 'outfile':
                params_txt.write(','.join([arg, str(user_val)]) + '\n')
    if verbose:
        print('Parameter output written to', os.path.join(save_path, module_string + '.csv'))


def update_timestring() -> str:
    """Generate a timestamp string in ``YYYY_MM_DD_HH_MM_SS`` format.

    Returns:
        Formatted timestamp of the current local time.
    """
    now = datetime.datetime.now()
    timestring = '%.4d_%.2d_%.2d_%.2d_%.2d_%.2d' % (now.year, now.month, now.day, now.hour, now.minute, now.second)
    return timestring