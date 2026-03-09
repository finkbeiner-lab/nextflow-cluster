"""Helper functions for montage-based segmentation mask I/O and cell data persistence.

Provides utilities for saving segmentation masks as TIFF files and updating
celldata/intensitycelldata tables in the PostgreSQL database. Includes both
single-row and batch-optimized variants for database operations.
"""

import os
import uuid
import gc
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import tifffile

from sql import Database


def save_mask(
    mask: np.ndarray,
    image_file: str,
    savedir: str,
) -> str:
    """Save a segmentation mask as an uncompressed 16-bit TIFF.

    The output filename is derived from *image_file* by stripping the TIFF
    extension and appending ``_ENCODED.tif``.

    Args:
        mask: 2-D array of labelled regions (values are cast to uint16).
        image_file: Original image path; only the basename is used.
        savedir: Directory in which to write the mask file.

    Returns:
        Absolute path to the saved mask file.
    """
    name = os.path.basename(image_file)
    name = name.split('.t')[0]  # Strip .tif / .tiff suffix
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    savepath = os.path.join(savedir, name + '_ENCODED.tif')

    # uint16 with no compression for maximum OpenCV compatibility
    mask_uint16 = mask.astype(np.uint16)
    tifffile.imwrite(savepath, mask_uint16, compression=None)
    return savepath


def save_masks_batch(
    masks_data: List[Tuple[np.ndarray, str]],
    savedir: str,
) -> List[str]:
    """Save multiple segmentation masks in one call for better I/O throughput.

    Args:
        masks_data: List of (mask_array, source_image_path) tuples.
        savedir: Directory in which to write all mask files.

    Returns:
        List of absolute paths to the saved mask files, in the same order as
        *masks_data*.
    """
    if not os.path.exists(savedir):
        os.makedirs(savedir)

    saved_paths: List[str] = []
    for mask, image_file in masks_data:
        name = os.path.basename(image_file)
        name = name.split('.t')[0]
        savepath = os.path.join(savedir, name + '_ENCODED.tif')
        mask_uint16 = mask.astype(np.uint16)
        tifffile.imwrite(savepath, mask_uint16, compression=None)
        saved_paths.append(savepath)

    return saved_paths


def convert_numpy_types(
    data: Union[Dict[str, Any], List[Dict[str, Any]], Any],
) -> Union[Dict[str, Any], List[Dict[str, Any]], Any]:
    """Recursively convert numpy scalar types to native Python types.

    This is necessary before passing data to SQLAlchemy, which does not
    accept numpy scalars in INSERT statements.

    Args:
        data: A dict, list of dicts, or any other value.

    Returns:
        The same structure with numpy scalars replaced by their Python
        equivalents (via ``np.generic.item()``).
    """
    if isinstance(data, list):
        return [convert_numpy_types(d) for d in data]

    if isinstance(data, dict):
        return {
            k: (v.item() if isinstance(v, np.generic) else v)
            for k, v in data.items()
        }

    return data


# ---------------------------------------------------------------------------
# Single-row database update functions
# ---------------------------------------------------------------------------
# NOTE: ``update_celldataand_intensitycelldata`` (no underscore between
# "celldata" and "and") and ``update_celldata_and_intensitycelldata`` are
# functionally identical.  The first is the original legacy name with a typo;
# the second was added later with the corrected spelling.  Both are kept for
# backward compatibility with callers that may reference either name.
# ---------------------------------------------------------------------------


def update_celldataand_intensitycelldata(
    row: Any,
    props_df: pd.DataFrame,
    Db: Database,
) -> None:
    """Insert cell and intensity-cell records for a single tile (legacy typo name).

    This function has the same behaviour as
    :func:`update_celldata_and_intensitycelldata`.  It exists only because
    older callers reference the misspelled name (missing underscores around
    "and").  Prefer the correctly-spelled variant for new code.

    Args:
        row: A tiledata row (namedtuple-like) with attributes
            ``experimentdata_id``, ``welldata_id``, ``id``, and
            ``channeldata_id``.
        props_df: DataFrame of regionprops for the segmented cells.
        Db: Active ``Database`` connection used for reads and writes.
    """
    celldata_dcts: List[Dict[str, Any]] = []
    check_celldata_dcts: List[Dict[str, Any]] = []
    intensitycelldata_dcts: List[Dict[str, Any]] = []
    for i, prow in props_df.iterrows():
        ident = uuid.uuid4()
        check_celldata_dcts.append(dict(experimentdata_id=row.experimentdata_id,
                                        welldata_id=row.welldata_id,
                                        tiledata_id=row.id,))

        celldata_dcts.append(
            dict(id=ident,
                 experimentdata_id=row.experimentdata_id,
                 welldata_id=row.welldata_id,
                 tiledata_id=row.id,
                 randomcellid_montage=prow.label,
                 centroid_x=prow['centroid_weighted-1'],
                 centroid_y=prow['centroid_weighted-0'],
                 perimeter=prow.perimeter,
                 area=prow.area,
                 solidity=prow.solidity,
                 extent=prow.extent,
                 eccentricity=prow.eccentricity,
                 axis_major_length=prow.axis_major_length,
                 axis_minor_length=prow.axis_minor_length,
                 ))
        intensitycelldata_dcts.append(dict(experimentdata_id=row.experimentdata_id,
                                           welldata_id=row.welldata_id,
                                           tiledata_id=row.id,
                                           celldata_id=ident,
                                           channeldata_id=row.channeldata_id,
                                           intensity_max=prow.intensity_max,
                                           intensity_mean=prow.intensity_mean,
                                           intensity_min=prow.intensity_min))
    # Delete existing cells for this tile before inserting new ones
    if not props_df.empty:
        for check_dct in check_celldata_dcts:
            Db.delete_based_on_duplicate_name(tablename='celldata', kwargs=check_dct)
            Db.delete_based_on_duplicate_name(tablename='intensitycelldata', kwargs=check_dct)
        celldata_dcts = convert_numpy_types(celldata_dcts)
        Db.add_row(tablename='celldata', dct=celldata_dcts)
        intensitycelldata_dcts = convert_numpy_types(intensitycelldata_dcts)
        Db.add_row(tablename='intensitycelldata', dct=intensitycelldata_dcts)


def batch_update_celldataand_intensitycelldata(
    batch_data: List[Tuple[Any, pd.DataFrame]],
    Db: Database,
) -> None:
    """Batch-insert cell and intensity-cell records for many tiles (legacy typo name).

    Compared to calling :func:`update_celldataand_intensitycelldata` in a
    loop, this function accumulates all rows first and performs chunked
    deletes and inserts to reduce database round-trips.

    Args:
        batch_data: List of (tiledata_row, regionprops_DataFrame) tuples.
        Db: Active ``Database`` connection.
    """
    if not batch_data:
        return

    # Early exit when every DataFrame is empty
    total_cells: int = sum(len(props_df) for _, props_df in batch_data if not props_df.empty)
    if total_cells == 0:
        return

    celldata_batch: List[Dict[str, Any]] = []
    intensitycelldata_batch: List[Dict[str, Any]] = []

    # Collect unique tile IDs for bulk deletion
    tiledata_ids: set = set()

    for row, props_df in batch_data:
        if props_df.empty:
            continue

        tiledata_ids.add(row.id)

        for _, prow in props_df.iterrows():
            ident = uuid.uuid4()

            celldata_batch.append({
                'id': ident,
                'experimentdata_id': row.experimentdata_id,
                'welldata_id': row.welldata_id,
                'tiledata_id': row.id,
                'randomcellid_montage': prow.label,
                'centroid_x': prow['centroid_weighted-1'],
                'centroid_y': prow['centroid_weighted-0'],
                'perimeter': prow.perimeter,
                'area': prow.area,
                'solidity': prow.solidity,
                'extent': prow.extent,
                'eccentricity': prow.eccentricity,
                'axis_major_length': prow.axis_major_length,
                'axis_minor_length': prow.axis_minor_length,
            })

            intensitycelldata_batch.append({
                'experimentdata_id': row.experimentdata_id,
                'welldata_id': row.welldata_id,
                'tiledata_id': row.id,
                'celldata_id': ident,
                'channeldata_id': row.channeldata_id,
                'intensity_max': prow.intensity_max,
                'intensity_mean': prow.intensity_mean,
                'intensity_min': prow.intensity_min
            })

    # Batch delete existing data for all affected tiles (chunked)
    if tiledata_ids:
        chunk_size = 50
        tiledata_list = list(tiledata_ids)

        for i in range(0, len(tiledata_list), chunk_size):
            chunk = tiledata_list[i:i+chunk_size]
            for tiledata_id in chunk:
                Db.delete_based_on_duplicate_name(tablename='celldata',
                                                  kwargs={'tiledata_id': tiledata_id})
                Db.delete_based_on_duplicate_name(tablename='intensitycelldata',
                                                  kwargs={'tiledata_id': tiledata_id})

    # Batch insert new data (chunked to limit memory)
    if celldata_batch:
        celldata_batch = convert_numpy_types(celldata_batch)
        chunk_size = 1000
        for i in range(0, len(celldata_batch), chunk_size):
            chunk = celldata_batch[i:i+chunk_size]
            Db.add_row(tablename='celldata', dct=chunk)

    if intensitycelldata_batch:
        intensitycelldata_batch = convert_numpy_types(intensitycelldata_batch)
        chunk_size = 1000
        for i in range(0, len(intensitycelldata_batch), chunk_size):
            chunk = intensitycelldata_batch[i:i+chunk_size]
            Db.add_row(tablename='intensitycelldata', dct=chunk)

    # Explicitly free large lists
    del celldata_batch, intensitycelldata_batch
    gc.collect()


def update_celldata_and_intensitycelldata(
    row: Any,
    props_df: pd.DataFrame,
    Db: Database,
) -> None:
    """Insert cell and intensity-cell records for a single tile.

    Deletes any existing celldata / intensitycelldata rows for the tile
    identified by *row*, then inserts new rows derived from *props_df*.

    This is the correctly-spelled equivalent of
    :func:`update_celldataand_intensitycelldata`.

    Args:
        row: A tiledata row (namedtuple-like) with attributes
            ``experimentdata_id``, ``welldata_id``, ``id``, and
            ``channeldata_id``.
        props_df: DataFrame of regionprops for the segmented cells.
        Db: Active ``Database`` connection.
    """
    celldata_dcts: List[Dict[str, Any]] = []
    check_celldata_dcts: List[Dict[str, Any]] = []
    intensitycelldata_dcts: List[Dict[str, Any]] = []
    for i, prow in props_df.iterrows():
        ident = uuid.uuid4()
        check_celldata_dcts.append(dict(experimentdata_id=row.experimentdata_id,
                                        welldata_id=row.welldata_id,
                                        tiledata_id=row.id,))

        celldata_dcts.append(
            dict(id=ident,
                 experimentdata_id=row.experimentdata_id,
                 welldata_id=row.welldata_id,
                 tiledata_id=row.id,
                 randomcellid_montage=prow.label,
                 centroid_x=prow['centroid_weighted-1'],
                 centroid_y=prow['centroid_weighted-0'],
                 perimeter=prow.perimeter,
                 area=prow.area,
                 solidity=prow.solidity,
                 extent=prow.extent,
                 eccentricity=prow.eccentricity,
                 axis_major_length=prow.axis_major_length,
                 axis_minor_length=prow.axis_minor_length,
                 ))
        intensitycelldata_dcts.append(dict(experimentdata_id=row.experimentdata_id,
                                           welldata_id=row.welldata_id,
                                           tiledata_id=row.id,
                                           celldata_id=ident,
                                           channeldata_id=row.channeldata_id,
                                           intensity_max=prow.intensity_max,
                                           intensity_mean=prow.intensity_mean,
                                           intensity_min=prow.intensity_min))
    # Delete existing cells for this tile before inserting new ones
    if not props_df.empty:
        for check_dct in check_celldata_dcts:
            Db.delete_based_on_duplicate_name(tablename='celldata', kwargs=check_dct)
            Db.delete_based_on_duplicate_name(tablename='intensitycelldata', kwargs=check_dct)
        celldata_dcts = convert_numpy_types(celldata_dcts)
        Db.add_row(tablename='celldata', dct=celldata_dcts)
        intensitycelldata_dcts = convert_numpy_types(intensitycelldata_dcts)
        Db.add_row(tablename='intensitycelldata', dct=intensitycelldata_dcts)


def batch_update_celldata_and_intensitycelldata(
    batch_data: List[Tuple[Any, pd.DataFrame]],
    Db: Database,
) -> None:
    """Batch-insert cell and intensity-cell records for many tiles.

    Accumulates all celldata and intensitycelldata rows, performs bulk
    deletes by ``tiledata_id``, then bulk inserts the new records.

    Args:
        batch_data: List of (tiledata_row, regionprops_DataFrame) tuples.
        Db: Active ``Database`` connection.
    """
    if not batch_data:
        return

    tiledata_ids: set = set()
    celldata_batch: List[Dict[str, Any]] = []
    intensitycelldata_batch: List[Dict[str, Any]] = []

    for row, props_df in batch_data:
        tiledata_ids.add(row.id)

        for _, prow in props_df.iterrows():
            ident = uuid.uuid4()

            celldata_batch.append({
                'id': ident,
                'experimentdata_id': row.experimentdata_id,
                'welldata_id': row.welldata_id,
                'tiledata_id': row.id,
                'randomcellid_montage': prow.label,
                'centroid_x': prow['centroid_weighted-1'],
                'centroid_y': prow['centroid_weighted-0'],
                'perimeter': prow.perimeter,
                'area': prow.area,
                'solidity': prow.solidity,
                'extent': prow.extent,
                'eccentricity': prow.eccentricity,
                'axis_major_length': prow.axis_major_length,
                'axis_minor_length': prow.axis_minor_length,
            })

            intensitycelldata_batch.append({
                'experimentdata_id': row.experimentdata_id,
                'welldata_id': row.welldata_id,
                'tiledata_id': row.id,
                'celldata_id': ident,
                'channeldata_id': row.channeldata_id,
                'intensity_max': prow.intensity_max,
                'intensity_mean': prow.intensity_mean,
                'intensity_min': prow.intensity_min
            })

    # Delete existing data for all affected tiles
    for tiledata_id in tiledata_ids:
        Db.delete_based_on_duplicate_name(tablename='celldata',
                                          kwargs={'tiledata_id': tiledata_id})
        Db.delete_based_on_duplicate_name(tablename='intensitycelldata',
                                          kwargs={'tiledata_id': tiledata_id})

    # Bulk insert new records
    if celldata_batch:
        celldata_batch = convert_numpy_types(celldata_batch)
        Db.add_row(tablename='celldata', dct=celldata_batch)

    if intensitycelldata_batch:
        intensitycelldata_batch = convert_numpy_types(intensitycelldata_batch)
        Db.add_row(tablename='intensitycelldata', dct=intensitycelldata_batch)


def batch_update_celldata_and_intensitycelldata_no_delete(
    batch_data: List[Tuple[Any, pd.DataFrame]],
    Db: Database,
) -> None:
    """Batch-insert cell and intensity-cell records without deleting first.

    Same as :func:`batch_update_celldata_and_intensitycelldata` but skips
    the per-tile delete step.  The caller is responsible for performing
    bulk deletes separately (e.g. via :meth:`Segmentation.bulk_delete_celldata`).

    Args:
        batch_data: List of (tiledata_row, regionprops_DataFrame) tuples.
        Db: Active ``Database`` connection.
    """
    if not batch_data:
        return

    celldata_batch: List[Dict[str, Any]] = []
    intensitycelldata_batch: List[Dict[str, Any]] = []

    for row, props_df in batch_data:
        for _, prow in props_df.iterrows():
            ident = uuid.uuid4()

            celldata_batch.append({
                'id': ident,
                'experimentdata_id': row.experimentdata_id,
                'welldata_id': row.welldata_id,
                'tiledata_id': row.id,
                'randomcellid_montage': prow.label,
                'centroid_x': prow['centroid_weighted-1'],
                'centroid_y': prow['centroid_weighted-0'],
                'perimeter': prow.perimeter,
                'area': prow.area,
                'solidity': prow.solidity,
                'extent': prow.extent,
                'eccentricity': prow.eccentricity,
                'axis_major_length': prow.axis_major_length,
                'axis_minor_length': prow.axis_minor_length,
            })

            intensitycelldata_batch.append({
                'experimentdata_id': row.experimentdata_id,
                'welldata_id': row.welldata_id,
                'tiledata_id': row.id,
                'celldata_id': ident,
                'channeldata_id': row.channeldata_id,
                'intensity_max': prow.intensity_max,
                'intensity_mean': prow.intensity_mean,
                'intensity_min': prow.intensity_min
            })

    # Insert without deleting; caller handles deletion separately
    if celldata_batch:
        celldata_batch = convert_numpy_types(celldata_batch)
        Db.add_row(tablename='celldata', dct=celldata_batch)

    if intensitycelldata_batch:
        intensitycelldata_batch = convert_numpy_types(intensitycelldata_batch)
        Db.add_row(tablename='intensitycelldata', dct=intensitycelldata_batch)
