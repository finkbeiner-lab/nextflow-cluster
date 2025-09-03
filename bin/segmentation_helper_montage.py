"""Helper functions for segmentation"""
import os
import cv2
import uuid
import pandas as pd
from sql import Database
import numpy as np

def save_mask(mask, image_file, savedir):
    """Save mask with optimized I/O"""
    name = os.path.basename(image_file)
    name = name.split('.t')[0]  # split by tiff suffix
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    savepath = os.path.join(savedir, name + '_ENCODED.tif')
    
    # Use optimized compression for faster I/O
    cv2.imwrite(savepath, mask, [cv2.IMWRITE_TIFF_COMPRESSION, 1])
    return savepath

def save_masks_batch(masks_data, savedir):
    """Batch save multiple masks for better I/O performance"""
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    
    saved_paths = []
    for mask, image_file in masks_data:
        name = os.path.basename(image_file)
        name = name.split('.t')[0]
        savepath = os.path.join(savedir, name + '_ENCODED.tif')
        cv2.imwrite(savepath, mask, [cv2.IMWRITE_TIFF_COMPRESSION, 1])
        saved_paths.append(savepath)
    
    return saved_paths

def convert_numpy_types(data):
    # If it's a list of dicts, convert each one
    if isinstance(data, list):
        return [convert_numpy_types(d) for d in data]

    # If it's a single dict
    elif isinstance(data, dict):
        return {k: (v.item() if isinstance(v, np.generic) else v) for k, v in data.items()}

    # Otherwise, return as-is
    return data

def update_celldata_and_intensitycelldata(row, props_df: pd.DataFrame, Db: Database):
    """Legacy function - kept for backward compatibility"""
    celldata_dcts = []
    check_celldata_dcts = []
    intensitycelldata_dcts = []
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
    # clears all cells in tile
    if not props_df.empty:
        for check_dct in check_celldata_dcts:
            Db.delete_based_on_duplicate_name(tablename='celldata', kwargs=check_dct)
            Db.delete_based_on_duplicate_name(tablename='intensitycelldata', kwargs=check_dct)
        celldata_dcts = convert_numpy_types(celldata_dcts)
        Db.add_row(tablename='celldata', dct=celldata_dcts)
        intensitycelldata_dcts = convert_numpy_types(intensitycelldata_dcts)
        Db.add_row(tablename='intensitycelldata', dct=intensitycelldata_dcts)

def batch_update_celldata_and_intensitycelldata(batch_data, Db: Database):
    """Optimized batch update function for multiple tiles"""
    if not batch_data:
        return
    
    # Group by tiledata_id for efficient deletion
    tiledata_ids = set()
    celldata_batch = []
    intensitycelldata_batch = []
    
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
    
    # Batch delete existing data
    for tiledata_id in tiledata_ids:
        Db.delete_based_on_duplicate_name(tablename='celldata', 
                                        kwargs={'tiledata_id': tiledata_id})
        Db.delete_based_on_duplicate_name(tablename='intensitycelldata', 
                                        kwargs={'tiledata_id': tiledata_id})
    
    # Batch insert new data
    if celldata_batch:
        celldata_batch = convert_numpy_types(celldata_batch)
        Db.add_row(tablename='celldata', dct=celldata_batch)
    
    if intensitycelldata_batch:
        intensitycelldata_batch = convert_numpy_types(intensitycelldata_batch)
        Db.add_row(tablename='intensitycelldata', dct=intensitycelldata_batch)


