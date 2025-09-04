"""Helper functions for segmentation - OPTIMIZED VERSION"""
import os
import tifffile  # Faster than cv2 for TIFF files
import uuid
import pandas as pd
from sql import Database
import numpy as np
from typing import List, Tuple, Dict, Any
import gc


def save_mask(mask, image_file, savedir):
    """Save mask with optimized I/O using tifffile - OpenCV compatible format"""
    name = os.path.basename(image_file)
    name = name.split('.t')[0]  # split by tiff suffix
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    savepath = os.path.join(savedir, name + '_ENCODED.tif')
    
    # Use tifffile with OpenCV-compatible format
    # Ensure the mask is uint16 and use no compression for maximum compatibility
    mask_uint16 = mask.astype(np.uint16)
    tifffile.imwrite(savepath, mask_uint16, compression=None)
    return savepath

def save_masks_batch(masks_data, savedir):
    """Batch save multiple masks for better I/O performance using tifffile - OpenCV compatible"""
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    
    saved_paths = []
    for mask, image_file in masks_data:
        name = os.path.basename(image_file)
        name = name.split('.t')[0]
        savepath = os.path.join(savedir, name + '_ENCODED.tif')
        # Use tifffile with OpenCV-compatible format
        mask_uint16 = mask.astype(np.uint16)
        tifffile.imwrite(savepath, mask_uint16, compression=None)
        saved_paths.append(savepath)
    
    return saved_paths

def convert_numpy_types(data):
    """Convert numpy types to Python native types - OPTIMIZED"""
    # If it's a list of dicts, convert each one
    if isinstance(data, list):
        return [convert_numpy_types(d) for d in data]

    # If it's a single dict
    elif isinstance(data, dict):
        return {k: (v.item() if isinstance(v, np.generic) else v) for k, v in data.items()}

    # Otherwise, return as-is
    return data

def update_celldataand_intensitycelldata(row, props_df: pd.DataFrame, Db: Database):
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

def batch_update_celldataand_intensitycelldata(batch_data, Db: Database):
    """HIGHLY OPTIMIZED batch update function for multiple tiles"""
    if not batch_data:
        return
    
    # Pre-allocate containers with estimated sizes
    total_cells = sum(len(props_df) for _, props_df in batch_data if not props_df.empty)
    if total_cells == 0:
        return
    
    celldata_batch = []
    intensitycelldata_batch = []
    
    # Pre-allocate lists for better memory management
    celldata_batch.reserve(total_cells) if hasattr(celldata_batch, 'reserve') else None
    intensitycelldata_batch.reserve(total_cells) if hasattr(intensitycelldata_batch, 'reserve') else None
    
    # Group by tiledata_id for efficient deletion
    tiledata_ids = set()
    
    # Process all data in a single pass
    for row, props_df in batch_data:
        if props_df.empty:
            continue
            
        tiledata_ids.add(row.id)
        
        # Use vectorized operations where possible
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
    
    # Batch delete existing data for all tiles
    if tiledata_ids:
        # Use bulk delete if available, otherwise process in chunks
        chunk_size = 50  # Process deletions in chunks to avoid memory issues
        tiledata_list = list(tiledata_ids)
        
        for i in range(0, len(tiledata_list), chunk_size):
            chunk = tiledata_list[i:i+chunk_size]
            for tiledata_id in chunk:
                Db.delete_based_on_duplicate_name(tablename='celldata', 
                                                kwargs={'tiledata_id': tiledata_id})
                Db.delete_based_on_duplicate_name(tablename='intensitycelldata', 
                                                kwargs={'tiledata_id': tiledata_id})
    
    # Batch insert new data with optimized chunking
    if celldata_batch:
        celldata_batch = convert_numpy_types(celldata_batch)
        # Insert in chunks to avoid memory issues
        chunk_size = 1000
        for i in range(0, len(celldata_batch), chunk_size):
            chunk = celldata_batch[i:i+chunk_size]
            Db.add_row(tablename='celldata', dct=chunk)
    
    if intensitycelldata_batch:
        intensitycelldata_batch = convert_numpy_types(intensitycelldata_batch)
        # Insert in chunks to avoid memory issues
        chunk_size = 1000
        for i in range(0, len(intensitycelldata_batch), chunk_size):
            chunk = intensitycelldata_batch[i:i+chunk_size]
            Db.add_row(tablename='intensitycelldata', dct=chunk)
    
    # Clean up memory
    del celldata_batch, intensitycelldata_batch
    gc.collect()

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

def batch_update_celldata_and_intensitycelldata_no_delete(batch_data, Db: Database):
    """Optimized batch update function for multiple tiles - SKIPS DELETES for bulk optimization"""
    if not batch_data:
        return
    
    celldata_batch = []
    intensitycelldata_batch = []
    
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
    
    # Batch insert new data (deletes will be done in bulk at the end)
    if celldata_batch:
        celldata_batch = convert_numpy_types(celldata_batch)
        Db.add_row(tablename='celldata', dct=celldata_batch)
    
    if intensitycelldata_batch:
        intensitycelldata_batch = convert_numpy_types(intensitycelldata_batch)
        Db.add_row(tablename='intensitycelldata', dct=intensitycelldata_batch)

