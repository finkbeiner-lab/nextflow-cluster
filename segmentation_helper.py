"""Helper functions for segmentation"""
import os
import cv2
import uuid
import pandas as pd
from sql import Database


def save_mask(mask, image_file, savedir):
    name = os.path.basename(image_file)
    name = name.split('.t')[0]  # split by tiff suffix
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    savepath = os.path.join(savedir, name + '_ENCODED.tif')
    cv2.imwrite(savepath, mask)
    return savepath


def update_celldata_and_intensitycelldata(row, props_df: pd.DataFrame, Db: Database):
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
                 randomcellid=prow.label,
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
        Db.add_row(tablename='celldata', dct=celldata_dcts)
        Db.add_row(tablename='intensitycelldata', dct=intensitycelldata_dcts)


