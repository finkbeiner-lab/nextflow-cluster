"""Helper functions for puncta"""
import os
import cv2
import uuid
import pandas as pd
from sql import Database


def update_punctadata_and_intensitypunctadata(row, props_df: pd.DataFrame, Db: Database):
    punctadata_dcts = []
    intensitypunctadata_dcts = []
    check_punctadata_dct = dict(experimentdata_id=row.experimentdata_id,
                                          welldata_id=row.welldata_id,
                                          tiledata_id=row.tiledata_id,
                                          )
    for i, prow in props_df.iterrows():
        ident = uuid.uuid4()
        

        punctadata_dcts.append(
            dict(id=ident,
                 experimentdata_id=row.experimentdata_id,
                 welldata_id=row.welldata_id,
                 tiledata_id=row.tiledata_id,
                 celldata_id=row.id,
                 randompunctaid=prow.label,
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
        intensitypunctadata_dcts.append(dict(experimentdata_id=row.experimentdata_id,
                                             welldata_id=row.welldata_id,
                                             tiledata_id=row.tiledata_id,
                                             celldata_id=row.id,
                                             punctadata_id=ident,
                                             channeldata_id=row.channeldata_id,
                                             intensity_max=prow.intensity_max,
                                             intensity_mean=prow.intensity_mean,
                                             intensity_min=prow.intensity_min))
    # clears all cells in tile
    Db.delete_based_on_duplicate_name(tablename='punctadata', kwargs=check_punctadata_dct)
    Db.delete_based_on_duplicate_name(tablename='intensitypunctadata', kwargs=check_punctadata_dct)
    Db.add_row(tablename='punctadata', dct=punctadata_dcts)
    Db.add_row(tablename='intensitypunctadata', dct=intensitypunctadata_dcts)