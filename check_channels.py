#!/usr/bin/env python
"""Quick script to check what channels are registered in the database for an experiment"""

import sys
sys.path.append('/gladstone/finkbeiner/imaging-home/vgramas/nextflow-cluster')

from sql import Database
import pandas as pd

experiment = 'JAKZF-AlphDOP12-SiRICC2-11132025'

Db = Database()
exp_uuid = Db.get_table_uuid('experimentdata', dict(experiment=experiment))

if exp_uuid is None:
    print(f"❌ Experiment '{experiment}' not found in database!")
    sys.exit(1)

# Get all channels for this experiment
channeldata_df = Db.get_df_from_query('channeldata', dict(experimentdata_id=exp_uuid))

if channeldata_df.empty:
    print(f"❌ No channels registered for experiment '{experiment}'")
    sys.exit(1)

# Get unique channels
unique_channels = channeldata_df['channel'].unique()
print(f"\n✅ Found {len(unique_channels)} unique channel(s) registered for experiment '{experiment}':")
print(f"   Channels: {sorted(unique_channels.tolist())}")

# Also show channels per well (sample a few wells)
print(f"\n📊 Sample channels per well (showing first 5 wells):")
welldata_df = Db.get_df_from_query('welldata', dict(experimentdata_id=exp_uuid))
channeldata_with_wells = pd.merge(channeldata_df, welldata_df[['id', 'well']], 
                                   left_on='welldata_id', right_on='id', how='left')

for well in sorted(channeldata_with_wells['well'].unique())[:5]:
    well_channels = channeldata_with_wells[channeldata_with_wells['well'] == well]['channel'].unique()
    print(f"   {well}: {sorted(well_channels.tolist())}")

