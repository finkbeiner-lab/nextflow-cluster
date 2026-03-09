#!/usr/bin/env python
"""Check what channels are actually in tiledata vs channeldata"""

import sys
sys.path.append('/gladstone/finkbeiner/imaging-home/vgramas/nextflow-cluster')

from sql import Database
import pandas as pd

experiment = 'JAKZF-AlphDOP12-SiRICC2-11132025'
well = 'B03'  # The well you're trying to process
channel = 'FITC'  # The channel you're trying to use

Db = Database()
exp_uuid = Db.get_table_uuid('experimentdata', dict(experiment=experiment))

if exp_uuid is None:
    print(f"❌ Experiment '{experiment}' not found in database!")
    sys.exit(1)

# Get welldata
welldata_df = Db.get_df_from_query('welldata', dict(experimentdata_id=exp_uuid, well=well))
if welldata_df.empty:
    print(f"❌ Well '{well}' not found for experiment '{experiment}'")
    sys.exit(1)

well_uuid = welldata_df['id'].iloc[0]
print(f"\n✅ Found well {well} (UUID: {well_uuid})")

# Get channeldata for this well
channeldata_df = Db.get_df_from_query('channeldata', dict(experimentdata_id=exp_uuid, welldata_id=well_uuid))
print(f"\n📊 Channels in channeldata for well {well}:")
if channeldata_df.empty:
    print("   No channels found!")
else:
    channels_in_channeldata = channeldata_df['channel'].unique()
    print(f"   {sorted(channels_in_channeldata.tolist())}")

# Get tiledata for this well
tiledata_df = Db.get_df_from_query('tiledata', dict(experimentdata_id=exp_uuid, welldata_id=well_uuid))
print(f"\n📊 Tiledata rows for well {well}: {len(tiledata_df)} rows")

if tiledata_df.empty:
    print("   ❌ No tiledata rows found for this well!")
    sys.exit(1)

# Get channeldata_ids from tiledata
tiledata_channel_ids = tiledata_df['channeldata_id'].unique()
print(f"   Found {len(tiledata_channel_ids)} unique channeldata_id(s)")

# Match channeldata_ids to channel names
all_channeldata_df = Db.get_df_from_query('channeldata', dict(experimentdata_id=exp_uuid))
merged = pd.merge(tiledata_df[['channeldata_id']].drop_duplicates(), 
                  all_channeldata_df[['id', 'channel']], 
                  left_on='channeldata_id', right_on='id', how='left')

print(f"\n📊 Channels actually linked in tiledata for well {well}:")
if merged['channel'].isna().any():
    print("   ⚠️  Some tiledata rows have channeldata_id that don't match any channeldata!")
    print(f"   Missing matches: {merged[merged['channel'].isna()]['channeldata_id'].tolist()}")
    
channels_in_tiledata = merged['channel'].dropna().unique()
print(f"   {sorted(channels_in_tiledata.tolist())}")

# Check specifically for FITC
print(f"\n🔍 Checking specifically for channel '{channel}':")

# Get FITC channeldata_id for this well
fitc_channeldata = channeldata_df[channeldata_df['channel'] == channel]
if fitc_channeldata.empty:
    print(f"   ❌ No '{channel}' channel found in channeldata for well {well}")
else:
    fitc_channeldata_id = fitc_channeldata['id'].iloc[0]
    print(f"   ✅ '{channel}' channeldata_id: {fitc_channeldata_id}")
    
    # Check if any tiledata rows use this channeldata_id
    fitc_tiledata = tiledata_df[tiledata_df['channeldata_id'] == fitc_channeldata_id]
    if fitc_tiledata.empty:
        print(f"   ❌ PROBLEM: No tiledata rows link to '{channel}' channeldata_id!")
        print(f"   \n   This means:")
        print(f"   - '{channel}' exists in channeldata (from template)")
        print(f"   - But no image files with '{channel}' in the filename were registered")
        print(f"   \n   Possible causes:")
        print(f"   1. Registration was run with --chosen_channels that excluded '{channel}'")
        print(f"   2. Registration was run with --channels_toggle exclude and '{channel}' was excluded")
        print(f"   3. Well {well} doesn't have any '{channel}' image files")
        print(f"   \n   Solution: Re-register the experiment including '{channel}' channel")
    else:
        print(f"   ✅ Found {len(fitc_tiledata)} tiledata rows for '{channel}'")
        print(f"   Timepoints: {sorted(fitc_tiledata['timepoint'].unique().tolist())}")
        
        # Show sample filenames
        if 'filename' in fitc_tiledata.columns:
            sample_filenames = fitc_tiledata['filename'].head(3).tolist()
            print(f"\n   Sample '{channel}' filenames:")
            for fname in sample_filenames:
                print(f"     {fname}")

# Check if FITC is in channeldata but not in tiledata
if channel in channels_in_channeldata and channel not in channels_in_tiledata:
    print(f"\n⚠️  SUMMARY:")
    print(f"   '{channel}' exists in channeldata but NO tiledata rows link to it!")
    print(f"   This means your image filenames don't contain '{channel}' as the channel name.")
    print(f"   \n   Solution: Check your filenames - they might use a different channel name.")
    
    # Show sample filenames from other channels
    if 'filename' in tiledata_df.columns and len(tiledata_df) > 0:
        sample_filenames = tiledata_df['filename'].head(3).tolist()
        print(f"\n   Sample filenames from tiledata (other channels):")
        for fname in sample_filenames:
            print(f"     {fname}")

