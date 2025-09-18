#!/usr/bin/env python

"""Montage images or masks - WITH ENHANCED TIMING"""
import argparse
from normalization import Normalize
import datetime
import logging
import numpy as np
import os
import imageio
from sql import Database
import pdb
from time import time

logger = logging.getLogger("Montage")
# logger.propagate = False
now = datetime.datetime.now()
TIMESTAMP = '%d%02d%02d%02d%02d' % (now.year, now.month, now.day, now.hour, now.minute)
print(f'🚀 Starting montage processing at {now.strftime("%Y-%m-%d %H:%M:%S")}')
fink_log_dir = './finkbeiner_logs'
if not os.path.exists(fink_log_dir):
    os.makedirs(fink_log_dir)
logname = os.path.join(fink_log_dir, f'Montage-log_{TIMESTAMP}.log')
fh = logging.FileHandler(logname)
# fh.setLevel(logging.DEBUG)
logger.addHandler(fh)
logger.warning('Running MONTAGE WITH ENHANCED TIMING from Database.')


class Montage:
    def __init__(self, opt):
        self.opt = opt
        self.Db = Database()  # 🔹 Initialize database connection
        logger.warning('Montage class initialized.')
        self.Norm = Normalize(self.opt)
        _, self.analysisdir = self.Norm.get_raw_and_analysis_dir()
        self.montage_folder_name = 'MontagedImages'
        self.montagedir = os.path.join(self.analysisdir, self.montage_folder_name)
        if self.opt.tiletype!='filename':
            self.opt.img_norm_name = 'identity'
        
        # Performance monitoring variables
        self.start_time = None
        self.total_montages = 0
        self.processed_montages = 0
        self.db_time = 0
        self.img_time = 0
        
        # Batch database update variables
        self.batch_updates = []
        self.batch_size = 20  # Process updates in batches of 20 for better progress visibility

    def run(self, savebool=True):
        self.start_time = time()
        tiledata_df = self.Norm.get_df_for_training(['channeldata'])
        # tiledata_df = self.Norm.get_flatfields()
        tiledata_df = tiledata_df.sort_values(by=['timepoint', 'well', 'tile',])

        groups = tiledata_df.groupby(by=['timepoint', 'well', 'channel'])
        self.total_montages = len(groups)
        
        montage_start_time = time()
        for name, df in groups:
            self.single_montage(df)
        
        total_time = time() - self.start_time
        montage_time = time() - montage_start_time
        
        # Process any remaining batch updates
        if self.batch_updates:
            self.process_batch_updates()
        
        print(f'✅ MONTAGE COMPLETED in {total_time:.2f}s ({total_time/60:.2f} min)')
        
        # Write progress marker for Nextflow to track
        with open(self.opt.outfile, 'w') as f:
            f.write(f'Montaged Images.')
        
        # Create a progress marker file that Nextflow can monitor
        progress_file = os.path.join(os.path.dirname(self.opt.outfile), 'MONTAGE_COMPLETE.txt')
        with open(progress_file, 'w') as f:
            f.write(f'Montage completed at {datetime.datetime.now().isoformat()}\n')
            f.write(f'Processing time: {total_time:.2f}s\n')
            f.write(f'Montages processed: {self.total_montages}\n')
        
        # Log timing to file
        logger.warning(f'Completed montage processing in {total_time:.2f}s')
        logger.warning(f'Database operations: {self.db_time:.2f}s')
        logger.warning(f'Image processing: {self.img_time:.2f}s')
        if self.total_montages > 0:
            avg_time_per_montage = total_time / self.total_montages
            logger.warning(f'Average time per montage: {avg_time_per_montage:.2f}s')
        
        print('✅ Done.')

    def process_batch_updates(self):
        """Process database updates in batches for better performance"""
        if not self.batch_updates:
            return
            
        batch_start_time = time()
        print(f'🔄 Processing batch of {len(self.batch_updates)} database updates...')
        
        # Group updates by field type for more efficient batch operations
        updates_by_field = {}
        for update in self.batch_updates:
            field = update['update_field']
            if field not in updates_by_field:
                updates_by_field[field] = []
            updates_by_field[field].append(update)
        
        # Process each field type in batches
        for field, updates in updates_by_field.items():
            if len(updates) > 1:
                # Use batch update if available, otherwise fall back to individual updates
                try:
                    # Try to use a batch update method if it exists
                    if hasattr(self.Db, 'batch_update'):
                        # Prepare batch data for batch_update method
                        batch_data = []
                        for update in updates:
                            batch_data.append({
                                'experimentdata_id': update['experimentdata_id'],
                                'welldata_id': update['welldata_id'],
                                'channeldata_id': update['channeldata_id'],
                                'timepoint': update['timepoint'],
                                field: update['savepath']
                            })
                        
                        # Perform batch update
                        self.Db.batch_update('tiledata', batch_data, [field])
                        print(f'✅ Batch updated {len(updates)} {field} records')
                    else:
                        # Fall back to individual updates if batch_update not available
                        for update in updates:
                            self.Db.update(
                                'tiledata',
                                update_dct={update['update_field']: update['savepath']},
                                kwargs={
                                    'experimentdata_id': update['experimentdata_id'],
                                    'welldata_id': update['welldata_id'],
                                    'channeldata_id': update['channeldata_id'],
                                    'timepoint': update['timepoint']
                                }
                            )
                        print(f'✅ Updated {len(updates)} {field} records individually')
                except Exception as e:
                    print(f'⚠️ Batch update failed for {field}, falling back to individual updates: {e}')
                    # Fall back to individual updates on error
                    for update in updates:
                        try:
                            self.Db.update(
                                'tiledata',
                                update_dct={update['update_field']: update['savepath']},
                                kwargs={
                                    'experimentdata_id': update['experimentdata_id'],
                                    'welldata_id': update['welldata_id'],
                                    'channeldata_id': update['channeldata_id'],
                                    'timepoint': update['timepoint']
                                }
                            )
                        except Exception as update_error:
                            print(f'❌ Failed to update {field} for T{update["timepoint"]}: {update_error}')
            else:
                # Single update
                update = updates[0]
                try:
                    self.Db.update(
                        'tiledata',
                        update_dct={update['update_field']: update['savepath']},
                        kwargs={
                            'experimentdata_id': update['experimentdata_id'],
                            'welldata_id': update['welldata_id'],
                            'channeldata_id': update['channeldata_id'],
                            'timepoint': update['timepoint']
                        }
                    )
                except Exception as e:
                    print(f'❌ Failed to update {update["update_field"]} for T{update["timepoint"]}: {e}')
        
        batch_time = time() - batch_start_time
        self.db_time += batch_time
        print(f'✅ Batch processing completed in {batch_time:.2f}s')
        
        # Clear the processed batch
        self.batch_updates = []

    def single_montage(self, df, savebool=True):
        """
                # Get Robo0/3/4 montage order indexes relative to
        # regular left to right and top to bottom order.
        # like
        # 3 2 1                 1 2 3
        # 4 5 6     relative to 4 5 6
        # 9 8 7                 7 8 9
        # But here we start from 0
        """
        montage_start_time = time()
        images = []
        savepath = None
        mont = None
        overlap = 0  # TODO: use overlap?
        df = df.sort_values('tile')
        if len(df) == df.tile.max() and not df[self.opt.tiletype].isna().any():
            well= df.well.iloc[0]
            timepoint = int(df.timepoint.iloc[0])
            print(f'🔬 Montage: {well} T{timepoint} ({len(df)} tiles)')
            
            if self.opt.img_norm_name != 'identity' and self.opt.tiletype=='filename':
                bg_start_time = time()
                self.Norm.get_background_image(df, well, timepoint)#kaushik edit
                bg_time = time() - bg_start_time
                # Add back the background calculation message with timing
                print(f'Calculated background image for {well} at T{timepoint} in {bg_time}')
            
            
            
            img_start_time = time()
            for i, row in df.iterrows():
                f = row[self.opt.tiletype]
                # overlap = row.overlap
                if not savepath:
                    name = os.path.basename(f)
                    name = name.split('.t')[0] + '_MONTAGE.tif'
                    welldir = os.path.join(self.montagedir, row.well)
                    if not os.path.exists(welldir):
                        os.makedirs(welldir)
                    savepath = os.path.join(welldir, name)
                img = imageio.v3.imread(f)
                
                cleaned_im = self.Norm.image_bg_correction[self.opt.img_norm_name](img, row.well, row.timepoint)
                images.append(cleaned_im)
            
            img_time = time() - img_start_time
            self.img_time += img_time
            
            if well in self.Norm.backgrounds and timepoint in self.Norm.backgrounds[well]:
                del self.Norm.backgrounds[well][timepoint]
        num_tiles = len(images)
       
        logger.warning(f'Num tiles: {num_tiles}')
        side = int(np.sqrt(num_tiles))
        if num_tiles:
            montage_creation_start = time()
            h, w = np.shape(images[0])
            mont = np.zeros((int(h * side), int(w * side)), dtype=np.uint16)
            for i in range(side):
                for j in range(side):
                    #TODO: map montages for legacy montage, new montages, and ixm montages
                    if self.opt.montage_pattern == 'legacy':
                        if i%2==0:
                            k = side - (j+1)
                        else:
                            k = j
                    else:
                        k = j
                    mont[i * h:(i + 1) * h, j * w:(j + 1) * w] = images[i * side + k]
            
            montage_creation_start = time()
            
            if savebool:
                imageio.v3.imwrite(savepath, mont)
            
            # Add back the montage creation completed message
            montage_creation_time = time() - montage_creation_start
            print(f'⚡ Montage creation completed in {montage_creation_time:.2f}s')
            
            # if os.path.exists(savepath):
            if os.path.exists(savepath):
                # Get IDs
                experimentdata_id = self.Db.get_table_uuid(
                    'experimentdata',
                    dict(experiment=self.opt.experiment)
                )
                welldata_id = self.Db.get_table_uuid(
                    'welldata',
                    dict(experimentdata_id=experimentdata_id, well=well)
                )
                channel = df.channel.iloc[0]
                channeldata_id = self.Db.get_table_uuid(
                    'channeldata',
                    dict(
                        experimentdata_id=experimentdata_id,
                        welldata_id=welldata_id,
                        channel=channel
                    )
                )

                # Choose your new tiledata column
                if self.opt.tiletype == 'filename':
                    update_field = 'newimagemontage'
                elif self.opt.tiletype == 'maskpath':
                    update_field = 'newmaskmontage'
                else:
                    update_field = 'newtrackedmontage'

                                # Database update will be handled in batch later
                # Just log that we're preparing the update
                logger.warning(
                    f'Prepared {update_field} update for {well} T{timepoint}'
                )

        # Add to batch updates instead of immediate update
        self.batch_updates.append({
            'update_field': update_field,
            'savepath': savepath,
            'experimentdata_id': experimentdata_id,
            'welldata_id': welldata_id,
            'channeldata_id': channeldata_id,
            'timepoint': int(timepoint)
        })
        
        # Process batch if it reaches the batch size
        if len(self.batch_updates) >= self.batch_size:
            self.process_batch_updates()

    
    # Update progress
        self.processed_montages += 1
        total_montage_time = time() - montage_start_time
        
        print(f'✅ {well} T{timepoint}: {total_montage_time:.2f}s')
        print()  # Add blank line between timepoints
        
        return mont


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input_dict',
        help='path to pickle',
        default=f'/gladstone/finkbeiner/linsley/josh/GALAXY/YD-Transdiff-XDP-Survival1-102822/GXYTMP/tmp.pkl'
    )
    parser.add_argument(
        '--outfile',
        help='Text status',
        default=f'/gladstone/finkbeiner/linsley/GXYTMPS/Nextflow-tmp/GALAXY/YD-Transdiff-XDP-Survival1-102822/GXYTMP/tmp_output.txt'
    )
    
    parser.add_argument('--experiment', default='JAK-COR7508012023-GEDI', type=str)
    parser.add_argument('--tiletype', default='maskpath', choices=['filename', 'maskpath', 'trackedmaskpath'], type=str,
                        help='Montage image, binary mask, or tracked mask.')
    parser.add_argument('--img_norm_name', default='subtraction', choices=['division', 'subtraction', 'identity'], type=str,
                        help='Image normalization method using flatfield image.')
    parser.add_argument('--montage_pattern',default='standard', choices=['standard', 'legacy'], help="Montage snaking with 3 2 1 4 5 6 9 8 7 pattern.")
    parser.add_argument("--wells_toggle", default='include',
                        help="Chose whether to include or exclude specified wells.")
    parser.add_argument("--timepoints_toggle", default='include',
                        help="Chose whether to include or exclude specified timepoints.")
    parser.add_argument("--channels_toggle", default='include',
                        help="Chose whether to include or exclude specified channels.")
    parser.add_argument("--chosen_wells", "-cw", 
                        dest="chosen_wells", default='C15',
                        help="Specify wells to include or exclude")
    parser.add_argument("--chosen_timepoints", "-ct",
                        dest="chosen_timepoints", default='',
                        help="Specify timepoints to include or exclude.")
    parser.add_argument("--chosen_channels", "-cc",
                        dest="chosen_channels", default='all',
                        help="Specify channels to include or exclude.")
    parser.add_argument("--image_overlap", "-io",
                        dest="image_overlap", default='all',
                        help="Specify amount of overlap")
    parser.add_argument('--tile', default=0, type=int, help="Select single tile to segment. Default is to segment all tiles.")
    args = parser.parse_args()
    
    Mt = Montage(args)
    Mt.run()
