#!/opt/conda/bin/python
"""
Aligns tile images using phase cross-correlation and DFT fallback.
For morphology channel: computes and saves shift.
For others: reuses shift from shift_dict file.
"""

import os
import cv2
import argparse
import numpy as np
import pandas as pd
from skimage import exposure,transform
from skimage.registration import phase_cross_correlation
import imreg_dft as ird
from sql import Database
from tqdm import tqdm
import imageio
import pickle
from db_util import Ops
import datetime

# # #Original
def cross_correlation_dft_combo(ref_img, img):
    try:
        shift, error, _ = phase_cross_correlation(ref_img, img)
        if np.linalg.norm(shift) > 0.5:
            tform = transform.SimilarityTransform(translation=(-shift[1], -shift[0]))
            shifted_img = transform.warp(img, tform, preserve_range=True).astype(img.dtype)
            return shifted_img, shift, 'phase'
    except Exception as e:
        print(f"Phase cross-correlation failed: {e}")

    try:
        result = ird.translation(ref_img, img)
        shift = result['tvec']
        tform = transform.SimilarityTransform(translation=(-shift[1], -shift[0]))
        shifted_img = transform.warp(img, tform, preserve_range=True).astype(img.dtype)
        return shifted_img, shift, 'dft'
    except Exception as e:
        print(f"DFT alignment failed: {e}")
        return img, (0, 0), 'none'

def align_tiles(opt):
    ops = Ops(opt)
    tiledata = ops.get_df_for_training(['channeldata'])

    tiledata = tiledata[tiledata["filename"].str.endswith(".tif")]
    tiledata = tiledata[tiledata["channel"] != '']

    if "timepoint" not in tiledata.columns:
        raise KeyError("tiledata table must contain a 'timepoint' column")

    db = Database()
    _, analysisdir = ops.get_raw_and_analysis_dir()
    savedir_root = os.path.join(analysisdir, 'AlignedTiles')
    os.makedirs(savedir_root, exist_ok=True)

    # Load precomputed shift_dict if provided and not morphology_channel
    use_saved_shifts = False
    shift_dict = {}
    if hasattr(opt, 'shift_dict') and opt.shift_dict and os.path.exists(opt.shift_dict):
        with open(opt.shift_dict, 'rb') as f:
            shift_dict = pickle.load(f)
        use_saved_shifts = True
        print(f"📁 Loaded shift dictionary from {opt.shift_dict}")

    all_groups = tiledata.groupby(["well", "tile", "channeldata_id"])

    # Create new shift dict if morphology channel
    new_shift_dict = {}
    for (well, tile, channel_id), group in tqdm(all_groups, desc="Aligning tiles"):
        group = group.sort_values("timepoint")
        running_shift = np.array([0.0, 0.0])

        ref_row = group.iloc[0]
        ref_img = cv2.imread(ref_row.filename, cv2.IMREAD_UNCHANGED)
        if ref_img is None:
            print(f"⚠️ Reference image unreadable: {ref_row.filename}")
            continue

        prev_img = ref_img  # Initialize for cumulative alignment

        for i, (_, row) in enumerate(group.iterrows()):
            out_dir = os.path.join(savedir_root, row.well)
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, os.path.basename(row.filename).replace(".tif", "_ALIGNED.tif"))

            if os.path.exists(out_path):
                db.update("tiledata", {"alignedtilepath": out_path}, {"id": row.tiledata_id})
                continue

            img = cv2.imread(row.filename, cv2.IMREAD_UNCHANGED)
            if img is None:
                print(f"⚠️ Skipping unreadable image: {row.filename}")
                continue

            if i == 0:
                aligned = img.copy()
                print(f"🔹 Using {row.filename} as reference (T0)")
            else:
                aligned, shift, method = cross_correlation_dft_combo(prev_img, img)
                running_shift += shift
                print(f"✅ {row.filename} aligned using {method} | Δshift={shift} | Total shift={running_shift}")

            imageio.imwrite(out_path, aligned.astype(np.uint16))
            db.update("tiledata", {"alignedtilepath": out_path}, {"id": row.tiledata_id})

            prev_img = img  # Update reference for next alignment


# Original
    # for (well, tile, channel_id), group in tqdm(all_groups, desc="Aligning tiles"):
    #     group = group.sort_values("timepoint")
    #     ref_row = group[group.timepoint == 1].iloc[0] ## KS edit for IXM use group.timepoint==1 else use timepoint ==0
    #     ref_img = cv2.imread(ref_row.filename, cv2.IMREAD_UNCHANGED)

    #     for _, row in group.iterrows():
    #         base_name = os.path.basename(row.filename).replace(".tif", "_ALIGNED.tif")
    #         out_dir = os.path.join(savedir_root, row.well)
    #         os.makedirs(out_dir, exist_ok=True)
    #         out_path = os.path.join(out_dir, base_name)

    #         if os.path.exists(out_path):
    #             db.update("tiledata", {"alignedtilepath": out_path}, {"id": row.tiledata_id})
    #             continue

    #         img = cv2.imread(row.filename, cv2.IMREAD_UNCHANGED)
    #         if img is None or ref_img is None:
    #             print(f"Warning: Could not read {row.filename}")
    #             continue

    #         key = (row.well, row.tile, row.timepoint)

    #         if row.channel == opt.morphology_channel:
    #             aligned, shift, method = cross_correlation_dft_combo(ref_img, img)
    #             new_shift_dict[key] = shift
    #             print(f"✅ Aligned morphology channel {row.channel} using {method}")
    #         elif use_saved_shifts and key in shift_dict:
    #             shift = shift_dict[key]
    #             tform = transform.SimilarityTransform(translation=(-shift[1], -shift[0]))
    #             aligned = transform.warp(img, tform, preserve_range=True).astype(img.dtype)
    #             print(f"📎 Applied saved shift to {row.channel}")
    #         else:
    #             print(f"❌ No shift available for {row.channel} at {key} — skipping")
    #             continue

    #         imageio.imwrite(out_path, aligned.astype(np.uint16))
    #         db.update("tiledata", {"alignedtilepath": out_path}, {"id": row.tiledata_id})

    # Save new shift_dict if working on morphology channel
    if not use_saved_shifts:
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        shift_path = os.path.join(savedir_root, f"calculated_shift_{timestamp}.dict")
        with open(shift_path, 'wb') as f:
            pickle.dump(new_shift_dict, f)
        print(f"\n💾 Saved shift dictionary to {shift_path}")

    print("\n✅ Tile alignment complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Align tile images to T0 using phase correlation + DFT fallback.")
    parser.add_argument("--experiment", type=str, required=True)
    parser.add_argument("--morphology_channel", type=str, required=True)
    parser.add_argument("--wells_toggle", default='include')
    parser.add_argument("--timepoints_toggle", default='include')
    parser.add_argument("--channels_toggle", default='include')
    parser.add_argument("--chosen_wells", default='all')
    parser.add_argument("--chosen_timepoints", default='all')
    parser.add_argument("--chosen_channels", default='all')
    parser.add_argument("--tile", type=int, default=0)
    parser.add_argument("--shift_dict", type=str, default='')

    args = parser.parse_args()

    align_tiles(args)






# #!/opt/conda/bin/python
# """
# Aligns tile images across timepoints using phase cross-correlation and DFT fallback.
# Saves aligned tiles to AlignedTiles/ and updates alignedtilepath in tiledata table.
# """
# import os
# import cv2
# import argparse
# import numpy as np
# import pandas as pd
# from skimage import transform
# from skimage.registration import phase_cross_correlation
# import imreg_dft as ird
# from sql import Database
# from tqdm import tqdm
# import imageio
# from db_util import Ops

# # def cross_correlation_dft_combo(ref_img, img):
# #     # Normalize both images to 8-bit for robust registration
# #     ref_norm = cv2.normalize(ref_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
# #     img_norm = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

# #     try:
# #         shift, error, _ = phase_cross_correlation(ref_norm, img_norm)
# #         if np.linalg.norm(shift) > 0.5:
# #             M = np.float32([[1, 0, -shift[1]], [0, 1, -shift[0]]])
# #             aligned = cv2.warpAffine(
# #                 img, M,
# #                 (img.shape[1], img.shape[0]),
# #                 flags=cv2.INTER_NEAREST,
# #                 borderMode=cv2.BORDER_CONSTANT,
# #                 borderValue=0
# #             )
# #             return aligned, shift, 'phase'
# #     except Exception as e:
# #         print(f"[WARN] Phase correlation failed: {e}")

# #     try:
# #         result = ird.translation(ref_norm, img_norm)
# #         shift = result['tvec']
# #         M = np.float32([[1, 0, -shift[1]], [0, 1, -shift[0]]])
# #         aligned = cv2.warpAffine(
# #             img, M,
# #             (img.shape[1], img.shape[0]),
# #             flags=cv2.INTER_NEAREST,
# #             borderMode=cv2.BORDER_CONSTANT,
# #             borderValue=0
# #         )
# #         return aligned, shift, 'dft'
# #     except Exception as e:
# #         print(f"[ERROR] DFT alignment failed: {e}")
# #         return img, (0, 0), 'none'


# def cross_correlation_dft_combo(ref_img, img):
#     try:
#         shift, error, _ = phase_cross_correlation(ref_img, img)
#         if np.linalg.norm(shift) > 0.5:
#             tform = transform.SimilarityTransform(translation=(-shift[1], -shift[0]))
#             shifted_img = transform.warp(img, tform, preserve_range=True).astype(img.dtype)
#             return shifted_img, shift, 'phase'
#     except Exception as e:
#         print(f"Phase cross-correlation failed: {e}")

#     try:
#         result = ird.translation(ref_img, img)
#         shift = result['tvec']
#         tform = transform.SimilarityTransform(translation=(-shift[1], -shift[0]))
#         shifted_img = transform.warp(img, tform, preserve_range=True).astype(img.dtype)
#         return shifted_img, shift, 'dft'
#     except Exception as e:
#         print(f"DFT alignment failed: {e}")
#         return img, (0, 0), 'none'

# def align_tiles(opt):
#     ops = Ops(opt)
#     tiledata = ops.get_df_for_training(['channeldata'])

#     tiledata = tiledata[tiledata["filename"].str.endswith(".tif")]
#     tiledata = tiledata[tiledata["channel"] != '']

#     if "timepoint" not in tiledata.columns:
#         raise KeyError("tiledata table must contain a 'timepoint' column")

#     ref_tiles = tiledata[tiledata["channel"] == opt.morphology_channel]
#     grouped = ref_tiles.groupby(["well", "tile", "channeldata_id"])

#     db = Database()
#     _, analysisdir = ops.get_raw_and_analysis_dir()
#     savedir_root = os.path.join(analysisdir, 'AlignedTiles')
#     os.makedirs(savedir_root, exist_ok=True)

#     for (well, tile, channel_id), group in tqdm(grouped, desc="Aligning tiles"):
#         group = group.sort_values("timepoint")
#         if group[group.timepoint == 0].empty:
#             print(f"Skipping {well} tile {tile} - no T0 image")
#             continue

#         ref_row = group[group.timepoint == 0].iloc[0]
#         ref_img = cv2.imread(ref_row.filename, cv2.IMREAD_UNCHANGED)

#         for _, row in group.iterrows():
#             base_name = os.path.basename(row.filename).replace(".tif", "_ALIGNED.tif")
#             out_dir = os.path.join(savedir_root, row.well)
#             os.makedirs(out_dir, exist_ok=True)
#             out_path = os.path.join(out_dir, base_name)

#             if os.path.exists(out_path):
#                 print(f"🔁 Already exists, updating DB for: {row.tiledata_id} → {out_path}")
#                 db.update("tiledata", {"alignedtilepath": out_path}, {"id": row.tiledata_id})
#                 continue

#             # if os.path.exists(out_path):
#             #     db.update("tiledata", {"alignedtilepath": out_path}, {"id": row.tiledata_id})
#             #     continue

#             img = cv2.imread(row.filename, cv2.IMREAD_UNCHANGED)
#             if img is None or ref_img is None:
#                 print(f"Warning: Could not read {row.filename}")
#                 continue

#             aligned, shift, method = cross_correlation_dft_combo(ref_img, img)

#             print(f"✅ Saving aligned + updating DB: {row.tiledata_id} → {out_path}")
#             # imageio.imwrite(out_path, (aligned * 65535).astype(np.uint16))

#             imageio.imwrite(out_path, aligned.astype(np.uint16))
            

#             db.update("tiledata", {"alignedtilepath": out_path}, {"id": row.tiledata_id})

#     print("\n✅ Tile alignment complete.")

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Align tile images to T0 using phase correlation + DFT fallback.")
#     parser.add_argument("--experiment", type=str, required=True, help="Experiment name (from experimentdata table)")
#     parser.add_argument("--morphology_channel", type=str, required=True, help="Morphology channel (e.g., GFP-DMD1)")
#     parser.add_argument("--wells_toggle", default='include')
#     parser.add_argument("--timepoints_toggle", default='include')
#     parser.add_argument("--channels_toggle", default='include')
#     parser.add_argument("--chosen_wells", default='all')
#     parser.add_argument("--chosen_timepoints", default='all')
#     parser.add_argument("--chosen_channels", default='all')
#     parser.add_argument("--tile", type=int, default=0, help="Tile selection for Ops compatibility")

#     args = parser.parse_args()

#     align_tiles(args)
