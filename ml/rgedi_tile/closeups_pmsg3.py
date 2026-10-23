#!/usr/bin/env python
"""Full-resolution close-up tiles for the hevo-pmsG-3 deck animation (slide: plate time-lapse -> zoom).

Usage (inside the SIF, /gladstone bound):  python closeups_pmsg3.py OUT_DIR T1 T2 ...
Reads OUT_DIR/plate_display_settings.json (written by plate_grid_pmsg3.py for the same timepoints) so the close-ups use
exactly the plate's pooled display (FITC lo/hi, RFP lo/hi, gamma) and GFP-green / RFP-magenta merge. Writes
OUT_DIR/closeups/<tag>_<TP>.jpg, one raw 2048 px tile per selected well, downsampled to 1024 px.
"""
import glob, json, os, sys
import numpy as np, tifffile
from PIL import Image

RAW = "/gladstone/finkbeiner/robodata/Robo3Images/hevo-pmsG-3"
# (tag, well, tile): centre tile (5) of the 0 ms and 1 s wells for each line, replicate row C
SEL = [("TP0388_0ms", "C7", 5), ("TP0388_1s", "C10", 5), ("TP0149_0ms", "C2", 5), ("TP0149_1s", "C5", 5)]
OUT_PX = 1024

def main():
    out, tps = sys.argv[1], sys.argv[2:]
    disp = json.load(open(os.path.join(out, "plate_display_settings.json")))
    G, R, gamma = disp["fitc"], disp["rfp"], disp["gamma"]
    st = lambda a, lh: np.clip((a.astype(np.float32) - lh[0]) / (lh[1] - lh[0]), 0, 1) ** gamma
    os.makedirs(os.path.join(out, "closeups"), exist_ok=True)
    for tp in tps:
        for tag, well, tile in SEL:
            f = glob.glob(f"{RAW}/{well}/*_{tp}_0-0_{well}_{tile}_FITC_*.tif")
            r = glob.glob(f"{RAW}/{well}/*_{tp}_0-0_{well}_{tile}_RFP_*.tif")
            if not f or not r:
                print("missing", tag, tp); continue
            gn, rn = st(tifffile.imread(f[0]), G), st(tifffile.imread(r[0]), R)
            img = Image.fromarray((np.stack([rn, gn, rn], -1) * 255).astype(np.uint8))
            img.resize((OUT_PX, OUT_PX), Image.LANCZOS).save(os.path.join(out, "closeups", f"{tag}_{tp}.jpg"), quality=88)
        print("closeups done", tp, flush=True)

if __name__ == "__main__":
    main()
