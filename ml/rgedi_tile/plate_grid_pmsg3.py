#!/usr/bin/env python
"""Whole-plate view for hevo-pmsG-3: every well's 3x3 montage placed at its plate position (rows B-E,
cols 2-11), GFP green / RFP magenta, one display per channel shared by every well and every timepoint.

Usage (inside the SIF, /gladstone bound):
    python plate_grid_pmsg3.py OUT_DIR T0 T1 [...]

Stitch: Robo3 2048 px tiles, snake 1 2 3 / 6 5 4 / 7 8 9, 186 px overlap, last tile wins.
Display: FITC p2-p99.7, RFP p2-p99.9 of the pooled pixels of all wells x all given timepoints (each well
weighted equally), gamma 0.7; failed frames (dark: p99 < 500, blown out: p1 >= 65000) are drawn and tagged
but left out of the range. Writes plate_<TP>.jpg per timepoint, zoom_<first>_vs_<last>.jpg (center 1024 px crop of
the ZOOM wells: GFP | RFP | merge at the first and last timepoint) and plate_display_settings.json.
"""
import glob
import json
import os
import sys
from multiprocessing import Pool

import numpy as np
import tifffile
from PIL import Image, ImageDraw, ImageFont

RAW = "/gladstone/finkbeiner/robodata/Robo3Images/hevo-pmsG-3"
GRID = [[1, 2, 3], [6, 5, 4], [7, 8, 9]]
TS, OV = 2048, 186
ST = TS - OV
N = 2 * ST + TS
DS = 9                       # well montage downsample -> ~641 px per well
ROWS, COLS = "BCDE", list(range(2, 12))
LINES = [("TP0149", range(2, 7)), ("TP0388", range(7, 12))]
DOSE = {2: 0, 3: 50, 4: 500, 5: 1000, 6: 5000, 7: 0, 8: 50, 9: 500, 10: 1000, 11: 5000}
GAMMA, DARK_P99, BLOWN_P1 = 0.7, 500, 65000
ZOOM = [("C", 2), ("C", 6), ("C", 7), ("C", 11)]
C0 = (N - 1024) // 2
GAP, BLOCK_GAP, BORDER = 14, 44, 2
BG, BORDER_RGB, INK, INK2 = (18, 18, 18), (95, 95, 95), (235, 235, 235), (160, 160, 160)


def font(size, bold=False):
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    try:
        return ImageFont.truetype(f"/usr/share/fonts/truetype/dejavu/{name}", size)
    except OSError:
        return ImageFont.load_default()


def dose_label(ms):
    return "0 ms" if ms == 0 else (f"{ms // 1000} s" if ms >= 1000 else f"{ms} ms")


def montage(well, tp, ch):
    m = np.zeros((N, N), np.uint16)
    for i in range(3):
        for j in range(3):
            f = glob.glob(f"{RAW}/{well}/*_{tp}_0-0_{well}_{GRID[i][j]}_{ch}_*.tif")[0]
            m[i * ST:i * ST + TS, j * ST:j * ST + TS] = tifffile.imread(f)
    return m


def load_well(job):
    """Stitch one well/timepoint; return the plate-view thumbnail, optional zoom crop, failure flag, display sample."""
    tp, r, c, want_crop = job
    g, rr = montage(f"{r}{c}", tp, "FITC"), montage(f"{r}{c}", tp, "RFP")
    sm = (g[::DS, ::DS].astype(np.float32), rr[::DS, ::DS].astype(np.float32))
    cr = ((g[C0:C0 + 1024:2, C0:C0 + 1024:2].astype(np.float32), rr[C0:C0 + 1024:2, C0:C0 + 1024:2].astype(np.float32))
          if want_crop else None)
    s = g[::24, ::24]
    flag = ("dark frame" if np.percentile(s, 99) < DARK_P99
            else "FITC blown out" if np.percentile(s, 1) >= BLOWN_P1 else "")
    return tp, r, c, sm, cr, flag, s.ravel(), rr[::24, ::24].ravel()


def main():
    out, tps = sys.argv[1], sys.argv[2:] or ["T0", "T1"]
    small, crops, flags, pool_g, pool_r = {}, {}, {}, [], []
    jobs = [(tp, r, c, (r, c) in ZOOM and tp in (tps[0], tps[-1])) for tp in tps for r in ROWS for c in COLS]
    with Pool(int(os.environ.get("SLURM_CPUS_PER_TASK", "4"))) as pool:
        for tp, r, c, sm, cr, flag, sg, sr in pool.imap_unordered(load_well, jobs):
            small[(tp, r, c)] = sm
            flags[(tp, r, c)] = flag
            if cr is not None:
                crops[(tp, r, c)] = cr
            if not flag:
                pool_g.append(sg); pool_r.append(sr)
    pg, pr = np.concatenate(pool_g).astype(np.float32), np.concatenate(pool_r).astype(np.float32)
    G = (float(np.percentile(pg, 2)), float(np.percentile(pg, 99.7)))
    R = (float(np.percentile(pr, 2)), float(np.percentile(pr, 99.9)))
    print(f"display FITC {G[0]:.0f}-{G[1]:.0f}  RFP {R[0]:.0f}-{R[1]:.0f}  from {len(pool_g)} well-timepoints", flush=True)

    def stretch(a, lh):
        return np.clip((a - lh[0]) / (lh[1] - lh[0]), 0, 1) ** GAMMA

    W = small[(tps[0], ROWS[0], COLS[0])][0].shape[0]
    left, top, foot = 60, 96, 40
    xs, x = {}, left
    for k, c in enumerate(COLS):
        xs[c] = x
        x += W + (BLOCK_GAP if c == 6 else GAP)
    width = x - GAP + 24
    height = top + len(ROWS) * (W + GAP) - GAP + 16 + foot
    f_head, f_col, f_row, f_tag, f_foot = font(26, True), font(19), font(28, True), font(15), font(16)
    for tp in tps:
        canvas = Image.new("RGB", (width, height), BG)
        d = ImageDraw.Draw(canvas)
        for name, cols in LINES:
            x0, x1 = xs[cols[0]], xs[cols[-1]] + W
            d.text(((x0 + x1) // 2, 10), name, fill=INK, font=f_head, anchor="mt")
            d.line([(x0, 44), (x1, 44)], fill=BORDER_RGB, width=1)
        for c in COLS:
            d.text((xs[c] + W // 2, 54), f"{c}  ·  {dose_label(DOSE[c])}", fill=INK2, font=f_col, anchor="mt")
        for i, r in enumerate(ROWS):
            y = top + i * (W + GAP)
            d.text((left // 2, y + W // 2), r, fill=INK2, font=f_row, anchor="mm")
            for c in COLS:
                g, rr = small[(tp, r, c)]
                gn, rn = stretch(g, G), stretch(rr, R)
                tile = Image.fromarray((np.stack([rn, gn, rn], -1) * 255).astype(np.uint8))
                canvas.paste(tile, (xs[c], y))
                d.rectangle([xs[c] - BORDER, y - BORDER, xs[c] + W + BORDER - 1, y + W + BORDER - 1],
                            outline=BORDER_RGB, width=BORDER)
                if flags[(tp, r, c)]:
                    d.text((xs[c] + 8, y + W - 8), flags[(tp, r, c)], fill=INK2, font=f_tag, anchor="ld")
        hrs = 4 * int(tp[1:])
        d.text((left, height - foot // 2),
               f"hevo-pmsG-3 (Robo3)  {tp} ({hrs} h)   GFP/FITC green {G[0]:.0f}-{G[1]:.0f}, RFP magenta {R[0]:.0f}-{R[1]:.0f}, "
               f"gamma {GAMMA}; one display for every well and timepoint ({'/'.join(tps)}), failed frames excluded from the range.",
               fill=INK2, font=f_foot, anchor="lm")
        canvas.save(f"{out}/plate_{tp}.jpg", quality=90)
        print(f"wrote {out}/plate_{tp}.jpg {canvas.size}", flush=True)
    # zoom: first vs last timepoint, GFP | RFP | merge per zoom well
    rows = []
    for r, c in ZOOM:
        panels = []
        for tp in dict.fromkeys((tps[0], tps[-1])):
            g, rr = crops[(tp, r, c)]
            gn, rn = stretch(g, G), stretch(rr, R)
            tag = f"{r}{c} {'TP0149' if c < 7 else 'TP0388'} {dose_label(DOSE[c])} {tp} ({4 * int(tp[1:])} h)"
            tag += f"  {flags[(tp, r, c)]}" if flags[(tp, r, c)] else ""
            for img, sub in ((np.repeat((gn * 255).astype(np.uint8)[..., None], 3, 2), "GFP"),
                             (np.repeat((rn * 255).astype(np.uint8)[..., None], 3, 2), "RFP"),
                             ((np.stack([rn, gn, rn], -1) * 255).astype(np.uint8), "merge")):
                p = Image.fromarray(img); ImageDraw.Draw(p).text((6, 4), f"{tag} {sub}", fill=(255, 255, 255), font=font(14, True),
                                                                     stroke_width=2, stroke_fill=(0, 0, 0))
                panels.append(np.pad(np.array(p), ((0, 6), (0, 6), (0, 0)), constant_values=30))
        rows.append(np.concatenate(panels, 1))
    z = Image.fromarray(np.concatenate(rows, 0))
    zc = Image.new("RGB", (z.width, z.height + 30), BG); zc.paste(z, (0, 0))
    ImageDraw.Draw(zc).text((8, z.height + 15), f"Center 1024 px crop at half res. Same display as the plate views: GFP {G[0]:.0f}-{G[1]:.0f}, "
                            f"RFP {R[0]:.0f}-{R[1]:.0f}, gamma {GAMMA}.", fill=INK2, font=f_foot, anchor="lm")
    zc.save(f"{out}/zoom_{tps[0]}_vs_{tps[-1]}.jpg", quality=90)
    print(f"wrote {out}/zoom_{tps[0]}_vs_{tps[-1]}.jpg", flush=True)
    json.dump(dict(fitc=G, rfp=R, gamma=GAMMA, timepoints=tps,
                   failed_frames=sorted(f"{r}{c} {tp}: {v}" for (tp, r, c), v in flags.items() if v)),
              open(f"{out}/plate_display_settings.json", "w"), indent=1)


if __name__ == "__main__":
    main()
