"""pmsG-3 (Robo3) per-line grids at one timepoint, one shared display per channel pooled over all panels.
Robo3 stitch: 2048 px tiles, snake 1 2 3 / 6 5 4 / 7 8 9 (solved from edge NCC on 6 wells), 186 px overlap
(edge-NCC optimum; nominal 10% = 205), last tile wins."""
import glob, json, sys, numpy as np, tifffile
from PIL import Image, ImageDraw, ImageFont
RAW = "/gladstone/finkbeiner/robodata/Robo3Images/hevo-pmsG-3"
GRID = [[1, 2, 3], [6, 5, 4], [7, 8, 9]]; TS, OV = 2048, 186; ST = TS - OV; N = 2 * ST + TS
# one or more timepoints; the display range is pooled over all of them so every grid shares it
TPS = sys.argv[1:] or ["T1"]
TP = TPS[-1]
GAMMA = 0.7
LINE = {c: "TP0149" for c in range(2, 7)} | {c: "TP0388" for c in range(7, 12)}
DOSE = {2: 0, 3: 50, 4: 500, 5: 1000, 6: 5000, 7: 0, 8: 50, 9: 500, 10: 1000, 11: 5000}
ZOOM = [("C", 2), ("C", 6), ("C", 7), ("C", 11)]
# failed acquisitions are shown (labelled) but kept out of the display range, otherwise they set the black or
# white point for every real well: dark frames (camera offset only, ~200 counts) and blown-out frames (all 65535)
DARK_P99, BLOWN_P1 = 500, 65000
def dl(ms): return "0 ms" if ms == 0 else (f"{ms // 1000} s" if ms >= 1000 else f"{ms} ms")
def mont(w, tp, ch):
    m = np.zeros((N, N), np.uint16)
    for i in range(3):
        for j in range(3):
            m[i*ST:i*ST+TS, j*ST:j*ST+TS] = tifffile.imread(glob.glob(f"{RAW}/{w}/*_{tp}_0-0_{w}_{GRID[i][j]}_{ch}_*.tif")[0])
    return m
C0 = (N - 1024) // 2; SL = (slice(C0, C0 + 1024), slice(C0, C0 + 1024))
store, sg, sr = {}, [], []
ZTPS = list(dict.fromkeys(["T0", TP]))
need = [(r, c, tp) for tp in TPS for r in "BCDE" for c in range(2, 12)] + [(r, c, t) for r, c in ZOOM for t in ZTPS]
for r, c, tp in dict.fromkeys(need):
    g, rr = mont(f"{r}{c}", tp, "FITC"), mont(f"{r}{c}", tp, "RFP")
    store[(r, c, tp)] = dict(g=g[::10, ::10].astype(np.float32), r=rr[::10, ::10].astype(np.float32),
                             gz=g[SL].astype(np.float32), rz=rr[SL].astype(np.float32))
    gs_ = g[::24, ::24]
    dark = float(np.percentile(gs_, 99)) < DARK_P99
    blown = float(np.percentile(gs_, 1)) >= BLOWN_P1
    store[(r, c, tp)]["dark"] = dark or blown
    store[(r, c, tp)]["flag"] = "DARK FRAME" if dark else ("FITC BLOWN OUT" if blown else "")
    if not (dark or blown):
        sg.append(g[::24, ::24].ravel()); sr.append(rr[::24, ::24].ravel())
G = (float(np.percentile(np.concatenate(sg), 2)), float(np.percentile(np.concatenate(sg), 99.7)))
R = (float(np.percentile(np.concatenate(sr), 2)), float(np.percentile(np.concatenate(sr), 99.9)))
darks = sorted({f"{r}{c} {tp} ({v['flag']})" for (r, c, tp), v in store.items() if v["dark"]})
print(f"pmsG-3 shared display: FITC {G[0]:.0f}-{G[1]:.0f}  RFP {R[0]:.0f}-{R[1]:.0f}  gamma {GAMMA}  from {len(sg)} panels; failed frames (excluded): {darks}")
def n(a, lh): return np.clip((a - lh[0]) / (lh[1] - lh[0]), 0, 1) ** GAMMA
def merge(g, r): g, r = n(g, G), n(r, R); return (np.stack([r, g, r], -1) * 255).astype(np.uint8)
def gray(a, lh): return (np.repeat(n(a, lh)[..., None], 3, 2) * 255).astype(np.uint8)
try: FONT = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 15)
except Exception: FONT = ImageFont.load_default()
def lab(im, txt):
    p = Image.fromarray(im); ImageDraw.Draw(p).text((6, 4), txt, fill=(255, 255, 255), font=FONT, stroke_width=2, stroke_fill=(0, 0, 0)); return np.array(p)
pad = lambda a, right=6: np.pad(a, ((0, 6), (0, right), (0, 0)), constant_values=30)
def footer(img, extra=""):
    f = Image.fromarray(np.full((30, img.shape[1], 3), 20, np.uint8))
    ImageDraw.Draw(f).text((8, 7), f"pmsG-3 (Robo3), every panel same display: FITC/GFP (green) {G[0]:.0f}-{G[1]:.0f}, RFP (magenta) {R[0]:.0f}-{R[1]:.0f}, "
                                   f"gamma {GAMMA}. Raw counts, no per-panel/per-tile adjustment; failed frames (dark / blown out) excluded from the range. {extra}", fill=(220, 220, 220), font=FONT)
    return np.concatenate([img, np.array(f)], 0)
log = []
for TPO in TPS:
  rows = []
  for r in "BCDE":
    t = []
    for c in range(2, 12):
        s = store[(r, c, TPO)]
        t.append(pad(lab(merge(s["g"], s["r"]), f"{r}{c} {LINE[c]} | {dl(DOSE[c])}" + (f"  {s['flag']}" if s["flag"] else "")), 40 if c == 6 else 6))
        log.append(dict(panel=f"{r}{c} {TPO}", fitc_lo=G[0], fitc_hi=G[1], rfp_lo=R[0], rfp_hi=R[1], gamma=GAMMA,
                        fitc_pct_saturated=round(100 * float((s["g"] >= G[1]).mean()), 2), rfp_pct_saturated=round(100 * float((s["r"] >= R[1]).mean()), 2)))
    rows.append(np.concatenate(t, 1))
  hrs = 4 * int(TPO[1:])
  Image.fromarray(footer(np.concatenate(rows, 0), f"{TPO} ({hrs} h). Rows B-E = replicates; cols 2-6 TP0149, 7-11 TP0388. Range pooled over {'/'.join(TPS)}.")).save(f"/tmp/ah_rt/pmsg3_lines_overview_{TPO}.jpg", quality=88)
rows = []
for r, c in ZOOM:
    panels = []
    for tp in ZTPS:
        s = store[(r, c, tp)]; g, rr = s["gz"][::2, ::2], s["rz"][::2, ::2]
        tag = f"{r}{c} {LINE[c]} {dl(DOSE[c])} {tp}" + (f" {s['flag']}" if s["flag"] else "")
        panels += [lab(gray(g, G), f"{tag} GFP"), lab(gray(rr, R), f"{tag} RFP"), lab(merge(g, rr), "merge")]
    rows.append(np.concatenate([pad(p) for p in panels], 1))
Image.fromarray(footer(np.concatenate(rows, 0), "Center 1024 px crop, shown at half res.")).save(f"/tmp/ah_rt/pmsg3_lines_zoom_{TP}.jpg", quality=88)
json.dump(dict(fitc=G, rfp=R, gamma=GAMMA, dark_frame_panels=darks, tile_order=GRID, tile_px=TS, overlap_px=OV, panels=log), open(f"/tmp/ah_rt/pmsg3_display_settings_{'_'.join(TPS)}.json", "w"), indent=1)
print("distinct display settings across panels:", len({(p["fitc_lo"], p["fitc_hi"], p["rfp_lo"], p["rfp_hi"]) for p in log}),
      "| max % saturated FITC", max(p["fitc_pct_saturated"] for p in log), "RFP", max(p["rfp_pct_saturated"] for p in log))
