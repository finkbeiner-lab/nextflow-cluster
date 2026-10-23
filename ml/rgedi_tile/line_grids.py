"""Render the per-line overview + zoom grids for BOTH plates with ONE display setting per channel,
computed from the pooled pixels of every panel of both experiments (each panel weighted equally)."""
import glob, json, numpy as np, tifffile
from PIL import Image, ImageDraw
RAW = {"pmsG-1": "/gladstone/finkbeiner/robodata/Robo4Images/hevo-pmsG-1",
       "pmsG-2": "/gladstone/finkbeiner/robodata/Robo4Images/hevo-pmsG-2"}
GRID = [[1, 2, 3], [6, 5, 4], [7, 8, 9]]
GAMMA = 0.7
def mont(exp, w, tp, ch):   # raw robo4_serpentine stitch, 1080 px stride, last tile wins (== MontagedImages)
    m = np.zeros((3360, 3360), np.uint16)
    for i in range(3):
        for j in range(3):
            m[i*1080:i*1080+1200, j*1080:j*1080+1200] = tifffile.imread(glob.glob(f"{RAW[exp]}/{w}/*_{tp}_0-0_{w}_{GRID[i][j]}_{ch}_*.tif")[0])
    return m
PLATES = {
    "pmsG-1": dict(line={"C": "TP0357", "D": "TP0359", "E": "TP0388", "F": "TP0398"},
                   cols=[("3", "T0", "500 ms T0"), ("3", "T18", "500 ms 72h"), ("6", "T0", "10 s T0"),
                         ("6", "T6", "10 s 24h"), ("6", "T12", "10 s 48h"), ("6", "T18", "10 s 72h")],
                   zoom=("6", [("T0", "T0"), ("T18", "72h")], "10s")),
    "pmsG-2": dict(line={"C": "TP057", "D": "TP057", "E": "TP059", "F": "TP059"},
                   cols=[("8", "T0", "0 ms T0"), ("8", "T7", "0 ms 28h"), ("3", "T0", "250 ms T0"), ("3", "T7", "250 ms 28h"),
                         ("7", "T0", "5 s T0"), ("7", "T4", "5 s 16h"), ("7", "T7", "5 s 28h")],
                   zoom=("7", [("T0", "T0"), ("T7", "28h")], "5s")),
}
SL = (slice(1168, 2192), slice(1168, 2192))
store, samp_g, samp_r = {}, [], []
for exp, P in PLATES.items():
    need = {(r, c, tp) for r in "CDEF" for c, tp, _ in P["cols"]} | {(r, P["zoom"][0], tp) for r in "CDEF" for tp, _ in P["zoom"][1]}
    for r, c, tp in sorted(need):
        g, rr = mont(exp, f"{r}{c}", tp, "Epi-GFP16"), mont(exp, f"{r}{c}", tp, "Epi-RFP16")
        store[(exp, r, c, tp)] = dict(g6=g[::6, ::6].astype(np.float32), r6=rr[::6, ::6].astype(np.float32),
                                      gz=g[SL].astype(np.float32), rz=rr[SL].astype(np.float32))
        samp_g.append(g[::16, ::16].ravel()); samp_r.append(rr[::16, ::16].ravel())   # equal pixels per panel
gs, rs = np.concatenate(samp_g).astype(np.float32), np.concatenate(samp_r).astype(np.float32)
G = (float(np.percentile(gs, 2)), float(np.percentile(gs, 99.7)))
R = (float(np.percentile(rs, 2)), float(np.percentile(rs, 99.9)))
print(f"shared display: GFP {G[0]:.0f}-{G[1]:.0f}  RFP {R[0]:.0f}-{R[1]:.0f}  gamma {GAMMA}  from {len(samp_g)} panels")
def n(a, lo_hi): return np.clip((a - lo_hi[0]) / (lo_hi[1] - lo_hi[0]), 0, 1) ** GAMMA
def merge(g, r):
    g, r = n(g, G), n(r, R); return (np.stack([r, g, r], -1) * 255).astype(np.uint8)
def gray(a, lo_hi): return (np.repeat(n(a, lo_hi)[..., None], 3, 2) * 255).astype(np.uint8)
def lab(im, txt):
    p = Image.fromarray(im); d = ImageDraw.Draw(p)
    for dx, dy in ((1,1),(-1,-1),(1,-1),(-1,1),(2,2)): d.text((6+dx, 4+dy), txt, fill=(0,0,0))
    d.text((6, 4), txt, fill=(255,255,255)); return np.array(p)
pad = lambda a: np.pad(a, ((0, 6), (0, 6), (0, 0)), constant_values=30)
def footer(img):
    f = np.full((26, img.shape[1], 3), 20, np.uint8); p = Image.fromarray(f); d = ImageDraw.Draw(p)
    d.text((8, 7), f"Same display for every panel of pmsG-1 and pmsG-2: GFP (green) {G[0]:.0f}-{G[1]:.0f}, "
                   f"RFP16 (magenta) {R[0]:.0f}-{R[1]:.0f}, gamma {GAMMA}. Raw counts, no per-panel or per-tile adjustment.", fill=(220,220,220))
    return np.concatenate([img, np.array(p)], 0)
log = []
for exp, P in PLATES.items():
    rows = []
    for r in "CDEF":
        t = []
        for c, tp, name in P["cols"]:
            s = store[(exp, r, c, tp)]
            t.append(lab(merge(s["g6"], s["r6"]), f"{r}{c} {P['line'][r]} | {name}"))
            log.append(dict(exp=exp, panel=f"{r}{c} {tp} overview", gfp_lo=G[0], gfp_hi=G[1], rfp_lo=R[0], rfp_hi=R[1], gamma=GAMMA,
                            gfp_pct_saturated=round(100 * float((s["g6"] >= G[1]).mean()), 2),
                            rfp_pct_saturated=round(100 * float((s["r6"] >= R[1]).mean()), 2)))
        rows.append(np.concatenate([pad(x) for x in t], 1))
    Image.fromarray(footer(np.concatenate(rows, 0))).save(f"/tmp/ah_rt/{exp}_lines_overview.jpg", quality=88)
    rows = []
    zc, zt, zname = P["zoom"]
    for r in "CDEF":
        panels = []
        for tp, name in zt:
            s = store[(exp, r, zc, tp)]; g, rr = s["gz"][::2, ::2], s["rz"][::2, ::2]
            panels += [lab(gray(g, G), f"{r}{zc} {P['line'][r]} {zname} {name} GFP"), lab(gray(rr, R), f"{r}{zc} {P['line'][r]} {zname} {name} RFP"), lab(merge(g, rr), "merge")]
        rows.append(np.concatenate([pad(p) for p in panels], 1))
    Image.fromarray(footer(np.concatenate(rows, 0))).save(f"/tmp/ah_rt/{exp}_lines_zoom.jpg", quality=88)
json.dump(dict(gfp=G, rfp=R, gamma=GAMMA, method="pooled percentiles over all panels of both plates: GFP p2/p99.7, RFP p2/p99.9",
               panels=log), open("/tmp/ah_rt/display_settings.json", "w"), indent=1)
# verification: every panel used the identical numbers
u = {(p["gfp_lo"], p["gfp_hi"], p["rfp_lo"], p["rfp_hi"], p["gamma"]) for p in log}
print(f"distinct display settings across {len(log)} overview panels: {len(u)}")
import collections
for exp in PLATES:
    ps = [p for p in log if p["exp"] == exp]
    print(exp, "max % saturated GFP", max(p["gfp_pct_saturated"] for p in ps), " RFP", max(p["rfp_pct_saturated"] for p in ps))
