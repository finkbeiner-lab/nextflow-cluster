"""GEDI2 (Jeremy v3) on MONTAGED data — measurement + Plot 1 (threshold-setting).

Faithful to gedi_reference/: per-tile per-timepoint background, but the 'tile' is the
source 3x3 sub-tile recovered from each cell's montage centroid (PER-OBSERVATION), and
rfp/gfp are re-measured as the LARGEST CONNECTED COMPONENT of each tracked cell's seg
label. Produces gedi_cells.csv (well,tile,timepoint,track_id,area,rfp_mean,gfp_mean) +
platemap.csv, then Plot 1: pass-1 GEDI2 across time faceted by condition, NO threshold
line (per rules Section 3 — the user sets the threshold from this plot).

Usage: gedi_montage_v1.py <out_dir> <wells|all>
"""
import sys, os, csv
import numpy as np, pandas as pd
import imageio.v3 as iio
from skimage import measure
from scipy import ndimage
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from sqlalchemy import create_engine, text

EXP='8b244215-31b5-410b-9d00-8837ad9e1d71'
GFP='Epi-GFP16'; RFP='Epi-RFP16-2'
OUT=sys.argv[1]; WELLS=sys.argv[2] if len(sys.argv)>2 else 'all'
os.makedirs(OUT, exist_ok=True)
# montage geometry (robo4_serpentine, 1200px tiles, 10% overlap, last-tile-wins)
TILE=1200; OV=120; STRIDE=TILE-OV; GRID=[[1,2,3],[6,5,4],[7,8,9]]
def tile_of(cy,cx):
    i=min(int(cy//STRIDE),2); j=min(int(cx//STRIDE),2); return GRID[i][j]

pw=None
for r in csv.DictReader(open('/gladstone/finkbeiner/lab/GALAXY_INFO/pass.csv')):
    if 'pw' in r: pw=r['pw'].strip(); break
eng=create_engine(f'postgresql://postgres:{pw}@fb-postgres01.gladstone.internal:5432/galaxy')

# ---- platemap + analysisdir + frame_time (real hours from time_imaged) ----
with eng.connect() as c:
    adir=c.execute(text('SELECT analysisdir FROM experimentdata WHERE id=:x'),{'x':EXP}).scalar()
    wells=c.execute(text('SELECT DISTINCT w.well, w.celltype, w.condition FROM welldata w WHERE w.experimentdata_id=:x ORDER BY 1'),{'x':EXP}).fetchall()
pm=pd.DataFrame(wells, columns=['well','celltype','condition'])
pm['kind']=pm['celltype'].astype(str)+'|'+pm['condition'].astype(str)   # our 'condition' = cell-line | dose
pm.to_csv(f'{OUT}/platemap.csv', index=False)
well_list=sorted(pm.well) if WELLS=='all' else WELLS.split(',')

def paths(well, tp, ch):
    with eng.connect() as c:
        row=c.execute(text("""SELECT t.alignedmontagemaskpath, t.newimagemontage FROM tiledata t
          JOIN channeldata ch ON t.channeldata_id=ch.id JOIN welldata w ON t.welldata_id=w.id
          WHERE w.well=:w AND ch.channel=:c AND t.timepoint=:t AND t.experimentdata_id=:x LIMIT 1"""),
          {'w':well,'c':ch,'t':tp,'x':EXP}).fetchone()
    return (row[0], row[1]) if row else (None,None)

# tracking CSV -> per (well,tp,track_id) centroid (from GFP rows)
csvp=os.path.join(adir, 'hevo-pmsG-1_tracked_montage_summary.csv')
trk=pd.read_csv(csvp)
tcol='timepoint' if 'timepoint' in trk.columns else 'Timepoint'
trk=trk[trk.MeasurementTag==GFP][['well','tracked_id',tcol,'centroid_x','centroid_y']].rename(columns={tcol:'tp'})
trk=trk.dropna(subset=['centroid_x','centroid_y']); trk['tp']=trk.tp.astype(int)

def largest_cc_measure(mask, gfp_img, rfp_img, labels_needed):
    """per seg label in labels_needed: largest connected component -> area, gfp_mean, rfp_mean."""
    out={}
    props={p.label:p for p in measure.regionprops(mask)}
    for L in labels_needed:
        p=props.get(int(L))
        if p is None: continue
        minr,minc,maxr,maxc=p.bbox
        crop=(mask[minr:maxr,minc:maxc]==L)
        cc,n=ndimage.label(crop)
        if n==0: continue
        if n>1:
            big=np.argmax(np.bincount(cc.ravel())[1:])+1; sel=cc==big
        else:
            sel=crop
        ys,xs=np.where(sel); ys=ys+minr; xs=xs+minc
        out[int(L)]=(len(ys), float(gfp_img[ys,xs].mean()), float(rfp_img[ys,xs].mean()))
    return out

rows=[]
for well in well_list:
    tps=sorted(int(x) for x in trk[trk.well==well].tp.unique())
    for tp in tps:
        mpath,_=paths(well,tp,GFP)                    # seg mask (GFP)
        _,gimg=paths(well,tp,GFP)                     # GFP intensity montage
        _,rimg=paths(well,tp,RFP)                     # RFP-2 intensity montage
        if not (mpath and gimg and rimg and os.path.exists(mpath) and os.path.exists(gimg) and os.path.exists(rimg)):
            continue
        mask=iio.imread(mpath).astype(np.int32)
        G=iio.imread(gimg).astype(np.float32); R=iio.imread(rimg).astype(np.float32)
        H,W=mask.shape
        sub=trk[(trk.well==well)&(trk.tp==tp)]
        # map each tracked cell centroid -> seg label
        cyc=np.clip(sub.centroid_y.round().astype(int).values,0,H-1)
        cxc=np.clip(sub.centroid_x.round().astype(int).values,0,W-1)
        seglab=mask[cyc,cxc]
        need=set(int(l) for l in seglab if l>0)
        meas=largest_cc_measure(mask,G,R,need)
        for tid,cy,cx,L in zip(sub.tracked_id.values, cyc, cxc, seglab):
            if L<=0 or int(L) not in meas: continue
            area,gm,rm=meas[int(L)]
            rows.append((well, tile_of(cy,cx), int(tp), int(tid), area, rm, gm))
    print(f"  {well}: {sum(1 for r in rows if r[0]==well)} cell-observations", flush=True)

gc=pd.DataFrame(rows, columns=['well','tile','timepoint','track_id','area','rfp_mean','gfp_mean'])
gc.to_csv(f'{OUT}/gedi_cells.csv', index=False)
print(f"gedi_cells.csv: {len(gc)} rows, {gc.groupby(['well','track_id']).ngroups} tracks", flush=True)

# ---- pass-1 GEDI2 + Plot 1 (NO threshold; user sets it) ----
gc=gc.merge(pm[['well','celltype','condition']], on='well', how='left')
med=gc.groupby(['well','tile','timepoint']).rfp_mean.median().rename('tileMedRFP')
gc=gc.merge(med, on=['well','tile','timepoint'], how='left')
gc['gedi2']=(gc.rfp_mean-gc.tileMedRFP)/gc.gfp_mean
gc['hours']=gc.timepoint*4.0   # fallback; frame_time integration is a follow-up (rules Section 7)

lines=sorted(gc.celltype.dropna().unique()); doses=sorted(gc.condition.dropna().unique())
fig,ax=plt.subplots(len(lines),len(doses),figsize=(4*len(doses),3*len(lines)),squeeze=False,sharex=True,sharey=True)
for i,ln in enumerate(lines):
    for j,ds in enumerate(doses):
        a=ax[i][j]; g=gc[(gc.celltype==ln)&(gc.condition==ds)]
        if len(g): a.scatter(g.hours, g.gedi2, s=2, alpha=.15, color='0.3')
        a.set_ylim(-0.05,0.3)  # zoomed to show the live-cell band edge (rules Section 3.2)
        if i==0: a.set_title(str(ds),fontsize=9)
        if j==0: a.set_ylabel(f'{ln}\nGEDI2',fontsize=9)
        if i==len(lines)-1: a.set_xlabel('hours')
fig.suptitle('Plot 1 — GEDI2 (pass-1) across time, faceted by line x dose (NO threshold: set it from here)',fontsize=12)
fig.tight_layout(rect=[0,0,1,0.97]); fig.savefig(f'{OUT}/plot1_gedi2.png',dpi=115)
print(f"saved {OUT}/plot1_gedi2.png", flush=True)
print("GEDI2 (pass1) percentiles by timepoint (for threshold intuition):")
for tp in [0,4,8,12,16,18]:
    v=gc[gc.timepoint==tp].gedi2
    if len(v): print(f"  T{tp}: p50={v.median():.4f} p90={v.quantile(.9):.4f} p99={v.quantile(.99):.4f} n={len(v)}", flush=True)
