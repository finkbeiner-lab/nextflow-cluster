"""Tune Ultrack on F6: sweep max_distance x appear/disappear penalties, report
track-length / dropout vs the proximity baseline. labels_to_contours computed ONCE
and reused across configs. Metrics only (no tiff writes)."""
import csv, os, shutil
import numpy as np, pandas as pd
import imageio.v3 as iio
from sqlalchemy import create_engine, text
from ultrack import MainConfig, Tracker
from ultrack.utils import labels_to_contours

EXP='hevo-pmsG-1'; WELL='F6'; CH='Epi-GFP16'; TMAX=18
WORK='/gladstone/finkbeiner/home/aholub/GXYTMPS/ULTRACK_SWEEP'
pw=None
for r in csv.DictReader(open('/gladstone/finkbeiner/lab/GALAXY_INFO/pass.csv')):
    if 'pw' in r: pw=r['pw'].strip(); break
eng=create_engine(f'postgresql://postgres:{pw}@fb-postgres01.gladstone.internal:5432/galaxy')
with eng.connect() as c:
    xid=c.execute(text('SELECT id FROM experimentdata WHERE experiment=:e'),{'e':EXP}).scalar()
    adir=c.execute(text('SELECT analysisdir FROM experimentdata WHERE id=:x'),{'x':xid}).scalar()
    rows=c.execute(text("""SELECT t.timepoint, t.alignedmontagemaskpath FROM tiledata t
      JOIN channeldata ch ON t.channeldata_id=ch.id JOIN welldata w ON t.welldata_id=w.id
      WHERE w.well=:w AND ch.channel=:c AND t.experimentdata_id=:x AND t.alignedmontagemaskpath IS NOT NULL
      ORDER BY t.timepoint"""),{'w':WELL,'c':CH,'x':xid}).fetchall()
seen={}
for tp,mp in rows:
    if mp and os.path.exists(mp) and int(tp) not in seen: seen[int(tp)]=mp
items=sorted(seen.items()); tps=[t for t,_ in items]
labels=np.stack([iio.imread(mp).astype(np.int32) for _,mp in items],0)
print(f'{WELL}: {len(tps)} timepoints, {labels.shape}', flush=True)
foreground,contours=labels_to_contours(labels, sigma=1.0)

def metrics(gg):
    g=gg.groupby('track_id')['t'].agg(['min','max','nunique'])
    n=len(g); t0=g[g['min']==0]; nt0=len(t0); surv=int((t0['max']==TMAX).sum())
    return dict(tracks=n, t0=nt0, med=float(g['nunique'].median()),
                surv=surv, survpct=100*surv/max(nt0,1))

# proximity baseline
p=pd.read_csv(f'{adir}/{EXP}_tracked_montage_summary.csv')
p=p[(p.well==WELL)&(p.MeasurementTag==CH)]
ptp='timepoint' if 'timepoint' in p.columns else 'Timepoint'
pg=p.rename(columns={'tracked_id':'track_id',ptp:'t'})[['track_id','t']]
pm=metrics(pg)
print(f"\n{'config':<34}{'tracks':>8}{'@T0':>6}{'med_len':>9}{'T0->end%':>10}")
print(f"{'PROXIMITY (md450+motion)':<34}{pm['tracks']:>8}{pm['t0']:>6}{pm['med']:>9.0f}{pm['survpct']:>9.0f}%")

grid=[(md,aw) for md in (300,450,600) for aw in (-0.001,-1.0)]
best=None
for md,aw in grid:
    wd=f'{WORK}/md{md}_aw{aw}'; shutil.rmtree(wd,ignore_errors=True); os.makedirs(wd,exist_ok=True)
    cfg=MainConfig(); cfg.data_config.working_dir=wd
    cfg.segmentation_config.min_area=500; cfg.segmentation_config.max_area=40000
    cfg.linking_config.max_distance=md; cfg.linking_config.max_neighbors=5
    cfg.tracking_config.solver_name=''
    cfg.tracking_config.appear_weight=aw; cfg.tracking_config.disappear_weight=aw
    tr=Tracker(cfg); tr.track(foreground=foreground, contours=contours, overwrite='all')
    tdf,_=tr.to_tracks_layer(include_parents=True)
    tcol='t' if 't' in tdf.columns else 'time'; tdf=tdf.rename(columns={tcol:'t'})
    m=metrics(tdf)
    tag=f'md={md} appear/disappear={aw}'
    print(f"{tag:<34}{m['tracks']:>8}{m['t0']:>6}{m['med']:>9.0f}{m['survpct']:>9.0f}%", flush=True)
    if best is None or m['survpct']>best[1]: best=(tag,m['survpct'])
    shutil.rmtree(wd,ignore_errors=True)
print(f"\nBEST Ultrack config by T0->end survival: {best[0]} ({best[1]:.0f}%) vs proximity {pm['survpct']:.0f}%")
