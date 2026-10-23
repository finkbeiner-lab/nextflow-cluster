"""Visual: cell-track trajectories on the F6 montage, proximity vs Ultrack (tuned).
Each track drawn as a line through its centroids over time, colored by track id.
Read-only. Runs in the main SIF (matplotlib)."""
import csv, numpy as np, pandas as pd
import imageio.v3 as iio
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from sqlalchemy import create_engine, text
EXP='hevo-pmsG-1'; WELL='F6'; CH='Epi-GFP16'; BG_TP=9
D='/gladstone/finkbeiner/home/aholub/GXYTMPS/GXYTMP-hevo-pmsG-1'
pw=None
for r in csv.DictReader(open('/gladstone/finkbeiner/lab/GALAXY_INFO/pass.csv')):
    if 'pw' in r: pw=r['pw'].strip(); break
eng=create_engine(f'postgresql://postgres:{pw}@fb-postgres01.gladstone.internal:5432/galaxy')
with eng.connect() as c:
    xid=c.execute(text('SELECT id FROM experimentdata WHERE experiment=:e'),{'e':EXP}).scalar()
    bg=c.execute(text("""SELECT t.newimagemontage FROM tiledata t JOIN channeldata ch ON t.channeldata_id=ch.id
      JOIN welldata w ON t.welldata_id=w.id WHERE w.well=:w AND ch.channel=:c AND t.timepoint=:t
      AND t.experimentdata_id=:x AND t.newimagemontage IS NOT NULL LIMIT 1"""),{'w':WELL,'c':CH,'t':BG_TP,'x':xid}).scalar()
img=iio.imread(bg).astype(np.float32); lo,hi=np.percentile(img,1),np.percentile(img,99.5)
disp=np.clip((img-lo)/(hi-lo+1e-9),0,1)

# proximity trajectories
p=pd.read_csv(f'{D}/{EXP}_tracked_montage_summary.csv'); p=p[(p.well==WELL)&(p.MeasurementTag==CH)]
ptp='timepoint' if 'timepoint' in p.columns else 'Timepoint'
prox=pd.DataFrame({'tid':p.tracked_id.values,'tt':p[ptp].values,'x':p.centroid_x.values,'y':p.centroid_y.values})
# ultrack trajectories (winning config output); the CSV has both 't' (frame idx) and 'timepoint'
u=pd.read_csv(f'{D}/{EXP}_ultrack_tracks.csv'); u=u[u.well==WELL]
utp='timepoint' if 'timepoint' in u.columns else 't'
ult=pd.DataFrame({'tid':u.track_id.values,'tt':u[utp].values,'x':u.x.values,'y':u.y.values})

def draw(ax,df,title):
    ax.imshow(disp,cmap='gray'); ax.axis('off'); ax.set_title(title,fontsize=12)
    rng=np.random.default_rng(0)
    long=df.groupby('tid').tt.nunique(); keep=long[long>=10].index  # only tracks >=10 frames, for legibility
    for tid in keep:
        g=df[df.tid==tid].sort_values('tt')
        ax.plot(g.x,g.y,'-',lw=0.6,alpha=0.8,color=plt.cm.hsv(rng.random()))
    ax.text(0.01,0.99,f'{len(keep)} tracks ≥10 frames',transform=ax.transAxes,va='top',color='w',fontsize=10,
            bbox=dict(fc='k',alpha=.5,pad=2))

fig,ax=plt.subplots(1,2,figsize=(20,10))
draw(ax[0],prox,'Proximity (md450+motion)')
draw(ax[1],ult,'Ultrack (md=600, appear/disappear=-1.0)')
fig.suptitle('F6 track trajectories on the montage — tracks lasting ≥10 timepoints',fontsize=14)
fig.tight_layout(rect=[0,0,1,0.96]); fig.savefig(f'{D}/MINISOG/track_overlay_F6.png',dpi=110)
print('saved track_overlay_F6.png')
