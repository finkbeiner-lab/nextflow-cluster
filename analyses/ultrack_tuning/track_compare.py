"""Compare Ultrack vs proximity tracking on F6 for the metric that matters to GEDI:
track length / dropout (fraction of T0 cells tracked to the last frame). Read-only."""
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
D='/gladstone/finkbeiner/home/aholub/GXYTMPS/GXYTMP-hevo-pmsG-1'
WELL='F6'; TMAX=18

# Ultrack tracks
u=pd.read_csv(f'{D}/hevo-pmsG-1_ultrack_tracks.csv'); u=u[u.well==WELL]
utp='timepoint' if 'timepoint' in u.columns else 't'
ug=u.groupby('track_id')[utp].agg(['min','max','nunique'])

# Proximity tracks (from the montage tracking summary, GFP rows)
p=pd.read_csv(f'{D}/hevo-pmsG-1_tracked_montage_summary.csv')
p=p[(p.well==WELL)&(p.MeasurementTag=='Epi-GFP16')]
ptp='timepoint' if 'timepoint' in p.columns else 'Timepoint'
pg=p.groupby('tracked_id')[ptp].agg(['min','max','nunique'])

def stats(g,name):
    n=len(g); t0=g[g['min']==0]; n_t0=len(t0)
    surv=t0[t0['max']==TMAX]; n_surv=len(surv)
    print(f'{name}: tracks={n}  present@T0={n_t0}  '
          f'median_len={g["nunique"].median():.0f}  mean_len={g["nunique"].mean():.1f}  '
          f'full_len(19)={int((g["nunique"]==TMAX+1).sum())}  '
          f'T0->end survivors={n_surv} ({100*n_surv/max(n_t0,1):.0f}% of T0; dropout={100*(1-n_surv/max(n_t0,1)):.0f}%)')
    return g['nunique'].values
uL=stats(ug,'Ultrack   '); pL=stats(pg,'Proximity ')

fig,ax=plt.subplots(1,2,figsize=(13,5))
bins=np.arange(1,21)
ax[0].hist(pL,bins=bins,alpha=.6,label=f'Proximity (n={len(pL)})',color='tab:gray')
ax[0].hist(uL,bins=bins,alpha=.6,label=f'Ultrack (n={len(uL)})',color='tab:blue')
ax[0].set_xlabel('track length (timepoints)'); ax[0].set_ylabel('# tracks'); ax[0].legend(); ax[0].set_title('F6 track-length distribution')
# survival-curve style: fraction of tracks lasting >= k frames (of T0-present)
for L,name,c in [(pL,'Proximity','tab:gray'),(uL,'Ultrack','tab:blue')]:
    ks=np.arange(1,20); frac=[ (L>=k).sum() for k in ks]
    ax[1].plot(ks,frac,marker='o',ms=3,label=name,color=c)
ax[1].set_xlabel('track length >= k timepoints'); ax[1].set_ylabel('# tracks'); ax[1].legend(); ax[1].set_title('F6 track persistence')
fig.suptitle('Ultrack vs proximity tracking — F6 (hevo-pmsG-1)')
fig.tight_layout(); fig.savefig(f'{D}/MINISOG/track_compare_F6.png',dpi=115); print('saved track_compare_F6.png')
