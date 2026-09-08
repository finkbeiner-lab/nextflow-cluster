#!/usr/bin/env python3
"""GEDI2 death analysis v2 — corrected rules.
 (1) per-timepoint tile-median background, EXCLUDING post-death observations (2-pass)
 (2) death = GEDI2 > 0.025 on >=1 frame ; death time = first crossing
 (3) KM CENSORING: every track present+alive at T0 is KEPT. A track lost while still alive
     is CENSORED at its last observed frame (event=0) -- it contributes at-risk time up to
     that point and then leaves the risk set. A lost track is NOT a death and is NOT
     excluded from the cohort. Completeness is judged against each well's own final frame.
 (4) no observations counted after a cell's death
 usage: gedi_v2.py <plate_dir> <hal3|hal4>
"""
import sys, os
import numpy as np, pandas as pd
THR=0.025
BASE=sys.argv[1]; PLATE=sys.argv[2]
TIER_IDX={1:0,2:1,3:2,4:3,5:3,6:0,7:1,8:2,9:2,10:3,11:0,12:1,13:1,14:2,15:3,16:0}
def tiers_h3(w): return [0,500,5000,10000] if w in ("G8","G10") else ([0,1000,2500,7500] if w in ("G9","G11") else [0,0,10000,10000])
H4_TITRATION={"G8":[0,500,5000,10000],"G9":[0,1000,2500,7500],
              "G10":[0,500,5000,10000],"G11":[0,1000,2500,7500]}
def tiers_h4(w): return H4_TITRATION.get(w,[0,0,10000,10000])
tiers = tiers_h3 if PLATE=="hal3" else tiers_h4
# Hal4 special wells: whole-well stim state (B5 no tiles stimulated, B6 all tiles stimulated)
WHOLE_WELL = {"B5":0, "B6":10000} if PLATE=="hal4" else {}
d=pd.read_csv(f"{BASE}/gedi_cells.csv",on_bad_lines="skip")
d=d[d.well!="well"].copy()
for c in ["tile","timepoint","track_id","area","rfp_mean","gfp_mean"]: d[c]=pd.to_numeric(d[c],errors="coerce")
d=d.dropna(subset=["tile","timepoint","rfp_mean","gfp_mean"])
d[["tile","timepoint","track_id"]]=d[["tile","timepoint","track_id"]].astype(int)
d=d.drop_duplicates(["well","tile","timepoint","track_id"])
d["stim"]=[WHOLE_WELL[w] if w in WHOLE_WELL else tiers(w)[TIER_IDX[t]] for w,t in zip(d.well,d.tile)]
d["cell"]=d.well+"_t"+d.tile.astype(str)+"_id"+d.track_id.astype(str)
Tmin=int(d.timepoint.min())
WELL_TMAX=d.groupby("well").timepoint.max().to_dict()   # per-well final frame
d=d.sort_values(["cell","timepoint"]).reset_index(drop=True)

def derive(df, med_mask=None):
    """compute gedi2 using per-timepoint tile median over rows where med_mask is True"""
    src = df if med_mask is None else df[med_mask]
    med = src.groupby(["well","tile","timepoint"]).rfp_mean.median().rename("tileMedRFP")
    out = df.merge(med, on=["well","tile","timepoint"], how="left")
    out["gedi2"]=(out.rfp_mean-out.tileMedRFP)/out.gfp_mean
    return out

def call_death(df):
    g=df.groupby("cell",sort=False)
    ab=df.gedi2>THR
    df=df.assign(_ab=ab,_abtp=np.where(ab,df.timepoint,np.nan))
    g=df.groupby("cell",sort=False)
    agg=g.agg(well=("well","first"),tile=("tile","first"),stim=("stim","first"),
              first_tp=("timepoint","first"),last_tp=("timepoint","last"),
              first_r=("gedi2","first"),n_above=("_ab","sum"),death_tp=("_abtp","min"))
    return df,agg

# ---- pass 1: background over all rows, get provisional deaths ----
d1=derive(d)
d1,agg1=call_death(d1)
prov_death=agg1.death_tp.to_dict()
# mask: keep rows up to and including death frame (drop post-death observations)
dt=d1.cell.map(prov_death)
keep_rows=~(dt.notna() & (d1.timepoint>dt))
# ---- pass 2: recompute background excluding post-death observations ----
d2=derive(d1[keep_rows].drop(columns=["tileMedRFP","gedi2","_ab","_abtp"]))
d2,agg=call_death(d2)
# ---- inclusion rules ----
agg["event"]=(agg.n_above>=1).astype(int)                       # death = >=1 frame above
agg["time"]=np.where(agg.event==1, agg.death_tp, agg.last_tp).astype(float)
present_alive = (agg.first_tp==Tmin) & (agg.first_r<=THR)        # present + alive at T0
# KM CENSORING: keep every present+alive cell. Deaths = event 1 at first crossing;
# tracks lost while alive are CENSORED at their last observed frame (event 0) - never counted as deaths.
well_tmax     = agg.well.map(WELL_TMAX)
agg["censored_early"]=((agg.event==0)&(agg.last_tp<well_tmax)).astype(int)
surv=agg[present_alive].copy()
surv=surv.reset_index()[["cell","well","tile","stim","time","event","last_tp","n_above","censored_early"]]
pm=pd.read_csv(f"{BASE}/platemap.csv"); surv=surv.merge(pm,on="well",how="left")
surv.to_csv(f"{BASE}/gedi_survival_v2.csv",index=False)
# also save truncated per-cell trace (no post-death rows) for plots
d2[["well","tile","timepoint","track_id","cell","stim","kind" if "kind" in d2.columns else "well","rfp_mean","gfp_mean","tileMedRFP","gedi2"]].to_csv(f"{BASE}/gedi_traces_v2.csv",index=False) if False else None
d2.merge(pm,on="well",how="left").to_csv(f"{BASE}/gedi_traces_v2.csv",index=False)
n_all=len(agg); n_pa=int((present_alive).sum()); n_final=len(surv)
print(f"[{PLATE}] tracks={n_all}  present+alive@T0={n_pa}  analysed={n_final}  of which censored-early={int(surv.censored_early.sum())}")
print(f"[{PLATE}] deaths={int(surv.event.sum())} ({100*surv.event.mean():.1f}%)")
