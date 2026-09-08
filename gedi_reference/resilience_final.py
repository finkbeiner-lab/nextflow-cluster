import sys, numpy as np, pandas as pd
from math import erfc,sqrt
B=sys.argv[1]; PLATE=sys.argv[2]
L3={"T1":"ALA 10µM","T2":"ALA 3µM","T3":"ALA 1µM","T4":"CA 3µM","T5":"CA 1µM","T6":"CA 0.3µM",
 "T7":"Cocktail 4µg/mL","T8":"Cocktail 0.4µg/mL","T9":"Cocktail 0.04µg/mL","T10":"3-part Lo",
 "T11":"3-part Med","T12":"Hi-ALA","T13":"Hi-CA","T14":"Hi-cocktail","P1":"Sulforaphane 0.1µM",
 "P2":"KI-696 0.1µM","N1":"A485 10µM (+ctrl)","V":"DMSO"}
L4={"V":"DMSO","A":"A485 10µM (+ctrl)","K0":"Cocktail 0.04","K1":"Cocktail 0.4","K2":"Cocktail 0.8","K3":"Cocktail 1.6",
 "C1":"CA 1µM","C2":"CA 3µM","C3":"CA 10µM","C4":"CA 20µM","L1":"ALA 0.03µM","L2":"ALA 0.1µM","L3":"ALA 0.3µM","L4":"ALA 1µM",
 "X1":"Ck0.4+CA1","X2":"Ck0.4+CA3","X3":"Ck0.8+CA1","X4":"Ck0.8+CA3","X5":"Ck0.8+CA10",
 "X6":"Ck0.4+CA1+ALA0.3","X7":"Ck0.4+CA1+ALA0.03","S1":"Sulforaphane 1µM","S2":"Sulforaphane 0.1µM"}
LAB=L3 if PLATE=="hal3" else L4
s=pd.read_csv(f"{B}/gedi_survival_v2.csv"); s["label"]=s.kind.map(LAB)
def km_ci(sub):
    t=np.asarray(sub.time,float);e=np.asarray(sub.event,int);S=1.;vs=0.;nr=len(t)
    for x in np.unique(t):
        d=int(((t==x)&(e==1)).sum());c=int(((t==x)&(e==0)).sum())
        if d>0 and nr>0: S*=(1-d/nr); vs+= d/(nr*(nr-d)) if nr>d else 0
        nr-=d+c
    F=1-S; se=S*sqrt(vs) if vs>0 else 0
    return F, max(0,F-1.96*se), min(1,F+1.96*se)
def lr(a,b):
    t=np.r_[a.time,b.time];e=np.r_[a.event,b.event];g=np.r_[np.zeros(len(a)),np.ones(len(b))]
    tt=np.unique(t[e==1]);O=E=V=0.
    for x in tt:
        n=(t>=x).sum();n1=(t[g==0]>=x).sum();dd=((t==x)&(e==1)).sum();d1=((t[g==0]==x)&(e[g==0]==1)).sum()
        if n<2:continue
        E+=dd*n1/n;O+=d1;V+=dd*(n1/n)*(1-n1/n)*(n-dd)/(n-1) if n>1 else 0
    return erfc(abs((O-E)/sqrt(V))/sqrt(2)) if V>0 else np.nan
def bh(p):
    p=np.asarray(p,float);n=len(p);o=np.argsort(p);q=np.empty(n);prev=1.
    for r,i in enumerate(o[::-1]):
        k=n-r;v=min(prev,p[i]*n/k);q[i]=v;prev=v
    return q
st=s[s.stim==10000]; un=s[s.stim==0]
dmso=st[st.kind=="V"]; base,blo,bhi=km_ci(dmso); dun,_,_=km_ci(un[un.kind=="V"])
print(f"{PLATE.upper()}  DMSO stim death={base:.3f} [{blo:.3f}-{bhi:.3f}] n={len(dmso)} | unstim={dun:.3f} n={len(un[un.kind=='V'])}")
rows=[]
for k in st.kind.dropna().unique():
    if k=="V" or k not in LAB: continue
    sub=st[st.kind==k]
    if len(sub)<20: continue
    d,lo,hi=km_ci(sub); su=un[un.kind==k]
    u=km_ci(su)[0] if len(su)>=20 else np.nan
    rows.append(dict(label=LAB[k],n=len(sub),cens=int(sub.censored_early.sum()),
        unstim=round(u,3),death=round(d,3),ci=f"{lo:.3f}-{hi:.3f}",
        prot_pp=round((base-d)*100,1),rel=round((base-d)/base,3),p=lr(dmso,sub)))
r=pd.DataFrame(rows); r["q"]=bh(r.p.values)
r["sig"]=np.where(r.q<0.001,"***",np.where(r.q<0.01,"**",np.where(r.q<0.05,"*","ns")))
r=r.sort_values("prot_pp",ascending=False)
r.to_csv(f"{B}/resilience_FINAL.csv",index=False)
print(r[["label","n","cens","unstim","death","ci","prot_pp","rel","q","sig"]].to_string(index=False))
