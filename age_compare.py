"""age_compare.py - exploratory comparison of Adaptive vs Youngest First (8 metrics, Holm over 8)."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import wilcoxon, rankdata
ROOT=Path(__file__).resolve().parent
CFG=["n_processes","n_resources","contention"]; SEED="seed"
M={"Recovery cost":"recovery_cost","Work lost":"work_lost","Terminated processes":"terminated","Recovery time":"recovery_time",
   "Throughput":"throughput","Repeat proxy":"repeat_proxy","Max victimisations":"max_victimisations","Jain progress fairness":"jain"}
main=pd.read_csv(ROOT/"main/results_main.csv"); age=pd.read_csv(ROOT/"age_baseline/results_age_baseline.csv")
ad=main[main.policy=="Adaptive"]
def mat(d,c): return d.sort_values(CFG+[SEED]).pivot_table(index=CFG,columns=SEED,values=c).to_numpy()
rng=np.random.default_rng(20261123); rows=[]
for lab,c in M.items():
    d=mat(ad,c)-mat(age,c); cd=d.mean(axis=1)
    try: p=wilcoxon(cd).pvalue
    except ValueError: p=1.0
    x=cd[cd!=0]; r=rankdata(abs(x)); rb=(r[x>0].sum()-r[x<0].sum())/r.sum()
    b=[d[rng.integers(0,27,27)][:,rng.integers(0,30,30)].mean() for _ in range(1500)]
    lo,hi=np.percentile(b,[2.5,97.5])
    rows.append([lab,ad[c].mean(),age[c].mean(),cd.mean(),lo,hi,p,rb])
t=pd.DataFrame(rows,columns=["metric","adaptive","youngest","diff","lo","hi","p","rb"])
o=t.p.argsort().to_numpy(); run=0; h=np.empty(8)
for k,i in enumerate(o): run=max(run,(8-k)*t.p.iloc[i]); h[i]=min(1,run)
t["holm"]=h; t.to_csv(ROOT/"table6_age_baseline.csv",index=False)
print(t.round(5).to_string())
for c in ["recovery_cost","terminated","max_victimisations","repeat_proxy"]:
    print(c, age.groupby("contention")[c].mean().round(2).to_dict())
