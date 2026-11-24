import pandas as pd, numpy as np
from scipy.stats import wilcoxon, rankdata
from statsmodels.stats.multitest import multipletests
from pathlib import Path
import json, hashlib

ROOT=Path(__file__).resolve().parent
main=pd.read_csv(ROOT/'main/results_main.csv')
supp=pd.read_csv(ROOT/'supplementary/results_supplementary.csv')
metrics={
 'Recovery cost':'recovery_cost',
 'Work lost':'work_lost',
 'Terminated processes':'terminated',
 'Recovery time':'recovery_time',
 'Throughput':'throughput',
 'Repeat proxy':'repeat_proxy',
 'Max victimisations':'max_victimisations',
 'Jain progress fairness':'jain',
}
baselines=['Minimum Cost','Lowest Priority','Minimum History','Fixed Weight']
# Average paired differences over seeds within each of the 27 configurations.
wide=main.pivot_table(index=['n_processes','n_resources','contention','seed'],columns='policy',values=list(metrics.values()))
config_diffs={}
for label,col in metrics.items():
  for base in baselines:
    dd=(wide[col]['Adaptive']-wide[col][base]).rename('difference').reset_index()
    config_diffs[(label,base)]=dd.groupby(['n_processes','n_resources','contention']).difference.mean().to_numpy()

# Deterministic two-stage bootstrap: resample 27 configurations, then 30 seeds
# within each selected configuration. Each configuration receives equal weight.
rng=np.random.default_rng(20261123)
B=1500
rows=[]
raw_p=[]
for (label,base), cfgvals in config_diffs.items():
  # Wilcoxon on 27 configuration-level paired differences
  stat,p=wilcoxon(cfgvals, alternative='two-sided', zero_method='wilcox', method='auto')
  # matched-pairs rank-biserial correlation, signed in direction of Adaptive-baseline
  nz=cfgvals[cfgvals!=0]
  ranks=rankdata(np.abs(nz),method='average')
  wplus=ranks[nz>0].sum(); wminus=ranks[nz<0].sum()
  rbc=(wplus-wminus)/(wplus+wminus) if (wplus+wminus) else 0.0
  # Bootstrap at run level with 2-stage cluster resampling
  confs=main[['n_processes','n_resources','contention']].drop_duplicates().sort_values(['n_processes','n_resources','contention']).to_records(index=False)
  # Precompute per-config arrays [30 seeds] of paired diffs
  arrays=[]
  for c in confs:
    mask=(main.n_processes==c[0])&(main.n_resources==c[1])&(main.contention==c[2])
    sub=main.loc[mask].pivot(index='seed',columns='policy',values=metrics[label])
    arrays.append((sub['Adaptive']-sub[base]).to_numpy())
  boot=np.empty(B)
  for b in range(B):
    selected=rng.integers(0,len(arrays),size=len(arrays))
    vals=[]
    for j in selected:
      a=arrays[j]
      seed_idx=rng.integers(0,len(a),size=len(a))
      vals.append(a[seed_idx].mean())
    boot[b]=np.mean(vals)
  lo,hi=np.percentile(boot,[2.5,97.5])
  rows.append({'Metric':label,'Comparison':f'Adaptive vs {base}','Mean difference':cfgvals.mean(),
               '95% CI lower':lo,'95% CI upper':hi,'Wilcoxon statistic':stat,'Raw p':p,'Rank-biserial':rbc})
  raw_p.append(p)
res=pd.DataFrame(rows)
res['Holm p']=multipletests(raw_p,method='holm')[1]
res['Sig.']=np.where(res['Holm p']<0.05,'Yes','No')
res.to_csv(ROOT/'table3_statistical_results.csv',index=False,float_format='%.8f')

# Table 4 supplementary means: exact metrics in manuscript's table
supp_metrics=['recovery_cost','work_lost','terminated','repeat_proxy','jain','p95_turnaround']
label_map={'recovery_cost':'Recovery cost','work_lost':'Work lost','terminated':'Terminated','repeat_proxy':'Repeat proxy','jain':'Jain index','p95_turnaround':'P95 turnaround'}
t4=supp.groupby('policy')[supp_metrics].mean().rename(columns=label_map).reset_index().rename(columns={'policy':'Variant'})
order=['Adaptive','Ablation - cost','Ablation - history','Ablation - priority','Ablation - resource','AdaptiveHighWeights','AdaptiveLowWeights']
t4['__order']=t4.Variant.map({v:i for i,v in enumerate(order)})
t4=t4.sort_values('__order').drop(columns='__order')
t4.to_csv(ROOT/'table4_supplementary_means.csv',index=False,float_format='%.6f')

# Main overall summary
main.groupby('policy')[list(metrics.values())+['completed','unfinished','mean_turnaround','p95_turnaround','p99_turnaround']].mean().to_csv(ROOT/'main_policy_means.csv',float_format='%.8f')

# Human-readable run report
meta_main=json.loads((ROOT/'main/results_main.meta.json').read_text())
meta_supp=json.loads((ROOT/'supplementary/results_supplementary.meta.json').read_text())
report=[]
report.append('Deadlock recovery experiment rerun report')
report.append('========================================')
report.append(f"Main runs: {meta_main['rows']} | sanity problems: {len(meta_main['sanity_check_problems'])} | SHA256 CSV: {meta_main['csv_sha256']}")
report.append(f"Supplementary runs: {meta_supp['rows']} | sanity problems: {len(meta_supp['sanity_check_problems'])} | SHA256 CSV: {meta_supp['csv_sha256']}")
means=main.groupby('policy').recovery_cost.mean()
report.append('\nMain recovery cost means:')
for p,v in means.items(): report.append(f'  {p}: {v:.6f}')
report.append(f"\nAdaptive mean recovery cost in this dataset: {means['Adaptive']:.6f}")
report.append('\nTable 3: new paired statistics from stated eight outcomes, 27 configuration-level differences, Wilcoxon + Holm over 32 tests; two-stage bootstrap 1,500 replicates (seed 20261123).')
report.append('Important: exact bootstrap implementation/seed was not present in the supplied materials; these CIs are reproducible new estimates, not claimed to reproduce missing original analysis.py.')
report.append('\nTable 4: means from supplementary 1,890-run dataset; Adaptive default is the matched reference over seeds 0-9.')
(ROOT/'RERUN_REPORT.txt').write_text('\n'.join(report))
print('\n'.join(report))
print('\nTABLE 3 preview:')
print(res[['Metric','Comparison','Mean difference','95% CI lower','95% CI upper','Holm p','Rank-biserial','Sig.']].to_string(index=False))
print('\nTABLE 4:')
print(t4.to_string(index=False))
