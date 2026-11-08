"""Recompute and audit all two-stage outcomes from retained records/checkpoints."""
import json, sys, statistics, hashlib, argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'source'))
import two_stage as exp

parser=argparse.ArgumentParser()
parser.add_argument('--results',type=Path,default=ROOT/'results')
out=parser.parse_args().results
rows=json.loads((out/'run_records.json').read_text())
assert len(rows)==210 and all(r['status']=='ok' for r in rows)
assert len({exp.stem(r) for r in rows})==210
controls=json.loads((out/'control_replays.json').read_text())
assert len(controls)==210
audit=[]
for i,row in enumerate(rows):
    c,saved=exp.prepare(row);m=exp.construct(row,c)
    p=out/'checkpoints'/(exp.stem(row)+'.pt')
    m.load_state_dict(torch.load(p,map_location='cpu',weights_only=True))
    _,source=exp.paths(row['study'])
    cp=source/'checkpoints'/(exp.stem(dict(row,method=row['stage1_method']))+'.pt')
    frozen=torch.load(cp,map_location='cpu',weights_only=True)
    assert all(torch.equal(m.tau_net.state_dict()[k],v) for k,v in frozen.items())
    assert exp.state_hash(m.tau_net)==row['frozen_tau_sha256']
    metrics,grid,delay,truth=exp.eval_model(m,row,c,saved)
    assert all(np.isclose(v,row[k],rtol=1e-12,atol=1e-15) for k,v in metrics.items())
    curve=pd.read_csv(out/'curves'/(exp.stem(row)+'.csv'))
    assert np.array_equal(curve.predicted_delay.to_numpy().astype(np.float32),delay)
    old=exp.archived_row(row,row['stage1_method'])
    assert metrics['delay_rmse']==old['delay_rmse']
    assert row['stage2_updates']==280 and row['total_updates']==old['updates']+280
    assert row['tau_unchanged'] and row['field_parameters_updated']
    audit.append({'run':exp.stem(row),'frozen_delay_exact':True,'forecast_and_delay_replay':True,'checkpoint_sha256':exp.sha(p)})
    if (i+1)%30==0:print('AUDITED',i+1,flush=True)
exp.atomic(out/'audit.json',{'status':'passed','checkpoint_replays':len(audit),'checks':audit,
    'scope':'Consistency audit using the same retained numerical implementation, not independent scientific replication.'})

all_rows=rows+controls
groups={}
for r in all_rows:
    key=(r['study'],r['noise_std'],r['observation_step'],r['rollout_step'],r['data_seed'],r['method'])
    groups.setdefault(key,[]).append(r)
datasets=[]
for key,group in sorted(groups.items()):
    assert len(group)==3
    n,sigma,obs,h,seed,method=key
    datasets.append({'study':n,'noise_std':sigma,'observation_step':obs,'rollout_step':h,'data_seed':seed,'method':method,
        'forecast_mse':statistics.median(r['forecast_mse'] for r in group),
        'delay_rmse':statistics.median(r['delay_rmse'] for r in group)})
pd.DataFrame(datasets).to_csv(out/'dataset_summaries.csv',index=False)
conditions=sorted({(r['study'],r['noise_std'],r['observation_step'],r['rollout_step']) for r in datasets})
pairs=[];summary=[];absolute=[]
for n,sigma,obs,h in conditions:
    selected=[r for r in datasets if (r['study'],r['noise_std'],r['observation_step'],r['rollout_step'])==(n,sigma,obs,h)]
    seeds=sorted({r['data_seed'] for r in selected});assert len(seeds)==5
    d={(r['data_seed'],r['method']):r for r in selected}
    for method in ('P3','P4','T1','T2'):
        absolute.append({'study':n,'noise_std':sigma,'observation_step':obs,'rollout_step':h,'method':method,
            'marginal_median_forecast_mse':statistics.median(d[s,method]['forecast_mse'] for s in seeds),
            'marginal_median_delay_rmse':statistics.median(d[s,method]['delay_rmse'] for s in seeds)})
    for numerator,denominator in [('P3','T1'),('P3','T2'),('P4','T1'),('P4','T2')]:
        for metric in ('forecast_mse','delay_rmse'):
            ratios=[]
            for s in seeds:
                num=d[s,numerator][metric];den=d[s,denominator][metric];ratios.append(num/den)
                pairs.append({'study':n,'noise_std':sigma,'observation_step':obs,'rollout_step':h,'data_seed':s,
                    'numerator':numerator,'denominator':denominator,'metric':metric,'numerator_value':num,'denominator_value':den,'ratio':num/den})
            summary.append({'study':n,'noise_std':sigma,'observation_step':obs,'rollout_step':h,'numerator':numerator,'denominator':denominator,'metric':metric,
                'median_paired_ratio':statistics.median(ratios),'numerator_wins':sum(v<1 for v in ratios),'datasets':5,'min_ratio':min(ratios),'max_ratio':max(ratios)})
pd.DataFrame(pairs).to_csv(out/'dataset_pairs.csv',index=False)
pd.DataFrame(absolute).to_csv(out/'absolute_summary.csv',index=False)
pd.DataFrame(summary).to_csv(out/'paired_summary.csv',index=False)
exp.atomic(out/'summary.json',{'complete':True,'new_field_fits':210,'replayed_joint_controls':210,'failed_fits':0,'conditions':len(conditions),'comparisons':summary,
    'interpretation':'P3/T1 and P3/T2 below one favor joint fitting. Ratios summarize paired dataset medians. No cross-study pooled success claim; no new independent datasets.'})
print(pd.DataFrame(summary).query('numerator=="P3" and metric=="forecast_mse"').to_string(index=False),flush=True)
