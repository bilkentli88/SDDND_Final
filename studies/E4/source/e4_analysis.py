"""Fixed selection rule and descriptive paired summaries; no outcome tuning."""
import itertools,json
from pathlib import Path
import numpy as np

def write(path,value):
 path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(path)

def cell(m,s):return f'{m}|sigma{s:g}'

def choose(rows,c):
 out={}
 for m,s in itertools.product(c['methods'],c['noise_levels']):
  trials=[]
  for lr,b in itertools.product(c['learning_rates'],map(int,c['schedules'])):
   g=[r for r in rows if (r['method'],r['noise_std'],r['learning_rate'],r['budget'])==(m,s,lr,b)]
   eligible=len(g)==len(c['calibration_data_seeds']) and {r['data_seed'] for r in g}==set(c['calibration_data_seeds']) and all(r['status']=='ok' for r in g)
   trials.append(dict(learning_rate=lr,budget=b,eligible=eligible,score=float(np.mean([r['h0.025_validation_noisy_mse'] for r in g])) if eligible else None))
  eligible=[t for t in trials if t['eligible']];best=min(eligible,key=lambda t:(t['score'],t['budget'],t['learning_rate'])) if eligible else None
  out[cell(m,s)]=dict(selected=best,trials=trials,available=best is not None)
 return out

def summarize(rows,direct,c,root):
 import pandas as pd
 outcomes=['absolute_delay_error','h0.00625_test_clean_mse'];blocks=[];medians=[];paired=[]
 for m,s,seed in itertools.product(c['methods'],c['noise_levels'],c['final_data_seeds']):
  g=[r for r in rows if (r['method'],r['noise_std'],r['data_seed'])==(m,s,seed)]
  if len(g)==len(c['optimization_seeds']) and {r['optimization_seed'] for r in g}==set(c['optimization_seeds']) and all(r['status']=='ok' for r in g):
   blocks.append(dict(method=m,noise_std=s,data_seed=seed,**{k:float(np.median([r[k] for r in g])) for k in outcomes}))
 for m,s in itertools.product(c['methods'],c['noise_levels']):
  g=[b for b in blocks if (b['method'],b['noise_std'])==(m,s)]
  if len(g)==len(c['final_data_seeds']):medians.append(dict(method=m,noise_std=s,**{k:float(np.median([b[k] for b in g])) for k in outcomes},**{k+'_range':[min(b[k] for b in g),max(b[k] for b in g)] for k in outcomes}))
 pairs=[('External-C+anchors','External-C'),('Structured-C+anchors','Structured-C'),('Structured-C','External-C'),('Structured-C+anchors','External-C+anchors')]
 for (a,b),s,k in itertools.product(pairs,c['noise_levels'],outcomes):
  av={r['data_seed']:r[k] for r in blocks if (r['method'],r['noise_std'])==(a,s)};bv={r['data_seed']:r[k] for r in blocks if (r['method'],r['noise_std'])==(b,s)}
  complete=set(av)==set(bv)==set(c['final_data_seeds']);vals=[dict(data_seed=d,numerator=av[d],denominator=bv[d],ratio=av[d]/bv[d] if bv[d]!=0 else None,difference=av[d]-bv[d]) for d in c['final_data_seeds']] if complete else []
  ratios=[v['ratio'] for v in vals if v['ratio'] is not None]
  paired.append(dict(numerator_method=a,denominator_method=b,noise_std=s,outcome=k,complete=complete,datasets=vals,median_ratio=float(np.median(ratios)) if complete and len(ratios)==len(vals) else None,ratio_range=[min(ratios),max(ratios)] if ratios else None,median_difference=float(np.median([v['difference'] for v in vals])) if complete else None,dataset_wins=sum(v['difference']<0 for v in vals) if complete else None))
 result=dict(expected_final_fits=len(c['methods'])*len(c['noise_levels'])*len(c['final_data_seeds'])*len(c['optimization_seeds']),recorded_final_fits=len(rows),successful_final_fits=sum(r['status']=='ok' for r in rows),failed_final_fits=sum(r['status']!='ok' for r in rows),direct_estimates=direct,dataset_medians=blocks,method_medians=medians,paired_comparisons=paired,independent_base_noise_datasets=len(c['final_data_seeds']),interpretation='Descriptive paired results. Optimizer repeats and noise levels are not independent datasets; no significance claim.')
 write(root/'summary.json',result);pd.DataFrame(rows).to_csv(root/'final_runs.csv',index=False);pd.DataFrame(blocks).to_csv(root/'dataset_medians.csv',index=False);pd.DataFrame(direct).to_csv(root/'direct_anchors.csv',index=False)
 return result
