"""Dataset-level summaries; optimizer repeats are never treated as datasets."""
import numpy as np
import pandas as pd
from e3_fit import atomic_json

FULL=('P3','P4'); ONLY=('A1','A2'); DIRECT=('Linear','PCHIP')


def specifications(c):
    for obs in c['observation_steps']:
        for seed in c['data_seeds']:
            for method in DIRECT:
                yield obs,None,seed,None,method
            for opt in c['optimization_seeds']:
                for method in ONLY:
                    yield obs,None,seed,opt,method
                for step in c['rollout_steps']:
                    for method in FULL:
                        yield obs,step,seed,opt,method


def key(row):
    return tuple(row[k] for k in ('observation_step','rollout_step','data_seed','optimization_seed','method'))


def summarize(rows,c,output):
    assert len({key(r) for r in rows})==len(rows)
    frame=pd.DataFrame(rows);frame.to_csv(output/'runs.csv',index=False)
    expected=set(specifications(c));lookup={key(r):r for r in rows}
    complete=set(lookup)==expected and all(r['status']=='ok' for r in rows)
    ds=[];values={}
    for obs in c['observation_steps']:
        for method in FULL+ONLY+DIRECT:
            steps=c['rollout_steps'] if method in FULL else [None]
            for step in steps:
                for seed in c['data_seeds']:
                    opts=c['optimization_seeds'] if method not in DIRECT else [None]
                    group=[lookup.get((obs,step,seed,opt,method)) for opt in opts]
                    valid=all(r is not None and r['status']=='ok' for r in group)
                    row={'observation_step':obs,'rollout_step':step,'data_seed':seed,'method':method,
                         'complete':valid,'expected_runs':len(opts),'successful_runs':sum(r is not None and r['status']=='ok' for r in group)}
                    metrics=['delay_rmse','anchor_rmse']+(['refined_clean_test_mse','native_clean_test_mse'] if method in FULL else [])
                    for metric in metrics:
                        row[metric]=float(np.median([r[metric] for r in group])) if valid else None
                    ds.append(row)
                    if valid:values[(obs,step,seed,method)]=row
    pd.DataFrame(ds).to_csv(output/'dataset_summaries.csv',index=False)
    pairs=[]
    def pair(kind,label,metric,seed,nkey,dkey,**extra):
        a,b=values.get(nkey),values.get(dkey)
        valid=a is not None and b is not None and b[metric]>0
        pairs.append({'contrast':kind,'label':label,'metric':metric,'data_seed':seed,
                      'complete':valid,'numerator':a[metric] if a else None,'denominator':b[metric] if b else None,
                      'ratio':a[metric]/b[metric] if valid else None,
                      'difference':a[metric]-b[metric] if valid else None,**extra})
    coarse_obs,fine_obs=c['observation_steps'];coarse_h,fine_h=c['rollout_steps']
    for seed in c['data_seeds']:
        for method in FULL:
            for obs in c['observation_steps']:
                for metric in ('delay_rmse','refined_clean_test_mse'):
                    pair('solver_refinement',f'{method}, obs={obs:g}',metric,seed,
                         (obs,fine_h,seed,method),(obs,coarse_h,seed,method),method=method,observation_step=obs,rollout_step=None)
            for step in c['rollout_steps']:
                for metric in ('delay_rmse','refined_clean_test_mse'):
                    pair('observation_refinement',f'{method}, h={step:g}',metric,seed,
                         (fine_obs,step,seed,method),(coarse_obs,step,seed,method),method=method,observation_step=None,rollout_step=step)
        for method in ONLY+DIRECT:
            pair('observation_refinement',method,'delay_rmse',seed,
                 (fine_obs,None,seed,method),(coarse_obs,None,seed,method),method=method,observation_step=None,rollout_step=None)
        for obs in c['observation_steps']:
            for step in c['rollout_steps']:
                for comparator in ('P4','A1','A2','Linear','PCHIP'):
                    pair('P3_comparison',f'obs={obs:g}, h={step:g}, P3/{comparator}','delay_rmse',seed,
                         (obs,step,seed,'P3'),(obs,step if comparator=='P4' else None,seed,comparator),
                         method='P3',comparator=comparator,observation_step=obs,rollout_step=step)
    pf=pd.DataFrame(pairs);pf.to_csv(output/'dataset_pairs.csv',index=False)
    comparisons=[]
    for (kind,label,metric),group in pf.groupby(['contrast','label','metric'],sort=False):
        valid=bool(complete and len(group)==len(c['data_seeds']) and group.complete.all())
        ratios=group.ratio.dropna().to_numpy()
        comparisons.append({'contrast':kind,'label':label,'metric':metric,'complete':valid,
            'median_paired_ratio':float(np.median(ratios)) if valid else None,
            'minimum_paired_ratio':float(min(ratios)) if valid else None,
            'maximum_paired_ratio':float(max(ratios)) if valid else None,
            'datasets_with_improvement':int((ratios<1).sum()) if valid else None,
            'datasets':len(ratios),'practical_criterion_met':bool(np.median(ratios)<=c['practical_ratio_threshold'] and (ratios<1).sum()>=c['practical_min_dataset_wins']) if valid and kind=='P3_comparison' and len(ratios)==5 else None})
    verdicts=[]
    for obs in c['observation_steps']:
        for step in c['rollout_steps']:
            found=[r for r in comparisons if r['label'] in (f'obs={obs:g}, h={step:g}, P3/A1',f'obs={obs:g}, h={step:g}, P3/A2')]
            verdicts.append({'observation_step':obs,'rollout_step':step,
                'secondary_practical_joint_advantage':all(r['practical_criterion_met'] for r in found) if complete and len(c['data_seeds'])==5 and len(found)==2 else None})
    result={'protocol_id':c['protocol_id'],'mode':c['execution_mode'],'complete':bool(complete),
       'expected_records':len(expected),'actual_records':len(rows),'trained_neural_fits':sum(r['status']=='ok' and r['method'] not in DIRECT for r in rows),
       'direct_functions':sum(r['status']=='ok' and r['method'] in DIRECT for r in rows),
       'numerical_failures':sum(r['status']=='failed' for r in rows),
       'independent_base_noise_datasets':len(c['data_seeds']),'optimization_repeats':len(c['optimization_seeds']),
       'comparisons':comparisons,'secondary_cell_comparisons':verdicts,
       'interpretation':'Descriptive numerical sensitivity; no asymptotic convergence, significance or general-superiority verdict.'}
    atomic_json(output/'summary.json',result)
    return result
