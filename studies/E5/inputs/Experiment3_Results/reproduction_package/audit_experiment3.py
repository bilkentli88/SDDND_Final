"""Reconstruct raw data, replay every checkpoint, and independently recompute summaries."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics
import numpy as np
import pandas as pd
import torch
import e3_data as data
import e3_neural as neural
import e3_summary as summaries
import run_experiment3 as run


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--results',type=Path,required=True);args=p.parse_args()
    out=args.results.resolve();manifest=json.loads((out/'manifest.json').read_text())
    c=manifest['signature']['execution_config']
    assert manifest['signature']['config_file_sha256']==run.sha(run.HERE/'protocol.json')
    for name,digest in manifest['signature']['source_sha256'].items():assert run.sha(run.HERE/name)==digest
    rows=json.loads((out/'run_records.json').read_text());summary=json.loads((out/'summary.json').read_text())
    expected=set(summaries.specifications(c));assert {summaries.key(r) for r in rows}==expected
    assert len(expected)==len(rows)
    reference,convergence=data.refined_reference(data.resolution_config(c,c['fine_observation_step']))
    assert convergence==json.loads((out/'reference_convergence.json').read_text())
    saved={};rawchecks=anchorchecks=0
    for obs in c['observation_steps']:
        for seed in c['data_seeds']:
            block=data.observed_case(reference,c,obs,seed);saved[(obs,seed)]=block
            z=np.load(out/'observations'/f'{run.obs_slug(obs,seed)}.npz')
            for split in ('train','test'):
                for k,v in block[split].items():
                    if k=='metadata':
                        for mk,mv in v.items():np.testing.assert_array_equal(z[split+'_meta_'+mk],mv)
                    else:np.testing.assert_array_equal(z[split+'_'+k],v)
            for k,v in block['anchor'].items():np.testing.assert_array_equal(z['anchor_'+k],v)
            rawchecks+=1;anchorchecks+=len(block['anchor']['states'])
    coarse,fine=c['observation_steps']
    for seed in c['data_seeds']:
        for split in ('train','test'):
            for k in ('clean','noisy'):np.testing.assert_array_equal(saved[(coarse,seed)][split][k],saved[(fine,seed)][split][k][:,::2])
    grid=np.linspace(*c['delay_grid']);checkpoints={};trained=direct=trajectory=failed=0
    for row in rows:
        obs,step,seed,opt,method=summaries.key(row);name=run.stem(row)
        block=saved[(obs,seed)];ec=data.resolution_config(c,obs,step)
        err=block['anchor']['delays']-data.tau_true(block['anchor']['states'],c)
        assert np.isclose(row['anchor_rmse'],np.sqrt(np.mean(err**2)),rtol=1e-14)
        if row['status']=='failed':
            assert (out/f'{name}_failure.txt').exists();failed+=1;continue
        assert row['status']=='ok'
        curve=pd.read_csv(out/'curves'/f'{name}.csv')
        np.testing.assert_allclose(curve.state,grid,rtol=0,atol=1e-14)
        if method not in summaries.DIRECT:
            model=neural.RestrictedModel(ec) if method in summaries.FULL else neural.LearnedTau(ec)
            file=out/'checkpoints'/f'{name}.pt'
            model.load_state_dict(torch.load(file,map_location='cpu',weights_only=True))
            assert run.state_hash(model)==row['final_parameters_hash']
            prediction=neural.delay_curve(model,method,grid)
            np.testing.assert_array_equal(prediction,curve.predicted_delay.to_numpy().astype(np.float32))
            assert row['updates']==(c['anchor_long_updates'] if method=='A2' else c['anchor_only_updates'])
            neural.set_seed(row['model_seed']);initial=neural.RestrictedModel(ec)
            assert run.state_hash(initial.tau_net)==row['initial_tau_hash']
            if method in summaries.FULL:
                metrics=run.evaluate(model,block,ec)
                for k,v in metrics.items():assert np.isclose(v,row[k],rtol=1e-12,atol=1e-15),(name,k,v,row[k])
                trajectory+=len(c['evaluation_steps'])
            checkpoints[file.name]=run.sha(file);trained+=1
            prediction=prediction.astype(np.float64)
        else:
            a=block['anchor']
            prediction=np.interp(grid,a['states'],a['delays']) if method=='Linear' else run.PchipInterpolator(a['states'],a['delays'],extrapolate=False)(grid)
            np.testing.assert_allclose(curve.predicted_delay,prediction,rtol=0,atol=1e-14);direct+=1
        for k,v in run.delay_metrics(prediction,grid,c).items():assert np.isclose(v,row[k],rtol=1e-12,atol=1e-15)
    for seed in c['data_seeds']:
        for opt in c['optimization_seeds']:
            group=[r for r in rows if r['data_seed']==seed and r['optimization_seed']==opt and r['status']=='ok']
            assert len({r['initial_tau_hash'] for r in group})==1
    # Independent statistics from raw records; do not call summarize().
    ds={}
    for r in pd.read_csv(out/'dataset_summaries.csv').to_dict('records'):
        obs,step,seed,method=r['observation_step'],r['rollout_step'],r['data_seed'],r['method']
        step=None if pd.isna(step) else step
        group=[x for x in rows if x['observation_step']==obs and x['rollout_step']==step and x['data_seed']==seed and x['method']==method]
        valid=all(x['status']=='ok' for x in group) and len(group)==(1 if method in summaries.DIRECT else len(c['optimization_seeds']))
        assert bool(r['complete'])==valid
        vals={}
        if valid:
            metrics=['delay_rmse','anchor_rmse']+(['refined_clean_test_mse','native_clean_test_mse'] if method in summaries.FULL else [])
            for m in metrics:
                vals[m]=statistics.median(x[m] for x in group)
                assert np.isclose(vals[m],r[m],rtol=1e-12,atol=1e-16)
            ds[(obs,step,seed,method)]=vals
    pairs=pd.read_csv(out/'dataset_pairs.csv').to_dict('records');groups=defaultdict(list)
    for r in pairs:
        kind,method,seed,metric=r['contrast'],r['method'],r['data_seed'],r['metric']
        obs=r['observation_step'];step=r['rollout_step']
        if kind=='solver_refinement':
            a,b=(obs,c['rollout_steps'][1],seed,method),(obs,c['rollout_steps'][0],seed,method)
        elif kind=='observation_refinement':
            step=None if pd.isna(step) else step
            a,b=(c['observation_steps'][1],step,seed,method),(c['observation_steps'][0],step,seed,method)
        else:
            assert kind=='P3_comparison'
            a=(obs,step,seed,'P3');b=(obs,step if r['comparator']=='P4' else None,seed,r['comparator'])
        if a in ds and b in ds:
            ratio=ds[a][metric]/ds[b][metric]
            assert np.isclose(r['numerator'],ds[a][metric],rtol=1e-12,atol=1e-16)
            assert np.isclose(r['denominator'],ds[b][metric],rtol=1e-12,atol=1e-16)
            assert np.isclose(r['ratio'],ratio,rtol=1e-12,atol=1e-15)
            groups[(kind,r['label'],metric)].append(ratio)
    for r in summary['comparisons']:
        ratios=groups[(r['contrast'],r['label'],r['metric'])]
        if summary['complete']:
            assert len(ratios)==len(c['data_seeds'])
            for key,fn in [('median_paired_ratio',statistics.median),('minimum_paired_ratio',min),('maximum_paired_ratio',max)]:
                assert np.isclose(r[key],fn(ratios),rtol=1e-12)
            assert r['datasets_with_improvement']==sum(x<1 for x in ratios)
            if r['practical_criterion_met'] is not None:
                assert r['practical_criterion_met']==(statistics.median(ratios)<=.8 and sum(x<1 for x in ratios)>=4)
        else:assert r['median_paired_ratio'] is None
    for cell in summary['secondary_cell_comparisons']:
        if summary['complete'] and len(c['data_seeds'])==5:
            prefix=f"obs={cell['observation_step']:g}, h={cell['rollout_step']:g}, P3/"
            a=[r for r in summary['comparisons'] if r['label'] in (prefix+'A1',prefix+'A2')]
            assert len(a)==2 and cell['secondary_practical_joint_advantage']==all(r['practical_criterion_met'] for r in a)
        else:assert cell['secondary_practical_joint_advantage'] is None
    assert len(list((out/'checkpoints').glob('*.pt')))==trained==summary['trained_neural_fits']
    assert len(list((out/'curves').glob('*.csv')))==trained+direct
    assert direct==summary['direct_functions'] and failed==summary['numerical_failures']
    audit={'status':'passed','scope':'Consistency/replay audit, not independent scientific replication.',
           'trained_checkpoint_delay_replays':trained,'frozen_solver_trajectory_metric_sets_replayed':trajectory,
           'direct_function_replays':direct,'raw_observation_sets_reconstructed':rawchecks,
           'anchor_entries_recomputed':anchorchecks,'shared_noise_pairings_verified':True,
           'initial_hashes_reconstructed':trained,'cross_cell_initialization_groups':len(c['data_seeds'])*len(c['optimization_seeds']),
           'numerical_failures_retained':failed,'reference_refinement_reproduced':True,
           'all_dataset_statistics_and_contrasts_recomputed':True,'frozen_sources_unchanged':True,
           'neural_csv_precision':'Exact float32 round-trip','checkpoint_sha256':checkpoints}
    run.atomic_json(out/'independent_audit.json',audit)
    print(json.dumps({k:v for k,v in audit.items() if k!='checkpoint_sha256'},indent=2))


if __name__=='__main__':main()
