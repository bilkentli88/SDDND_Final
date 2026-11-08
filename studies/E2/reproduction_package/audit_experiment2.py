"""Reconstruct observations and replay E2 checkpoints independently of the fit loop."""
import argparse
import json
from pathlib import Path
import statistics

import numpy as np
import pandas as pd
import torch

import e2_data as data
import e2_neural as neural
import run_experiment2 as run

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--results',required=True,type=Path)
args=parser.parse_args()
output=args.results.resolve()
c=run.load_config(run.HERE/'protocol.json')
manifest=json.loads((output/'manifest.json').read_text())
assert manifest['signature']['execution_config']['execution_mode']=='full'
assert manifest['signature']['config_file_sha256']==run.sha(run.HERE/'protocol.json')
for name,digest in manifest['signature']['source_sha256'].items():
    assert run.sha(run.HERE/name)==digest
rows=json.loads((output/'run_records.json').read_text())
assert len(rows)==280
key=lambda r:(r['design'],r['noise_std'],r['data_seed'],r['optimization_seed'],r['method'])
assert len({key(r) for r in rows})==280
assert sum(r['status']=='abstain_structural' for r in rows)==80
failed=[r for r in rows if r['status']=='failed']
reference,convergence=data.refined_reference(c)
stored_convergence=json.loads((output/'reference_convergence.json').read_text())
assert convergence==stored_convergence
grid=np.linspace(*c['delay_grid'])
saved_cases={};reconstructed=anchor_checks=initial_hash_checks=0
for sigma in c['noise_levels']:
    for design in c['designs']:
        for seed in c['data_seeds']:
            name=run.slug(design,sigma,seed)
            z=np.load(output/'observations'/f'{name}.npz')
            train,test=reference[(design,'train')],reference[(design,'test')]
            np.testing.assert_array_equal(z['clean_train'],train['clean'])
            np.testing.assert_array_equal(z['clean_test'],test['clean'])
            noisy_train=data.add_noise(train['clean'],sigma,seed)
            noisy_test=data.add_noise(test['clean'],sigma,seed+1000)
            np.testing.assert_array_equal(z['noisy_train'],noisy_train)
            np.testing.assert_array_equal(z['noisy_test'],noisy_test)
            anchor=data.estimate_anchors(noisy_train,c['train_states'],design,c)
            for k,value in anchor.items():
                np.testing.assert_allclose(z['anchor_'+k],value,rtol=0,atol=1e-14,equal_nan=True)
            for split,meta in [('train',train['metadata']),('test',test['metadata'])]:
                for k,value in meta.items():
                    np.testing.assert_array_equal(z[split+'_'+k],value)
            assert bool(anchor['eligible'].all())==(design=='informative')
            if design=='degenerate':
                assert not anchor['eligible'].any()
                assert np.isnan(anchor['delays']).all()
            saved_cases[(design,sigma,seed)]={'train':train,'test':test,
                'noisy_train':noisy_train,'noisy_test':noisy_test,'anchor':anchor}
            reconstructed+=1;anchor_checks+=len(anchor['states'])
            for opt in c['optimization_seeds']:
                group=[r for r in rows if r['design']==design and r['noise_std']==sigma and r['data_seed']==seed and r['optimization_seed']==opt and r['method'] in run.NEURAL and r['status']=='ok']
                if group:
                    assert len({r['initial_tau_hash'] for r in group})==1
                    initial_hash_checks+=1

curve_checks=trajectory_checks=direct_checks=0
checkpoint_hashes={};degenerate_states={}
for row in rows:
    design,sigma,seed,method,opt=(row[k] for k in ('design','noise_std','data_seed','method','optimization_seed'))
    case=run.slug(design,sigma,seed)
    stem=f'{case}_{method}'+(f'_opt{opt}' if opt is not None else '')
    saved=saved_cases[(design,sigma,seed)]
    if row['status']=='abstain_structural':
        assert design=='degenerate' and method in ('A1','A2')+run.DIRECT
        assert row['valid_anchor_count']==0 and not row['history_geometry_eligible']
        assert 'delay_rmse' not in row
        continue
    if row['status']=='failed':
        assert method in run.NEURAL
        assert (output/f'{stem}_failure.txt').is_file()
        continue
    curve=pd.read_csv(output/'curves'/f'{stem}.csv')
    assert len(curve)==201
    np.testing.assert_allclose(curve.state,grid,rtol=0,atol=1e-14)
    if method in run.NEURAL:
        model=neural.RestrictedModel(c) if method in ('P3','P4') else neural.LearnedTau(c)
        path=output/'checkpoints'/f'{stem}.pt'
        state=torch.load(path,map_location='cpu',weights_only=True)
        model.load_state_dict(state)
        assert run.state_hash(model)==row['final_parameters_hash']
        prediction=neural.delay_curve(model,method,grid)
        np.testing.assert_array_equal(prediction,curve.predicted_delay.to_numpy().astype(np.float32))
        prediction=prediction.astype(np.float64)
        assert row['updates']==(c['anchor_long_updates'] if method=='A2' else c['anchor_only_updates'])
        checkpoint_hashes[path.name]=run.sha(path)
        curve_checks+=1
        if method in ('P3','P4'):
            metrics=run.trajectory_metrics(model,saved['train'],saved['test'],saved['noisy_train'],saved['noisy_test'],c)
            for k,value in metrics.items():
                assert np.isclose(value,row[k],rtol=1e-12,atol=1e-15)
            trajectory_checks+=1
            if design=='degenerate':
                degenerate_states[(sigma,seed,opt,method)]=state
    else:
        anchor=saved['anchor']
        prediction=np.interp(grid,anchor['states'],anchor['delays']) if method=='Linear' else run.PchipInterpolator(anchor['states'],anchor['delays'],extrapolate=False)(grid)
        np.testing.assert_allclose(prediction,curve.predicted_delay,rtol=0,atol=1e-14)
        direct_checks+=1
    metrics=run.delay_metrics(prediction,grid,c)
    for k,value in metrics.items():
        assert np.isclose(value,row[k],rtol=1e-12,atol=1e-15)

equal_pairs=0
for sigma in c['noise_levels']:
    for seed in c['data_seeds']:
        for opt in c['optimization_seeds']:
            p3=degenerate_states.get((sigma,seed,opt,'P3'))
            p4=degenerate_states.get((sigma,seed,opt,'P4'))
            if p3 is not None and p4 is not None:
                assert set(p3)==set(p4)
                assert all(torch.equal(p3[k],p4[k]) for k in p3)
                equal_pairs+=1

summary=json.loads((output/'summary.json').read_text())
assert summary['numerical_failures']==len(failed)
assert summary['trained_neural_fits']==curve_checks
assert summary['direct_functions']==direct_checks==20
assert summary['degenerate_P3_P4_pairs_equal']==equal_pairs
for sigma in c['noise_levels']:
    level=summary['per_noise'][f'{sigma:.4f}']
    for comparator in ('A1','A2','P4','Linear','PCHIP'):
        ratios=[]
        for seed in c['data_seeds']:
            subset=[r for r in rows if r['design']=='informative' and r['noise_std']==sigma and r['data_seed']==seed]
            a=[r['delay_rmse'] for r in subset if r['method']=='P3' and r['status']=='ok']
            b=[r['delay_rmse'] for r in subset if r['method']==comparator and r['status']=='ok']
            expected=3 if comparator in run.NEURAL else 1
            if len(a)==3 and len(b)==expected:
                ratios.append(statistics.median(a)/statistics.median(b))
        check=level['dataset_comparisons'][comparator]
        assert check['complete_datasets']==len(ratios)
        if ratios:
            assert np.isclose(check['median_dataset_ratio'],statistics.median(ratios),rtol=1e-14)
        assert check['datasets_with_P3_improvement']==sum(r<1 for r in ratios)
        met=len(ratios)==5 and statistics.median(ratios)<=.8 and sum(r<1 for r in ratios)>=4
        assert met==check['descriptive_criterion_met']
    if summary['complete']:
        assert level['consistent_practical_joint_advantage']==all(level['dataset_comparisons'][m]['descriptive_criterion_met'] for m in ('A1','A2'))
    else:
        assert level['consistent_practical_joint_advantage'] is None
if summary['complete']:
    assert summary['consistent_practical_joint_advantage_at_both_noise_levels']==all(v['consistent_practical_joint_advantage'] for v in summary['per_noise'].values())
assert len(list((output/'checkpoints').glob('*.pt')))==curve_checks
assert len(list((output/'curves').glob('*.csv')))==curve_checks+direct_checks
audit={'status':'passed','scope':'Consistency and replay audit, not independent scientific replication.',
       'numerical_failures_retained':len(failed),'trained_checkpoint_replays':curve_checks,
       'trajectory_metric_replays':trajectory_checks,'direct_function_replays':direct_checks,
       'raw_condition_dataset_reconstructions':reconstructed,'anchor_entries_recomputed':anchor_checks,
       'paired_initial_delay_hash_checks':initial_hash_checks,'degenerate_model_pairs_bitwise_equal':equal_pairs,
       'structural_abstention_records_verified':80,'reference_refinement_reproduced':True,
       'condition_level_comparisons_and_decision_recomputed':True,'frozen_source_hashes_unchanged':True,
       'curve_precision':'Neural curves exactly round-trip through float32 CSV; direct curves checked to absolute 1e-14.',
       'checkpoint_sha256':checkpoint_hashes}
run.atomic_json(output/'independent_audit.json',audit)
print(json.dumps({k:v for k,v in audit.items() if k!='checkpoint_sha256'},indent=2))
