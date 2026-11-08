"""Independently replay checkpoints and recompute the completed E1 summaries."""
import argparse
import json
from pathlib import Path
import statistics
import sys

import numpy as np
import pandas as pd
import torch

import run_experiment1 as runner

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--repo',required=True,type=Path)
parser.add_argument('--results',required=True,type=Path)
args=parser.parse_args()
output=args.results.resolve()
config=runner.load_config(runner.HERE/'protocol.json')
manifest=json.loads((output/'manifest.json').read_text())
assert manifest['signature']['execution_config']['execution_mode']=='full'
assert manifest['signature']['runner_sha256']==runner.sha(runner.HERE/'run_experiment1.py')
assert manifest['signature']['protocol_sha256']==runner.sha(runner.HERE/'ASOC_Experiment1_Protocol.md')
assert manifest['signature']['config_file_sha256']==runner.sha(runner.HERE/'protocol.json')
data,source=runner.load_data_module(args.repo,config)
sys.path.insert(0,str(source))
import sdde_core as core

rows=json.loads((output/'run_records.json').read_text())
assert len(rows)==70 and all(r['status']=='ok' for r in rows)
keys={(r['data_seed'],r['optimization_seed'],r['method']) for r in rows}
assert len(keys)==70
assert len(list((output/'checkpoints').glob('*.pt')))==60
assert len(list((output/'curves').glob('*.csv')))==70
assert len(list((output/'observations').glob('*.npz')))==5
grid=np.linspace(*config['delay_grid'])
paired_hash_checks=0
checkpoint_checks=trajectory_checks=0
raw_data={}
for seed in config['data_seeds']:
    saved=np.load(output/'observations'/f'data{seed}.npz')
    clean,_=data.make_strong_dataset(tuple(config['train_states']))
    noisy=data.add_observation_noise(clean,config['noise_std'],seed)
    assert np.array_equal(clean,saved['clean_train'])
    assert np.array_equal(noisy,saved['noisy_train'])
    states,anchors=data.estimate_profile_anchors(noisy,tuple(config['train_states']))
    assert np.array_equal(states,saved['anchor_states'])
    assert np.array_equal(anchors,saved['anchor_delays'])
    raw_data[seed]={name:saved[name] for name in saved.files}
    for opt in config['optimization_seeds']:
        group=[r for r in rows if r['data_seed']==seed and r['optimization_seed']==opt]
        assert len(group)==4 and len({r['initial_tau_sha256'] for r in group})==1
        paired_hash_checks+=1

checkpoint_hashes={}
for row in rows:
    method,seed,opt=row['method'],row['data_seed'],row['optimization_seed']
    stem=f'{method}_data{seed}' + (f'_opt{opt}' if opt is not None else '')
    curve=pd.read_csv(output/'curves'/f'{stem}.csv')
    assert len(curve)==201 and np.allclose(curve.state,grid,rtol=0,atol=1e-14)
    prediction=curve.predicted_delay.to_numpy()
    if method in runner.NEURAL:
        model=core.RestrictedLearnedModel() if method in ('P3','P4') else core.LearnedTau()
        path=output/'checkpoints'/f'{stem}.pt'
        model.load_state_dict(torch.load(path,map_location='cpu',weights_only=True))
        replay=runner.learned_curve(model,method,grid)
        # pandas writes a float32 column using decimal strings that round-trip
        # to float32; interpreting those strings directly as float64 is not
        # expected to recover extra decimal precision.
        assert np.array_equal(replay,prediction.astype(np.float32))
        prediction=replay.astype(np.float64)
        assert row['updates']==(config['anchor_long_updates'] if method=='A2' else 280)
        checkpoint_hashes[path.name]=runner.sha(path)
        checkpoint_checks+=1
        if method in ('P3','P4'):
            saved=raw_data[seed]
            replay_metrics=runner.trajectory_metrics(core,model,saved['clean_train'],saved['clean_test'],saved['noisy_test'])
            assert all(np.isclose(value,row[key],rtol=1e-12,atol=1e-15) for key,value in replay_metrics.items())
            trajectory_checks+=1
    else:
        saved=raw_data[seed]
        direct=runner.direct_curves(grid,saved['anchor_states'],saved['anchor_delays'])
        assert np.allclose(direct[runner.DIRECT.index(method)],prediction,rtol=0,atol=1e-14)
    error=prediction-data.tau_true_np(grid)
    assert np.isclose(np.sqrt(np.mean(error**2)),row['delay_rmse'],rtol=1e-12,atol=1e-15)

summary=json.loads((output/'summary.json').read_text())
assert summary['complete'] and summary['failed_fits']==0
checks={}
for comparator in ('A1','A2','P4','Linear','PCHIP'):
    ratios=[]
    for seed in config['data_seeds']:
        p3=statistics.median(r['delay_rmse'] for r in rows if r['data_seed']==seed and r['method']=='P3')
        control=statistics.median(r['delay_rmse'] for r in rows if r['data_seed']==seed and r['method']==comparator)
        ratios.append(p3/control)
    stored=summary['dataset_comparisons'][comparator]
    assert np.isclose(statistics.median(ratios),stored['median_dataset_ratio'],rtol=1e-14)
    assert sum(r<1 for r in ratios)==stored['datasets_with_P3_improvement']
    meets=statistics.median(ratios)<=config['practical_ratio_threshold'] and sum(r<1 for r in ratios)>=config['practical_min_dataset_wins']
    assert meets==stored['descriptive_criterion_met']
    checks[comparator]=meets
assert summary['consistent_practical_joint_advantage']==(checks['A1'] and checks['A2'])
audit={'status':'passed','checkpoint_delay_curve_replays':checkpoint_checks,
       'full_model_trajectory_metric_replays':trajectory_checks,
       'direct_function_replays':10,'raw_dataset_and_anchor_reconstructions':5,
       'paired_initialization_hash_checks':paired_hash_checks,
       'all_70_delay_RMSE_values_recomputed':True,
       'all_five_dataset_level_comparisons_recomputed':True,
       'protocol_runner_and_config_hashes_unchanged':True,
       'curve_csv_precision':'Neural CSVs checked by exact float32 round-trip; direct curves checked to 1e-14 absolute tolerance.',
       'recorded_failures':0,'checkpoint_sha256':checkpoint_hashes,
       'scope':'Replay and consistency audit of the completed fixed experiment, not independent scientific replication.'}
runner.atomic_json(output/'independent_audit.json',audit)
print(json.dumps({k:v for k,v in audit.items() if k!='checkpoint_sha256'},indent=2))
