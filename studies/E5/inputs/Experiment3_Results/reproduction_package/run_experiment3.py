"""Fixed factorial resolution study; see ASOC_Experiment3_Protocol.md."""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
from datetime import datetime,timezone
import json
import multiprocessing
from pathlib import Path
import platform
import traceback
import numpy as np
import pandas as pd
import scipy
from scipy.interpolate import PchipInterpolator
import e3_data as data
from e3_fit import sha,atomic_json,state_hash,fit_model,delay_metrics
import e3_summary as summary

HERE=Path(__file__).resolve().parent


def load_config(path):
    c=json.loads(Path(path).read_text())
    assert c['protocol_id']=='ASOC-E3-v1.0'
    assert c['observation_steps']==[.05,.025] and c['rollout_steps']==[.025,.0125]
    assert c['noise_levels']==[.0005] and c['designs']==['informative']
    assert c['training_phases']==[[.5,90],[1.,190]]
    assert c['anchor_only_updates']==280 and c['anchor_long_updates']==1200
    assert len(c['data_seeds'])==len(set(c['data_seeds']))==5
    assert len(c['optimization_seeds'])==len(set(c['optimization_seeds']))==3
    assert set(c['data_seeds']).isdisjoint(c['smoke']['data_seeds'])
    return c


def obs_slug(obs,seed):return f'obs{obs:g}_data{seed}'


def stem(row):
    return obs_slug(row['observation_step'],row['data_seed'])+(
        f"_h{row['rollout_step']:g}" if row['rollout_step'] is not None else '')+'_'+row['method']+(
        f"_opt{row['optimization_seed']}" if row['optimization_seed'] is not None else '')


def evaluate(model,saved,c):
    import torch
    import e3_neural as neural
    result={}
    for step in c['evaluation_steps']:
        ec=data.resolution_config(c,c['fine_observation_step'],step)
        values={}
        with torch.no_grad():
            for split in ('train','test'):
                block=saved[split]
                pred=neural.rollout(model,neural.tensors(block['metadata']),int(round(c['horizon']/ec['observation_step'])),ec).numpy()
                if not np.isfinite(pred).all():raise FloatingPointError('Nonfinite post-fit trajectory')
                values[split]=pred
                result[f'eval_h{step:g}_clean_{split}_mse']=float(np.mean((pred-block['fine_clean'])**2))
                result[f'eval_h{step:g}_observed_{split}_mse']=float(np.mean((pred-block['fine_noisy'])**2))
        result[f'eval_h{step:g}_minimum_predicted_observation_state']=float(min(p.min() for p in values.values()))
        result[f'eval_h{step:g}_nonpositive_observation_count']=int(sum((p<=0).sum() for p in values.values()))
    for split in ('train','test'):
        result[f'native_clean_{split}_mse']=result[f"eval_h{c['rollout_step']:g}_clean_{split}_mse"]
        result[f'refined_clean_{split}_mse']=result[f"eval_h{min(c['evaluation_steps']):g}_clean_{split}_mse"]
    return result


def train_job(row,c,saved,output):
    import torch
    import e3_neural as neural
    try:
        model,fit=fit_model(c,row['method'],row['model_seed'],saved['train']['metadata'],saved['train']['noisy'],saved['anchor'])
        grid=np.linspace(*c['delay_grid'])
        prediction=neural.delay_curve(model,row['method'],grid)
        row.update(status='ok',**fit,**delay_metrics(prediction,grid,c))
        if row['method'] in summary.FULL:row.update(evaluate(model,saved,c))
        torch.save(model.state_dict(),output/'checkpoints'/f'{stem(row)}.pt')
        pd.DataFrame({'state':grid,'predicted_delay':prediction,'true_delay':data.tau_true(grid,c)}).to_csv(output/'curves'/f'{stem(row)}.csv',index=False)
    except (FloatingPointError,RuntimeError) as error:
        row.update(status='failed',error=f'{type(error).__name__}: {error}')
        (output/f'{stem(row)}_failure.txt').write_text(traceback.format_exc())
    return row


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=HERE/'protocol.json')
    parser.add_argument('--output',type=Path,default=HERE/'experiment3_results')
    parser.add_argument('--mode',choices=['smoke','full'],default='full')
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    c=load_config(args.config)
    if args.mode=='smoke':c.update(c['smoke'])
    c['execution_mode']=args.mode
    import torch
    import e3_neural as neural
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=True)
    source_names=['run_experiment3.py','e3_data.py','e3_neural.py','e3_fit.py','e3_summary.py','ASOC_Experiment3_Protocol.md','ASOC_Experiment2_Model_Check.md']
    signature={'config_file_sha256':sha(args.config),'source_sha256':{n:sha(HERE/n) for n in source_names},
        'execution_config':c,'python':platform.python_version(),'torch':torch.__version__,'numpy':np.__version__,'pandas':pd.__version__,'scipy':scipy.__version__}
    manifest_path=output/'manifest.json'
    if manifest_path.exists():
        if not args.resume:raise FileExistsError('Use --resume for an existing output')
        manifest=json.loads(manifest_path.read_text())
        if manifest['signature']!=signature:raise ValueError('Resume source/settings/software mismatch')
    else:
        if any(output.iterdir()):raise FileExistsError('Nonempty uninitialized output')
        atomic_json(manifest_path,{'frozen_before_reference_generation_utc':datetime.now(timezone.utc).isoformat(),
            'signature':signature,'device':'cpu','threads_per_worker':1,'worker_processes':c['workers'],
            'interpretation':'Internal prospective protocol, not public preregistration.'})
    rc=data.resolution_config(c,c['fine_observation_step'])
    reference,convergence=data.refined_reference(rc)
    atomic_json(output/'reference_convergence.json',convergence)
    print('REFERENCE',json.dumps(convergence),flush=True)
    for folder in ('checkpoints','curves','observations'):(output/folder).mkdir(exist_ok=True)
    records=output/'run_records.json'
    rows=json.loads(records.read_text()) if records.exists() else []
    done={summary.key(r) for r in rows};assert len(done)==len(rows)
    for row in rows:
        if row['status']=='ok':
            assert (output/'curves'/f'{stem(row)}.csv').exists()
            if row['method'] not in summary.DIRECT:assert (output/'checkpoints'/f'{stem(row)}.pt').exists()
    cases={};jobs=[]
    for obs in c['observation_steps']:
        for seed in c['data_seeds']:
            saved=data.observed_case(reference,c,obs,seed);cases[(obs,seed)]=saved
            fields={}
            for split in ('train','test'):
                fields.update({split+'_'+k:v for k,v in saved[split].items() if k!='metadata'})
                fields.update({split+'_meta_'+k:v for k,v in saved[split]['metadata'].items()})
            fields.update({'anchor_'+k:v for k,v in saved['anchor'].items()})
            np.savez_compressed(output/'observations'/f'{obs_slug(obs,seed)}.npz',**fields)
    grid=np.linspace(*c['delay_grid'])
    def commit(row):
        rows.append(row);rows.sort(key=stem);atomic_json(records,rows);done.add(summary.key(row))
        print(f"DONE {len(rows)}: {stem(row)} status={row['status']} delay_RMSE={row.get('delay_rmse')}",flush=True)
    for obs,step,seed,opt,method in summary.specifications(c):
        saved=cases[(obs,seed)];anchor=saved['anchor']
        error=anchor['delays']-data.tau_true(anchor['states'],c)
        row={'observation_step':obs,'rollout_step':step,'data_seed':seed,'optimization_seed':opt,
             'method':method,'noise_std':c['noise_levels'][0],'anchor_rmse':float(np.sqrt(np.mean(error**2))),
             'anchor_mae':float(np.mean(np.abs(error))),'valid_anchor_count':int(anchor['eligible'].sum())}
        if summary.key(row) in done:continue
        if method in summary.DIRECT:
            prediction=np.interp(grid,anchor['states'],anchor['delays']) if method=='Linear' else PchipInterpolator(anchor['states'],anchor['delays'],extrapolate=False)(grid)
            row.update(status='ok',**delay_metrics(prediction,grid,c))
            pd.DataFrame({'state':grid,'predicted_delay':prediction,'true_delay':data.tau_true(grid,c)}).to_csv(output/'curves'/f'{stem(row)}.csv',index=False)
            commit(row)
        else:
            row['model_seed']=opt+100*c['data_seeds'].index(seed)
            ec=data.resolution_config(c,obs,step)
            jobs.append((row,ec,saved,output))
    with ProcessPoolExecutor(max_workers=c['workers'],mp_context=multiprocessing.get_context('spawn')) as pool:
        pending={}
        for job in jobs:
            with (output/'events.jsonl').open('a') as f:
                f.write(json.dumps({'event':'fit_submit_or_restart','run':stem(job[0]),'utc':datetime.now(timezone.utc).isoformat()})+'\n')
            pending[pool.submit(train_job,*job)]=job[0]
        for future in as_completed(pending):commit(future.result())
    # Paired initialization check spans methods and all numerical conditions.
    for seed in c['data_seeds']:
        for opt in c['optimization_seeds']:
            group=[r for r in rows if r['data_seed']==seed and r['optimization_seed']==opt and r['status']=='ok']
            assert len({r['initial_tau_hash'] for r in group})<=1
    result=summary.summarize(rows,c,output)
    print(json.dumps({k:v for k,v in result.items() if k!='comparisons'},indent=2),flush=True)
    if not result['complete']:raise SystemExit('Incomplete; all failures retained, complete comparisons withheld')


if __name__=='__main__':main()
