"""Fixed ASOC E2 logistic SDDE study; protocol and model check are co-shipped."""
from __future__ import annotations
import argparse
import copy
from datetime import datetime,timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import time
import traceback

import numpy as np
import pandas as pd
import scipy
from scipy.interpolate import PchipInterpolator

import e2_data as data

HERE=Path(__file__).resolve().parent
NEURAL=('P3','P4','A1','A2')
DIRECT=('Linear','PCHIP')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path,value):
    path=Path(path);temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temp.replace(path)


def load_config(path):
    c=json.loads(Path(path).read_text())
    assert c['protocol_id']=='ASOC-E2-v1.0'
    assert len(c['data_seeds'])==len(set(c['data_seeds']))==5
    assert len(c['optimization_seeds'])==len(set(c['optimization_seeds']))==3
    assert len(c['history_c'])==len(c['history_d'])==4
    assert c['designs']==['informative','degenerate']
    assert tuple(c['neural_methods'])==NEURAL and tuple(c['direct_methods'])==DIRECT
    assert sum(epochs for _,epochs in c['schedule'])==c['anchor_only_updates']==280
    assert c['anchor_long_updates']==1200
    assert set(c['data_seeds']).isdisjoint(c['smoke']['data_seeds'])
    assert c['rollout_step']<c['delay_bounds'][0]
    return c


def state_hash(model):
    digest=hashlib.sha256()
    for key,tensor in model.state_dict().items():
        digest.update(key.encode());digest.update(tensor.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def fit_model(c,method,seed,meta,target,anchor):
    import torch
    import e2_neural as neural
    neural.set_seed(seed)
    model=neural.RestrictedModel(c)
    initial_tau=state_hash(model.tau_net)
    only=method in ('A1','A2')
    if only:
        model=copy.deepcopy(model.tau_net);tau=model
    else:
        tau=model.tau
    parameters=list(model.parameters())
    neural.set_seed(seed+404)
    optimizer=torch.optim.Adam(parameters,lr=c['learning_rate'],weight_decay=c['weight_decay'])
    valid=anchor['eligible']
    assert not only or valid.any()
    states=torch.tensor(anchor['states'][valid],dtype=torch.float32).reshape(-1,1)
    delays=torch.tensor(anchor['delays'][valid],dtype=torch.float32).reshape(-1,1)
    smooth_grid=torch.linspace(*c['smoothness_grid']).reshape(-1,1)
    mt=neural.tensors(meta);y=torch.tensor(target,dtype=torch.float32)
    schedule=[(0,c['anchor_only_updates' if method=='A1' else 'anchor_long_updates'])] if only else c['schedule']
    clipped=updates=0;started=time.perf_counter()
    for steps,epochs in schedule:
        weights=torch.linspace(0.6,1.4,steps+1).reshape(1,-1) if steps else None
        for _ in range(epochs):
            optimizer.zero_grad(set_to_none=True)
            if only:
                loss=c['anchor_weight']*torch.mean((tau(states)-delays)**2)
            else:
                prediction=neural.rollout(model,mt,steps,c)
                loss=torch.mean(weights*(prediction-y[:,:steps+1])**2)
                if method=='P3' and valid.any():
                    loss=loss+c['anchor_weight']*torch.mean((tau(states)-delays)**2)
            curve=tau(smooth_grid)
            loss=loss+c['smoothness_weight']*torch.mean((curve[1:]-curve[:-1])**2)
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite training objective')
            loss.backward()
            norm=torch.nn.utils.clip_grad_norm_(parameters,c['gradient_clip'],error_if_nonfinite=True)
            clipped+=int(float(norm)>c['gradient_clip']);optimizer.step();updates+=1
    if not all(torch.isfinite(p).all() for p in parameters):
        raise FloatingPointError('Nonfinite final parameter')
    with torch.no_grad():
        anchor_mse=float(torch.mean((tau(states)-delays)**2)) if valid.any() else None
        curve=tau(smooth_grid)
        smoothness=float(torch.mean((curve[1:]-curve[:-1])**2))
        trajectory_fit=None
        if not only:
            steps=c['schedule'][-1][0]
            prediction=neural.rollout(model,mt,steps,c)
            weights=torch.linspace(0.6,1.4,steps+1).reshape(1,-1)
            trajectory_fit=float(torch.mean(weights*(prediction-y[:,:steps+1])**2))
        objective=(trajectory_fit or 0.)+(c['anchor_weight']*(anchor_mse or 0.) if method!='P4' else 0.)+c['smoothness_weight']*smoothness
        if not all(math.isfinite(v) for v in (smoothness,objective)):
            raise FloatingPointError('Nonfinite final objective')
    return model,{'initial_tau_hash':initial_tau,'final_parameters_hash':state_hash(model),
       'updates':updates,'clipped_updates':clipped,'training_seconds':time.perf_counter()-started,
       'anchor_fit_mse':anchor_mse,'final_smoothness':smoothness,
       'final_trajectory_fit':trajectory_fit,'final_objective':objective,
       'parameter_count':sum(p.numel() for p in parameters)}


def delay_metrics(prediction,grid,c):
    if not np.isfinite(prediction).all():
        raise FloatingPointError('Nonfinite delay prediction')
    error=prediction-data.tau_true(grid,c)
    return {'delay_rmse':float(np.sqrt(np.mean(error**2))),
       'delay_mae':float(np.mean(np.abs(error))),
       'delay_max_abs_error':float(np.max(np.abs(error)))}


def trajectory_metrics(model,train,test,noisy_train,noisy_test,c):
    import torch
    import e2_neural as neural
    with torch.no_grad():
        steps=int(round(c['horizon']/c['observation_step']))
        a=neural.rollout(model,neural.tensors(train['metadata']),steps,c).numpy()
        b=neural.rollout(model,neural.tensors(test['metadata']),steps,c).numpy()
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise FloatingPointError('Nonfinite trajectory prediction')
    return {'clean_train_mse':float(np.mean((a-train['clean'])**2)),
       'observed_train_mse':float(np.mean((a-noisy_train)**2)),
       'clean_test_mse':float(np.mean((b-test['clean'])**2)),
       'observed_test_mse':float(np.mean((b-noisy_test)**2)),
       'minimum_predicted_state':float(min(a.min(),b.min())),
       'nonpositive_prediction_count':int((a<=0).sum()+(b<=0).sum())}


def slug(design,sigma,seed):
    return f'{design}_sigma{sigma:.4f}_data{seed}'


def summarize(rows,c,output):
    frame=pd.DataFrame(rows);frame.to_csv(output/'runs.csv',index=False)
    ndata,nopt=len(c['data_seeds']),len(c['optimization_seeds'])
    ds_rows=[];pair_rows=[];per_noise={};complete=True
    for sigma in c['noise_levels']:
        pairs={comp:[] for comp in ('A1','A2','P4','Linear','PCHIP')}
        for seed in c['data_seeds']:
            values={}
            for method in NEURAL+DIRECT:
                group=frame[frame.design.eq('informative') & frame.noise_std.eq(sigma) & frame.data_seed.eq(seed) & frame.method.eq(method)]
                expected=nopt if method in NEURAL else 1
                good=group[group.status.eq('ok')]
                valid=len(group)==len(good)==expected
                complete &= valid
                value=float(good.delay_rmse.median()) if valid else None
                if valid:
                    values[method]=value
                ds_rows.append({'noise_std':sigma,'data_seed':seed,'method':method,
                    'expected_runs':expected,'successful_runs':len(good),'complete':valid,
                    'median_delay_rmse':value})
            for comparator in pairs:
                valid='P3' in values and comparator in values and values[comparator]>0
                ratio=values['P3']/values[comparator] if valid else None
                if valid:
                    pairs[comparator].append(ratio)
                pair_rows.append({'noise_std':sigma,'data_seed':seed,'comparator':comparator,
                    'complete':valid,'P3_rmse':values.get('P3'),'comparator_rmse':values.get(comparator),
                    'ratio':ratio,'difference':values['P3']-values[comparator] if valid else None})
        comparisons={}
        for comparator,ratios in pairs.items():
            comparisons[comparator]={'complete_datasets':len(ratios),
              'median_dataset_ratio':float(np.median(ratios)) if ratios else None,
              'min_dataset_ratio':float(min(ratios)) if ratios else None,
              'max_dataset_ratio':float(max(ratios)) if ratios else None,
              'datasets_with_P3_improvement':sum(r<1 for r in ratios),
              'descriptive_criterion_met':bool(len(ratios)==ndata==5 and np.median(ratios)<=c['practical_ratio_threshold'] and sum(r<1 for r in ratios)>=c['practical_min_dataset_wins'])}
        computable=all(comparisons[m]['complete_datasets']==ndata for m in ('A1','A2'))
        verdict=bool(comparisons['A1']['descriptive_criterion_met'] and comparisons['A2']['descriptive_criterion_met']) if complete and computable and ndata==5 else None
        per_noise[f'{sigma:.4f}']={'dataset_comparisons':comparisons,'consistent_practical_joint_advantage':verdict}
    pd.DataFrame(ds_rows).to_csv(output/'dataset_summaries.csv',index=False)
    pd.DataFrame(pair_rows).to_csv(output/'dataset_pairs.csv',index=False)
    trained=abstained=failed=0
    degenerate_equal=[]
    for sigma in c['noise_levels']:
        for design in c['designs']:
            for seed in c['data_seeds']:
                for method in NEURAL+DIRECT:
                    group=frame[frame.design.eq(design)&frame.noise_std.eq(sigma)&frame.data_seed.eq(seed)&frame.method.eq(method)]
                    expected=nopt if method in NEURAL else 1
                    status='abstain_structural' if design=='degenerate' and method in ('A1','A2')+DIRECT else 'ok'
                    complete &= len(group)==expected and group.status.eq(status).all()
                if design=='degenerate':
                    for opt in c['optimization_seeds']:
                        group=frame[frame.design.eq(design)&frame.noise_std.eq(sigma)&frame.data_seed.eq(seed)&frame.optimization_seed.eq(opt)&frame.method.isin(('P3','P4'))]
                        equal=len(group)==2 and group.status.eq('ok').all() and group.final_parameters_hash.nunique()==1
                        degenerate_equal.append(bool(equal))
    trained=int((frame.method.isin(NEURAL)&frame.status.eq('ok')).sum())
    abstained=int(frame.status.eq('abstain_structural').sum())
    failed=int(frame.status.eq('failed').sum())
    if not complete:
        for level in per_noise.values():
            level['consistent_practical_joint_advantage']=None
    verdicts=[v['consistent_practical_joint_advantage'] for v in per_noise.values()]
    across=bool(all(verdicts)) if complete and all(v is not None for v in verdicts) and len(c['noise_levels'])==2 else None
    result={'mode':c['execution_mode'],'complete':bool(complete),
      'independent_base_noise_replications':ndata,'optimization_repeats_per_dataset':nopt,
      'conditions_paired_across_noise_and_history':True,
      'trained_neural_fits':trained,'direct_functions':int((frame.method.isin(DIRECT)&frame.status.eq('ok')).sum()),
      'structural_abstention_records':abstained,'numerical_failures':failed,
      'degenerate_P3_P4_pairs_equal':sum(degenerate_equal),
      'degenerate_P3_P4_pairs_expected':len(degenerate_equal),
      'per_noise':per_noise,'consistent_practical_joint_advantage_at_both_noise_levels':across,
      'inference':'Descriptive controlled study. No calibrated confidence or deterministic certification.'}
    atomic_json(output/'summary.json',result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=HERE/'protocol.json')
    parser.add_argument('--output',type=Path,default=HERE/'experiment2_results')
    parser.add_argument('--mode',choices=['full','smoke'],default='full')
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    c=load_config(args.config)
    if args.mode=='smoke':
        c.update(c['smoke'])
    c['execution_mode']=args.mode
    import torch
    import e2_neural as neural
    assert torch.get_num_threads()==1
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=True)
    source_names=['run_experiment2.py','e2_data.py','e2_neural.py','ASOC_Experiment2_Protocol.md','ASOC_Experiment2_Model_Check.md']
    signature={'config_file_sha256':sha(args.config),'source_sha256':{n:sha(HERE/n) for n in source_names},
        'execution_config':c,'python':platform.python_version(),'torch':torch.__version__,
        'numpy':np.__version__,'pandas':pd.__version__,'scipy':scipy.__version__}
    manifest_path=output/'manifest.json'
    if manifest_path.exists():
        if not args.resume:
            raise FileExistsError('Initialized output requires --resume with identical source and settings.')
        manifest=json.loads(manifest_path.read_text())
        if manifest['signature']!=signature:
            raise ValueError('Resume rejected: source, protocol, settings, mode or software changed.')
    else:
        if any(output.iterdir()):
            raise FileExistsError('Nonempty output without manifest')
        atomic_json(manifest_path,{'frozen_before_reference_generation_utc':datetime.now(timezone.utc).isoformat(),
            'signature':signature,'device':'cpu','threads':1,'dtype':'float32',
            'platform':platform.platform(),'interpretation':'Internal prospective protocol; not public preregistration.'})
    reference,ref_record=data.refined_reference(c)
    atomic_json(output/'reference_convergence.json',ref_record)
    print('Reference convergence:',ref_record,flush=True)
    records_path=output/'run_records.json'
    rows=json.loads(records_path.read_text()) if records_path.exists() else []
    key=lambda r:(r['design'],r['noise_std'],r['data_seed'],r['optimization_seed'],r['method'])
    done={key(r) for r in rows}
    assert len(done)==len(rows)
    for folder in ('checkpoints','curves','observations'):
        (output/folder).mkdir(exist_ok=True)
    grid=np.linspace(*c['delay_grid'])
    def commit(row):
        rows.append(row);atomic_json(records_path,rows);done.add(key(row))
    def curve_file(stem,prediction):
        pd.DataFrame({'state':grid,'predicted_delay':prediction,'true_delay':data.tau_true(grid,c)}).to_csv(output/'curves'/f'{stem}.csv',index=False)
    for sigma in c['noise_levels']:
        for design in c['designs']:
            train,test=reference[(design,'train')],reference[(design,'test')]
            for data_index,seed in enumerate(c['data_seeds']):
                case=slug(design,sigma,seed)
                noisy_train=data.add_noise(train['clean'],sigma,seed)
                noisy_test=data.add_noise(test['clean'],sigma,seed+1000)
                anchor=data.estimate_anchors(noisy_train,c['train_states'],design,c)
                assert bool(anchor['eligible'].all())==(design=='informative')
                if design=='degenerate':
                    assert not anchor['eligible'].any()
                np.savez_compressed(output/'observations'/f'{case}.npz',
                    clean_train=train['clean'],noisy_train=noisy_train,clean_test=test['clean'],noisy_test=noisy_test,
                    **{'train_'+k:v for k,v in train['metadata'].items()},
                    **{'test_'+k:v for k,v in test['metadata'].items()},
                    **{'anchor_'+k:v for k,v in anchor.items()})
                valid=anchor['eligible']
                mae=float(np.mean(np.abs(anchor['delays'][valid]-data.tau_true(anchor['states'][valid],c)))) if valid.any() else None
                common={'design':design,'noise_std':sigma,'data_seed':seed,
                    'valid_anchor_count':int(valid.sum()),'anchor_mae':mae,
                    'history_geometry_eligible':design=='informative'}
                for method in DIRECT:
                    row={**common,'optimization_seed':None,'method':method}
                    if key(row) in done:
                        continue
                    if not valid.any():
                        row.update(status='abstain_structural',reason='Zero centered history signature')
                    else:
                        states,delays=anchor['states'][valid],anchor['delays'][valid]
                        prediction=np.interp(grid,states,delays) if method=='Linear' else PchipInterpolator(states,delays,extrapolate=False)(grid)
                        row.update(status='ok',anchor_fit_mse=0.,**delay_metrics(prediction,grid,c))
                        curve_file(f'{case}_{method}',prediction)
                    commit(row)
                for opt in c['optimization_seeds']:
                    hashes=set()
                    for method in NEURAL:
                        row={**common,'optimization_seed':opt,'model_seed':opt+100*data_index,'method':method}
                        stem=f'{case}_{method}_opt{opt}'
                        if key(row) in done:
                            existing=next(r for r in rows if key(r)==key(row))
                            if existing.get('initial_tau_hash'):
                                hashes.add(existing['initial_tau_hash'])
                            if existing['status']=='ok' and not (output/'checkpoints'/f'{stem}.pt').exists():
                                raise FileNotFoundError('Recorded completed checkpoint is missing')
                            continue
                        if method in ('A1','A2') and not valid.any():
                            row.update(status='abstain_structural',reason='No structurally eligible anchors')
                            commit(row);continue
                        with (output/'events.jsonl').open('a') as f:
                            f.write(json.dumps({'event':'fit_start_or_restart','run':stem,'utc':datetime.now(timezone.utc).isoformat()})+'\n')
                        print('TRAIN',stem,flush=True)
                        try:
                            model,fit=fit_model(c,method,row['model_seed'],train['metadata'],noisy_train,anchor)
                            prediction=neural.delay_curve(model,method,grid)
                            metrics=delay_metrics(prediction,grid,c)
                            if method in ('P3','P4'):
                                metrics.update(trajectory_metrics(model,train,test,noisy_train,noisy_test,c))
                            row.update(status='ok',**fit,**metrics)
                            hashes.add(fit['initial_tau_hash'])
                            torch.save(model.state_dict(),output/'checkpoints'/f'{stem}.pt')
                            curve_file(stem,prediction)
                            print(f"DONE {stem}: delay_RMSE={row['delay_rmse']:.6g}",flush=True)
                        except (FloatingPointError,RuntimeError) as error:
                            row.update(status='failed',error=f'{type(error).__name__}: {error}')
                            (output/f'{stem}_failure.txt').write_text(traceback.format_exc())
                            print('FAILED',stem,row['error'],flush=True)
                        commit(row)
                    if len(hashes)>1:
                        raise AssertionError('Paired methods did not start with identical delay parameters')
    result=summarize(rows,c,output)
    print(json.dumps(result,indent=2))
    if not result['complete']:
        raise SystemExit('Incomplete experiment; retain failures and do not issue a complete-case practical verdict.')
    if result['degenerate_P3_P4_pairs_equal']!=result['degenerate_P3_P4_pairs_expected']:
        raise AssertionError('Degenerate P3/P4 equality failed')


if __name__=='__main__':
    main()
