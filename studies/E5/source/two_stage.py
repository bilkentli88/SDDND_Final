"""Complete frozen-delay controls on the archived E1--E3 data.

Architecture/rollout sources for E2/E3 are retained unchanged in vendor/.
The E1 adapter implements the published discrete-history Euler scheme and is
checked against stored checkpoints/metrics before the experiment is run.
"""
from __future__ import annotations
import argparse, copy, hashlib, importlib.util, json, math, os, platform
from pathlib import Path
import time, traceback
from datetime import datetime, timezone
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import numpy as np
import pandas as pd
import torch

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
torch.set_num_threads(1)

def module(name):
    spec=importlib.util.spec_from_file_location(name,HERE/'vendor'/f'{name}.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
N2=module('e2_neural');N3=module('e3_neural')

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def state_hash(model):
    h=hashlib.sha256()
    for k,v in model.state_dict().items():h.update(k.encode());h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()
def atomic(path,obj):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n');tmp.replace(path)
def paths(study):
    base=ROOT/'inputs'/f'Experiment{study}_Results'
    return base/'reproduction_package',base/f'experiment{study}_results'
def config(study):return json.loads((paths(study)[0]/'protocol.json').read_text())

class FirstModel(N2.RestrictedModel):
    def derivative(self,x,past):
        return self.a_net(x)+(0.05+torch.nn.functional.softplus(self.b_net(x)))*past

def first_rollout(model,histories,steps):
    # Histories include -0.50,...,0 at the archived 0.05 spacing.
    h=.05;H=10;seq=list(histories.unbind(1));rows=histories.shape[0]
    for i in range(steps):
        x=seq[-1].reshape(rows,1)
        delay=model.tau(x).reshape(-1)
        pos=H+i-delay/h
        left=torch.floor(pos).long().clamp(0,len(seq)-2)
        weight=pos-left.to(pos.dtype)
        values=torch.stack(seq,dim=1)
        past=(1-weight)*values.gather(1,left[:,None]).squeeze(1)+weight*values.gather(1,(left+1)[:,None]).squeeze(1)
        seq.append(seq[-1]+h*model.derivative(x,past[:,None]).squeeze(1))
    return torch.stack(seq,dim=1)

def stem(row):
    n=row['study'];d=row['data_seed'];o=row['optimization_seed'];m=row['method']
    if n==1:return f'{m}_data{d}_opt{o}'
    if n==2:return f'informative_sigma{row["noise_std"]:g}_data{d}_{m}_opt{o}'
    return f'obs{row["observation_step"]:g}_data{d}'+(f'_h{row["rollout_step"]:g}' if m in ('P3','P4','T1','T2') else '')+f'_{m}_opt{o}'

def cases():
    out=[]
    for study in (1,2,3):
        _,res=paths(study)
        rows=json.loads((res/'run_records.json').read_text())
        for r in rows:
            if r['method']!='P3' or r['status']!='ok':continue
            if study==2 and r['design']!='informative':continue
            q={k:r[k] for k in ['data_seed','optimization_seed','model_seed']}
            q.update(study=study,noise_std=(.0005 if study!=2 else r['noise_std']),
                     observation_step=(r['observation_step'] if study==3 else .05),
                     rollout_step=(r['rollout_step'] if study==3 else (.05 if study==1 else .025)))
            out.append(q)
    return sorted(out,key=lambda r:(r['study'],r['noise_std'],r['observation_step'],r['rollout_step'],r['data_seed'],r['optimization_seed']))

def prepare(row):
    n=row['study'];c=config(n);_,res=paths(n)
    if n==1:
        f=res/'observations'/f'data{row["data_seed"]}.npz'
    elif n==2:
        f=res/'observations'/f'informative_sigma{row["noise_std"]:g}_data{row["data_seed"]}.npz'
    else:
        f=res/'observations'/f'obs{row["observation_step"]:g}_data{row["data_seed"]}.npz'
        c['observation_step']=row['observation_step'];c['rollout_step']=row['rollout_step']
        c['schedule']=[(int(round(t/c['observation_step'])),updates) for t,updates in c['training_phases']]
    saved=dict(np.load(f));return c,saved

def construct(row,c):
    N2.set_seed(row['model_seed'])
    if row['study']==1:return FirstModel({'delay_bounds':[.15,.50]})
    return (N2 if row['study']==2 else N3).RestrictedModel(c)

def predict(model,row,c,saved,split,step=None):
    n=row['study']
    if n==1:
        h=torch.tensor(saved[f'clean_{split}'][:,:11],dtype=torch.float32)
        return first_rollout(model,h,20)[:,10:]
    nn=N2 if n==2 else N3
    meta={k:torch.tensor(saved[f'{split}_'+('meta_' if n==3 else '')+k],dtype=torch.float32) for k in ['s','c','d']}
    ec=copy.deepcopy(c)
    if step is not None:ec['rollout_step']=step
    if n==3:ec['observation_step']=.025
    return nn.rollout(model,meta,int(round(1/ec['observation_step'])),ec)

def eval_model(model,row,c,saved):
    result={};n=row['study']
    with torch.no_grad():
        for split in ('train','test'):
            p=predict(model,row,c,saved,split).numpy()
            truth=saved[f'clean_{split}'][:,10:] if n==1 else saved[f'{split}_fine_clean' if n==3 else f'clean_{split}']
            assert p.shape==truth.shape
            if not np.isfinite(p).all():raise FloatingPointError('nonfinite forecast')
            result[f'native_clean_{split}_mse']=float(np.mean((p-truth)**2))
            if n==3:
                refined=predict(model,row,c,saved,split,.00625).numpy()
                if not np.isfinite(refined).all():raise FloatingPointError('nonfinite refined forecast')
                result[f'refined_clean_{split}_mse']=float(np.mean((refined-truth)**2))
    result['forecast_mse']=result['refined_clean_test_mse' if n==3 else 'native_clean_test_mse']
    grid=np.linspace(*c['delay_grid'])
    with torch.no_grad():pred=model.tau(torch.tensor(grid,dtype=torch.float32)[:,None]).squeeze(1).numpy()
    truth=.30+.08*np.tanh(1.30*grid) if n==1 else c['model']['tau_center']+c['model']['tau_amplitude']*np.tanh(c['model']['tau_slope']*(grid-c['model']['tau_state_center']))
    result['delay_rmse']=float(np.sqrt(np.mean((pred-truth)**2)))
    return result,grid,pred,truth

def archived_row(row,method):
    _,res=paths(row['study']);records=json.loads((res/'run_records.json').read_text())
    selected=[]
    for r in records:
        if r.get('method')!=method or r.get('data_seed')!=row['data_seed'] or r.get('optimization_seed')!=row['optimization_seed']:continue
        if row['study']==2 and (r['design']!='informative' or r['noise_std']!=row['noise_std']):continue
        if row['study']==3 and (r['observation_step']!=row['observation_step'] or (method in ('P3','P4') and r['rollout_step']!=row['rollout_step'])):continue
        selected.append(r)
    assert len(selected)==1,(row,method,len(selected));return selected[0]

def train_fields(model,row,c,saved,schedule=None):
    parameters=list(model.a_net.parameters())+list(model.b_net.parameters())
    frozen_before=state_hash(model.tau_net);fields_before=state_hash(model.a_net)+state_hash(model.b_net)
    for p in model.tau_net.parameters():p.requires_grad_(False)
    N2.set_seed(row['model_seed']+404)
    opt=torch.optim.Adam(parameters,lr=c['learning_rate'],weight_decay=c['weight_decay'])
    y=torch.tensor(saved['noisy_train'][:,10:] if row['study']==1 else saved['train_noisy' if row['study']==3 else 'noisy_train'],dtype=torch.float32)
    if row['study']!=1:
        nn=N2 if row['study']==2 else N3
        meta={k:torch.tensor(saved['train_'+('meta_' if row['study']==3 else '')+k],dtype=torch.float32) for k in ['s','c','d']}
    else:h=torch.tensor(saved['clean_train'][:,:11],dtype=torch.float32)
    started=time.perf_counter();clipped=updates=0
    for steps,epochs in (schedule or c['schedule']):
        weight=torch.linspace(.6,1.4,steps+1)[None,:]
        for _ in range(epochs):
            opt.zero_grad(set_to_none=True)
            pred=first_rollout(model,h,steps)[:,10:] if row['study']==1 else nn.rollout(model,meta,steps,c)
            loss=torch.mean(weight*(pred-y[:,:steps+1])**2)
            if not torch.isfinite(loss):raise FloatingPointError('nonfinite objective')
            loss.backward();norm=torch.nn.utils.clip_grad_norm_(parameters,c['gradient_clip'],error_if_nonfinite=True)
            clipped+=int(float(norm)>c['gradient_clip']);opt.step();updates+=1
    assert state_hash(model.tau_net)==frozen_before
    assert all(p.grad is None for p in model.tau_net.parameters())
    assert fields_before!=state_hash(model.a_net)+state_hash(model.b_net)
    return {'stage2_updates':updates,'stage2_seconds':time.perf_counter()-started,'clipped_updates':clipped,'frozen_tau_sha256':frozen_before,'tau_unchanged':True,'field_parameters_updated':True}

def train_job(row,out):
    out=Path(out);row=copy.deepcopy(row);c,saved=prepare(row)
    _,res=paths(row['study']);base='A1' if row['method']=='T1' else 'A2'
    base_row=archived_row(row,base);cp=res/'checkpoints'/f'{stem(dict(row,method=base))}.pt'
    try:
        model=construct(row,c)
        initial=state_hash(model.tau_net)
        assert initial==base_row.get('initial_tau_hash',base_row.get('initial_tau_sha256'))
        model.tau_net.load_state_dict(torch.load(cp,map_location='cpu',weights_only=True))
        delay_before=model.tau_net(torch.tensor(np.linspace(*c['delay_grid']),dtype=torch.float32)[:,None]).detach().numpy().copy()
        fit=train_fields(model,row,c,saved)
        metrics,grid,pred,true=eval_model(model,row,c,saved)
        assert np.array_equal(pred,delay_before.squeeze(1))
        assert np.isclose(metrics['delay_rmse'],base_row['delay_rmse'],rtol=1e-12,atol=1e-15)
        row.update(status='ok',**metrics,**fit,initial_tau_sha256=initial,stage1_method=base,
                   stage1_updates=base_row['updates'],total_updates=base_row['updates']+fit['stage2_updates'],
                   stage1_checkpoint_sha256=sha(cp),stage1_recorded_seconds=base_row['training_seconds'])
        torch.save(model.state_dict(),out/'checkpoints'/f'{stem(row)}.pt')
        pd.DataFrame({'state':grid,'predicted_delay':pred,'true_delay':true}).to_csv(out/'curves'/f'{stem(row)}.csv',index=False)
    except (FloatingPointError,RuntimeError) as e:
        row.update(status='failed',error=f'{type(e).__name__}: {e}')
        (out/(stem(row)+'_failure.txt')).write_text(traceback.format_exc())
    return row

def replay_controls():
    out=[];checks=[]
    for row in cases():
        c,saved=prepare(row);_,res=paths(row['study'])
        for method in ('P3','P4'):
            q=dict(row,method=method);old=archived_row(row,method)
            model=construct(row,c);model.load_state_dict(torch.load(res/'checkpoints'/f'{stem(q)}.pt',map_location='cpu',weights_only=True))
            metrics,*_=eval_model(model,row,c,saved)
            target=old['refined_clean_test_mse' if row['study']==3 else 'clean_test_mse']
            checks.append({'run':stem(q),'study':row['study'],'old_forecast_mse':target,'replay_forecast_mse':metrics['forecast_mse'],'relative_difference':abs(metrics['forecast_mse']-target)/max(abs(target),1e-30)})
            assert np.isclose(metrics['delay_rmse'],old['delay_rmse'],rtol=1e-12,atol=1e-15)
            out.append(dict(q,status='ok',**metrics,source='archived checkpoint replay',training_updates=old['updates']))
    return out,checks

def main():
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['validate','full'],default='validate');p.add_argument('--workers',type=int,default=8);p.add_argument('--output',type=Path,default=ROOT/'results');args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    if args.mode=='validate':
        controls,checks=replay_controls()
        atomic(args.output/'control_replays.json',controls);atomic(args.output/'replay_checks.json',checks)
        print('REPLAYS',len(checks),'max relative error',max(x['relative_difference'] for x in checks),flush=True)
        smoke=[]
        for n in (1,2,3):
            row=next(r for r in cases() if r['study']==n);c,saved=prepare(row);m=construct(row,c)
            # A fixture only: no scientific two-stage checkpoint is assessed.
            x=torch.tensor([[.2 if n==1 else .8]],requires_grad=True);m.tau_net.requires_grad_(False)
            g=torch.autograd.grad(m.tau(x).sum(),x)[0];assert torch.isfinite(g).all() and torch.any(g!=0)
            fit=train_fields(m,row,c,saved,[(c['schedule'][0][0],2),(c['schedule'][-1][0],2)])
            smoke.append(dict(study=n,state_derivative_retained=float(g),**fit))
        atomic(args.output/'smoke_checks.json',smoke);print('SMOKE passed',flush=True);return
    checks=json.loads((args.output/'replay_checks.json').read_text())
    assert max(x['relative_difference'] for x in checks)<1e-4
    signature={'sources':{str(f.relative_to(HERE)):sha(f) for f in HERE.rglob('*') if f.is_file() and f.suffix in ('.py','.json','.md','.txt')},'python':platform.python_version(),'torch':torch.__version__,'numpy':np.__version__,'workers':args.workers,'threads_per_worker':1}
    manifest=args.output/'manifest.json'
    if manifest.exists():assert json.loads(manifest.read_text())['signature']==signature
    else:atomic(manifest,{'frozen_before_scientific_stage2_fits_utc':datetime.now(timezone.utc).isoformat(),'signature':signature,'status':'Extension after inspection of original E1--E3 outcomes; not independent validation or public preregistration.'})
    for name in ('checkpoints','curves'):(args.output/name).mkdir(exist_ok=True)
    records=args.output/'run_records.json';rows=json.loads(records.read_text()) if records.exists() else []
    done={stem(r) for r in rows}
    jobs=[dict(r,method=m) for r in cases() for m in ('T1','T2') if stem(dict(r,method=m)) not in done]
    print('FITS',len(jobs),'already recorded',len(rows),flush=True)
    with ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        pending={pool.submit(train_job,r,args.output):r for r in jobs}
        for f in as_completed(pending):
            row=f.result();rows.append(row);rows.sort(key=lambda r:(r['study'],stem(r)));atomic(records,rows)
            print('DONE',len(rows),stem(row),row['status'],flush=True)
    pd.DataFrame(rows).to_csv(args.output/'runs.csv',index=False)
    atomic(args.output/'completion.json',{'recorded':len(rows),'expected':210,'failures':sum(r['status']!='ok' for r in rows),'completed_utc':datetime.now(timezone.utc).isoformat()})
    assert len(rows)==210

if __name__=='__main__':main()
