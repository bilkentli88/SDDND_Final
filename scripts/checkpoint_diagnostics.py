#!/usr/bin/env python3
"""Measure cross-platform checkpoint replay differences without changing audits.

This diagnostic reports differences; it does not relax or replace the frozen
strict audit criteria and does not issue a bitwise-reproduction pass verdict.
"""
from pathlib import Path
import argparse,importlib.util,json,math,platform,sys
import numpy as np,pandas as pd,torch

ROOT=Path(__file__).resolve().parents[1]
torch.set_num_threads(1)

class Differences:
    def __init__(self):self.values={}
    def add(self,key,actual,recorded):
        x=np.asarray(actual,dtype=float);y=np.asarray(recorded,dtype=float)
        assert x.shape==y.shape and np.isfinite(x).all() and np.isfinite(y).all(),key
        d=np.abs(x-y);nonzero=np.abs(y)>0
        v=self.values.setdefault(key,{'values':0,'different_values':0,'maximum_absolute_difference':0.,'maximum_relative_difference_nonzero_reference':0.})
        v['values']+=int(x.size);v['different_values']+=int(np.count_nonzero(d))
        v['maximum_absolute_difference']=max(v['maximum_absolute_difference'],float(np.max(d)))
        if nonzero.any():v['maximum_relative_difference_nonzero_reference']=max(v['maximum_relative_difference_nonzero_reference'],float(np.max(d[nonzero]/np.abs(y[nonzero]))))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--study',required=True,choices=['E1','E2','E3','E4','E5'])
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--e1-repo',type=Path,default=ROOT/'.repro-deps/nsddnew_e1')
    a=p.parse_args();d=ROOT/'studies'/a.study
    source=d/('reproduction_package' if a.study in ('E1','E2','E3') else 'source')
    sys.path.insert(0,str(source));diff=Differences();fresh=[];count=0
    result=d/(f'experiment{a.study[1]}_results' if a.study in ('E1','E2','E3') else 'results')
    if a.study in ('E1','E2','E3'):
        n=int(a.study[1]);run=__import__(f'run_experiment{n}')
        c=json.loads((source/'protocol.json').read_text())
        if n==1:
            data,core_source=run.load_data_module(a.e1_repo,c);sys.path.insert(0,str(core_source));neural=__import__('sdde_core')
        else:
            neural=__import__(f'e{n}_neural');data=__import__(f'e{n}_data')
        rows=json.loads((result/'run_records.json').read_text());grid=np.linspace(*c['delay_grid']);cache={}
        for old in rows:
            row=dict(old)
            if old['status']!='ok' or old['method'] not in ('P3','P4','A1','A2'):
                fresh.append(row);continue
            method=old['method']
            if n==1:
                stem=f'{method}_data{old["data_seed"]}_opt{old["optimization_seed"]}'
                model=neural.RestrictedLearnedModel() if method in ('P3','P4') else neural.LearnedTau();ec=c
                name=f'data{old["data_seed"]}'
            elif n==2:
                name=run.slug(old['design'],old['noise_std'],old['data_seed'])
                stem=f'{name}_{method}_opt{old["optimization_seed"]}'
                model=neural.RestrictedModel(c) if method in ('P3','P4') else neural.LearnedTau(c);ec=c
            else:
                stem=run.stem(old);name=run.obs_slug(old['observation_step'],old['data_seed'])
                ec=data.resolution_config(c,old['observation_step'],old['rollout_step'])
                model=neural.RestrictedModel(ec) if method in ('P3','P4') else neural.LearnedTau(ec)
            model.load_state_dict(torch.load(result/'checkpoints'/(stem+'.pt'),weights_only=True,map_location='cpu'))
            pred=run.learned_curve(model,method,grid) if n==1 else neural.delay_curve(model,method,grid)
            curve=pd.read_csv(result/'curves'/(stem+'.csv')).predicted_delay.to_numpy().astype(np.float32)
            diff.add('delay_curve',pred,curve)
            truth=data.tau_true_np(grid) if n==1 else data.tau_true(grid,c)
            row['delay_rmse']=float(np.sqrt(np.mean((pred.astype(float)-truth)**2)))
            diff.add('delay_rmse',row['delay_rmse'],old['delay_rmse'])
            if method in ('P3','P4'):
                if name not in cache:cache[name]=dict(np.load(result/'observations'/(name+'.npz')))
                z=cache[name]
                if n==1:metrics=run.trajectory_metrics(neural,model,z['clean_train'],z['clean_test'],z['noisy_test'])
                elif n==2:
                    blocks={s:{'metadata':{k:z[s+'_'+k] for k in ('s','c','d')},'clean':z['clean_'+s]} for s in ('train','test')}
                    metrics=run.trajectory_metrics(model,blocks['train'],blocks['test'],z['noisy_train'],z['noisy_test'],c)
                else:
                    blocks={s:{'metadata':{k:z[s+'_meta_'+k] for k in ('s','c','d')},**{k:z[s+'_'+k] for k in ('clean','noisy','fine_clean','fine_noisy')}} for s in ('train','test')}
                    metrics=run.evaluate(model,blocks,ec)
                for k,v in metrics.items():diff.add(k,v,old[k]);row[k]=v
            fresh.append(row);count+=1
    elif a.study=='E4':
        import e4_data as data,e4_neural as neural
        c=json.loads((result/'manifest.json').read_text())['signature']['config']
        rows=json.loads((result/'calibration_records.json').read_text())+json.loads((result/'final_records.json').read_text());cache={}
        for old in rows:
            row=dict(old)
            stem=f"{row['stage']}_sigma{row['noise_std']:g}_data{row['data_seed']}_{row['method']}_lr{row['learning_rate']:g}_u{row['budget']}_seed{row['model_seed']}"
            model=neural.Model(row['method'],c['initial_delay']);model.load_state_dict(torch.load(result/'checkpoints'/(stem+'.pt'),weights_only=True,map_location='cpu'))
            name=f'sigma{row["noise_std"]:g}_data{row["data_seed"]}.npz'
            if name not in cache:cache[name]=data.load(result/'observations'/name)
            values=neural.evaluate(model,cache[name],c,row['stage'])
            for k,v in values.items():diff.add(k,v,old[k]);row[k]=v
            fresh.append(row);count+=1
    else:
        import two_stage as exp
        new=json.loads((result/'run_records.json').read_text());controls=json.loads((result/'control_replays.json').read_text())
        rows=new+controls
        for old in rows:
            row=dict(old);c,z=exp.prepare(row);model=exp.construct(row,c)
            if row['method'] in ('T1','T2'):
                checkpoint=result/'checkpoints'/(exp.stem(row)+'.pt')
            else:
                _,previous=exp.paths(row['study']);checkpoint=previous/'checkpoints'/(exp.stem(row)+'.pt')
            model.load_state_dict(torch.load(checkpoint,weights_only=True,map_location='cpu'))
            if row['method'] in ('T1','T2'):
                _,previous=exp.paths(row['study']);stage1=exp.stem(dict(row,method=row['stage1_method']))
                frozen=torch.load(previous/'checkpoints'/(stage1+'.pt'),weights_only=True,map_location='cpu')
                assert all(torch.equal(model.tau_net.state_dict()[k],v) for k,v in frozen.items())
            values,grid,pred,truth=exp.eval_model(model,row,c,z)
            for k,v in values.items():diff.add(k,v,old[k]);row[k]=v
            fresh.append(row);count+=1
    a.output.mkdir(parents=True,exist_ok=True)
    report={'status':'diagnostic_completed','study':a.study,'checkpoint_evaluations':count,
            'runtime':{'python':platform.python_version(),'torch':torch.__version__,'numpy':np.__version__,'cpu_capability':torch.backends.cpu.get_cpu_capability(),'threads':torch.get_num_threads()},
            'bitwise_or_strict_audit_pass_claimed':False,
            'differences':diff.values,
            'scope':'Measures numeric drift using retained sources and weights. Frozen strict audits and records are unchanged; no tolerance-based pass verdict is assigned.'}
    (a.output/'diagnostic.json').write_text(json.dumps(report,indent=2)+'\n')
    (a.output/'replayed_records.json').write_text(json.dumps(fresh,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='differences'},indent=2))

if __name__=='__main__':main()
