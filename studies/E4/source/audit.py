"""Consistency replay; shared model/solver, independent score/selection arithmetic."""
import argparse,itertools,json
from pathlib import Path
import numpy as np
import torch,torchdde
import e4_data as data
import e4_neural as neural
from e4_analysis import write
from run import HERE,FROZEN,sha,source,stem,case_name

def audit(root,architecture):
 manifest=json.loads((root/'manifest.json').read_text());sig=manifest['signature'];c=sig['config'];checks=[]
 def check(name,value):
  checks.append(dict(check=name,passed=bool(value)))
  if not value:raise AssertionError(name)
 for n in FROZEN:check('frozen '+n,sha(HERE/n)==sig['source_sha256'][n])
 solver=Path(torchdde.__file__).resolve().parent.parent
 check('solver provenance',source(solver,c['solver_commit'])==sig['solver']);check('architecture provenance',source(architecture,c['architecture_source_commit'])==sig['architecture'])
 check('runtime torch',torch.__version__==sig['runtime']['torch']);check('runtime numpy',np.__version__==sig['runtime']['numpy'])
 ref,conv=data.reference(c);check('reference',conv==json.loads((root/'reference_convergence.json').read_text()));cases={}
 for s,d in itertools.product(c['noise_levels'],c['calibration_data_seeds']+c['final_data_seeds']):
  expected=data.observed(ref,c,d,s);actual=data.load(root/'observations'/case_name(s,d));cases[(s,d)]=actual
  for split in ['train','validation','test']:
   for k in ['clean','noisy','fine_clean','fine_noisy']:check(f'data/{s}/{d}/{split}/{k}',np.array_equal(expected[split][k],actual[split][k]))
   for k in ['s','c','d']:check(f'meta/{s}/{d}/{split}/{k}',np.array_equal(expected[split]['metadata'][k],actual[split]['metadata'][k]))
  for k,v in expected['anchors'].items():check(f'anchors/{s}/{d}/{k}',np.array_equal(v,actual['anchors'][k]))
 cal=json.loads((root/'calibration_records.json').read_text());final=json.loads((root/'final_records.json').read_text());lock=json.loads((root/'selection_lock.json').read_text())
 check('calibration complete',len(cal)==len(c['methods'])*len(c['noise_levels'])*len(c['learning_rates'])*len(c['schedules'])*len(c['calibration_data_seeds']))
 check('calibration hash',sha(root/'calibration_records.json')==lock['calibration_records_sha256'])
 check('unique records',len({stem(r) for r in cal+final})==len(cal)+len(final));independent={}
 for m,s in itertools.product(c['methods'],c['noise_levels']):
  trials=[]
  for lr,b in itertools.product(c['learning_rates'],map(int,c['schedules'])):
   g=[r for r in cal if (r['method'],r['noise_std'],r['learning_rate'],r['budget'])==(m,s,lr,b)]
   ok=len(g)==len(c['calibration_data_seeds']) and {r['data_seed'] for r in g}==set(c['calibration_data_seeds']) and all(r['status']=='ok' for r in g)
   trials.append(dict(learning_rate=lr,budget=b,eligible=ok,score=sum(r['h0.025_validation_noisy_mse'] for r in g)/len(g) if ok else None))
  eligible=[t for t in trials if t['eligible']];best=sorted(eligible,key=lambda t:(t['score'],t['budget'],t['learning_rate']))[0] if eligible else None
  independent[f'{m}|sigma{s:g}']=dict(selected=best,trials=trials,available=best is not None)
 check('independent selection arithmetic',independent==lock['selections'])
 expected=[]
 for m,s,(i,d),opt in itertools.product(c['methods'],c['noise_levels'],enumerate(c['final_data_seeds']),c['optimization_seeds']):
  if independent[f'{m}|sigma{s:g}']['available']:expected.append((m,s,d,opt,opt+100*i))
 check('expected final cells',sorted(expected)==sorted((r['method'],r['noise_std'],r['data_seed'],r['optimization_seed'],r['model_seed']) for r in final))
 events=[json.loads(t) for t in (root/'events.jsonl').read_text().splitlines()];check('lock before final submissions',all(e['utc']>=lock['locked_utc'] for e in events if e['run'].startswith('final_')))
 maxdiff=0.;replayed=0;initgroups={}
 for r in cal+final:
  if r['status']!='ok':check('failure retained '+stem(r),(root/'failures'/f'{stem(r)}.txt').exists());continue
  neural.seed(r['model_seed']);m=neural.Model(r['method'],c['initial_delay']);check('initial hash '+stem(r),neural.state_hash(m)==r['initial_parameters_hash'])
  initgroups.setdefault((r['method'].split('+')[0],r['model_seed']),set()).add(r['initial_parameters_hash'])
  m.load_state_dict(torch.load(root/'checkpoints'/f'{stem(r)}.pt',map_location='cpu',weights_only=True));check('checkpoint hash '+stem(r),neural.state_hash(m)==r['final_parameters_hash'])
  check('finite weights',all(torch.isfinite(p).all() for p in m.parameters()));check('updates',r['updates']==r['budget']==sum(x[1] for x in c['schedules'][str(r['budget'])]))
  if r['stage']=='final':
   selected=independent[f"{r['method']}|sigma{r['noise_std']:g}"]['selected'];check('selected final settings',(r['learning_rate'],r['budget'])==(selected['learning_rate'],selected['budget']))
  case=cases[(r['noise_std'],r['data_seed'])];est=float(m.delays.detach()[0]);values=dict(estimated_delay=est,absolute_delay_error=abs(est-c['generator']['true_delay']))
  specs=[(c['training_step'],c['observation_step'],['train','validation'])] if r['stage']=='calibration' else [(h,c['fine_observation_step'],['train','test']) for h in c['evaluation_steps']]
  with torch.no_grad():
   for h,obs,splits in specs:
    predictions=[]
    for split in splits:
     y=neural.rollout(m,case[split]['metadata'],c,obs,h).numpy().astype(float);predictions.append(y);prefix='' if r['stage']=='calibration' else 'fine_'
     for name in ['clean','noisy']:
      residual=y-case[split][prefix+name];values[f'h{h:g}_{split}_{name}_mse']=float(np.sum(residual**2)/residual.size)
    values[f'h{h:g}_minimum_observation']=float(min(y.min() for y in predictions));values[f'h{h:g}_nonpositive_observations']=int(sum((y<=0).sum() for y in predictions))
   pred=neural.rollout(m,case['train']['metadata'],c,c['observation_step'],c['training_step']);target=torch.tensor(case['train']['noisy'],dtype=torch.float32)
   tr=float(torch.mean(torch.linspace(.6,1.4,pred.shape[1]).reshape(1,-1)*(pred-target)**2));an=float(torch.mean((m.delays[0]-torch.tensor(case['anchors']['delays'],dtype=torch.float32))**2))
   values.update(final_weighted_training_mse=tr,final_anchor_fit_mse=an,final_objective=tr+(c['anchor_weight']*an if r['method'].endswith('+anchors') else 0))
  for k,v in values.items():maxdiff=max(maxdiff,abs(v-r[k]));check('metric '+stem(r)+'/'+k,np.isclose(v,r[k],rtol=1e-9,atol=1e-12))
  replayed+=1
 check('paired initialization',all(len(v)==1 for v in initgroups.values()))
 direct=json.loads((root/'direct_records.json').read_text())
 for r in direct:
  est=float(np.clip(cases[(r['noise_std'],r['data_seed'])]['anchors']['delays'].mean(),*c['delay_bounds']));check('direct '+str((r['noise_std'],r['data_seed'])),est==r['estimated_delay'] and abs(est-c['generator']['true_delay'])==r['absolute_delay_error'])
 summary=json.loads((root/'summary.json').read_text());blocks=summary['dataset_medians'];outcomes=['absolute_delay_error','h0.00625_test_clean_mse']
 for b in blocks:
  g=[r for r in final if (r['method'],r['noise_std'],r['data_seed'])==(b['method'],b['noise_std'],b['data_seed']) and r['status']=='ok'];check('three repeats',len(g)==len(c['optimization_seeds']))
  for k in outcomes:check('dataset median '+k,float(np.median([r[k] for r in g]))==b[k])
 for p in summary['paired_comparisons']:
  if not p['complete']:continue
  ratios=[];differences=[]
  for d in p['datasets']:
   a=next(b[p['outcome']] for b in blocks if (b['method'],b['noise_std'],b['data_seed'])==(p['numerator_method'],p['noise_std'],d['data_seed']));b=next(b[p['outcome']] for b in blocks if (b['method'],b['noise_std'],b['data_seed'])==(p['denominator_method'],p['noise_std'],d['data_seed']))
   check('paired values',a==d['numerator'] and b==d['denominator']);differences.append(a-b)
   if b!=0:ratios.append(a/b)
  check('paired difference',float(np.median(differences))==p['median_difference']);check('paired wins',sum(v<0 for v in differences)==p['dataset_wins'])
  if len(ratios)==len(p['datasets']):check('paired ratio',float(np.median(ratios))==p['median_ratio'])
 for m in summary['method_medians']:
  g=[b for b in blocks if (b['method'],b['noise_std'])==(m['method'],m['noise_std'])];check('five datasets',len(g)==len(c['final_data_seeds']))
  for k in outcomes:check('marginal median',float(np.median([b[k] for b in g]))==m[k]);check('marginal range',[min(b[k] for b in g),max(b[k] for b in g)]==m[k+'_range'])
 report=dict(status='passed',checkpoints_replayed=replayed,calibration_records=len(cal),final_records=len(final),observation_datasets=len(cases),direct_estimates=len(direct),checks=len(checks),maximum_metric_difference=maxdiff,scope='Consistency replay with independent score/selection arithmetic; shared architecture/solver, not independent research replication.')
 write(root/'audit_checks.json',checks);write(root/'audit_report.json',report);return report

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--results',type=Path,required=True);p.add_argument('--architecture',type=Path,required=True);a=p.parse_args();print(json.dumps(audit(a.results.resolve(),a.architecture.resolve()),indent=2))
