"""Restartable fixed E4 recovery run with separately saved stage boundaries."""
import argparse,hashlib,itertools,json,multiprocessing,platform,subprocess,time,traceback
from concurrent.futures import ProcessPoolExecutor,as_completed
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import torch,torchdde
import e4_data as data
import e4_neural as neural
from e4_analysis import write,choose,cell,summarize

HERE=Path(__file__).resolve().parent
FROZEN=['protocol.json','PROTOCOL.md','e4_data.py','e4_neural.py','e4_analysis.py','run.py','audit.py','validate.py']
def utc():return datetime.now(timezone.utc).isoformat()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def case_name(s,d):return f'sigma{s:g}_data{d}.npz'
def stem(r):return f"{r['stage']}_sigma{r['noise_std']:g}_data{r['data_seed']}_{r['method']}_lr{r['learning_rate']:g}_u{r['budget']}_seed{r['model_seed']}"

def source(root,commit):
 root=Path(root).resolve();actual=subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip();assert actual==commit
 assert not subprocess.check_output(['git','-C',str(root),'diff','HEAD','--name-only'],text=True).strip()
 names=subprocess.check_output(['git','-C',str(root),'ls-files','-z']).decode().split('\0')
 return dict(commit=commit,tracked_sha256={n:sha(root/n) for n in names if n})

def signature(c,solver,architecture):
 assert Path(torchdde.__file__).resolve().parent.parent==solver.resolve()
 return dict(config=c,source_sha256={n:sha(HERE/n) for n in FROZEN},solver=source(solver,c['solver_commit']),architecture=source(architecture,c['architecture_source_commit']),runtime=dict(python=platform.python_version(),torch=torch.__version__,numpy=np.__version__,workers=c['workers'],threads_per_worker=1,device='cpu'))

def worker(r,c,root):
 start=time.perf_counter();r=dict(r)
 try:
  case=data.load(root/'observations'/case_name(r['noise_std'],r['data_seed']));m,fit,curve=neural.fit(r,case,c);r.update(fit)
  path=root/'checkpoints'/f'{stem(r)}.pt';temp=path.with_suffix('.tmp');torch.save(m.state_dict(),temp);temp.replace(path)
  write(root/'curves'/f'{stem(r)}.json',curve);r.update(neural.evaluate(m,case,c,r['stage']),status='ok')
 except Exception as e:
  r.update(status='failed',error=f'{type(e).__name__}: {e}');(root/'failures'/f'{stem(r)}.txt').write_text(traceback.format_exc())
 r['total_job_seconds']=time.perf_counter()-start;write(root/'individual_records'/f'{stem(r)}.json',r);return r

def execute(specs,c,root,stage,limit):
 record=root/f'{stage}_records.json';rows=json.loads(record.read_text()) if record.exists() else []
 done={stem(r) for r in rows};expected={stem(r) for r in specs};assert len(done)==len(rows) and done.issubset(expected)
 # Recover an atomic worker record if its parent was interrupted before commit.
 for p in (root/'individual_records').glob(stage+'_*.json'):
  r=json.loads(p.read_text())
  if stem(r) not in done:assert stem(r) in expected;rows.append(r);done.add(stem(r))
 jobs=[r for r in specs if stem(r) not in done]
 if limit:jobs=jobs[:limit]
 with ProcessPoolExecutor(max_workers=c['workers'],mp_context=multiprocessing.get_context('spawn')) as pool:
  futures=[]
  for r in jobs:
   with (root/'events.jsonl').open('a') as f:f.write(json.dumps(dict(event='submit_or_restart',utc=utc(),run=stem(r)))+'\n')
   futures.append(pool.submit(worker,r,c,root))
  for f in as_completed(futures):
   r=f.result();rows.append(r);rows.sort(key=stem);write(record,rows);print(f"{stage} {len(rows)}/{len(specs)} {stem(r)} {r['status']} seconds={r['total_job_seconds']:.1f}",flush=True)
 rows.sort(key=stem);write(record,rows);return rows,len(rows)==len(specs)

def main():
 p=argparse.ArgumentParser();p.add_argument('--results',type=Path,required=True);p.add_argument('--solver',type=Path,required=True);p.add_argument('--architecture',type=Path,required=True);p.add_argument('--stage',choices=['calibration','final'],required=True);p.add_argument('--mode',choices=['smoke','full'],default='full');p.add_argument('--limit',type=int,default=0);a=p.parse_args()
 c=json.loads((HERE/'protocol.json').read_text())
 if a.mode=='smoke':c.update(c['smoke'])
 c['mode']=a.mode;root=a.results.resolve();root.mkdir(parents=True,exist_ok=True);sig=signature(c,a.solver,a.architecture)
 path=root/'manifest.json'
 if path.exists():assert json.loads(path.read_text())['signature']==sig,'Source/settings/runtime changed'
 else:
  assert not any(root.iterdir());write(path,dict(frozen_before_this_recovery_run_utc=utc(),signature=sig,recovery_disclosure='Original attempt outputs were lost during workspace maintenance. Its calibration completion and selected configurations were seen. This is a reconstructed rerun under the same scientific settings, with eight rather than four one-thread workers; not a fresh public preregistration.'))
 for folder in ['observations','checkpoints','curves','failures','individual_records']:(root/folder).mkdir(exist_ok=True)
 ref,convergence=data.reference(c);write(root/'reference_convergence.json',convergence)
 for s,d in itertools.product(c['noise_levels'],c['calibration_data_seeds']+c['final_data_seeds']):
  path=root/'observations'/case_name(s,d)
  if not path.exists():data.save(path,data.observed(ref,c,d,s))
 if a.stage=='calibration':
  assert not (root/'final_records.json').exists(),'Calibration cannot be altered after final fitting'
  specs=[dict(stage='calibration',method=m,noise_std=s,learning_rate=lr,budget=b,data_seed=d,model_seed=c['calibration_model_seed']+100*i) for m,s,lr,b,(i,d) in itertools.product(c['methods'],c['noise_levels'],c['learning_rates'],map(int,c['schedules']),enumerate(c['calibration_data_seeds']))]
  rows,complete=execute(specs,c,root,'calibration',a.limit)
  if complete:
   lock=dict(locked_utc=utc(),calibration_records_sha256=sha(root/'calibration_records.json'),selections=choose(rows,c))
   path=root/'selection_lock.json'
   if path.exists():assert json.loads(path.read_text())['selections']==lock['selections']
   else:write(path,lock)
   print('CALIBRATION LOCKED',flush=True)
 else:
  lock=json.loads((root/'selection_lock.json').read_text());assert lock['calibration_records_sha256']==sha(root/'calibration_records.json')
  specs=[];direct=[]
  for s,(i,d) in itertools.product(c['noise_levels'],enumerate(c['final_data_seeds'])):
   case=data.load(root/'observations'/case_name(s,d));est=float(np.clip(case['anchors']['delays'].mean(),*c['delay_bounds']))
   direct.append(dict(method='Anchor-C',noise_std=s,data_seed=d,estimated_delay=est,absolute_delay_error=abs(est-c['generator']['true_delay'])))
   for m,opt in itertools.product(c['methods'],c['optimization_seeds']):
    selected=lock['selections'][cell(m,s)]['selected']
    if selected is not None:specs.append(dict(stage='final',method=m,noise_std=s,data_seed=d,optimization_seed=opt,model_seed=opt+100*i,learning_rate=selected['learning_rate'],budget=selected['budget']))
  write(root/'direct_records.json',direct);rows,complete=execute(specs,c,root,'final',a.limit)
  if complete:
   summarize(rows,direct,c,root);write(root/'execution_complete.json',dict(completed_utc=utc(),final_records=len(rows),successful=sum(r['status']=='ok' for r in rows)));print('FINAL COMPLETE',flush=True)

if __name__=='__main__':main()
