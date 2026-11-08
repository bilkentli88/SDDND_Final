"""Constant-delay data and observed-data anchors; truth stays outside fitting."""
import numpy as np

def metadata(states,c):
 return dict(s=np.repeat(np.array(states,float),4),c=np.tile(c['history_c'],len(states)),d=np.tile(c['history_d'],len(states)))

def history(t,m,c):
 return m['s']+c['history_amplitude']*(m['c']*np.sin(c['history_frequency']*t)+m['d']*(1-np.cos(c['history_frequency']*t)))

def simulate(m,h,c):
 n=int(round(c['horizon']/h));stride=int(round(c['fine_observation_step']/h));p=c['generator']
 assert np.isclose(stride*h,c['fine_observation_step']) and h<c['delay_bounds'][0]
 y=np.empty((len(m['s']),n+1));y[:,0]=m['s']
 for i in range(n):
  def rhs(t,x):
   q=t-p['true_delay']
   if q<0:past=history(q,m,c)
   else:
    left=max(0,int(np.floor(q/h)));w=q/h-left;assert left+1<=i
    past=(1-w)*y[:,left]+w*y[:,left+1]
   return p['growth']*x*(1-past/p['capacity'])
  t=i*h;x=y[:,i];a=rhs(t,x);b=rhs(t+h/2,x+h*a/2);d=rhs(t+h/2,x+h*b/2);e=rhs(t+h,x+h*d)
  y[:,i+1]=x+h*(a+2*b+2*d+e)/6
 assert np.isfinite(y).all() and (y>0).all()
 return y[:,::stride]

def reference(c):
 splits=['train','validation','test'];ms={s:metadata(c[s+'_states'],c) for s in splits}
 merged={k:np.concatenate([ms[s][k] for s in splits]) for k in ('s','c','d')};previous=None;checks=[]
 for h in c['reference_steps']:
  result=simulate(merged,h,c)
  if previous is not None:
   delta=float(np.max(abs(result-previous)));checks.append(dict(coarser_step=old,finer_step=h,max_difference=delta,passes=delta<=c['reference_tolerance']))
   if checks[-1]['passes']:break
  previous=result;old=h
 else:raise RuntimeError('Reference convergence failed')
 out={};offset=0
 for s in splits:
  n=len(ms[s]['s']);out[s]=dict(metadata=ms[s],fine_clean=result[offset:offset+n]);offset+=n
 return out,dict(selected_step=h,comparisons=checks,tolerance=c['reference_tolerance'],observation_grid_step=c['fine_observation_step'],scope='Empirical 40-trajectory convergence, not a rigorous enclosure.')

def anchors(noisy,c):
 n=int(round(c['slope_window_duration']/c['observation_step']))+1;t=np.arange(n)*c['observation_step']
 grid=np.linspace(*c['delay_bounds'],c['delay_profile_points']);est=[];slopes=[];profiles=[];norms=[]
 for i,s in enumerate(c['train_states']):
  coef=np.polyfit(t/t[-1],noisy[i*4:i*4+4,:n].T,deg=c['slope_polynomial_degree']);v=coef[-2]/t[-1];slopes.append(v)
  q=np.array([history(-r,metadata([s],c),c) for r in grid]);w=q-q.mean(axis=1,keepdims=True);norm=np.linalg.norm(w,axis=1)
  assert norm.min()>c['geometry_norm_threshold'];v=v-v.mean();res=np.maximum(float(v@v)-(w@v)**2/norm**2,0)/4
  est.append(grid[int(res.argmin())]);profiles.append(res);norms.append(norm.min())
 return dict(delays=np.array(est),slope_estimates=np.array(slopes),profiles=np.array(profiles),minimum_signature_norm=np.array(norms))

def observed(ref,c,seed,sigma):
 stride=int(round(c['observation_step']/c['fine_observation_step']));out={}
 for split,offset in [('train',0),('validation',2000),('test',1000)]:
  b=ref[split];clean=b['fine_clean'];noisy=clean.copy();noisy[:,1:]+=sigma*np.random.default_rng(seed+offset).normal(size=noisy[:,1:].shape)
  out[split]=dict(metadata=b['metadata'],fine_clean=clean,fine_noisy=noisy,clean=clean[:,::stride],noisy=noisy[:,::stride])
 out['anchors']=anchors(out['train']['noisy'],c);return out

def save(path,case):
 fields={}
 for s in ['train','validation','test']:
  for k,v in case[s].items():
   if k=='metadata':fields.update({s+'_meta_'+m:a for m,a in v.items()})
   else:fields[s+'_'+k]=v
 fields.update({'anchors_'+k:v for k,v in case['anchors'].items()});np.savez_compressed(path,**fields)

def load(path):
 with np.load(path) as f:
  out={s:{k:f[s+'_'+k] for k in ['clean','noisy','fine_clean','fine_noisy']} for s in ['train','validation','test']}
  for s in out:out[s]['metadata']={k:f[s+'_meta_'+k] for k in ['s','c','d']}
  out['anchors']={k[8:]:f[k] for k in f.files if k.startswith('anchors_')}
 return out
