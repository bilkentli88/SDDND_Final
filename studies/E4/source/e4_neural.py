"""Generic PyTorch architectures; pinned external Euler/direct solver only."""
import hashlib,random,time
import numpy as np
import torch
from torch import nn
from torchdde import integrate,Euler,ConstantStepSizeController
torch.set_num_threads(1)

def seed(s):random.seed(s);np.random.seed(s);torch.manual_seed(s)

def mlp(w):return nn.Sequential(nn.Linear(1,w),nn.Tanh(),nn.Linear(w,w),nn.Tanh(),nn.Linear(w,1))

class Model(nn.Module):
 def __init__(self,method,initial):
  super().__init__();self.delays=nn.Parameter(torch.tensor([initial],dtype=torch.float32));self.external=method.startswith('External')
  if self.external:
   dims=[2,32,32,32,32,1];layers=[]
   for i,(a,b) in enumerate(zip(dims,dims[1:])):
    layers.append(nn.Linear(a,b))
    if i<len(dims)-2:layers.append(nn.ReLU())
   self.field=nn.Sequential(*layers)
  else:self.a_net=mlp(28);self.b_net=mlp(22)
 def forward(self,t,x,args,*,history):
  if self.external:return self.field(torch.cat([x,*history],dim=-1))
  return self.a_net(x)-(0.05+torch.nn.functional.softplus(self.b_net(x)))*history[0]

def state_hash(model):
 h=hashlib.sha256()
 for k,v in model.state_dict().items():h.update(k.encode());h.update(v.detach().numpy().tobytes())
 return h.hexdigest()

def rollout(model,meta,c,obs,h,T=None):
 T=c['horizon'] if T is None else T;ts=torch.arange(int(round(T/obs))+1,dtype=torch.float32)*obs
 m={k:torch.tensor(v,dtype=torch.float32).reshape(-1,1) for k,v in meta.items()}
 def history(t):return m['s']+c['history_amplitude']*(m['c']*torch.sin(c['history_frequency']*t)+m['d']*(1-torch.cos(c['history_frequency']*t)))
 y=integrate(model,Euler(),ts[0],ts[-1],ts,history,None,stepsize_controller=ConstantStepSizeController(),dt0=torch.tensor(h),delays=model.delays,discretize_then_optimize=True,max_steps=2048)
 assert y.shape==(len(meta['s']),len(ts),1) and torch.equal(y[:,0],m['s']);return y[:,:,0]

def fit(r,case,c):
 seed(r['model_seed']);m=Model(r['method'],c['initial_delay']);initial=state_hash(m);pid=id(m.delays);ps=list(m.parameters())
 opt=torch.optim.Adam([{'params':[p for n,p in m.named_parameters() if n!='delays'],'weight_decay':c['field_weight_decay']},{'params':[m.delays],'weight_decay':0.}],lr=r['learning_rate'])
 target=torch.tensor(case['train']['noisy'],dtype=torch.float32);anchors=torch.tensor(case['anchors']['delays'],dtype=torch.float32)
 updates=clipped=projected=boundary=0;curve=[];start=time.perf_counter()
 for T,count in c['schedules'][str(r['budget'])]:
  n=int(round(T/c['observation_step']));weights=torch.linspace(.6,1.4,n+1).reshape(1,-1)
  for _ in range(count):
   opt.zero_grad(set_to_none=True);y=rollout(m,case['train']['metadata'],c,c['observation_step'],c['training_step'],T)
   loss=torch.mean(weights*(y-target[:,:n+1])**2)
   if r['method'].endswith('+anchors'):loss=loss+c['anchor_weight']*torch.mean((m.delays[0]-anchors)**2)
   if not torch.isfinite(loss):raise FloatingPointError('Nonfinite loss')
   loss.backward()
   if any(p.grad is None or not torch.isfinite(p.grad).all() for p in ps):raise FloatingPointError('Invalid gradient')
   norm=torch.nn.utils.clip_grad_norm_(ps,c['gradient_clip'],error_if_nonfinite=True);clipped+=int(float(norm)>c['gradient_clip']);opt.step()
   with torch.no_grad():
    before=float(m.delays[0]);m.delays.clamp_(*c['delay_bounds']);projected+=int(before!=float(m.delays[0]));boundary+=int(float(m.delays[0])<=.1500001 or float(m.delays[0])>=.4999999)
   assert id(m.delays)==pid;updates+=1
   if updates==1 or updates%20==0 or updates==r['budget']:curve.append(dict(update=updates,loss_before_update=float(loss.detach()),delay_after_update=float(m.delays.detach()[0])))
 assert updates==r['budget'] and all(torch.isfinite(p).all() for p in ps)
 with torch.no_grad():
  y=rollout(m,case['train']['metadata'],c,c['observation_step'],c['training_step']);tr=float(torch.mean(torch.linspace(.6,1.4,y.shape[1]).reshape(1,-1)*(y-target)**2));an=float(torch.mean((m.delays[0]-anchors)**2))
 return m,dict(initial_parameters_hash=initial,final_parameters_hash=state_hash(m),parameter_count=sum(p.numel() for p in ps),updates=updates,clipped_updates=clipped,projected_updates=projected,boundary_updates=boundary,training_seconds=time.perf_counter()-start,final_weighted_training_mse=tr,final_anchor_fit_mse=an,final_objective=tr+(c['anchor_weight']*an if r['method'].endswith('+anchors') else 0),optimizer_parameter_identity_preserved=True),curve

def evaluate(m,case,c,stage):
 est=float(m.delays.detach()[0]);out=dict(estimated_delay=est,absolute_delay_error=abs(est-c['generator']['true_delay']))
 specs=[(c['training_step'],c['observation_step'],['train','validation'])] if stage=='calibration' else [(h,c['fine_observation_step'],['train','test']) for h in c['evaluation_steps']]
 with torch.no_grad():
  for h,obs,splits in specs:
   preds=[]
   for s in splits:
    y=rollout(m,case[s]['metadata'],c,obs,h).numpy().astype(float)
    if not np.isfinite(y).all():raise FloatingPointError('Nonfinite prediction')
    preds.append(y);prefix='' if stage=='calibration' else 'fine_'
    for name in ['clean','noisy']:out[f'h{h:g}_{s}_{name}_mse']=float(np.mean((y-case[s][prefix+name])**2))
   out[f'h{h:g}_minimum_observation']=float(min(y.min() for y in preds));out[f'h{h:g}_nonpositive_observations']=int(sum((y<=0).sum() for y in preds))
 return out
