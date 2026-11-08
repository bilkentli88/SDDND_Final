"""Reserved component checks, including direct-mode finite differences."""
import copy,json,math
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torchdde import integrate,Euler,ConstantStepSizeController
import e4_neural as neural
import e4_data as data
from e4_analysis import choose,cell

def gradient_fixture(feedback):
 class Field(nn.Module):
  def __init__(self):
   super().__init__();self.delays=nn.Parameter(torch.tensor([.313],dtype=torch.float64));self.beta=nn.Parameter(torch.tensor(.12 if feedback else .1,dtype=torch.float64));self.alpha=nn.Parameter(torch.tensor(-.45 if feedback else .4*math.exp(.5*.313),dtype=torch.float64))
  def forward(self,t,x,args,*,history):return self.beta*x+self.alpha*history[0]
 m=Field();T=.8 if feedback else .2;h=.0125 if feedback else .005
 s=torch.tensor([.45,.70,.95,1.2],dtype=torch.float64).reshape(-1,1);cv=torch.tensor([-1,0,1,0],dtype=torch.float64).reshape(-1,1);dv=torch.tensor([0,1,0,-1],dtype=torch.float64).reshape(-1,1)
 def hist(t):return s+.18*(cv*torch.sin(math.pi*t)+dv*(1-torch.cos(math.pi*t))) if feedback else torch.exp(.5*t).reshape(1,1)
 ts=torch.linspace(0,T,int(round(T/h))+1,dtype=torch.float64)
 def loss():return integrate(m,Euler(),ts[0],ts[-1],ts,hist,None,stepsize_controller=ConstantStepSizeController(),dt0=torch.tensor(h,dtype=torch.float64),delays=m.delays,discretize_then_optimize=True,max_steps=2048)[:,-1].square().mean()
 value=loss();value.backward();records=[]
 for name,p in m.named_parameters():
  analytic=float(p.grad.reshape(-1)[0]);delta=1e-6;original=p.detach().clone()
  with torch.no_grad():p.copy_(original+delta);plus=float(loss());p.copy_(original-delta);minus=float(loss());p.copy_(original)
  fd=(plus-minus)/(2*delta);relative=abs(analytic-fd)/max(abs(fd),1e-10);assert relative<1e-5,(name,analytic,fd)
  records.append(dict(parameter=name,returned_gradient=analytic,finite_difference=fd,relative_error=relative))
 return dict(feedback=feedback,records=records)

def main():
 c=json.loads((Path(__file__).parent/'protocol.json').read_text());rows=[]
 for d in c['calibration_data_seeds']:
  for lr in c['learning_rates']:
   for b in map(int,c['schedules']):rows.append(dict(method='External-C',noise_std=.0005,learning_rate=lr,budget=b,data_seed=d,status='ok',**{'h0.025_validation_noisy_mse':1.}))
 selected=choose(rows,c)[cell('External-C',.0005)]['selected'];assert (selected['budget'],selected['learning_rate'])==(280,.0005)
 failed=copy.deepcopy(rows)
 for r in failed:
  if (r['learning_rate'],r['budget'])==(.0005,280):r['status']='failed'
 assert choose(failed,c)[cell('External-C',.0005)]['selected']['learning_rate']==.002
 assert not choose(rows[:1],c)[cell('External-C',.0005)]['available']
 assert not choose([],c)[cell('External-C',.0005)]['available']
 for name,count in [('External-C',3298),('Structured-C',1471)]:
  neural.seed(930002);a=neural.Model(name,.325);neural.seed(930002);b=neural.Model(name+'+anchors',.325)
  assert neural.state_hash(a)==neural.state_hash(b) and sum(p.numel() for p in a.parameters())==count
  y=neural.rollout(a,data.metadata(c['train_states'],c),c,.05,.025,.5);y.square().mean().backward();assert a.delays.grad.abs().max()>0
  pid=id(a.delays);opt=torch.optim.Adam([a.delays],lr=1.);opt.step()
  with torch.no_grad():a.delays.clamp_(*c['delay_bounds'])
  assert id(a.delays)==pid==id(opt.param_groups[0]['params'][0])
 ref,conv=data.reference(c);lo=data.observed(ref,c,930001,.0005);hi=data.observed(ref,c,930001,.0015)
 for s in ['train','validation','test']:
  assert np.allclose(3*(lo[s]['noisy']-lo[s]['clean']),hi[s]['noisy']-hi[s]['clean'],atol=1e-15,rtol=0)
  assert np.array_equal(lo[s]['noisy'][:,0],lo[s]['metadata']['s'])
 print(json.dumps(dict(status='passed',direct_gradient_fixtures=[gradient_fixture(False),gradient_fixture(True)],reference=conv,other_checks=['selection tie/missing/failure','paired initialization','parameter counts','delay gradient','projection identity','paired noise/exact initials']),indent=2))

if __name__=='__main__':main()
