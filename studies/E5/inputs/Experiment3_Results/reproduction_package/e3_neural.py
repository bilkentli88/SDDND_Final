"""Matched neural architectures and differentiable Euler rollout for E3.

MLP construction follows NSDDNew (Aykut T. Altay, MIT); see its retained notice.
The negative-gain prior and analytic trigonometric history rollout are inherited from E2.
"""
import random
import numpy as np
import torch
from torch import nn

torch.set_num_threads(1)


def set_seed(seed):
    torch.manual_seed(seed);np.random.seed(seed);random.seed(seed)


class MLP(nn.Module):
    def __init__(self,width):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(1,width),nn.Tanh(),nn.Linear(width,width),nn.Tanh(),nn.Linear(width,1))
    def forward(self,x):
        return self.net(x)


class LearnedTau(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.net=MLP(20)
        self.minimum,self.maximum=config['delay_bounds']
    def forward(self,x):
        return self.minimum+(self.maximum-self.minimum)*torch.sigmoid(self.net(x))


class RestrictedModel(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.a_net=MLP(28);self.b_net=MLP(22);self.tau_net=LearnedTau(config)
    def tau(self,x):
        return self.tau_net(x)
    def derivative(self,x,delayed):
        gain=-(0.05+torch.nn.functional.softplus(self.b_net(x)))
        return self.a_net(x)+gain*delayed


def tensors(meta,dtype=torch.float32):
    return {key:torch.tensor(value,dtype=dtype) for key,value in meta.items()}


def rollout(model,meta,observation_steps,config,step=None):
    h=config['rollout_step'] if step is None else step
    substeps=int(round(config['observation_step']/h))
    assert np.isclose(substeps*h,config['observation_step']) and h<config['delay_bounds'][0]
    s,c,d=meta['s'],meta['c'],meta['d']
    sequence=[s]
    for index in range(observation_steps*substeps):
        x=sequence[-1].unsqueeze(-1)
        delay=model.tau(x).squeeze(-1)
        query=index*h-delay
        analytic=s+config['history_amplitude']*(c*torch.sin(config['history_frequency']*query)+d*(1-torch.cos(config['history_frequency']*query)))
        if index==0:
            past=s
        else:
            values=torch.stack(sequence,dim=1)
            left=torch.floor(query/h).long().clamp(0,index-1)
            weight=query/h-left.to(query.dtype)
            before=torch.gather(values,1,left[:,None]).squeeze(-1)
            after=torch.gather(values,1,(left+1)[:,None]).squeeze(-1)
            past=(1-weight)*before+weight*after
        delayed=torch.where(query<0,analytic,past)
        derivative=model.derivative(x,delayed[:,None]).squeeze(-1)
        sequence.append(sequence[-1]+h*derivative)
    return torch.stack(sequence,dim=1)[:,::substeps]


def delay_curve(model,method,grid):
    model.eval()
    with torch.no_grad():
        x=torch.tensor(grid,dtype=torch.float32).reshape(-1,1)
        return (model(x) if method in ('A1','A2') else model.tau(x)).squeeze(-1).numpy()
