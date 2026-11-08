"""Fixed-budget fitting routines adapted from Experiment 2 (MIT notice retained)."""
import copy
import hashlib
import json
import math
from pathlib import Path
import time
import numpy as np
import e3_data as data

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def atomic_json(path,value):
    path=Path(path);temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temp.replace(path)

def state_hash(model):
    digest=hashlib.sha256()
    for key,tensor in model.state_dict().items():
        digest.update(key.encode());digest.update(tensor.detach().cpu().numpy().tobytes())
    return digest.hexdigest()

def fit_model(c,method,seed,meta,target,anchor):
    import torch
    import e3_neural as neural
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
