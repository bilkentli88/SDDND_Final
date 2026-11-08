"""Logistic SDDE, exact history geometry, RK4 reference and observed anchors."""
from __future__ import annotations
import numpy as np
import copy


def resolution_config(config, observation_step, rollout_step=None):
    c=copy.deepcopy(config)
    c['observation_step']=float(observation_step)
    c['rollout_step']=float(rollout_step or config['rollout_steps'][0])
    n=int(round(c['slope_window_duration']/observation_step))
    assert np.isclose(n*observation_step,c['slope_window_duration'])
    c['slope_window_points']=n+1
    c['schedule']=[(int(round(t/observation_step)),updates) for t,updates in c['training_phases']]
    assert all(np.isclose(steps*observation_step,t) for (steps,_),(t,_) in zip(c['schedule'],c['training_phases']))
    return c


def observed_case(reference,config,observation_step,seed):
    stride=int(round(observation_step/config['fine_observation_step']))
    assert np.isclose(stride*config['fine_observation_step'],observation_step)
    out={}
    for split in ('train','test'):
        fine=reference[('informative',split)]
        noisy=add_noise(fine['clean'],config['noise_levels'][0],seed+(1000 if split=='test' else 0))
        out[split]={'metadata':fine['metadata'],'clean':fine['clean'][:,::stride],
                    'noisy':noisy[:,::stride],'fine_clean':fine['clean'],'fine_noisy':noisy}
    c=resolution_config(config,observation_step)
    out['anchor']=estimate_anchors(out['train']['noisy'],c['train_states'],'informative',c)
    assert out['anchor']['eligible'].all()
    return out


def tau_true(s, config):
    p=config['model']
    return p['tau_center']+p['tau_amplitude']*np.tanh(p['tau_slope']*(np.asarray(s)-p['tau_state_center']))


def metadata(states, design, config):
    n=len(config['history_c'])
    return {'s':np.repeat(np.asarray(states,dtype=float),n),
            'c':np.tile(np.asarray(config['history_c'],dtype=float),len(states)) if design=='informative' else np.zeros(n*len(states)),
            'd':np.tile(np.asarray(config['history_d'],dtype=float),len(states)) if design=='informative' else np.zeros(n*len(states))}


def history(theta, s, c, d, config):
    w=config['history_frequency']
    return s+config['history_amplitude']*(c*np.sin(w*theta)+d*(1-np.cos(w*theta)))


def signatures(state, delay_grid, design, config):
    c=np.asarray(config['history_c']) if design=='informative' else np.zeros(4)
    d=np.asarray(config['history_d']) if design=='informative' else np.zeros(4)
    return history(-np.asarray(delay_grid)[:,None],state,c[None,:],d[None,:],config)


def initial_responses(state, design, config):
    q=signatures(state,[float(tau_true(state,config))],design,config)[0]
    return config['model']['growth']*state*(1-q/config['model']['capacity'])


def profile(slopes,state,design,config):
    grid=np.linspace(*config['delay_bounds'],config['delay_profile_points'])
    q=signatures(state,grid,design,config)
    w=q-q.mean(axis=1,keepdims=True)
    norms=np.linalg.norm(w,axis=1)
    y=np.asarray(slopes)-np.mean(slopes)
    eligible=bool(norms.min()>config['geometry_norm_threshold'])
    if eligible:
        residual=np.maximum(float(y@y)-(w@y)**2/(norms**2),0.)/len(y)
        anchor=float(grid[int(np.argmin(residual))])
    else:
        residual=np.full(len(grid),float(y@y)/len(y))
        anchor=np.nan
    return anchor,eligible,residual,float(norms.min())


def estimate_anchors(noisy,states,design,config):
    window=config['slope_window_points']
    times=np.arange(window)*config['observation_step']
    scale=times[-1]
    n=len(config['history_c'])
    anchors,eligible,profiles,norms,all_slopes=[],[],[],[],[]
    for i,state in enumerate(states):
        block=noisy[i*n:(i+1)*n,:window]
        coefficient=np.polyfit(times/scale,block.T,deg=config['slope_polynomial_degree'])
        slopes=coefficient[-2]/scale
        anchor,valid,p,norm=profile(slopes,float(state),design,config)
        anchors.append(anchor);eligible.append(valid);profiles.append(p);norms.append(norm);all_slopes.append(slopes)
    return {'states':np.asarray(states,dtype=float),'delays':np.asarray(anchors),
            'eligible':np.asarray(eligible,dtype=bool),'profiles':np.asarray(profiles),
            'minimum_signature_norm':np.asarray(norms),'slope_estimates':np.asarray(all_slopes)}


def add_noise(clean, sigma, seed):
    noisy=np.asarray(clean).copy()
    noisy[:,1:]+=sigma*np.random.default_rng(seed).normal(size=noisy[:,1:].shape)
    return noisy


def simulate_reference(meta, step, config):
    nsteps=int(round(config['horizon']/step))
    stride=int(round(config['observation_step']/step))
    assert np.isclose(nsteps*step,config['horizon']) and np.isclose(stride*step,config['observation_step'])
    assert step<config['delay_bounds'][0]
    s,c,d=meta['s'],meta['c'],meta['d']
    count=len(s)
    values=np.empty((count,nsteps+1),dtype=np.float64)
    values[:,0]=s
    row=np.arange(count)
    growth,capacity=config['model']['growth'],config['model']['capacity']
    for index in range(nsteps):
        def rhs(t,x):
            query=t-tau_true(x,config)
            delayed=history(query,s,c,d,config)
            positive=query>=0
            if positive.any():
                assert index>0 and np.max(query[positive])<index*step+1e-12
                left=np.clip(np.floor(query/step).astype(int),0,index-1)
                weight=query/step-left
                past=(1-weight)*values[row,left]+weight*values[row,left+1]
                delayed=np.where(positive,past,delayed)
            return growth*x*(1-delayed/capacity)
        t=index*step;x=values[:,index]
        k1=rhs(t,x)
        k2=rhs(t+step/2,x+step*k1/2)
        k3=rhs(t+step/2,x+step*k2/2)
        k4=rhs(t+step,x+step*k3)
        values[:,index+1]=x+step*(k1+2*k2+2*k3+k4)/6
    assert np.isfinite(values).all() and (values>0).all()
    return values[:,::stride]


def refined_reference(config):
    specs=[(design,split) for design in config['designs'] for split in ('train','test')]
    metas=[metadata(config[split+'_states'],design,config) for design,split in specs]
    merged={key:np.concatenate([m[key] for m in metas]) for key in ('s','c','d')}
    previous=None;checks=[];chosen=None
    for step in config['reference_steps']:
        result=simulate_reference(merged,step,config)
        if previous is not None:
            difference=float(np.max(np.abs(result-previous)))
            checks.append({'coarser_step':previous_step,'finer_step':step,
                           'max_observation_difference':difference,
                           'passes':difference<=config['reference_tolerance']})
            if difference<=config['reference_tolerance']:
                chosen=step
                break
        previous=result;previous_step=step
    if chosen is None:
        raise RuntimeError('Reference refinement failed the prespecified tolerance: '+str(checks))
    output={};offset=0
    for (design,split),meta in zip(specs,metas):
        count=len(meta['s'])
        output[(design,split)]={'clean':result[offset:offset+count],'metadata':meta}
        offset+=count
    return output,{'selected_step':chosen,'tolerance':config['reference_tolerance'],
                   'comparisons':checks,'scope':'Empirical convergence at observation times; not a rigorous solver-error enclosure.'}
