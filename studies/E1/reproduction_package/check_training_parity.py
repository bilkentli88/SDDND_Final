"""Check the new P3/P4 training loop against the pinned original on old data."""
import argparse
import json
from pathlib import Path
import sys

import torch

import run_experiment1 as runner

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--repo',required=True,type=Path)
parser.add_argument('--output',required=True,type=Path)
args=parser.parse_args()
config=runner.load_config(runner.HERE/'protocol.json')
data,source=runner.load_data_module(args.repo,config)
sys.path.insert(0,str(source))
import sdde_core as core
import run_tnnls_strengthening as original

config.update(config['smoke'])
clean,_=data.make_strong_dataset(tuple(config['train_states']))
noisy=data.add_observation_noise(clean,config['noise_std'],20260901)
states,anchors=data.estimate_profile_anchors(noisy,tuple(config['train_states']))
results=[]
for method in ('P3','P4'):
    seed=20261001
    candidate,_=runner.fit_model(core,config,method,seed,
        clean[:,:core.HIST_STEPS+1],noisy[:,core.HIST_STEPS:],states,anchors)
    match=next(c for c in original.PRINCIPAL_CONFIGURATIONS if c.config_id==method)
    reference=original.make_model(match,seed,None)
    reference,_=original.train_model(reference,clean[:,:core.HIST_STEPS+1],
        noisy[:,core.HIST_STEPS:],seed+404,config['schedule'],
        anchor_states=states if method=='P3' else None,
        anchor_delays=anchors if method=='P3' else None,
        anchor_weight=config['anchor_weight'] if method=='P3' else 0.0,
        smoothness_weight=config['smoothness_weight'])
    candidate_state,reference_state=candidate.state_dict(),reference.state_dict()
    assert set(candidate_state)==set(reference_state)
    assert all(torch.equal(candidate_state[k],reference_state[k]) for k in candidate_state)
    results.append({'method':method,'all_parameters_bitwise_equal_to_original':True,
                    'updates':sum(e for _,e in config['schedule'])})
payload={'scope':'Old-seed smoke-budget implementation parity; no new experiment evidence.',
         'repository_commit':config['commit'],'torch':torch.__version__,
         'data_seed':20260901,'model_seed':20261001,'checks':results}
args.output.write_text(json.dumps(payload,indent=2)+'\n')
print(json.dumps(payload,indent=2))
