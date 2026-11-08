"""Prospective ASOC E1 ablation. See the immutable protocol shipped alongside.

Torch is imported only for training; --verify-only works without it.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import math
import platform
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
from scipy.interpolate import PchipInterpolator

HERE = Path(__file__).resolve().parent
NEURAL = ("P3", "P4", "A1", "A2")
DIRECT = ("Linear", "PCHIP")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path, data):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def load_config(path):
    c = json.loads(Path(path).read_text())
    assert c['protocol_id'] == 'ASOC-E1-v1.0'
    assert len(c['data_seeds']) == len(set(c['data_seeds'])) == 5
    assert len(c['optimization_seeds']) == len(set(c['optimization_seeds'])) == 3
    assert set(c['data_seeds']).isdisjoint(range(20260901, 20260906))
    assert tuple(c['neural_methods']) == NEURAL
    assert tuple(c['direct_methods']) == DIRECT
    assert sum(epochs for _, epochs in c['schedule']) == c['anchor_only_updates']
    assert c['anchor_long_updates'] > c['anchor_only_updates']
    return c


def load_data_module(repo, c):
    repo = Path(repo).resolve()
    commit = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()
    if commit != c['commit']:
        raise ValueError(f"Expected commit {c['commit']}; found {commit}")
    source = repo / 'experiments' / 'strengthening'
    subprocess.run(['git', '-C', str(repo), 'diff', '--exit-code', 'HEAD', '--',
                    'experiments/strengthening'], check=True, capture_output=True)
    spec = importlib.util.spec_from_file_location('numpy_data', source / 'numpy_data.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # sdde_core imports this module by its original short name.
    sys.modules['numpy_data'] = module
    assert math.isclose(module.DT, 0.05)
    assert module.HIST_STEPS == 10 and module.N_STEPS == 20
    assert math.isclose(module.TAU_MIN, 0.15) and math.isclose(module.TAU_MAX, 0.50)
    assert len(module.R_GRID) == 351
    return module, source


def verify_archived_anchors(repo, c, data):
    records = pd.read_csv(Path(repo) / 'results/confirmation/reference/step5_confirmation_runs.csv')
    clean, _ = data.make_strong_dataset(tuple(c['train_states']))
    checks = []
    for seed in sorted(records.data_seed.unique()):
        noisy = data.add_observation_noise(clean, c['noise_std'], int(seed))
        assert np.array_equal(noisy[:, :data.HIST_STEPS+1], clean[:, :data.HIST_STEPS+1])
        states, anchors = data.estimate_profile_anchors(noisy, tuple(c['train_states']))
        mae = float(np.mean(np.abs(anchors-data.tau_true_np(states))))
        assert np.allclose(records.loc[records.data_seed.eq(seed), 'anchor_mae'], mae,
                           rtol=1e-12, atol=1e-14)
        grid = np.linspace(*c['delay_grid'])
        linear, pchip = direct_curves(grid, states, anchors)
        assert np.allclose(np.interp(states, states, anchors), anchors)
        assert np.allclose(PchipInterpolator(states, anchors)(states), anchors)
        assert np.isfinite(linear).all() and np.isfinite(pchip).all()
        checks.append({'archived_data_seed':int(seed), 'anchor_mae_matches':True,
                       'history_and_initial_state_unchanged':True})
    assert len(checks) == 5
    return checks


def direct_curves(grid, states, anchors):
    return (np.interp(grid, states, anchors),
            PchipInterpolator(states, anchors, extrapolate=False)(grid))


def curve_metrics(prediction, grid, states, anchors, data):
    if not np.isfinite(prediction).all():
        raise FloatingPointError('Nonfinite delay prediction')
    delta = prediction-data.tau_true_np(grid)
    return {'delay_rmse':float(np.sqrt(np.mean(delta**2))),
            'delay_mae':float(np.mean(np.abs(delta))),
            'delay_max_abs_error':float(np.max(np.abs(delta)))}


def state_hash(module):
    digest = hashlib.sha256()
    for key, tensor in module.state_dict().items():
        digest.update(key.encode())
        digest.update(tensor.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def fit_model(core, c, method, seed, histories, targets, states, anchors):
    import torch
    core.set_seed(seed)
    model = core.RestrictedLearnedModel().to(core.DEVICE)
    initial_hash = state_hash(model.tau_net)
    delay_only = method in ('A1', 'A2')
    if delay_only:
        # Preserve the exact initial delay weights generated after the a/b nets.
        model = copy.deepcopy(model.tau_net)
        parameters = list(model.parameters())
        tau = model
    else:
        parameters = list(model.parameters())
        tau = model.tau
    core.set_seed(seed+404)
    optimizer = torch.optim.Adam(parameters, lr=c['learning_rate'], weight_decay=c['weight_decay'])
    s = torch.tensor(states, dtype=torch.float32).reshape(-1, 1)
    r = torch.tensor(anchors, dtype=torch.float32).reshape(-1, 1)
    smooth_grid = torch.linspace(*c['smoothness_grid']).reshape(-1, 1)
    h = torch.tensor(histories, dtype=torch.float32)
    y = torch.tensor(targets, dtype=torch.float32)
    schedule = [(0, c['anchor_only_updates' if method == 'A1' else 'anchor_long_updates'])] if delay_only else c['schedule']
    clipped = updates = 0
    final_fit = None
    started = time.perf_counter()
    for steps, epochs in schedule:
        weights = torch.linspace(0.6, 1.4, steps+1).reshape(1, -1) if steps else None
        for _ in range(epochs):
            optimizer.zero_grad(set_to_none=True)
            if delay_only:
                fit = None
                loss = c['anchor_weight']*torch.mean((tau(s)-r)**2)
            else:
                full = core.differentiable_rollout(model, h, steps)
                predicted = full[:, core.HIST_STEPS:core.HIST_STEPS+steps+1]
                fit = torch.mean(weights*(predicted-y[:, :steps+1])**2)
                loss = fit
                if method == 'P3':
                    loss = loss+c['anchor_weight']*torch.mean((tau(s)-r)**2)
            smooth = tau(smooth_grid)
            loss = loss+c['smoothness_weight']*torch.mean((smooth[1:]-smooth[:-1])**2)
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite training objective')
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(parameters, c['gradient_clip'], error_if_nonfinite=True)
            clipped += int(float(norm) > c['gradient_clip'])
            optimizer.step()
            updates += 1
            final_fit = float(fit.detach()) if fit is not None else None
    if not all(torch.isfinite(p).all() for p in parameters):
        raise FloatingPointError('Nonfinite final parameter')
    with torch.no_grad():
        anchor_mse = float(torch.mean((tau(s)-r)**2))
        smooth = tau(smooth_grid)
        final_smoothness = float(torch.mean((smooth[1:]-smooth[:-1])**2))
        final_trajectory_fit = None
        if not delay_only:
            last_steps = c['schedule'][-1][0]
            full = core.differentiable_rollout(model,h,last_steps)
            predicted = full[:,core.HIST_STEPS:core.HIST_STEPS+last_steps+1]
            weights = torch.linspace(0.6,1.4,last_steps+1).reshape(1,-1)
            final_trajectory_fit = float(torch.mean(weights*(predicted-y[:,:last_steps+1])**2))
        final_objective = ((final_trajectory_fit or 0.0)
            + (c['anchor_weight']*anchor_mse if method != 'P4' else 0.0)
            + c['smoothness_weight']*final_smoothness)
        if not all(math.isfinite(v) for v in (anchor_mse,final_smoothness,final_objective)):
            raise FloatingPointError('Nonfinite final objective component')
    return model, {'initial_tau_sha256':initial_hash, 'updates':updates,
                   'clipped_updates':clipped, 'training_seconds':time.perf_counter()-started,
                   'last_preupdate_trajectory_fit':final_fit,
                   'final_trajectory_fit':final_trajectory_fit,'final_objective':final_objective,
                   'anchor_fit_mse':anchor_mse, 'final_smoothness':final_smoothness,
                   'parameter_count':sum(p.numel() for p in parameters)}


def learned_curve(model, method, grid):
    import torch
    model.eval()
    with torch.no_grad():
        x = torch.tensor(grid, dtype=torch.float32).reshape(-1, 1)
        return (model(x) if method in ('A1', 'A2') else model.tau(x)).squeeze(-1).cpu().numpy()


def trajectory_metrics(core, model, clean_train, clean_test, noisy_test):
    a = core.rollout_numpy(model, clean_train[:, :core.HIST_STEPS+1])[:, core.HIST_STEPS:]
    b = core.rollout_numpy(model, clean_test[:, :core.HIST_STEPS+1])[:, core.HIST_STEPS:]
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise FloatingPointError('Nonfinite trajectory prediction')
    return {'clean_train_mse':float(np.mean((a-clean_train[:, core.HIST_STEPS:])**2)),
            'clean_test_mse':float(np.mean((b-clean_test[:, core.HIST_STEPS:])**2)),
            'noisy_test_mse':float(np.mean((b-noisy_test[:, core.HIST_STEPS:])**2))}


def summarize(rows, c, output):
    frame = pd.DataFrame(rows)
    frame.to_csv(output/'runs.csv', index=False)
    n_opt, n_data = len(c['optimization_seeds']), len(c['data_seeds'])
    dataset_rows, pairs = [], []
    all_complete = True
    for seed in c['data_seeds']:
        values = {}
        for method in NEURAL+DIRECT:
            group = frame[frame.data_seed.eq(seed) & frame.method.eq(method)]
            expected = n_opt if method in NEURAL else 1
            success = group[group.status.eq('ok')]
            complete = len(group) == expected and len(success) == expected
            all_complete &= complete
            value = float(success.delay_rmse.median()) if complete else None
            if complete:
                values[method] = value
            dataset_rows.append({'data_seed':seed,'method':method,'expected_runs':expected,
                                 'recorded_runs':len(group),'successful_runs':len(success),
                                 'complete':complete,'median_delay_rmse':value})
        for comparator in ('A1', 'A2', 'P4', 'Linear', 'PCHIP'):
            valid = 'P3' in values and comparator in values and values[comparator] > 0
            pairs.append({'data_seed':seed,'comparator':comparator,'complete':valid,
                          'P3_rmse':values.get('P3'), 'comparator_rmse':values.get(comparator),
                          'ratio':values['P3']/values[comparator] if valid else None,
                          'difference':values['P3']-values[comparator] if valid else None})
    pd.DataFrame(dataset_rows).to_csv(output/'dataset_summaries.csv', index=False)
    pd.DataFrame(pairs).to_csv(output/'dataset_pairs.csv', index=False)
    pair_summary = {}
    for comparator in ('A1', 'A2', 'P4', 'Linear', 'PCHIP'):
        ratios = [r['ratio'] for r in pairs if r['comparator'] == comparator and r['complete']]
        complete = len(ratios) == n_data
        pair_summary[comparator] = {'complete_datasets':len(ratios),
            'median_dataset_ratio':float(np.median(ratios)) if ratios else None,
            'min_dataset_ratio':float(min(ratios)) if ratios else None,
            'max_dataset_ratio':float(max(ratios)) if ratios else None,
            'datasets_with_P3_improvement':sum(r < 1 for r in ratios),
            'descriptive_criterion_met':bool(complete and n_data == 5 and
                np.median(ratios) <= c['practical_ratio_threshold'] and
                sum(r < 1 for r in ratios) >= c['practical_min_dataset_wins'])}
    primary_computable = all(pair_summary[m]['complete_datasets'] == n_data for m in ('A1','A2'))
    verdict = bool(pair_summary['A1']['descriptive_criterion_met'] and pair_summary['A2']['descriptive_criterion_met']) if all_complete and primary_computable and n_data == 5 else None
    result = {'complete':bool(all_complete),'mode':c.get('execution_mode','full'),
              'independent_noise_datasets':n_data,'optimization_repeats_per_dataset':n_opt,
              'recorded_neural_fits':int(frame.method.isin(NEURAL).sum()),
              'recorded_direct_functions':int(frame.method.isin(DIRECT).sum()),
              'failed_fits':int(frame.status.ne('ok').sum()),
              'dataset_comparisons':pair_summary,
              'consistent_practical_joint_advantage':verdict,
              'inference':'Descriptive only; no statistical confidence or deterministic certification.'}
    atomic_json(output/'summary.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', required=True, type=Path)
    parser.add_argument('--config', type=Path, default=HERE/'protocol.json')
    parser.add_argument('--output', type=Path, default=HERE/'experiment1_results')
    parser.add_argument('--mode', choices=['full','smoke'], default='full')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    c = load_config(args.config)
    data, source = load_data_module(args.repo, c)
    checks = verify_archived_anchors(args.repo, c, data)
    print('PASS: five archived anchor checks, direct-interpolant and exact-history checks.', flush=True)
    if args.verify_only:
        print(json.dumps({'verification_only':True,'checks':checks}, indent=2))
        return
    import torch
    sys.path.insert(0, str(source))
    import sdde_core as core
    assert torch.get_num_threads() == 1 and core.DEVICE.type == 'cpu'
    if args.mode == 'smoke':
        c.update(c['smoke'])
    c['execution_mode'] = args.mode
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    protocol_path = HERE/'ASOC_Experiment1_Protocol.md'
    signature = {'config_file_sha256':sha(args.config), 'runner_sha256':sha(__file__),
                 'protocol_sha256':sha(protocol_path),'execution_config':c,
                 'source_sha256':{p.name:sha(p) for p in [source/'sdde_core.py',source/'numpy_data.py']},
                 'python':platform.python_version(),'torch':torch.__version__,
                 'numpy':np.__version__,'pandas':pd.__version__,'scipy':scipy.__version__}
    manifest_path = output/'manifest.json'
    if manifest_path.exists():
        if not args.resume:
            raise FileExistsError('Output already initialized; use --resume with the same package and environment.')
        manifest = json.loads(manifest_path.read_text())
        if manifest['signature'] != signature:
            raise ValueError('Resume refused: configuration, protocol, source, mode or software changed.')
    else:
        if any(output.iterdir()):
            raise FileExistsError('Nonempty output without manifest; select a fresh directory.')
        atomic_json(manifest_path, {'frozen_before_training_utc':datetime.now(timezone.utc).isoformat(),
                    'signature':signature,'archived_anchor_checks':checks,
                    'primary_unit':'independently generated observation-noise dataset',
                    'device':'cpu','threads':1,'dtype':'float32',
                    'platform':platform.platform(),'machine':platform.machine(),
                    'processor':platform.processor()})
    records_path = output/'run_records.json'
    rows = json.loads(records_path.read_text()) if records_path.exists() else []
    keys = [(r['data_seed'], r['optimization_seed'], r['method']) for r in rows]
    if len(keys) != len(set(keys)):
        raise ValueError('Duplicate run records')
    done = set(keys)
    grid = np.linspace(*c['delay_grid'])
    clean_train, _ = data.make_strong_dataset(tuple(c['train_states']))
    clean_test, _ = data.make_strong_dataset(tuple(c['test_states']))
    (output/'checkpoints').mkdir(exist_ok=True)
    (output/'curves').mkdir(exist_ok=True)
    (output/'observations').mkdir(exist_ok=True)
    for data_index, seed in enumerate(c['data_seeds']):
        noisy_train = data.add_observation_noise(clean_train, c['noise_std'], seed)
        noisy_test = data.add_observation_noise(clean_test, c['noise_std'], seed+1000)
        states, anchors = data.estimate_profile_anchors(noisy_train, tuple(c['train_states']))
        np.savez_compressed(output/'observations'/f'data{seed}.npz',
            clean_train=clean_train,noisy_train=noisy_train,clean_test=clean_test,
            noisy_test=noisy_test,anchor_states=states,anchor_delays=anchors)
        for method, prediction in zip(DIRECT,direct_curves(grid,states,anchors)):
            key = (seed,None,method)
            if key in done:
                continue
            row = {'data_seed':seed,'optimization_seed':None,'method':method,'status':'ok',
                   'anchor_fit_mse':0.0, **curve_metrics(prediction,grid,states,anchors,data)}
            pd.DataFrame({'state':grid,'predicted_delay':prediction,
                          'true_delay':data.tau_true_np(grid)}).to_csv(output/'curves'/f'{method}_data{seed}.csv',index=False)
            rows.append(row)
            atomic_json(records_path,rows)
            done.add(key)
        for opt in c['optimization_seeds']:
            model_seed = opt+100*data_index
            hashes = set()
            for method in NEURAL:
                key = (seed,opt,method)
                stem = f'{method}_data{seed}_opt{opt}'
                if key in done:
                    existing = next(r for r in rows if (r['data_seed'],r['optimization_seed'],r['method']) == key)
                    if existing.get('initial_tau_sha256'):
                        hashes.add(existing['initial_tau_sha256'])
                    if existing['status'] == 'ok' and not (output/'checkpoints'/f'{stem}.pt').exists():
                        raise FileNotFoundError('Completed checkpoint missing; restore complete results before resuming.')
                    print(f'SKIP recorded {stem}',flush=True)
                    continue
                with (output/'events.jsonl').open('a') as f:
                    f.write(json.dumps({'event':'fit_start_or_restart','run':stem,
                        'utc':datetime.now(timezone.utc).isoformat()})+'\n')
                print(f'TRAIN {stem}',flush=True)
                row = {'data_seed':seed,'optimization_seed':opt,'model_seed':model_seed,'method':method}
                try:
                    model, fit = fit_model(core,c,method,model_seed,
                        clean_train[:,:core.HIST_STEPS+1],noisy_train[:,core.HIST_STEPS:],states,anchors)
                    prediction = learned_curve(model,method,grid)
                    metrics = curve_metrics(prediction,grid,states,anchors,data)
                    if method in ('P3','P4'):
                        metrics.update(trajectory_metrics(core,model,clean_train,clean_test,noisy_test))
                    hashes.add(fit['initial_tau_sha256'])
                    torch.save(model.state_dict(),output/'checkpoints'/f'{stem}.pt')
                    pd.DataFrame({'state':grid,'predicted_delay':prediction,
                        'true_delay':data.tau_true_np(grid)}).to_csv(output/'curves'/f'{stem}.csv',index=False)
                    row.update(status='ok',**fit,**metrics)
                    print(f"DONE {stem}: delay_RMSE={row['delay_rmse']:.6g}",flush=True)
                except (FloatingPointError, RuntimeError) as error:
                    row.update(status='failed',error=f'{type(error).__name__}: {error}')
                    (output/f'{stem}_failure.txt').write_text(traceback.format_exc())
                    print(f"FAILED {stem}: {row['error']}",flush=True)
                rows.append(row)
                atomic_json(records_path,rows)
                done.add(key)
            if len(hashes) > 1:
                raise AssertionError('Initial delay weights differ within a paired run')
    summary = summarize(rows,c,output)
    print(json.dumps(summary,indent=2))
    if not summary['complete']:
        raise SystemExit('Incomplete experiment: inspect failed fits; no complete-case primary verdict was issued.')


if __name__ == '__main__':
    main()
