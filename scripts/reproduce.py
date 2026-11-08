#!/usr/bin/env python3
"""Verify the experiment materials or replay and rerun studies E1--E5."""
from __future__ import annotations
import argparse
import collections
import csv
import hashlib
import json
import math
import os
import shutil
import statistics
import subprocess
import sys
from contextlib import nullcontext
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

ROOT=Path(__file__).resolve().parents[1]
FROZEN_SOURCES=ROOT/'provenance/frozen_sources.zip'
STUDIES=('E1','E2','E3','E4','E5')
COUNTS={'E1':(70,60),'E2':(280,180),'E3':(200,180),'E4':(184,184),'E5':(210,210)}

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def load(path):return json.loads(path.read_text())
def layout(study):
    d=ROOT/'studies'/study
    if study in ('E1','E2','E3'):return d/'reproduction_package',d/f'experiment{study[1]}_results'
    return d/'source',d/'results'

def paired_delay(rows,method,condition):
    ratios=[]
    for seed in sorted({r['data_seed'] for r in rows if condition(r)}):
        a=[r['delay_rmse'] for r in rows if condition(r) and r['data_seed']==seed and r['method']=='P3' and r['status']=='ok']
        b=[r['delay_rmse'] for r in rows if condition(r) and r['data_seed']==seed and r['method']==method and r['status']=='ok']
        if len(a)!=3 or len(b)!=3:raise AssertionError((seed,method,len(a),len(b)))
        ratios.append(statistics.median(a)/statistics.median(b))
    assert len(ratios)==5
    return {'median_paired_ratio':statistics.median(ratios),'wins':sum(r<1 for r in ratios),'datasets':5}

def frozen_sha(path, originals):
    relative = path.relative_to(ROOT).as_posix()
    if relative in originals:
        return hashlib.sha256(originals[relative]).hexdigest()
    return sha(path)

def stage_frozen_replay(source, destination):
    """Restore the original package names and entry points for strict replay."""
    shutil.copytree(source, destination)
    prefix = source.relative_to(ROOT).as_posix() + '/'
    with ZipFile(FROZEN_SOURCES) as archive:
        for name in archive.namelist():
            if name.startswith(prefix):
                (destination / name.removeprefix(prefix)).write_bytes(archive.read(name))
    for pattern in ('Experiment*_Protocol.md', 'Experiment*_Model_Check.md'):
        for document in source.glob(pattern):
            shutil.copy2(document, destination / f'ASOC_{document.name}')

def verify_appendix_h():
    study = ROOT / 'studies/sensitivity'
    source = study / 'neural/reproduction'
    neural = study / 'neural/results'
    profile = study / 'results'
    signature = load(neural / 'manifest.json')['signature']
    for directory, hashes in (
        (source, signature['source_sha256']),
        (ROOT, load(profile / 'manifest.json')['source_sha256']),
    ):
        for name, expected in hashes.items():
            if directory == ROOT:
                name = name.replace('studies/R1_robustness/', 'studies/sensitivity/', 1)
            assert sha(directory / name) == expected, name

    with (neural / 'runs.csv').open(newline='') as f:
        runs = list(csv.DictReader(f))
    checkpoints = list((neural / 'cases').glob('*/checkpoints/*.pt'))
    assert len(runs) == len(checkpoints) == load(neural / 'complete.json')['completed_fits'] == 240
    protocol = load(source / 'r1_protocol.json')
    data_seeds = {str(seed) for seed in protocol['data_seeds']}
    optimizer_seeds = {str(seed) for seed in protocol['optimization_seeds']}
    fits = collections.defaultdict(list)
    for run in runs:
        assert run['status'] == 'ok', run
        key = (run['noise_std'], run['observation_step'], run['data_seed'], run['method'])
        fits[key].append(run)
    assert all({r['optimization_seed'] for r in rows} == optimizer_seeds for rows in fits.values())
    with (neural / 'paired_summary.csv').open(newline='') as f:
        paired = list(csv.DictReader(f))
    assert len({(r['noise_std'], r['observation_step'], r['metric']) for r in paired}) == len(paired) == 8
    for row in paired:
        condition = (row['noise_std'], row['observation_step'])
        ratios = []
        for seed in data_seeds:
            p3, p4 = (
                statistics.median(float(r[row['metric']]) for r in fits[(*condition, seed, method)])
                for method in ('P3', 'P4')
            )
            ratios.append(p3 / p4)
        assert len(ratios) == int(row['n_datasets']) == 10
        assert math.isclose(statistics.median(ratios), float(row['median_paired_ratio']), rel_tol=1e-12)
        assert sum(ratio < 1 for ratio in ratios) == int(row['numerator_lower'])

    fields = ('noise_sd', 'observation_step', 'response_estimator', 'window')
    datasets = collections.defaultdict(list)
    states = collections.defaultdict(list)
    for filename, grouped in (('per_dataset.csv', datasets), ('per_state.csv', states)):
        with (profile / filename).open(newline='') as f:
            for row in csv.DictReader(f):
                grouped[tuple(row[field] for field in fields)].append(row)
    with (profile / 'summary.csv').open(newline='') as f:
        summaries = list(csv.DictReader(f))
    assert len(summaries) == len(datasets) == len(states) == 12
    for row in summaries:
        key = tuple(row[field] for field in fields)
        assert len({r['data_seed'] for r in datasets[key]}) == len(datasets[key]) == int(row['data_draws']) == 40
        assert len(states[key]) == int(row['state_draw_pairs']) == 160
        for metric, values in (
            ('median_dataset_rmse', [float(r['delay_rmse']) for r in datasets[key]]),
            ('median_interval_width', [float(r['interval_width']) for r in states[key]]),
        ):
            assert math.isclose(statistics.median(values), float(row[metric]), rel_tol=1e-12)
        for field, summary_field in (
            ('accepted', 'accepted'),
            ('accepted_large_error', 'accepted_large_errors'),
            ('interval_contains_truth', 'interval_contains_truth'),
        ):
            assert sum(r[field] == 'True' for r in states[key]) == int(row[summary_field])
    return {
        'neural_fits': len(runs), 'neural_checkpoints': len(checkpoints),
        'neural_paired_summary_rows': len(paired), 'profile_summary_rows': len(summaries),
        'profile_base_noise_datasets': len({r['data_seed'] for rows in datasets.values() for r in rows}),
        'frozen_source_hashes_match': True,
    }

def verify(studies):
    manifest=load(ROOT/'checksums/files.sha256.json')
    for name,expected in manifest['files'].items():
        path=ROOT/name
        if not path.is_file() or sha(path)!=expected:
            raise AssertionError(f'Missing or changed supplied file: {name}')
    report={'file_hashes_verified':len(manifest['files']),'studies':{}}
    with ZipFile(FROZEN_SOURCES) as archive:
        originals = {name: archive.read(name) for name in archive.namelist()}
    for study in studies:
        source,result=layout(study);m=load(result/'manifest.json');sig=m['signature']
        if study=='E1':
            for name,key in [('run_experiment1.py','runner_sha256'),('protocol.json','config_file_sha256'),('Experiment1_Protocol.md','protocol_sha256')]:assert frozen_sha(source/name, originals)==sig[key]
            for name,value in sig['source_sha256'].items():assert sha(ROOT/'experiments/strengthening'/name)==value
        elif study in ('E2','E3'):
            assert sha(source/'protocol.json')==sig['config_file_sha256']
            for name,value in sig['source_sha256'].items():assert frozen_sha(source/name.removeprefix('ASOC_'), originals)==value
        else:
            field='source_sha256' if study=='E4' else 'sources'
            for name,value in sig[field].items():assert frozen_sha(source/name, originals)==value
        rows=(load(result/'calibration_records.json')+load(result/'final_records.json')) if study=='E4' else load(result/'run_records.json')
        checkpoints=list((result/'checkpoints').glob('*.pt'))
        assert (len(rows),len(checkpoints))==COUNTS[study],study
        statuses=collections.Counter(r['status'] for r in rows)
        assert not (set(statuses)-{'ok','abstain_structural'}),statuses
        if study=='E2':assert statuses['abstain_structural']==80
        record={'records':len(rows),'trained_checkpoints':len(checkpoints),'status_counts':dict(statuses),'frozen_source_hashes_match':True}
        if study in ('E1','E2','E3'):
            results=[]
            conditions=[{}] if study=='E1' else ([{'noise_std':s,'design':'informative'} for s in (.0005,.0015)] if study=='E2' else [{'observation_step':o,'rollout_step':h} for o in (.05,.025) for h in (.025,.0125)])
            for condition in conditions:
                for method in ('P4','A1','A2'):
                    # E3 anchor-only fits are shared across solver steps.
                    selected=[]
                    for r in rows:
                        wanted=dict(condition)
                        if study=='E3' and r['method'] in ('A1','A2'):
                            wanted.pop('rollout_step',None)
                        if all(r.get(k)==v for k,v in wanted.items()):selected.append(r)
                    results.append(dict(condition=condition,comparator=method,**paired_delay(selected,method,lambda r:True)))
            record['paired_delay_summaries']=results
        elif study=='E4':
            summary=load(result/'summary.json');final=load(result/'final_records.json')
            assert len(final)==120
            for pair in summary['paired_comparisons']:
                if not pair['complete']:continue
                ratios=[v['numerator']/v['denominator'] for v in pair['datasets']]
                assert math.isclose(statistics.median(ratios),pair['median_ratio'],rel_tol=1e-12)
            record['original_replay_audit']=load(result/'audit_report.json')
        else:
            controls=load(result/'control_replays.json');assert len(controls)==210
            with (result/'dataset_pairs.csv').open(newline='') as f:pairs=list(csv.DictReader(f))
            with (result/'paired_summary.csv').open(newline='') as f:summaries=list(csv.DictReader(f))
            for s in summaries:
                group=[r for r in pairs if all(r[k]==s[k] for k in ('study','noise_std','observation_step','rollout_step','numerator','denominator','metric'))]
                ratios=[float(r['numerator_value'])/float(r['denominator_value']) for r in group]
                assert len(ratios)==5
                assert math.isclose(statistics.median(ratios),float(s['median_paired_ratio']),rel_tol=1e-12)
                assert sum(v<1 for v in ratios)==int(s['numerator_wins'])
            record.update(replayed_control_records=len(controls),paired_summary_rows=len(summaries))
        report['studies'][study]=record
    report['appendix_h'] = verify_appendix_h()
    print(json.dumps(report,indent=2))
    return report

def commands(args,study,source):
    dest=args.output/study
    python=sys.executable
    if args.action=='diagnose':
        return [[python,str(ROOT/'scripts/checkpoint_diagnostics.py'),'--study',study,'--output',str(dest),'--e1-repo',str(args.e1_repo)]]
    if args.action=='replay':
        if study=='E1':return [[python,str(source/'audit_completed_run.py'),'--repo',str(args.e1_repo),'--results',str(dest)]]
        if study in ('E2','E3'):return [[python,str(source/f'audit_experiment{study[1]}.py'),'--results',str(dest)]]
        if study=='E4':return [[python,str(source/'audit.py'),'--results',str(dest),'--architecture',str(args.architecture)]]
        return [[python,str(source.parent/'analyze.py'),'--results',str(dest)]]
    mode='smoke' if args.action=='smoke' else 'full'
    if study in ('E1','E2','E3'):
        command=[python,str(source/f'run_experiment{study[1]}.py'),'--mode',mode,'--output',str(dest),'--resume']
        if study=='E1':command+=['--repo',str(args.e1_repo)]
        return [command]
    if study=='E4':
        return [[python,str(source/'run.py'),'--results',str(dest),'--solver',str(args.solver),'--architecture',str(args.architecture),'--stage',stage,'--mode',mode] for stage in ('calibration','final')]
    result=[[python,str(source/'two_stage.py'),'--mode','validate','--output',str(dest)]]
    if args.action=='train':result.append([python,str(source/'two_stage.py'),'--mode','full','--workers',str(args.workers),'--output',str(dest)])
    return result

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['verify','replay','diagnose','smoke','train'])
    p.add_argument('--studies',nargs='+',choices=STUDIES,default=list(STUDIES))
    p.add_argument('--output',type=Path)
    p.add_argument('--e1-repo',type=Path,default=ROOT/'.repro-deps/nsddnew_e1')
    p.add_argument('--solver',type=Path,default=ROOT/'.repro-deps/torchdde')
    p.add_argument('--architecture',type=Path,default=ROOT/'.repro-deps/learnable_delays')
    p.add_argument('--workers',type=int,default=8)
    p.add_argument('--dry-run',action='store_true')
    args=p.parse_args()
    if args.action=='verify':verify(args.studies);return
    args.output=(args.output or ROOT/'generated/reproduction'/args.action).resolve()
    for field in ('e1_repo','solver','architecture'):setattr(args,field,getattr(args,field).resolve())
    if not args.dry_run:
        if 'E1' in args.studies and not (args.e1_repo/'.git').exists():p.error('Run: python scripts/setup_dependencies.py E1')
        if 'E4' in args.studies and not all((x/'.git').exists() for x in (args.solver,args.architecture)):p.error('Run: python scripts/setup_dependencies.py E4')
        for study in args.studies:
            dest=args.output/study
            if dest.exists():p.error(f'Output already exists: {dest}. Choose a new --output directory; supplied records will not be overwritten.')
            # Ensure an explicit custom output cannot modify archived research data.
            if dest==ROOT or ROOT/'studies' in dest.parents or ROOT/'results' in dest.parents:p.error('Choose an output outside the retained study/results directories.')
        args.output.mkdir(parents=True,exist_ok=True)
    env=os.environ.copy()
    if 'E4' in args.studies:
        env['PYTHONPATH']=str(args.solver)+(os.pathsep+env['PYTHONPATH'] if env.get('PYTHONPATH') else '')
    for study in args.studies:
        source,archived=layout(study);dest=args.output/study
        replay = args.action == 'replay' and not args.dry_run
        if replay:
            shutil.copytree(archived, dest)
        frozen_replay = replay and study in ('E1','E2','E3')
        with (TemporaryDirectory(prefix='.frozen-', dir=args.output) if frozen_replay else nullcontext()) as folder:
            if folder is not None:
                staged = Path(folder) / 'source'
                stage_frozen_replay(source, staged)
                source = staged
            for command in commands(args,study,source):
                print(study+': '+subprocess.list2cmdline(command),flush=True)
                if not args.dry_run:subprocess.run(command,cwd=source,env=env,check=True)

if __name__=='__main__':main()
