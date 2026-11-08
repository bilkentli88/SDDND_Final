"""Build the completed E2 report and reproducible archives after the audit."""
from pathlib import Path
import json
import shutil
import subprocess
import sys
import zipfile

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent
RESULTS=ROOT.parent/'asoc_experiment2_full'/'experiment2_results'
OUT=ROOT.parents[1]/'output'/'asoc_experiment2'
OUT.mkdir(parents=True,exist_ok=True)
c=json.loads((ROOT/'protocol.json').read_text())
s=json.loads((RESULTS/'summary.json').read_text())
audit=json.loads((RESULTS/'independent_audit.json').read_text())
manifest=json.loads((RESULTS/'manifest.json').read_text())
ref=json.loads((RESULTS/'reference_convergence.json').read_text())
assert audit['status']=='passed'
dataset=pd.read_csv(RESULTS/'dataset_summaries.csv')
pairs=pd.read_csv(RESULTS/'dataset_pairs.csv')
runs=pd.read_csv(RESULTS/'runs.csv')
methods=['P3','P4','A1','A2','Linear','PCHIP']
comparators=['A1','A2','P4','Linear','PCHIP']
verdict=s['consistent_practical_joint_advantage_at_both_noise_levels']
decision='met' if verdict is True else 'not met' if verdict is False else 'inconclusive because required evidence is incomplete'
main_table='| Method | Noise SD 0.0005 | Noise SD 0.0015 |\n| --- | ---: | ---: |\n'
for method in methods:
    values=[]
    for sigma in c['noise_levels']:
        group=dataset[dataset.method.eq(method)&dataset.noise_std.eq(sigma)]
        values.append(f'{group.median_delay_rmse.median():.6f}' if group.complete.all() and len(group)==5 else 'Incomplete')
    main_table+='| '+method+' | '+' | '.join(values)+' |\n'
comparison_table='| Noise SD | Comparator | Median paired ratio | P3 improves | Chosen practical criterion |\n| --- | --- | ---: | ---: | --- |\n'
paired_tables=[]
for sigma in c['noise_levels']:
    level=s['per_noise'][f'{sigma:.4f}']
    for method in comparators:
        x=level['dataset_comparisons'][method]
        value=f"{x['median_dataset_ratio']:.3f}" if x['median_dataset_ratio'] is not None else 'Undefined'
        comparison_table+=f"| {sigma:g} | {method} | {value} | {x['datasets_with_P3_improvement']}/5 | {'Met' if x['descriptive_criterion_met'] else 'Not met'} |\n"
    table=f'### Noise SD {sigma:g}: all five paired ratios\n\n'
    table+='| Base noise seed | P3/A1 | P3/A2 | P3/P4 | P3/Linear | P3/PCHIP |\n| --- | ---: | ---: | ---: | ---: | ---: |\n'
    for seed in c['data_seeds']:
        cells=[]
        for method in comparators:
            group=pairs[pairs.noise_std.eq(sigma)&pairs.data_seed.eq(seed)&pairs.comparator.eq(method)]
            value=group.ratio.iloc[0]
            cells.append(f'{value:.3f}' if np.isfinite(value) else 'Undefined')
        table+='| '+str(seed)+' | '+' | '.join(cells)+' |\n'
    paired_tables.append(table)
domain_count=int(runs.nonpositive_prediction_count.fillna(0).sum())
domain_fits=int(runs.nonpositive_prediction_count.fillna(0).gt(0).sum())
full_model_fits=int((runs.method.isin(['P3','P4']) & runs.status.eq('ok')).sum())
clip_count=int(runs.clipped_updates.fillna(0).sum())
gain_floor=1.1*.45/1.5
minimum_norm=2*np.sqrt(2)*.18*np.sin(np.pi*.15/2)
separation=gain_floor*minimum_norm*np.sin(np.pi*.35/2)/.35
box_max=1.2*np.exp(1.1)
box_min=.45*np.exp(1.1*(1-box_max/1.5))
validation={'scope':'Analytic design quantities and execution checks, not noisy learning performance.',
    'target_state_gain_magnitude_lower_bound':gain_floor,
    'centered_history_norm_lower_bound':float(minimum_norm),
    'ideal_unsquared_profile_separation_lower_bound':float(separation),
    'analytic_true_solution_positive_box':[float(box_min),float(box_max)],
    'preparation_tests_passed':11,'reserved_seed_smoke_complete':True}
(ROOT/'model_validation.json').write_text(json.dumps(validation,indent=2)+'\n')
full_software=manifest['signature']
report=f'''# ASOC Experiment 2: completed second-mechanism study

This study uses a synthetic state-dependent delayed logistic mechanism and a new four-history trigonometric design. It checks generality beyond Experiment 1's original scalar mechanism. It is not validation on measured population data.

## Main result and fixed decision

Completed actual neural fits: **{s['trained_neural_fits']}**. Informative direct functions: **{s['direct_functions']}**. Expected structural abstention records: **{s['structural_abstention_records']}**. Recorded numerical failures: **{s['numerical_failures']}**.

The prespecified criterion for a consistent practical joint-fitting advantage against both anchor-only controls at both noise levels was **{decision}**. The criterion remains the chosen 20% median paired improvement and at least 4/5 dataset improvements against **both A1 and A2**, evaluated separately at each noise level. It is a descriptive target, not statistical significance or a submission requirement. The complete results below must accompany any claim of improvement.

## Mathematical fit and necessary implementation changes

The true equation is `x' = 1.1*x*(1 - delayed_x/1.5)` with delay `0.28 + 0.07*tanh(2*(x-0.8))`. Thus the instantaneous and delayed coefficients are `a(s)=1.1*s` and `b(s)=-(1.1/1.5)*s`. The neural models learn their own unknown coefficient fields; these true coefficient functions are not supplied as model parameters. Both neural dynamics models share a known negative-gain prior.

The true gain magnitude is at least 0.33 on the target interval [0.45,1.20]. The four positive trigonometric histories yield a centered signature whose coefficient ratio is `tan(pi*r/2)`, strictly increasing on the delay interval [0.15,0.50]. The history map is therefore analytically projectively injective. Its norm is at least {minimum_norm:.6f}; the ideal unsquared distance profile has a conservative uniform separation lower bound of {separation:.6f}. These ideal quantities do not create a deterministic certificate under Gaussian observation noise.

`ASOC_Experiment2_Model_Check.md` contains the full calculation, positive-solution bounds, and the explicit **right-hand initial response** convention. Derivative compatibility at the initial-history junction is not assumed. The theorem being tested is the fixed matched-state algebra; the study does not assert an iff characterization of the full trajectory observation operator. Four observed training anchors also do not certify delay estimates across the entire 201-point evaluation grid.

## Protocol, data and numerical prerequisite

The full-run manifest was written before reference generation at `{manifest['frozen_before_reference_generation_utc']}` and hashes the protocol, model check, configuration, data code, model code and runner. The audit confirms those inputs remain unchanged. This was an internal prospective protocol, not a public preregistration. Reserved-seed smoke results are separate.

The design has two noise levels (0.0005 and 0.0015), five independent **base-noise realizations**, three paired model initializations per dataset, and informative/degenerate history conditions. Standardized noise is reused across levels and history designs, so these blocks are paired. They must not be described as 20 independent datasets. There are five independent base-noise replications per condition. Histories and initial states are exact, and only post-initial observations receive noise.

There are 16 training and 12 held-out trajectories per design, observed every 0.05 through time 1.0. The reference RK4 method-of-steps refinement selected step {ref['selected_step']:g}. The maximum consecutive-refinement difference at observation times was {ref['comparisons'][-1]['max_observation_difference']:.3e}, below the fixed 1e-6 criterion across all train/test trajectories and both history designs. This is an empirical convergence check, not a rigorous error enclosure.

The learned solver uses Euler substeps of 0.025 and analytic prescribed histories before time zero. This does not replace the pending study that changes observation resolution, rebuilds anchors and retrains. Several design factors differ from Experiment 1 simultaneously; this study tests transfer across settings, not the isolated causal effect of each changed factor.

## Delay-recovery results on informative histories

P3 is rollout plus profile anchors; P4 is rollout only. A1 fits the identical delay subnetwork only to anchors for 280 updates; A2 uses 1,200 updates. Both direct interpolants use the same four anchors. The final iterate is used throughout, without test-based tuning. A1/A2 and the interpolants define delay functions only, so no complete-trajectory comparison is claimed for them.

The table summarizes the median across the five **within-dataset median** delay RMSE values. Each within-dataset neural median uses three optimization repeats; each direct method contributes one value.

{main_table}
The primary paired ratios below are computed within datasets before their median is taken. A ratio of the separate medians above is a different statistic and must not be substituted for the median paired ratio.

{comparison_table}
{chr(10).join(paired_tables)}
All run-level values and dataset summaries are retained in `runs.csv`, `dataset_summaries.csv` and `dataset_pairs.csv`. A longer anchor-only optimization budget is a prespecified sensitivity control, not a guarantee that the best possible direct estimator has been found. No hypothesis-test or population-confidence claim is made from five base-noise replications.

## Degenerate histories and structural abstention

All four constant histories at each matched state have zero centered delayed signature. The history-only gate rejects all such anchors. Consequently the anchor-only methods and interpolants issue **{s['structural_abstention_records']} structural abstention records**, rather than reporting a grid endpoint as an estimate. P3 then has the same loss and initialization as P4.

Saved P3/P4 parameters are bitwise identical in **{s['degenerate_P3_P4_pairs_equal']}/{s['degenerate_P3_P4_pairs_expected']} paired degenerate runs**. This verifies the specified algorithmic negative control. It concerns failure of the matched-initial-state mechanism, not universal impossibility of identifying a delay using any possible full-trajectory experiment. Structural abstention here is deliberately based on exact known history geometry; it is not evidence of calibrated uncertainty under noisy histories.

Model-domain diagnostics found {domain_count} nonpositive predicted observation values. Of the {full_model_fits} evaluated full-model fits, {domain_fits} had any such value; these diagnostics were retained rather than used to exclude fits. Recorded clipping events across all trained models: {clip_count}.

## Audit and implications for the manuscript

The audit replayed {audit['trained_checkpoint_replays']} neural checkpoint delay curves, {audit['trajectory_metric_replays']} full-model trajectory-metric sets and {audit['direct_function_replays']} direct functions. It reconstructed {audit['raw_condition_dataset_reconstructions']} condition/dataset observation sets and {audit['anchor_entries_recomputed']} anchor entries, checked paired initializations, independently recomputed the condition-level summaries and fixed decision rule, and verified source hashes. This is consistency/replay validation, not independent scientific replication.

The runtime used Python {full_software['python']}, PyTorch {full_software['torch']}, NumPy {full_software['numpy']}, pandas {full_software['pandas']} and SciPy {full_software['scipy']}, on one CPU thread. All available outcomes, including competitive baselines and failures, are preserved.

The article can now report a second admissible mechanism and a distinct analytically justified history family. Its empirical claim about joint fitting must follow the per-noise comparisons above and retain Experiment 1's mixed findings. The theoretical identifiability contribution does not require P3 to beat every estimator. Avoid general neural-superiority, calibrated-confidence, or unconditional deterministic-certification claims.

The remaining priorities are end-to-end training-resolution robustness, a compatible external neural-method comparison, and integration/consolidation of the main article and supplement. A further control motivated by these findings is a complete two-stage model: fit the delay network from anchors, freeze its parameters, then learn the instantaneous and gain fields from the same trajectories. Because the existing A1/A2 controls supply only delay functions, their competitive delay errors do not settle the relative forecasting performance of that complete pipeline. Such a new comparison requires its own prospective protocol and should report both delay and trajectory metrics.

A focused `ASOC_Neural_Comparator_Audit.md` is included: the released Monsel et al. state-dependent benchmark supplies its analytic delay, whereas the 2026 GNDDE paper explicitly describes adaptive delay-function identification and warrants a full compatibility/reproduction check. No external-method result is claimed in this study.

## Reproducible files

- `ASOC_Experiment2_Results.zip`: all raw observations, anchors, profiles, curves, checkpoints, manifests, full log, audit, report, paired-ratio figures, mathematical note and code; smoke results are separate.
- `ASOC_Experiment2_Run_Package.zip`: the fixed protocol and self-contained execution/validation/audit code, plus Colab notebook; it embeds no full-study performance outcomes.
- `ASOC_Experiment2_Colab.ipynb`: standalone reproduction notebook.
- `ASOC_Experiment2_Model_Check.md`: mathematical admissibility and history-design verification.

The scientific figures display all five paired ratios and their median, with the preselected 0.80 threshold. They show no confidence intervals or significance labels. Code and outputs supplied here have not been pushed to the original public repository. This package does not establish submission readiness.
'''
report_path=OUT/'ASOC_Experiment2_Report.md'
report_path.write_text(report)
package_files=['README.md','ASOC_Experiment2_Model_Check.md','ASOC_Experiment2_Protocol.md',
 'protocol.json','e2_data.py','e2_neural.py','run_experiment2.py','validate_experiment2.py',
 'audit_experiment2.py','requirements.txt','ASOC_Experiment2_Colab.ipynb','build_notebook.py',
 'preparation_validation.log','neural_smoke_validation.log','smoke_validation.json','model_validation.json',
 'ASOC_Neural_Comparator_Audit.md','NSDDNew_LICENSE.txt','make_experiment2_figure.py']
subprocess.run([sys.executable,str(ROOT/'make_experiment2_figure.py'),'--results',str(RESULTS),
                '--output',str(OUT/'figures')],check=True)
for name in ('ASOC_Experiment2_Colab.ipynb','ASOC_Experiment2_Model_Check.md'):
    shutil.copy2(ROOT/name,OUT/name)
with zipfile.ZipFile(OUT/'ASOC_Experiment2_Run_Package.zip','w',zipfile.ZIP_DEFLATED) as archive:
    for name in package_files:
        archive.write(ROOT/name,name)
with zipfile.ZipFile(OUT/'ASOC_Experiment2_Results.zip','w',zipfile.ZIP_DEFLATED) as archive:
    for path in sorted(RESULTS.rglob('*')):
        if path.is_file():
            archive.write(path,'experiment2_results/'+str(path.relative_to(RESULTS)))
    smoke=ROOT.parent/'asoc_experiment2_smoke'
    for path in sorted(smoke.rglob('*')):
        if path.is_file():
            archive.write(path,'smoke_results/'+str(path.relative_to(smoke)))
    for name in package_files:
        archive.write(ROOT/name,'reproduction_package/'+name)
    archive.write(report_path,report_path.name)
    archive.write(ROOT.parent/'asoc_experiment2_full_run.log','full_run_console.log')
    archive.write(ROOT.parent/'asoc_experiment2_full'/'audit.log','audit.log')
    archive.write(ROOT/'build_delivery.py','report_generation/build_delivery.py')
    for path in sorted((OUT/'figures').iterdir()):
        archive.write(path,'figures/'+path.name)
for path in OUT.glob('*.zip'):
    with zipfile.ZipFile(path) as archive:
        assert archive.testzip() is None
    print(path.name,path.stat().st_size,'CRC passed')
print('Completed report and reproducible archives prepared.')
