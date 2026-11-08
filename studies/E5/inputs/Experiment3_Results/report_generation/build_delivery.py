"""Package the audited numerical study; data-derived tables retain every condition."""
from pathlib import Path
import json
import shutil
import subprocess
import sys
import zipfile
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent
RESULTS=ROOT.parent/'asoc_experiment3_full'/'experiment3_results'
OUT=ROOT.parents[1]/'output'/'asoc_experiment3'

ASSESSMENT='''## Scientific assessment

- **Small solver-step sensitivity for P3 delay recovery in this setting.** Halving the training step gives median paired delay-RMSE ratios 0.9991 at observation spacing 0.05 and 1.0015 at spacing 0.025. Across the five dataset medians at both spacings, the ratios range from 0.9962 to 1.0077. This supports a limited statement about these tested steps and budgets, not solver-independent identification or a convergence rate.
- **Denser observations do not uniformly improve delay recovery.** Halving observation spacing gives P3 median paired delay-RMSE ratios 1.1227 and 1.1269 at the two fixed solver steps, with improvement in only 2/5 datasets in each comparison. The corresponding P4 median ratios are 1.0771 and 1.0377. The physical slope window stays fixed; the reported finite pipeline is not an asymptotic consistency experiment. Changes cannot be attributed solely to slope bias or optimization without further evidence.
- **Delay improvement over rollout-only training persists in these cells.** P3/P4 median paired delay-RMSE ratios range from 0.3045 to 0.3812, with P3 lower in all five datasets in every cell. These are five paired datasets observed in four conditions, not 20 independent improvements. Earlier experiments, especially Experiment 2's higher-noise exceptions, remain part of the evidence.
- **The added gain over the matched anchor-only network remains modest.** P3/A1 median paired ratios are 0.9601–0.9619: approximately 3.8–4.0% lower delay RMSE, with 3/5 wins at coarse sampling and 5/5 at fine sampling. P3/A2 ratios are 0.8473–0.8685, with 5/5 wins. Neither comparison reaches the 20% median-improvement target in any cell; the secondary joint-advantage criterion fails in all four cells. Competitive direct estimators must stay visible.
- **Delay accuracy does not establish a forecasting advantage.** P4 has lower marginal median clean held-out MSE than P3 in all four cells, at both native and common refined evaluation steps. This is a statement about the displayed marginal summaries, not an added paired hypothesis test. The delay-only A1/A2 controls do not resolve forecasting performance of a complete two-stage pipeline.

The revision can now report a controlled separation of sampling and solver effects, retaining the unfavorable sampling and baseline results. It should frame the method around identifiability diagnostics and carefully scoped empirical delay recovery. A general superiority claim for joint neural fitting is unsupported. A compatible external neural-method comparison remains outstanding before the planned consolidation of the main article and supplement.

### Draft manuscript paragraph

To distinguish sampling effects from discretization of the learned dynamics, we crossed observation spacings 0.05 and 0.025 with Euler training steps 0.025 and 0.0125 for the informative logistic example at noise standard deviation 0.0005. The anchor polynomial retained its physical window [0,0.20], and shared observation times used identical noise. Five new noise datasets and three paired initializations produced 180 neural fits. Halving the training step changed P3's dataset-median delay RMSE by less than 0.8% in every paired dataset, whereas denser sampling did not consistently improve recovery. P3 improved on rollout-only delay errors in all five datasets in each resolution cell, but its median paired improvement over the matched 280-update anchor-only network was approximately 4%. The prespecified descriptive 20% target against both anchor-only controls was not met. Frozen-checkpoint trajectory evaluations at a common refined step retained the distinction between predictive fit and delay recovery; these finite-resolution results do not establish asymptotic convergence or general superiority.
'''


def table(headers,rows):
    return '| '+' | '.join(headers)+' |\n| '+' | '.join('---' for _ in headers)+' |\n'+''.join('| '+' | '.join(str(x) for x in row)+' |\n' for row in rows)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    c=json.loads((ROOT/'protocol.json').read_text());s=json.loads((RESULTS/'summary.json').read_text())
    a=json.loads((RESULTS/'independent_audit.json').read_text());manifest=json.loads((RESULTS/'manifest.json').read_text())
    ref=json.loads((RESULTS/'reference_convergence.json').read_text());assert a['status']=='passed' and s['complete']
    ds=pd.read_csv(RESULTS/'dataset_summaries.csv');pairs=pd.read_csv(RESULTS/'dataset_pairs.csv');runs=pd.read_csv(RESULTS/'runs.csv')
    def med(obs,h,method,metric='delay_rmse'):
        sel=ds[ds.observation_step.eq(obs)&ds.method.eq(method)]
        sel=sel[sel.rollout_step.eq(h)] if h is not None else sel[sel.rollout_step.isna()]
        assert len(sel)==5 and sel.complete.all()
        return float(sel[metric].median())
    absolute=[];forecast=[];secondary=[];anchors=[]
    for obs in c['observation_steps']:
        for h in c['rollout_steps']:
            absolute.append([obs,h]+[f'{med(obs,h if m in ("P3","P4") else None,m):.6f}' for m in ('P3','P4','A1','A2','Linear','PCHIP')])
            forecast.append([obs,h]+[f'{med(obs,h,m,metric):.3e}' for m in ('P3','P4') for metric in ('native_clean_test_mse','refined_clean_test_mse')])
            for comp in ('P4','A1','A2','Linear','PCHIP'):
                r=next(x for x in s['comparisons'] if x['label']==f'obs={obs:g}, h={h:g}, P3/{comp}')
                secondary.append([obs,h,'P3/'+comp,f"{r['median_paired_ratio']:.3f}",f"{r['minimum_paired_ratio']:.3f}–{r['maximum_paired_ratio']:.3f}",f"{r['datasets_with_improvement']}/5",'Met' if r['practical_criterion_met'] else 'Not met'])
        anchors.append([obs,5 if obs==.05 else 9,f'{med(obs,None,"A1","anchor_rmse"):.6f}'])
    sensitivity=[];five=[]
    for r in s['comparisons']:
        if r['contrast']=='P3_comparison':continue
        label=('Solver: ' if r['contrast']=='solver_refinement' else 'Observation: ')+r['label']
        metric='Delay RMSE' if r['metric']=='delay_rmse' else 'Refined held-out MSE'
        sensitivity.append([label,metric,f"{r['median_paired_ratio']:.3f}",f"{r['minimum_paired_ratio']:.3f}–{r['maximum_paired_ratio']:.3f}",f"{r['datasets_with_improvement']}/5"])
        g=pairs[pairs.contrast.eq(r['contrast'])&pairs.label.eq(r['label'])&pairs.metric.eq(r['metric'])].sort_values('data_seed')
        five.append([label,metric]+[f'{v:.3f}' for v in g.ratio])
    counts=[]
    full=runs[runs.method.isin(['P3','P4'])&runs.status.eq('ok')]
    for h in c['evaluation_steps']:
        col=f'eval_h{h:g}_nonpositive_observation_count'
        counts.append([h,len(full),int(full[col].sum()),int(full[col].gt(0).sum())])
    software=manifest['signature']
    report=f'''# ASOC Experiment 3: completed numerical-resolution study

The fixed study crosses observation spacing with the learned solver's internal step, rebuilds anchors at each sampling resolution, and retrains all methods. It uses only the synthetic logistic mechanism with informative histories and noise SD 0.0005. It does not establish numerical robustness for the original scalar example, the higher noise level, degenerate histories or measured data.

## Completion and interpretation

**{s['trained_neural_fits']} neural fits, {s['direct_functions']} direct functions, {s['numerical_failures']} numerical failures.** The 120 P3/P4 fits cover all four resolution cells; the 60 A1/A2 fits are shared across solver settings because they use anchors without a rollout solver. All 200 outcomes are retained.

The study measures finite numerical sensitivity. It does not define a binary robustness verdict or prove asymptotic convergence. All condition tables and unfavorable ratios remain visible.

{ASSESSMENT}

## Fixed design and numerical prerequisite

Observation spacings: 0.05 and 0.025. Euler training steps: 0.025 and 0.0125. These are varied independently. Reference trajectories and post-initial Gaussian noise are generated on the fine grid, then downsampled; shared times have identical values. Histories and initial states are exact. Finer sampling provides more observations, so this is not a fixed-observation-budget comparison.

The degree-two right-hand slope fit always uses the physical window [0,0.20]: five observations at spacing 0.05, nine at 0.025. All four profile anchors are reconstructed from those observations. The architecture, loss coefficients, optimizer, 90 updates through T=0.5 followed by 190 through T=1, and final-iterate selection are fixed. A1/A2 use 280/1200 anchor-only updates. Neither true delays nor held-out errors tune fitting.

There are five independent new base-noise datasets and three paired initializations per dataset. Resolution cells and optimizer repeats are paired, not independent replications. Reported absolute errors are medians across the five within-dataset medians over three optimizer repeats. Reported ratios are formed within each dataset first, then their median is taken; these two summary operations must not be interchanged. No significance or confidence-interval claim is made.

The manifest freezes scientific inputs at `{manifest['frozen_before_reference_generation_utc']}`, before reference generation. This is an internal prospective protocol, not public preregistration. All source hashes remain unchanged. Reserved-seed smoke results are separate. The 14 validation checks cover the geometry, exact sampling/noise pairing, physical-window invariance, polynomial slope fixture, rollout sampling invariance, gradient, reference initial-segment comparison and accounting/missing-data behavior.

Float64 RK4 method-of-steps refinement selected step {ref['selected_step']:g}. The maximum consecutive-refinement difference was {ref['comparisons'][-1]['max_observation_difference']:.3e}, below the fixed 1e-6 prerequisite on the fine observation grid across all 28 training/test trajectories. Analytic negative-time histories and linear computed-past interpolation are used. This empirical check is not an error enclosure or a global fourth-order claim.

The retained Experiment 2 model note supplies the mathematical assumptions. In particular, the positive result concerns matched right-hand initial responses with a scalar affine delayed channel. Four training anchors do not certify a whole state interval, and Gaussian noise does not create deterministic localization certificates.

## Delay errors and anchor estimates

{table(['Observation spacing','Training step','P3','P4','A1','A2','Linear','PCHIP'],absolute)}
A1/A2/direct values repeat across the two training-step rows because the same fitted functions are reused. They are not additional independent fits.

{table(['Observation spacing','Window samples','Median anchor RMSE at four training states'],anchors)}
Anchor RMSE uses the known delay only for post-estimation evaluation. It is not an estimator-selection criterion.

## Paired numerical contrasts

Every ratio is fine/coarse. Values below one indicate lower error at the finer resolution. Solver contrasts hold observation spacing fixed; observation contrasts hold the training solver fixed. The held-out MSE contrasts use the common refined evaluation step 0.00625 and common observation grid 0.025 for both checkpoints, so an evaluation-grid change cannot explain them.

{table(['Contrast','Metric','Median paired ratio','Range across datasets','Finer improves'],sensitivity)}
### All five paired numerical ratios

{table(['Contrast','Metric']+[str(x) for x in c['data_seeds']],five)}
## Frozen-checkpoint trajectory evaluation

Every full model is evaluated on the common observation grid 0.025 at Euler steps 0.025, 0.0125 and 0.00625. Parameters remain frozen. The following table separates evaluation at the model's training step from evaluation at 0.00625. A1/A2 and direct interpolants supply delay functions only and have no complete-trajectory scores.

{table(['Observation spacing','Training step','P3 native-step MSE','P3 refined-step MSE','P4 native-step MSE','P4 refined-step MSE'],forecast)}
These are clean held-out trajectory MSE values. Intermediate-step, training and noisy-target errors remain in `runs.csv`; no score selected a checkpoint. Three frozen evaluation steps are a sensitivity check, not an asymptotic convergence demonstration. Very low native-solver error may include compensation for that solver's discretization.

## Secondary joint-versus-baseline comparisons

For continuity with earlier studies, the chosen descriptive practical target is median paired P3/comparator delay-RMSE ratio <=0.80 with at least 4/5 dataset improvements. A joint-advantage label requires this against both A1 and A2 in a cell. It is not a hypothesis test, a proof of superiority, or a journal requirement.

{table(['Observation spacing','Training step','Comparison','Median paired ratio','Dataset range','P3 improves','Descriptive target'],secondary)}
The full `dataset_pairs.csv` retains every individual baseline ratio. Longer anchor-only optimization is a fixed sensitivity control, not a claim that the strongest possible direct estimator has been found.

## Domain diagnostics and replay audit

{table(['Evaluation step','Full-model checkpoints','Nonpositive predicted observation values','Checkpoints with any such value'],counts)}
Counts refer to observation values, not every internal point or a proof of continuous-time positivity. Diagnostics do not exclude fits. Total recorded clipped updates: {int(runs.clipped_updates.fillna(0).sum())}.

The passed audit reconstructed {a['raw_observation_sets_reconstructed']} observation sets and {a['anchor_entries_recomputed']} anchor entries, replayed {a['trained_checkpoint_delay_replays']} checkpoint delay curves, {a['frozen_solver_trajectory_metric_sets_replayed']} frozen-solver trajectory-metric sets and {a['direct_function_replays']} direct functions, reconstructed all neural initial delay hashes, verified 15 paired initialization groups, and independently recomputed all dataset medians, contrasts and secondary decisions. This is consistency/replay validation, not independent scientific replication.

Runtime: Python {software['python']}, PyTorch {software['torch']}, NumPy {software['numpy']}, pandas {software['pandas']}, SciPy {software['scipy']}; four CPU workers, one PyTorch thread each. Neural CSV curves round-trip exactly through float32; direct curves are checked to absolute 1e-14.

## Reproducible deliverables

- `ASOC_Experiment3_Results.zip`: complete raw study, checkpoints, curves, observations/anchors/profiles, source/configuration/protocol, audits, full console log, this report and scientific figures; smoke results are separate.
- `ASOC_Experiment3_Run_Package.zip`: self-contained fixed protocol, code, validation and Colab notebook without full-study performance outcomes.
- `ASOC_Experiment3_Colab.ipynb`: standalone unexecuted reproduction notebook; upload the run-package ZIP.

The figures show all five paired datasets and their median for P3/P4, with a tie line at one. A1/A2/direct-method observation contrasts remain in the tables. The figures contain no confidence intervals or significance labels. No files have been pushed to the original public repository. This study alone does not establish submission readiness.
'''
    report_path=OUT/'ASOC_Experiment3_Report.md';report_path.write_text(report)
    subprocess.run([sys.executable,str(ROOT/'make_experiment3_figures.py'),'--results',str(RESULTS),'--output',str(OUT/'figures')],check=True)
    shutil.copy2(ROOT/'ASOC_Experiment3_Colab.ipynb',OUT/'ASOC_Experiment3_Colab.ipynb')
    files=['README.md','protocol.json','ASOC_Experiment3_Protocol.md','ASOC_Experiment2_Model_Check.md',
           'e3_data.py','e3_fit.py','e3_neural.py','e3_summary.py','run_experiment3.py','validate_experiment3.py',
           'audit_experiment3.py','requirements.txt','NSDDNew_LICENSE.txt','ASOC_Experiment3_Colab.ipynb',
           'build_notebook.py','make_experiment3_figures.py','preparation_validation.log']
    with zipfile.ZipFile(OUT/'ASOC_Experiment3_Run_Package.zip','w',zipfile.ZIP_DEFLATED) as z:
        for name in files:z.write(ROOT/name,name)
    with zipfile.ZipFile(OUT/'ASOC_Experiment3_Results.zip','w',zipfile.ZIP_DEFLATED) as z:
        for path in sorted(RESULTS.rglob('*')):
            if path.is_file():z.write(path,'experiment3_results/'+str(path.relative_to(RESULTS)))
        smoke=ROOT.parent/'asoc_experiment3_smoke'/'smoke_results'
        for path in sorted(smoke.rglob('*')):
            if path.is_file():z.write(path,'smoke_results/'+str(path.relative_to(smoke)))
        for name in files:z.write(ROOT/name,'reproduction_package/'+name)
        for name in ('neural_smoke_validation.log','smoke_audit.log'):z.write(ROOT/name,'smoke_results/'+name)
        z.write(ROOT/'build_delivery.py','report_generation/build_delivery.py')
        z.write(report_path,report_path.name)
        z.write(ROOT.parent/'asoc_experiment3_full_run.log','full_run_console.log')
        z.write(ROOT.parent/'asoc_experiment3_full'/'audit.log','audit.log')
        for path in sorted((OUT/'figures').iterdir()):z.write(path,'figures/'+path.name)
    for path in OUT.glob('*.zip'):
        with zipfile.ZipFile(path) as z:assert z.testzip() is None
        print(path.name,path.stat().st_size,'CRC passed')


if __name__=='__main__':main()
