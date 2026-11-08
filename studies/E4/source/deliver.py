"""Format already-frozen summaries and plot scientific results; no fitting."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent
def fmt(x):return 'unavailable' if x is None else f'{x:.6g}'
def table(head,rows):return '\n'.join(['| '+' | '.join(head)+' |','| '+' | '.join(['---']*len(head))+' |']+['| '+' | '.join(str(v).replace('|',r'\|') for v in r)+' |' for r in rows])+'\n'

def main():
 p=argparse.ArgumentParser();p.add_argument('--results',type=Path,required=True);a=p.parse_args();root=a.results.resolve()
 manifest=json.loads((root/'manifest.json').read_text());c=manifest['signature']['config'];s=json.loads((root/'summary.json').read_text());audit=json.loads((root/'audit_report.json').read_text());cal=json.loads((root/'calibration_records.json').read_text());runs=json.loads((root/'final_records.json').read_text());lock=json.loads((root/'selection_lock.json').read_text());conv=json.loads((root/'reference_convergence.json').read_text())
 lines=['# Experiment4: completed constant-delay comparison','',
 f"Completed {len(cal)} calibration fits and {len(runs)} final fits, with {sum(r['status']!='ok' for r in cal)} calibration failures and {sum(r['status']!='ok' for r in runs)} final failures, plus {len(s['direct_estimates'])} direct Anchor-C estimates. All eight configurations were selected from separate calibration data and locked before final fitting.",'',
 '**Recovery disclosure.** Workspace maintenance deleted the original attempt before durable saving. Earlier progress messages had shown its calibration completion and selected settings. The results below come from a reconstructed, fully rerun and audited package, using the same scientific design/seeds and eight instead of four one-thread CPU workers. The recorded rerun uses NumPy2.3.5 (the earlier attempt used2.4.2), and all eight selected mean validation scores reproduce the earlier printed values exactly. This is not a newly blind preregistration. Sources were frozen before this rerun; no settings were changed based on the final outcomes.','',
 '## Primary outcomes','',
 'Take the median of three optimizer repeats within each dataset, then the median over five independent base-noise datasets. Noise levels reuse standardized draws and are paired conditions, not ten independent datasets. Forecast outcome is clean held-out trajectory MSE at the prespecified common Euler step0.00625 on observation grid0.025. Scalar absolute delay error is a separate primary outcome. Anchor-C estimates only a delay and has no forecast score.','']
 rows=[]
 for noise in c['noise_levels']:
  for method in c['methods']:
   m=next((v for v in s['method_medians'] if (v['method'],v['noise_std'])==(method,noise)),None)
   rows.append([f'{noise:g}',method,fmt(m['absolute_delay_error']) if m else 'unavailable',fmt(m['h0.00625_test_clean_mse']) if m else 'unavailable'])
  direct=[v['absolute_delay_error'] for v in s['direct_estimates'] if v['noise_std']==noise];rows.append([f'{noise:g}','Anchor-C',fmt(float(np.median(direct))),'—'])
 lines += [table(['Noise SD','Method','Median absolute delay error','Median clean test MSE'],rows),'',
 'The constant-delay subcase compares two architectures that can represent the target delay. External-C estimates an unrestricted current/delayed-state field. Structured-C imposes an additional affine delayed-state channel and negative-gain prior; cross-architecture rankings include these different restrictions. External-C+anchors is our adaptation of the released dimensions/solver, not an unchanged published method.','',
 '## Within-architecture anchor comparisons','',
 'Ratios are formed within each dataset after taking its optimizer median. Below1 favors the anchored numerator. These paired median ratios need not equal ratios of the marginal medians above. All five ratios/ranges are reported later.','']
 rows=[]
 for noise in c['noise_levels']:
  for family in ['External-C','Structured-C']:
   vals={p['outcome']:p for p in s['paired_comparisons'] if (p['numerator_method'],p['denominator_method'],p['noise_std'])==(family+'+anchors',family,noise)}
   d=vals['absolute_delay_error'];f=vals['h0.00625_test_clean_mse'];rows.append([f'{noise:g}',family,fmt(d['median_ratio']),str(d['dataset_wins'])+'/5',fmt(f['median_ratio']),str(f['dataset_wins'])+'/5'])
 lines += [table(['Noise SD','Architecture','Delay error ratio','Delay wins','Forecast MSE ratio','Forecast wins'],rows),'',
 'These compare validation-calibrated pipelines. When selected rates differ, the final comparison is not a pure fixed-configuration anchor-loss intervention. Within-architecture initialization is paired, and all candidates received the same validation grid. Equal update budgets do not imply equal compute costs. No population significance, confidence interval or new arbitrary20% success criterion is claimed.','',
 '## All paired dataset outcomes','']
 rows=[]
 for p in s['paired_comparisons']:
  rows.append([f"{p['noise_std']:g}",p['numerator_method']+' / '+p['denominator_method'],'Delay' if p['outcome']=='absolute_delay_error' else 'Forecast',fmt(p['median_ratio']),str(p['dataset_wins'])+'/5' if p['complete'] else 'withheld',', '.join(fmt(v['ratio']) for v in p['datasets'])])
 lines += [table(['Noise SD','Numerator / denominator','Outcome','Median ratio','Wins','All five dataset ratios'],rows),'',
 'Every dataset value, marginal range and paired difference is in summary.json; dataset_medians.csv and final_runs.csv support inspection. Missing/failed required fits withhold complete comparisons rather than being silently dropped. Ratios with zero denominator are undefined; differences remain available.','',
 '## Calibration and numerical audit','']
 rows=[]
 for key,val in lock['selections'].items():
  b=val['selected'];rows.append([key,fmt(b['learning_rate']) if b else 'unavailable',b['budget'] if b else '—',fmt(b['score']) if b else '—'])
 lines += [table(['Method/noise','Selected Adam rate','Updates','Mean noisy validation MSE'],rows),'',
 'Each method/noise cell received four choices: rates0.0005/0.002 crossed with280/1200 updates on two separate calibration datasets, for64 actual fits. Selection used noisy validation trajectory MSE only, never delay/test error. Ties prefer shorter budget then smaller rate. Candidate failure makes that configuration ineligible; no completed failure is retried. The final update is the checkpoint, without early stopping. The bounded grid does not prove optimal tuning.','',
 f"Reference RK4 step {conv['selected_step']:g} passed the first consecutive refinement check, maximum difference {conv['comparisons'][-1]['max_difference']:.6g} across40 trajectories on the0.025 grid (tolerance1e-6). This is empirical convergence, not a rigorous enclosure. Every frozen final model was evaluated at Euler steps0.025/0.0125/0.00625 without checkpoint selection.",'',
 f"Consistency audit: **{audit['status']}**, {audit['checkpoints_replayed']} checkpoints replayed, {audit['observation_datasets']} raw observation datasets regenerated, {audit['direct_estimates']} direct estimates and {audit['checks']} checks. Maximum metric difference {audit['maximum_metric_difference']:.6g}. Selection and score arithmetic were recomputed independently; frozen source/third-party hashes, lock timing, initialization pairing and summaries were checked. Shared architecture/solver code means this is consistency replay, not independent scientific replication.",'',
 'The separate reserved eight-fit smoke and direct finite-difference gradient checks passed before the rerun; they are not comparative evidence. The archived adjoint route was excluded following the prior documented gradient failures. No third-party source was modified.','',
 'Per-fit records retain training/total job time, parameter counts/hashes, clipping/projection/boundary counts, predicted observation minima and nonpositive observation counts at every step. Nonpositive counts concern sampled observations and do not certify continuous-time positivity. Finite but unfavorable outcomes remain included. Peak memory was not measured per fit.','',
 table(['Stage','Successful fits','Sum of training job seconds'],[[label,sum(r['status']=='ok' for r in group),fmt(sum(r.get('training_seconds',0) for r in group))] for label,group in [('Calibration',cal),('Final',runs)]]),'',
 'Summed training durations are job times, not elapsed wall time, and exclude some startup/evaluation overhead. The manifest records actual software versions and eight one-thread CPU workers.','',
 '## Model and source scope','',
 'Generate x′=1.1x[1−x(t−0.28)/1.5], with16 training,12 validation,12 test trajectories, exact prescribed trigonometric histories/initial states, horizon1, noisy postinitial fitting observations every0.05. No true delay, coefficient or derivative target is used in the optimizer, anchor estimator or selector. Every learned delay starts at0.325 and is projected in place to[0.15,0.50]. Profile anchors reuse training observations via right-side degree2 polynomial slope fits on[0,0.20]; they add no independent labels.','',
 'External-C has four width32/ReLU hidden layers (3,298 parameters); Structured-C has width28/22, two-hidden-layer tanh instantaneous/gain networks (1,471 parameters) with negative gain. All methods use the unchanged pinned torchdde Euler/direct-backpropagation path and fitting step0.025. Analytic history/t0=0, known bounds, scalarizing our delay branch, common fixed Adam schedule and curriculum are disclosed adaptations from the source paper’s native observed-prefix/adaptive/OneCycle experiment. This is not a reproduction of its original training experiment or adjoint algorithm.','',
 'Source: [Monsel et al., DOI10.1017/dce.2026.10053](https://doi.org/10.1017/dce.2026.10053), [accessible preprint](https://arxiv.org/html/2410.02843v2), architecture commit9b573c4bed5efee9fe9c6b765044dbd506b5bf07, solver commit4888c63dc84e966f6c6553277adc3851a2cca4fc. Archive DOIs:10.5281/zenodo.20113438 and10.5281/zenodo.20113429. Repositories are obtained separately, with exact tracked-file hashes in the manifest; they are not bundled.','',
 '## Interpretation','',
 'Use this as a restricted external comparison and report recovery/forecast tradeoffs explicitly. The comparison cannot establish superiority over a method learning an unknown state-dependent delay function. The unavailable GNDDE full methods remain an unresolved comparison, not a demonstrated incompatibility. Retain Anchor-C, all unfavorable conditions and E1–E3’s original criteria/results.','',
 'R1’s abstract currently emphasizes one archived71.7% improvement. Keep it only with its study attribution; revise the overall abstract/conclusion to reflect the mixed newer evidence and strong anchor-only controls. Distinguish the original positive-gain scalar system from the newer negative-gain logistic system. Replace the old statement that a regenerated-data/retrained resolution study is still needed, because E3 supplies that study. Preserve fixed matched-slope theorem scope and conditional error-envelope requirements; empirical resampling remains a sensitivity diagnostic.','',
 'The next manuscript revision should consolidate the main article to eight coherent sections, place full protocols and dataset tables in the supplement, integrate E1–E4, update code availability truthfully, and compile/inspect both complete Overleaf projects. This experiment package itself is not the revised article or a submission-ready project.','',
 f"This rerun source freeze: {manifest['frozen_before_this_recovery_run_utc']}. Configuration lock: {lock['locked_utc']}. See manifest.json, selection_lock.json, all trial/final records and audit files for exact evidence.",'']
 text='\n'.join(lines)
 for a,b in {'Experiment4':'Experiment 4','NumPy2':'NumPy 2','used2.4':'used 2.4','step0.00625':'step 0.00625','grid0.025':'grid 0.025','Below1':'Below 1','arbitrary20%':'arbitrary 20%','rates0.0005':'rates 0.0005','with280':'with 280','for64':'for 64','across40':'across 40','the0.025':'the 0.025','steps0.025':'steps 0.025','with16':'with 16',',12 validation':', 12 validation',',12 test':', 12 test','horizon1':'horizon 1','every0.05':'every 0.05','at0.325':'at 0.325','to[0.15':'to [0.15','degree2':'degree 2','on[0,0.20]':'on [0,0.20]','width32':'width 32','width28/22':'width 28/22','step0.025':'step 0.025','archived71.7':'archived 71.7'}.items():text=text.replace(a,b)
 findings='The constant-delay comparison does not support a general anchor advantage. Median paired delay error improved in two of the four architecture/noise comparisons and worsened in two. Median paired clean forecast MSE increased in all four comparisons (ratios 1.504–2.773). These are descriptive condition-wise results over the same five base-noise datasets, not four independent replications. The marginal and paired summaries below are both retained, because they can give different rankings.\n\n'
 text=text.replace('## Primary outcomes\n',findings+'## Primary outcomes\n',1)
 (HERE/'REPORT.md').write_text(text)
 plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
 fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained');colors=['#4676a8','#8ab5d6','#a6583c','#e2a184','#666666']
 for col,noise in enumerate(c['noise_levels']):
  for row,outcome in enumerate(['absolute_delay_error','h0.00625_test_clean_mse']):
   ax=axes[row,col];methods=c['methods']+(['Anchor-C'] if row==0 else []);allvalues=[]
   for i,method in enumerate(methods):
    records=s['direct_estimates'] if method=='Anchor-C' else s['dataset_medians'];values=[v[outcome] for v in records if (v['method'],v['noise_std'])==(method,noise)];allvalues+=values
    ax.scatter(i+np.linspace(-.1,.1,len(values)),values,s=28,color=colors[i],zorder=3)
    if values:ax.plot([i-.19,i+.19],[np.median(values)]*2,color='black',lw=2,zorder=4)
   if allvalues and min(allvalues)>0:ax.set_yscale('log')
   ax.set_xticks(range(len(methods)),[m.replace('+anchors','\n+anchors') for m in methods],rotation=20,ha='right');ax.grid(axis='y',alpha=.25);ax.set_title(f'Noise SD {noise:g}');ax.set_ylabel('Absolute delay error' if row==0 else 'Clean held-out trajectory MSE')
 fig.suptitle('Constant-delay study: five base-noise datasets\nNeural dots: medians of three optimizer repeats; bars: medians across datasets')
 for suffix in ['png','pdf']:fig.savefig(HERE/f'constant_delay_outcomes.{suffix}',dpi=180)
 plt.close(fig);print('Report and scientific figures written.')

if __name__=='__main__':main()
