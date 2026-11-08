# ASOC Experiment 4: completed constant-delay comparison

Completed 64 calibration fits and 120 final fits, with 0 calibration failures and 0 final failures, plus 10 direct Anchor-C estimates. All eight configurations were selected from separate calibration data and locked before final fitting.

**Recovery disclosure.** Workspace maintenance deleted the original attempt before durable saving. Earlier progress messages had shown its calibration completion and selected settings. The results below come from a reconstructed, fully rerun and audited package, using the same scientific design/seeds and eight instead of four one-thread CPU workers. The recorded rerun uses NumPy 2.3.5 (the earlier attempt used 2.4.2), and all eight selected mean validation scores reproduce the earlier printed values exactly. This is not a newly blind preregistration. Sources were frozen before this rerun; no settings were changed based on the final outcomes.

The constant-delay comparison does not support a general anchor advantage. Median paired delay error improved in two of the four architecture/noise comparisons and worsened in two. Median paired clean forecast MSE increased in all four comparisons (ratios 1.504–2.773). These are descriptive condition-wise results over the same five base-noise datasets, not four independent replications. The marginal and paired summaries below are both retained, because they can give different rankings.

## Primary outcomes

Take the median of three optimizer repeats within each dataset, then the median over five independent base-noise datasets. Noise levels reuse standardized draws and are paired conditions, not ten independent datasets. Forecast outcome is clean held-out trajectory MSE at the prespecified common Euler step 0.00625 on observation grid 0.025. Scalar absolute delay error is a separate primary outcome. Anchor-C estimates only a delay and has no forecast score.

| Noise SD | Method | Median absolute delay error | Median clean test MSE |
| --- | --- | --- | --- |
| 0.0005 | External-C | 0.0442148 | 2.26536e-06 |
| 0.0005 | External-C+anchors | 0.0296625 | 3.70003e-06 |
| 0.0005 | Structured-C | 0.0413662 | 2.65468e-06 |
| 0.0005 | Structured-C+anchors | 0.0289841 | 9.2317e-06 |
| 0.0005 | Anchor-C | 0.02975 | — |
| 0.0015 | External-C | 0.0275929 | 2.78558e-06 |
| 0.0015 | External-C+anchors | 0.0366534 | 3.85948e-06 |
| 0.0015 | Structured-C | 0.0408346 | 2.7948e-06 |
| 0.0015 | Structured-C+anchors | 0.0358495 | 7.12466e-06 |
| 0.0015 | Anchor-C | 0.03675 | — |


The constant-delay subcase compares two architectures that can represent the target delay. External-C estimates an unrestricted current/delayed-state field. Structured-C imposes an additional affine delayed-state channel and negative-gain prior; cross-architecture rankings include these different restrictions. External-C+anchors is our adaptation of the released dimensions/solver, not an unchanged published method.

## Within-architecture anchor comparisons

Ratios are formed within each dataset after taking its optimizer median. Below 1 favors the anchored numerator. These paired median ratios need not equal ratios of the marginal medians above. All five ratios/ranges are reported later.

| Noise SD | Architecture | Delay error ratio | Delay wins | Forecast MSE ratio | Forecast wins |
| --- | --- | --- | --- | --- | --- |
| 0.0005 | External-C | 1.04539 | 2/5 | 1.59806 | 1/5 |
| 0.0005 | Structured-C | 0.912969 | 3/5 | 2.52982 | 0/5 |
| 0.0015 | External-C | 0.72609 | 3/5 | 1.50383 | 1/5 |
| 0.0015 | Structured-C | 1.57964 | 2/5 | 2.77273 | 0/5 |


These compare validation-calibrated pipelines. When selected rates differ, the final comparison is not a pure fixed-configuration anchor-loss intervention. Within-architecture initialization is paired, and all candidates received the same validation grid. Equal update budgets do not imply equal compute costs. No population significance, confidence interval or new arbitrary 20% success criterion is claimed.

## All paired dataset outcomes

| Noise SD | Numerator / denominator | Outcome | Median ratio | Wins | All five dataset ratios |
| --- | --- | --- | --- | --- | --- |
| 0.0005 | External-C+anchors / External-C | Delay | 1.04539 | 2/5 | 1.04539, 0.588171, 1.3466, 0.564246, 1.04839 |
| 0.0005 | External-C+anchors / External-C | Forecast | 1.59806 | 1/5 | 2.06671, 0.935654, 1.64189, 1.59806, 1.13194 |
| 0.0015 | External-C+anchors / External-C | Delay | 0.72609 | 3/5 | 3.17175, 0.187646, 0.561137, 0.72609, 2.0697 |
| 0.0015 | External-C+anchors / External-C | Forecast | 1.50383 | 1/5 | 2.14618, 0.681141, 1.56389, 1.025, 1.50383 |
| 0.0005 | Structured-C+anchors / Structured-C | Delay | 0.912969 | 3/5 | 0.912969, 0.316258, 0.207002, 1.28573, 2.75136 |
| 0.0005 | Structured-C+anchors / Structured-C | Forecast | 2.52982 | 0/5 | 6.57331, 2.18066, 3.12615, 2.3987, 2.52982 |
| 0.0015 | Structured-C+anchors / Structured-C | Delay | 1.57964 | 2/5 | 1.97948, 0.0681217, 0.226515, 1.57964, 3.19244 |
| 0.0015 | Structured-C+anchors / Structured-C | Forecast | 2.77273 | 0/5 | 17.87, 1.70141, 2.13362, 2.77273, 3.09494 |
| 0.0005 | Structured-C / External-C | Delay | 1.09154 | 2/5 | 1.09154, 1.71583, 5.86429, 0.428814, 0.370262 |
| 0.0005 | Structured-C / External-C | Forecast | 1.48894 | 0/5 | 1.17186, 2.01367, 1.61134, 1.00369, 1.48894 |
| 0.0015 | Structured-C / External-C | Delay | 1.55467 | 2/5 | 1.55467, 2.1179, 2.63688, 0.449573, 0.630204 |
| 0.0015 | Structured-C / External-C | Forecast | 1.13247 | 2/5 | 0.74712, 1.83604, 1.13247, 0.67479, 1.7875 |
| 0.0005 | Structured-C+anchors / External-C+anchors | Delay | 0.953268 | 5/5 | 0.953268, 0.922596, 0.901471, 0.977127, 0.971699 |
| 0.0005 | Structured-C+anchors / External-C+anchors | Forecast | 3.32768 | 0/5 | 3.72717, 4.69311, 3.06798, 1.50654, 3.32768 |
| 0.0015 | Structured-C+anchors / External-C+anchors | Delay | 0.972067 | 4/5 | 0.970263, 0.768871, 1.06444, 0.978067, 0.972067 |
| 0.0015 | Structured-C+anchors / External-C+anchors | Forecast | 3.67874 | 0/5 | 6.22083, 4.58622, 1.54503, 1.82538, 3.67874 |


Every dataset value, marginal range and paired difference is in summary.json; dataset_medians.csv and final_runs.csv support inspection. Missing/failed required fits withhold complete comparisons rather than being silently dropped. Ratios with zero denominator are undefined; differences remain available.

## Calibration and numerical audit

| Method/noise | Selected Adam rate | Updates | Mean noisy validation MSE |
| --- | --- | --- | --- |
| External-C\|sigma0.0005 | 0.0005 | 1200 | 2.32493e-06 |
| External-C\|sigma0.0015 | 0.002 | 1200 | 4.03181e-06 |
| External-C+anchors\|sigma0.0005 | 0.002 | 1200 | 2.80018e-06 |
| External-C+anchors\|sigma0.0015 | 0.002 | 1200 | 5.70673e-06 |
| Structured-C\|sigma0.0005 | 0.002 | 1200 | 3.13972e-06 |
| Structured-C\|sigma0.0015 | 0.002 | 1200 | 4.76331e-06 |
| Structured-C+anchors\|sigma0.0005 | 0.002 | 1200 | 4.95644e-06 |
| Structured-C+anchors\|sigma0.0015 | 0.002 | 1200 | 5.64599e-06 |


Each method/noise cell received four choices: rates 0.0005/0.002 crossed with 280/1200 updates on two separate calibration datasets, for 64 actual fits. Selection used noisy validation trajectory MSE only, never delay/test error. Ties prefer shorter budget then smaller rate. Candidate failure makes that configuration ineligible; no completed failure is retried. The final update is the checkpoint, without early stopping. The bounded grid does not prove optimal tuning.

Reference RK4 step 0.00125 passed the first consecutive refinement check, maximum difference 7.70659e-08 across 40 trajectories on the 0.025 grid (tolerance1e-6). This is empirical convergence, not a rigorous enclosure. Every frozen final model was evaluated at Euler steps 0.025/0.0125/0.00625 without checkpoint selection.

Consistency audit: **passed**, 184 checkpoints replayed, 14 raw observation datasets regenerated, 10 direct estimates and 4988 checks. Maximum metric difference 0. Selection and score arithmetic were recomputed independently; frozen source/third-party hashes, lock timing, initialization pairing and summaries were checked. Shared architecture/solver code means this is consistency replay, not independent scientific replication.

The separate reserved eight-fit smoke and direct finite-difference gradient checks passed before the rerun; they are not comparative evidence. The archived adjoint route was excluded following the prior documented gradient failures. No third-party source was modified.

Per-fit records retain training/total job time, parameter counts/hashes, clipping/projection/boundary counts, predicted observation minima and nonpositive observation counts at every step. Nonpositive counts concern sampled observations and do not certify continuous-time positivity. Finite but unfavorable outcomes remain included. Peak memory was not measured per fit.

| Stage | Successful fits | Sum of training job seconds |
| --- | --- | --- |
| Calibration | 64 | 1977.32 |
| Final | 120 | 5881.01 |


Summed training durations are job times, not elapsed wall time, and exclude some startup/evaluation overhead. The manifest records actual software versions and eight one-thread CPU workers.

## Model and source scope

Generate x′=1.1x[1−x(t−0.28)/1.5], with 16 training, 12 validation, 12 test trajectories, exact prescribed trigonometric histories/initial states, horizon 1, noisy postinitial fitting observations every 0.05. No true delay, coefficient or derivative target is used in the optimizer, anchor estimator or selector. Every learned delay starts at 0.325 and is projected in place to [0.15,0.50]. Profile anchors reuse training observations via right-side degree 2 polynomial slope fits on [0,0.20]; they add no independent labels.

External-C has four width 32/ReLU hidden layers (3,298 parameters); Structured-C has width 28/22, two-hidden-layer tanh instantaneous/gain networks (1,471 parameters) with negative gain. All methods use the unchanged pinned torchdde Euler/direct-backpropagation path and fitting step 0.025. Analytic history/t0=0, known bounds, scalarizing our delay branch, common fixed Adam schedule and curriculum are disclosed adaptations from the source paper’s native observed-prefix/adaptive/OneCycle experiment. This is not a reproduction of its original training experiment or adjoint algorithm.

Source: [Monsel et al., DOI10.1017/dce.2026.10053](https://doi.org/10.1017/dce.2026.10053), [accessible preprint](https://arxiv.org/html/2410.02843v2), architecture commit9b573c4bed5efee9fe9c6b765044dbd506b5bf07, solver commit4888c63dc84e966f6c6553277adc3851a2cca4fc. Archive DOIs:10.5281/zenodo.20113438 and10.5281/zenodo.20113429. Repositories are obtained separately, with exact tracked-file hashes in the manifest; they are not bundled.

## Consequences for the ASOC revision

Use this as a restricted external comparison and report recovery/forecast tradeoffs explicitly. The comparison cannot establish superiority over a method learning an unknown state-dependent delay function. The unavailable GNDDE full methods remain an unresolved comparison, not a demonstrated incompatibility. Retain Anchor-C, all unfavorable conditions and E1–E3’s original criteria/results.

R1’s abstract currently emphasizes one archived 71.7% improvement. Keep it only with its study attribution; revise the overall abstract/conclusion to reflect the mixed newer evidence and strong anchor-only controls. Distinguish the original positive-gain scalar system from the newer negative-gain logistic system. Replace the old statement that a regenerated-data/retrained resolution study is still needed, because E3 supplies that study. Preserve fixed matched-slope theorem scope and conditional error-envelope requirements; empirical resampling remains a sensitivity diagnostic.

The next manuscript revision should consolidate the main article to eight coherent sections, place full protocols and dataset tables in the supplement, integrate E1–E4, update code availability truthfully, and compile/inspect both complete Overleaf projects. This experiment package itself is not the revised article or a submission-ready project.

This rerun source freeze: 2026-10-20T08:57:42.750944+00:00. Configuration lock: 2026-10-20T09:02:13.344130+00:00. See manifest.json, selection_lock.json, all trial/final records and audit files for exact evidence.
