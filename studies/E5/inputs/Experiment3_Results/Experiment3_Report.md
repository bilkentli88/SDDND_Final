# ASOC Experiment 3: completed numerical-resolution study

The fixed study crosses observation spacing with the learned solver's internal step, rebuilds anchors at each sampling resolution, and retrains all methods. It uses only the synthetic logistic mechanism with informative histories and noise SD 0.0005. It does not establish numerical robustness for the original scalar example, the higher noise level, degenerate histories or measured data.

## Completion and interpretation

**180 neural fits, 20 direct functions, 0 numerical failures.** The 120 P3/P4 fits cover all four resolution cells; the 60 A1/A2 fits are shared across solver settings because they use anchors without a rollout solver. All 200 outcomes are retained.

The study measures finite numerical sensitivity. It does not define a binary robustness verdict or prove asymptotic convergence. All condition tables and unfavorable ratios remain visible.

## Scientific assessment

- **Small solver-step sensitivity for P3 delay recovery in this setting.** Halving the training step gives median paired delay-RMSE ratios 0.9991 at observation spacing 0.05 and 1.0015 at spacing 0.025. Across the five dataset medians at both spacings, the ratios range from 0.9962 to 1.0077. This supports a limited statement about these tested steps and budgets, not solver-independent identification or a convergence rate.
- **Denser observations do not uniformly improve delay recovery.** Halving observation spacing gives P3 median paired delay-RMSE ratios 1.1227 and 1.1269 at the two fixed solver steps, with improvement in only 2/5 datasets in each comparison. The corresponding P4 median ratios are 1.0771 and 1.0377. The physical slope window stays fixed; the reported finite pipeline is not an asymptotic consistency experiment. Changes cannot be attributed solely to slope bias or optimization without further evidence.
- **Delay improvement over rollout-only training persists in these cells.** P3/P4 median paired delay-RMSE ratios range from 0.3045 to 0.3812, with P3 lower in all five datasets in every cell. These are five paired datasets observed in four conditions, not 20 independent improvements. Earlier experiments, especially Experiment 2's higher-noise exceptions, remain part of the evidence.
- **The added gain over the matched anchor-only network remains modest.** P3/A1 median paired ratios are 0.9601–0.9619: approximately 3.8–4.0% lower delay RMSE, with 3/5 wins at coarse sampling and 5/5 at fine sampling. P3/A2 ratios are 0.8473–0.8685, with 5/5 wins. Neither comparison reaches the 20% median-improvement target in any cell; the secondary joint-advantage criterion fails in all four cells. Competitive direct estimators must stay visible.
- **Delay accuracy does not establish a forecasting advantage.** P4 has lower marginal median clean held-out MSE than P3 in all four cells, at both native and common refined evaluation steps. This is a statement about the displayed marginal summaries, not an added paired hypothesis test. The delay-only A1/A2 controls do not resolve forecasting performance of a complete two-stage pipeline.

The revision can now report a controlled separation of sampling and solver effects, retaining the unfavorable sampling and baseline results. It should frame the method around identifiability diagnostics and carefully scoped empirical delay recovery. A general superiority claim for joint neural fitting is unsupported. A compatible external neural-method comparison remains outstanding before the planned consolidation of the main article and supplement.

### Draft manuscript paragraph

To distinguish sampling effects from discretization of the learned dynamics, we crossed observation spacings 0.05 and 0.025 with Euler training steps 0.025 and 0.0125 for the informative logistic example at noise standard deviation 0.0005. The anchor polynomial retained its physical window [0,0.20], and shared observation times used identical noise. Five new noise datasets and three paired initializations produced 180 neural fits. Halving the training step changed P3's dataset-median delay RMSE by less than 0.8% in every paired dataset, whereas denser sampling did not consistently improve recovery. P3 improved on rollout-only delay errors in all five datasets in each resolution cell, but its median paired improvement over the matched 280-update anchor-only network was approximately 4%. The prespecified descriptive 20% target against both anchor-only controls was not met. Frozen-checkpoint trajectory evaluations at a common refined step retained the distinction between predictive fit and delay recovery; these finite-resolution results do not establish asymptotic convergence or general superiority.


## Fixed design and numerical prerequisite

Observation spacings: 0.05 and 0.025. Euler training steps: 0.025 and 0.0125. These are varied independently. Reference trajectories and post-initial Gaussian noise are generated on the fine grid, then downsampled; shared times have identical values. Histories and initial states are exact. Finer sampling provides more observations, so this is not a fixed-observation-budget comparison.

The degree-two right-hand slope fit always uses the physical window [0,0.20]: five observations at spacing 0.05, nine at 0.025. All four profile anchors are reconstructed from those observations. The architecture, loss coefficients, optimizer, 90 updates through T=0.5 followed by 190 through T=1, and final-iterate selection are fixed. A1/A2 use 280/1200 anchor-only updates. Neither true delays nor held-out errors tune fitting.

There are five independent new base-noise datasets and three paired initializations per dataset. Resolution cells and optimizer repeats are paired, not independent replications. Reported absolute errors are medians across the five within-dataset medians over three optimizer repeats. Reported ratios are formed within each dataset first, then their median is taken; these two summary operations must not be interchanged. No significance or confidence-interval claim is made.

The manifest freezes scientific inputs at `2026-10-19T20:39:58.541874+00:00`, before reference generation. This is an internal prospective protocol, not public preregistration. All source hashes remain unchanged. Reserved-seed smoke results are separate. The 14 validation checks cover the geometry, exact sampling/noise pairing, physical-window invariance, polynomial slope fixture, rollout sampling invariance, gradient, reference initial-segment comparison and accounting/missing-data behavior.

Float64 RK4 method-of-steps refinement selected step 0.00125. The maximum consecutive-refinement difference was 1.560e-07, below the fixed 1e-6 prerequisite on the fine observation grid across all 28 training/test trajectories. Analytic negative-time histories and linear computed-past interpolation are used. This empirical check is not an error enclosure or a global fourth-order claim.

The retained Experiment 2 model note supplies the mathematical assumptions. In particular, the positive result concerns matched right-hand initial responses with a scalar affine delayed channel. Four training anchors do not certify a whole state interval, and Gaussian noise does not create deterministic localization certificates.

## Delay errors and anchor estimates

| Observation spacing | Training step | P3 | P4 | A1 | A2 | Linear | PCHIP |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.05 | 0.025 | 0.021657 | 0.062670 | 0.019687 | 0.031076 | 0.028139 | 0.031298 |
| 0.05 | 0.0125 | 0.021805 | 0.064080 | 0.019687 | 0.031076 | 0.028139 | 0.031298 |
| 0.025 | 0.025 | 0.025683 | 0.063785 | 0.026717 | 0.028340 | 0.028587 | 0.025834 |
| 0.025 | 0.0125 | 0.025651 | 0.066494 | 0.026717 | 0.028340 | 0.028587 | 0.025834 |

A1/A2/direct values repeat across the two training-step rows because the same fitted functions are reused. They are not additional independent fits.

| Observation spacing | Window samples | Median anchor RMSE at four training states |
| --- | --- | --- |
| 0.05 | 5 | 0.037365 |
| 0.025 | 9 | 0.036399 |

Anchor RMSE uses the known delay only for post-estimation evaluation. It is not an estimator-selection criterion.

## Paired numerical contrasts

Every ratio is fine/coarse. Values below one indicate lower error at the finer resolution. Solver contrasts hold observation spacing fixed; observation contrasts hold the training solver fixed. The held-out MSE contrasts use the common refined evaluation step 0.00625 and common observation grid 0.025 for both checkpoints, so an evaluation-grid change cannot explain them.

| Contrast | Metric | Median paired ratio | Range across datasets | Finer improves |
| --- | --- | --- | --- | --- |
| Solver: P3, obs=0.05 | Delay RMSE | 0.999 | 0.996–1.007 | 3/5 |
| Solver: P3, obs=0.05 | Refined held-out MSE | 1.012 | 0.962–1.013 | 2/5 |
| Solver: P3, obs=0.025 | Delay RMSE | 1.001 | 0.997–1.008 | 2/5 |
| Solver: P3, obs=0.025 | Refined held-out MSE | 1.005 | 0.964–1.017 | 1/5 |
| Observation: P3, h=0.025 | Delay RMSE | 1.123 | 0.586–1.239 | 2/5 |
| Observation: P3, h=0.025 | Refined held-out MSE | 1.004 | 0.922–1.201 | 2/5 |
| Observation: P3, h=0.0125 | Delay RMSE | 1.127 | 0.588–1.232 | 2/5 |
| Observation: P3, h=0.0125 | Refined held-out MSE | 1.005 | 0.926–1.203 | 2/5 |
| Solver: P4, obs=0.05 | Delay RMSE | 1.023 | 0.986–1.100 | 1/5 |
| Solver: P4, obs=0.05 | Refined held-out MSE | 1.031 | 1.021–1.189 | 0/5 |
| Solver: P4, obs=0.025 | Delay RMSE | 1.000 | 0.980–1.065 | 3/5 |
| Solver: P4, obs=0.025 | Refined held-out MSE | 1.030 | 1.024–1.198 | 0/5 |
| Observation: P4, h=0.025 | Delay RMSE | 1.077 | 1.010–1.109 | 0/5 |
| Observation: P4, h=0.025 | Refined held-out MSE | 0.935 | 0.897–1.003 | 4/5 |
| Observation: P4, h=0.0125 | Delay RMSE | 1.038 | 0.984–1.125 | 1/5 |
| Observation: P4, h=0.0125 | Refined held-out MSE | 0.938 | 0.895–1.011 | 4/5 |
| Observation: A1 | Delay RMSE | 1.179 | 0.619–1.391 | 2/5 |
| Observation: A2 | Delay RMSE | 0.912 | 0.642–1.349 | 3/5 |
| Observation: Linear | Delay RMSE | 0.936 | 0.704–1.064 | 3/5 |
| Observation: PCHIP | Delay RMSE | 0.814 | 0.737–1.030 | 3/5 |

### All five paired numerical ratios

| Contrast | Metric | 20261301 | 20261302 | 20261303 | 20261304 | 20261305 |
| --- | --- | --- | --- | --- | --- | --- |
| Solver: P3, obs=0.05 | Delay RMSE | 0.998 | 1.007 | 0.999 | 1.004 | 0.996 |
| Solver: P3, obs=0.05 | Refined held-out MSE | 1.013 | 1.012 | 0.999 | 1.013 | 0.962 |
| Solver: P3, obs=0.025 | Delay RMSE | 0.999 | 1.001 | 1.003 | 1.008 | 0.997 |
| Solver: P3, obs=0.025 | Refined held-out MSE | 1.017 | 1.013 | 1.000 | 1.005 | 0.964 |
| Observation: P3, h=0.025 | Delay RMSE | 0.728 | 1.239 | 1.123 | 0.586 | 1.221 |
| Observation: P3, h=0.025 | Refined held-out MSE | 0.922 | 0.995 | 1.004 | 1.138 | 1.201 |
| Observation: P3, h=0.0125 | Delay RMSE | 0.729 | 1.232 | 1.127 | 0.588 | 1.222 |
| Observation: P3, h=0.0125 | Refined held-out MSE | 0.926 | 0.997 | 1.005 | 1.129 | 1.203 |
| Solver: P4, obs=0.05 | Delay RMSE | 0.986 | 1.100 | 1.010 | 1.023 | 1.063 |
| Solver: P4, obs=0.05 | Refined held-out MSE | 1.026 | 1.021 | 1.031 | 1.039 | 1.189 |
| Solver: P4, obs=0.025 | Delay RMSE | 1.000 | 0.989 | 0.980 | 1.042 | 1.065 |
| Solver: P4, obs=0.025 | Refined held-out MSE | 1.030 | 1.024 | 1.026 | 1.037 | 1.198 |
| Observation: P4, h=0.025 | Delay RMSE | 1.109 | 1.094 | 1.077 | 1.018 | 1.010 |
| Observation: P4, h=0.025 | Refined held-out MSE | 0.935 | 0.936 | 0.932 | 0.897 | 1.003 |
| Observation: P4, h=0.0125 | Delay RMSE | 1.125 | 0.984 | 1.045 | 1.038 | 1.012 |
| Observation: P4, h=0.0125 | Refined held-out MSE | 0.938 | 0.940 | 0.928 | 0.895 | 1.011 |
| Observation: A1 | Delay RMSE | 0.689 | 1.391 | 1.179 | 0.619 | 1.240 |
| Observation: A2 | Delay RMSE | 0.806 | 0.912 | 1.039 | 0.642 | 1.349 |
| Observation: Linear | Delay RMSE | 0.739 | 1.063 | 1.064 | 0.704 | 0.936 |
| Observation: PCHIP | Delay RMSE | 0.737 | 1.026 | 1.030 | 0.814 | 0.765 |

## Frozen-checkpoint trajectory evaluation

Every full model is evaluated on the common observation grid 0.025 at Euler steps 0.025, 0.0125 and 0.00625. Parameters remain frozen. The following table separates evaluation at the model's training step from evaluation at 0.00625. A1/A2 and direct interpolants supply delay functions only and have no complete-trajectory scores.

| Observation spacing | Training step | P3 native-step MSE | P3 refined-step MSE | P4 native-step MSE | P4 refined-step MSE |
| --- | --- | --- | --- | --- | --- |
| 0.05 | 0.025 | 7.764e-05 | 7.905e-05 | 7.495e-05 | 7.059e-05 |
| 0.05 | 0.0125 | 7.947e-05 | 7.998e-05 | 7.348e-05 | 7.205e-05 |
| 0.025 | 0.025 | 7.627e-05 | 7.866e-05 | 7.098e-05 | 6.610e-05 |
| 0.025 | 0.0125 | 7.890e-05 | 7.972e-05 | 6.938e-05 | 6.771e-05 |

These are clean held-out trajectory MSE values. Intermediate-step, training and noisy-target errors remain in `runs.csv`; no score selected a checkpoint. Three frozen evaluation steps are a sensitivity check, not an asymptotic convergence demonstration. Very low native-solver error may include compensation for that solver's discretization.

## Secondary joint-versus-baseline comparisons

For continuity with earlier studies, the chosen descriptive practical target is median paired P3/comparator delay-RMSE ratio <=0.80 with at least 4/5 dataset improvements. A joint-advantage label requires this against both A1 and A2 in a cell. It is not a hypothesis test, a proof of superiority, or a journal requirement.

| Observation spacing | Training step | Comparison | Median paired ratio | Dataset range | P3 improves | Descriptive target |
| --- | --- | --- | --- | --- | --- | --- |
| 0.05 | 0.025 | P3/P4 | 0.333 | 0.279–0.659 | 5/5 | Met |
| 0.05 | 0.025 | P3/A1 | 0.962 | 0.847–1.100 | 3/5 | Not met |
| 0.05 | 0.025 | P3/A2 | 0.850 | 0.607–0.961 | 5/5 | Not met |
| 0.05 | 0.025 | P3/Linear | 0.770 | 0.585–0.912 | 5/5 | Met |
| 0.05 | 0.025 | P3/PCHIP | 0.838 | 0.487–0.931 | 5/5 | Not met |
| 0.05 | 0.0125 | P3/P4 | 0.305 | 0.274–0.667 | 5/5 | Met |
| 0.05 | 0.0125 | P3/A1 | 0.961 | 0.844–1.108 | 3/5 | Not met |
| 0.05 | 0.0125 | P3/A2 | 0.847 | 0.609–0.959 | 5/5 | Not met |
| 0.05 | 0.0125 | P3/Linear | 0.775 | 0.583–0.910 | 5/5 | Met |
| 0.05 | 0.0125 | P3/PCHIP | 0.837 | 0.486–0.929 | 5/5 | Not met |
| 0.025 | 0.025 | P3/P4 | 0.377 | 0.161–0.433 | 5/5 | Met |
| 0.025 | 0.025 | P3/A1 | 0.961 | 0.834–0.980 | 5/5 | Not met |
| 0.025 | 0.025 | P3/A2 | 0.869 | 0.554–0.947 | 5/5 | Not met |
| 0.025 | 0.025 | P3/Linear | 0.897 | 0.579–0.926 | 5/5 | Not met |
| 0.025 | 0.025 | P3/PCHIP | 0.913 | 0.493–1.038 | 4/5 | Not met |
| 0.025 | 0.0125 | P3/P4 | 0.381 | 0.155–0.432 | 5/5 | Met |
| 0.025 | 0.0125 | P3/A1 | 0.960 | 0.832–0.981 | 5/5 | Not met |
| 0.025 | 0.0125 | P3/A2 | 0.867 | 0.558–0.948 | 5/5 | Not met |
| 0.025 | 0.0125 | P3/Linear | 0.897 | 0.583–0.929 | 5/5 | Not met |
| 0.025 | 0.0125 | P3/PCHIP | 0.916 | 0.496–1.040 | 4/5 | Not met |

The full `dataset_pairs.csv` retains every individual baseline ratio. Longer anchor-only optimization is a fixed sensitivity control, not a claim that the strongest possible direct estimator has been found.

## Domain diagnostics and replay audit

| Evaluation step | Full-model checkpoints | Nonpositive predicted observation values | Checkpoints with any such value |
| --- | --- | --- | --- |
| 0.025 | 120 | 0 | 0 |
| 0.0125 | 120 | 0 | 0 |
| 0.00625 | 120 | 0 | 0 |

Counts refer to observation values, not every internal point or a proof of continuous-time positivity. Diagnostics do not exclude fits. Total recorded clipped updates: 0.

The passed audit reconstructed 10 observation sets and 40 anchor entries, replayed 180 checkpoint delay curves, 360 frozen-solver trajectory-metric sets and 20 direct functions, reconstructed all neural initial delay hashes, verified 15 paired initialization groups, and independently recomputed all dataset medians, contrasts and secondary decisions. This is consistency/replay validation, not independent scientific replication.

Runtime: Python 3.12.14, PyTorch 2.10.0+cpu, NumPy 2.4.2, pandas 2.2.3, SciPy 1.17.0; four CPU workers, one PyTorch thread each. Neural CSV curves round-trip exactly through float32; direct curves are checked to absolute 1e-14.

## Reproducible deliverables

- `ASOC_Experiment3_Results.zip`: complete raw study, checkpoints, curves, observations/anchors/profiles, source/configuration/protocol, audits, full console log, this report and scientific figures; smoke results are separate.
- `ASOC_Experiment3_Run_Package.zip`: self-contained fixed protocol, code, validation and Colab notebook without full-study performance outcomes.
- `ASOC_Experiment3_Colab.ipynb`: standalone unexecuted reproduction notebook; upload the run-package ZIP.

The figures show all five paired datasets and their median for P3/P4, with a tie line at one. A1/A2/direct-method observation contrasts remain in the tables. The figures contain no confidence intervals or significance labels. No files have been pushed to the original public repository. This study alone does not establish submission readiness.
