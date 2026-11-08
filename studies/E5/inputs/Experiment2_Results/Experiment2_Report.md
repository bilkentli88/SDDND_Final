# ASOC Experiment 2: completed second-mechanism study

This study uses a synthetic state-dependent delayed logistic mechanism and a new four-history trigonometric design. It checks generality beyond Experiment 1's original scalar mechanism. It is not validation on measured population data.

## Main result and fixed decision

Completed actual neural fits: **180**. Informative direct functions: **20**. Expected structural abstention records: **80**. Recorded numerical failures: **0**.

The prespecified criterion for a consistent practical joint-fitting advantage against both anchor-only controls at both noise levels was **not met**. The criterion remains the chosen 20% median paired improvement and at least 4/5 dataset improvements against **both A1 and A2**, evaluated separately at each noise level. It is a descriptive target, not statistical significance or a submission requirement. The complete results below must accompany any claim of improvement.

## Mathematical fit and necessary implementation changes

The true equation is `x' = 1.1*x*(1 - delayed_x/1.5)` with delay `0.28 + 0.07*tanh(2*(x-0.8))`. Thus the instantaneous and delayed coefficients are `a(s)=1.1*s` and `b(s)=-(1.1/1.5)*s`. The neural models learn their own unknown coefficient fields; these true coefficient functions are not supplied as model parameters. Both neural dynamics models share a known negative-gain prior.

The true gain magnitude is at least 0.33 on the target interval [0.45,1.20]. The four positive trigonometric histories yield a centered signature whose coefficient ratio is `tan(pi*r/2)`, strictly increasing on the delay interval [0.15,0.50]. The history map is therefore analytically projectively injective. Its norm is at least 0.118851; the ideal unsquared distance profile has a conservative uniform separation lower bound of 0.058551. These ideal quantities do not create a deterministic certificate under Gaussian observation noise.

`ASOC_Experiment2_Model_Check.md` contains the full calculation, positive-solution bounds, and the explicit **right-hand initial response** convention. Derivative compatibility at the initial-history junction is not assumed. The theorem being tested is the fixed matched-state algebra; the study does not assert an iff characterization of the full trajectory observation operator. Four observed training anchors also do not certify delay estimates across the entire 201-point evaluation grid.

## Protocol, data and numerical prerequisite

The full-run manifest was written before reference generation at `2026-10-19T20:08:10.187979+00:00` and hashes the protocol, model check, configuration, data code, model code and runner. The audit confirms those inputs remain unchanged. This was an internal prospective protocol, not a public preregistration. Reserved-seed smoke results are separate.

The design has two noise levels (0.0005 and 0.0015), five independent **base-noise realizations**, three paired model initializations per dataset, and informative/degenerate history conditions. Standardized noise is reused across levels and history designs, so these blocks are paired. They must not be described as 20 independent datasets. There are five independent base-noise replications per condition. Histories and initial states are exact, and only post-initial observations receive noise.

There are 16 training and 12 held-out trajectories per design, observed every 0.05 through time 1.0. The reference RK4 method-of-steps refinement selected step 0.00125. The maximum consecutive-refinement difference at observation times was 1.550e-07, below the fixed 1e-6 criterion across all train/test trajectories and both history designs. This is an empirical convergence check, not a rigorous error enclosure.

The learned solver uses Euler substeps of 0.025 and analytic prescribed histories before time zero. This does not replace the pending study that changes observation resolution, rebuilds anchors and retrains. Several design factors differ from Experiment 1 simultaneously; this study tests transfer across settings, not the isolated causal effect of each changed factor.

## Delay-recovery results on informative histories

P3 is rollout plus profile anchors; P4 is rollout only. A1 fits the identical delay subnetwork only to anchors for 280 updates; A2 uses 1,200 updates. Both direct interpolants use the same four anchors. The final iterate is used throughout, without test-based tuning. A1/A2 and the interpolants define delay functions only, so no complete-trajectory comparison is claimed for them.

The table summarizes the median across the five **within-dataset median** delay RMSE values. Each within-dataset neural median uses three optimization repeats; each direct method contributes one value.

| Method | Noise SD 0.0005 | Noise SD 0.0015 |
| --- | ---: | ---: |
| P3 | 0.020937 | 0.054719 |
| P4 | 0.063069 | 0.063221 |
| A1 | 0.020307 | 0.068953 |
| A2 | 0.042326 | 0.109558 |
| Linear | 0.033260 | 0.088672 |
| PCHIP | 0.034162 | 0.100362 |

The primary paired ratios below are computed within datasets before their median is taken. A ratio of the separate medians above is a different statistic and must not be substituted for the median paired ratio.

| Noise SD | Comparator | Median paired ratio | P3 improves | Chosen practical criterion |
| --- | --- | ---: | ---: | --- |
| 0.0005 | A1 | 0.972 | 3/5 | Not met |
| 0.0005 | A2 | 0.655 | 4/5 | Met |
| 0.0005 | P4 | 0.382 | 5/5 | Met |
| 0.0005 | Linear | 0.695 | 4/5 | Met |
| 0.0005 | PCHIP | 0.648 | 4/5 | Met |
| 0.0015 | A1 | 0.968 | 3/5 | Not met |
| 0.0015 | A2 | 0.544 | 5/5 | Met |
| 0.0015 | P4 | 0.845 | 3/5 | Not met |
| 0.0015 | Linear | 0.633 | 4/5 | Met |
| 0.0015 | PCHIP | 0.545 | 4/5 | Met |

### Noise SD 0.0005: all five paired ratios

| Base noise seed | P3/A1 | P3/A2 | P3/P4 | P3/Linear | P3/PCHIP |
| --- | ---: | ---: | ---: | ---: | ---: |
| 20261201 | 1.031 | 0.495 | 0.382 | 0.630 | 0.613 |
| 20261202 | 1.529 | 1.372 | 0.169 | 1.128 | 1.067 |
| 20261203 | 0.972 | 0.661 | 0.433 | 0.732 | 0.667 |
| 20261204 | 0.946 | 0.655 | 0.580 | 0.695 | 0.648 |
| 20261205 | 0.967 | 0.432 | 0.214 | 0.450 | 0.448 |

### Noise SD 0.0015: all five paired ratios

| Base noise seed | P3/A1 | P3/A2 | P3/P4 | P3/Linear | P3/PCHIP |
| --- | ---: | ---: | ---: | ---: | ---: |
| 20261201 | 0.879 | 0.528 | 1.088 | 0.684 | 0.587 |
| 20261202 | 1.023 | 0.977 | 2.035 | 1.032 | 1.172 |
| 20261203 | 0.968 | 0.557 | 0.840 | 0.633 | 0.545 |
| 20261204 | 1.021 | 0.455 | 0.670 | 0.540 | 0.463 |
| 20261205 | 0.572 | 0.544 | 0.845 | 0.627 | 0.535 |

All run-level values and dataset summaries are retained in `runs.csv`, `dataset_summaries.csv` and `dataset_pairs.csv`. A longer anchor-only optimization budget is a prespecified sensitivity control, not a guarantee that the best possible direct estimator has been found. No hypothesis-test or population-confidence claim is made from five base-noise replications.

## Degenerate histories and structural abstention

All four constant histories at each matched state have zero centered delayed signature. The history-only gate rejects all such anchors. Consequently the anchor-only methods and interpolants issue **80 structural abstention records**, rather than reporting a grid endpoint as an estimate. P3 then has the same loss and initialization as P4.

Saved P3/P4 parameters are bitwise identical in **30/30 paired degenerate runs**. This verifies the specified algorithmic negative control. It concerns failure of the matched-initial-state mechanism, not universal impossibility of identifying a delay using any possible full-trajectory experiment. Structural abstention here is deliberately based on exact known history geometry; it is not evidence of calibrated uncertainty under noisy histories.

Model-domain diagnostics found 0 nonpositive predicted observation values. Of the 120 evaluated full-model fits, 0 had any such value; these diagnostics were retained rather than used to exclude fits. Recorded clipping events across all trained models: 0.

## Audit and implications for the manuscript

The audit replayed 180 neural checkpoint delay curves, 120 full-model trajectory-metric sets and 20 direct functions. It reconstructed 20 condition/dataset observation sets and 80 anchor entries, checked paired initializations, independently recomputed the condition-level summaries and fixed decision rule, and verified source hashes. This is consistency/replay validation, not independent scientific replication.

The runtime used Python 3.12.14, PyTorch 2.10.0+cpu, NumPy 2.4.2, pandas 2.2.3 and SciPy 1.17.0, on one CPU thread. All available outcomes, including competitive baselines and failures, are preserved.

The article can now report a second admissible mechanism and a distinct analytically justified history family. Its empirical claim about joint fitting must follow the per-noise comparisons above and retain Experiment 1's mixed findings. The theoretical identifiability contribution does not require P3 to beat every estimator. Avoid general neural-superiority, calibrated-confidence, or unconditional deterministic-certification claims.

The remaining priorities are end-to-end training-resolution robustness, a compatible external neural-method comparison, and integration/consolidation of the main article and supplement. A further control motivated by these findings is a complete two-stage model: fit the delay network from anchors, freeze its parameters, then learn the instantaneous and gain fields from the same trajectories. Because the existing A1/A2 controls supply only delay functions, their competitive delay errors do not settle the relative forecasting performance of that complete pipeline. Such a new comparison requires its own prospective protocol and should report both delay and trajectory metrics.

A focused `ASOC_Neural_Comparator_Audit.md` is included: the released Monsel et al. state-dependent benchmark supplies its analytic delay, whereas the 2026 GNDDE paper explicitly describes adaptive delay-function identification and warrants a full compatibility/reproduction check. No external-method result is claimed in this study.

## Reproducible files

- `ASOC_Experiment2_Results.zip`: all raw observations, anchors, profiles, curves, checkpoints, manifests, full log, audit, report, paired-ratio figures, mathematical note and code; smoke results are separate.
- `ASOC_Experiment2_Run_Package.zip`: the fixed protocol and self-contained execution/validation/audit code, plus Colab notebook; it embeds no full-study performance outcomes.
- `ASOC_Experiment2_Colab.ipynb`: standalone reproduction notebook.
- `ASOC_Experiment2_Model_Check.md`: mathematical admissibility and history-design verification.

The scientific figures display all five paired ratios and their median, with the preselected 0.80 threshold. They show no confidence intervals or significance labels. Code and outputs supplied here have not been pushed to the original public repository. This package does not establish submission readiness.
