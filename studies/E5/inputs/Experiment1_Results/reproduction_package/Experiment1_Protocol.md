# ASOC Experiment 1: joint fitting versus anchor-only delay learning

Protocol version 1.0, prepared 19 October 2026. This protocol is fixed before generating the new experiment's outcomes. It is an internal prospective protocol, not a public preregistration. The earlier direct-interpolation pilot remains exploratory and uses different noise seeds.

## Introduction review and purpose

The R1 introduction now connects the paper to reliable neural system identification: a good forecast does not by itself justify interpreting a learned delay as a physical memory mechanism. It explains the structured neural learner, the reuse of training observations as delay anchors, and the separation between conditional deterministic guarantees and empirical sensitivity diagnostics. These are appropriate central claims to retain.

The unresolved question is computational: does joint trajectory fitting improve delay recovery beyond fitting a neural function to the same four anchors? More journal citations cannot answer that question. This experiment addresses it directly. The current introduction can remain the working version while the experiment is run; the empirical contribution paragraph should be finalized after the outcomes are known. No claim of broad application superiority or calibrated empirical confidence should be added.

## Fixed comparison

Use the scalar mechanism, histories, solver, network classes, and profile estimator from <https://github.com/bilkentli88/NSDDNew> at commit `ee1014a80760dd720e680c7597a3b985f0f4b609`. The repository is a pinned external dependency, not bundled or modified. The runner checks both the commit and tracked experiment-source integrity.

| Method | Training information | Budget | Role |
| --- | --- | --- | --- |
| P3 | Full noisy training trajectories plus four profile anchors | Original 280-update schedule | Joint learner |
| P4 | Same noisy trajectories, no anchor term | Original 280-update schedule | Trajectory-only control |
| A1 | Four anchors only | 280 updates | Delay-network control with matched update count |
| A2 | Four anchors only | 1,200 updates | Prespecified longer-optimization sensitivity control |
| Linear | Four anchors only | Deterministic interpolation | Direct baseline |
| PCHIP | Four anchors only | Deterministic interpolation | Direct baseline |

A1 and A2 use the exact delay subnetwork copied from the corresponding full model's initial state. They do not train an instantaneous term or delayed gain and do not define full forward models. All four neural methods begin with identical delay parameters within each dataset/optimization pair. A2 is always run; it is not added conditionally after seeing A1's errors.

## Data and replications

- New independent observation-noise seeds: **20261020, 20261021, 20261022, 20261023, 20261024**. These are pseudorandom seed labels, not execution dates.
- Optimization seed labels: **20261101, 20261102, 20261103**. For zero-based dataset index `d`, initialize the full model using `optimization_seed + 100*d`; use `model_seed + 404` for training RNG initialization, matching the original convention.
- Noise standard deviation: **0.0005**, added only to observations after time zero. Designed histories and initial states stay exact.
- Training initial states: `[-0.75, -0.25, 0.25, 0.75]`, five histories per state, yielding 20 trajectories. Held-out trajectory states: `[-0.50, 0, 0.50]`, yielding 15 trajectories. Use independent held-out observation noise with seed `data_seed + 1000` solely for secondary evaluation.
- Horizon: 1.0; history interval: `[-0.50, 0]`; simulation and rollout step: 0.05. This preserves the original experiment and does **not** establish solver-convergence robustness.
- Delay interval: `[0.15, 0.50]`; anchor-profile delay grid: 351 points. Estimate each anchor using the original degree-two local polynomial on the first five observation times including time zero.
- No post-training assessment probes, certificate filtering, derivative labels, or true-delay labels enter any fit.

The clean simulated trajectories are the same controlled histories under the same mechanism; the five replications are independent noise realizations. This experiment tests the missing ablation and does not add a second physical mechanism or application domain.

## Models, losses, and optimization

The delay network has two tanh hidden layers of width 20 and a sigmoid output scaled to the delay interval. P3/P4 retain the original instantaneous network (two width-28 tanh layers) and gain network (two width-22 tanh layers, gain `0.05 + softplus(output)`).

P3/P4 use the original staged rollout schedule: 90 updates over the first 10 future steps, followed by 190 updates over all 20 future steps. The trajectory loss averages squared residuals weighted linearly from 0.6 to 1.4 over the selected observation times, including time zero.

P3's loss is trajectory loss plus `0.08 * anchor_MSE + 0.0002 * smoothness`. P4 omits the anchor term. Smoothness is the mean squared first difference of the delay on the original 100-point state grid `[-1.5, 1.5]`. A1/A2 use **exactly** `0.08 * anchor_MSE + 0.0002 * smoothness`, without trajectory loss. This removes the trajectory component while retaining the delay branch's penalty scaling.

All neural fits use Adam, learning rate 0.002, weight decay 0.000001, float32, one CPU thread, and the final iterate. P3/P4 retain the original global gradient clipping at norm 5; A1/A2 apply the same threshold to their trainable delay parameters. Record clipping frequency: because the sets of trainable parameters differ, the clipping multiplier need not be equal across methods. This is an ablation of the implemented training procedures, not an assertion that all optimizer effects are isolated.

No profile initialization, early stopping on evaluation error, learning-rate search, regularization search, test-based checkpoint selection, or extra optimization seeds are permitted in this protocol. No noiseless or harder-noise cell is added after seeing these outcomes; those require a separately recorded extension.

## Metrics and primary summaries

Evaluate every delay function on the same 201-point grid over `[-0.75, 0.75]`, without extrapolating the direct interpolants. Record delay RMSE, MAE, maximum absolute error, anchor-fit MSE, and the full predicted curve. True delays are used only to compute evaluation metrics. For P3/P4 additionally record clean training, clean held-out, and noisy held-out trajectory MSE, plus training time and final objective components. Leave trajectory fields missing for A1/A2 and the interpolants.

For each noisy dataset, take the median delay RMSE over the three optimization repeats of each neural method. Each direct interpolant has one value per dataset. The **primary estimands** are the five dataset-level ratios `median(P3 RMSE)/median(A1 RMSE)` and `median(P3 RMSE)/median(A2 RMSE)`. Report all five ratios, their median and range, absolute paired differences, and the number below one. Also report P3/P4, P3/Linear, and P3/PCHIP as secondary comparisons. Do not count an interpolant three times as independent data and do not present all 15 optimization runs as 15 independent noise datasets.

A prespecified descriptive criterion for a consistent practical joint-fitting advantage is a median dataset-level RMSE ratio at most **0.80** and improvement in at least **4/5 datasets**, against **both** A1 and A2. The 20% threshold is a chosen practical effect size, not a power calculation or statistical significance test. Report whether it is met; retain all results if it is not met. With only five noise datasets, this criterion does not establish population-level superiority. Report the methods' absolute errors even when this criterion is met.

No statistical-confidence or deterministic-certification conclusion is part of this experiment. If A1/A2 match or improve on P3, revise the paper's neural-benefit claim accordingly; the structural theory may still hold.

## Fixed accounting and failure policy

A complete run contains **60 neural fits** (five datasets × three seeds × four methods), **10 deterministic baseline functions**, and five sets of four anchors. Save final checkpoints, raw observations, anchors, delay curves, all run-level metrics, initialization hashes, package-source hashes, repository commit, software versions, and configuration hash.

Write the execution manifest and protocol/configuration hashes before the first fit. Resume only when the mode, configuration, protocol, runner, and source hashes match. Resume skips completed run records, including recorded numerical failures; it does not silently retry failed seeds. If interrupted during a fit, restarting the identical fit is an infrastructure resumption, and its restart is logged. Nonfinite losses, gradients, parameters, or predictions produce a recorded failed fit. A complete-case primary verdict is withheld if any required fit fails; report failure counts and available descriptive results instead of silently reducing the denominator.

Smoke mode uses **one old archived noise seed and one old optimization seed**, with four updates for P3/P4/A1 and eight for A2. It checks execution, initialization pairing, finite outputs, and result accounting. It is excluded from the experiment and must not be cited as scientific evidence.

## Run and return

Open `ASOC_Experiment1_Colab.ipynb` in Google Colab. Upload `ASOC_Experiment1_Run_Package.zip` when its first cell prompts. Run the cells in order; a CPU runtime is sufficient. The notebook installs the pinned training dependencies, checks the repository, verifies the archived anchors, runs the smoke check, and then runs the fixed full experiment. Leave the Colab session connected until the final results ZIP downloads. Runtime depends on the session; no completion-time estimate has been benchmarked.

Return **ASOC_Experiment1_Results.zip** for audit and manuscript integration. The run package itself contains no full-experiment outcomes. Preparation checks cover data/anchor consistency, protocol accounting, summary logic, notebook syntax, and package integrity. A separate isolated CPU environment is used to check neural execution; its validation record is supplied separately. The notebook repeats the smoke check before starting its full training run.
