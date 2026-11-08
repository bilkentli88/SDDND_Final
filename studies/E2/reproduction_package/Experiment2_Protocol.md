# Experiment 2: prospective protocol

Version ASOC-E2-v1.0, prepared 19 October 2026. This internal protocol is fixed before the new noisy-learning outcomes; it is not an external preregistration. `ASOC_Experiment2_Model_Check.md` supplies the mathematical scope, history-diversity proof, and initial right-derivative convention.

## Question and fixed design

Test whether the profile-anchor learning results extend to a state-dependent delayed logistic mechanism with **negative** delayed gain and a new four-history **trigonometric** design. Use two observation-noise levels and a structurally degenerate history control. The specific delay law is synthetic; this is an application-motivated controlled study, not validation on measured population data.

All numerical settings are in `protocol.json`. They are selected before the experiment and are not tuned against its delay errors. The target state interval is [0.45,1.20]; the delay interval is [0.15,0.50]. Histories, inputs (zero), and initial values are known exactly. The four training states generate 16 trajectories; the three held-out states generate 12 trajectories. The time horizon is 1.0, with observations every 0.05.

Matched responses are the right-hand initial slopes. For informative histories, profiles use the original degree-two polynomial estimate on the first five observation times including time zero and a 351-point delay grid. The normalized squared projection residual eliminates the unknown instantaneous term and gain. The true delay is used only for simulation, evaluation, and separately labelled algebraic checks; it is not supplied to the estimator or any fit.

## Geometry and structural abstention

The history gate requires the minimum centered-signature norm over the profile grid to exceed 1e-8. The informative design has analytically proved injective projective geometry and passes this gate independently of the noisy response. Passing this structural gate is not candidate acceptance, an error guarantee, or a deterministic certificate. The degenerate design has four identical constant histories per state and zero centered signature, so all its anchors are rejected. A numerical grid minimum of a flat profile is never presented as a delay estimate.

For informative histories, compare P3 (rollout plus anchors), P4 (rollout only), A1 (anchor-only delay network, 280 updates), A2 (anchor-only delay network, 1,200 updates), linear interpolation, and PCHIP. Each neural method has the identical initial delay subnetwork within a paired run. P3/P4 also share all initial dynamics parameters. Both use the same known negative-gain prior. The architecture widths/depths and optimizer/penalty settings follow Experiment 1, apart from the necessary gain-sign change and the state interval used for smoothness.

For degenerate histories, P3 receives zero anchor weight and is therefore mathematically the same training procedure as P4. Both are run and their saved parameters checked for equality. A1, A2, linear interpolation, and PCHIP record structural abstention and do not fabricate a fitted delay. P3/P4 delay errors in this condition are diagnostic only, with mechanistic eligibility false. The control concerns the matched-initial-state identification mechanism, not every possible use of complete trajectories.

## Simulation and learned rollout

Generate clean reference trajectories by fixed-step method-of-steps RK4 with analytic prescribed histories and piecewise-linear interpolation of computed past values. Every reference step is smaller than the minimum delay, so RK stages never require as-yet-unknown future values. Use the refinement ladder 0.0025, 0.00125, 0.000625, 0.0003125. Stop at the first consecutive pair for which the maximum difference at observation times across **all** train/test trajectories and both history designs is at most 1e-6. Use that pair's finer result for all conditions. If no pair qualifies, abort before neural fitting and report the failed numerical prerequisite. Record every evaluated refinement comparison.

This is an empirical convergence check, not a rigorous uniform solver-error enclosure. The clean-data check may not be used to choose a more favorable learning mechanism or history design.

The learned solver is explicit Euler with internal step 0.025 and two substeps per observation interval. It evaluates prescribed histories analytically for negative query times and differentiably interpolates computed states for positive query times. The same solver is used by P3 and P4. Training uses 90 updates over the first 10 observation steps, followed by 190 updates over all 20 observation steps. This experiment does not itself establish end-to-end training-resolution robustness; that remains the subsequent resolution study.

## Replications, noise, and budgets

Use five base noise seeds 20261201–20261205 and three optimization seed labels 20261211–20261213. These integers are pseudorandom labels, not dates. The initialized model seed is `optimization_seed + 100*data_index`; training resets its RNG with `model_seed + 404` as in Experiment 1.

Gaussian noise standard deviations are 0.0005 and 0.0015, affecting only post-initial observations. Reuse the same standardized noise draws across noise levels and history designs for a given base seed. Thus there are **five independent base-noise replications per condition**, with pairing across conditions; the four condition blocks do not create 20 independent replications. Held-out observation noise uses `data_seed + 1000` and is used solely for evaluation.

Adam uses learning rate 0.002, weight decay 1e-6, clipping threshold 5, float32, and one CPU thread. P3 uses trajectory loss + 0.08 × anchor MSE + 0.0002 × delay smoothness. P4 omits the anchor term. A1/A2 use the same anchor and smoothness penalties with no trajectory loss. Smoothness uses 100 equally spaced states on [0.45,1.20]. There is no profile initialization, test-based model selection, early stopping, hyperparameter search, or adaptive addition of seeds/budgets.

Expected completed accounting: **180 neural fits** (120 informative and 60 degenerate), **20 informative direct functions**, and **80 structural-abstention records** (60 anchor-only and 20 direct in the degenerate condition). Save 180 final checkpoints and all corresponding curves, raw observations, anchors, geometry decisions, settings, hashes, failures, and runtime versions. All learned trajectories are additionally checked for nonpositive predicted values; these are reported as model-domain diagnostics and are not silently removed.

## Evaluation and fixed interpretation

For each noise level separately, evaluate delay RMSE, MAE, maximum absolute error, anchor-fit error, and predicted curves on 201 states in [0.45,1.20]. For P3/P4 also report clean/observed training and held-out trajectory errors. Delay-only methods have no trajectory-error entry. Four observed training anchors do not certify dense-grid functional recovery; the grid metric evaluates interpolation against synthetic truth.

Take each neural method's median RMSE over three optimization repeats **within each dataset**, then compute paired P3/comparator ratios and absolute differences across the five independent base seeds. Direct methods contribute one function per informative dataset, not three independent repetitions. Publish all five paired values, the median and range, and the improvement count. Report P3/P4 and both direct baselines as well as A1/A2.

Retain Experiment 1's chosen descriptive criterion: a median dataset-level ratio at most 0.80 and improvement in at least 4/5 datasets, against **both A1 and A2**, separately at each noise level. A claim of consistent practical joint-fitting advantage across these two noise levels requires that criterion at **both** levels. This is a chosen practical target, not a power calculation, significance test, or submission requirement. Do not lower it after observing results. A failure to meet it narrows the empirical learning claim without changing the structural theorem.

Degenerate controls are evaluated through structural abstention and P3/P4 equality; no deterministic recovery or calibrated-confidence claims are made. Gaussian noise supplies no deterministic uniform profile-error envelope automatically. Keep conditional theoretical certification separate from these empirical results.

## Execution, smoke test, and failure policy

Write a manifest hashing this protocol, the mathematical note, configuration, model/data code, and runner before generating the reference data or fitting a model. Resume requires matching files, software, execution mode, and settings. Existing completed records, including failures, are retained. Interrupted fits may restart with their original seed and log entry; failed fits are not silently replaced.

Nonfinite objectives, gradients, parameters, or predictions are recorded as numerical failures. Do not compute a complete-case practical-advantage verdict with missing required fits. Structural abstention is an expected observation-design outcome, not a numerical failure. Record all denominators explicitly.

Smoke mode uses reserved seeds 900001/900002, one noise level, both history designs, four updates for P3/P4/A1 and eight for A2. It checks execution and accounting, not scientific performance. It is kept separate from the full experiment. Algebraic tests use oracle slopes only to verify the derived formula and profile; they supply no training labels.

After the fixed run, audit saved observations/anchors, initialization hashes, checkpoints, delay curves, trajectory metrics, condition-level summaries, and the practical decision rule. Report all outcomes, including competitive anchor-only/direct estimates and failed prerequisites. Any changed protocol requires a separately recorded extension.
