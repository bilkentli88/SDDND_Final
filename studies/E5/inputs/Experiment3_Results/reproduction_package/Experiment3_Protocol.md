# Experiment 3: observation and training-solver resolution

Protocol ASOC-E3-v1.0. This is an internal prospective protocol, frozen by source hashes before reference generation and full-study fitting. It is not a public preregistration. It follows the mixed joint-versus-anchor-only results of Experiments 1 and 2 and must not replace them.

## Scope and fixed factorial design

Use the synthetic state-dependent delayed logistic mechanism, positive trigonometric histories, unknown neural coefficient fields and known negative-gain prior from Experiment 2. The co-shipped `ASOC_Experiment2_Model_Check.md` gives the algebra and its limitations. Only informative histories and noise SD 0.0005 are included. This study does not cover the higher noise level, degenerate histories, the original scalar mechanism, or measured data.

Cross observation spacing 0.05/0.025 with learned Euler step 0.025/0.0125. Both solver steps divide both observation spacings. Reference trajectories are generated once on the fine observation grid and downsampled exactly. Generate noise once on the fine grid for each split and base seed, then downsample it: shared observation times have exactly the same noise. Initial states and prescribed histories are exact. The finer observation condition contains more observations; this is not a fixed-data-budget comparison.

Use five new independent base-noise seeds 20261301–20261305 and three optimization seeds 20261311–20261313, with model seed `optimization_seed + 100*dataset_index`. Seed integers are identifiers, not dates. Pair initial parameters across all methods and resolution cells within a dataset/optimization repeat. Held-out noise seed is base seed +1000. Conditions and optimizer repeats are not independent datasets.

## Anchors and training

Re-estimate the right-hand slopes and profile anchors separately at each observation resolution. Keep the physical polynomial window [0,0.20] and degree two fixed: five samples at spacing 0.05 and nine at 0.025. Thus added observations change sampling density, not window duration. Candidate-delay grid: 351 points on [0.15,0.50]. Four observed anchors do not certify an entire state interval. No true delay enters anchor estimation, fitting or model selection.

P3: rollout plus anchors. P4: rollout only. Fit each in all four resolution cells. A1: the same initialized delay branch, anchor-only for 280 updates. A2: anchor-only for 1200 updates. Fit A1/A2 once per observation resolution, dataset and initialization; they have no rollout solver and must not be counted four times. Linear/PCHIP each use the same four anchors once per observation resolution/dataset.

Physical training phases remain T=0.5 for 90 updates, T=1 for 190 updates. Mean observation loss has weights linearly spanning 0.6 to 1.4; denser observations mean more terms. Architecture, Adam learning rate 0.002, weight decay 1e-6, anchor weight 0.08, smoothness weight 2e-4, clipping at 5 and final-iterate selection remain fixed. This is a fixed optimization-budget study, not proof that all optimization problems converge.

Full accounting: 120 P3/P4 fits, 60 A1/A2 fits, 20 direct functions; 180 actual neural fits, 200 records, 10 observation/anchor sets. Four CPU worker processes, each with one PyTorch thread, execute independent fits. Completion order is irrelevant to seeds and paired summaries.

## Numerical prerequisite and evaluation

Use float64 method-of-steps RK4 with analytic negative-time histories and linear interpolation of computed past values. Reference step ladder: 0.0025, 0.00125, 0.000625, 0.0003125. Select the first consecutive pair with maximum observation-time difference <=1e-6 across all 28 train/test trajectories on the fine grid; abort before fitting otherwise. This empirical check is not a rigorous error enclosure and linear past interpolation prevents claiming global fourth-order convergence.

Delay RMSE is measured on the fixed 201-state grid [0.45,1.20]. All trained full models are evaluated on the common observation grid 0.025, using each frozen checkpoint at Euler steps 0.025, 0.0125 and 0.00625. Record train/test clean and noisy MSE, minimum predicted observation state and nonpositive observation counts at every evaluation step. Evaluations do not update parameters or select checkpoints. Native-step and common refined-step (0.00625) errors are reported separately. Exact true coefficients and delay are used only for reference generation, labelled validation fixtures and post-fit evaluation.

## Prespecified summaries and interpretation

The replication unit is the base-noise dataset. First take the median over three optimization repeats within each cell/dataset, then form paired dataset ratios; report all five ratios, their median/range and number below one. Do not substitute ratios of marginal medians for median paired ratios.

Primary numerical contrasts: fine/coarse solver at each fixed observation spacing for P3/P4; fine/coarse observations at each fixed solver step for P3/P4; observation contrasts for A1/A2/Linear/PCHIP. Report delay RMSE and common refined-step clean held-out trajectory MSE where a full dynamics model exists. Also report anchor RMSE and paired P3/P4 and P3/A1/A2 comparisons in every cell. A1/A2 supply delay functions only, so they have no trajectory comparison.

For continuity only, retain the previous descriptive practical comparison: median paired P3/comparator delay-RMSE ratio <=0.80 and >=4/5 dataset improvements; a joint-advantage label requires this against both A1 and A2. This is secondary to numerical sensitivity, is not significance or a journal requirement, and cannot establish general superiority. No automatic numerical-convergence or robustness verdict is defined: report the actual sensitivity. This two-level training study and three-level frozen evaluation are not asymptotic convergence proofs.

Retain every numerical failure and unfavorable outcome. Missing or failed fits withhold complete-cell comparisons; do not rerun failed seeds, tune hyperparameters, change stopping rules or exclude outliers after observing full results. Resume may only continue unfinished jobs under identical hashed sources/settings. Record all attempts. Smoke seeds 910001/910002 and reduced 4/8-update budgets are separate and excluded from scientific summaries.

The protocol, configuration, model note and all scientific execution modules are hashed before reference generation. The later replay audit reconstructs raw observations/anchors, verifies initializations and checkpoint metrics, and recomputes summaries independently of the training loop. It is consistency validation, not independent scientific replication.
