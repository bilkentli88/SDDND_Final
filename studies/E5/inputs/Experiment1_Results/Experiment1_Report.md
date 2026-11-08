# ASOC Experiment 1: completed ablation and revision decision

Prepared 19 October 2026. All 60 neural fits and 10 direct-baseline functions completed with no recorded numerical failures. This is a new fixed-protocol experiment, separate from the earlier exploratory interpolation pilot. It remains a controlled synthetic study of one mechanism.

## Main finding

Profile anchors strongly improve rollout-only learning on these fresh noise realizations. The evidence for an additional delay-recovery advantage from joint trajectory fitting over an anchor-only delay network is **modest and mixed across summaries**. The complete prespecified practical-advantage criterion was **not met**.

P3 improves delay RMSE against the matched-budget anchor-only network A1 in four of five datasets. Its median dataset-level RMSE ratio is 0.862, a paired reduction of 13.8%. That is below the chosen 20% practical threshold. P3 meets the descriptive criterion against the longer-budget A2 control, but the protocol required it against **both** controls. This decision has not been changed after observing the outcomes.

## Protocol and execution

The protocol, configuration, and runner were hashed before the first full-experiment fit. The execution manifest records the freeze time as `2026-10-19T19:21:57.881920+00:00`. Their hashes still match the delivered sources. This was an internal prospective protocol, not an external preregistration.

Five new independent observation-noise realizations (seeds 20261020–20261024) were used, with three paired model initializations per dataset. These are five noise datasets, not 15 independent datasets. All methods reuse the same training observations or anchors extracted from those observations. The clean histories and scalar mechanism are the original controlled design; no new application domain or physical mechanism is added by this experiment.

P3 is the rollout-plus-anchor learner; P4 is its rollout-only control. A1 is the identical delay subnetwork trained on anchors for 280 updates; A2 uses 1,200 updates. P3/P4 retain the original 280-update staged rollout schedule. Linear and PCHIP interpolate the four anchors. All methods are evaluated on the same 201-point state grid. A1/A2 and the interpolants are delay-only functions; no complete trajectory-model comparison is implied for them.

There were no clipping events in any fit. The penalty scaling, optimizer, final-iterate evaluation, and lack of test-based tuning follow the supplied protocol. The new P3/P4 implementation also matched the pinned original's parameters bit for bit on an old-seed, reduced-budget parity check before the fresh run.

## Dataset-level results

All entries are delay RMSE. Neural entries are medians over the three optimization repeats within that dataset. Direct methods contribute one function per dataset.

| Noise seed | P3 | P4 | A1 | A2 | Linear | PCHIP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 20261020 | 0.007886 | 0.027774 | 0.011302 | 0.010771 | 0.010031 | 0.010221 |
| 20261021 | 0.006549 | 0.028278 | 0.005273 | 0.005965 | 0.005280 | 0.006234 |
| 20261022 | 0.007728 | 0.027562 | 0.008963 | 0.009801 | 0.010536 | 0.010513 |
| 20261023 | 0.005439 | 0.027292 | 0.006026 | 0.006381 | 0.006767 | 0.006532 |
| 20261024 | 0.002791 | 0.021611 | 0.005331 | 0.013722 | 0.010347 | 0.013237 |
| Median across five datasets | 0.006549 | 0.027562 | 0.006026 | 0.009801 | 0.010031 | 0.010221 |

The median across datasets of A1's within-dataset error is lower than P3's corresponding median (0.006026 versus 0.006549). At the same time, P3 improves in four of five paired dataset comparisons. These statements use different summaries: a ratio of separate medians is not the median of paired ratios. Both are reported to avoid implying overall superiority from a favorable choice of summary.

## Fixed comparisons

Each ratio below compares P3's within-dataset median with the corresponding comparator value, followed by the median of the five ratios. Values below one favor P3.

| Comparator | Median dataset-level P3/comparator ratio | P3 improves | Fixed practical criterion |
| --- | ---: | ---: | --- |
| A1 | 0.862 | 4/5 | Not met |
| A2 | 0.789 | 4/5 | Met |
| P4 | 0.232 | 5/5 | Met |
| Linear | 0.786 | 4/5 | Met |
| PCHIP | 0.772 | 4/5 | Met |

The fixed descriptive rule requires a median ratio at most 0.80 and improvement in at least four of five datasets, against **both A1 and A2**. Because A1 does not meet the 0.80 threshold, the joint criterion is false. The rule is a chosen practical effect-size criterion, not a statistical significance test or proof of population-level superiority.

P3 improves on P4 in all five datasets. For clean held-out trajectory MSE, P3's within-dataset median improves on P4 in four of five datasets; these values are in `runs.csv`. This does not supply trajectory-error comparisons for the delay-only baselines.

## Consequence for the article

Keep the R1 introduction's distinction between prediction and mechanistic delay recovery, and keep its explicit scalar/fixed-matched-slope scope. The new experiment supports emphasizing **the usefulness of the profile-anchor objective within the tested structured learner**. It does not support a broad claim that joint neural fitting is uniformly better than learning directly from estimated anchors.

In the next manuscript revision, include the full A1/A2 comparison alongside the direct baselines and state that the preselected practical criterion was not met. The theoretical identifiability results do not depend on P3 winning this ablation. The empirical learning contribution should be described as conditional and design-specific.

The next priorities are a second admissible scalar mechanism/history family and an end-to-end observation/solver-resolution study with regenerated data and retraining. Preserve this completed experiment as it stands; any extension needs a new recorded protocol. ASOC fit still depends on the final methodological contribution and experimental evidence, not target-journal citation counts. No submission-readiness conclusion follows from this one new experiment.

## Verification and files

The saved checkpoints replayed all 60 neural delay curves and all 30 P3/P4 trajectory-metric sets. The audit also reconstructed all five saved noisy datasets and their anchors, replayed the 10 direct functions, checked 15 paired initialization hashes, recomputed all 70 delay-RMSE values, and independently checked the five comparator summaries and decision rule. Curve CSVs preserve the neural functions through exact float32 round-trip; checkpoints permit full-precision replay.

The runtime used Python 3.12.14, PyTorch 2.10.0+cpu, NumPy 2.4.2, pandas 2.2.3, and SciPy 1.17.0, on one CPU thread. This replay audit checks consistency; it is not independent scientific replication.

- `ASOC_Experiment1_Results.zip`: complete run records, checkpoints, observed arrays, anchors, curves, protocol, code, manifests, console log, parity checks, smoke outputs, and audit.
- `ASOC_Experiment1_Run_Package.zip`: executable code, fixed protocol, validation materials, and the Colab notebook, with no full-experiment outcomes embedded in the run package.
- `ASOC_Experiment1_Colab.ipynb`: standalone notebook for reproducing the experiment.
- `ASOC_Experiment1_Protocol.md`: prospective protocol retained unchanged after execution.
- `ASOC_Experiment1_Report.md`: this interpretation and accounting record.

The original public repository is pinned to `ee1014a80760dd720e680c7597a3b985f0f4b609`. New code/results are supplied in these archives and have not been published to that repository.
