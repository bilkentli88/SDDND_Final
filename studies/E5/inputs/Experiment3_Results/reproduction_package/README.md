# ASOC Experiment 3: resolution study

Read `ASOC_Experiment3_Protocol.md` before execution. This self-contained CPU package crosses observation spacing and learned Euler step for the informative logistic example at noise SD 0.0005. It uses five new base-noise datasets, three paired initializations, 180 neural fits and 20 direct functions. The run package contains no full-study learning outcomes.

The model note from Experiment 2 is retained under its original filename because the mechanism/history family is unchanged. It gives exact matched-initial-state geometry, not a characterization of the full trajectory observation operator. Gaussian observation noise is not turned into a deterministic certificate.

## Execution

Python 3.12 was used for the delivered run. Use an isolated environment. Install CPU PyTorch 2.10.0 from its official wheel index, then `pip install -r requirements.txt`. The runner uses four independent worker processes with one PyTorch thread each; seeds are task-specific, so process completion order does not affect the results.

```bash
python validate_experiment3.py
python run_experiment3.py --mode smoke --output smoke_results --resume
python audit_experiment3.py --results smoke_results
python run_experiment3.py --mode full --output experiment3_results --resume
python audit_experiment3.py --results experiment3_results
```

The Colab notebook performs these steps and exports a complete results ZIP. Select the run-package ZIP when prompted. Smoke outputs are excluded from the full scientific summaries. Identical-source interrupted jobs can resume; recorded numerical failures are retained and are not silently retried. Do not edit a running study's source or protocol. A changed protocol requires a new output and an explicitly labelled new study.

## Main outputs

- `manifest.json`: actual runtime, source/configuration hashes and a freeze timestamp before reference generation.
- `reference_convergence.json`: prespecified reference refinement at the common observation times.
- `observations/`: fine/common and native observations, exact history metadata, estimated slopes, anchors and profiles. Shared times have identical noise.
- `checkpoints/`, `curves/`, `events.jsonl`: every completed model, delay function and submitted training attempt.
- `run_records.json`, `runs.csv`: all outcomes, including failures.
- `dataset_summaries.csv`: within-dataset medians over optimization repeats.
- `dataset_pairs.csv`: all five dataset-level paired ratios/differences for every prespecified contrast.
- `summary.json`: complete-cell numerical contrasts and secondary practical comparisons.
- `independent_audit.json`: raw-data reconstruction, checkpoint replay, independently recomputed summaries and checkpoint hashes.

`native_clean_test_mse` uses the training solver step on the common observation grid 0.025. `refined_clean_test_mse` uses Euler step 0.00625 on that same grid. The other `eval_h...` columns retain the intermediate step. All are frozen-checkpoint evaluations, with no parameter update or selection. Domain counts concern predicted observation values, not every internal/continuous-time value. A1/A2 and direct interpolants are delay functions without a complete dynamics model, so trajectory scores are absent.

## Interpretation

This finite two-level training study measures numerical sensitivity, not asymptotic convergence. Sampling refinement adds observations and changes slope estimates over a fixed physical window; it is not a fixed-observation-budget study. Five base-noise datasets are the independent replications. The 15 optimization repeats within a condition, and the paired resolution cells, must not be counted as independent noise datasets. Report actual unfavorable ratios and the prior experiments alongside these results. No general neural-superiority, uncertainty-calibration or submission-readiness claim is established.

Architecture construction is adapted from NSDDNew by Aykut T. Altay, MIT, pinned in earlier work at commit ee1014a80760dd720e680c7597a3b985f0f4b609. The retained license applies to that code. No external repository checkout is required to run this package. No external neural method is claimed to have been reproduced here.
