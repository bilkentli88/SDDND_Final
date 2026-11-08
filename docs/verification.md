# Integration verification

## Files, sources and saved summaries

The distribution preserves the earlier S0 numerical files and all E1--E5
scientific sources, frozen protocols, observations, trained checkpoints and
recorded outcomes. README files and repository-level utilities were updated.
The unused historical manuscript archive in E5 was excluded; its name and
hash are recorded in `material_sources.json`.

`python scripts/reproduce.py verify` checks the delivered checksum manifest, frozen
source/protocol hashes, checkpoint and record counts, and the principal paired
summaries. Those checks passed. The original S0 record/summary verification and
regeneration of the E4/E5 paired plots also completed. Smoke/full training
commands were checked in dry-run mode; a new full training campaign was not run.

## Strict replay and checkpoint diagnostics

The unchanged strict audits were attempted for E1--E5. Each stopped at a
numerical comparison in the current environment. These audits include exact
float32 delay-curve comparisons or trajectory-score tolerances tighter than
the differences observed below. They are **not reported as passing**.
The original successful audit reports are retained as historical records.

Diagnostics then evaluated all 814 scientific checkpoints: 60 in E1, 180 in
E2, 180 in E3, 184 calibration/final fits in E4, and 210 new field fits in E5.
They also evaluated E5's 210 reused P3/P4 controls, for 1,024 checkpoint
evaluations in total. All diagnostic evaluations completed. E5's frozen delay
tensors matched their stage-one inputs exactly. No original weights, results
or audit thresholds were altered.

The runtime used Python 3.12.14, CPU PyTorch 2.10.0+cpu and one Torch thread
with AVX512 CPU capability. NumPy was 2.4.2 for E1--E3 and 2.3.5 for E4--E5.
E4 used the pinned third-party sources specified in the reproduction guide.

| Study | Largest absolute delay-score difference | Largest absolute trajectory-MSE difference |
|---|---:|---:|
| E1 | 1.36e-9 RMSE | 2.22e-10 |
| E2 | 2.92e-9 RMSE | 1.91e-9 |
| E3 | 2.02e-9 RMSE | 1.33e-9 |
| E4 | 0 scalar-delay error | 1.64e-9 |
| E5, including reused controls | 2.92e-9 RMSE | 1.33e-9 |

Pointwise delay-curve differences in E1--E3 were at most 8.95e-8.
The per-metric counts and maximum relative differences are retained in
[`verification/`](verification/). These diagnostics measure drift; they do
not assign a new numerical pass threshold or establish independent replication.

## Paired summaries at printed precision

The archived and newly evaluated scores were compared using the same
within-dataset optimizer medians and paired dataset ratios. The check covered
85 comparisons: E1--E3 delay comparisons, E4 architecture/noise outcome pairs,
and every E5 paired-summary row. All 85 retained their win counts; 84 retained
their ratios at the checked printed precision.

The paired ratios in main Tables 2, 4 and 5 in V15 retained their printed values. The one
rounding change concerns an additional E5 comparison: P4/T2 forecast MSE in
E2 at noise SD 0.0005 changed from 1.0268510542 to 1.0268343032 (1.0269 versus
1.0268 at four decimal places), with two wins among five datasets in both
cases. The largest absolute paired-ratio difference across the 85 comparisons
was 1.76e-5. Archived manuscript values are preserved.

Full comparison details are in
[`verification/paired_summary_comparison.json`](verification/paired_summary_comparison.json).

## Appendix H record verification

The repository check also verifies the frozen source fingerprints for both
sensitivity studies and the 240 neural fit and checkpoint records. It
recomputes the eight paired median ratios and win counts from the per-fit
CSV and checks profile medians and interval counts across the 12 settings
from the dataset and state records. This verification uses the archived
neural fits and does not retrain them.
