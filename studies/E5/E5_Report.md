# E5 complete two-stage forecasting control

All 210 new field fits completed with zero failures. All 210 saved complete
predictors passed checkpoint replay, exact frozen-delay and score checks.
Another 210 archived P3/P4 checkpoints were replayed before fitting.

T1 completes A1 (280 delay updates); T2 completes A2 (1,200 delay updates).
Both add 280 field updates using the same observations and original paired
field initialization. The delay is frozen; its state derivative is retained.
The controls receive extra total computation compared with P3/P4.

## Paired forecast results

Ratios below one favor P3. Parentheses count P3 wins out of five datasets.
Each dataset score first takes the median of its three optimizer repeats.

| Condition | P3/T1 | P3/T2 |
|---|---:|---:|
| E1, SD 0.0005 | 1.0383 (2/5) | 1.0307 (2/5) |
| E2, SD 0.0005 | 0.9676 (4/5) | 0.9825 (3/5) |
| E2, SD 0.0015 | 0.9446 (5/5) | 0.7503 (5/5) |
| E3, obs. 0.05, h 0.025 | 0.9553 (5/5) | 1.0136 (2/5) |
| E3, obs. 0.05, h 0.0125 | 0.9629 (4/5) | 1.0170 (2/5) |
| E3, obs. 0.025, h 0.025 | 0.9575 (4/5) | 1.0096 (2/5) |
| E3, obs. 0.025, h 0.0125 | 0.9591 (4/5) | 1.0080 (2/5) |

P3 has a lower median paired forecast error than T1 in six conditions and
than T2 in two. E1 and all four E3 conditions favor T2 by this statistic.
The largest median improvement for P3 is against T2 in E2 at SD 0.0015:
ratio 0.7503, with five of five wins. These condition counts are descriptive;
paired noise/resolution cells are not independent evidence. No method was
chosen using the test results. P4 has lower median paired forecast error than
both controls in all four E3 cells, while its delay recovery is worse.

T1/T2 delays are exactly A1/A2, so there is no new independent delay-recovery
evidence. The original joint delay-advantage target remains unmet. The study
does not establish that joint fitting is necessary or uniformly superior.

## Scope and reproducibility

This is an extension after the original outcomes were inspected. All data,
anchors and stage-one weights are reused; it is not independent confirmation
or public preregistration. The new protocol and scientific training source
were frozen before the new field fits. E2/E3 retain the archived numerical
source unchanged. E1 uses a documented discrete-history Euler adapter because
the original numerical source was unavailable in the recovered packages.
Across 30 E1 archived controls its maximum relative forecast-score difference
is 7.1166e-6, with exact delay RMSE. E2/E3 scores match exactly. E5 uses common
harness scores for all methods. This is a consistency audit, not independent
replication or proof of exact E1 source parity.

Degenerate E2 histories have no eligible anchor estimate and keep their earlier
abstention result. E4 is not extended. Five base datasets per original study,
unequal total computation, synthetic mechanisms, a fixed training curriculum,
the E1 adapter and reused data limit conclusions. No calibrated uncertainty or
equal-compute superiority claim is made.

`dataset_summaries.csv`, `dataset_pairs.csv`, `absolute_summary.csv` and
`paired_summary.csv` retain the full arithmetic, including P4 comparisons.
`run_records.json` retains per-fit metrics and timing. `audit.json` retains
all checkpoint checks; `manifest.json` records the pre-fit source freeze.
