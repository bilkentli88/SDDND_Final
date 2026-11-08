# Identifiability-Aware Learning of State-Dependent Delays in Neural Differential Equations

Code, synthetic observations, trained checkpoints and numerical records for
the accompanying main article and supplement. This repository contains the
scalar and oscillator studies (S0), extensions E1--E5, and both sensitivity
studies in Appendix H. Their code and archived results are included in the
repository.

## Start here

```bash
python scripts/reproduce.py verify
```

This standard-library check verifies the supplied file hashes, frozen numerical
sources, record counts and checkpoints, and recomputes the principal paired
summaries from retained records. It does not retrain or load neural weights.
For checkpoint/trajectory replay and training commands, read
[the reproduction guide](docs/reproduction.md).

## Studies and manuscript results

| Study | Material | Location | Main manuscript connection |
|---|---|---|---|
| S0 | Scalar mechanism ablation, fresh-noise comparison, profile diagnostics, noise/domain/oscillator checks | Existing `experiments/`, `results/`, `checkpoints/` | Supplement G; main Sections 7.1, 7.2 and 7.6 |
| E1 | First scalar mechanism; rollout, joint, anchor-only and interpolation controls | `studies/E1/` | Main Table 2; Supplement E.2 and Table E.2 |
| E2 | Delayed logistic mechanism, paired noise levels and degenerate histories | `studies/E2/` | Main Table 2; Supplement E.3 and Table E.3 |
| E3 | Crossed observation and training-solver resolutions | `studies/E3/` | Main Table 2; Supplement E.4 and Tables E.4--E.5 |
| E4 | Adapted constant-delay architecture comparison | `studies/E4/` | Main Table 5 and Figure 3; Supplement F |
| E5 | Complete frozen-delay two-stage predictors on informative E1--E3 cases | `studies/E5/` | Main Table 4 and Figure 2; Supplement E.5 and Table E.6 |
| Sensitivity | Noise and sampling contrasts for neural fits and separate response-estimator comparisons | `studies/sensitivity/` | Main Section 7.3 and Table 3; Supplement H |

E1--E5 retain 60, 180, 180, 184 and 210 trained scientific checkpoints,
respectively (814 total), in addition to the earlier studies and separately
labelled smoke checks. E4 includes 64 calibration fits and 120 final fits.
E5 includes 210 newly fitted field models and the retained E1--E3 inputs;
it does not add independent datasets.

The [Appendix H studies](studies/sensitivity/) report 240 neural fits on ten
paired base-noise datasets and profile-only diagnostics on 40 additional
base-noise datasets. Main Section 7.3, Table 3 and Appendix H report the
sensitivity results.

## Reproduction levels

1. **File and record verification:** `python scripts/reproduce.py verify` needs only
   Python 3.12+ and preserves the supplied records.
2. **Checkpoint and trajectory replay:** `python scripts/reproduce.py replay ...`
   runs the original audits in a generated working copy. Follow the study's
   recorded environment and dependency instructions in the guide.
3. **Numerical diagnostics:** `python scripts/reproduce.py diagnose ...` evaluates
   the retained checkpoints and reports differences from the archived scores.
   It leaves the original strict audit criteria unchanged.
4. **Training:** `python scripts/reproduce.py smoke ...` performs the original short
   checks; `python scripts/reproduce.py train ...` runs the fixed scientific protocols
   into a new generated directory.

The integration check verified the supplied files and saved summaries. Strict
replay did not pass the original exact/tight numerical comparisons on the
current CPU runtime. Diagnostics completed for all 814 E1--E5 checkpoints and
210 reused controls; the paired ratios in main Tables 2, 4 and 5 retain their printed
values and win counts. See [the quantified results](docs/verification.md)
for the scope and one rounding difference in an additional E5 comparison.

The numerical implementations and protocols that generated the retained
results are unchanged. E1 checks the original NSDDNew commit; the dependency
setup command obtains that historical checkout. E4 checks pinned third-party
solver/architecture sources, obtained by the same setup tool and kept outside
tracked research records. External software retains its own license.

The original S0 workflow remains available:

```bash
python -m pip install -r requirements-analysis.txt
python scripts/reproduce_all.py --mode verify
```

See [the full material map](docs/experiment_mapping.md),
[reproduction instructions](docs/reproduction.md), and
[recorded integration checks](docs/verification.md).
Study manifests record source hashes, seeds and execution environments.
Generated outputs go to `generated/reproduction/`; archived weights and observations
are retained in their study directories.

For updating GitHub and the manuscript's availability statement, follow
[the publication checklist](docs/repository_publication.md).

## Interpretation

The positive theorem concerns a scalar affine delayed-state channel and a fixed
matched-slope inverse problem. Forecast accuracy and delay recovery are
reported separately. Profile resampling is an empirical sensitivity diagnostic;
deterministic localization requires the stated uniform error envelope.
Independent units are base-noise datasets, with optimizer repeats nested within
them. E4 uses adapted constant-delay pipelines. T1/T2 in E5 receive additional
stage-one updates; they preserve the original A1/A2 delays.

## Citation and license

The accompanying manuscript is **Identifiability-Aware Learning of
State-Dependent Delays in Neural Differential Equations**, by Aykut Tayyip
Altay. Bibliographic metadata are in [CITATION.cff](CITATION.cff).
The project license and reused-code notices are retained. Cite the final
published article when publication details become available.
