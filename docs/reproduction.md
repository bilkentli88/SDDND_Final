# Reproducing the experiments

Run repository-level commands from the repository root. Scientific source,
protocols and archived numerical records are retained in `studies/E1` through
`studies/E5` and `studies/sensitivity`. New outputs go to `generated/reproduction/`.
All experiment materials are available in this repository.

## File and numerical-record verification

```bash
python scripts/reproduce.py verify
```

This needs only Python's standard library. It checks all delivered file hashes,
source/protocol fingerprints, scientific record and checkpoint counts, and the
main paired summaries. E2's 80 structural-abstention records are counted
separately from numerical failures. The check neither imports PyTorch nor
replays trajectories. File identity and numerical checkpoint replay are
different verification levels.

The original replay reports are retained alongside the experiment records.
Fresh integration/replay checks for this distribution are described in
`verification.md`.

## Appendix H sensitivity analyses

The [Appendix H study guide](../studies/sensitivity/README.md) gives the
commands, recorded environments and result files for the paired neural study
and the profile-only response-estimation study. The neural smoke run uses a
small separate dataset. The profile study uses NumPy and the exact E2/E3 files
already included in this repository.

## Environments for E1--E5

The numerical runs used Python 3.12.14 and CPU PyTorch 2.10.0+cpu. E1--E3 used
NumPy 2.4.2, pandas 2.2.3 and SciPy 1.17.0. E4--E5 used NumPy 2.3.5; E4 also
used jaxtyping 0.3.11. Use separate environments for the two recorded NumPy
versions when running the strict replay audits.

For E1--E3, create and activate a Python 3.12 environment, then:

```bash
python -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-e123.txt
```

For E4--E5, use a second Python 3.12 environment and:

```bash
python -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-e45.txt
```

The root `requirements-analysis.txt`, `requirements.txt` and `environment.yml`
are the original S0 environment specifications. They remain available for the
earlier workflow. Pinning versions records the tested environments; different
hardware, libraries or math backends can still change floating-point outcomes.

## Historical and third-party source dependencies

E1 checks the original NSDDNew commit used to generate its archived data. E4 checks the original solver and
architecture source commits. Obtain those clean checkouts once:

```bash
python scripts/setup_dependencies.py E1 E4
```

This writes ignored dependency checkouts to `.repro-deps/`:

| Dependency | Commit | Use |
|---|---|---|
| NSDDNew | `ee1014a80760dd720e680c7597a3b985f0f4b609` | E1 historical source/provenance check |
| onofriosmr/torchdde | `4888c63dc84e966f6c6553277adc3851a2cca4fc` | E4 Euler/direct-backpropagation solver |
| onofriosmr/learnable_delays | `9b573c4bed5efee9fe9c6b765044dbd506b5bf07` | E4 architecture-source provenance |

Git and an internet connection are needed for this setup. The dependency tool
checks the commit and rejects changed tracked files. It never changes the
working research records. External projects retain their own licenses. The
wrapper puts the pinned torchdde checkout on its subprocess import path; a
separate editable install is unnecessary when using `scripts/reproduce.py`.

Existing checkouts can be selected with `--e1-repo`, `--solver` and
`--architecture` on the wrapper. The original standalone E4 commands remain
in its frozen protocol/source; they expect torchdde to be installed/importable.

## Checkpoint and trajectory replay

In the E1--E3 environment:

```bash
python scripts/reproduce.py replay --studies E1 E2 E3
```

In the E4--E5 environment:

```bash
python scripts/reproduce.py replay --studies E4 E5
```

Each command copies the archived result directory to
`generated/reproduction/replay/E1`, etc., and runs the archived audit. For
E1--E3, the launcher restores the frozen runners and their historical package
paths in a temporary directory. This keeps the original source checks and
archived weights, observations and summaries intact. Existing output
directories are refused; use a new `--output` parent for another replay, e.g.:

```bash
python scripts/reproduce.py replay --studies E5 --output generated/reproduction/replay_again
```

The original scripts perform the following consistency checks:

| Study | Replay scope |
|---|---|
| E1 | 60 delay checkpoints, 30 full-model trajectory metrics, ten direct functions, raw observations/anchors and paired summaries |
| E2 | 180 checkpoints, 120 trajectory metric sets, observations/anchors, degenerate controls and summaries |
| E3 | 180 checkpoints, 360 trajectory metric sets, shared-noise/resolution pairing and summaries |
| E4 | 184 calibration/final checkpoints, source/runtime hashes, data, locked validation selection and paired outcomes |
| E5 | 210 complete frozen-delay checkpoints, equality with their stage-one delays, forecast/delay scores and paired summaries |

These are consistency replays of the retained studies, not independent data
replications. They do not rerun all training.

The integration check on the current CPU runtime did not meet the original
audits' exact/tight numerical criteria, despite matching the recorded package
versions and source hashes. The archived successful audit reports describe
their original executions; they are not fresh pass reports for this runtime.
See [the measured differences](verification.md).

## Numerical diagnostics across runtimes

To evaluate every retained checkpoint and measure differences without changing
the original audits, use the corresponding environment:

```bash
python scripts/reproduce.py diagnose --studies E1 E2 E3
python scripts/reproduce.py diagnose --studies E4 E5
```

This writes `diagnostic.json` and `replayed_records.json` to a new
`generated/reproduction/diagnose/E1`, etc. The report gives maximum absolute and relative
differences for each evaluated metric. E5 includes the 210 reused P3/P4 controls
in addition to its 210 new field checkpoints and checks exact equality of the
frozen stage-one delay tensors. Diagnostics assign no tolerance-based pass
verdict and do not retrain. Original numerical records remain the source for
the manuscript tables.

## Smoke checks and full training

Select studies appropriate to the active environment:

```bash
python scripts/reproduce.py smoke --studies E1 E2 E3
python scripts/reproduce.py smoke --studies E4 E5
python scripts/reproduce.py train --studies E1 E2 E3
python scripts/reproduce.py train --studies E4 E5
```

Use `--dry-run` to see commands first. New outputs go under
`generated/reproduction/smoke/` or `generated/reproduction/train/`. Output parents can be
changed with `--output`; archived result directories cannot be selected.
The wrapper refuses existing study output directories. For an interrupted
training run, use the original script command printed by `--dry-run` and its
original resume mechanism, which checks source/configuration/runtime identity.

For E4, smoke/full execution runs calibration before the final stage, retaining
validation-based configuration selection. For E5, `smoke` runs the original
control replays and four-update implementation checks; `train` follows them
with all 210 fixed-delay field fits. E5 defaults to eight one-thread workers;
`--workers` changes only its new training execution setting and manifest.

Smoke outputs are execution checks, separate from scientific study outcomes.
All original failure-accounting and abstention logic remains in the unchanged
numerical runners.

## Figures and original S0 checks

The main paper's vector figure assets are in `figures/experiment_assets/reference/`.
Regenerate the E4/E5 paired-comparison plots from the retained per-dataset
records using:

```bash
python scripts/build_figures.py
```

These regenerated plots use the same numerical values and aggregation; PDF
bytes/fonts may differ across Matplotlib versions. The conceptual diagram is
a supplied vector illustration, not a plot estimated from data. Earlier S0
figure/table generation remains available through:

```bash
python scripts/reproduce_all.py --mode verify
```
