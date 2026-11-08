# E1: completed experiment materials

This directory contains the completed study of the first scalar mechanism and anchor-only/interpolation controls.
`experiment1_results/` contains the scientific observations, checkpoints, curves, run records,
summaries and original replay audit. `smoke_results/` contains separate execution
checks. `reproduction_package/` contains the unchanged numerical source and
frozen protocol, together with the original notebook utilities.

Use the repository-level commands in [the reproduction guide](../../docs/reproduction.md).
From the repository root:

```bash
python scripts/reproduce.py verify --studies E1
python scripts/reproduce.py replay --studies E1
python scripts/reproduce.py smoke --studies E1
python scripts/reproduce.py train --studies E1
```

Replay and training require the dependencies described in the guide. Generated
outputs are separate from these retained records. The original notebook utilities
describe the historical standalone package workflow; the repository-level runner
is the entry point for this consolidated distribution.
