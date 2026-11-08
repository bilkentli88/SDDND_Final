# E4: completed constant-delay architecture comparison

`results/` contains 64 calibration fits, the locked validation selection, 120
final fits, observations, all 184 scientific checkpoints and the 4,988-check
original replay audit. `source/` contains the unchanged numerical implementation
and frozen protocol. `reserved_smoke/` contains separate execution checks.

From the repository root, follow the environment instructions in
[the reproduction guide](../../docs/reproduction.md), then run:

```bash
python scripts/setup_dependencies.py E4
python scripts/reproduce.py replay --studies E4
```

The dependency tool obtains the exact solver and architecture source commits.
These third-party projects keep their original licenses and are not bundled.
The comparison uses common Euler/direct-backpropagation settings and adapted
constant-delay architectures; the original protocol documents those choices.
Archived source and provenance records are preserved without changing the
scientific settings or numerical results.
