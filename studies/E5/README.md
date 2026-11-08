# E5: complete frozen-delay predictors

`source/` contains the frozen two-stage implementation and its protocol;
`results/` contains all 210 complete trained predictors, 210 replayed P3/P4
controls, curves, per-fit records, paired summaries and the original audit.
`inputs/` retains the E1--E3 input archives and their extracted records.
Those repeated inputs preserve the source layout used by the numerical audit.

From the repository root, follow [the reproduction guide](../../docs/reproduction.md):

```bash
python scripts/reproduce.py verify --studies E5
python scripts/reproduce.py replay --studies E5
python scripts/reproduce.py smoke --studies E5
python scripts/reproduce.py train --studies E5
```

Replay operates on a generated results copy. T1/T2 preserve the A1/A2 delays;
field training uses the existing E1--E3 observations and seeds. Total update
budgets are 560/1,480 versus 280 for P3/P4. The E1 evaluation adapter and its
measured discrepancy are documented in the unchanged protocol.
The original source hashes and numerical inputs are retained. A historical
manuscript ZIP, unused by every runner/audit, is excluded from this distribution.
