# Appendix H sensitivity studies

These studies support Section 7.3 and Table 3 of the main article and
Appendix H of the supplement.

| Analysis | Reproduction instructions | Archived results |
|---|---|---|
| H.1. Paired P3/P4 neural fits on ten base-noise datasets | [Neural study guide](neural/reproduction/README.md) | [Neural results](neural/results/) |
| H.2. Profile-only response estimation on 40 separate base-noise datasets | Instructions below | [Profile results](results/) |

From the repository root, run `python scripts/reproduce.py verify` to check the
file and source hashes, fit counts and numerical summaries from the archived
records.

## H.2. Profile-only response estimation

The protocol crosses two observation noise levels, two sampling intervals
and three response estimators on the synthetic E2/E3 logistic mechanism.
It uses the E2/E3 protocols and E3 data module already in the repository.
The run manifest records their exact SHA-256 hashes.

The recorded environment is Python 3.12.14 and NumPy 2.3.5. In a clean
Python 3.12 environment, run from the repository root:

```bash
python -m pip install numpy==2.3.5
python studies/sensitivity/run_profile_pilot.py --output generated/reproduction/appendix_h/profile
```

Choose a fresh output directory. The outputs include a reference-solver
refinement check, per-state and per-dataset CSVs, and summaries for Tables
H.3 and H.4. Each base-noise dataset is shared across settings and contributes
four matched states. The 2,000 resampling draws per state characterize the
specified measurement noise. The intervals omit response-estimation bias and
carry no coverage guarantee.
