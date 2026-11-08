# Neural sensitivity experiment

Manuscript Identifiability-Aware Learning of State-Dependent Delays in Neural Differential Equations.

## Design

The run crosses observation noise SD 0.0005 and 0.0015 with observation spacings 0.050 and 0.025 in the submitted E2/E3 logistic mechanism. The training Euler step is 0.025. The E3 solver, response fit, architecture, objective and optimizer schedule are retained. Each fit gets 280 optimizer updates. P3 uses the original E3 polynomial-profile anchors and P4 uses the same rollout objective without the anchor penalty.

There are 10 fresh base-noise datasets with three optimizer seeds per dataset, giving 240 fits. Each dataset uses one standardized fine-grid noise draw across both noise scales. The coarser observation sequence is a subset of the finer one. The starts, known histories and initialization are paired across conditions and arms. The observations at the initial time and the prescribed histories are exact.

The primary recovery metric is delay RMSE on the original E3 grid of 201 states. The separate forecast metric is MSE against clean held-out trajectories at the original three test states and four histories per state, evaluated at observation spacing 0.025 with Euler step 0.00625. The E3 native-step forecast score is also recorded.

## Run

The recorded CPU environment uses Python 3.13.18, PyTorch 2.14.1+cpu,
NumPy 2.4.6, pandas 2.2.3 and SciPy 1.16.3. Use a separate environment from the
profile study. From the repository root, use a new output path for each mode
or runtime.

```bash
python studies/sensitivity/neural/reproduction/run_r1_neural.py \
  --mode smoke --output generated/reproduction/appendix_h/neural-smoke --workers 2
python studies/sensitivity/neural/reproduction/run_r1_neural.py \
  --mode full --output generated/reproduction/appendix_h/neural-full --workers 2
```

The smoke run uses eight short fits and a separate seed pair to exercise every noise/sampling/method combination. Its scores are for execution checks. Repeating the full command resumes incomplete fits when the run signature matches. Use one active runner per output directory. The manifest records source hashes, runtime versions, settings and the full fit count. The scripts `e3_data.py`, `e3_neural.py`, `e3_fit.py`, `e3_summary.py` and `run_experiment3.py`, and the E3 `protocol.json`, are from the submitted experiment.

## Results

- `runs.csv` records all fits, including raw profile-anchor error, delay recovery and held-out forecast metrics.
- `dataset_medians.csv` takes medians across the three optimizer repeats within each dataset.
- `paired_by_dataset.csv` and `paired_summary.csv` compare P3 and P4 within each noise and sampling cell.
- `sensitivity_summary.csv` contrasts high versus low noise and fine versus coarse sampling within each method, holding the other factor fixed.
- `cases/` holds observations, delay curves and model state dictionaries. `records/` holds per-fit completion records. The source, seed and environment information is in `manifest.json`.

The independent units for these descriptive comparisons are the 10 base-noise datasets. Statewise errors, the four conditions and optimizer repeats are nested or paired. Ratios use medians of per-dataset paired ratios, with the middle half of those ratios reported descriptively. Ratios below one favor the numerator. The study concerns one synthetic system with informative matched initial histories.
