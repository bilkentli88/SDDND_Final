# ASOC Experiment 2: delayed logistic mechanism

The package contains the fixed protocol, mathematical model/design check, self-contained runner, validation tests, and a Colab notebook. The state-dependent lag law is a synthetic choice motivated by delayed logistic growth; no measured population dataset is used.

Read `ASOC_Experiment2_Model_Check.md` first. It verifies the affine delayed-state form, negative nonzero gain on a positive state domain, analytic projective diversity of four trigonometric histories, and the right-hand initial-response convention. It also states the limits of dense-grid interpolation metrics and the degenerate-history control.

For Colab, upload `ASOC_Experiment2_Colab.ipynb` using **File → Upload notebook**, then run its cells in order. Upload `ASOC_Experiment2_Run_Package.zip` when prompted. A CPU runtime is sufficient; the code uses one CPU thread. The notebook performs algebra/gradient/accounting checks, a reduced smoke run, the full fixed study, and a checkpoint/data audit. Download the final results ZIP. Actual execution time depends on the session.

For local execution, install PyTorch 2.10.0, NumPy 2.4.2, pandas, and SciPy, then run:

```bash
python3 validate_experiment2.py
python3 run_experiment2.py --mode smoke --output smoke_results --resume
python3 run_experiment2.py --mode full --output experiment2_results --resume
python3 audit_experiment2.py --results experiment2_results
```

The fixed full design has two noise levels, five independent base-noise replications (paired across conditions), three paired model initializations, informative and degenerate histories, 180 actual neural fits, 20 direct functions, and 80 expected structural-abstention records. Structural abstention is not numerical failure. The learned rollout step is 0.025, the observation step is 0.05, and the reference solver is refined by the prespecified rule. This is not yet an end-to-end training-resolution study.

The manifest guards source/configuration/software consistency for resume. Keep failed fits and all raw outputs. Do not change settings after looking at scientific results under the same protocol identifier. The predeclared 20% practical criterion is descriptive; Gaussian noise and a history-only geometry gate do not create deterministic recovery guarantees or calibrated confidence.

The public NSDDNew architecture/training conventions are adapted with the negative-gain prior and new rollout/history design; its MIT notice is retained in `NSDDNew_LICENSE.txt`. No checkout of the original repository is needed to execute this self-contained package. The code is a new revision artifact and has not been pushed to the public repository.

`make_experiment2_figure.py` is optional analysis code for the paired-ratio figure. It requires Matplotlib in addition to the training dependencies and takes `--results experiment2_results --output figures`. The figure displays all five paired dataset ratios and their median; it shows no confidence interval or significance test.
