# ASOC Experiment 1 run package

This package prepares the missing comparison between joint trajectory fitting and a delay network trained only on profile anchors. It does not contain new neural results. The manuscript's earlier direct-interpolation pilot remains separate.

1. Open `ASOC_Experiment1_Colab.ipynb` at <https://colab.research.google.com/> using **File → Upload notebook**.
2. Run the notebook cells in order. Upload the complete `ASOC_Experiment1_Run_Package.zip` when prompted by the first cell.
3. Use the standard CPU runtime. The notebook checks the archived data, runs a small execution smoke check, and starts the full 60-fit experiment.
4. Download and return **ASOC_Experiment1_Results.zip** using the final cell. Preserve it even if it contains recorded failed fits; failures are part of the audit.

Read `ASOC_Experiment1_Protocol.md` for the exact comparison, seeds, losses, summary definitions, limitations, and failure policy. `protocol.json` is the machine-readable configuration. `run_experiment1.py` implements it. `validate_package.py` checks archived-data consistency, aggregation/failure accounting using synthetic fixtures, and notebook syntax. `preparation_validation.log` records those checks. A separate isolated CPU environment checks neural execution; `neural_smoke_validation.json` records its outcome. The notebook repeats the smoke step before full training.

The public reproduction repository is downloaded at a fixed commit in Colab. The package does not modify that repository or depend on your Overleaf projects. The code runs on CPU and does not require a GPU. Results may vary by software/hardware; the manifest records versions and refuses a mixed-environment resume. The training-loop parity check compares P3/P4 with the pinned original implementation on an old-seed smoke budget. `NSDDNew_LICENSE.txt` preserves the original repository's MIT notice for the reused training logic.

Local alternative, after installing NumPy, pandas, SciPy, and PyTorch and checking out the pinned repository:

```bash
python3 validate_package.py --repo /path/to/NSDDNew
python3 run_experiment1.py --repo /path/to/NSDDNew --mode smoke --output smoke_results --resume
python3 run_experiment1.py --repo /path/to/NSDDNew --mode full --output experiment1_results --resume
```

Smoke mode uses old seeds and reduced training; it supplies no scientific evidence. Full mode uses five new noise realizations and three paired initializations. A1/A2 and the direct interpolants produce delay functions only, so trajectory-error fields for them remain missing.
