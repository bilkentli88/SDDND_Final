#!/usr/bin/env python3
"""Run the paired E2/E3 neural robustness experiment for the ASOC revision."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from itertools import product
import json
import multiprocessing
import os
from pathlib import Path
import platform

import numpy as np
import pandas as pd
import scipy

import e3_data as data
from e3_fit import atomic_json, sha
from run_experiment3 import load_config, stem, train_job

HERE = Path(__file__).resolve().parent
SOURCE_FILES = (
    "run_r1_neural.py", "r1_protocol.json", "e3_data.py", "e3_neural.py",
    "e3_fit.py", "e3_summary.py", "run_experiment3.py", "protocol.json",
)
INDEX_FIELDS = ["noise_std", "observation_step", "data_seed"]


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def run_id(row):
    return f"sd{row['noise_std']:g}_{stem(row)}"


def case_directory(output, sigma):
    return output / "cases" / f"sd{sigma:g}"


def expected_fits(protocol):
    for sigma, step, seed, opt, method in product(
        protocol["noise_levels"], protocol["observation_steps"], protocol["data_seeds"],
        protocol["optimization_seeds"], protocol["methods"],
    ):
        yield {
            "noise_std": sigma, "observation_step": step, "data_seed": seed,
            "optimization_seed": opt, "method": method, "rollout_step": protocol["rollout_step"],
            "model_seed": opt + 100 * protocol["data_seeds"].index(seed),
        }


def load_protocol(mode):
    base = load_config(HERE / "protocol.json")
    protocol = read_json(HERE / "r1_protocol.json")
    if base["protocol_id"] != protocol["base_protocol"]:
        raise ValueError("Wrong frozen E3 base protocol.")
    if protocol["observation_steps"] != base["observation_steps"]:
        raise ValueError("Sampling spacings must match the submitted E3 study.")
    if mode == "smoke":
        smoke = protocol["smoke"]
        protocol.update(data_seeds=smoke["data_seeds"], optimization_seeds=smoke["optimization_seeds"])
        base["training_phases"] = smoke["training_phases"]
    return base, protocol


def prepared_cases(base, protocol, output):
    reference_config = data.resolution_config(base, base["fine_observation_step"], protocol["rollout_step"])
    reference, convergence = data.refined_reference(reference_config)
    atomic_json(output / "reference_convergence.json", convergence)
    cases = {}
    for sigma in protocol["noise_levels"]:
        base_for_noise = {**base, "noise_levels": [sigma], "evaluation_steps": protocol["evaluation_steps"]}
        directory = case_directory(output, sigma)
        for name in ("checkpoints", "curves", "observations"):
            (directory / name).mkdir(parents=True, exist_ok=True)
        for step, seed in product(protocol["observation_steps"], protocol["data_seeds"]):
            config = data.resolution_config(base_for_noise, step, protocol["rollout_step"])
            saved = data.observed_case(reference, base_for_noise, step, seed)
            fields = {f"anchor_{name}": value for name, value in saved["anchor"].items()}
            for split in ("train", "test"):
                block = saved[split]
                fields.update({f"{split}_{name}": value for name, value in block.items() if name != "metadata"})
                fields.update({f"{split}_meta_{name}": value for name, value in block["metadata"].items()})
            path = directory / "observations" / f"obs{step:g}_data{seed}.npz"
            if not path.exists():
                pending = path.with_suffix(".tmp.npz")
                np.savez_compressed(pending, **fields)
                pending.replace(path)
            cases[(sigma, step, seed)] = config, saved, directory
    return cases


def summarize_comparison(group, numerator, denominator):
    a = group[numerator]
    b = group[denominator]
    ratio = a[b > 0] / b[b > 0]
    return {
        "n_datasets": len(group), "n_valid_ratios": len(ratio),
        "median_numerator": a.median(), "median_denominator": b.median(),
        "median_paired_ratio": ratio.median() if len(ratio) else np.nan,
        "ratio_q25": ratio.quantile(0.25) if len(ratio) else np.nan,
        "ratio_q75": ratio.quantile(0.75) if len(ratio) else np.nan,
        "median_paired_difference": (a - b).median(),
        "numerator_lower": int((a < b).sum()), "ties": int((a == b).sum()),
    }


def export_summaries(rows, protocol, output):
    if len(rows) != len(list(expected_fits(protocol))) or any(row["status"] != "ok" for row in rows):
        raise ValueError("Complete successful fit records are required for paired summaries.")
    frame = pd.DataFrame(rows)
    paired_key = ["data_seed", "optimization_seed"]
    if (frame.groupby(paired_key)["initial_tau_hash"].nunique() != 1).any():
        raise ValueError("Delay-branch initialization differs within a paired block.")
    metrics = protocol["metrics"]
    if not np.isfinite(frame[metrics].to_numpy()).all() or (frame[metrics] < 0).any().any():
        raise ValueError("Invalid metric in fit records.")

    medians = frame.groupby([*INDEX_FIELDS, "method"])[metrics].median().reset_index()
    medians.to_csv(output / "dataset_medians.csv", index=False)
    by_dataset, paired_summary, sensitivity = [], [], []
    for metric in metrics:
        pairs = medians.pivot(index=INDEX_FIELDS, columns="method", values=metric).reset_index()
        pairs.columns.name = None
        pairs["metric"] = metric
        pairs["P3_over_P4"] = pairs["P3"] / pairs["P4"].where(pairs["P4"] > 0)
        by_dataset.append(pairs)
        for (sigma, step), group in pairs.groupby(["noise_std", "observation_step"]):
            paired_summary.append({
                "noise_std": sigma, "observation_step": step, "metric": metric,
                **summarize_comparison(group, "P3", "P4"),
            })

        for name, varied, fixed, top, bottom in (
            ("high_over_low_noise", "noise_std", "observation_step",
             max(protocol["noise_levels"]), min(protocol["noise_levels"])),
            ("fine_over_coarse_sampling", "observation_step", "noise_std",
             min(protocol["observation_steps"]), max(protocol["observation_steps"])),
        ):
            for (method, fixed_value), group in medians.groupby(["method", fixed]):
                contrast = group.pivot(index="data_seed", columns=varied, values=metric)
                sensitivity.append({
                    "contrast": name, "method": method, "metric": metric,
                    "fixed_factor": fixed, "fixed_value": fixed_value,
                    "numerator_level": top, "denominator_level": bottom,
                    **summarize_comparison(contrast, top, bottom),
                })
    pd.concat(by_dataset, ignore_index=True).to_csv(output / "paired_by_dataset.csv", index=False)
    pd.DataFrame(paired_summary).to_csv(output / "paired_summary.csv", index=False)
    pd.DataFrame(sensitivity).to_csv(output / "sensitivity_summary.csv", index=False)


def experiment(output, mode, workers):
    import torch

    base, protocol = load_protocol(mode)
    expected = {run_id(row): row for row in expected_fits(protocol)}
    signature = {
        "mode": mode, "device": "cpu", "torch": torch.__version__,
        "numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__,
        "python": platform.python_version(), "machine": platform.machine(),
        "source_sha256": {name: sha(HERE / name) for name in SOURCE_FILES},
    }
    output.mkdir(parents=True, exist_ok=True)
    manifest = output / "manifest.json"
    if manifest.exists():
        if read_json(manifest)["signature"] != signature:
            raise ValueError("Source, mode or software differs from the saved run. Use a new output path.")
    elif any(output.iterdir()):
        raise ValueError(f"Cannot initialize a nonempty output directory without a manifest: {output}")
    else:
        atomic_json(manifest, {
            "signature": signature, "frozen_before_fit_utc": datetime.now(timezone.utc).isoformat(),
            "protocol": protocol, "base_config": base, "expected_fits": len(expected),
        })

    records = output / "records"
    records.mkdir(exist_ok=True)
    complete = {}
    for path in sorted(records.glob("*.json")):
        row = read_json(path)
        name = path.stem
        if name not in expected or any(row.get(key) != value for key, value in expected[name].items()):
            raise ValueError(f"Unexpected fit record: {path.name}")
        directory = case_directory(output, row["noise_std"])
        checkpoint = directory / "checkpoints" / f"{stem(row)}.pt"
        curve = directory / "curves" / f"{stem(row)}.csv"
        if row["status"] == "ok" and checkpoint.is_file() and curve.is_file():
            complete[name] = row

    pending = {name: row for name, row in expected.items() if name not in complete}
    print(f"{mode}: {len(complete)} complete, {len(pending)} pending, {len(expected)} planned.", flush=True)
    if pending:
        (output / "complete.json").unlink(missing_ok=True)
        cases = prepared_cases(base, protocol, output)
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
            futures = {}
            for name, row in pending.items():
                case = cases[(row["noise_std"], row["observation_step"], row["data_seed"])]
                futures[pool.submit(train_job, row, *case)] = name
            finished = 0
            for future in as_completed(futures):
                name = futures[future]
                row = future.result()
                if row["status"] == "ok":
                    complete[name] = row
                else:
                    with (output / "failed_attempts.jsonl").open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps({"run": name, "utc": datetime.now(timezone.utc).isoformat(),
                                                 "error": row["error"]}) + "\n")
                atomic_json(records / f"{name}.json", row)
                finished += 1
                if row["status"] != "ok" or finished % 8 == 0 or finished == len(pending):
                    print(f"{finished}/{len(pending)} processed this session. {len(complete)}/{len(expected)} complete.", flush=True)

    rows = [read_json(records / f"{name}.json") for name in expected]
    pd.DataFrame(rows).to_csv(output / "runs.csv", index=False)
    if len(complete) != len(expected):
        raise SystemExit("Some fits failed. Inspect runs.csv and rerun to retry incomplete fits.")
    export_summaries(rows, protocol, output)
    atomic_json(output / "complete.json", {
        "mode": mode, "completed_fits": len(rows), "n_datasets": len(protocol["data_seeds"]),
    })
    print(f"Finished {len(rows)} successful {mode} fits.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "full"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    experiment(args.output.resolve(), args.mode, args.workers)


if __name__ == "__main__":
    main()
