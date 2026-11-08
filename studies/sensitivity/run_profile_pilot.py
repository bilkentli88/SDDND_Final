#!/usr/bin/env python3
"""Paired synthetic sensitivity of E3 profile anchors to noise, sampling and response estimation."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
from itertools import product
import json
from pathlib import Path
import platform
import sys

import numpy as np

STUDY_DIR = Path(__file__).resolve().parent
ROOT = STUDY_DIR.parents[1]
E3 = ROOT / "studies/E3/reproduction_package"
sys.path.insert(0, str(E3))
import e3_data as data


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def response_weights(method, window, config):
    step = config["observation_step"]
    n_steps = round(window / step)
    if n_steps < 1 or not np.isclose(window, n_steps * step):
        raise ValueError(f"{window} must be a positive multiple of {step}")
    if method == "polynomial":
        if n_steps + 1 != config["slope_window_points"]:
            raise ValueError("The polynomial window must match E3.")
        scaled_times = np.arange(n_steps + 1) / n_steps
        coeffs = np.polyfit(scaled_times, np.eye(n_steps + 1), config["slope_polynomial_degree"])
        return coeffs[-2] / (n_steps * step)
    weights = np.zeros(n_steps + 1)
    weights[[0, -1]] = [-1.0 / window, 1.0 / window]
    return weights


def response_anchors(observations, method, weights, config):
    states = np.asarray(config["train_states"], dtype=float)
    if method == "polynomial":
        anchors = data.estimate_anchors(observations, states, "informative", config)
        return anchors["delays"], anchors["eligible"], anchors["slope_estimates"]
    n_histories = len(config["history_c"])
    responses = (observations[:, :len(weights)] @ weights).reshape(len(states), n_histories)
    estimates = [data.profile(row, state, "informative", config) for row, state in zip(responses, states)]
    return (
        np.array([estimate[0] for estimate in estimates]),
        np.array([estimate[1] for estimate in estimates]),
        responses,
    )


def resampled_delays(response, draws, response_noise_sd, unit_signatures, grid):
    perturbed = response + response_noise_sd * draws
    centered = perturbed - perturbed.mean(axis=1, keepdims=True)
    # Minimizing the E3 squared profile is equivalent to maximizing squared projection.
    return grid[np.argmax((centered @ unit_signatures.T) ** 2, axis=1)]


def run_profile_conditions(reference, e3, protocol):
    states = np.asarray(e3["train_states"], dtype=float)
    n_histories = len(e3["history_c"])
    grid = np.linspace(*e3["delay_bounds"], e3["delay_profile_points"])
    signatures = data.signatures(states[0], grid, "informative", e3)
    centered = signatures - signatures.mean(axis=1, keepdims=True)
    unit_signatures = centered / np.linalg.norm(centered, axis=1, keepdims=True)
    # In E3 each state adds only a constant offset, removed by centering.
    for state in states[1:]:
        values = data.signatures(state, grid, "informative", e3)
        np.testing.assert_allclose(values - values.mean(axis=1, keepdims=True), centered, atol=1e-14)
    generating_delays = data.tau_true(states, e3)
    clean_fine = reference[("informative", "train")]["clean"]
    start = protocol["data_seed_start"]
    seeds = range(start, start + protocol["data_seed_count"])
    draws = {
        seed: np.random.default_rng(
            np.random.SeedSequence([protocol["perturbation_seed"], seed])
        ).normal(size=(len(states), protocol["perturbation_count"], n_histories))
        for seed in seeds
    }

    state_rows, dataset_rows, summaries = [], [], []
    conditions = product(protocol["noise_levels"], protocol["observation_steps"],
                         protocol["response_windows"].items())
    for sigma, step, (method, window) in conditions:
        config = data.resolution_config(e3, step)
        weights = response_weights(method, window, config)
        stride = round(step / e3["fine_observation_step"])
        clean = clean_fine[:, ::stride]
        clean_anchors, _, _ = response_anchors(clean, method, weights, config)
        # E3 noise is iid across histories and positive times; the initial state is exact.
        response_noise_sd = sigma * float(np.linalg.norm(weights[1:]))
        condition = {"noise_sd": sigma, "observation_step": step,
                     "response_estimator": method, "window": window}
        current_states, current_datasets = [], []
        for seed in seeds:
            observed = data.add_noise(clean_fine, sigma, seed)[:, ::stride]
            anchors, eligible, responses = response_anchors(observed, method, weights, config)
            if not eligible.all():
                raise ValueError("The informative E3 pilot requires eligible history geometry.")
            abs_errors = np.abs(anchors - generating_delays)
            dataset_states = []
            for index, state in enumerate(states):
                perturbed_delays = resampled_delays(
                    responses[index], draws[seed][index], response_noise_sd, unit_signatures, grid
                )
                lower, upper = np.quantile(perturbed_delays, protocol["interval_quantiles"])
                width = float(upper - lower)
                accepted = bool(width <= protocol["maximum_interval_width"]
                                and lower <= anchors[index] <= upper)
                dataset_states.append({
                    **condition, "data_seed": seed, "state": state,
                    "true_delay": generating_delays[index], "delay_estimate": anchors[index],
                    "clean_delay_estimate": clean_anchors[index], "absolute_error": abs_errors[index],
                    "interval_lower": lower, "interval_upper": upper, "interval_width": width,
                    "interval_contains_truth": bool(lower <= generating_delays[index] <= upper),
                    "estimate_at_grid_boundary": bool(anchors[index] in (grid[0], grid[-1])),
                    "accepted": accepted,
                    "accepted_large_error": bool(
                        accepted and abs_errors[index] > protocol["large_delay_error_threshold"]
                    ),
                })
            current_states.extend(dataset_states)
            current_datasets.append({
                **condition, "data_seed": seed,
                "delay_rmse": float(np.sqrt(np.mean(abs_errors ** 2))),
                "accepted_states": sum(row["accepted"] for row in dataset_states),
                "accepted_large_errors": sum(row["accepted_large_error"] for row in dataset_states),
            })

        errors = [row["delay_rmse"] for row in current_datasets]
        accepted_errors = [row["absolute_error"] for row in current_states if row["accepted"]]
        summaries.append({
            **condition, "data_draws": len(current_datasets), "state_draw_pairs": len(current_states),
            "median_dataset_rmse": float(np.median(errors)),
            "q25_dataset_rmse": float(np.quantile(errors, 0.25)),
            "q75_dataset_rmse": float(np.quantile(errors, 0.75)),
            "noiseless_grid_rmse": float(np.sqrt(np.mean((clean_anchors - generating_delays) ** 2))),
            "response_noise_sd": response_noise_sd,
            "median_interval_width": float(np.median([row["interval_width"] for row in current_states])),
            "accepted": len(accepted_errors), "abstained": len(current_states) - len(accepted_errors),
            "median_accepted_abs_error": float(np.median(accepted_errors)) if accepted_errors else None,
            "accepted_large_errors": sum(row["accepted_large_error"] for row in current_states),
            "interval_contains_truth": sum(row["interval_contains_truth"] for row in current_states),
            "estimates_at_grid_boundary": sum(row["estimate_at_grid_boundary"] for row in current_states),
        })
        state_rows.extend(current_states)
        dataset_rows.extend(current_datasets)
    return state_rows, dataset_rows, summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="New directory for generated pilot results.")
    args = parser.parse_args()
    e2_config_path = ROOT / "studies/E2/reproduction_package/protocol.json"
    e3_config_path = E3 / "protocol.json"
    protocol_path = STUDY_DIR / "protocol.json"
    e2, e3, protocol = (read_json(path) for path in (e2_config_path, e3_config_path, protocol_path))
    for key in ("model", "delay_bounds", "history_amplitude", "history_frequency",
                "history_c", "history_d", "train_states", "test_states"):
        if e2[key] != e3[key]:
            raise ValueError(f"E2 and E3 disagree on {key}")
    if protocol["noise_levels"] != e2["noise_levels"] or protocol["observation_steps"] != e3["observation_steps"]:
        raise ValueError("Pilot observation settings differ from E2/E3.")

    args.output.mkdir(parents=True, exist_ok=False)
    sources = [e2_config_path, e3_config_path, E3 / "e3_data.py", protocol_path, Path(__file__).resolve()]
    write_json(args.output / "manifest.json", {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(), "numpy": np.__version__,
        "source_sha256": {path.relative_to(ROOT).as_posix(): file_hash(path) for path in sources},
        "protocol": protocol,
    })
    fine = data.resolution_config(e3, e3["fine_observation_step"])
    reference, convergence = data.refined_reference(fine)
    state_rows, dataset_rows, summary = run_profile_conditions(reference, e3, protocol)
    write_json(args.output / "reference_convergence.json", convergence)
    write_csv(args.output / "per_state.csv", state_rows)
    write_csv(args.output / "per_dataset.csv", dataset_rows)
    write_csv(args.output / "summary.csv", summary)
    print(f"Completed {len(summary)} conditions, {len(dataset_rows)} dataset rows, and {len(state_rows)} state rows.")


if __name__ == "__main__":
    main()
