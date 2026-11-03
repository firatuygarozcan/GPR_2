"""Equivalence checks between the sklearn reference and the GPyTorch "shared" model.

Run from seasonal/ after the sklearn reference run (and, for part b, after the
GPyTorch run and both exports):

    python gpytorch_equivalence.py --part a   # fixed sklearn hyperparameters
    python gpytorch_equivalence.py --part b   # independently optimized GPyTorch run

Part a rebuilds every season's training and test inputs exactly like
seasonal_evaluation.py, loads the fitted sklearn model of that season, gives
its hyperparameters to the GPyTorch model as fixed values and compares the log
marginal likelihood and every test prediction (mean and std, all 43 buses).

Part b compares the exported results of the two full runs: hyperparameters,
metrics per season and scope, and the predictions themselves.
"""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from config import MODEL_BACKEND, NETWORK, SEASON_ORDER, TEST_BATCH_DAYS, TEST_NOISE_SEED, TRAIN_NOISE_SEED
from dsse import (
    add_noise_by_day,
    build_voltage_dataset,
    get_reference_buses,
    load_ground_truth,
    load_network,
    seasonal_run_directory,
)
from gpytorch_models import GPyTorchDSSEGaussianProcess

RESULTS_ROOT = Path(__file__).resolve().parent.parent / "results"
REFERENCE = "reference_sklearn"
VARIANT = "gpytorch_shared"


def chunked(items, size):
    for start in range(0, len(items), size):
        yield items[start : start + size]


def sklearn_hyperparameters(model):
    kernel = model.model.kernel_
    return {
        "outputscale": float(kernel.k1.k1.constant_value),
        "lengthscale": float(kernel.k1.k2.length_scale),
        "noise": float(kernel.k2.noise_level),
    }


def part_a(out_dir):
    if MODEL_BACKEND != "sklearn":
        raise RuntimeError("Part a reads the sklearn reference run; unset DSSE_MODEL_BACKEND.")
    run_dir = seasonal_run_directory(NETWORK)
    with open(run_dir / "run_config.json", encoding="utf-8") as file:
        run_config = json.load(file)
    reference = pd.read_parquet(RESULTS_ROOT / REFERENCE / "predictions.parquet")

    net = load_network(NETWORK)
    all_buses = [int(bus) for bus in net.bus.index]
    reference_buses = get_reference_buses(net)
    measured_buses = [int(bus) for bus in run_config["measured_buses"]]
    target_columns = [f"bus_{int(bus)}_vm_pu" for bus in run_config["prediction_buses"]]

    rows = []
    for season in SEASON_ORDER:
        sk = joblib.load(run_dir / run_config["trained_model_files"][season])
        truth = load_ground_truth(net, NETWORK, run_config["training_days_by_season"][season])
        X_clean, y_train, meta, _, _ = build_voltage_dataset(truth, measured_buses, all_buses, reference_buses)
        X_train = add_noise_by_day(X_clean, meta, TRAIN_NOISE_SEED)
        Y_train = y_train[target_columns]
        hyperparameters = sklearn_hyperparameters(sk)
        gp = GPyTorchDSSEGaussianProcess().fit(X_train, Y_train, hyperparameters=hyperparameters)

        # Same scaler statistics prove that the inputs were rebuilt identically.
        scaler_difference = max(
            float(np.max(np.abs(gp.scaler.mean_ - sk.scaler.mean_))),
            float(np.max(np.abs(gp.scaler.scale_ - sk.scaler.scale_) / sk.scaler.scale_)),
        )

        sk_mean, sk_std, gp_mean, gp_std, steps = [], [], [], [], []
        for batch_days in chunked(run_config["test_days_by_season"][season], TEST_BATCH_DAYS):
            truth = load_ground_truth(net, NETWORK, batch_days)
            X_clean, _, test_meta, _, _ = build_voltage_dataset(truth, measured_buses, all_buses, reference_buses)
            X_test = add_noise_by_day(X_clean, test_meta, TEST_NOISE_SEED)
            mean, std = sk.predict(X_test)
            sk_mean.append(mean[target_columns]), sk_std.append(std[target_columns])
            mean, std = gp.predict(X_test)
            gp_mean.append(mean[target_columns]), gp_std.append(std[target_columns])
            steps.append(test_meta["simbench_step"].astype(int))
        sk_mean, sk_std = pd.concat(sk_mean).to_numpy(), pd.concat(sk_std).to_numpy()
        gp_mean, gp_std = pd.concat(gp_mean).to_numpy(), pd.concat(gp_std).to_numpy()
        steps = pd.concat(steps).to_numpy()

        # The recomputed sklearn predictions must be the stored reference ones.
        stored = reference[reference["season"] == season].pivot(index="simbench_step", columns="bus", values="mean")
        stored = stored.loc[steps, [int(column.split("_")[1]) for column in target_columns]].to_numpy()

        gp_lml = float(gp.kernel_summary().iloc[0]["log_marginal_likelihood"])
        sk_lml = float(sk.model.log_marginal_likelihood_value_)
        rows.append(
            {
                "season": season,
                "n_test_timesteps": int(len(steps)),
                "n_predictions": int(sk_mean.size),
                **{f"sklearn_{name}": value for name, value in hyperparameters.items()},
                "sklearn_log_marginal_likelihood": sk_lml,
                "gpytorch_log_marginal_likelihood": gp_lml,
                "lml_relative_difference": abs(gp_lml - sk_lml) / abs(sk_lml),
                "scaler_max_difference": scaler_difference,
                "max_abs_mean_difference_pu": float(np.max(np.abs(gp_mean - sk_mean))),
                "max_mean_difference_over_std": float(np.max(np.abs(gp_mean - sk_mean) / sk_std)),
                "max_relative_std_difference": float(np.max(np.abs(gp_std - sk_std) / sk_std)),
                "max_abs_std_difference_pu": float(np.max(np.abs(gp_std - sk_std))),
                "sklearn_vs_stored_reference_max_abs_mean_difference_pu": float(np.max(np.abs(sk_mean - stored))),
            }
        )
        print(pd.DataFrame(rows[-1:]).T.to_string(header=False))

    result = pd.DataFrame(rows)
    result.to_csv(out_dir / "equivalence_fixed_hyperparameters.csv", index=False)
    return result


def part_b(out_dir):
    reference_dir, variant_dir = RESULTS_ROOT / REFERENCE, RESULTS_ROOT / VARIANT
    keys = ["season", "evaluation_scope"]
    sk_metrics = pd.read_csv(reference_dir / "metrics.csv")
    gp_metrics = pd.read_csv(variant_dir / "metrics.csv")
    metrics = sk_metrics.merge(gp_metrics, on=keys, suffixes=("_sklearn", "_gpytorch"), validate="one_to_one")
    for name in ["rmse_pu", "mae_pu", "max_abs_error_pu", "mean_pi_width_95_pu"]:
        metrics[f"{name}_relative_difference_percent"] = 100 * (
            metrics[f"{name}_gpytorch"] / metrics[f"{name}_sklearn"] - 1
        )
    metrics["coverage_95_difference_points"] = 100 * (
        metrics["coverage_95_gpytorch"] - metrics["coverage_95_sklearn"]
    )
    metrics["rmse_within_2_percent"] = metrics["rmse_pu_relative_difference_percent"].abs() < 2
    metrics["coverage_within_1_point"] = metrics["coverage_95_difference_points"].abs() < 1
    metrics.to_csv(out_dir / "comparison_metrics_vs_sklearn.csv", index=False)

    columns = ["season", "constant_value", "rbf_length_scale", "white_noise_level", "log_marginal_likelihood"]
    sk_hyper = pd.read_csv(reference_dir / "hyperparameters.csv")[columns]
    gp_hyper = pd.read_csv(variant_dir / "hyperparameters.csv")[columns]
    hyper = sk_hyper.merge(gp_hyper, on="season", suffixes=("_sklearn", "_gpytorch"), validate="one_to_one")
    for name in columns[1:]:
        hyper[f"{name}_relative_difference_percent"] = 100 * (
            hyper[f"{name}_gpytorch"] / hyper[f"{name}_sklearn"] - 1
        )
    hyper["lml_gain_gpytorch_minus_sklearn"] = (
        hyper["log_marginal_likelihood_gpytorch"] - hyper["log_marginal_likelihood_sklearn"]
    )
    hyper.to_csv(out_dir / "comparison_hyperparameters_vs_sklearn.csv", index=False)

    join = ["season", "simbench_step", "bus"]
    sk_pred = pd.read_parquet(reference_dir / "predictions.parquet", columns=join + ["y_true", "mean", "std"])
    gp_pred = pd.read_parquet(variant_dir / "predictions.parquet", columns=join + ["y_true", "mean", "std"])
    pred = sk_pred.merge(gp_pred, on=join, suffixes=("_sklearn", "_gpytorch"), validate="one_to_one")
    if not np.array_equal(pred["y_true_sklearn"].to_numpy(), pred["y_true_gpytorch"].to_numpy()):
        raise RuntimeError("The two runs do not share identical ground-truth targets.")
    rows = []
    for season in SEASON_ORDER + ["pooled"]:
        part = pred if season == "pooled" else pred[pred["season"] == season]
        mean_diff = part["mean_gpytorch"] - part["mean_sklearn"]
        std_ratio = part["std_gpytorch"] / part["std_sklearn"]
        rows.append(
            {
                "season": season,
                "n_predictions": int(len(part)),
                "rms_mean_difference_pu": float(np.sqrt(np.mean(mean_diff**2))),
                "max_abs_mean_difference_pu": float(np.max(np.abs(mean_diff))),
                "max_mean_difference_over_std": float(np.max(np.abs(mean_diff) / part["std_sklearn"])),
                "median_std_ratio": float(np.median(std_ratio)),
                "min_std_ratio": float(np.min(std_ratio)),
                "max_std_ratio": float(np.max(std_ratio)),
            }
        )
    predictions = pd.DataFrame(rows)
    predictions.to_csv(out_dir / "comparison_predictions_vs_sklearn.csv", index=False)
    return metrics, hyper, predictions


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--part", choices=["a", "b"], required=True)
    args = parser.parse_args()
    out_dir = RESULTS_ROOT / VARIANT
    out_dir.mkdir(parents=True, exist_ok=True)
    with pd.option_context("display.width", 200, "display.max_columns", 50):
        if args.part == "a":
            print(part_a(out_dir).to_string(index=False))
        else:
            for frame in part_b(out_dir):
                print(frame.to_string(index=False))


if __name__ == "__main__":
    main()
