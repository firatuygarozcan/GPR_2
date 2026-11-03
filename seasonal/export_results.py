"""Export one finished seasonal run into the common long-format result files.

Run from seasonal/ after seasonal_evaluation.py has finished:

    python export_results.py --variant reference_sklearn

The experiment code is not touched. This script only reads the files that the
seasonal run already saved (daily_diagnostics, kernel hyperparameters and
run_config.json) and writes one variant folder under ``results/``:

    predictions.parquet   one row per (season, test timestep, predicted bus)
    metrics.csv           RMSE, MAE, max error, coverage_95, PI width
    hyperparameters.csv   learned kernel hyperparameters per season
    manifest.json         provenance, consistency checks and file hashes

``predictions.parquet`` is the format shared by every model variant. Its first
seven columns are fixed: season, timestamp, bus, measured_flag, y_true, mean,
std. ``simbench_step`` is kept as well because the SimBench local-clock
timestamp repeats on the autumn DST day.
"""

import argparse
import hashlib
import json
import platform
import subprocess
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np
import pandas as pd

from config import NETWORK, SEASON_ORDER, TEST_NOISE_SEED, TRAIN_NOISE_SEED
from dsse import GAUSSIAN_95_MULTIPLIER, seasonal_run_directory

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_ROOT = PROJECT_ROOT / "results"

COMMON_COLUMNS = ["season", "timestamp", "bus", "measured_flag", "y_true", "mean", "std"]
EXTRA_COLUMNS = ["simbench_step", "date", "day_of_year", "step_in_day", "bus_name"]
SCOPES = ["unmeasured_only", "all_predicted_buses"]
POOLED = "pooled"

# Metric name in this export -> column name in the pipeline's own metric files.
PIPELINE_METRIC_COLUMNS = {
    "n_predictions": "n_evaluated_voltage_predictions",
    "mae_pu": "voltage_mae_pu",
    "rmse_pu": "voltage_rmse_pu",
    "coverage_95": "voltage_coverage_95",
    "mean_pi_width_95_pu": "voltage_mean_interval_width_95_pu",
}
EXTRA_TABLES = ["optimization_log", "task_parameters", "task_covariance"]
PACKAGES = ["numpy", "pandas", "scipy", "scikit-learn", "pandapower", "simbench", "torch", "gpytorch"]


def load_predictions(run_dir):
    """Read every daily diagnostics file into the common long table."""
    files = sorted((run_dir / "daily_diagnostics").glob("day_*.csv.gz"))
    if not files:
        raise RuntimeError(f"No daily diagnostics found in {run_dir}.")

    frames = [
        pd.read_csv(path, dtype={"timestamp": str, "time": str}, float_precision="round_trip")
        for path in files
    ]
    raw = pd.concat(frames, ignore_index=True)
    frame = pd.DataFrame(
        {
            "season": raw["season"].astype(str),
            "timestamp": raw["timestamp"].astype(str),
            "bus": raw["bus"].astype(int),
            "measured_flag": raw["bus_role"].eq("measured"),
            "y_true": raw["ground_truth_vm_pu"].astype(float),
            "mean": raw["predicted_vm_pu"].astype(float),
            "std": raw["sigma_vm_pu"].astype(float),
            "simbench_step": raw["simbench_step"].astype(int),
            "date": raw["date"].astype(str),
            "day_of_year": raw["day_of_year"].astype(int),
            "step_in_day": raw["step_in_day"].astype(int),
            "bus_name": raw["simbench_name"].astype(str),
        }
    )
    season_rank = {season: rank for rank, season in enumerate(SEASON_ORDER)}
    frame = frame.sort_values(
        ["season", "simbench_step", "bus"], key=lambda col: col.map(season_rank) if col.name == "season" else col
    ).reset_index(drop=True)
    return frame[COMMON_COLUMNS + EXTRA_COLUMNS]


def validate_predictions(frame, run_config):
    """Fail if the long table does not cover exactly the run's test set."""
    if frame.duplicated(["season", "simbench_step", "bus"]).any():
        raise RuntimeError("Duplicate (season, simbench_step, bus) rows in predictions.")
    if not np.isfinite(frame[["y_true", "mean", "std"]].to_numpy()).all():
        raise RuntimeError("Non-finite y_true/mean/std values in predictions.")

    prediction_buses = sorted(int(bus) for bus in run_config["prediction_buses"])
    measured = set(int(bus) for bus in run_config["measured_buses"])
    if sorted(frame["bus"].unique().tolist()) != prediction_buses:
        raise RuntimeError("Predicted bus set differs from run_config prediction_buses.")
    expected_flags = frame["bus"].isin(measured)
    if not frame["measured_flag"].equals(expected_flags):
        raise RuntimeError("measured_flag differs from run_config measured_buses.")

    for season in SEASON_ORDER:
        days = sorted(frame.loc[frame["season"] == season, "day_of_year"].unique().tolist())
        if days != sorted(run_config["test_days_by_season"][season]):
            raise RuntimeError(f"{season}: exported days differ from the run's test days.")
    return {
        "rows": int(len(frame)),
        "prediction_buses": len(prediction_buses),
        "measured_prediction_buses": int(frame.loc[frame["measured_flag"], "bus"].nunique()),
        "test_timesteps_by_season": {
            season: int(frame.loc[frame["season"] == season, "simbench_step"].nunique())
            for season in SEASON_ORDER
        },
    }


def scope_mask(frame, scope):
    if scope == "unmeasured_only":
        return ~frame["measured_flag"]
    if scope == "all_predicted_buses":
        return pd.Series(True, index=frame.index)
    raise KeyError(f"Unknown evaluation scope: {scope}")


def compute_metrics(frame):
    """Same formulas as seasonal_evaluation.py, plus the maximum absolute error."""
    rows = []
    for season in SEASON_ORDER + [POOLED]:
        season_frame = frame if season == POOLED else frame[frame["season"] == season]
        for scope in SCOPES:
            part = season_frame[scope_mask(season_frame, scope)]
            true = part["y_true"].to_numpy(float)
            pred = part["mean"].to_numpy(float)
            std = part["std"].to_numpy(float)
            error = pred - true
            lower = pred - GAUSSIAN_95_MULTIPLIER * std
            upper = pred + GAUSSIAN_95_MULTIPLIER * std
            rows.append(
                {
                    "season": season,
                    "evaluation_scope": scope,
                    "n_predictions": int(true.size),
                    "rmse_pu": float(np.sqrt(np.mean(error**2))),
                    "mae_pu": float(np.mean(np.abs(error))),
                    "max_abs_error_pu": float(np.max(np.abs(error))),
                    "coverage_95": float(np.mean((true >= lower) & (true <= upper))),
                    "mean_pi_width_95_pu": float(np.mean(upper - lower)),
                }
            )
    return pd.DataFrame(rows)


def check_against_pipeline(metrics, run_dir, tolerance=1e-9):
    """Recomputed metrics must reproduce the pipeline's own metric files."""
    seasonal = pd.read_csv(run_dir / "seasonal_metrics.csv")
    pooled = pd.read_csv(run_dir / "pooled_overall_metrics.csv").assign(season=POOLED)
    reference = pd.concat([seasonal, pooled], ignore_index=True)

    worst = {}
    for name, column in PIPELINE_METRIC_COLUMNS.items():
        merged = metrics.merge(
            reference[["season", "evaluation_scope", column]],
            on=["season", "evaluation_scope"],
            how="left",
            validate="one_to_one",
        )
        ours = merged[name].to_numpy(float)
        theirs = merged[column].to_numpy(float)
        relative = np.abs(ours - theirs) / np.maximum(np.abs(theirs), 1e-300)
        worst[name] = float(np.max(relative))
    if max(worst.values()) > tolerance:
        raise RuntimeError(f"Exported metrics differ from the pipeline files: {worst}")
    return {"max_relative_difference": worst, "tolerance": tolerance, "passed": True}


def package_versions():
    versions = {"python": platform.python_version()}
    for package in PACKAGES:
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = None
    return versions


def git_state():
    def run(*args):
        return subprocess.run(
            ["git", *args], cwd=PROJECT_ROOT, capture_output=True, text=True, check=False
        ).stdout.strip()

    return {"commit": run("rev-parse", "HEAD"), "dirty_files": run("status", "--porcelain").splitlines()}


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for block in iter(lambda: file.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--variant", required=True, help="Folder name under results/, e.g. reference_sklearn")
    parser.add_argument("--run-dir", type=Path, default=None, help="Finished seasonal run directory")
    parser.add_argument("--results-root", type=Path, default=RESULTS_ROOT)
    args = parser.parse_args()

    run_dir = Path(args.run_dir) if args.run_dir else seasonal_run_directory(NETWORK)
    out_dir = Path(args.results_root) / args.variant
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "run_config.json", encoding="utf-8") as file:
        run_config = json.load(file)

    predictions = load_predictions(run_dir)
    coverage = validate_predictions(predictions, run_config)
    metrics = compute_metrics(predictions)
    consistency = check_against_pipeline(metrics, run_dir)
    hyperparameters = pd.read_csv(run_dir / "voltage_kernel_hyperparameters.csv")

    files = {
        "predictions": out_dir / "predictions.parquet",
        "metrics": out_dir / "metrics.csv",
        "hyperparameters": out_dir / "hyperparameters.csv",
    }
    predictions.to_parquet(files["predictions"], index=False, compression="zstd")
    metrics.to_csv(files["metrics"], index=False)
    hyperparameters.to_csv(files["hyperparameters"], index=False)

    # Backends with their own optimizer (GPyTorch) also save per-season tables:
    # every optimizer start, and for multi-task variants the per-bus parameters.
    for name in EXTRA_TABLES:
        season_files = sorted(run_dir.glob(f"{name}_*.csv"))
        if season_files:
            files[name] = out_dir / f"{name}.csv"
            pd.concat([pd.read_csv(path) for path in season_files], ignore_index=True).to_csv(
                files[name], index=False
            )

    manifest = {
        "variant": args.variant,
        "source_run_directory": str(run_dir),
        "git": git_state(),
        "environment": package_versions(),
        "experiment": {
            key: run_config.get(key)
            for key in [
                "network",
                "experiment",
                "model_type",
                "model_backend",
                "gp_variant",
                "gp_icm_rank",
                "outputs_correlated",
                "kernel",
                "gpr_alpha",
                "use_measurement_noise",
                "noise_strategy",
                "sensor_ratio",
                "sensor_seed",
                "measured_buses",
                "reference_buses",
                "prediction_buses",
                "evaluation_scopes",
                "training_days_by_season",
                "test_days_by_season",
            ]
        },
        "noise_seeds": {"train": TRAIN_NOISE_SEED, "test": TEST_NOISE_SEED},
        "gaussian_95_multiplier": GAUSSIAN_95_MULTIPLIER,
        "common_columns": COMMON_COLUMNS,
        "extra_columns": EXTRA_COLUMNS,
        "coverage": coverage,
        "consistency_with_pipeline_metrics": consistency,
        "files": {name: {"name": path.name, "sha256": sha256(path)} for name, path in files.items()},
    }
    with open(out_dir / "manifest.json", "w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2)

    print(f"Exported {coverage['rows']} prediction rows to {out_dir}")
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
