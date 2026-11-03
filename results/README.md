# Results in the common format

Each model variant has one folder here, written by `seasonal/export_results.py`
from a finished seasonal run (`python export_results.py --variant <name>`).

| File | Content |
|---|---|
| `predictions.parquet` | One row per season × test timestep × predicted (non-reference) bus |
| `metrics.csv` | Per season and pooled, for `unmeasured_only` and `all_predicted_buses`: RMSE, MAE, max absolute error, coverage_95, mean 95% PI width (all in pu) |
| `hyperparameters.csv` | Learned kernel hyperparameters per season, as reported by the model's `kernel_summary()` |
| `manifest.json` | Git commit, package versions, split/sensor/seed settings, consistency check against the pipeline's own metric files, file hashes |
| `optimization_log.csv` | Only for backends with their own optimizer (GPyTorch): every start, its final hyperparameters and log marginal likelihood |
| `task_parameters.csv` | ICM only: per-bus noise (normalized and pu²), diagonal of B and signal variance per season |
| `task_covariance.csv` | ICM only: the full learned task covariance B and its correlation, long format, per season |

Variant folders:

| Folder | Run |
|---|---|
| `reference_sklearn/` | Default run (`MODEL_BACKEND = "sklearn"`), original code |
| `gpytorch_shared/` | `DSSE_MODEL_BACKEND=gpytorch DSSE_GP_VARIANT=shared`; also holds the equivalence checks written by `seasonal/gpytorch_equivalence.py` (`equivalence_fixed_hyperparameters.csv`, `comparison_*_vs_sklearn.csv`) |
| `gpytorch_icm_r1/`, `_r2/`, `_r4/` | `DSSE_MODEL_BACKEND=gpytorch DSSE_GP_VARIANT=icm DSSE_ICM_RANK=<r>`: ICM with rank-r task covariance and per-bus noise |

ICM note: with free per-bus noise (bounds 1e-8 to 1, taken from the sklearn
WhiteKernel) the likelihood optimum is degenerate. For 10–30 of the 43 buses
per season the noise reaches the lower bound and the model interpolates the
noisy training targets (winter: median in-sample residual of those buses
2e-5 to 3e-3 in normalized units, against a shared-model noise std of
0.11–0.16). The training log marginal likelihood rises from 4–9·10⁴ (shared)
to 2.6–4.7·10⁵, while the pooled test RMSE grows by a factor of 2.0–2.7. An
independent check (one Cholesky per whitened task eigenvalue, no
eigendecomposition of K) reproduces the saved winter likelihoods (relative
difference ≤ 5e-6) and the saved test means of two checked days (≤ 5e-7 pu). At r ≥ 2
the predictive std is computed as a difference of large numbers (signal
variance up to 10⁶ in normalized units), so it is accurate only to about 1%.
Every ICM start stopped at the L-BFGS iteration limit (200).

`predictions.parquet` columns:

| Column | Meaning |
|---|---|
| `season` | winter / spring / summer / autumn (model that made the prediction) |
| `timestamp` | Original SimBench local-clock label `dd.mm.yyyy HH:MM` |
| `bus` | pandapower internal bus index (calculation key) |
| `measured_flag` | True if the bus is one of the 13 measured buses |
| `y_true` | Clean power-flow voltage magnitude (pu) |
| `mean` | Predicted mean (pu) |
| `std` | Predicted standard deviation (pu); 95% PI is `mean ± 1.96·std` |
| `simbench_step` | Unique SimBench row index (timestamps repeat on the autumn DST day) |
| `date`, `day_of_year`, `step_in_day` | Calendar metadata |
| `bus_name` | Exact SimBench bus name |

Variants that add variance components (for example input-uncertainty runs) append
extra columns after these; the first seven columns never change.
