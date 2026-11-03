# Results in the common format

Each model variant has one folder here, written by `seasonal/export_results.py`
from a finished seasonal run (`python export_results.py --variant <name>`).

| File | Content |
|---|---|
| `predictions.parquet` | One row per season × test timestep × predicted (non-reference) bus |
| `metrics.csv` | Per season and pooled, for `unmeasured_only` and `all_predicted_buses`: RMSE, MAE, max absolute error, coverage_95, mean 95% PI width (all in pu) |
| `hyperparameters.csv` | Learned kernel hyperparameters per season, as reported by the model's `kernel_summary()` |
| `manifest.json` | Git commit, package versions, split/sensor/seed settings, consistency check against the pipeline's own metric files, file hashes |

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
