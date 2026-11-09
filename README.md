# ECE 9603/9063 Assignment 1: Hourly Bike-Sharing Demand Forecasting

Forecasts total hourly bike rentals (`cnt`) in the UCI Bike Sharing dataset
(Capital Bikeshare, Washington D.C., 2011–2012, 17,379 hours) and compares:

| Model | Approach | Transformation applied (inside the model's pipeline) |
|---|---|---|
| Seasonal naive (baseline) | Demand at the same hour one week earlier | none |
| Multivariate Linear Regression | Parametric, OLS | One-hot `season`, `mnth`, `hr`, `weekday`, `weathersit`; standardize `temp`, `atemp`, `hum`, `windspeed` |
| Support Vector Regression | Kernel method, RBF | sin/cos encoding of `hr`, `mnth`, `weekday`; one-hot `season`; standardize all features **and** the target |
| Random Forest Regressor | Tree ensemble, bagging | none (raw features) |

All three learned models use **scikit-learn default hyperparameters** (no tuning,
per the assignment). Only `random_state=42` is fixed for the Random Forest.

**Evaluation**
1. Chronological 80/20 hold-out: train on the first 13,903 hours
   (2011-01-01 to 2012-08-07 11:00), test on the last 3,476 hours (to 2012-12-31).
2. 5-fold expanding-window time-series cross-validation on the training period only
   (the hold-out test set is never used for it).

Metrics: MAE, RMSE, R². Negative predictions are clipped to 0.

## Group responsibilities

| Member | Responsible for |
|---|---|
| Aseel (Aseelqrsh) | Support Vector Regression |
| Dhyey Panchal (dhyeypanchal) | Random Forest Regressor |
| Both | Data preparation, Linear Regression and naive baselines, evaluation |

## How to run

Requires Python 3.9+.

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the full pipeline FROM THE REPOSITORY ROOT (paths are relative)
python src/train_forecast.py
```

Runtime is about 30 seconds on a laptop. The dataset is included at `data/hour.csv`
(source: https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset).

`python src/eda.py` on its own regenerates only Figure 1 and the attribute table.

## Outputs (`assignment1/results/`)

| File | Contents |
|---|---|
| `table1_attribute_summary.csv` | Type, range and statistics of the 12 inputs and the target |
| `table2_holdout_results.csv` | MAE / RMSE / R² for every model on the hold-out test set |
| `table3_cv_summary.csv` | Mean and std of each metric over the 5 CV folds |
| `cv_per_fold_results.csv`, `cv_fold_periods.csv` | Per-fold metrics and date ranges |
| `fig1_hourly_demand_profile.png` | Average demand by hour, working days vs weekends/holidays |
| `fig2_forecast_week.png` | Actual vs forecast for one test week (1–7 Oct 2012) |
| `fig3_error_by_hour.png` | Test-set MAE by hour of day for each model |
| `fig4_cv_fold_mae.png` | MAE of each model on each CV fold |
| `report_facts.md` | Every number used in the report, generated from the run |

## Code structure

```
src/
  data_loader.py     load data, timestamp, feature list, chronological split
  models.py          model pipelines (per-model transformations) + seasonal naive
  evaluate.py        metrics and figures
  eda.py             attribute table and Figure 1
  train_forecast.py  main entry point: runs everything above
  inspect_dataset.py quick dataset inspection (optional)
notebooks/           exploratory notebook (optional)
```

## Use of generative AI

Generative AI (Claude, by Anthropic) was used to help write and restructure the
code in this repository and to assist with drafting the written report
(permitted by the instructor). All results and numbers come from running this code.
