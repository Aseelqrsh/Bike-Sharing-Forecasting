"""
ECE 9603/9063 - Data Analytics Foundations
Assignment 1: Forecasting
Script: train_forecast.py (main entry point - run from the repository root)

Pipeline:
  1. Load data, exploratory summary + Figure 1
  2. Chronological 80/20 hold-out: fit on first 80% of hours, test on last 20%
  3. Expanding-window time-series cross-validation (5 folds) inside the
     training period only, so the hold-out test set stays untouched
  4. Export tables, figures and a plain fact sheet (report_facts.md) with every
     number quoted in the report
"""

import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_loader import DROPPED_COLS, FEATURE_COLS, chronological_split, load_data
from eda import run_eda
from evaluate import compute_metrics, plot_cv_folds, plot_error_by_hour, plot_forecast_week
from models import MODEL_BUILDERS, fit_predict, n_model_inputs, seasonal_naive

# Early CV folds contain months/seasons not yet seen in training; OneHotEncoder
# encodes these as all-zeros (noted in report_facts.md), so silence its warning.
warnings.filterwarnings('ignore', message='Found unknown categories')

RESULTS_DIR = os.path.join('assignment1', 'results')
N_CV_SPLITS = 5
ALL_MODELS = ['Seasonal naive'] + list(MODEL_BUILDERS)


def evaluate_split(full_df, train_df, test_df):
    """Fits every model on train_df, predicts test_df. Returns metrics, predictions, timings."""
    preds, metrics, timings, extra = {}, {}, {}, {}
    y_test = test_df['cnt'].values

    t0 = time.time()
    preds['Seasonal naive'], extra['naive_fallbacks'] = seasonal_naive(full_df, test_df)
    timings['Seasonal naive'] = time.time() - t0

    for name, builder in MODEL_BUILDERS.items():
        t0 = time.time()
        _, preds[name] = fit_predict(builder, train_df, test_df)
        timings[name] = time.time() - t0

    for name in ALL_MODELS:
        metrics[name] = compute_metrics(y_test, preds[name])
    return metrics, preds, timings, extra


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print('=' * 64)
    print('ECE 9603/9063 - Assignment 1: Forecasting pipeline')
    print('=' * 64)

    # 1. Data + EDA ----------------------------------------------------------
    df = load_data('data/hour.csv')
    attr_table, profile = run_eda(df, RESULTS_DIR)
    print(f'\n[1/4] Loaded {len(df)} hourly rows; Figure 1 + attribute table saved.')

    # 2. Chronological hold-out ---------------------------------------------
    train_df, test_df = chronological_split(df, train_ratio=0.8)
    print(f'\n[2/4] Hold-out: train {len(train_df)} h '
          f'({train_df.timestamp.min():%Y-%m-%d %H:%M} -> {train_df.timestamp.max():%Y-%m-%d %H:%M}), '
          f'test {len(test_df)} h ({test_df.timestamp.min():%Y-%m-%d %H:%M} -> '
          f'{test_df.timestamp.max():%Y-%m-%d %H:%M})')
    ho_metrics, ho_preds, ho_times, ho_extra = evaluate_split(df, train_df, test_df)

    holdout_table = pd.DataFrame([
        {'Model': m, 'MAE': round(v['MAE'], 2), 'RMSE': round(v['RMSE'], 2),
         'R2': round(v['R2'], 4), 'Fit+predict time (s)': round(ho_times[m], 2)}
        for m, v in ho_metrics.items()])
    holdout_table.to_csv(os.path.join(RESULTS_DIR, 'table2_holdout_results.csv'), index=False)
    print(holdout_table.to_string(index=False))

    # 3. Time-series cross-validation on the training period ---------------
    print(f'\n[3/4] {N_CV_SPLITS}-fold expanding-window time-series CV (training period only)...')
    tscv = TimeSeriesSplit(n_splits=N_CV_SPLITS)
    cv_rows, fold_rows = [], []
    for k, (tr_idx, va_idx) in enumerate(tscv.split(train_df), start=1):
        tr, va = train_df.iloc[tr_idx], train_df.iloc[va_idx]
        m, _, _, _ = evaluate_split(df, tr, va)
        fold_rows.append({'Fold': k, 'Train hours': len(tr),
                          'Months in validation unseen in training':
                              ', '.join(str(m) for m in sorted(set(va.mnth) - set(tr.mnth))) or 'none',
                          'Train period': f'{tr.timestamp.min():%Y-%m-%d} to {tr.timestamp.max():%Y-%m-%d}',
                          'Validation hours': len(va),
                          'Validation period': f'{va.timestamp.min():%Y-%m-%d} to {va.timestamp.max():%Y-%m-%d}'})
        for name, v in m.items():
            cv_rows.append({'Fold': k, 'Model': name, **v})
        print(f'      fold {k}: ' + ', '.join(f"{n} MAE={v['MAE']:.1f}" for n, v in m.items()))
    cv_df = pd.DataFrame(cv_rows)
    cv_df.round(4).to_csv(os.path.join(RESULTS_DIR, 'cv_per_fold_results.csv'), index=False)
    folds_df = pd.DataFrame(fold_rows)
    folds_df.to_csv(os.path.join(RESULTS_DIR, 'cv_fold_periods.csv'), index=False)

    cv_summary = (cv_df.groupby('Model', sort=False)[['MAE', 'RMSE', 'R2']]
                  .agg(['mean', 'std']).reindex(ALL_MODELS))
    cv_summary.columns = [f'{a} {b}' for a, b in cv_summary.columns]
    cv_summary.round(4).to_csv(os.path.join(RESULTS_DIR, 'table3_cv_summary.csv'))
    print(cv_summary.round(3).to_string())

    # 4. Figures + fact sheet ---------------------------------------------
    print('\n[4/4] Writing figures and fact sheet...')
    week_start = '2012-10-01'
    plot_forecast_week(test_df, ho_preds, os.path.join(RESULTS_DIR, 'fig2_forecast_week.png'), start=week_start)
    plot_error_by_hour(test_df, ho_preds, os.path.join(RESULTS_DIR, 'fig3_error_by_hour.png'))
    plot_cv_folds(cv_df, os.path.join(RESULTS_DIR, 'fig4_cv_fold_mae.png'))

    write_fact_sheet(df, train_df, test_df, attr_table, profile, holdout_table,
                     cv_summary, folds_df, ho_preds, ho_extra, week_start)
    print(f'      Done. Outputs in {RESULTS_DIR}/')


def write_fact_sheet(df, train_df, test_df, attr_table, profile, holdout_table,
                     cv_summary, folds_df, ho_preds, ho_extra, week_start):
    """Plain list of verified numbers for the report (data, not report text)."""
    y = df['cnt']
    expected_hours = int((df.timestamp.max() - df.timestamp.min()) / pd.Timedelta(hours=1)) + 1
    wd, we = profile[1], profile[0]
    y_test = test_df['cnt'].values

    def peak(series, hours):
        s = series.loc[hours]
        return f'{s.idxmax()}:00 (~{s.max():.0f} bikes)'

    def pct_better(a, b):
        return 100 * (a - b) / a

    mae = holdout_table.set_index('Model')['MAE']
    hr_err = {n: pd.Series(np.abs(y_test - p)).groupby(test_df['hr'].values).mean()
              for n, p in ho_preds.items()}
    rush = [7, 8, 9, 16, 17, 18, 19]
    test_wd = test_df['workingday'].values == 1
    rush_mask = np.isin(test_df['hr'].values, rush) & test_wd

    L = ['# Report fact sheet (auto-generated by src/train_forecast.py)', '',
         'Numbers only - write the report text yourselves.', '',
         '## Data',
         f'- Rows (hours): {len(df)}; raw columns: {df.shape[1] - 1} (excluding added timestamp)',
         f'- Period: {df.timestamp.min():%Y-%m-%d %H:%M} to {df.timestamp.max():%Y-%m-%d %H:%M}',
         f'- Hours in that period: {expected_hours}; missing hours: {expected_hours - len(df)}',
         f'- Missing values (NaN) in any column: {int(df.isnull().sum().sum())}',
         f'- Dropped columns: {", ".join(DROPPED_COLS)}',
         f'- Predictive attributes used ({len(FEATURE_COLS)}): {", ".join(FEATURE_COLS)}',
         f'- Target cnt: mean {y.mean():.2f}, median {y.median():.0f}, std {y.std():.2f}, '
         f'min {y.min()}, max {y.max()}, skewness {y.skew():.2f}',
         f'- Mean cnt 2011: {df[df.yr == 0].cnt.mean():.1f}; 2012: {df[df.yr == 1].cnt.mean():.1f}',
         f'- Mean cnt in training set: {train_df.cnt.mean():.1f}; in test set: {test_df.cnt.mean():.1f}',
         f'- weathersit=4 occurrences: train {int((train_df.weathersit == 4).sum())}, '
         f'test {int((test_df.weathersit == 4).sum())}',
         '', '### Attribute summary (also table1_attribute_summary.csv)', '',
         attr_table.round(3).astype(object).where(attr_table.notna(), '-').to_markdown(index=False), '',
         '### Hourly profile (Figure 1)',
         f'- Working days: morning peak {peak(wd, range(5, 12))}, evening peak {peak(wd, range(14, 21))}, '
         f'lowest {wd.idxmin()}:00 (~{wd.min():.0f})',
         f'- Weekends/holidays: peak {peak(we, range(0, 24))}, lowest {we.idxmin()}:00 (~{we.min():.0f})',
         '', '## Evaluation setup',
         f'- Hold-out: train = first 80% = {len(train_df)} h '
         f'({train_df.timestamp.min():%Y-%m-%d %H:%M} to {train_df.timestamp.max():%Y-%m-%d %H:%M}); '
         f'test = last 20% = {len(test_df)} h ({test_df.timestamp.min():%Y-%m-%d %H:%M} to '
         f'{test_df.timestamp.max():%Y-%m-%d %H:%M})',
         f'- Time-series CV: {N_CV_SPLITS} expanding-window folds on the training period only:', '',
         folds_df.to_markdown(index=False), '',
         '- Metrics: MAE, RMSE, R^2 (scikit-learn); negative predictions clipped to 0',
         f'- Seasonal naive used a fallback (day-before or last value) for {ho_extra["naive_fallbacks"]} '
         f'of {len(test_df)} test hours because the week-earlier hour is missing',
         '',
         '## Model inputs after each model\'s own transformation',
         f'- Linear Regression: {n_model_inputs(MODEL_BUILDERS["Linear Regression"], train_df)} columns '
         '(one-hot season/mnth/hr/weekday/weathersit with first level dropped; temp/atemp/hum/windspeed '
         'standardized; yr/holiday/workingday as 0/1)',
         f'- SVR: {n_model_inputs(MODEL_BUILDERS["SVR (RBF)"], train_df)} columns '
         '(sin/cos of hr, mnth, weekday; one-hot season; all standardized); target standardized, '
         'predictions converted back to bikes',
         '- Random Forest: 12 raw columns, no scaling',
         '- Hyperparameters: scikit-learn defaults for all three '
         '(LinearRegression(); SVR kernel=rbf, C=1.0, epsilon=0.1, gamma=scale; '
         'RandomForestRegressor n_estimators=100, random_state=42)',
         '', '## Hold-out results (Table 2)', '',
         holdout_table.to_markdown(index=False), '',
         f'- Random Forest MAE vs Linear Regression: {pct_better(mae["Linear Regression"], mae["Random Forest"]):.1f}% lower',
         f'- Random Forest MAE vs SVR: {pct_better(mae["SVR (RBF)"], mae["Random Forest"]):.1f}% lower',
         f'- Random Forest MAE vs Seasonal naive: {pct_better(mae["Seasonal naive"], mae["Random Forest"]):.1f}% lower',
         '', '## Time-series CV summary (Table 3, mean and std over folds)', '',
         cv_summary.round(3).to_markdown(), '',
         '## Error by hour (Figure 3, hold-out test set)',
         ]
    for n, e in hr_err.items():
        L.append(f'- {n}: worst hour {e.idxmax()}:00 (MAE {e.max():.1f}); best hour {e.idxmin()}:00 (MAE {e.min():.1f})')
    L.append(f'- MAE in working-day rush hours ({rush[0]}-9 and 16-19, {int(rush_mask.sum())} test hours) '
             'vs all other test hours:')
    for n, p in ho_preds.items():
        err = np.abs(y_test - p)
        L.append(f'  - {n}: rush {err[rush_mask].mean():.1f} vs other {err[~rush_mask].mean():.1f}')
    L += ['', '## Figure 2 window',
          f'- One full week of the test set: {week_start} (Monday) to 7 days later (Sunday)', '']
    with open(os.path.join(RESULTS_DIR, 'report_facts.md'), 'w') as f:
        f.write('\n'.join(L))


if __name__ == '__main__':
    main()
