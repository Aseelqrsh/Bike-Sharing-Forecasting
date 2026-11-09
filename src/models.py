"""
ECE 9603/9063 - Data Analytics Foundations
Assignment 1: Forecasting
Module: models.py
Purpose: Builds the forecasting models as scikit-learn Pipelines. Each
         pipeline contains the data transformation specific to that algorithm,
         so transformers are always fitted on the training data only.

  0. Seasonal-naive baseline: demand at the same hour one week earlier
  1. Multivariate Linear Regression (OLS) - one-hot categorical features
  2. Support Vector Regression (RBF kernel) - cyclical sin/cos encoding,
     standardized features and standardized target
  3. Random Forest Regressor - raw features (trees are scale-invariant)

All three learned models use scikit-learn's DEFAULT hyperparameters
(no tuning, as required by Assignment 1). Only random_state is fixed for
reproducibility of the Random Forest.
"""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.svm import SVR

from data_loader import BINARY_COLS, CATEGORICAL_COLS, CONTINUOUS_COLS, FEATURE_COLS

RANDOM_STATE = 42

# Period of each cyclical attribute (used for sin/cos encoding in SVR)
CYCLE_PERIODS = {'hr': 24, 'mnth': 12, 'weekday': 7}


# ---------------------------------------------------------------------------
# Model 1: Multivariate Linear Regression
# ---------------------------------------------------------------------------
def build_linear_regression():
    """OLS on one-hot encoded categoricals. Without one-hot encoding, OLS would
    treat hour 17 as '17 times' hour 1 and could only fit one slope per hour."""
    preprocess = ColumnTransformer([
        ('onehot', OneHotEncoder(drop='first', handle_unknown='ignore'), CATEGORICAL_COLS),
        ('scale', StandardScaler(), CONTINUOUS_COLS),
        ('binary', 'passthrough', BINARY_COLS),
    ])
    return Pipeline([('preprocess', preprocess), ('model', LinearRegression())])


# ---------------------------------------------------------------------------
# Model 2: Support Vector Regression (RBF)
# ---------------------------------------------------------------------------
def _cyclical_encode(X):
    """Replaces hr, mnth, weekday with sin/cos pairs so that e.g. hour 23 and
    hour 0 are close together in the RBF kernel's Euclidean distance."""
    X = pd.DataFrame(X, columns=FEATURE_COLS).copy()
    for col, period in CYCLE_PERIODS.items():
        angle = 2 * np.pi * X[col] / period
        X[f'{col}_sin'] = np.sin(angle)
        X[f'{col}_cos'] = np.cos(angle)
        X = X.drop(columns=col)
    # season is nominal (4 values): one-hot it
    for s in [1, 2, 3, 4]:
        X[f'season_{s}'] = (X['season'] == s).astype(float)
    X = X.drop(columns='season')
    return X.values.astype(float)


def build_svr():
    """RBF-kernel SVR with default C=1.0, epsilon=0.1, gamma='scale'.
    The target is standardized (and inverted after prediction) because the
    default C and epsilon assume a target of roughly unit scale; on raw counts
    up to ~1000 the default model would barely fit anything."""
    pipe = Pipeline([
        ('cyclical', FunctionTransformer(_cyclical_encode)),
        ('scale', StandardScaler()),
        ('model', SVR(kernel='rbf')),
    ])
    return TransformedTargetRegressor(regressor=pipe, transformer=StandardScaler())


# ---------------------------------------------------------------------------
# Model 3: Random Forest Regressor
# ---------------------------------------------------------------------------
def build_random_forest():
    """100 bagged regression trees (default). Raw features: tree splits do not
    depend on feature scale, so no scaling or encoding is needed."""
    return RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)


MODEL_BUILDERS = {
    'Linear Regression': build_linear_regression,
    'SVR (RBF)': build_svr,
    'Random Forest': build_random_forest,
}


def fit_predict(builder, train_df, test_df, target_col='cnt'):
    """Fits a fresh model on train_df and predicts test_df.
    Negative predictions are clipped to 0 (demand cannot be negative)."""
    model = builder()
    model.fit(train_df[FEATURE_COLS], train_df[target_col].values)
    y_pred = model.predict(test_df[FEATURE_COLS])
    return model, np.clip(y_pred, 0, None)


# ---------------------------------------------------------------------------
# Model 0: Seasonal-naive baseline
# ---------------------------------------------------------------------------
def seasonal_naive(full_df, test_df, lag_hours=168):
    """Predicts each test hour with the observed demand at the same hour one
    week (168 h) earlier. This only uses past values, which an operator would
    have. The dataset has 165 missing hours; if the lagged hour is missing we
    fall back to the same hour one day earlier, then to the most recent
    earlier observation.

    Returns (predictions, number_of_fallbacks)."""
    series = full_df.set_index('timestamp')['cnt']
    preds, fallbacks = [], 0
    for ts in test_df['timestamp']:
        week_ago = ts - pd.Timedelta(hours=lag_hours)
        day_ago = ts - pd.Timedelta(hours=24)
        if week_ago in series.index:
            preds.append(series[week_ago])
        elif day_ago in series.index:
            preds.append(series[day_ago]); fallbacks += 1
        else:
            preds.append(series[series.index < ts].iloc[-1]); fallbacks += 1
    return np.array(preds, dtype=float), fallbacks


def n_model_inputs(builder, train_df):
    """Number of input columns the model actually sees after its own encoding."""
    model = builder()
    if isinstance(model, TransformedTargetRegressor):
        pre = Pipeline(model.regressor.steps[:-1])
    elif isinstance(model, Pipeline):
        pre = Pipeline(model.steps[:-1])
    else:
        return len(FEATURE_COLS)
    return pre.fit_transform(train_df[FEATURE_COLS]).shape[1]
