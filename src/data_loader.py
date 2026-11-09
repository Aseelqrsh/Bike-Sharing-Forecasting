"""
ECE 9603/9063 - Data Analytics Foundations
Assignment 1: Forecasting
Module: data_loader.py
Purpose: Loads the UCI Bike Sharing (hourly) dataset, builds a timestamp,
         selects the 12 predictive attributes (dropping leaking/ID columns),
         and performs the chronological train/test split.

Feature scaling/encoding is NOT done here: each model applies its own
transformation inside its scikit-learn Pipeline (see models.py), and every
transformer is fitted on training data only.
"""

import os
import pandas as pd

# 12 predictive attributes used by all models
CATEGORICAL_COLS = ['season', 'mnth', 'hr', 'weekday', 'weathersit']
BINARY_COLS = ['yr', 'holiday', 'workingday']
CONTINUOUS_COLS = ['temp', 'atemp', 'hum', 'windspeed']
FEATURE_COLS = ['season', 'yr', 'mnth', 'hr', 'holiday', 'weekday',
                'workingday', 'weathersit', 'temp', 'atemp', 'hum', 'windspeed']
TARGET_COL = 'cnt'

# Dropped: instant (row ID), dteday (encoded by yr/mnth/weekday/hr),
# casual + registered (cnt = casual + registered -> target leakage)
DROPPED_COLS = ['instant', 'dteday', 'casual', 'registered']


def load_data(data_path="data/hour.csv"):
    """Loads hour.csv and adds a 'timestamp' column (date + hour)."""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at: {data_path}")
    df = pd.read_csv(data_path)
    df['timestamp'] = pd.to_datetime(df['dteday']) + pd.to_timedelta(df['hr'], unit='h')
    df = df.sort_values('timestamp').reset_index(drop=True)
    return df


def chronological_split(df, train_ratio=0.8):
    """First train_ratio of the hours -> training set, the rest -> test set.
    No shuffling, so the models never see the future."""
    split_idx = int(len(df) * train_ratio)
    return df.iloc[:split_idx].copy(), df.iloc[split_idx:].copy()


if __name__ == "__main__":
    df = load_data()
    train, test = chronological_split(df)
    print(f"Total: {len(df)}  Train: {len(train)}  Test: {len(test)}")
    print(f"Train: {train.timestamp.min()} -> {train.timestamp.max()}")
    print(f"Test:  {test.timestamp.min()} -> {test.timestamp.max()}")
