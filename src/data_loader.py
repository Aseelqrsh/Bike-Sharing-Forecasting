"""
Assignment 1: Forecasting
Module: data_loader.py
Purpose: Handles data ingestion, feature selection (avoiding target leakage),
         chronological train-test splitting, and feature standardization.
"""

import os
import pandas as pd
from sklearn.preprocessing import StandardScaler

FEATURE_COLS = [
    'season', 'yr', 'mnth', 'hr', 'holiday', 
    'weekday', 'workingday', 'weathersit', 
    'temp', 'atemp', 'hum', 'windspeed'
]
TARGET_COL = 'cnt'

def load_and_preprocess_data(data_path="data/hour.csv", train_ratio=0.8):
    """
    Loads the bike sharing dataset, isolates predictive features,
    performs a chronological train/test split, and scales features.
    
    Returns:
        dict: Processed training and testing arrays, the fitted scaler,
              and original feature names.
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at: {data_path}")
        
    df = pd.read_csv(data_path)
    
    # 1. Feature isolation (X) and target isolation (y)
    X = df[FEATURE_COLS]
    y = df[TARGET_COL].values
    
    # 2. Chronological Train/Test Split (No shuffling to prevent lookahead bias)
    split_idx = int(len(df) * train_ratio)
    
    X_train = X.iloc[:split_idx]
    y_train = y[:split_idx]
    X_test = X.iloc[split_idx:]
    y_test = y[split_idx:]
    
    # 3. Standardize features (Mean = 0, Std = 1)
    # Fit strictly on X_train to prevent data leakage
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return {
        'X_train': X_train_scaled,
        'X_test': X_test_scaled,
        'y_train': y_train,
        'y_test': y_test,
        'scaler': scaler,
        'feature_names': FEATURE_COLS,
        'total_samples': len(df),
        'train_samples': len(X_train),
        'test_samples': len(X_test)
    }

if __name__ == "__main__":
    # Test data loader
    data = load_and_preprocess_data()
    print("DataLoader test successful:")
    print(f"  Training samples: {data['train_samples']}")
    print(f"  Testing samples:  {data['test_samples']}")
    print(f"  Features count:   {len(data['feature_names'])}")
