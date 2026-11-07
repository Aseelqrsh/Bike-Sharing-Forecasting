"""
Assignment 1: Forecasting
Module: evaluate.py
Purpose: Computes evaluation metrics (MAE, RMSE, R²) and generates
         comparative figures and tables for the final report.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def compute_metrics(y_true, y_pred, model_name, paradigm):
    """
    Computes MAE, RMSE, and R² for a model's predictions.
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    return {
        'Model / Algorithm': model_name,
        'Model Paradigm': paradigm,
        'MAE (Bikes)': round(mae, 2),
        'RMSE (Bikes)': round(rmse, 2),
        'R² Score': round(r2, 4)
    }

def save_comparison_table(metrics_list, output_path):
    """
    Exports a list of metric dictionaries to a CSV table.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df = pd.DataFrame(metrics_list)
    df.to_csv(output_path, index=False)
    return df

def plot_forecast_comparison(y_test, predictions_dict, output_path, window_start=100, window_hours=72):
    """
    Plots ground truth against model forecasts over a continuous window.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    sns.set_theme(style="whitegrid", font_scale=1.1)
    plt.figure(figsize=(13, 6))
    
    window_end = window_start + window_hours
    hours = np.arange(window_hours)
    
    # 1. Ground truth
    plt.plot(
        hours, 
        y_test[window_start:window_end], 
        label='Actual Demand (Ground Truth)', 
        color='#111111', 
        linewidth=2.5, 
        marker='o', 
        markersize=4
    )
    
    # Colors and styles for each model
    styles = {
        'Multivariate Linear Regression': {'color': '#d62728', 'linestyle': '--', 'linewidth': 2.0},
        'Support Vector Regression (RBF)': {'color': '#1f77b4', 'linestyle': '-', 'linewidth': 2.2}
    }
    
    # 2. Model forecasts
    for name, y_pred in predictions_dict.items():
        style = styles.get(name, {'color': 'gray', 'linestyle': '-', 'linewidth': 1.5})
        plt.plot(
            hours, 
            y_pred[window_start:window_end], 
            label=f"{name} (Forecast)", 
            **style
        )
        
    plt.title(f"Comparative Forecasting: Actual vs. Models ({window_hours}-Hour Test Window)", 
              fontsize=13, fontweight='bold', pad=12)
    plt.xlabel("Consecutive Hours in Test Set", fontsize=11)
    plt.ylabel("Hourly Bike Rental Count (cnt)", fontsize=11)
    plt.legend(frameon=True, loc='upper right')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
