"""
ECE 9603/9063 - Data Analytics Foundations
Assignment 1: Forecasting
Script: train_forecast.py (Master Pipeline Entry Point)
Purpose: Coordinates data loading, trains the 3 distinct forecasting models,
         evaluates metrics, and exports comparison tables & figures.
"""

import os
import sys

# Ensure src directory is in sys.path
sys.path.append(os.path.dirname(__file__))

from data_loader import load_and_preprocess_data
from models import train_linear_regression, train_svr, train_random_forest
from evaluate import compute_metrics, save_comparison_table, plot_forecast_comparison

def main():
    print("=" * 60)
    print("ECE 9603/9063 - Assignment 1: Forecasting Master Pipeline")
    print("=" * 60)
    
    # 1. Load and preprocess data
    print("\n[1/4] Loading and standardizing dataset...")
    data = load_and_preprocess_data(data_path="data/hour.csv")
    print(f"      Training samples: {data['train_samples']} (First 80% chronologically)")
    print(f"      Testing samples:  {data['test_samples']} (Last 20% unseen future)")
    
    X_train, X_test = data['X_train'], data['X_test']
    y_train, y_test = data['y_train'], data['y_test']
    
    # 2. Train Model 1: Multivariate Linear Regression
    print("\n[2/4] Training Model 1: Multivariate Linear Regression (OLS)...")
    lr_model, y_pred_lr = train_linear_regression(X_train, y_train, X_test)
    m_lr = compute_metrics(y_test, y_pred_lr, 'Multivariate Linear Regression', 'Parametric / Linear Hyperplane')
    print(f"      LR -> MAE: {m_lr['MAE (Bikes)']}, RMSE: {m_lr['RMSE (Bikes)']}, R²: {m_lr['R² Score']}")
    
    # 3. Train Model 2: Support Vector Regression (SVR)
    print("\n      Training Model 2: Support Vector Regression (RBF Kernel)...")
    svr_model, y_pred_svr = train_svr(X_train, y_train, X_test)
    m_svr = compute_metrics(y_test, y_pred_svr, 'Support Vector Regression (RBF)', 'Non-Parametric / Kernel Dual Space')
    print(f"      SVR -> MAE: {m_svr['MAE (Bikes)']}, RMSE: {m_svr['RMSE (Bikes)']}, R²: {m_svr['R² Score']}")
    
    # 4. Train Model 3: Random Forest Regressor
    print("\n      Training Model 3: Random Forest Regressor (Ensemble Bagging)...")
    rf_model, y_pred_rf = train_random_forest(X_train, y_train, X_test)
    m_rf = compute_metrics(y_test, y_pred_rf, 'Random Forest Regressor', 'Non-Parametric / Tree Bagging Ensemble')
    print(f"      RF -> MAE: {m_rf['MAE (Bikes)']}, RMSE: {m_rf['RMSE (Bikes)']}, R²: {m_rf['R² Score']}")
    
    # 5. Export Comparative Evaluation Table
    print("\n[3/4] Exporting comparison deliverables...")
    results_dir = os.path.join("assignment1", "results")
    table_path = os.path.join(results_dir, "model_comparison_results.csv")
    comparison_table = save_comparison_table([m_lr, m_svr, m_rf], table_path)
    print(f"      Saved table to: {table_path}\n")
    print(comparison_table.to_string(index=False))
    
    # 6. Export 72-Hour Comparative Forecast Plot
    print("\n[4/4] Generating 72-hour forecast comparison plot...")
    fig_path = os.path.join(results_dir, "fig2_model_forecast_comparison.png")
    plot_forecast_comparison(
        y_test, 
        {
            'Multivariate Linear Regression': y_pred_lr,
            'Support Vector Regression (RBF)': y_pred_svr,
            'Random Forest Regressor': y_pred_rf
        },
        fig_path,
        window_start=100,
        window_hours=72
    )
    print(f"      Saved Figure 2 to: {fig_path}")
    
    print("\n" + "=" * 60)
    print("Pipeline Finished Successfully!")
    print("=" * 60)

if __name__ == "__main__":
    main()
