"""
Assignment 1: Forecasting
Module: models.py
Purpose: Encapsulates model training and inference for:
         1. Multivariate Linear Regression (Parametric linear baseline)
         2. Support Vector Regression (Kernel dual space)
"""

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR

def train_linear_regression(X_train, y_train, X_test):
    """
    Fits an Ordinary Least Squares (OLS) Multivariate Linear Regression model.
    """
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred = np.clip(y_pred, 0, None)
    return model, y_pred

def train_svr(X_train, y_train, X_test, kernel='rbf', C=100.0, epsilon=10.0):
    """
    Fits a Support Vector Regression (SVR) model with an RBF kernel.
    """
    model = SVR(kernel=kernel, C=C, epsilon=epsilon)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred = np.clip(y_pred, 0, None)
    return model, y_pred
