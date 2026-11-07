import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. Load the dataset
df = pd.read_csv("data/hour.csv")

# 2. Define the 12 predictive features and the target
features = [
    'season', 'yr', 'mnth', 'hr', 'holiday', 
    'weekday', 'workingday', 'weathersit', 
    'temp', 'atemp', 'hum', 'windspeed'
]
target = 'cnt'

# 3. Chronological 80/20 train-test split
split_idx = int(len(df) * 0.8)

train_df = df.iloc[:split_idx]
test_df = df.iloc[split_idx:]

X_train = train_df[features]
y_train = train_df[target].values

X_test = test_df[features]
y_test = test_df[target].values

print(f"Total dataset samples: {len(df)}")
print(f"Training set:          {len(X_train)} samples (first 80% chronologically)")
print(f"Testing set:           {len(X_test)} samples (last 20% unseen future)")

# 4. Standardize features (Mean = 0, Std = 1)
# Fit strictly on training data, then transform both train and test
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\nFeatures standardized successfully!")

# 5. Model 1: Multivariate Linear Regression
print("\n" + "=" * 50)
print("Training Model 1: Multivariate Linear Regression")
print("=" * 50)

lr_model = LinearRegression()
lr_model.fit(X_train_scaled, y_train)

# Predict on unseen future test set
y_pred_lr = lr_model.predict(X_test_scaled)
y_pred_lr = np.clip(y_pred_lr, 0, None)

mae_lr = mean_absolute_error(y_test, y_pred_lr)
rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_lr))
r2_lr = r2_score(y_test, y_pred_lr)

print(f"Linear Regression Results (on Unseen Test Set):")
print(f"  MAE:  {mae_lr:.2f} bikes")
print(f"  RMSE: {rmse_lr:.2f} bikes")
print(f"  R²:   {r2_lr:.4f}")

# 6. Model 2: Support Vector Regression (SVR with RBF Kernel)
print("\n" + "=" * 50)
print("Training Model 2: Support Vector Regression (RBF)")
print("=" * 50)

svr_model = SVR(kernel='rbf', C=100.0, epsilon=10.0)
svr_model.fit(X_train_scaled, y_train)

# Predict on unseen future test set
y_pred_svr = svr_model.predict(X_test_scaled)
y_pred_svr = np.clip(y_pred_svr, 0, None)

mae_svr = mean_absolute_error(y_test, y_pred_svr)
rmse_svr = np.sqrt(mean_squared_error(y_test, y_pred_svr))
r2_svr = r2_score(y_test, y_pred_svr)

print(f"SVR Results (on Unseen Test Set):")
print(f"  MAE:  {mae_svr:.2f} bikes")
print(f"  RMSE: {rmse_svr:.2f} bikes")
print(f"  R²:   {r2_svr:.4f}")
