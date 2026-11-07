import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

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
