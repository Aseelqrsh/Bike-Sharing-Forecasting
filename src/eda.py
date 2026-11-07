import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load the dataset
df = pd.read_csv("data/hour.csv")

# 2. Print dataset dimensions
print("Data shape:", df.shape)

# 3. Check column names and missing values
print("\nColumns:", df.columns.tolist())
print("\nMissing values per column:\n", df.isnull().sum())

# 4. Define features (X) and target (y)
feature_cols = [
    'season', 'yr', 'mnth', 'hr', 'holiday', 
    'weekday', 'workingday', 'weathersit', 
    'temp', 'atemp', 'hum', 'windspeed'
]
target_col = 'cnt'

X = df[feature_cols]
y = df[target_col]
print(f"\nFeatures shape (X): {X.shape}")
print(f"Target shape (y): {y.shape}")

# 5. Compute summary statistics for numerical attributes & target
results_dir = os.path.join("assignment1", "results")
os.makedirs(results_dir, exist_ok=True)

summary_cols = ['temp', 'atemp', 'hum', 'windspeed', 'cnt']
stats_table = df[summary_cols].describe().T[['mean', 'std', 'min', '25%', '50%', '75%', 'max']]
stats_table.columns = ['Mean', 'Std Dev', 'Min', 'Q1 (25%)', 'Median (50%)', 'Q3 (75%)', 'Max']

print("\n=== Summary Statistics Table ===")
print(stats_table)

# Save to CSV for the report table
stats_table_path = os.path.join(results_dir, "dataset_summary_statistics.csv")
stats_table.to_csv(stats_table_path)
print(f"\nSaved summary table to: {stats_table_path}")

# 6. Generate Figure 1: Hourly Demand Profile (Working Day vs. Weekend)
sns.set_theme(style="whitegrid", font_scale=1.1)

hourly_pattern = df.groupby(['hr', 'workingday'])['cnt'].mean().reset_index()
hourly_pattern['Day Type'] = hourly_pattern['workingday'].map({
    1: 'Working Day (Commute Spikes: 8 AM & 5 PM)',
    0: 'Weekend / Holiday (Leisure Curve)'
})

plt.figure(figsize=(9, 5))
sns.lineplot(
    data=hourly_pattern,
    x='hr',
    y='cnt',
    hue='Day Type',
    marker='o',
    linewidth=2.5,
    palette=['#1f77b4', '#ff7f0e']
)

plt.title("Hourly Bike Rental Demand: Working Day vs. Weekend", fontsize=13, fontweight='bold', pad=12)
plt.xlabel("Hour of the Day (0 to 23)", fontsize=11)
plt.ylabel("Average Bike Rental Count (cnt)", fontsize=11)
plt.xticks(range(0, 24))
plt.legend(title="", frameon=True)
plt.tight_layout()

fig1_path = os.path.join(results_dir, "fig1_hourly_demand_profile.png")
plt.savefig(fig1_path, dpi=300)
plt.close()
print(f"Saved Figure 1 to: {fig1_path}")
