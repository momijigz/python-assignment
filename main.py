import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt
import seaborn as sns

# Loading data
df_train = pd.read_csv("train.csv")
df_ideal = pd.read_csv("ideal.csv")
df_test = pd.read_csv("test.csv")

# Melting the dataframe to long format suitable for sns lineplot
df_melted = df_train.melt(id_vars="x", var_name="Function", value_name="y")

# Plotting using Seaborn
sns.set(style="whitegrid")
plt.figure(figsize=(8, 5))
sns.lineplot(data=df_melted, x="x", y="y", hue="Function")

plt.title("Function Visualization (y1, y2, y3, y4 vs x)")
plt.xlabel("x")
plt.ylabel("Function values")
plt.legend(title="Function")
plt.tight_layout()
plt.show()


# Identifying columns
train_ys = [col for col in df_train.columns if col.startswith('y')]
ideal_ys = [col for col in df_ideal.columns if col.startswith('y')]

# Matching training functions to ideal functions using least squares
best_matches = {}
max_deviations = {}

for train_col in train_ys:
    min_error = float('inf')
    best_ideal_col = None
    for ideal_col in ideal_ys:
        error = np.sum((df_train[train_col] - df_ideal[ideal_col]) ** 2)
        if error < min_error:
            min_error = error
            best_ideal_col = ideal_col
    best_matches[train_col] = best_ideal_col
    max_dev = np.max(np.abs(df_train[train_col] - df_ideal[best_ideal_col]))
    max_deviations[best_ideal_col] = max_dev

print("Matched Ideal Functions:")
for k, v in best_matches.items():
    print(f"{k} → {v} (max deviation = {max_deviations[v]:.4f})")

# Matching test (x, y) to one of the 4 ideal functions
print("\n Test Results:\n")
print(f"{'x':>8} {'y':>12} {'Deviation':>12} {'Ideal Function':>15}")

# Creating list to hold test match results
matched_points = []
unmatched_points = []

# Classifying test points
for _, row in df_test.iterrows():
    x_test = row['x']
    y_test = row['y']

    matched = False
    best_func = None
    best_dev = float('inf')

    for ideal_col in best_matches.values():
        row_ideal = df_ideal.loc[np.isclose(df_ideal['x'], x_test)]
        if row_ideal.empty:
            continue
        y_ideal = row_ideal[ideal_col].values[0]
        deviation = abs(y_test - y_ideal)
        threshold = max_deviations[ideal_col] * sqrt(2)

        if deviation <= threshold and deviation < best_dev:
            best_func = ideal_col
            best_dev = deviation
            matched = True

    if matched:
        matched_points.append((x_test, y_test, best_func))
        print(f"{x_test:8.2f} {y_test:12.6f} {best_dev:12.6f} {best_func:>15}")
    else:
        unmatched_points.append((x_test, y_test))
        print(f"{x_test:8.2f} {y_test:12.6f} {'-'*12} {'No Match':>15}")

# Converting to DataFrames for plotting
df_matched = pd.DataFrame(matched_points, columns=['x', 'y', 'ideal_function'])
df_unmatched = pd.DataFrame(unmatched_points, columns=['x', 'y'])

# Plotting matched and unmatched points
plt.figure(figsize=(10, 6))
plt.scatter(df_matched['x'], df_matched['y'], c='green', label='Matched', alpha=0.7, marker='o')
plt.scatter(df_unmatched['x'], df_unmatched['y'], c='red', label='Unmatched', alpha=0.7, marker='x')

plt.xlabel("x")
plt.ylabel("y")
plt.title("Scatterplot: Matched vs. Unmatched Test Points")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()