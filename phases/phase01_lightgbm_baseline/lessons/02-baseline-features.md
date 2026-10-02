# Lesson 02: Baseline Features

## Explain

**Why 3 Features?**

For our baseline, we use exactly 3 features to establish a minimal but meaningful starting point:

1. **TransactionAmt** — transaction amount. Fraudulent transactions often have unusual amounts (very high or very low).

2. **time_since_creation** — time since account creation (normalized). New accounts are higher risk; fraudsters often create accounts and immediately transact.

3. **n_prior_transactions** — number of prior transactions by the same card. Low counts indicate new/risky cards; high counts suggest established legitimate usage.

**Why Minimal?**

Starting with minimal features helps us:
- Understand the baseline performance ceiling
- Measure the incremental value of adding complexity (graph features)
- Avoid overfitting to noise in the training data

## Code

```python
import pandas as pd
import numpy as np

# Load sample data
from phase01_lightgbm_baseline.data_loader import DataLoader

loader = DataLoader()
train_df, test_df = loader.load_data(sample_limit=500)

# Check baseline features
baseline_features = ["TransactionAmt", "time_since_creation", "n_prior_transactions"]
print("Baseline features:")
print(train_df[baseline_features].describe())

# Fraud rate by feature
print("\nFraud rate by TransactionAmt quartile:")
train_df["amt_quartile"] = pd.qcut(train_df["TransactionAmt"], 4, labels=False)
print(train_df.groupby("amt_quartile")["isFraud"].mean())
```

## Visualize

```python
import matplotlib.pyplot as plt
import seaborn as sns

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Transaction amount distribution
sns.histplot(data=train_df, x="TransactionAmt", hue="isFraud", 
             bins=50, ax=axes[0], palette="Set1")
axes[0].set_title("Transaction Amount by Fraud")
axes[0].set_xscale("log")

# Time since creation
sns.histplot(data=train_df, x="time_since_creation", hue="isFraud",
             bins=20, ax=axes[1], palette="Set1")
axes[1].set_title("Time Since Creation")

# Prior transactions
sns.histplot(data=train_df, x="n_prior_transactions", hue="isFraud",
             bins=20, ax=axes[2], palette="Set1")
axes[2].set_title("Prior Transactions")

plt.tight_layout()
plt.show()
```

## Practice

**Exercise:** Add a 4th feature and measure its impact.

Create a new feature `dist_ratio = dist1 / (dist2 + 1)` and measure how it affects model performance.

1. Add the feature to the baseline
2. Train the model with 4 features
3. Compare PR-AUC with the 3-feature baseline

Does the 4th feature improve performance?

## Solution

```python
from phase01_lightgbm_baseline.lightgbm_baseline import LightGBMWrapper
from sklearn.model_selection import train_test_split

# Prepare data
train_df["dist_ratio"] = train_df["dist1"] / (train_df["dist2"] + 1)
test_df["dist_ratio"] = test_df["dist1"] / (test_df["dist2"] + 1)

# 3-feature baseline
features_3 = ["TransactionAmt", "time_since_creation", "n_prior_transactions"]
X_train_3 = train_df[features_3].values
y_train = train_df["isFraud"].values
X_test_3 = test_df[features_3].values
y_test = test_df["isFraud"].values

model_3 = LightGBMWrapper()
metrics_3 = model_3.full_pipeline(X_train_3, y_train, X_test_3, y_train, X_test_3, y_test)

# 4-feature model
features_4 = features_3 + ["dist_ratio"]
X_train_4 = train_df[features_4].values
X_test_4 = test_df[features_4].values

model_4 = LightGBMWrapper()
metrics_4 = model_4.full_pipeline(X_train_4, y_train, X_test_4, y_train, X_test_4, y_test)

print(f"3 features PR-AUC: {metrics_3['pr_auc']:.4f}")
print(f"4 features PR-AUC: {metrics_4['pr_auc']:.4f}")
print(f"Improvement: {metrics_4['pr_auc'] - metrics_3['pr_auc']:+.4f}")
```
