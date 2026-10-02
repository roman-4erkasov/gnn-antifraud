# Lesson 01: Introduction to LightGBM

## Explain

**What is LightGBM?**

LightGBM (Light Gradient Boosting Machine) is a fast, distributed, high-performance gradient boosting framework based on decision tree algorithms. It's designed for efficiency and scalability, making it ideal for large-scale machine learning tasks like fraud detection.

**Why Gradient Boosting?**

Gradient boosting builds an ensemble of weak learners (typically decision trees) sequentially. Each new tree corrects the errors of the previous ones, gradually improving prediction accuracy. Key advantages:

- **Handles imbalanced data well** — fraud detection typically has <5% positive cases
- **Feature importance** — tells you which features matter most
- **Robust to outliers** — trees are less sensitive to extreme values
- **No feature scaling needed** — works with raw features

**Why LightGBM for Fraud Detection?**

1. **Speed** — trains 10-20x faster than traditional GBDT
2. **Memory efficient** — uses histogram-based algorithms
3. **Handles categorical features** — no need for one-hot encoding
4. **Built-in early stopping** — prevents overfitting

## Code

```python
import lightgbm as lgb
import numpy as np
from sklearn.model_selection import train_test_split

# Generate synthetic fraud data
np.random.seed(42)
n_samples = 1000
n_features = 3

X = np.random.randn(n_samples, n_features)
y = np.random.binomial(1, 0.05, n_samples)  # 5% fraud rate

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Create LightGBM datasets
train_data = lgb.Dataset(X_train, label=y_train)
test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

# Set parameters
params = {
    "objective": "binary",
    "metric": "binary_logloss",
    "verbosity": -1,
    "scale_pos_weight": np.sum(y_train == 0) / np.sum(y_train == 1),
}

# Train model
model = lgb.train(
    params,
    train_data,
    num_boost_round=100,
    valid_sets=[test_data],
    callbacks=[lgb.early_stopping(10), lgb.log_evaluation(0)],
)

# Predict
predictions = model.predict(X_test)
print(f"Predictions range: [{predictions.min():.4f}, {predictions.max():.4f}]")
```

## Visualize

```python
import matplotlib.pyplot as plt

# Feature importance
lgb.plot_importance(model, max_num_features=10)
plt.title("Feature Importance")
plt.tight_layout()
plt.show()
```

## Practice

**Exercise:** Predict what happens when you change `n_estimators` (number of trees).

1. Train a model with `num_boost_round=50`
2. Train another model with `num_boost_round=200`
3. Compare the training time and prediction accuracy

What do you observe? How does the number of trees affect model performance?

## Solution

```python
import time

# Model with 50 trees
start = time.time()
model_50 = lgb.train(params, train_data, num_boost_round=50)
time_50 = time.time() - start
pred_50 = model_50.predict(X_test)

# Model with 200 trees
start = time.time()
model_200 = lgb.train(params, train_data, num_boost_round=200)
time_200 = time.time() - start
pred_200 = model_200.predict(X_test)

print(f"50 trees: {time_50:.3f}s, predictions mean: {pred_50.mean():.4f}")
print(f"200 trees: {time_200:.3f}s, predictions mean: {pred_200.mean():.4f}")
```

**Observation:** More trees generally improve accuracy but increase training time. Early stopping helps find the optimal number of trees automatically.
