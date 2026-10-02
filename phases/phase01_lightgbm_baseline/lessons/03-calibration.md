# Lesson 03: Calibration

## Explain

**Why Calibration Matters in Fraud Detection**

Machine learning models often output poorly calibrated probabilities. A model might predict 0.8 for many transactions, but if only 5% are actually fraudulent, those predictions are overconfident.

Calibration ensures that when a model predicts probability p, the actual probability of fraud is close to p. This is critical for:

- **Risk scoring** — setting thresholds for manual review
- **Resource allocation** — prioritizing high-risk cases
- **Regulatory compliance** — demonstrating model reliability

**What is Isotonic Regression?**

Isotonic regression is a non-parametric calibration method that fits a monotonically increasing function to map raw predictions to calibrated probabilities. Unlike Platt scaling (logistic regression), isotonic regression makes no assumptions about the shape of the calibration curve.

Advantages:
- **Flexible** — adapts to any calibration pattern
- **Non-parametric** — no assumptions about distribution
- **Handles imbalanced data** — works well for fraud detection

## Code

```python
import numpy as np
from phase01_lightgbm_baseline.lightgbm_baseline import LightGBMWrapper
from phase01_lightgbm_baseline.data_loader import DataLoader

# Load data
loader = DataLoader()
train_df, test_df = loader.load_data(sample_limit=500)

# Prepare features
features = ["TransactionAmt", "time_since_creation", "n_prior_transactions"]
X_train = train_df[features].values
y_train = train_df["isFraud"].values
X_test = test_df[features].values
y_test = test_df["isFraud"].values

# Train model
model = LightGBMWrapper()
model.train(X_train, y_train, X_test, y_test)

# Get raw predictions
raw_probs = model.predict(X_test)

# Calibrate
calibrated_probs = model.calibrate(raw_probs, y_test)

print("Raw predictions:")
print(f"  Mean: {raw_probs.mean():.4f}")
print(f"  Std: {raw_probs.std():.4f}")

print("\nCalibrated predictions:")
print(f"  Mean: {calibrated_probs.mean():.4f}")
print(f"  Std: {calibrated_probs.std():.4f}")
```

## Visualize

```python
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve

# Calibration curve
fig, ax = plt.subplots(figsize=(8, 6))

# Raw predictions
prob_true_raw, prob_pred_raw = calibration_curve(y_test, raw_probs, n_bins=10)
ax.plot(prob_pred_raw, prob_true_raw, marker='o', label='Raw', linewidth=2)

# Calibrated predictions
prob_true_cal, prob_pred_cal = calibration_curve(y_test, calibrated_probs, n_bins=10)
ax.plot(prob_pred_cal, prob_true_cal, marker='s', label='Calibrated', linewidth=2)

# Perfect calibration
ax.plot([0, 1], [0, 1], '--', color='gray', label='Perfect calibration')

ax.set_xlabel('Mean predicted probability')
ax.set_ylabel('Fraction of positives')
ax.set_title('Calibration Curve')
ax.legend()
plt.tight_layout()
plt.show()
```

## Practice

**Exercise:** Try different smoothing parameters for isotonic regression.

The `IsotonicCalibrator` class has a `threshold` parameter. Try different values (0.3, 0.5, 0.7) and observe how they affect:
1. The calibration curve
2. The Brier score
3. The distribution of calibrated probabilities

## Solution

```python
from src.utils.calibration import IsotonicCalibrator
from sklearn.metrics import brier_score_loss

thresholds = [0.3, 0.5, 0.7]

for threshold in thresholds:
    calibrator = IsotonicCalibrator(threshold=threshold)
    calibrator.fit(raw_probs, y_test)
    cal_probs = calibrator.predict(raw_probs)
    
    brier = brier_score_loss(y_test, cal_probs)
    print(f"Threshold {threshold}: Brier score = {brier:.4f}")
```

**Observation:** The threshold parameter affects the decision boundary but not the calibration quality. Lower Brier scores indicate better calibration.
