# Lesson 03 — Probability Calibration

## Explain

A classifier's **raw scores** are not necessarily probabilities. LightGBM with the
`binary` objective plus `scale_pos_weight` deliberately inflates scores to catch
more fraud, so a raw score of `0.8` may not mean "80% chance of fraud". Downstream
systems (risk pricing, expected-loss models, threshold policies) need **calibrated
probabilities** where the score equals the true frequency.

**Isotonic regression** is a flexible, non-parametric calibrator. It fits a
monotonically non-decreasing step function mapping raw scores to observed
frequencies. It is stronger than Platt scaling when you have enough validation
data, at the cost of being less smooth.

Calibration is measured with **proper scoring rules**:

- **Brier score** — mean squared error of probabilities (lower is better).
- **Log-loss** — penalizes confident wrong probabilities (lower is better).

`LightGBMWrapper.calibrate(raw_probs, val_labels)` fits an `IsotonicCalibrator` on
validation data and returns calibrated probabilities. `full_pipeline` reports the
Brier/log-loss **deltas** between calibrated and raw scores.

## Code

```python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from phase01_lightgbm_baseline import LightGBMWrapper

X, y = make_classification(
    n_samples=6000,
    n_features=10,
    n_informative=5,
    weights=[0.965, 0.035],
    flip_y=0.02,
    random_state=7,
)
X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.4, stratify=y, random_state=7)
X_val, X_te, y_val, y_te = train_test_split(X_tmp, y_tmp, test_size=0.5, stratify=y_tmp, random_state=7)

model = LightGBMWrapper()
model.train(X_tr, y_tr, X_val, y_val)

raw_val = model.predict(X_val)
cal_val = model.calibrate(raw_val, y_val)          # fit isotonic on validation
raw_te = model.predict(X_te)
cal_te = model.calibrator.predict(raw_te)

print(f"raw  mean score    : {raw_te.mean():.4f}  (true rate {y_te.mean():.4f})")
print(f"cal  mean score    : {cal_te.mean():.4f}")
```

## Visualize

```python
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, log_loss

frac_raw, mean_raw = calibration_curve(y_te, raw_te, n_bins=10, strategy="quantile")
frac_cal, mean_cal = calibration_curve(y_te, cal_te, n_bins=10, strategy="quantile")

fig, ax = plt.subplots(figsize=(5, 5))
ax.plot([0, 1], [0, 1], ls="--", color="grey", label="perfect")
ax.plot(mean_raw, frac_raw, "o-", label=f"raw  (Brier {brier_score_loss(y_te, raw_te):.3f})")
ax.plot(mean_cal, frac_cal, "s-", label=f"cal  (Brier {brier_score_loss(y_te, cal_te):.3f})")
ax.set(xlabel="Predicted probability", ylabel="Observed fraud rate", title="Reliability diagram")
ax.legend()
fig.tight_layout()
fig.savefig("lesson03_calibration.png", dpi=120)
print("saved lesson03_calibration.png")
print(f"raw log-loss: {log_loss(y_te, np.clip(raw_te, 1e-9, 1 - 1e-9)):.4f}")
print(f"cal log-loss: {log_loss(y_te, np.clip(cal_te, 1e-9, 1 - 1e-9)):.4f}")
```

## Practice

Compare raw vs isotonic scores on **Brier score and log-loss**. Which improves
more? Why might log-loss improve less than the Brier score?

Open **[../exercises/03-raw-vs-isotonic.ipynb](../exercises/03-raw-vs-isotonic.ipynb)**
to sweep several seeds.

## Solution

Isotonic calibration pushes points toward the diagonal, so **Brier score typically
improves** (or stays flat). Log-loss can improve less — or occasionally worsen —
because isotonic regression can produce **hard 0/1 outputs** for extreme bins, and
log-loss strongly punishes a confident wrong 0/1 prediction. **Takeaway:**
calibration fixes the *meaning* of scores; always verify it on a held-out set
rather than assuming it helps every proper scoring rule.
