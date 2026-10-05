# Lesson 01 — Introduction to LightGBM

## Explain

**LightGBM** is a gradient-boosting framework that builds an ensemble of decision
trees sequentially. Each new tree is fitted to the residual errors of the current
ensemble, so the model keeps correcting its own mistakes.

Why gradient boosting suits fraud detection:

- **Tabular native** — transaction data is tabular, where boosted trees still beat
  deep learning.
- **Handles imbalance** — via `scale_pos_weight` and the `binary` objective.
- **Fast and memory-efficient** — histogram-based splits and leaf-wise growth
  train on millions of rows in seconds.
- **Early stopping** — we hold out a validation set and stop when the validation
  `binary_logloss` stops improving, preventing overfitting.

Key parameters in `LightGBMWrapper`:

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `n_estimators` | 100 | Maximum number of boosting rounds |
| `max_depth` | 6 | Maximum tree depth |
| `learning_rate` | 0.1 | Shrinkage applied to each tree |
| `scale_pos_weight` | auto | Negative/positive ratio for imbalance |
| `random_state` | 42 | Reproducibility seed |

## Code

```python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Import the phase wrapper (installed via `uv sync`).
from phase01_lightgbm_baseline import LightGBMWrapper

X, y = make_classification(
    n_samples=5000,
    n_features=12,
    n_informative=6,
    weights=[0.965, 0.035],
    flip_y=0.01,
    random_state=42,
)
X_tr, X_temp, y_tr, y_temp = train_test_split(
    X, y, test_size=0.4, stratify=y, random_state=42
)
X_val, X_te, y_val, y_te = train_test_split(
    X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
)

model = LightGBMWrapper(n_estimators=100, max_depth=6, learning_rate=0.1)
model.train(X_tr, y_tr, X_val, y_val)

print(f"best_iteration      : {model.best_iteration}")
print(f"scale_pos_weight    : {model.scale_pos_weight:.2f}")

probs = model.predict(X_te)
print(f"prediction range    : [{probs.min():.4f}, {probs.max():.4f}]")
```

## Visualize

```python
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

importances = model.model.feature_importance(importance_type="gain")
order = np.argsort(importances)[::-1]

fig, ax = plt.subplots(figsize=(8, 4))
ax.bar(range(len(importances)), importances[order], color="seagreen")
ax.set(xlabel="Feature index (ranked)", ylabel="Gain", title="LightGBM feature importance")
fig.tight_layout()
fig.savefig("lesson01_importance.png", dpi=120)
print("saved lesson01_importance.png")
```

## Practice

Predict what happens when you raise `n_estimators` from 100 to 200 and rerun the
`Code` block. Does `best_iteration` change? Does the model overfit the small
synthetic set?

Open **[../exercises/01-n-estimators.ipynb](../exercises/01-n-estimators.ipynb)**
to compare `n_estimators=50` vs `200` on held-out PR-AUC.

## Solution

With early stopping (`stopping_rounds=10`), raising `n_estimators` merely raises
the **ceiling** on boosting rounds; training stops once validation loss plateaus.
So `best_iteration` typically stays close to its earlier value and held-out PR-AUC
is nearly unchanged. The extra rounds only matter when the model is still
improving. **Takeaway:** early stopping makes `n_estimators` a budget, not a
hyper-parameter you must tune precisely.
