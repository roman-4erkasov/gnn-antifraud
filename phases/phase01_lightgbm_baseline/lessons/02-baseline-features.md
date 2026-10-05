# Lesson 02 — Baseline Features

## Explain

We deliberately start with the **smallest defensible model**: three features that
carry real signal and are cheap to compute. Everything later (graph features,
GNNs) must beat this reference point to justify its complexity.

The three baseline features (see `src/utils/features.py`):

1. **`TransactionAmt`** — the transaction amount. Fraudulent transactions often
   have unusual amounts relative to a customer's history.
2. **`time_since_creation`** — `TransactionDT` min-max normalized to `[0, 1]`.
   `TransactionDT` is a time offset (seconds from a reference), so it acts as a
   proxy for **account age / recency**: newer accounts are riskier.
3. **`n_prior_transactions`** — number of earlier transactions on the same card
   (`groupby(card1).cumcount()`). A burst of first-time activity on a card is
   suspicious.

`prepare_baseline_features(df, params)` returns a copy of the frame with the two
derived columns added. The `TransactionDT` min/max are fitted once on the
**training split** by `fit_baseline_feature_params(train_df)` and then applied to
validation/test, so the normalization never sees held-out rows.

## Code

```python
import pandas as pd

BASELINE_FEATURES = ["TransactionAmt", "time_since_creation", "n_prior_transactions"]


def fit_baseline_feature_params(df, card_col="card1"):
    """Mirror of src/utils/features.fit_baseline_feature_params (shown inline)."""
    dt = df["TransactionDT"]
    return {"dt_min": float(dt.min()), "dt_max": float(dt.max())}


def prepare_baseline_features(df, params=None, card_col="card1"):
    """Mirror of src/utils/features.prepare_baseline_features (shown inline).

    When ``params`` is omitted the normalization is fitted on ``df`` itself
    (fine for a single frame); callers that split data should fit on the
    training split and pass the result here.
    """
    out = df.copy()
    if params is None:
        params = fit_baseline_feature_params(df, card_col=card_col)
    dt = out["TransactionDT"]
    dt_min, dt_max = params["dt_min"], params["dt_max"]
    if dt_max > dt_min:
        out["time_since_creation"] = ((dt - dt_min) / (dt_max - dt_min)).clip(0.0, 1.0)
    else:
        out["time_since_creation"] = 0.0
    out["n_prior_transactions"] = out.groupby(card_col).cumcount()
    return out


raw = pd.DataFrame(
    {
        "TransactionID": [10, 11, 12, 13, 14],
        "card1": [100, 100, 100, 200, 200],
        "TransactionAmt": [50.0, 12.5, 900.0, 30.0, 30.0],
        "TransactionDT": [1000, 2000, 3000, 1500, 2500],
        "isFraud": [0, 0, 1, 0, 0],
    }
)

train = raw.iloc[:3]
params = fit_baseline_feature_params(train)
features = prepare_baseline_features(raw, params)
print("params fitted on train:", params)
print(features[["card1", *BASELINE_FEATURES, "isFraud"]].to_string(index=False))
```

## Visualize

```python
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

rng = np.random.default_rng(0)
n = 2000
cards = rng.integers(0, 200, size=n)
df = pd.DataFrame(
    {
        "TransactionAmt": rng.lognormal(mean=3.0, sigma=1.0, size=n),
        "card1": cards,
        "TransactionDT": np.sort(rng.integers(0, 1_000_000, size=n)),
    }
)
df = prepare_baseline_features(df)
df["isFraud"] = (rng.random(n) < 0.035).astype(int)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, col in zip(axes, BASELINE_FEATURES):
    ax.hist(df[df.isFraud == 0][col], bins=30, alpha=0.6, label="legit", density=True)
    ax.hist(df[df.isFraud == 1][col], bins=30, alpha=0.6, label="fraud", density=True)
    ax.set(title=col, ylabel="density")
    ax.legend()
fig.tight_layout()
fig.savefig("lesson02_features.png", dpi=120)
print("saved lesson02_features.png")
```

## Practice

Add a fourth feature — for example `log1p(TransactionAmt)` or a
`hour_of_day = (TransactionDT // 3600) % 24` column — and measure the change in
held-out PR-AUC. Does it help?

Open **[../exercises/02-add-4th-feature.ipynb](../exercises/02-add-4th-feature.ipynb)**
for a guided experiment.

## Solution

A well-chosen fourth feature (e.g. a cyclical time-of-day encoding) often yields a
small PR-AUC gain, but the gain must be weighed against added complexity and
leakage risk. A redundant transform such as `log1p(TransactionAmt)` usually adds
almost nothing because tree splits are invariant to monotonic transforms of a
single feature. **Takeaway:** features must add *new information*, not just
re-express existing ones.
