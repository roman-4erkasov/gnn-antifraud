# Lesson 07 — Data Leakage and Splits

## Explain

**Data leakage** happens when information from the test set (or the future) sneaks
into training, producing metrics that look great but collapse in production.

Two leakage traps are especially relevant to Phase 01:

1. **Graph leakage.** Graph features (degree, community, RWSE) are computed from a
   transaction graph. If that graph is built on the **full** dataset — including
   validation/test transactions — then a test user's degree already reflects the
   test labels' structure. The correct procedure is to build the graph from the
   **training split only** and then join features onto validation/test
   transactions.

2. **Feature-normalization leakage.** `time_since_creation` is min-max normalized
   using the `TransactionDT` min/max. If the min/max is computed over the whole
   dataset before splitting, the training features encode global (test) time
   boundaries. Phase 01 avoids this with `fit_baseline_feature_params(train_df)`
   followed by `prepare_baseline_features(df, params)`, so the statistics come
   from the training split only (see the code below).

**Splits.** `split_train_val` uses a **random** 60/20/20 split (Design
Decision 13). For fraud, a **time-based** split is often more realistic because it
mimics deploying a model trained on the past to future transactions. Random splits
can overstate performance when fraud patterns drift over time, and they let
near-duplicate transactions appear in both train and test.

The pipeline helper `split_and_prepare` bundles the leak-free order: split first,
fit the feature parameters on the training split, then transform validation/test.
`train_graph_aware` additionally builds the graph on `train_split` only before
appending features.

## Code

```python
import numpy as np
import pandas as pd

from phase01_lightgbm_baseline import GraphFeatureExtractor

rng = np.random.default_rng(0)
n = 400
txns = pd.DataFrame(
    {
        "TransactionID": np.arange(n),
        "card1": rng.integers(0, 40, size=n),
        "ProductCD": rng.choice(list("WCHRS"), size=n),
        "TransactionDT": np.sort(rng.integers(0, 1_000_000, size=n)),
        "isFraud": (rng.random(n) < 0.04).astype(int),
    }
)

train_df = txns.iloc[:280].reset_index(drop=True)
test_df = txns.iloc[280:].reset_index(drop=True)

extractor = GraphFeatureExtractor()

# LEAK-FREE: graph built from the training split only.
graph_train = extractor.build_bipartite_graph(train_df)
table_train = extractor.compute_feature_table(graph_train)
_, test_clean = extractor.append_graph_features(train_df, test_df, table_train)

# LEAKING: graph built on the full dataset (train + test).
graph_full = extractor.build_bipartite_graph(txns)
table_full = extractor.compute_feature_table(graph_full)
_, test_leaky = extractor.append_graph_features(train_df, test_df, table_full)

col = "user_degree"
print(f"clean mean {col}: {test_clean[col].mean():.3f}")
print(f"leaky mean {col}: {test_leaky[col].mean():.3f}")
print(f"differing rows  : {(test_clean[col] != test_leaky[col]).sum()} / {len(test_df)}")

# --- Feature-normalization leakage ---
from src.utils.features import (
    fit_baseline_feature_params,
    prepare_baseline_features,
)

# LEAK-FREE: fit the TransactionDT min/max on the training split only.
params = fit_baseline_feature_params(train_df)
test_feat = prepare_baseline_features(test_df, params)

# LEAKING: fit on the full dataset (train + test).
leaky_params = fit_baseline_feature_params(txns)
test_feat_leaky = prepare_baseline_features(test_df, leaky_params)

print("clean train-fitted params:", params)
print("leaky full-data params  :", leaky_params)
print(
    "clean time_since_creation range:",
    (float(test_feat["time_since_creation"].min()),
     float(test_feat["time_since_creation"].max())),
)
print(
    "leaky time_since_creation range:",
    (float(test_feat_leaky["time_since_creation"].min()),
     float(test_feat_leaky["time_since_creation"].max())),
)
```

## Visualize

```python
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(7, 4))
ax.hist(test_clean[col], bins=20, alpha=0.6, label="leak-free (train graph)")
ax.hist(test_leaky[col], bins=20, alpha=0.6, label="leaky (full graph)")
ax.set(xlabel=col, ylabel="Test transactions", title="Graph-feature leakage")
ax.legend()
fig.tight_layout()
fig.savefig("lesson07_leakage.png", dpi=120)
print("saved lesson07_leakage.png")
```

## Practice

Compare a **random** split against a **time-ordered** split (sort by
`TransactionDT`, then cut). Train the same model on each and compare held-out
PR-AUC. Which is more pessimistic, and why might that be the honest number?

Open **[../exercises/07-leakage-and-splits.ipynb](../exercises/07-leakage-and-splits.ipynb)**
for a scripted comparison.

## Solution

The **time-ordered** split usually yields lower held-out PR-AUC because it forbids
the model from memorizing patterns that repeat across a random split and because
it must generalize across temporal drift. That lower number is often closer to
production reality. Building graph features on the full data inflates test degrees
(a form of leakage), so always construct the graph from the training split —
exactly what `train_graph_aware` does. **Takeaway:** decide the split *before*
feature engineering, and derive every statistic from the training portion only.
