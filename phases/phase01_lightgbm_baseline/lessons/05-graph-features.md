# Lesson 05 — Graph Features

## Explain

Once the graph is built we convert structure into **per-node features** that a
tabular model can consume. `GraphFeatureExtractor.compute_feature_table(graph)`
returns a table with one row per node:

- **degree / edge_count** — number of incident transaction edges. A merchant
  connected to many cards is high-risk; an active card has high out-degree.
- **clustering coefficient** — how tightly a node's neighbours interconnect.
  Fraud rings form dense local neighbourhoods.
- **degree centrality** — degree normalized by the maximum possible degree.
- **Louvain community label** — `community` (`python-louvain`) detects densely
  connected communities. Nodes in the same community share a fraud-risk context.
- **RWSE** — *Random-Walk Structural Embedding*: the diagonal of the k-step
  random-walk return probability matrix, `(P^k)[i, i]` for `k = 1..8`, where
  `P = D^-1 A` is the row-stochastic transition matrix. Each node gets an
  8-dimensional vector capturing its local structural role independent of node
  identity.

`append_graph_features(X_train, X_test, feature_table)` joins these onto the
transaction frames, producing `user_*` and `merchant_*` columns (missing nodes
filled with 0).

## Code

```python
import pandas as pd
import numpy as np

from phase01_lightgbm_baseline import GraphFeatureExtractor

rng = np.random.default_rng(0)
n = 300
txns = pd.DataFrame(
    {
        "TransactionID": np.arange(n),
        "card1": rng.integers(0, 60, size=n),
        "ProductCD": rng.choice(list("WCHRS"), size=n),
        "TransactionAmt": rng.lognormal(3, 1, size=n),
    }
)

extractor = GraphFeatureExtractor()
graph = extractor.build_bipartite_graph(txns)
table = extractor.compute_feature_table(graph)

print("feature table columns:", list(table.columns))
print(f"rows: {len(table)}")
print(table.head(4).to_string(index=False))

joined, _ = extractor.append_graph_features(txns, txns.copy(), table)
graph_cols = extractor.graph_feature_columns(table)
print(f"graph feature columns: {len(graph_cols)}")
print(f"joined shape: {joined.shape}")
```

## Visualize

```python
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

numeric = table.select_dtypes("number")
corr = numeric.corr(numeric_only=True)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
im = axes[0].imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
axes[0].set_xticks(range(len(corr)))
axes[0].set_xticklabels(corr.columns, rotation=90)
axes[0].set_yticks(range(len(corr)))
axes[0].set_yticklabels(corr.columns)
axes[0].set_title("Graph feature correlation")
fig.colorbar(im, ax=axes[0])

users = table[table["node_type"] == "user"]
axes[1].hist(users["community_label"], bins=max(2, users["community_label"].nunique()), color="tab:purple")
axes[1].set(xlabel="Community label", ylabel="Number of users", title="Community size distribution")
fig.tight_layout()
fig.savefig("lesson05_features.png", dpi=120)
print("saved lesson05_features.png")
```

## Practice

Group the feature table by `community_label` and compare mean degree and mean
`rwse_1` across communities. Do some communities look structurally different from
others?

Open **[../exercises/05-features-by-community.ipynb](../exercises/05-features-by-community.ipynb)**
for a guided aggregation.

## Solution

Community means typically differ — some communities are dense hubs (high mean
degree, high `rwse_1`) while others are sparse. This is exactly the signal the
graph-aware model can use: **community membership contextualizes a node's risk**
beyond its own transactions. **Takeaway:** graph features are most valuable when
they capture *relational* context that raw per-transaction features cannot.
