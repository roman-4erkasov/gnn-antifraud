# Lesson 06: Comparison and Stop Decision

## Explain

**How to Compare Models**

When comparing a baseline model (tabular features) with an enhanced model (baseline + graph features), we need to answer:

1. **Does the enhanced model perform better?** — measured by delta metrics (e.g., delta PR-AUC)
2. **Is the improvement statistically significant?** — measured by p-value from a permutation test
3. **Is the improvement practically significant?** — does it justify the added complexity?

**PR-AUC Delta**

PR-AUC (Precision-Recall Area Under Curve) is the primary metric for imbalanced classification. The delta tells us the absolute improvement:

- `delta_pr_auc = pr_auc_graph - pr_auc_baseline`

A positive delta means the graph model is better.

**Statistical Significance**

We use a **paired permutation test** to determine if the observed difference is statistically significant:

1. Compute the observed difference in predictions
2. Randomly permute the model labels (graph vs baseline)
3. Recompute the difference for each permutation
4. The p-value is the fraction of permutations where the permuted difference >= observed difference

A p-value < 0.05 means the difference is unlikely due to chance.

**Stop Decision**

We stop investing in graph features if:
- `delta_pr_auc < 0.01` (less than 1% improvement), OR
- `p_value > 0.05` (not statistically significant)

Otherwise, graphs add value and we continue.

## Code

```python
from phase01_lightgbm_baseline.comparison import (
    compute_comparison_report,
    test_significance,
    generate_stop_decision,
)
from phase01_lightgbm_baseline.lightgbm_baseline import LightGBMWrapper
from phase01_lightgbm_baseline.graph_features import GraphFeatureExtractor
from phase01_lightgbm_baseline.data_loader import DataLoader

# Load data
loader = DataLoader()
train_df, test_df = loader.load_data(sample_limit=500)

# Prepare baseline features
baseline_features = ["TransactionAmt", "time_since_creation", "n_prior_transactions"]
X_train = train_df[baseline_features].values
y_train = train_df["isFraud"].values
X_test = test_df[baseline_features].values
y_test = test_df["isFraud"].values

# Train baseline model
baseline_model = LightGBMWrapper()
baseline_metrics = baseline_model.full_pipeline(X_train, y_train, X_test, y_train, X_test, y_test)

# Prepare graph features
extractor = GraphFeatureExtractor()
train_graph, test_graph = extractor.append_graph_features(train_df.copy(), test_df.copy())
graph_features = baseline_features + ["user_degree", "merchant_degree", "user_clustering"]
X_train_graph = train_graph[graph_features].values
X_test_graph = test_graph[graph_features].values

# Train graph model
graph_model = LightGBMWrapper()
graph_metrics = graph_model.full_pipeline(X_train_graph, y_train, X_test_graph, y_train, X_test_graph, y_test)

# Compare
deltas = compute_comparison_report(baseline_metrics, graph_metrics)
print("Delta metrics:")
for key, value in deltas.items():
    print(f"  {key}: {value:+.4f}")
```

## Visualize

```python
import matplotlib.pyplot as plt
import numpy as np

# Comparison bar chart
fig, ax = plt.subplots(figsize=(10, 6))

metrics = ["pr_auc", "roc_auc", "brier_score"]
baseline_vals = [baseline_metrics[m] for m in metrics]
graph_vals = [graph_metrics[m] for m in metrics]

x = np.arange(len(metrics))
width = 0.35

ax.bar(x - width/2, baseline_vals, width, label='Baseline', alpha=0.7)
ax.bar(x + width/2, graph_vals, width, label='Graph', alpha=0.7)

ax.set_ylabel('Metric Value')
ax.set_title('Model Comparison')
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.legend()
plt.tight_layout()
plt.show()
```

## Practice

**Exercise:** Change the delta threshold and see how the stop decision changes.

1. Run the comparison with default threshold (0.01)
2. Change the threshold to 0.005 and 0.02
3. Observe how the stop decision changes

What does this tell you about the sensitivity of the decision?

## Solution

```python
# Test significance
raw_probs_baseline = baseline_model.predict(X_test)
raw_probs_graph = graph_model.predict(X_test_graph)
p_value = test_significance(raw_probs_baseline, raw_probs_graph)

print(f"P-value: {p_value:.4f}")

# Different thresholds
thresholds = [0.005, 0.01, 0.02]
delta_pr_auc = deltas["delta_pr_auc"]

for threshold in thresholds:
    decision = generate_stop_decision(delta_pr_auc, p_value, delta_threshold=threshold)
    print(f"Threshold {threshold}: {decision}")
```

**Observation:** Lower thresholds make it easier to continue (graphs add value), while higher thresholds require larger improvements. The choice depends on the cost of graph feature computation vs. the expected benefit.
