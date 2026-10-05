# Lesson 06 — Comparison and Stop Decision

## Explain

Adding graph features only makes sense if the graph-aware model **demonstrably
beats** the minimal baseline. Phase 01 defines a disciplined comparison:

1. **Delta metrics** — `compute_comparison_report(minimal, graph)` returns
   `delta_<metric>` = graph − minimal for every shared metric. The headline number
   is `delta_pr_auc`.
2. **Statistical significance** — a **paired permutation test**
   (`test_significance(minimal_probs, graph_probs)`) randomly flips the sign of
   each paired difference 10,000 times to build a null distribution, then reports
   the p-value: the fraction of permutations with an absolute mean difference at
   least as large as observed. This is a paired test, so it uses the fact that
   both models scored the *same* test transactions.
3. **Stop decision** — `generate_stop_decision(delta_pr_auc, p_value)` returns:
   - `"graph adds minimal value"` if `delta_pr_auc < 0.01` **or** `p_value > 0.05`
   - `"graph adds value"` otherwise.

The default thresholds are a 1-point PR-AUC improvement and a 5% significance
level. If graphs do not clear both bars, later phases (GNNs) have little reason to
continue.

## Code

```python
import numpy as np

from phase01_lightgbm_baseline import (
    compute_comparison_report,
    generate_stop_decision,
    test_significance,
)

rng = np.random.default_rng(42)
n_pos, n_neg = 40, 960

labels = np.concatenate([np.ones(n_pos), np.zeros(n_neg)]).astype(int)
minimal_probs = np.where(
    labels == 1,
    rng.uniform(0.3, 0.8, len(labels)),
    rng.uniform(0.0, 0.35, len(labels)),
)
graph_probs = np.clip(
    minimal_probs + labels * rng.normal(0.05, 0.03, len(labels)), 0, 1
)

from phase01_lightgbm_baseline.lightgbm_baseline import LightGBMWrapper  # noqa: F401
from src.utils.metrics import compute_metrics

minimal_metrics = compute_metrics(labels, minimal_probs, k=max(1, int(np.ceil(0.01 * len(labels)))))
graph_metrics = compute_metrics(labels, graph_probs, k=max(1, int(np.ceil(0.01 * len(labels)))))

deltas = compute_comparison_report(minimal_metrics, graph_metrics)
p_value = test_significance(minimal_probs, graph_probs, n_permutations=2000, seed=42)
decision = generate_stop_decision(deltas["delta_pr_auc"], p_value)

print(f"delta_pr_auc : {deltas['delta_pr_auc']:+.4f}")
print(f"p_value      : {p_value:.4f}")
print(f"decision     : {decision}")
```

## Visualize

```python
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

labels_ = list(deltas.keys())
values = list(deltas.values())

fig, ax = plt.subplots(figsize=(8, 4))
ax.barh(labels_, values, color="steelblue")
ax.axvline(0, color="black", linewidth=0.8)
ax.set_title("Graph-aware minus minimal (delta metrics)")
fig.tight_layout()
fig.savefig("lesson06_deltas.png", dpi=120)
print("saved lesson06_deltas.png")
```

## Practice

Re-run with `delta_threshold` set to `0.0` and then `0.05`. How does the stop
decision change? Which threshold would you defend to a stakeholder?

Open **[../exercises/06-delta-threshold.ipynb](../exercises/06-delta-threshold.ipynb)**
to sweep the threshold.

## Solution

With `delta_threshold=0.0` any positive, significant delta counts as "adds value".
With `0.05` the graph model must improve PR-AUC by a full 5 points — a high bar
that graph features on this data rarely clear. **Takeaway:** the threshold encodes
a **business decision** about the minimum improvement worth the added complexity;
it should be chosen by stakeholders, not by the analyst running the numbers.
