# Lesson 00 — Evaluation Metrics and Class Imbalance

## Explain

Fraud detection is a needle-in-a-haystack problem: in the IEEE-CIS dataset only
about **3.5%** of transactions are fraudulent. On data like this, **accuracy is a
trap**. A model that predicts "not fraud" for every single transaction is 96.5%
accurate and completely useless.

What we care about instead:

- **Precision** — of the transactions we flag, how many are truly fraud? High
  precision means few false alarms (fewer blocked good customers).
- **Recall** — of the true frauds, how many did we catch? High recall means fewer
  missed frauds.
- **PR-AUC** (area under the precision–recall curve) — summarizes the
  precision/recall trade-off across all thresholds. It is the **primary metric**
  for imbalanced problems because it only rewards performance on the positive
  (fraud) class.
- **ROC-AUC** — summarizes true-positive vs false-positive rate. It is useful but
  **over-optimistic** when the negative class dominates: the false-positive rate
  stays tiny even when many negatives are misclassified.
- **Precision@K** — precision among the **top-K** highest-scoring transactions.
  This mirrors the real operational setting: an analyst can only review the K
  riskiest cases. In this phase `K = ceil(0.01 * len(test))`, i.e. the top 1%.
- **Class weighting** — `scale_pos_weight = n_negatives / n_positives` tells
  LightGBM to treat each fraud as more important, counteracting the imbalance.

## Code

```python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    brier_score_loss,
    log_loss,
)

# A synthetic imbalanced dataset (~3.5% positives).
X, y = make_classification(
    n_samples=8000,
    n_features=10,
    n_informative=5,
    weights=[0.965, 0.035],
    flip_y=0.01,
    random_state=42,
)
print(f"Fraud rate: {y.mean():.4f}")

X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42
)

clf = LogisticRegression(class_weight="balanced", max_iter=1000)
clf.fit(X_tr, y_tr)
probs = clf.predict_proba(X_te)[:, 1]

k = max(1, int(np.ceil(0.01 * len(y_te))))
top_k = np.argsort(probs)[-k:]

print(f"PR-AUC          : {average_precision_score(y_te, probs):.4f}")
print(f"ROC-AUC         : {roc_auc_score(y_te, probs):.4f}")
print(f"Brier score     : {brier_score_loss(y_te, probs):.4f}")
print(f"Log-loss        : {log_loss(y_te, probs):.4f}")
print(f"Precision@{k:<4d}  : {y_te[top_k].mean():.4f}")
print(f"Accuracy        : {(clf.predict(X_te) == y_te).mean():.4f}")
```

## Visualize

```python
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, roc_curve, confusion_matrix

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

prec, rec, _ = precision_recall_curve(y_te, probs)
axes[0].plot(rec, prec, color="darkorange")
axes[0].axhline(y_te.mean(), ls="--", color="grey", label="fraud rate")
axes[0].set(xlabel="Recall", ylabel="Precision", title="Precision-Recall curve")
axes[0].legend()

fpr, tpr, _ = roc_curve(y_te, probs)
axes[1].plot(fpr, tpr, color="steelblue")
axes[1].plot([0, 1], [0, 1], ls="--", color="grey")
axes[1].set(xlabel="False Positive Rate", ylabel="True Positive Rate", title="ROC curve")

cm = confusion_matrix(y_te, (probs >= 0.5).astype(int))
im = axes[2].imshow(cm, cmap="Blues")
for (i, j), v in np.ndenumerate(cm):
    axes[2].text(j, i, str(v), ha="center", va="center")
axes[2].set(xlabel="Predicted", ylabel="Actual", title="Confusion matrix @0.5")
fig.colorbar(im, ax=axes[2])
fig.tight_layout()
fig.savefig("lesson00_curves.png", dpi=120)
print("saved lesson00_curves.png")
```

## Practice

Change the positive rate from `weights=[0.965, 0.035]` to a balanced
`weights=[0.5, 0.5]` and re-run. Observe how much **more** ROC-AUC and accuracy
move than PR-AUC. This is why PR-AUC is the metric we trust here.

Open the companion exercise **[../exercises/00-metrics-and-imbalance.ipynb](../exercises/00-metrics-and-imbalance.ipynb)**
to run this comparison systematically.

## Solution

When the classes are balanced, accuracy and ROC-AUC become meaningful and tend to
rise, while PR-AUC changes less dramatically because it was already focused on the
positive class. Under the original 3.5% rate, a trivially "all-negative" model
scores ~0.965 accuracy but a PR-AUC equal to the base rate — clearly worthless.
**Takeaway:** for imbalanced fraud detection, optimize PR-AUC and inspect
Precision@K, and always pair a metric with the confusion matrix at the operating
threshold.
