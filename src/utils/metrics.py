"""Shared metrics for model evaluation and statistical comparison."""

import numpy as np
from typing import Dict, Tuple, Optional
from sklearn.metrics import (
    precision_recall_curve,
    average_precision_score,
    roc_auc_score,
    brier_score_loss,
    log_loss,
    precision_score,
    recall_score,
    classification_report,
)


def compute_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    k: Optional[int] = None,
) -> Dict[str, float]:
    """Compute all standard classification metrics.

    Args:
        labels: Binary ground truth labels (0 or 1).
        probabilities: Predicted probabilities for the positive class.
        k: If provided, compute Precision@K using top-k predictions.

    Returns:
        Dictionary of metric name -> value.
    """
    labels = np.asarray(labels, dtype=int)
    probs = np.asarray(probabilities, dtype=float)

    metrics: Dict[str, float] = {}

    # PR-AUC (primary metric for imbalanced data)
    pr_auc = average_precision_score(labels, probs)
    metrics["pr_auc"] = float(pr_auc)

    # ROC-AUC
    roc_auc = roc_auc_score(labels, probs)
    metrics["roc_auc"] = float(roc_auc)

    # Brier score (calibration metric)
    bs = brier_score_loss(labels, probs)
    metrics["brier_score"] = float(bs)

    # Log-loss
    ll = log_loss(labels, np.clip(probs, 1e-10, 1 - 1e-10))
    metrics["log_loss"] = float(ll)

    # Precision@K (if specified)
    if k is not None and k > 0:
        top_k_indices = np.argsort(probs)[-k:]
        precision_at_k = np.mean(labels[top_k_indices])
        metrics["precision_at_k"] = float(precision_at_k)

    return metrics


def compute_confusion_matrix_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Compute confusion-matrix-derived metrics.

    Args:
        labels: Binary ground truth.
        probabilities: Predicted probabilities.
        threshold: Classification threshold.

    Returns:
        Dictionary with tp, fp, tn, fn, precision, recall, f1.
    """
    labels = np.asarray(labels, dtype=int)
    probs = np.asarray(probabilities, dtype=float)
    preds = (probs >= threshold).astype(int)

    tp = int(np.sum((preds == 1) & (labels == 1)))
    fp = int(np.sum((preds == 1) & (labels == 0)))
    tn = int(np.sum((preds == 0) & (labels == 0)))
    fn = int(np.sum((preds == 0) & (labels == 1)))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def compute_delta_metrics(
    model_metrics: Dict[str, float],
    baseline_metrics: Dict[str, float],
) -> Dict[str, float]:
    """Compute delta (improvement) of model metrics over baseline.

    Args:
        model_metrics: Metrics dictionary for the candidate model.
        baseline_metrics: Metrics dictionary for the baseline.

    Returns:
        Dictionary of delta metric name -> value.
    """
    deltas: Dict[str, float] = {}
    for key in model_metrics:
        if key in baseline_metrics:
            deltas[f"delta_{key}"] = model_metrics[key] - baseline_metrics[key]
    return deltas


def paired_permutation_test(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    n_permutations: int = 10000,
    seed: int = 42,
) -> Tuple[float, np.ndarray]:
    """Perform a paired permutation test for statistical significance.

    Args:
        scores_a: Model A predictions (or scores) per sample.
        scores_b: Model B predictions (or scores) per sample.
        n_permutations: Number of permutation iterations.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (p_value, permutation_distribution).
    """
    rng = np.random.RandomState(seed)
    diff_obs = np.mean(scores_a - scores_b)

    n = len(scores_a)
    if n != len(scores_b):
        raise ValueError("scores_a and scores_b must have the same length.")

    stats = np.zeros(n_permutations)
    for i in range(n_permutations):
        signs = rng.choice([-1, 1], size=n)
        stats[i] = np.mean(signs * (scores_a - scores_b))

    p_value = np.mean(np.abs(stats) >= np.abs(diff_obs))
    return float(p_value), stats
