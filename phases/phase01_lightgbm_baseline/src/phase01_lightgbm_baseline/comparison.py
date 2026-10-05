"""Comparison utilities for model evaluation."""

import sys
from pathlib import Path
from typing import Dict

import numpy as np

_project_root = Path(__file__).resolve().parents[4]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.utils.metrics import compute_delta_metrics, paired_permutation_test  # noqa: E402


def compute_comparison_report(
    minimal_metrics: Dict[str, float],
    graph_metrics: Dict[str, float],
) -> Dict[str, float]:
    """Compute delta metrics (graph − minimal)."""
    return compute_delta_metrics(graph_metrics, minimal_metrics)


def test_significance(
    raw_probs_minimal: np.ndarray,
    raw_probs_graph: np.ndarray,
    n_permutations: int = 10000,
    seed: int = 42,
) -> float:
    """Return the p-value of a paired permutation test between the two models."""
    p_value, _ = paired_permutation_test(
        raw_probs_graph,
        raw_probs_minimal,
        n_permutations=n_permutations,
        seed=seed,
    )
    return p_value


def generate_stop_decision(
    delta_pr_auc: float,
    p_value: float,
    delta_threshold: float = 0.01,
    significance_threshold: float = 0.05,
) -> str:
    """Return the Phase 1 stop decision.

    Graphs are deemed to add minimal value when the PR-AUC improvement is below
    ``delta_threshold`` or the difference is not statistically significant
    (``p_value > significance_threshold``).
    """
    if delta_pr_auc < delta_threshold or p_value > significance_threshold:
        return "graph adds minimal value"
    return "graph adds value"
