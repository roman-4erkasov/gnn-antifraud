"""Comparison utilities for model evaluation."""

from typing import Dict, Tuple

import numpy as np
import sys
from pathlib import Path

# Add project root to path for utils import
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.metrics import compute_delta_metrics, paired_permutation_test


def compute_comparison_report(
    minimal_metrics: Dict[str, float],
    graph_metrics: Dict[str, float],
) -> Dict[str, float]:
    """Compute comparison report between minimal and graph-aware models.
    
    Args:
        minimal_metrics: Metrics dictionary for minimal model.
        graph_metrics: Metrics dictionary for graph-aware model.
        
    Returns:
        Dictionary of delta metrics (delta_pr_auc, delta_brier_score, etc.).
    """
    return compute_delta_metrics(graph_metrics, minimal_metrics)


def check_significance(
    raw_probs_minimal: np.ndarray,
    raw_probs_graph: np.ndarray,
    n_permutations: int = 10000,
    seed: int = 42,
) -> float:
    """Check statistical significance of model difference.
    
    Args:
        raw_probs_minimal: Predictions from minimal model.
        raw_probs_graph: Predictions from graph-aware model.
        n_permutations: Number of permutation iterations.
        seed: Random seed for reproducibility.
        
    Returns:
        P-value from paired permutation test.
    """
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
    """Generate stop decision based on comparison criteria.
    
    Args:
        delta_pr_auc: Improvement in PR-AUC.
        p_value: P-value from statistical test.
        delta_threshold: Minimum delta threshold (default 0.01).
        significance_threshold: Maximum p-value threshold (default 0.05).
        
    Returns:
        Decision string: "graph adds value" or "graph adds minimal value".
    """
    if delta_pr_auc < delta_threshold or p_value > significance_threshold:
        return "graph adds minimal value"
    else:
        return "graph adds value"
