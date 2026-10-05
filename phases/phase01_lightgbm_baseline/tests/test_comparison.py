"""Tests for comparison and significance utilities (task 12.3)."""

import numpy as np
import pytest

from phase01_lightgbm_baseline.comparison import (
    compute_comparison_report,
    generate_stop_decision,
    test_significance as run_significance_test,
)


def test_compute_comparison_report_deltas():
    minimal = {"pr_auc": 0.30, "brier_score": 0.10}
    graph = {"pr_auc": 0.35, "brier_score": 0.08}
    deltas = compute_comparison_report(minimal, graph)
    assert deltas["delta_pr_auc"] == pytest.approx(0.05)
    assert deltas["delta_brier_score"] == pytest.approx(-0.02)


def test_test_significance_p_value_in_range():
    rng = np.random.RandomState(0)
    minimal = rng.rand(50)
    graph = minimal + rng.normal(0.01, 0.05, size=50)
    p_value = run_significance_test(minimal, graph, n_permutations=200, seed=1)
    assert 0.0 <= p_value <= 1.0


@pytest.mark.parametrize(
    "delta,p_value,expected",
    [
        (0.005, 0.01, "graph adds minimal value"),
        (0.02, 0.20, "graph adds minimal value"),
        (0.02, 0.01, "graph adds value"),
    ],
)
def test_generate_stop_decision(delta, p_value, expected):
    assert generate_stop_decision(delta, p_value) == expected


def test_stop_decision_boundaries():
    assert generate_stop_decision(0.01, 0.05) == "graph adds value"
    assert generate_stop_decision(0.0099, 0.05) == "graph adds minimal value"
