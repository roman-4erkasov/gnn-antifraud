"""Tests for comparison utilities."""

import numpy as np
import pytest

from phase01_lightgbm_baseline.comparison import (
    compute_comparison_report,
    check_significance,
    generate_stop_decision,
)


class TestComputeComparisonReport:
    """Tests for compute_comparison_report function."""

    def test_compute_comparison_report_returns_dict(self):
        """Test that compute_comparison_report returns a dictionary."""
        minimal_metrics = {"pr_auc": 0.5, "roc_auc": 0.6}
        graph_metrics = {"pr_auc": 0.55, "roc_auc": 0.65}
        
        deltas = compute_comparison_report(minimal_metrics, graph_metrics)
        
        assert isinstance(deltas, dict)

    def test_compute_comparison_report_correct_deltas(self):
        """Test that deltas are computed correctly."""
        minimal_metrics = {"pr_auc": 0.5, "roc_auc": 0.6, "brier_score": 0.1}
        graph_metrics = {"pr_auc": 0.55, "roc_auc": 0.65, "brier_score": 0.08}
        
        deltas = compute_comparison_report(minimal_metrics, graph_metrics)
        
        assert deltas["delta_pr_auc"] == pytest.approx(0.05)
        assert deltas["delta_roc_auc"] == pytest.approx(0.05)
        assert deltas["delta_brier_score"] == pytest.approx(-0.02)

    def test_compute_comparison_report_negative_delta(self):
        """Test that negative deltas are computed correctly."""
        minimal_metrics = {"pr_auc": 0.6}
        graph_metrics = {"pr_auc": 0.55}
        
        deltas = compute_comparison_report(minimal_metrics, graph_metrics)
        
        assert deltas["delta_pr_auc"] == pytest.approx(-0.05)

    def test_compute_comparison_report_zero_delta(self):
        """Test that zero deltas are computed correctly."""
        metrics = {"pr_auc": 0.5, "roc_auc": 0.6}
        
        deltas = compute_comparison_report(metrics, metrics)
        
        assert deltas["delta_pr_auc"] == pytest.approx(0.0)
        assert deltas["delta_roc_auc"] == pytest.approx(0.0)


class TestCheckSignificance:
    """Tests for check_significance function."""

    def test_check_significance_returns_float(self):
        """Test that check_significance returns a float."""
        np.random.seed(42)
        probs_a = np.random.rand(100)
        probs_b = np.random.rand(100)
        
        p_value = check_significance(probs_a, probs_b)
        
        assert isinstance(p_value, float)

    def test_check_significance_p_value_range(self):
        """Test that p-value is in [0, 1] range."""
        np.random.seed(42)
        probs_a = np.random.rand(100)
        probs_b = np.random.rand(100)
        
        p_value = check_significance(probs_a, probs_b)
        
        assert 0.0 <= p_value <= 1.0

    def test_check_significance_identical_predictions(self):
        """Test that identical predictions give high p-value."""
        np.random.seed(42)
        probs = np.random.rand(100)
        
        p_value = check_significance(probs, probs, n_permutations=1000)
        
        # Identical predictions should give p-value close to 1
        assert p_value > 0.5

    def test_check_significance_different_predictions(self):
        """Test that very different predictions give low p-value."""
        np.random.seed(42)
        probs_a = np.random.rand(100)
        probs_b = probs_a + 0.5  # Significantly different
        
        p_value = check_significance(probs_a, probs_b, n_permutations=1000)
        
        # Very different predictions should give low p-value
        assert p_value < 0.05


class TestGenerateStopDecision:
    """Tests for generate_stop_decision function."""

    def test_generate_stop_decision_graph_adds_value(self):
        """Test decision when graph adds value."""
        decision = generate_stop_decision(
            delta_pr_auc=0.02,
            p_value=0.01,
        )
        
        assert decision == "graph adds value"

    def test_generate_stop_decision_minimal_value_low_delta(self):
        """Test decision when delta is too small."""
        decision = generate_stop_decision(
            delta_pr_auc=0.005,
            p_value=0.01,
        )
        
        assert decision == "graph adds minimal value"

    def test_generate_stop_decision_minimal_value_high_pvalue(self):
        """Test decision when p-value is too high."""
        decision = generate_stop_decision(
            delta_pr_auc=0.02,
            p_value=0.10,
        )
        
        assert decision == "graph adds minimal value"

    def test_generate_stop_decision_boundary_delta(self):
        """Test decision at delta threshold boundary."""
        decision = generate_stop_decision(
            delta_pr_auc=0.01,
            p_value=0.01,
        )
        
        # At threshold, should be "graph adds value"
        assert decision == "graph adds value"

    def test_generate_stop_decision_boundary_pvalue(self):
        """Test decision at p-value threshold boundary."""
        decision = generate_stop_decision(
            delta_pr_auc=0.02,
            p_value=0.05,
        )
        
        # At threshold, should be "graph adds value"
        assert decision == "graph adds value"

    def test_generate_stop_decision_custom_thresholds(self):
        """Test decision with custom thresholds."""
        decision = generate_stop_decision(
            delta_pr_auc=0.015,
            p_value=0.03,
            delta_threshold=0.02,
            significance_threshold=0.05,
        )
        
        # Delta < threshold, so minimal value
        assert decision == "graph adds minimal value"

    def test_generate_stop_decision_both_conditions_fail(self):
        """Test decision when both conditions fail."""
        decision = generate_stop_decision(
            delta_pr_auc=0.005,
            p_value=0.10,
        )
        
        assert decision == "graph adds minimal value"
