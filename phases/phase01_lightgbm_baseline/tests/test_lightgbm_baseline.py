"""Tests for the LightGBM wrapper (task 12.1)."""

import numpy as np

from phase01_lightgbm_baseline.lightgbm_baseline import LightGBMWrapper


def test_init_defaults():
    wrapper = LightGBMWrapper()
    assert wrapper.scale_pos_weight is None
    assert wrapper.n_estimators == 100
    assert wrapper.max_depth == 6
    assert wrapper.learning_rate == 0.1
    assert wrapper.random_state == 42
    assert wrapper.model is None
    assert wrapper.best_iteration is None


def test_train_sets_best_iteration(synthetic_features):
    X_train, y_train, X_val, y_val, _, _ = synthetic_features
    wrapper = LightGBMWrapper(n_estimators=30)
    wrapper.train(X_train, y_train, X_val, y_val)
    assert wrapper.model is not None
    assert wrapper.best_iteration is not None
    assert wrapper.best_iteration >= 0


def test_predict_returns_probabilities_in_unit_interval(synthetic_features):
    X_train, y_train, X_val, y_val, X_test, _ = synthetic_features
    wrapper = LightGBMWrapper(n_estimators=30)
    wrapper.train(X_train, y_train, X_val, y_val)
    probs = wrapper.predict(X_test)
    assert probs.shape == (len(X_test),)
    assert np.all(probs >= 0.0) and np.all(probs <= 1.0)


def test_calibrate_changes_scores(synthetic_features):
    X_train, y_train, X_val, y_val, _, _ = synthetic_features
    wrapper = LightGBMWrapper(n_estimators=30)
    wrapper.train(X_train, y_train, X_val, y_val)
    raw = wrapper.predict(X_val)
    calibrated = wrapper.calibrate(raw, y_val)
    assert calibrated.shape == raw.shape
    assert not np.allclose(raw, calibrated)


def test_full_pipeline_returns_required_metric_keys(synthetic_features):
    X_train, y_train, X_val, y_val, X_test, y_test = synthetic_features
    wrapper = LightGBMWrapper(n_estimators=30)
    metrics = wrapper.full_pipeline(
        X_train, y_train, X_val, y_val, X_test, y_test, k=5
    )
    for key in ("pr_auc", "roc_auc", "brier_score", "log_loss", "precision_at_k"):
        assert key in metrics
    for key in ("raw_brier_score", "raw_log_loss", "brier_score_delta", "log_loss_delta"):
        assert key in metrics
    assert wrapper.raw_test_probs_ is not None
    assert wrapper.calibrated_test_probs_ is not None
