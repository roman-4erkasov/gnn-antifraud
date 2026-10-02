"""Tests for LightGBM baseline wrapper."""

import numpy as np
import pytest

from phase01_lightgbm_baseline.lightgbm_baseline import LightGBMWrapper


class TestLightGBMWrapperInit:
    """Tests for LightGBMWrapper initialization."""

    def test_init_defaults(self):
        """Test initialization with default parameters."""
        wrapper = LightGBMWrapper()
        assert wrapper.n_estimators == 100
        assert wrapper.max_depth == 6
        assert wrapper.learning_rate == 0.1
        assert wrapper.random_state == 42
        assert wrapper.scale_pos_weight is None
        assert wrapper.model is None
        assert wrapper.calibrator is None

    def test_init_custom_params(self):
        """Test initialization with custom parameters."""
        wrapper = LightGBMWrapper(
            scale_pos_weight=5.0,
            n_estimators=200,
            max_depth=8,
            learning_rate=0.05,
            random_state=123,
        )
        assert wrapper.scale_pos_weight == 5.0
        assert wrapper.n_estimators == 200
        assert wrapper.max_depth == 8
        assert wrapper.learning_rate == 0.05
        assert wrapper.random_state == 123


class TestLightGBMWrapperTrain:
    """Tests for LightGBMWrapper training."""

    @pytest.fixture
    def synthetic_data(self):
        """Generate synthetic data for testing."""
        np.random.seed(42)
        n_samples = 200
        n_features = 3
        
        X = np.random.randn(n_samples, n_features)
        y = np.random.binomial(1, 0.1, n_samples)
        
        # Split into train/val
        n_train = int(0.7 * n_samples)
        X_train, X_val = X[:n_train], X[n_train:]
        y_train, y_val = y[:n_train], y[n_train:]
        
        return X_train, y_train, X_val, y_val

    def test_train_sets_model(self, synthetic_data):
        """Test that training sets the model attribute."""
        X_train, y_train, X_val, y_val = synthetic_data
        wrapper = LightGBMWrapper()
        wrapper.train(X_train, y_train, X_val, y_val)
        
        assert wrapper.model is not None
        assert wrapper.best_iteration is not None
        assert wrapper.best_iteration > 0

    def test_train_computes_scale_pos_weight(self, synthetic_data):
        """Test that scale_pos_weight is computed if not provided."""
        X_train, y_train, X_val, y_val = synthetic_data
        wrapper = LightGBMWrapper()
        wrapper.train(X_train, y_train, X_val, y_val)
        
        assert wrapper.scale_pos_weight is not None
        assert wrapper.scale_pos_weight > 0


class TestLightGBMWrapperPredict:
    """Tests for LightGBMWrapper prediction."""

    @pytest.fixture
    def trained_model(self):
        """Train a model for testing."""
        np.random.seed(42)
        X = np.random.randn(200, 3)
        y = np.random.binomial(1, 0.1, 200)
        
        wrapper = LightGBMWrapper()
        wrapper.train(X[:140], y[:140], X[140:], y[140:])
        return wrapper, X[140:]

    def test_predict_returns_probabilities(self, trained_model):
        """Test that predict returns probabilities in [0, 1]."""
        wrapper, X_test = trained_model
        predictions = wrapper.predict(X_test)
        
        assert isinstance(predictions, np.ndarray)
        assert predictions.min() >= 0.0
        assert predictions.max() <= 1.0

    def test_predict_before_train_raises(self):
        """Test that predict before training raises RuntimeError."""
        wrapper = LightGBMWrapper()
        X = np.random.randn(10, 3)
        
        with pytest.raises(RuntimeError, match="Model must be trained"):
            wrapper.predict(X)


class TestLightGBMWrapperCalibrate:
    """Tests for LightGBMWrapper calibration."""

    @pytest.fixture
    def trained_model_with_predictions(self):
        """Train model and get predictions."""
        np.random.seed(42)
        X = np.random.randn(200, 3)
        y = np.random.binomial(1, 0.1, 200)
        
        wrapper = LightGBMWrapper()
        wrapper.train(X[:140], y[:140], X[140:], y[140:])
        raw_probs = wrapper.predict(X[140:])
        
        return wrapper, raw_probs, y[140:]

    def test_calibrate_changes_scores(self, trained_model_with_predictions):
        """Test that calibration changes prediction scores."""
        wrapper, raw_probs, y_val = trained_model_with_predictions
        calibrated_probs = wrapper.calibrate(raw_probs, y_val)
        
        assert isinstance(calibrated_probs, np.ndarray)
        assert calibrated_probs.min() >= 0.0
        assert calibrated_probs.max() <= 1.0
        # Calibrated should differ from raw
        assert not np.allclose(raw_probs, calibrated_probs)

    def test_calibrate_sets_calibrator(self, trained_model_with_predictions):
        """Test that calibration sets the calibrator attribute."""
        wrapper, raw_probs, y_val = trained_model_with_predictions
        wrapper.calibrate(raw_probs, y_val)
        
        assert wrapper.calibrator is not None
        assert wrapper.calibrator.is_fitted


class TestLightGBMWrapperFullPipeline:
    """Tests for LightGBMWrapper full pipeline."""

    @pytest.fixture
    def synthetic_data_split(self):
        """Generate synthetic data with train/val/test split."""
        np.random.seed(42)
        X = np.random.randn(300, 3)
        y = np.random.binomial(1, 0.1, 300)
        
        n_train = 180
        n_val = 60
        
        X_train, y_train = X[:n_train], y[:n_train]
        X_val, y_val = X[n_train:n_train + n_val], y[n_train:n_train + n_val]
        X_test, y_test = X[n_train + n_val:], y[n_train + n_val:]
        
        return X_train, y_train, X_val, y_val, X_test, y_test

    def test_full_pipeline_returns_metrics(self, synthetic_data_split):
        """Test that full_pipeline returns metrics dictionary."""
        X_train, y_train, X_val, y_val, X_test, y_test = synthetic_data_split
        
        wrapper = LightGBMWrapper()
        metrics = wrapper.full_pipeline(
            X_train, y_train,
            X_val, y_val,
            X_test, y_test,
        )
        
        assert isinstance(metrics, dict)
        assert "pr_auc" in metrics
        assert "roc_auc" in metrics
        assert "brier_score" in metrics
        assert "log_loss" in metrics

    def test_full_pipeline_with_k(self, synthetic_data_split):
        """Test that full_pipeline with k returns precision_at_k."""
        X_train, y_train, X_val, y_val, X_test, y_test = synthetic_data_split
        
        wrapper = LightGBMWrapper()
        metrics = wrapper.full_pipeline(
            X_train, y_train,
            X_val, y_val,
            X_test, y_test,
            k=10,
        )
        
        assert "precision_at_k" in metrics
        assert 0.0 <= metrics["precision_at_k"] <= 1.0

    def test_full_pipeline_metrics_values(self, synthetic_data_split):
        """Test that metrics have valid values."""
        X_train, y_train, X_val, y_val, X_test, y_test = synthetic_data_split
        
        wrapper = LightGBMWrapper()
        metrics = wrapper.full_pipeline(
            X_train, y_train,
            X_val, y_val,
            X_test, y_test,
        )
        
        assert 0.0 <= metrics["pr_auc"] <= 1.0
        assert 0.0 <= metrics["roc_auc"] <= 1.0
        assert 0.0 <= metrics["brier_score"] <= 1.0
        assert metrics["log_loss"] >= 0.0
