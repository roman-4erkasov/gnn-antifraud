"""LightGBM wrapper for fraud detection baseline."""

from typing import Dict, Optional

import lightgbm as lgb
import numpy as np
import sys
from pathlib import Path

# Add project root to path for utils import
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.calibration import IsotonicCalibrator
from src.utils.metrics import compute_metrics


class LightGBMWrapper:
    """LightGBM wrapper with calibration and evaluation.
    
    Provides train/predict/calibrate methods with early stopping and
    isotonic regression calibration.
    """

    def __init__(
        self,
        scale_pos_weight: Optional[float] = None,
        n_estimators: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        random_state: int = 42,
    ) -> None:
        """Initialize LightGBM wrapper.
        
        Args:
            scale_pos_weight: Weight for positive class (auto-computed if None).
            n_estimators: Maximum number of trees.
            max_depth: Maximum tree depth.
            learning_rate: Learning rate for boosting.
            random_state: Random seed for reproducibility.
        """
        self.scale_pos_weight = scale_pos_weight
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.random_state = random_state
        
        self.model: Optional[lgb.Booster] = None
        self.best_iteration: Optional[int] = None
        self.calibrator: Optional[IsotonicCalibrator] = None

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> None:
        """Train LightGBM model with early stopping.
        
        Args:
            X_train: Training features.
            y_train: Training labels.
            X_val: Validation features (for early stopping).
            y_val: Validation labels.
        """
        # Compute scale_pos_weight if not provided
        if self.scale_pos_weight is None:
            n_pos = np.sum(y_train == 1)
            n_neg = np.sum(y_train == 0)
            self.scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0
        
        # Create LightGBM datasets
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        
        # Set parameters
        params = {
            "objective": "binary",
            "metric": "binary_logloss",
            "verbosity": -1,
            "scale_pos_weight": self.scale_pos_weight,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "seed": self.random_state,
        }
        
        # Train with early stopping
        callbacks = [lgb.early_stopping(stopping_rounds=10), lgb.log_evaluation(period=0)]
        
        self.model = lgb.train(
            params,
            train_data,
            num_boost_round=self.n_estimators,
            valid_sets=[val_data],
            callbacks=callbacks,
        )
        
        self.best_iteration = self.model.best_iteration

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities for samples.
        
        Args:
            X: Feature matrix.
            
        Returns:
            Array of predicted probabilities in [0, 1].
        """
        if self.model is None:
            raise RuntimeError("Model must be trained before predicting.")
        
        return self.model.predict(X)

    def calibrate(
        self,
        raw_probs: np.ndarray,
        val_labels: np.ndarray,
    ) -> np.ndarray:
        """Calibrate raw probabilities using isotonic regression.
        
        Args:
            raw_probs: Raw probability predictions.
            val_labels: Validation labels for calibration.
            
        Returns:
            Calibrated probabilities.
        """
        self.calibrator = IsotonicCalibrator()
        self.calibrator.fit(raw_probs, val_labels)
        return self.calibrator.predict(raw_probs)

    def full_pipeline(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        k: Optional[int] = None,
    ) -> Dict[str, float]:
        """Run full pipeline: train -> calibrate -> predict -> evaluate.
        
        Args:
            X_train: Training features.
            y_train: Training labels.
            X_val: Validation features (for calibration).
            y_val: Validation labels.
            X_test: Test features.
            y_test: Test labels.
            k: If provided, compute Precision@K.
            
        Returns:
            Dictionary of metrics: pr_auc, roc_auc, brier_score, log_loss, precision_at_k.
        """
        # Train model
        self.train(X_train, y_train, X_val, y_val)
        
        # Get raw predictions on validation set for calibration
        raw_val_probs = self.predict(X_val)
        
        # Calibrate on validation set
        self.calibrate(raw_val_probs, y_val)
        
        # Get calibrated predictions on test set
        raw_test_probs = self.predict(X_test)
        calibrated_probs = self.calibrator.predict(raw_test_probs)
        
        # Compute metrics
        metrics = compute_metrics(y_test, calibrated_probs, k=k)
        
        return metrics
