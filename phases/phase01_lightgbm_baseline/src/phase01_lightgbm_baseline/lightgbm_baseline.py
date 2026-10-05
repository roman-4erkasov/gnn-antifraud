"""LightGBM wrapper for the fraud-detection baseline."""

import sys
from pathlib import Path
from typing import Dict, Optional

import lightgbm as lgb
import numpy as np

# Make the shared ``src.utils`` package importable when running from anywhere.
_project_root = Path(__file__).resolve().parents[4]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.utils.calibration import IsotonicCalibrator  # noqa: E402
from src.utils.metrics import compute_metrics  # noqa: E402


class LightGBMWrapper:
    """LightGBM wrapper with calibration and evaluation.

    Provides ``train``/``predict``/``calibrate`` with early stopping and
    isotonic-regression calibration. ``full_pipeline`` also measures the
    calibration impact (Brier score and log-loss before vs. after).
    """

    def __init__(
        self,
        scale_pos_weight: Optional[float] = None,
        n_estimators: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        random_state: int = 42,
    ) -> None:
        self.scale_pos_weight = scale_pos_weight
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.random_state = random_state

        self.model: Optional[lgb.Booster] = None
        self.best_iteration: Optional[int] = None
        self.calibrator: Optional[IsotonicCalibrator] = None

        # Populated by full_pipeline for downstream comparison/significance.
        self.raw_test_probs_: Optional[np.ndarray] = None
        self.calibrated_test_probs_: Optional[np.ndarray] = None

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> None:
        """Train LightGBM with early stopping on the validation set."""
        if self.scale_pos_weight is None:
            n_pos = int(np.sum(y_train == 1))
            n_neg = int(np.sum(y_train == 0))
            self.scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0

        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

        params = {
            "objective": "binary",
            "metric": "binary_logloss",
            "verbosity": -1,
            "scale_pos_weight": self.scale_pos_weight,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "seed": self.random_state,
        }

        callbacks = [
            lgb.early_stopping(stopping_rounds=10),
            lgb.log_evaluation(period=0),
        ]

        self.model = lgb.train(
            params,
            train_data,
            num_boost_round=self.n_estimators,
            valid_sets=[val_data],
            callbacks=callbacks,
        )
        self.best_iteration = self.model.best_iteration

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return raw probability predictions in ``[0, 1]``."""
        if self.model is None:
            raise RuntimeError("Model must be trained before predicting.")
        return self.model.predict(X)

    def calibrate(self, raw_probs: np.ndarray, val_labels: np.ndarray) -> np.ndarray:
        """Fit an isotonic calibrator on validation data and transform ``raw_probs``."""
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
        """Train → calibrate → predict → evaluate, including the calibration impact.

        Returns calibrated-probability metrics plus the raw/calibrated deltas:

        - ``pr_auc``, ``roc_auc``, ``brier_score``, ``log_loss`` (calibrated)
        - ``precision_at_k`` when ``k`` is provided
        - ``raw_brier_score``, ``raw_log_loss``
        - ``brier_score_delta``, ``log_loss_delta`` (calibrated − raw)
        """
        self.train(X_train, y_train, X_val, y_val)

        raw_val_probs = self.predict(X_val)
        self.calibrate(raw_val_probs, y_val)

        raw_test_probs = self.predict(X_test)
        calibrated_test_probs = self.calibrator.predict(raw_test_probs)

        self.raw_test_probs_ = raw_test_probs
        self.calibrated_test_probs_ = calibrated_test_probs

        metrics = compute_metrics(y_test, calibrated_test_probs, k=k)
        raw_metrics = compute_metrics(y_test, raw_test_probs, k=k)

        metrics["raw_brier_score"] = raw_metrics["brier_score"]
        metrics["raw_log_loss"] = raw_metrics["log_loss"]
        metrics["brier_score_delta"] = metrics["brier_score"] - raw_metrics["brier_score"]
        metrics["log_loss_delta"] = metrics["log_loss"] - raw_metrics["log_loss"]

        return metrics
