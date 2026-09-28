"""Shared calibration utilities for probabilistic model outputs."""

import numpy as np
from sklearn.isotonic import IsotonicRegression
from typing import Optional, Tuple


class IsotonicCalibrator:
    """Isotonic regression calibration wrapper.
    
    Fits on held-out validation data and transforms predictions to calibrated
    probabilities that minimize the Brier score.
    """

    def __init__(self, threshold: float = 0.5) -> None:
        self._ir: Optional[IsotonicRegression] = None
        self.threshold = threshold
        self.fitted_ = False

    def fit(self, raw_predictions: np.ndarray, labels: np.ndarray) -> "IsotonicCalibrator":
        """Fit isotonic regression on validation data.

        Args:
            raw_predictions: Model probability estimates (values clipped to [0, 1]).
            labels: Binary ground truth labels.

        Returns:
            Self for method chaining.
        """
        raw = np.asarray(raw_predictions, dtype=np.float64)
        y = np.asarray(labels, dtype=np.float64)
        clipped = np.clip(raw, 0.0, 1.0)
        self._ir = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        self._ir.fit(clipped, y)
        self.fitted_ = True
        return self

    def predict(self, raw_predictions: np.ndarray) -> np.ndarray:
        """Transform raw predictions through the fitted isotonic regression.

        Args:
            raw_predictions: Model probability estimates.

        Returns:
            Calibrated probabilities.

        Raises:
            RuntimeError: If calibrator has not been fitted yet.
        """
        if not self.fitted_:
            raise RuntimeError("Calibrator must be fitted before predicting.")
        raw = np.asarray(raw_predictions, dtype=np.float64)
        clipped = np.clip(raw, 0.0, 1.0)
        return self._ir.predict(clipped)

    def fit_predict(self, raw_predictions: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """Fit and predict in one step (useful for single-set evaluation).

        Args:
            raw_predictions: Model probability estimates.
            labels: Binary ground truth labels.

        Returns:
            Calibrated probabilities.
        """
        self.fit(raw_predictions, labels)
        return self.predict(raw_predictions)

    def get_prediction_intervals(
        self, raw_predictions: np.ndarray, coverage: float = 0.95
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute prediction intervals via conformal prediction.

        Args:
            raw_predictions: Model probability estimates.
            coverage: Target coverage level (e.g. 0.95 for 95% interval).

        Returns:
            Tuple of (lower_bounds, upper_bounds) arrays.
        """
        if not self.fitted_:
            raise RuntimeError("Calibrator must be fitted before computing intervals.")
        raw = np.asarray(raw_predictions, dtype=np.float64)
        abs_residuals = np.sort(np.abs(raw - np.clip(raw, 0.0, 1.0)))
        idx = int(np.ceil(coverage * len(abs_residuals)))
        idx = min(idx, len(abs_residuals) - 1)
        radius = abs_residuals[idx]
        calibrated = self.predict(raw)
        lower = np.clip(calibrated - radius, 0.0, 1.0)
        upper = np.clip(calibrated + radius, 0.0, 1.0)
        return lower, upper

    @property
    def is_fitted(self) -> bool:
        return self.fitted_
