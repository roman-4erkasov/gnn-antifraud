"""Phase 01: LightGBM baseline for fraud detection with graph-derived features."""

from phase01_lightgbm_baseline.comparison import (
    compute_comparison_report,
    generate_stop_decision,
    test_significance,
)
from phase01_lightgbm_baseline.data_loader import DataLoader
from phase01_lightgbm_baseline.graph_features import GraphFeatureExtractor
from phase01_lightgbm_baseline.lightgbm_baseline import LightGBMWrapper
from phase01_lightgbm_baseline.pipeline import split_train_val, train_graph_aware

__version__ = "0.1.0"

__all__ = [
    "DataLoader",
    "GraphFeatureExtractor",
    "LightGBMWrapper",
    "compute_comparison_report",
    "generate_stop_decision",
    "split_train_val",
    "test_significance",
    "train_graph_aware",
]
