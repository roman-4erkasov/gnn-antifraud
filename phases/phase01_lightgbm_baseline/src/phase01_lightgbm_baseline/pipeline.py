"""Graph-aware training pipeline for Phase 01."""

import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from phase01_lightgbm_baseline.graph_features import GraphFeatureExtractor
from phase01_lightgbm_baseline.lightgbm_baseline import LightGBMWrapper

_project_root = Path(__file__).resolve().parents[4]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.utils.features import BASELINE_FEATURES  # noqa: E402


def split_train_val(
    df: pd.DataFrame,
    train_frac: float = 0.6,
    val_frac: float = 0.2,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split a DataFrame into train and validation portions.

    The 60/20/20 convention is completed by the separate test split held by the
    caller, so ``test_frac == 1 - train_frac - val_frac``.
    """
    n = len(df)
    n_train = int(train_frac * n)
    n_val = int(val_frac * n)
    train_split = df.iloc[:n_train].copy()
    val_split = df.iloc[n_train : n_train + n_val].copy()
    return train_split, val_split


def train_graph_aware(
    data: Tuple[pd.DataFrame, pd.DataFrame],
    graph_extractor: GraphFeatureExtractor,
    k: Optional[int] = None,
    wrapper: Optional[LightGBMWrapper] = None,
) -> Tuple[LightGBMWrapper, Dict[str, float]]:
    """Build graph features, train a graph-aware LightGBM, and evaluate it.

    Args:
        data: ``(train_df, test_df)`` as returned by ``DataLoader.load_data``.
        graph_extractor: Configured :class:`GraphFeatureExtractor`. The graph is
            built from the training split only, to avoid leakage.
        k: Optional Precision@K cutoff. Defaults to the top 1% of the test set.
        wrapper: Optional pre-configured :class:`LightGBMWrapper`.

    Returns:
        Tuple ``(model, metrics_dict)``. ``model.raw_test_probs_`` holds the raw
        test-set probabilities for downstream significance testing.
    """
    train_df, test_df = data
    train_split, val_split = split_train_val(train_df)

    graph = graph_extractor.build_bipartite_graph(train_split)
    feature_table = graph_extractor.compute_feature_table(graph)

    train_g, val_g = graph_extractor.append_graph_features(
        train_split, val_split, feature_table
    )
    _, test_g = graph_extractor.append_graph_features(
        train_split, test_df, feature_table
    )

    feature_cols = BASELINE_FEATURES + graph_extractor.graph_feature_columns(
        feature_table
    )

    X_train = train_g[feature_cols].to_numpy(dtype=float)
    y_train = train_g["isFraud"].to_numpy(dtype=int)
    X_val = val_g[feature_cols].to_numpy(dtype=float)
    y_val = val_g["isFraud"].to_numpy(dtype=int)
    X_test = test_g[feature_cols].to_numpy(dtype=float)
    y_test = test_g["isFraud"].to_numpy(dtype=int)

    if k is None:
        k = max(1, int(np.ceil(0.01 * len(y_test))))

    model = wrapper if wrapper is not None else LightGBMWrapper()
    metrics = model.full_pipeline(
        X_train, y_train, X_val, y_val, X_test, y_test, k=k
    )

    return model, metrics
