"""Tests for the graph-aware pipeline entrypoints (tasks 12.5, 14.2)."""

import numpy as np
import pandas as pd

from phase01_lightgbm_baseline.graph_features import GraphFeatureExtractor
from phase01_lightgbm_baseline.lightgbm_baseline import LightGBMWrapper
from phase01_lightgbm_baseline.pipeline import (
    split_and_prepare,
    split_train_val,
    train_graph_aware,
)
from src.utils.features import (
    BASELINE_FEATURES,
    fit_baseline_feature_params,
    prepare_baseline_features,
)


def test_split_train_val_sizes(synthetic_transactions):
    df = synthetic_transactions
    train, val = split_train_val(df)
    assert len(train) == int(0.6 * len(df))
    assert len(val) == int(0.2 * len(df))


def test_split_and_prepare_is_leakage_free(synthetic_transactions):
    train_df = synthetic_transactions.iloc[:300].copy()
    test_df = synthetic_transactions.iloc[300:].copy()

    train_a, val_a, test_a, params_a = split_and_prepare((train_df, test_df))

    train_dt = train_df["TransactionDT"]
    pert_down = test_df.copy()
    pert_down["TransactionDT"] = np.linspace(
        float(train_dt.min()), float(train_dt.max()), len(pert_down)
    )
    train_b, val_b, test_b, params_b = split_and_prepare((train_df, pert_down))

    assert params_a == params_b
    assert params_a == fit_baseline_feature_params(split_train_val(train_df)[0])
    pd.testing.assert_frame_equal(train_a, train_b)
    pd.testing.assert_frame_equal(val_a, val_b)
    assert not np.allclose(
        test_a["time_since_creation"].to_numpy(),
        test_b["time_since_creation"].to_numpy(),
    )


def test_train_graph_aware_returns_model_and_metrics(synthetic_transactions):
    train_df = synthetic_transactions.iloc[:300].copy()
    test_df = synthetic_transactions.iloc[300:].copy()

    extractor = GraphFeatureExtractor()
    wrapper = LightGBMWrapper(n_estimators=30)
    model, metrics = train_graph_aware(
        (train_df, test_df), extractor, k=3, wrapper=wrapper
    )

    assert isinstance(model, LightGBMWrapper)
    assert model.model is not None
    assert model.raw_test_probs_ is not None
    for key in ("pr_auc", "roc_auc", "brier_score", "log_loss", "precision_at_k"):
        assert key in metrics
    for feature in BASELINE_FEATURES:
        assert feature in prepare_baseline_features(train_df).columns
