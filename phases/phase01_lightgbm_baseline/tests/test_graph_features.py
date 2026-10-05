"""Tests for the graph feature extractor (task 12.2)."""

import numpy as np
import pandas as pd

from phase01_lightgbm_baseline.graph_features import GraphFeatureExtractor
from src.utils.features import BASELINE_FEATURES, prepare_baseline_features


def test_build_bipartite_graph_adds_edges(synthetic_transactions):
    extractor = GraphFeatureExtractor()
    graph = extractor.build_bipartite_graph(synthetic_transactions)
    n_users = synthetic_transactions["card1"].nunique()
    n_merchants = synthetic_transactions["ProductCD"].nunique()
    assert graph.number_of_nodes() == n_users + n_merchants
    expected_edges = synthetic_transactions[["card1", "ProductCD"]].drop_duplicates().shape[0]
    assert graph.number_of_edges() == expected_edges


def test_compute_node_features_expected_keys(synthetic_transactions):
    extractor = GraphFeatureExtractor()
    graph = extractor.build_bipartite_graph(synthetic_transactions)
    features = extractor.compute_node_features(graph)
    assert len(features) == graph.number_of_nodes()
    for feats in features.values():
        for key in (
            "degree",
            "edge_count",
            "clustering_coefficient",
            "node_type",
            "degree_centrality",
        ):
            assert key in feats
        assert feats["edge_count"] == feats["degree"]


def test_detect_communities_returns_integer_labels(synthetic_transactions):
    extractor = GraphFeatureExtractor()
    graph = extractor.build_bipartite_graph(synthetic_transactions)
    partition = extractor.detect_communities(graph, node_type="user")
    assert len(partition) > 0
    labels = sorted(set(partition.values()))
    assert all(isinstance(label, (int, np.integer)) for label in labels)
    assert labels[0] >= 0
    assert max(labels) < len(labels)


def test_detect_communities_uses_user_projection():
    extractor = GraphFeatureExtractor()
    txn_df = pd.DataFrame(
        {
            "TransactionID": [1, 2, 3],
            "card1": ["u1", "u2", "u3"],
            "ProductCD": ["m1", "m1", "m2"],
        }
    )
    graph = extractor.build_bipartite_graph(txn_df)
    partition = extractor.detect_communities(graph, node_type="user")

    assert set(partition) == {"u1", "u2", "u3"}
    assert "m1" not in partition and "m2" not in partition
    assert partition["u1"] == partition["u2"]


def test_compute_rwse_shape(synthetic_transactions):
    extractor = GraphFeatureExtractor(rwse_dim=8)
    graph = extractor.build_bipartite_graph(synthetic_transactions)
    rwse = extractor.compute_rwse_features(graph)
    assert len(rwse) == graph.number_of_nodes()
    for vector in rwse.values():
        assert np.asarray(vector).shape == (8,)


def test_append_graph_features_increases_feature_count(synthetic_transactions):
    extractor = GraphFeatureExtractor()
    df = prepare_baseline_features(synthetic_transactions)
    train = df.iloc[:300].copy()
    test = df.iloc[300:].copy()

    graph = extractor.build_bipartite_graph(train)
    feature_table = extractor.compute_feature_table(graph)
    graph_cols = extractor.graph_feature_columns(feature_table)

    train_aug, test_aug = extractor.append_graph_features(train, test, feature_table)

    assert len(graph_cols) == 2 * (len(feature_table.columns) - 2)
    assert set(BASELINE_FEATURES).issubset(train_aug.columns)
    for col in graph_cols:
        assert col in train_aug.columns
        assert col in test_aug.columns
    assert len(BASELINE_FEATURES) + len(graph_cols) == 3 + len(graph_cols)
