"""Tests for graph feature extraction."""

import numpy as np
import pandas as pd
import pytest

from phase01_lightgbm_baseline.graph_features import GraphFeatureExtractor


class TestGraphFeatureExtractorInit:
    """Tests for GraphFeatureExtractor initialization."""

    def test_init_defaults(self):
        """Test initialization with default parameters."""
        extractor = GraphFeatureExtractor()
        assert extractor.user_col == "card1"
        assert extractor.merchant_col == "ProductCD"
        assert extractor.txn_col == "TransactionID"

    def test_init_custom_params(self):
        """Test initialization with custom parameters."""
        extractor = GraphFeatureExtractor(
            user_col="user_id",
            merchant_col="merchant_id",
            txn_col="txn_id",
        )
        assert extractor.user_col == "user_id"
        assert extractor.merchant_col == "merchant_id"
        assert extractor.txn_col == "txn_id"


class TestBuildBipartiteGraph:
    """Tests for bipartite graph construction."""

    @pytest.fixture
    def sample_transactions(self):
        """Generate sample transaction data."""
        return pd.DataFrame({
            "card1": [1, 1, 2, 2, 3, 3],
            "ProductCD": ["W", "H", "W", "C", "H", "S"],
            "TransactionID": range(6),
            "TransactionAmt": [100, 200, 150, 300, 250, 400],
        })

    def test_build_graph_has_nodes_and_edges(self, sample_transactions):
        """Test that built graph has nodes and edges."""
        extractor = GraphFeatureExtractor()
        G = extractor.build_bipartite_graph(sample_transactions)
        
        assert G.number_of_nodes() > 0
        assert G.number_of_edges() > 0

    def test_build_graph_correct_node_count(self, sample_transactions):
        """Test that graph has correct number of nodes."""
        extractor = GraphFeatureExtractor()
        G = extractor.build_bipartite_graph(sample_transactions)
        
        # 3 users + 4 merchants = 7 nodes
        assert G.number_of_nodes() == 7

    def test_build_graph_correct_edge_count(self, sample_transactions):
        """Test that graph has correct number of edges."""
        extractor = GraphFeatureExtractor()
        G = extractor.build_bipartite_graph(sample_transactions)
        
        # 6 transactions = 6 edges (some may be duplicates)
        assert G.number_of_edges() <= 6

    def test_build_graph_node_types(self, sample_transactions):
        """Test that nodes have correct type attributes."""
        extractor = GraphFeatureExtractor()
        G = extractor.build_bipartite_graph(sample_transactions)
        
        user_nodes = [n for n in G.nodes() if G.nodes[n].get("node_type") == "user"]
        merchant_nodes = [n for n in G.nodes() if G.nodes[n].get("node_type") == "merchant"]
        
        assert len(user_nodes) == 3
        assert len(merchant_nodes) == 4


class TestComputeNodeFeatures:
    """Tests for node feature computation."""

    @pytest.fixture
    def sample_graph(self):
        """Build a sample graph."""
        txn_df = pd.DataFrame({
            "card1": [1, 1, 2, 2, 3],
            "ProductCD": ["W", "H", "W", "C", "H"],
            "TransactionID": range(5),
        })
        extractor = GraphFeatureExtractor()
        return extractor.build_bipartite_graph(txn_df)

    def test_compute_node_features_returns_dict(self, sample_graph):
        """Test that compute_node_features returns a dictionary."""
        extractor = GraphFeatureExtractor()
        features = extractor.compute_node_features(sample_graph)
        
        assert isinstance(features, dict)
        assert len(features) == sample_graph.number_of_nodes()

    def test_compute_node_features_has_expected_keys(self, sample_graph):
        """Test that node features have expected keys."""
        extractor = GraphFeatureExtractor()
        features = extractor.compute_node_features(sample_graph)
        
        sample_node = list(features.keys())[0]
        node_feats = features[sample_node]
        
        assert "degree" in node_feats
        assert "clustering_coefficient" in node_feats
        assert "node_type" in node_feats
        assert "degree_centrality" in node_feats

    def test_compute_node_features_values_valid(self, sample_graph):
        """Test that node feature values are valid."""
        extractor = GraphFeatureExtractor()
        features = extractor.compute_node_features(sample_graph)
        
        for node, feats in features.items():
            assert feats["degree"] >= 0
            assert 0.0 <= feats["clustering_coefficient"] <= 1.0
            assert feats["node_type"] in ["user", "merchant"]
            assert 0.0 <= feats["degree_centrality"] <= 1.0


class TestDetectCommunities:
    """Tests for community detection."""

    @pytest.fixture
    def sample_graph(self):
        """Build a sample graph."""
        txn_df = pd.DataFrame({
            "card1": [1, 1, 2, 2, 3, 3, 4, 4],
            "ProductCD": ["W", "H", "W", "C", "H", "S", "W", "R"],
            "TransactionID": range(8),
        })
        extractor = GraphFeatureExtractor()
        return extractor.build_bipartite_graph(txn_df)

    def test_detect_communities_returns_dict(self, sample_graph):
        """Test that detect_communities returns a dictionary."""
        extractor = GraphFeatureExtractor()
        communities = extractor.detect_communities(sample_graph)
        
        assert isinstance(communities, dict)

    def test_detect_communities_integer_labels(self, sample_graph):
        """Test that community labels are integers."""
        extractor = GraphFeatureExtractor()
        communities = extractor.detect_communities(sample_graph)
        
        for node, label in communities.items():
            assert isinstance(label, int)

    def test_detect_communities_label_range(self, sample_graph):
        """Test that community labels are in valid range."""
        extractor = GraphFeatureExtractor()
        communities = extractor.detect_communities(sample_graph)
        
        labels = list(communities.values())
        assert min(labels) >= 0
        assert max(labels) < len(set(labels)) + 10  # Allow some buffer


class TestComputeRWSEFeatures:
    """Tests for RWSE feature computation."""

    @pytest.fixture
    def sample_graph(self):
        """Build a sample graph."""
        txn_df = pd.DataFrame({
            "card1": [1, 1, 2, 2, 3],
            "ProductCD": ["W", "H", "W", "C", "H"],
            "TransactionID": range(5),
        })
        extractor = GraphFeatureExtractor()
        return extractor.build_bipartite_graph(txn_df)

    def test_compute_rwse_returns_dict(self, sample_graph):
        """Test that compute_rwse_features returns a dictionary."""
        extractor = GraphFeatureExtractor()
        rwse = extractor.compute_rwse_features(sample_graph, n_walks=3, walk_length=2)
        
        assert isinstance(rwse, dict)
        assert len(rwse) == sample_graph.number_of_nodes()

    def test_compute_rwse_values_are_arrays(self, sample_graph):
        """Test that RWSE values are numpy arrays."""
        extractor = GraphFeatureExtractor()
        rwse = extractor.compute_rwse_features(sample_graph, n_walks=3, walk_length=2)
        
        for node, features in rwse.items():
            assert isinstance(features, np.ndarray)

    def test_compute_rwse_expected_shape(self, sample_graph):
        """Test that RWSE features have expected shape."""
        extractor = GraphFeatureExtractor()
        rwse = extractor.compute_rwse_features(sample_graph, n_walks=3, walk_length=2)
        
        sample_features = list(rwse.values())[0]
        assert sample_features.shape == (3,)  # [avg_length, avg_unique, avg_degree]


class TestAppendGraphFeatures:
    """Tests for appending graph features to DataFrames."""

    @pytest.fixture
    def sample_data(self):
        """Generate sample train/test data."""
        train_df = pd.DataFrame({
            "card1": [1, 1, 2, 2, 3, 3],
            "ProductCD": ["W", "H", "W", "C", "H", "S"],
            "TransactionID": range(6),
            "TransactionAmt": [100, 200, 150, 300, 250, 400],
            "time_since_creation": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            "n_prior_transactions": [0, 1, 0, 1, 0, 1],
            "isFraud": [0, 0, 0, 1, 0, 1],
        })
        
        test_df = pd.DataFrame({
            "card1": [1, 2, 3],
            "ProductCD": ["W", "H", "S"],
            "TransactionID": [6, 7, 8],
            "TransactionAmt": [120, 180, 350],
            "time_since_creation": [0.7, 0.8, 0.9],
            "n_prior_transactions": [2, 2, 2],
            "isFraud": [0, 0, 1],
        })
        
        return train_df, test_df

    def test_append_graph_features_increases_columns(self, sample_data):
        """Test that appending graph features increases column count."""
        train_df, test_df = sample_data
        extractor = GraphFeatureExtractor()
        
        original_cols = len(train_df.columns)
        train_graph, test_graph = extractor.append_graph_features(train_df, test_df)
        
        assert len(train_graph.columns) > original_cols
        assert len(test_graph.columns) > original_cols

    def test_append_graph_features_adds_expected_columns(self, sample_data):
        """Test that expected graph feature columns are added."""
        train_df, test_df = sample_data
        extractor = GraphFeatureExtractor()
        train_graph, test_graph = extractor.append_graph_features(train_df, test_df)
        
        expected_cols = [
            "user_degree", "user_clustering", "user_degree_centrality",
            "merchant_degree", "merchant_clustering", "merchant_degree_centrality",
        ]
        
        for col in expected_cols:
            assert col in train_graph.columns
            assert col in test_graph.columns

    def test_append_graph_features_no_nan_in_train(self, sample_data):
        """Test that train DataFrame has no NaN in graph features."""
        train_df, test_df = sample_data
        extractor = GraphFeatureExtractor()
        train_graph, _ = extractor.append_graph_features(train_df, test_df)
        
        graph_cols = ["user_degree", "merchant_degree", "user_clustering"]
        for col in graph_cols:
            if col in train_graph.columns:
                assert not train_graph[col].isna().any()
