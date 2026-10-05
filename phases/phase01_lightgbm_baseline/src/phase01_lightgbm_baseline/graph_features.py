"""Graph feature extraction for fraud detection."""

from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
import pandas as pd
from scipy import sparse


class GraphFeatureExtractor:
    """Extract graph-derived features from transaction data.

    Builds a bipartite user-merchant graph and computes node-level features:

    - degree (the number of incident transaction edges, also exposed as
      ``edge_count``), clustering coefficient, and degree centrality
    - Louvain community labels
    - RWSE: the diagonal of the k-step random-walk return probability matrix
      for k = 1..K (a K-dimensional vector per node)
    """

    def __init__(
        self,
        user_col: str = "card1",
        merchant_col: str = "ProductCD",
        txn_col: str = "TransactionID",
        rwse_dim: int = 8,
    ) -> None:
        self.user_col = user_col
        self.merchant_col = merchant_col
        self.txn_col = txn_col
        self.rwse_dim = rwse_dim

    def build_bipartite_graph(
        self,
        txn_df: pd.DataFrame,
        user_col: Optional[str] = None,
        merchant_col: Optional[str] = None,
    ) -> nx.Graph:
        """Build a bipartite user-merchant graph from transaction data."""
        user_col = user_col or self.user_col
        merchant_col = merchant_col or self.merchant_col

        graph = nx.Graph()

        users = txn_df[user_col].dropna().unique()
        merchants = txn_df[merchant_col].dropna().unique()
        graph.add_nodes_from(users, bipartite=0, node_type="user")
        graph.add_nodes_from(merchants, bipartite=1, node_type="merchant")

        for user, merchant in zip(txn_df[user_col], txn_df[merchant_col]):
            if pd.isna(user) or pd.isna(merchant):
                continue
            if graph.has_edge(user, merchant):
                graph[user][merchant]["weight"] += 1
            else:
                graph.add_edge(user, merchant, weight=1)

        return graph

    def compute_node_features(self, graph: nx.Graph) -> Dict:
        """Compute per-node features.

        Returns a mapping ``node -> feature dict`` with keys ``degree``,
        ``edge_count`` (equal to degree), ``clustering_coefficient``,
        ``node_type``, and ``degree_centrality``.
        """
        features: Dict = {}
        degrees = dict(graph.degree())
        clustering = nx.clustering(graph)
        centrality = nx.degree_centrality(graph)

        for node in graph.nodes():
            degree = degrees[node]
            features[node] = {
                "degree": degree,
                "edge_count": degree,
                "clustering_coefficient": clustering[node],
                "node_type": graph.nodes[node].get("node_type", "unknown"),
                "degree_centrality": centrality[node],
            }

        return features

    def detect_communities(
        self,
        graph: nx.Graph,
        node_type: str = "user",
    ) -> Dict:
        """Run Louvain community detection and return ``node -> label``."""
        import community as community_louvain

        partition = community_louvain.best_partition(graph)

        if node_type:
            partition = {
                node: label
                for node, label in partition.items()
                if graph.nodes[node].get("node_type") == node_type
            }

        return partition

    def compute_rwse_features(
        self,
        graph: nx.Graph,
        k_max: Optional[int] = None,
    ) -> Dict:
        """Compute RWSE = diagonal of k-step random-walk return probabilities.

        For the row-stochastic transition matrix ``P = D^-1 A`` the value for
        node ``i`` at step ``k`` is ``(P^k)[i, i]``. Returns a mapping
        ``node -> np.ndarray`` of shape ``(k_max,)``.

        Args:
            graph: NetworkX graph.
            k_max: Number of walk steps (defaults to ``self.rwse_dim``).
        """
        k_max = k_max or self.rwse_dim
        nodes = list(graph.nodes())
        n = len(nodes)
        if n == 0:
            return {}

        adjacency = nx.to_scipy_sparse_array(
            graph, nodelist=nodes, weight=None, dtype=float, format="csr"
        )
        degree = np.asarray(adjacency.sum(axis=1)).ravel()
        degree[degree == 0] = 1.0
        transition = (sparse.diags(1.0 / degree) @ adjacency).tocsr()

        out = np.zeros((n, k_max), dtype=float)
        power = transition.copy()
        for k in range(k_max):
            out[:, k] = power.diagonal()
            power = power @ transition

        return {node: out[i] for i, node in enumerate(nodes)}

    def compute_feature_table(self, graph: nx.Graph) -> pd.DataFrame:
        """Combine all node features into a single per-node table.

        Columns: ``node_id``, ``node_type``, ``degree``,
        ``clustering_coefficient``, ``degree_centrality``,
        ``community_label``, and ``rwse_1`` .. ``rwse_{rwse_dim}``.
        """
        node_features = self.compute_node_features(graph)
        communities = self.detect_communities(graph, node_type="user")
        rwse = self.compute_rwse_features(graph)

        rows: List[Dict] = []
        for node, feats in node_features.items():
            row = {
                "node_id": node,
                "node_type": feats["node_type"],
                "degree": feats["degree"],
                "clustering_coefficient": feats["clustering_coefficient"],
                "degree_centrality": feats["degree_centrality"],
                "community_label": (
                    communities.get(node, -1)
                    if feats["node_type"] == "user"
                    else -1
                ),
            }
            vector = rwse.get(node, np.zeros(self.rwse_dim))
            for i, value in enumerate(vector, start=1):
                row[f"rwse_{i}"] = float(value)
            rows.append(row)

        return pd.DataFrame(rows)

    @staticmethod
    def _base_feature_columns(graph_features: pd.DataFrame) -> List[str]:
        return [c for c in graph_features.columns if c not in ("node_id", "node_type")]

    def graph_feature_columns(self, graph_features: pd.DataFrame) -> List[str]:
        """Return the joined (prefixed) graph feature column names."""
        columns: List[str] = []
        for base in self._base_feature_columns(graph_features):
            columns.append(f"user_{base}")
            columns.append(f"merchant_{base}")
        return columns

    def append_graph_features(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        graph_features: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Join a node-feature table onto transaction DataFrames.

        Args:
            X_train: Training transactions.
            X_test: Test transactions.
            graph_features: Node-feature table from :meth:`compute_feature_table`.

        Returns:
            ``(X_train, X_test)`` with ``user_*`` and ``merchant_*`` columns.
        """
        user_col = self.user_col
        merchant_col = self.merchant_col
        base_cols = self._base_feature_columns(graph_features)

        user_features = graph_features[graph_features["node_type"] == "user"].copy()
        merchant_features = graph_features[
            graph_features["node_type"] == "merchant"
        ].copy()

        user_rename = {"node_id": user_col, **{c: f"user_{c}" for c in base_cols}}
        merchant_rename = {
            "node_id": merchant_col,
            **{c: f"merchant_{c}" for c in base_cols},
        }
        user_features = user_features.rename(columns=user_rename)
        merchant_features = merchant_features.rename(columns=merchant_rename)

        user_cols = [user_col] + [f"user_{c}" for c in base_cols]
        merchant_cols = [merchant_col] + [f"merchant_{c}" for c in base_cols]

        graph_feature_cols = [f"user_{c}" for c in base_cols] + [
            f"merchant_{c}" for c in base_cols
        ]

        def _join(df: pd.DataFrame) -> pd.DataFrame:
            joined = df.merge(
                user_features[user_cols], on=user_col, how="left"
            ).merge(merchant_features[merchant_cols], on=merchant_col, how="left")
            for col in graph_feature_cols:
                if col in joined.columns:
                    joined[col] = joined[col].fillna(0)
            return joined

        return _join(X_train), _join(X_test)
