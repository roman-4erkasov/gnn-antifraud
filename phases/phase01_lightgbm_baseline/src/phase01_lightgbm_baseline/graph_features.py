"""Graph feature extraction for fraud detection."""

from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
import pandas as pd
from scipy import sparse


class GraphFeatureExtractor:
    """Extract graph-derived features from transaction data.
    
    Builds a bipartite user-merchant graph and computes node-level features:
    - Degree, clustering coefficient
    - Community labels (Louvain)
    - Random walk structural embeddings (RWSE)
    """

    def __init__(
        self,
        user_col: str = "card1",
        merchant_col: str = "ProductCD",
        txn_col: str = "TransactionID",
    ) -> None:
        self.user_col = user_col
        self.merchant_col = merchant_col
        self.txn_col = txn_col

    def build_bipartite_graph(
        self,
        txn_df: pd.DataFrame,
        user_col: Optional[str] = None,
        merchant_col: Optional[str] = None,
    ) -> nx.Graph:
        """Build bipartite graph from transaction data.
        
        Args:
            txn_df: Transaction DataFrame.
            user_col: Column name for user identifier.
            merchant_col: Column name for merchant identifier.
            
        Returns:
            NetworkX bipartite graph with users and merchants as nodes.
        """
        user_col = user_col or self.user_col
        merchant_col = merchant_col or self.merchant_col
        
        G = nx.Graph()
        
        # Add user and merchant nodes with type attribute
        users = txn_df[user_col].unique()
        merchants = txn_df[merchant_col].unique()
        
        G.add_nodes_from(users, bipartite=0, node_type="user")
        G.add_nodes_from(merchants, bipartite=1, node_type="merchant")
        
        # Add edges for each transaction
        for _, row in txn_df.iterrows():
            user = row[user_col]
            merchant = row[merchant_col]
            if G.has_edge(user, merchant):
                G[user][merchant]["weight"] += 1
            else:
                G.add_edge(user, merchant, weight=1)
        
        return G

    def compute_node_features(self, graph: nx.Graph) -> Dict:
        """Compute per-node features from graph.
        
        Args:
            graph: NetworkX graph.
            
        Returns:
            Dictionary mapping node_id -> feature_dict with keys:
            - degree: node degree
            - clustering_coefficient: local clustering coefficient
            - node_type: "user" or "merchant"
            - degree_centrality: normalized degree centrality
        """
        features = {}
        
        # Compute graph-level metrics
        degrees = dict(graph.degree())
        clustering = nx.clustering(graph)
        centrality = nx.degree_centrality(graph)
        
        for node in graph.nodes():
            node_type = graph.nodes[node].get("node_type", "unknown")
            features[node] = {
                "degree": degrees[node],
                "clustering_coefficient": clustering[node],
                "node_type": node_type,
                "degree_centrality": centrality[node],
            }
        
        return features

    def detect_communities(
        self,
        graph: nx.Graph,
        node_type: str = "user",
    ) -> Dict:
        """Run Louvain community detection on graph.
        
        Args:
            graph: NetworkX graph.
            node_type: Filter to only return communities for this node type.
            
        Returns:
            Dictionary mapping node_id -> community_label (integer).
        """
        import community
        
        # Run Louvain on the full graph
        partition = community.best_partition(graph)
        
        # Filter by node type if specified
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
        n_walks: int = 10,
        walk_length: int = 8,
    ) -> Dict:
        """Compute random walk structural embeddings.
        
        Args:
            graph: NetworkX graph.
            n_walks: Number of random walks per node.
            walk_length: Length of each random walk.
            
        Returns:
            Dictionary mapping node_id -> RWSE features (numpy array).
        """
        rwse_features = {}
        nodes = list(graph.nodes())
        
        for node in nodes:
            walk_stats = []
            
            for _ in range(n_walks):
                walk = [node]
                current = node
                
                for _ in range(walk_length):
                    neighbors = list(graph.neighbors(current))
                    if neighbors:
                        current = np.random.choice(neighbors)
                        walk.append(current)
                    else:
                        break
                
                # Compute statistics from this walk
                walk_stats.append({
                    "length": len(walk),
                    "unique_nodes": len(set(walk)),
                    "mean_degree": np.mean([graph.degree(n) for n in walk]),
                })
            
            # Aggregate walk statistics
            avg_length = np.mean([s["length"] for s in walk_stats])
            avg_unique = np.mean([s["unique_nodes"] for s in walk_stats])
            avg_degree = np.mean([s["mean_degree"] for s in walk_stats])
            
            rwse_features[node] = np.array([avg_length, avg_unique, avg_degree])
        
        return rwse_features

    def append_graph_features(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        user_col: Optional[str] = None,
        merchant_col: Optional[str] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Append graph-derived features to transaction DataFrames.
        
        Args:
            train_df: Training DataFrame.
            test_df: Test DataFrame.
            user_col: Column name for user identifier.
            merchant_col: Column name for merchant identifier.
            
        Returns:
            Tuple of (train_df, test_df) with graph features appended.
        """
        user_col = user_col or self.user_col
        merchant_col = merchant_col or self.merchant_col
        
        # Build graph from training data
        graph = self.build_bipartite_graph(train_df, user_col, merchant_col)
        
        # Compute node features
        node_features = self.compute_node_features(graph)
        
        # Detect communities
        communities = self.detect_communities(graph, node_type="user")
        
        # Compute RWSE features
        rwse_features = self.compute_rwse_features(graph)
        
        # Create feature DataFrames
        feature_rows = []
        for node, feats in node_features.items():
            row = {
                "node_id": node,
                "node_type": feats["node_type"],
                "degree": feats["degree"],
                "clustering_coefficient": feats["clustering_coefficient"],
                "degree_centrality": feats["degree_centrality"],
            }
            
            # Add RWSE features
            rwse = rwse_features.get(node, np.zeros(3))
            row["rwse_avg_length"] = rwse[0]
            row["rwse_avg_unique"] = rwse[1]
            row["rwse_avg_degree"] = rwse[2]
            
            # Add community label for users
            if feats["node_type"] == "user":
                row["community_label"] = communities.get(node, -1)
            
            feature_rows.append(row)
        
        feature_df = pd.DataFrame(feature_rows)
        
        # Split by node type
        user_features = feature_df[feature_df["node_type"] == "user"].copy()
        merchant_features = feature_df[feature_df["node_type"] == "merchant"].copy()
        
        # Rename columns for joining
        user_features = user_features.rename(columns={
            "node_id": user_col,
            "degree": "user_degree",
            "clustering_coefficient": "user_clustering",
            "degree_centrality": "user_degree_centrality",
            "community_label": "user_community",
            "rwse_avg_length": "user_rwse_length",
            "rwse_avg_unique": "user_rwse_unique",
            "rwse_avg_degree": "user_rwse_degree",
        })
        
        merchant_features = merchant_features.rename(columns={
            "node_id": merchant_col,
            "degree": "merchant_degree",
            "clustering_coefficient": "merchant_clustering",
            "degree_centrality": "merchant_degree_centrality",
            "rwse_avg_length": "merchant_rwse_length",
            "rwse_avg_unique": "merchant_rwse_unique",
            "rwse_avg_degree": "merchant_rwse_degree",
        })
        
        # Join features to train and test DataFrames
        train_df = train_df.merge(
            user_features[[user_col, "user_degree", "user_clustering", 
                          "user_degree_centrality", "user_community",
                          "user_rwse_length", "user_rwse_unique", "user_rwse_degree"]],
            on=user_col,
            how="left",
        )
        
        train_df = train_df.merge(
            merchant_features[[merchant_col, "merchant_degree", "merchant_clustering",
                              "merchant_degree_centrality",
                              "merchant_rwse_length", "merchant_rwse_unique", 
                              "merchant_rwse_degree"]],
            on=merchant_col,
            how="left",
        )
        
        test_df = test_df.merge(
            user_features[[user_col, "user_degree", "user_clustering",
                          "user_degree_centrality", "user_community",
                          "user_rwse_length", "user_rwse_unique", "user_rwse_degree"]],
            on=user_col,
            how="left",
        )
        
        test_df = test_df.merge(
            merchant_features[[merchant_col, "merchant_degree", "merchant_clustering",
                              "merchant_degree_centrality",
                              "merchant_rwse_length", "merchant_rwse_unique",
                              "merchant_rwse_degree"]],
            on=merchant_col,
            how="left",
        )
        
        # Fill NaN values with 0
        graph_feature_cols = [
            "user_degree", "user_clustering", "user_degree_centrality", "user_community",
            "user_rwse_length", "user_rwse_unique", "user_rwse_degree",
            "merchant_degree", "merchant_clustering", "merchant_degree_centrality",
            "merchant_rwse_length", "merchant_rwse_unique", "merchant_rwse_degree",
        ]
        
        for col in graph_feature_cols:
            if col in train_df.columns:
                train_df[col] = train_df[col].fillna(0)
            if col in test_df.columns:
                test_df[col] = test_df[col].fillna(0)
        
        return train_df, test_df
