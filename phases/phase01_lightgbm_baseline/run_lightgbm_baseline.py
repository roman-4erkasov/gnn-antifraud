#!/usr/bin/env python3
"""Run LightGBM baseline pipeline with graph features."""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add phase src to path
phase_dir = Path(__file__).resolve().parent / "src"
sys.path.insert(0, str(phase_dir))

from phase01_lightgbm_baseline.data_loader import DataLoader
from phase01_lightgbm_baseline.lightgbm_baseline import LightGBMWrapper
from phase01_lightgbm_baseline.graph_features import GraphFeatureExtractor
from phase01_lightgbm_baseline.comparison import (
    compute_comparison_report,
    check_significance,
    generate_stop_decision,
)


def main():
    parser = argparse.ArgumentParser(description="Run LightGBM baseline with graph features")
    parser.add_argument("--data-path", type=str, default="../data", help="Path to shared data directory (relative to phase root)")
    parser.add_argument("--sample-limit", type=int, default=None, help="Limit samples for testing")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    print("=" * 80)
    print("LightGBM Baseline with Graph Features")
    print("=" * 80)
    print()

    # Load data
    print("Loading data...")
    loader = DataLoader(data_dir=args.data_path)
    train_df, test_df = loader.load_data(sample_limit=args.sample_limit)
    print(f"  Train samples: {len(train_df)}")
    print(f"  Test samples: {len(test_df)}")
    print(f"  Fraud rate (train): {train_df['isFraud'].mean():.4f}")
    print()

    # Prepare feature columns
    baseline_features = ["TransactionAmt", "time_since_creation", "n_prior_transactions"]
    
    # Split data: 60% train, 20% val, 20% test
    n_total = len(train_df)
    n_train = int(0.6 * n_total)
    n_val = int(0.2 * n_total)
    
    train_split = train_df.iloc[:n_train]
    val_split = train_df.iloc[n_train:n_train + n_val]
    test_split = test_df  # Use full test set
    
    # Train minimal model (baseline features only)
    print("Training minimal model (3 baseline features)...")
    X_train_min = train_split[baseline_features].values
    y_train_min = train_split["isFraud"].values
    X_val_min = val_split[baseline_features].values
    y_val_min = val_split["isFraud"].values
    X_test_min = test_split[baseline_features].values
    y_test_min = test_split["isFraud"].values
    
    minimal_model = LightGBMWrapper(random_state=args.random_state)
    minimal_metrics = minimal_model.full_pipeline(
        X_train_min, y_train_min,
        X_val_min, y_val_min,
        X_test_min, y_test_min,
        k=int(0.01 * len(y_test_min)),
    )
    
    print("  Minimal model metrics:")
    for metric, value in minimal_metrics.items():
        print(f"    {metric}: {value:.4f}")
    print()

    # Train graph-aware model
    print("Training graph-aware model (baseline + graph features)...")
    graph_extractor = GraphFeatureExtractor(user_col="card1", merchant_col="ProductCD")
    train_graph, test_graph = graph_extractor.append_graph_features(
        train_split.copy(), test_split.copy()
    )
    
    # Apply graph features to validation split
    val_graph = val_split.copy()
    # Build graph from training data
    graph = graph_extractor.build_bipartite_graph(train_split, "card1", "ProductCD")
    node_features = graph_extractor.compute_node_features(graph)
    communities = graph_extractor.detect_communities(graph, node_type="user")
    rwse_features = graph_extractor.compute_rwse_features(graph)
    
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
        "node_id": "card1",
        "degree": "user_degree",
        "clustering_coefficient": "user_clustering",
        "degree_centrality": "user_degree_centrality",
        "community_label": "user_community",
        "rwse_avg_length": "user_rwse_length",
        "rwse_avg_unique": "user_rwse_unique",
        "rwse_avg_degree": "user_rwse_degree",
    })
    
    merchant_features = merchant_features.rename(columns={
        "node_id": "ProductCD",
        "degree": "merchant_degree",
        "clustering_coefficient": "merchant_clustering",
        "degree_centrality": "merchant_degree_centrality",
        "rwse_avg_length": "merchant_rwse_length",
        "rwse_avg_unique": "merchant_rwse_unique",
        "rwse_avg_degree": "merchant_rwse_degree",
    })
    
    # Join features to val DataFrame
    val_graph = val_graph.merge(
        user_features[["card1", "user_degree", "user_clustering", 
                      "user_degree_centrality", "user_community",
                      "user_rwse_length", "user_rwse_unique", "user_rwse_degree"]],
        on="card1",
        how="left",
    )
    
    val_graph = val_graph.merge(
        merchant_features[["ProductCD", "merchant_degree", "merchant_clustering",
                          "merchant_degree_centrality",
                          "merchant_rwse_length", "merchant_rwse_unique", 
                          "merchant_rwse_degree"]],
        on="ProductCD",
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
        if col in val_graph.columns:
            val_graph[col] = val_graph[col].fillna(0)
    
    # Identify graph feature columns
    graph_feature_cols = [
        "user_degree", "user_clustering", "user_degree_centrality", "user_community",
        "user_rwse_length", "user_rwse_unique", "user_rwse_degree",
        "merchant_degree", "merchant_clustering", "merchant_degree_centrality",
        "merchant_rwse_length", "merchant_rwse_unique", "merchant_rwse_degree",
    ]
    
    all_features = baseline_features + graph_feature_cols
    
    X_train_graph = train_graph[all_features].values
    y_train_graph = train_graph["isFraud"].values
    X_val_graph = val_graph[all_features].values
    y_val_graph = val_graph["isFraud"].values
    X_test_graph = test_graph[all_features].values
    y_test_graph = test_graph["isFraud"].values
    
    graph_model = LightGBMWrapper(random_state=args.random_state)
    graph_metrics = graph_model.full_pipeline(
        X_train_graph, y_train_graph,
        X_val_graph, y_val_graph,
        X_test_graph, y_test_graph,
        k=int(0.01 * len(y_test_graph)),
    )
    
    print("  Graph-aware model metrics:")
    for metric, value in graph_metrics.items():
        print(f"    {metric}: {value:.4f}")
    print()

    # Compare models
    print("Comparing models...")
    deltas = compute_comparison_report(minimal_metrics, graph_metrics)
    
    print("  Delta metrics (graph - minimal):")
    for metric, value in deltas.items():
        print(f"    {metric}: {value:+.4f}")
    print()

    # Statistical significance
    print("Testing statistical significance...")
    raw_probs_minimal = minimal_model.predict(X_test_min)
    raw_probs_graph = graph_model.predict(X_test_graph)
    p_value = check_significance(raw_probs_minimal, raw_probs_graph, seed=args.random_state)
    print(f"  P-value: {p_value:.4f}")
    print()

    # Stop decision
    delta_pr_auc = deltas.get("delta_pr_auc", 0.0)
    decision = generate_stop_decision(delta_pr_auc, p_value)
    
    print("=" * 80)
    print("STOP DECISION")
    print("=" * 80)
    print(f"  Delta PR-AUC: {delta_pr_auc:+.4f}")
    print(f"  P-value: {p_value:.4f}")
    print(f"  Decision: {decision}")
    print("=" * 80)

    return 0


if __name__ == "__main__":
    sys.exit(main())
