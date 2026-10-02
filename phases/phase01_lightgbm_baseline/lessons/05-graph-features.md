# Lesson 05: Graph Features

## Explain

**Node-Level Features**

From the bipartite graph, we extract several node-level features:

1. **Degree** — number of connections. High-degree users transact with many merchants; high-degree merchants serve many users.

2. **Clustering Coefficient** — measures how connected a node's neighbors are. Low clustering indicates a user transacts with unrelated merchants (potentially suspicious).

3. **Community Labels (Louvain)** — detects communities of densely connected nodes. Fraudsters often form tight communities.

4. **RWSE (Random Walk Structural Embeddings)** — captures structural roles via random walks. Nodes with similar structural roles have similar embeddings.

**Why These Features?**

- **Degree** — captures transaction volume patterns
- **Clustering** — reveals isolated vs. connected behavior
- **Communities** — identifies fraud rings
- **RWSE** — captures higher-order structural patterns

## Code

```python
from phase01_lightgbm_baseline.graph_features import GraphFeatureExtractor
from phase01_lightgbm_baseline.data_loader import DataLoader

# Load data and build graph
loader = DataLoader()
train_df, _ = loader.load_data(sample_limit=500)

extractor = GraphFeatureExtractor(user_col="card1", merchant_col="ProductCD")
G = extractor.build_bipartite_graph(train_df)

# Compute node features
node_features = extractor.compute_node_features(G)

# Show sample
sample_node = list(node_features.keys())[0]
print(f"Node {sample_node} features:")
for key, value in node_features[sample_node].items():
    print(f"  {key}: {value}")

# Detect communities
communities = extractor.detect_communities(G, node_type="user")
print(f"\nCommunities detected: {len(set(communities.values()))}")

# Compute RWSE features
rwse = extractor.compute_rwse_features(G, n_walks=5, walk_length=4)
sample_rwse = list(rwse.values())[0]
print(f"\nRWSE features shape: {sample_rwse.shape}")
print(f"RWSE sample: {sample_rwse}")
```

## Visualize

```python
import matplotlib.pyplot as plt
import pandas as pd

# Append graph features to DataFrame
train_with_graph, _ = extractor.append_graph_features(train_df.copy(), train_df.copy())

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# User degree distribution
axes[0, 0].hist(train_with_graph["user_degree"], bins=30, edgecolor='black', alpha=0.7)
axes[0, 0].set_xlabel('User Degree')
axes[0, 0].set_ylabel('Count')
axes[0, 0].set_title('User Degree Distribution')

# Community size distribution
community_sizes = train_with_graph["user_community"].value_counts()
axes[0, 1].hist(community_sizes, bins=20, edgecolor='black', alpha=0.7)
axes[0, 1].set_xlabel('Community Size')
axes[0, 1].set_ylabel('Count')
axes[0, 1].set_title('Community Size Distribution')

# Correlation matrix of graph features
graph_cols = ["user_degree", "user_clustering", "user_degree_centrality", 
              "merchant_degree", "merchant_clustering"]
corr = train_with_graph[graph_cols].corr()
im = axes[1, 0].imshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
axes[1, 0].set_xticks(range(len(graph_cols)))
axes[1, 0].set_yticks(range(len(graph_cols)))
axes[1, 0].set_xticklabels(graph_cols, rotation=45, ha='right')
axes[1, 0].set_yticklabels(graph_cols)
axes[1, 0].set_title('Feature Correlation')
plt.colorbar(im, ax=axes[1, 0])

# Fraud rate by community
fraud_by_community = train_with_graph.groupby("user_community")["isFraud"].mean()
axes[1, 1].bar(range(len(fraud_by_community)), fraud_by_community.values, 
               edgecolor='black', alpha=0.7)
axes[1, 1].set_xlabel('Community ID')
axes[1, 1].set_ylabel('Fraud Rate')
axes[1, 1].set_title('Fraud Rate by Community')

plt.tight_layout()
plt.show()
```

## Practice

**Exercise:** Compare features by community.

1. Identify the top 3 communities by fraud rate
2. Compare their average degree, clustering coefficient, and RWSE features
3. What patterns do you observe?

Do high-fraud communities have distinct structural characteristics?

## Solution

```python
# Compute fraud rate by community
train_with_graph, _ = extractor.append_graph_features(train_df.copy(), train_df.copy())
community_stats = train_with_graph.groupby("user_community").agg({
    "isFraud": "mean",
    "user_degree": "mean",
    "user_clustering": "mean",
    "user_rwse_length": "mean",
}).sort_values("isFraud", ascending=False)

print("Top 5 communities by fraud rate:")
print(community_stats.head())

print("\nBottom 5 communities by fraud rate:")
print(community_stats.tail())
```

**Observation:** High-fraud communities often have distinct structural patterns — lower clustering (isolated behavior) or higher degree (many transactions). This validates that graph features capture fraud signals.
