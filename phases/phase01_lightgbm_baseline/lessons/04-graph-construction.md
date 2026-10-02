# Lesson 04: Graph Construction

## Explain

**Bipartite User-Merchant Graph**

In fraud detection, transactions naturally form a bipartite graph:
- **Users** (cardholders) on one side
- **Merchants** (stores) on the other side
- **Transactions** as edges connecting users to merchants

This structure captures fraud patterns:
- **Fraud rings** — groups of users transacting with the same suspicious merchants
- **Card testing** — fraudsters testing stolen cards at multiple merchants
- **Money mules** — users receiving fraudulent transactions

**Why Model as a Graph?**

Tabular models treat each transaction independently. Graph-based features capture relational patterns:
- A user transacting with many merchants is different from a user with few merchants
- A merchant with many fraud reports is suspicious
- Communities of users/merchants reveal fraud networks

## Code

```python
import networkx as nx
from phase01_lightgbm_baseline.graph_features import GraphFeatureExtractor
from phase01_lightgbm_baseline.data_loader import DataLoader

# Load data
loader = DataLoader()
train_df, _ = loader.load_data(sample_limit=500)

# Build graph
extractor = GraphFeatureExtractor(user_col="card1", merchant_col="ProductCD")
G = extractor.build_bipartite_graph(train_df)

print(f"Graph statistics:")
print(f"  Nodes: {G.number_of_nodes()}")
print(f"  Edges: {G.number_of_edges()}")
print(f"  Users: {sum(1 for n in G.nodes() if G.nodes[n].get('node_type') == 'user')}")
print(f"  Merchants: {sum(1 for n in G.nodes() if G.nodes[n].get('node_type') == 'merchant')}")

# Degree distribution
degrees = [G.degree(n) for n in G.nodes()]
print(f"\nDegree statistics:")
print(f"  Min: {min(degrees)}")
print(f"  Max: {max(degrees)}")
print(f"  Mean: {sum(degrees) / len(degrees):.2f}")
```

## Visualize

```python
import matplotlib.pyplot as plt
import numpy as np

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Degree distribution
degrees = [G.degree(n) for n in G.nodes()]
axes[0].hist(degrees, bins=30, edgecolor='black', alpha=0.7)
axes[0].set_xlabel('Degree')
axes[0].set_ylabel('Count')
axes[0].set_title('Node Degree Distribution')
axes[0].set_xscale('log')

# Graph visualization (small sample)
if G.number_of_nodes() < 100:
    pos = nx.spring_layout(G, seed=42)
    node_colors = ['lightblue' if G.nodes[n].get('node_type') == 'user' 
                   else 'lightgreen' for n in G.nodes()]
    nx.draw(G, pos, node_color=node_colors, node_size=50, 
            edge_color='gray', alpha=0.6, ax=axes[1])
    axes[1].set_title('Graph Structure (Sample)')
else:
    # For large graphs, show degree distribution on log scale
    sorted_degrees = sorted(degrees, reverse=True)
    axes[1].plot(sorted_degrees, marker='.', linestyle='none')
    axes[1].set_xlabel('Node Rank')
    axes[1].set_ylabel('Degree')
    axes[1].set_title('Degree Rank')
    axes[1].set_yscale('log')

plt.tight_layout()
plt.show()
```

## Practice

**Exercise:** Subsample the graph and compare statistics.

1. Take a random subset of 50% of transactions
2. Build a graph from the subset
3. Compare the number of nodes, edges, and average degree with the full graph

How does subsampling affect the graph structure?

## Solution

```python
# Full graph
G_full = extractor.build_bipartite_graph(train_df)

# Subsample 50%
train_sample = train_df.sample(frac=0.5, random_state=42)
G_sample = extractor.build_bipartite_graph(train_sample)

print("Full graph:")
print(f"  Nodes: {G_full.number_of_nodes()}")
print(f"  Edges: {G_full.number_of_edges()}")
print(f"  Avg degree: {sum(dict(G_full.degree()).values()) / G_full.number_of_nodes():.2f}")

print("\n50% sample:")
print(f"  Nodes: {G_sample.number_of_nodes()}")
print(f"  Edges: {G_sample.number_of_edges()}")
print(f"  Avg degree: {sum(dict(G_sample.degree()).values()) / G_sample.number_of_nodes():.2f}")
```

**Observation:** Subsampling reduces edges more than nodes, as many users/merchants appear in multiple transactions. The average degree decreases, potentially losing some relational information.
