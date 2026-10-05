# Lesson 04 — Graph Construction

## Explain

Fraud is rarely isolated: fraudsters reuse **cards**, **devices**, and **merchants**.
A transaction record alone hides these relationships, but a **graph** makes them
explicit.

We build a **bipartite graph** where the two node sets are:

- **users** (`card1`) and
- **merchants** (`ProductCD`).

Every transaction creates an **edge** between its user and its merchant, with an
edge `weight` counting how many times that pair transacted. In this bipartite view
a fraud ring appears as a dense cluster of cards connected to the same products.

`GraphFeatureExtractor.build_bipartite_graph(txn_df)` performs this construction
with NetworkX, tagging each node with `bipartite` (0 = user, 1 = merchant) and
`node_type`.

## Code

```python
import pandas as pd
import networkx as nx

from phase01_lightgbm_baseline import GraphFeatureExtractor

txns = pd.DataFrame(
    {
        "TransactionID": [1, 2, 3, 4, 5, 6],
        "card1": [100, 100, 101, 102, 102, 103],
        "ProductCD": ["W", "C", "W", "W", "H", "W"],
        "isFraud": [0, 0, 1, 0, 1, 1],
    }
)

extractor = GraphFeatureExtractor(user_col="card1", merchant_col="ProductCD")
graph = extractor.build_bipartite_graph(txns)

print(f"nodes: {graph.number_of_nodes()}  edges: {graph.number_of_edges()}")
print(f"is bipartite: {nx.is_bipartite(graph)}")

degrees = dict(graph.degree())
for node, degree in sorted(degrees.items(), key=lambda kv: -kv[1]):
    print(f"  {node!s:>5} ({graph.nodes[node]['node_type']:>8}): degree {degree}")
```

## Visualize

```python
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx

users = [n for n, d in graph.nodes(data=True) if d["node_type"] == "user"]
merchants = [n for n, d in graph.nodes(data=True) if d["node_type"] == "merchant"]
pos = nx.spring_layout(graph, seed=1)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
nx.draw_networkx_nodes(graph, pos, nodelist=users, node_color="tab:blue", node_size=400, ax=axes[0])
nx.draw_networkx_nodes(graph, pos, nodelist=merchants, node_color="tab:orange", node_size=400, ax=axes[0])
nx.draw_networkx_edges(graph, pos, ax=axes[0], alpha=0.6)
nx.draw_networkx_labels(graph, pos, ax=axes[0], font_size=8)
axes[0].set_title("Bipartite user-merchant graph")
axes[0].axis("off")

axes[1].hist(list(degrees.values()), bins=8, color="steelblue")
axes[1].set(xlabel="Degree", ylabel="Number of nodes", title="Degree distribution")
fig.tight_layout()
fig.savefig("lesson04_graph.png", dpi=120)
print("saved lesson04_graph.png")
```

## Practice

Subsample the transaction table (e.g. keep 50% of rows) and rebuild the graph.
Compare node counts, edge counts, and the shape of the degree distribution. How
robust are these statistics to sampling?

Open **[../exercises/04-subsampling-graph.ipynb](../exercises/04-subsampling-graph.ipynb)**
for a scripted comparison.

## Solution

Subsampling removes rare edges first, so **edge count falls faster than node
count** and the degree distribution loses its heavy tail. This matters because
graph features computed on a small sample (like our synthetic fixture) are
noisier. **Takeaway:** graph features are only as reliable as the graph's
coverage; report graph size alongside any graph-model result.
