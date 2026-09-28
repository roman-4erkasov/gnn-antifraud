## Context

Phase 1 created graph features in `phases/phase01_lightgbm_baseline/src/phase01_lightgbm_baseline/graph_features.py` (degree, community, RWSE). Phase 2 created a minimal GCN in `phases/phase02_minimal_gnn/src/phase02_minimal_gnn/minimal_gcn.py`. Both used basic graph-derived features. Now we need to systematically evaluate how different structural encoding strategies affect GCN performance. The existing infrastructure (data loading, graph building, GCN model, metrics) provides a solid foundation.

## Goals / Non-Goals

**Goals:**
- Implement 4 structural encoding techniques: ReCWE, RWSE, Spectral, Community detection.
- Train separate GCN models with each encoding under identical conditions.
- Evaluate each encoding with all metrics and produce a comparison table.
- Compute SHAP values for graph-aware LightGBM with each encoding and compare feature importance.
- Identify the best-performing encoding.

**Non-Goals:**
- Combining multiple encodings simultaneously (reserved for later if single best is insufficient).
- Designing new encoding schemes.
- Hyperparameter optimization per encoding.
- Visualizing encodings (t-SNE/UMAP).

## Decisions

### Decision 1: ReCWE via random walk visitation counts
**Choice:** Compute ReCWE by performing random walks from each node and recording the visitation frequency at each relative distance (k-hop).
**Rationale:** ReCWE captures the structural role of a node by how it's reached from other nodes. For fraud detection, fraudsters often have distinctive structural roles (e.g., bridge nodes, hub nodes).
**Alternatives considered:**
- Laplacian Eigenfunctions (as in PyG RPE) — similar expressive power but ReCWE is simpler to implement from scratch.
- Shortest path distances — simpler but ignores multi-path structural information.

### Decision 2: RWSE via random walk statistics
**Choice:** Compute RWSE features as: visit frequency, mean shortest path, clustering coefficient from walk trajectories, degree-weighted centrality.
**Rationale:** These statistics capture the local structural context of each node. They are complementary to ReCWE (which focuses on relative position) and can be computed from the same random walk infrastructure.
**Alternatives considered:**
- Graphlet degree awareness (GDA) — more expressive but computationally expensive for large graphs.
- Whirlpool encoding — newer, potentially better but less established.

### Decision 3: Spectral encoding via truncated Laplacian eigenvectors
**Choice:** Compute the normalized graph Laplacian and extract the k smallest non-trivial eigenvectors as node features (like Diffusion Embeddings).
**Rationale:** Spectral encoding captures global graph structure through eigenvectors. It is a well-established positional encoding technique (Kipf et al.).
**Alternatives considered:**
- Full eigendecomposition — too expensive for large graphs.
- Chebyshev polynomial approximation — more efficient but harder to implement correctly.
- Fiedler vector only — too coarse; we need k-dimensional embedding.

### Decision 4: Community detection via Louvain (python-louvain)
**Choice:** Use the `community_louvain` library (python-louvain) for Louvain community detection, producing integer labels per node.
**Rationale:** Louvain is fast, produces good quality communities, and the library is already referenced in the original tasks.md.
**Alternatives considered:**
- Label Propagation Algorithm (igraph) — comparable quality, slower convergence.
- Leiden algorithm — better theoretically but community_louvain is simpler.
- k-way spectral clustering — more complex, not necessary for ordinal community features.

### Decision 5: One-hot encoding for community labels
**Choice:** One-hot encode community labels as node features (n_communities binary features).
**Rationale:** One-hot preserves the full categorical information without imposing an ordinal relationship between community labels. For small graphs with few communities, this is feasible. For large graphs, fall back to ordinal encoding.
**Alternatives considered:**
- Ordinal encoding — simpler, fewer features, but imposes false ordering.
- Learned embedding per community — adds parameters; not needed for Phase 3.

### Decision 6: Identical training conditions across encodings
**Choice:** Use the exact same GCN hyperparameters, edge index, and train/val/test split for all encoding variants. Only the node feature matrix changes.
**Rationale:** Fair comparison requires identical conditions. Any difference in performance can be attributed solely to the encoding.

### Decision 7: Self-contained data loading per phase
**Choice:** Phase 03 implements its own `data_loader.py` that checks for the IEEE-CIS Fraud Detection dataset in the shared `data/` directory at project root, downloads it if missing, and loads phase-specific data. The loader handles graph construction with structural features (Laplacian eigenvectors, random walk features, community detection).
**Rationale:** Each phase (01, 02, 03, 04) shares the same dataset but has different data preparation needs. Self-contained loaders avoid cross-phase dependencies while the shared `data/` directory avoids redundant downloads. The loader encapsulates graph construction so downstream code (structural_encodings.py, train_gcn.py) receives ready-to-use graph data.
**Alternatives considered:**
- Shared data loading module in `src/utils/` — creates coupling between phases; each phase has different graph construction needs.
- Download data on every run — wasteful; the dataset is large and slow to download.

## Risks / Trade-offs

| Risk | Mitigation |
|------|-----------|
| Spectral encoding requires eigendecomposition — O(N^3) for dense graph | Use scipy.sparse.linalg.eigsh for sparse graphs; truncate to small k (e.g., 16). |
| Louvain community detection may produce many small communities in bipartite graph | Filter to user subgraph for community detection; use coarser resolution parameter. |
| RWSE and ReCWE may be redundant (both from random walks) | Use different statistics: ReCWE focuses on relative distance counts, RWSE focuses on aggregate statistics. |
| Many features from encodings could overwhelm GCN | Use feature normalization; GCN is generally robust to feature dimensionality. |

## Educational Content

### Lesson structure

Each lesson in `lessons/` follows a consistent 5-part pedagogical structure:

1. **Explain** — Introduce the mathematical concept and its relevance to structural encoding for fraud detection.
2. **Code** — Provide a minimal, runnable code snippet demonstrating the core algorithm.
3. **Visualize** — Include a figure or diagram showing the concept on a small example graph.
4. **Practice** — Pose an exercise for the reader to implement or extend the technique.
5. **Solution** — Provide the reference solution with commentary.

### Lesson topics

| Lesson | Topic | Key concepts |
|--------|-------|-------------|
| `01-spectral-graph-theory.md` | Spectral graph theory foundations | Adjacency matrix, degree matrix, graph Laplacian, eigenvalues |
| `02-laplacian-eigenvectors.md` | Laplacian eigenvectors as positional encodings | Normalized Laplacian, Fiedler vector, truncated eigendecomposition |
| `03-random-walk-features.md` | Random walk structural features | Transition matrix, visitation frequency, RWSE, ReCWE |
| `04-community-detection.md` | Community detection with Louvain | Modularity optimization, resolution parameter, community labels |
| `05-structural-features-for-fraud.md` | Applying structural features to fraud detection | Combining encodings, fraudster structural patterns, feature engineering |

### Notebooks

| Notebook | Purpose |
|----------|---------|
| `01-exploring-eigenvectors.ipynb` | Interactive exploration of Laplacian eigenvectors on a synthetic fraud graph — compute eigenvectors, visualize node embeddings in 2D, observe how fraud clusters in spectral space |
| `02-community-analysis.ipynb` | Louvain community detection on the fraud graph — detect communities, analyze community composition (fraud ratio per community), visualize community structure |

## Open Questions

- What value of k (number of eigenvectors) for spectral encoding? (Default: 16 — balances expressiveness and computation.)
- Should community labels be one-hot or ordinal encoded for the given graph size? (Decide based on n_communities: one-hot if < 50, ordinal otherwise.)
