## 1. Setup

- [ ] 1.1 Create project scaffold `phases/phase03_structural_encoding/` with `pyproject.toml`, `uv.lock`, and `src/phase03_structural_encoding/__init__.py`; verify `uv sync` succeeds
- [ ] 1.2 Add dependencies to `pyproject.toml`: numpy, scipy, networkx, community_louvain, torch, torch_geometric, lightgbm, shap; verify `uv sync` succeeds

## 2. Data loading

- [ ] 2.1 Create `phases/phase03_structural_encoding/src/phase03_structural_encoding/data_loader.py` with `DataLoader` class containing `check_data_exists()`, `download_data()`, and `load_data()` methods; `check_data_exists()` verifies IEEE-CIS data files in shared `data/` directory at project root; `download_data()` fetches dataset if missing; `load_data()` returns transaction and identity DataFrames; verify with `ast.parse`
- [ ] 2.2 Implement `prepare_graph_data(transactions_df, identity_df)` in `data_loader.py` that builds a bipartite graph from transaction-identity relationships and computes structural encoding features (Laplacian eigenvectors, random walk features, community detection labels); verify output is a graph object with node features compatible with `structural_encodings.py`
- [ ] 2.3 Verify `prepare_graph_data()` output is compatible with `structural_encodings.py` by running a smoke test: load data → build graph → pass to each encoding function; verify no shape mismatches or errors

## 3. ReCWE implementation

- [ ] 3.1 Create `phases/phase03_structural_encoding/src/phase03_structural_encoding/structural_encodings.py` with module docstring, imports (numpy, scipy, networkx), and `_compute_recwe` function stub; verify Python syntax
- [ ] 3.2 Implement `_compute_recwe(graph, n_walks=10, walk_length=8, max_distance=4)` performing random walks from each node and encoding visitation frequency at each relative distance (k-hop); verify output shape [n_nodes, max_distance]
- [ ] 3.3 Verify ReCWE captures structural role: on synthetic graph, hub node has different ReCWE pattern than leaf node

## 4. RWSE implementation

- [ ] 4.1 Implement `_compute_rwse(graph, n_walks=20, walk_length=16)` computing visit frequency, mean shortest path, clustering coefficient, degree-weighted centrality from random walks; verify output shape [n_nodes, 4]
- [ ] 4.2 Add configurable dimensionality to RWSE (default 8 features, pad with zeros if fewer features computed); verify shape flexibility

## 5. Spectral encoding implementation

- [ ] 5.1 Implement `_compute_spectral_encoding(graph, k=16)` computing normalized graph Laplacian L = I - D^(-1/2)AD^(-1/2) via scipy.sparse and extracting k smallest non-trivial eigenvectors; verify output shape [n_nodes, k]
- [ ] 5.2 Handle disconnected graphs by computing eigenvectors per component and padding; verify no errors on disconnected input

## 6. Community detection implementation

- [ ] 6.1 Implement `_compute_community_labels(graph, node_type='user')` running Louvain community detection via `community_louvain` on the subgraph of specified node type; verify output dict of node→int_label
- [ ] 6.2 Implement `_encode_community_labels(community_dict, all_nodes, encoding='onehot')` producing one-hot or ordinal encoding; verify shape [n_nodes, n_communities] or [n_nodes, 1]

## 7. GCN training with encodings

- [ ] 7.1 Create `phases/phase03_structural_encoding/src/phase03_structural_encoding/train_gcn.py` with `train_gcn_with_encoding(encoding_name, node_features, edge_index, node_labels, train_idx, val_idx, test_idx)` wrapper that calls `MinimalGCN` training with encoding-specific features; verify it trains and returns metrics dict (use `ast.parse` to verify module imports)
- [ ] 7.2 Implement `evaluate_all_encodings(data, graph)` running all 4 encodings → train GCN → evaluate → collect metrics; verify metrics dict per encoding
- [ ] 7.3 Print summary table: encoding × PR-AUC × ROC-AUC × Brier score × log-loss; verify output is readable

## 8. SHAP-based feature importance

- [ ] 8.1 Train graph-aware LightGBM models with each encoding (reuse `phases/phase01_lightgbm_baseline/src/phase01_lightgbm_baseline/lightgbm_baseline` module); verify models train on extended feature matrices (use `ast.parse` to verify module)
- [ ] 8.2 Compute SHAP values using `shap.TreeExplainer` for each LightGBM model; verify SHAP output shape [n_test_samples, n_features]
- [ ] 8.3 Extract mean absolute SHAP values per feature and produce per-encoding feature importance ranking; verify ranking lists features in descending order

## 9. Encoding comparison and best selection

- [ ] 9.1 Implement `select_best_encoding(encoding_results)` selecting encoding with highest PR-AUC, ties broken by lowest Brier score; verify correct selection on synthetic results
- [ ] 9.2 Produce comparison report with all encoding results, best encoding recommendation, and SHAP feature importance summaries

## 10. Interactive exercises (exercises/)

- [ ] 10.1 Create `phases/phase03_structural_encoding/exercises/` directory
- [ ] 10.2 Create `01-spectral-graph-theory.ipynb` with setup code and task to compute graph Laplacian eigenvalues
- [ ] 10.3 Create `02-laplacian-eigenvectors.ipynb` with task to compute and visualize Fiedler vector
- [ ] 10.4 Create `03-random-walk-features.ipynb` with task to compute RWSE features from random walks
- [ ] 10.5 Create `04-community-detection.ipynb` with task to run Louvain and analyze community structure
- [ ] 10.6 Create `05-structural-features-for-fraud.ipynb` with task to combine encodings and evaluate impact

## 11. Tests

- [ ] 11.1 Create `phases/phase03_structural_encoding/tests/test_structural_encodings.py` with tests for each encoding function — ReCWE output shape, RWSE output shape, spectral encoding shape, community labels valid range; verify with `pytest phases/phase03_structural_encoding/tests/test_structural_encodings.py -v`
- [ ] 11.2 Test spectral encoding on disconnected graph (multiple components); verify no errors
- [ ] 11.3 Test community detection on bipartite graph with known community structure; verify labels assign colluding pairs to same community

## 12. Educational content — lessons

- [ ] 12.1 Create `lessons/01-spectral-graph-theory.md` covering adjacency matrix, degree matrix, graph Laplacian, and eigenvalues; follow 5-part structure (Explain, Code, Visualize, Practice, Solution); Practice links to `../exercises/01-spectral-graph-theory.ipynb`; verify file renders correctly in Markdown
- [ ] 12.2 Create `lessons/02-laplacian-eigenvectors.md` covering normalized Laplacian, Fiedler vector, truncated eigendecomposition, and their use as positional encodings; follow 5-part structure; Practice links to `../exercises/02-laplacian-eigenvectors.ipynb`; verify code snippet runs
- [ ] 12.3 Create `lessons/03-random-walk-features.md` covering transition matrix, visitation frequency, RWSE, and ReCWE; follow 5-part structure; Practice links to `../exercises/03-random-walk-features.ipynb`; verify code snippet runs
- [ ] 12.4 Create `lessons/04-community-detection.md` covering modularity optimization, Louvain algorithm, resolution parameter, and community labels as features; follow 5-part structure; Practice links to `../exercises/04-community-detection.ipynb`; verify code snippet runs
- [ ] 12.5 Create `lessons/05-structural-features-for-fraud.md` covering fraudster structural patterns, combining encodings, and feature engineering strategies; follow 5-part structure; Practice links to `../exercises/05-structural-features-for-fraud.ipynb`; verify code snippet runs

## 13. Educational content — notebooks

- [ ] 13.1 Create `notebooks/01-exploring-eigenvectors.ipynb` — compute Laplacian eigenvectors on a synthetic fraud graph, visualize node embeddings in 2D, observe fraud clustering in spectral space; verify notebook executes end-to-end without errors
- [ ] 13.2 Create `notebooks/02-community-analysis.ipynb` — run Louvain community detection on the fraud graph, analyze community composition (fraud ratio per community), visualize community structure; verify notebook executes end-to-end without errors

## 14. Integration and verification

- [ ] 14.1 Run full test suite `pytest phases/phase03_structural_encoding/tests/ -v` and verify all tests pass including new structural encoding tests
- [ ] 14.2 Create run script `phases/phase03_structural_encoding/run_structural_encodings.py` that runs all 4 encodings end-to-end on synthetic data; verify output shows comparison table and best encoding recommendation; confirm exit code 0
