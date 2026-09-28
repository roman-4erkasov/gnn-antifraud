## 1. Data loading

- [ ] 1.1 Create `phases/phase02_minimal_gnn/src/phase02_minimal_gnn/data_loader.py` with module docstring, imports (os, pathlib, pandas), and `DataLoader(data_dir="data/")` class stub; verify Python syntax
- [ ] 1.2 Implement `DataLoader.check_data_exists()` checking if IEEE-CIS data exists in `data/` directory; verify returns boolean
- [ ] 1.3 Implement `DataLoader.download_data()` downloading IEEE-CIS dataset to `data/` if missing (or providing instructions); verify data directory is populated
- [ ] 1.4 Implement `DataLoader.load_data()` loading transactions, identity, and labels into DataFrames; verify returns correct data structures
- [ ] 1.5 Implement `DataLoader.prepare_graph_data()` converting transaction data to graph format (edge_index, node_features, labels); verify output is compatible with minimal_gcn.py

## 2. GCN model definition

- [ ] 2.1 Create `phases/phase02_minimal_gnn/src/phase02_minimal_gnn/` directory with `__init__.py`, verify directory and file exist
- [ ] 2.2 Create `phases/phase02_minimal_gnn/src/phase02_minimal_gnn/minimal_gcn.py` with module docstring, imports (torch, torch_geometric, torch.nn), and `MinimalGCN` class stub; verify Python syntax via `python -c "import ast; ast.parse(open('phases/phase02_minimal_gnn/src/phase02_minimal_gnn/minimal_gcn.py').read())"`
- [ ] 2.3 Implement `MinimalGCN.__init__` with `GCNConv` layers: `GCNConv(input_dim, hidden_dim)` → ReLU → `GCNConv(hidden_dim, 1)` → Sigmoid; verify by instantiating and checking layer types
- [ ] 2.4 Implement `MinimalGCN.forward(x, edge_index)` performing message passing via PyG `GCNConv`; verify output shape is `[n_nodes, 1]` with sigmoid output in [0, 1]

## 3. GCN training loop

- [ ] 3.1 Implement `train_gcn(node_features, edge_index, node_labels, train_idx, val_idx, test_idx, hidden_dim=16, lr=0.01, epochs=200, patience=20)` with Adam optimizer, BCEWithLogitsLoss, and early stopping on validation loss; verify training prints epoch/loss per epoch
- [ ] 3.2 Implement GCN prediction function `predict_gcn(model, node_features, edge_index, idx)` returning raw probabilities for specified node indices; verify predictions in [0, 1]
- [ ] 3.3 Implement GCN calibration via `IsotonicCalibrator` on validation predictions; verify calibrated output differs from raw on synthetic data

## 4. Node label assignment from transaction data

- [ ] 4.1 Implement `assign_node_labels(txn_df, fraud_col='isFraud', user_col='card1', txn_id_col='TransactionID')` assigning fraud label to user if ANY transaction is fraudulent; verify on synthetic data that user with one fraud tx gets label 1
- [ ] 4.2 Implement `build_graph_for_gcn(data, graph_features)` constructing PyG `Data` object from node features (user + merchant aggregates) and edge index (user→merchant transactions); verify `Data` object has x, edge_index, y attributes

## 5. GCN evaluation and metrics

- [ ] 5.1 Implement `evaluate_gcn(model, node_features, edge_index, test_idx, y_test)` computing PR-AUC, ROC-AUC, Brier score, log-loss, Precision@K via `src.utils.metrics`; verify metrics dict has all required keys
- [ ] 5.2 Implement `evaluate_gcn_calibrated(model, node_features, edge_index, test_idx, y_test, calibrator)` applying calibrator before computing metrics; verify delta metrics reported

## 6. Cross-model comparison

- [ ] 6.1 Create comparison function `compare_gcn_vs_lightgbm(gcn_metrics, lgb_minimal_metrics, lgb_graph_metrics)` using `src.utils.metrics.compute_delta_metrics` to compute all deltas; verify delta values match arithmetic difference
- [ ] 6.2 Implement statistical significance comparison using `src.utils.metrics.paired_permutation_test` comparing GCN vs each LightGBM variant on same test set; verify p-values in [0, 1]
- [ ] 6.3 Implement stop decision function checking if GCN ΔPR-AUC vs graph-aware LightGBM < 0.01 or p > 0.05; verify with boundary conditions

## 7. End-to-end run script

- [ ] 7.1 Add GCN run to `phases/phase02_minimal_gnn/run_gcn.py` script: build PyG graph → train GCN → evaluate → compare from `phase02_minimal_gnn.minimal_gcn`; verify script runs end-to-end with synthetic data
- [ ] 7.2 Print summary table with GCN, minimal-LightGBM, and graph-aware-LightGBM metrics side by side; verify output is readable
- [ ] 7.3 Print stop decision output (GNN adds value / minimal value); verify text matches comparison criteria

## 8. Tests

- [ ] 8.1 Create `phases/phase02_minimal_gnn/tests/test_minimal_gcn.py` with tests for `MinimalGCN` — init layer structure, forward pass output shape, prediction returns valid probabilities; verify with `pytest phases/phase02_minimal_gnn/tests/test_minimal_gcn.py -v`
- [ ] 8.2 Add tests for node label assignment — single fraud tx labels user 1, all-clean labels user 0; verify correctness
- [ ] 8.3 Add tests for graph construction — `Data` object has correct node/edge counts from synthetic bipartite graph; verify shapes match expectations

## 9. Educational content

- [ ] 9.1 Create `phases/phase02_minimal_gnn/lessons/` directory; verify directory exists
- [ ] 9.2 Create `phases/phase02_minimal_gnn/lessons/01-introduction-to-gcn.md` following 5-part structure (Explain/Code/Visualize/Practice/Solution); verify file exists and contains all 5 sections
- [ ] 9.3 Create `phases/phase02_minimal_gnn/lessons/02-graph-construction-for-gnn.md` following 5-part structure; verify file exists and contains all 5 sections
- [ ] 9.4 Create `phases/phase02_minimal_gnn/lessons/03-training-loop.md` following 5-part structure; verify file exists and contains all 5 sections
- [ ] 9.5 Create `phases/phase02_minimal_gnn/lessons/04-evaluation-metrics.md` following 5-part structure; verify file exists and contains all 5 sections
- [ ] 9.6 Create `phases/phase02_minimal_gnn/lessons/05-comparison-with-lightgbm.md` following 5-part structure; verify file exists and contains all 5 sections
- [ ] 9.7 Create `phases/phase02_minimal_gnn/notebooks/` directory; verify directory exists
- [ ] 9.8 Create `phases/phase02_minimal_gnn/notebooks/01-exploring-gcn-behavior.ipynb` with sections for loading graph, training GCN, visualizing node embeddings, and analyzing message passing; verify notebook is valid JSON and can be opened
- [ ] 9.9 Create `phases/phase02_minimal_gnn/notebooks/02-hyperparameter-sweep.ipynb` with sections for testing different learning rates, hidden dimensions, and epochs; verify notebook is valid JSON and can be opened

## 10. Integration and verification

- [ ] 10.1 Run full test suite `pytest phases/phase02_minimal_gnn/tests/ -v` and verify all tests pass including new GCN tests
- [ ] 10.2 Run `phases/phase02_minimal_gnn/run_gcn.py` end-to-end with synthetic data (`--sample-limit 50`) and verify: GCN trains, metrics reported, comparison table printed, stop decision output; confirm exit code 0
