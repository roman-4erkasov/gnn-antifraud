## 1. TGN (Temporal Graph Network) implementation

- [ ] 1.1 Create `phases/phase06_temporal_graphs/src/phase06_temporal_graphs/tgn.py` with module docstring, imports (torch, torch.nn, torch_scatter if available), and `TGNModel(node_dim, memory_dim, relation_dim, attention_dim)` class stub; verify Python syntax
- [ ] 1.2 Implement `_RelationEncoder` — relation encoder that maps (src_type, edge_type, dst_type) triplets to relation vectors of dimension `relation_dim`; verify output shape is correct
- [ ] 1.3 Implement `_MemoryUpdater` — node memory with RNN (LSTM or GRU) that updates memory states based on received messages; verify memory shape is (n_nodes, memory_dim) after update
- [ ] 1.4 Implement `_AttentionUpdate` — attention mechanism that weighs incoming messages when updating node memory, using attention dimension `attention_dim`; verify attention weights sum to 1
- [ ] 1.5 Implement `TGNModel.forward(node_features, edge_index, edge_types, time_diffs, memory)` processing a batch of time-stamped edges, updating memory, and producing node embeddings; verify output shape is (batch_edges, node_dim)
- [ ] 1.6 Implement `TGNModel.get_node_embeddings(node_ids)` producing current memory states for a given set of node IDs; verify output shape matches requested node count

## 2. GRN (Graph Recurrent Network) implementation

- [ ] 2.1 Create `phases/phase06_temporal_graphs/src/phase06_temporal_graphs/grn.py` with module docstring, imports (torch, torch.nn, torch_geometric.nn), and `GRNModel(gnn_encoder, rnn_type="lstm", hidden_dim=64)` class stub; verify Python syntax
- [ ] 2.2 Implement `GRNModel.forward(snapshot_sequence)` processing a sequence of PyG Data objects (snapshots) through GCN encoder then RNN layer; verify output is a list of hidden states (one per snapshot)
- [ ] 2.3 Implement `_GCNEncoder` — a simple GCN layer that encodes node features and edge_index from a snapshot into hidden representations; verify encoding produces (n_nodes, hidden_dim) output
- [ ] 2.4 Implement RNN state passing between consecutive snapshots — hidden state from snapshot t is passed as input to snapshot t+1's RNN; verify temporal coherence of hidden states
- [ ] 2.5 Implement `GRNModel.predict(snapshot)` producing final node predictions from the last snapshot's hidden state; verify output shape is (n_nodes, n_classes)
- [ ] 2.6 Implement `GRNModel.train_step(snapshot_sequence, labels)` performing a training step with cross-entropy loss on the specified snapshot's labels; verify loss is a scalar tensor

## 3. Rolling-window GCN implementation

- [ ] 3.1 Create `phases/phase06_temporal_graphs/src/phase06_temporal_graphs/rolling_gcn.py` with module docstring, imports (torch, torch_geometric.nn, networkx), and `RollingWindowGCN(window_size=None, n_windows=10)` class stub; verify Python syntax
- [ ] 3.2 Implement `_build_snapshots(edge_list, timestamps, n_windows)` splitting time-stamped edges into K consecutive windows based on edge ordering; verify each window has approximately equal edges and preserves temporal order
- [ ] 3.3 Implement `_build_static_graph(snapshot_edges, node_features)` converting window edges + features into a PyG Data object; verify output is a valid PyG Data with correct edge_index and x
- [ ] 3.4 Implement `RollingWindowGCN.train_on_windows(snapshots, labels)` training a GCN model independently on each temporal snapshot; verify n_models == n_windows and each model trains successfully
- [ ] 3.5 Implement `RollingWindowGCN.predict_latest()` producing predictions for the latest window using the model trained on that window (or nearest available); verify output shape is correct
- [ ] 3.6 Implement `RollingWindowGCN.evaluate_all_windows(test_labels)` evaluating each window's model on its test set and reporting per-window and aggregate PR-AUC; verify output contains per-window metrics

## 4. Temporal leakage detection

- [ ] 4.1 Create `phases/phase06_temporal_graphs/src/phase06_temporal_graphs/leakage_checker.py` with module docstring and `check_temporal_leakage(train_edges, test_edges, train_timestamps, test_timestamps)` function stub; verify Python syntax
- [ ] 4.2 Implement leakage detection — verify all test timestamps are strictly greater than max(train_timestamps); verify output is a boolean
- [ ] 4.3 Implement `_count_leaking_edges(train_edges, test_edges)` counting edges that appear in both train and test sets (regardless of timestamp); verify output is an integer >= 0
- [ ] 4.4 Implement `report_leakage(train_edges, test_edges, train_timestamps, test_timestamps)` returning a dict with fields: has_leakage (bool), max_train_timestamp, n_leaking_edges, leakage_details (list of leaking edge info); verify all fields present
- [ ] 4.5 Implement `apply_temporal_split(dataset, split_ratio=0.8)` performing a strict time-based train/test split ensuring no leakage; verify all test timestamps > max(train timestamps)

## 5. Temporal dataset loaders

- [ ] 5.1 Create `phases/phase06_temporal_graphs/src/phase06_temporal_graphs/data_loader.py` with module docstring, imports (os, urllib.request, zipfile, pandas, torch_geometric.data), and `TemporalDataLoader` class stub with methods `check_data_exists()`, `download_data()`, `load_pay_at_pump()`, `load_sungkyunkwan()`, `load_naver_plus_bank()`; verify Python syntax
- [ ] 5.2 Implement `TemporalDataLoader.check_data_exists()` checking if `data/pay-at-pump/`, `data/sungkyunkwan/`, `data/naver-plus-bank/` directories exist with required files; verify returns True if all data present, False with list of missing datasets otherwise
- [ ] 5.3 Implement `TemporalDataLoader.download_data()` downloading Pay-At-Pump, Sungkyunkwan, and Naver Plus Bank datasets into `data/` directory, extracting archives if needed; verify data directories created with expected files
- [ ] 5.4 Implement `TemporalDataLoader.load_pay_at_pump()` loading Pay-At-Pump dataset from `data/pay-at-pump/` with temporal features (timestamps, temporal splits); verify returns list of PyG Data objects with edge timestamps
- [ ] 5.5 Implement `TemporalDataLoader.load_sungkyunkwan()` loading Sungkyunkwan dataset from `data/sungkyunkwan/` as bipartite user-merchant graph snapshots with anomaly labels; verify returns list of PyG Data objects
- [ ] 5.6 Implement `TemporalDataLoader.load_naver_plus_bank()` loading Naver Plus Bank dataset from `data/naver-plus-bank/` with time-stamped transactions and anomaly labels; verify returns time-stamped edge events and node labels
- [ ] 5.7 Implement `_validate_temporal_data(loader_output)` utility that checks temporal data structure (correct timestamps, labels present, edges valid); verify it raises on malformed data

## 6. Model training and evaluation

- [ ] 6.1 Implement `train_tgn(dataset, epochs=50, lr=0.001)` training TGN on temporal edge events with cross-entropy loss; verify training prints loss per epoch, returns trained model
- [ ] 6.2 Implement `train_grn(snapshots, epochs=30, lr=0.001)` training GRN on snapshot sequences; verify training prints loss per epoch, returns trained model
- [ ] 6.3 Implement `train_rolling_gcn(snapshots, epochs=20, lr=0.001)` training GCN per window in rolling fashion; verify training prints loss per window, returns trained model
- [ ] 6.4 Implement `evaluate_tgn(model, test_events)` evaluating trained TGN on test edges/events and reporting PR-AUC, Brier score; verify output dict with metrics
- [ ] 6.5 Implement `evaluate_grn(model, test_snapshots)` evaluating trained GRN on test snapshots and reporting PR-AUC, Brier score; verify output dict with metrics
- [ ] 6.6 Implement `evaluate_rolling_gcn(model, test_snapshots)` evaluating trained rolling-GCN on test windows and reporting PR-AUC, Brier score; verify output dict with metrics
- [ ] 6.7 Implement `evaluate_static_gcn(snapshots, epochs=50)` evaluating static GCN on temporal snapshots (baseline); verify output dict with metrics comparable to temporal models

## 7. Temporal model comparison

- [ ] 7.1 Implement `run_temporal_comparison(dataset_name, loaders, all_models)` running all temporal models + static baselines on a dataset and returning comparison results; verify output dict with per-model metrics
- [ ] 7.2 Implement `_compute_comparison_metrics(models_results)` computing PR-AUC, Brier score, inference latency for each model; verify output is a summary dataframe or dict
- [ ] 7.3 Implement `print_comparison_table(results)` printing a human-readable comparison table of all models per dataset; verify output is well-formatted
- [ ] 7.4 Implement comparison on Pay-At-Pump dataset; verify results include all 5 models (TGN, GRN, rolling-GCN, static-GCN, LightGBM) with PR-AUC and Brier score
- [ ] 7.5 Implement comparison on Sungkyunkwan dataset; verify results include all 5 models with PR-AUC and Brier score
- [ ] 7.6 Implement comparison on Naver Plus Bank dataset; verify results include all 5 models with PR-AUC and Brier score
- [ ] 7.7 Implement stop decision check: if temporal models do not improve PR-AUC by >= 0.02 over static GCN on any dataset, print warning flag

## 8. XAI validation on temporal predictions

- [ ] 8.1 Implement `run_xai_on_temporal(model, snapshot, layer_name="conv_layer")` running GNNExplainer on temporal model predictions for a snapshot; verify output contains subgraph and feature explanations
- [ ] 8.2 Implement `_analyze_temporal_explanation(explanation, snapshot_timestamps)` analyzing whether highlighted edges/nodes are concentrated in recent time periods; verify output includes temporal distribution score
- [ ] 8.3 Implement `compare_xai_temporal_vs_static(model, snapshot)` comparing XAI explanations from temporal model vs. static GCN on the same snapshot; verify output includes explanation quality comparison
- [ ] 8.4 Implement `_report_xai_findings(xai_results)` producing structured findings on whether temporal models produce more temporally coherent explanations; verify output dict with findings

## 9. End-to-end temporal pipeline

- [ ] 9.1 Create `phases/phase06_temporal_graphs/run_temporal.py` that loads all three temporal datasets, trains all models, evaluates, compares, runs XAI validation, and prints summary; verify script runs without errors
- [ ] 9.2 Print summary table: dataset × model → metrics (PR-AUC, Brier score, latency); verify output is human-readable
- [ ] 9.3 Print XAI findings summary: whether temporal explanations capture temporal patterns better than static; verify output is human-readable
- [ ] 9.4 Add stop decision check: if temporal models do not improve PR-AUC by >= 0.02 over static GCN on ANY dataset, print "⚠ Temporal modeling adds minimal value for these datasets"
- [ ] 9.5 Save comparison results to `outputs/temporal_comparison.json`; verify file exists and is valid JSON

## 10. Interactive exercises (exercises/)

- [ ] 10.1 Create `phases/phase06_temporal_graphs/exercises/` directory
- [ ] 10.2 Create `01-introduction-to-temporal-graphs.ipynb` with task to build and visualize temporal snapshots
- [ ] 10.3 Create `02-temporal-graph-networks.ipynb` with task to configure and run TGN memory updates
- [ ] 10.4 Create `03-rolling-window-training.ipynb` with task to split data into windows and train per-window models
- [ ] 10.5 Create `04-temporal-leakage.ipynb` with task to detect and prevent temporal leakage
- [ ] 10.6 Create `05-grn-and-attention.ipynb` with task to run GRN on snapshot sequences
- [ ] 10.7 Create `06-temporal-evaluation.ipynb` with task to compare temporal vs static model metrics

## 11. Unit tests

- [ ] 11.1 Create `phases/phase06_temporal_graphs/tests/test_tgn.py` with tests for TGN model — verify relation encoder output shape, memory update correctness, forward pass produces correct embeddings; verify with pytest
- [ ] 11.2 Create `phases/phase06_temporal_graphs/tests/test_grn.py` with tests for GRN model — verify GCN encoder output shape, RNN state passing, predict produces correct output; verify with pytest
- [ ] 11.3 Create `phases/phase06_temporal_graphs/tests/test_rolling_gcn.py` with tests for rolling-window GCN — verify snapshot building, window splitting, training per window, predictions; verify with pytest
- [ ] 11.4 Create `phases/phase06_temporal_graphs/tests/test_leakage_checker.py` with tests for leakage detection — verify leakage detection on known leaking/non-leaking splits, temporal split correctness; verify with pytest
- [ ] 11.5 Create `phases/phase06_temporal_graphs/tests/test_dataset_loaders.py` with tests for temporal loaders — verify Pay-At-Pump temporal split, Sungkyunkwan snapshot loading, Naver Plus Bank loading; verify with pytest
- [ ] 11.6 Create `phases/phase06_temporal_graphs/tests/test_integration.py` with end-to-end test: generate small temporal graph, train all models, verify all models train and return metrics dict; verify with pytest

## 12. Educational content — lessons

- [ ] 12.1 Create `phases/phase06_temporal_graphs/lessons/01-introduction-to-temporal-graphs.md` with 5-part structure (Explain, Code, Visualize, Practice, Solution) covering why temporal modeling matters, static vs. temporal graphs, and time-stamped edges; verify all 5 sections present; link Practice section to `exercises/01-introduction-to-temporal-graphs.ipynb`
- [ ] 12.2 Create `phases/phase06_temporal_graphs/lessons/02-temporal-graph-networks.md` with 5-part structure covering TGN architecture: node memory, relation encoder, attention-based memory update; verify all 5 sections present; link Practice section to `exercises/02-temporal-graph-networks.ipynb`
- [ ] 12.3 Create `phases/phase06_temporal_graphs/lessons/03-rolling-window-training.md` with 5-part structure covering rolling-window GCN: window splitting, per-snapshot training, aggregation strategies; verify all 5 sections present; link Practice section to `exercises/03-rolling-window-training.ipynb`
- [ ] 12.4 Create `phases/phase06_temporal_graphs/lessons/04-temporal-leakage.md` with 5-part structure covering temporal leakage: definition, detection, prevention, and time-based train/test splits; verify all 5 sections present; link Practice section to `exercises/04-temporal-leakage.ipynb`
- [ ] 12.5 Create `phases/phase06_temporal_graphs/lessons/05-grn-and-attention.md` with 5-part structure covering GRN architecture: GCN encoder + RNN, and temporal attention mechanisms; verify all 5 sections present; link Practice section to `exercises/05-grn-and-attention.ipynb`
- [ ] 12.6 Create `phases/phase06_temporal_graphs/lessons/06-temporal-evaluation.md` with 5-part structure covering evaluating temporal models: PR-AUC over time, Brier score, latency comparison; verify all 5 sections present; link Practice section to `exercises/06-temporal-evaluation.ipynb`

## 13. Educational content — notebooks

- [ ] 13.1 Create `phases/phase06_temporal_graphs/notebooks/01-temporal-dynamics.ipynb` with cells that visualize temporal graph dynamics: plot event timelines, build temporal snapshots, observe graph structure evolution; verify notebook runs without errors
- [ ] 13.2 Create `phases/phase06_temporal_graphs/notebooks/02-rolling-training.ipynb` with cells that train rolling-window GCN step-by-step: split data into windows, train per-window models, compare predictions across windows; verify notebook runs without errors
- [ ] 13.3 Create `phases/phase06_temporal_graphs/notebooks/03-leakage-analysis.ipynb` with cells that demonstrate temporal leakage: create a leaking split, measure inflated metrics, apply proper temporal split, and compare; verify notebook runs without errors

## 14. Integration and verification

- [ ] 14.1 Run full test suite `pytest phases/phase06_temporal_graphs/tests/ -v` and verify all tests pass including new temporal tests
- [ ] 14.2 Run `phases/phase06_temporal_graphs/run_temporal.py` end-to-end with synthetic temporal data and verify: all three datasets processed, all models train, metrics reported, XAI validated, comparison table printed, stop decision check passes; confirm exit code 0
