## 1. Setup

- [ ] 1.1 Verify `phases/phase05_pattern_detection/` directory structure exists with `src/phase05_pattern_detection/`, `tests/`, `lessons/`, `notebooks/`, `exercises/` subdirectories; create any missing directories
- [ ] 1.2 Verify `pyproject.toml` or equivalent includes phase 05 dependencies (networkx, torch_geometric, lightgbm, matplotlib); add if missing

## 2. Data loading

- [ ] 2.1 Create `phases/phase05_pattern_detection/src/phase05_pattern_detection/data_loader.py` with module docstring, imports (os, pathlib, torch, numpy), and `DataLoader` class with `__init__(self, data_dir)` storing path to shared `data/` directory at project root; verify with ast.parse
- [ ] 2.2 Implement `DataLoader.check_data_exists()` returning True if synthetic pattern data files already exist in `data/`, False otherwise; verify returns boolean
- [ ] 2.3 Implement `DataLoader.generate_synthetic_data(n_samples_per_pattern=500, noise_levels=[0.0, 0.1, 0.2], node_feature_dim=30)` generating synthetic fraud pattern graphs and saving to `data/` directory; verify files are created on disk
- [ ] 2.4 Implement `generate_fraud_patterns(pattern_type, n_samples)` creating graphs with known motifs — Mitme (collusion rings), Cascade (hub-and-spoke stars), Money Mule (sequential chains); verify each graph has ground-truth pattern labels and structural annotations
- [ ] 2.5 Implement `DataLoader.load_data()` loading generated pattern data from `data/` directory and returning list of PyG `Data` objects; verify returned objects are compatible with `motif_detector.py` and `subgraph_matcher.py` input expectations
- [ ] 2.6 Write integration test verifying `DataLoader` round-trip: generate → save → load → confirm data integrity and compatibility with downstream pattern detection modules; verify with pytest

## 3. Synthetic data generators

- [ ] 3.1 Create `phases/phase05_pattern_detection/src/phase05_pattern_detection/generation.py` with module docstring, imports (networkx, torch, numpy, random), and `_generate_base_graph(pattern_type, n_nodes, noise_level)` function stub; verify Python syntax with ast.parse
- [ ] 3.2 Implement `generate_mitme_graph(n_pairs=5, n_benign=20, noise_level=0.0, node_feature_dim=30)` generating a graph with circular bidirectional edges between colluding pairs plus benign nodes; verify returned PyG `Data` object has correct edge_index shape and node features
- [ ] 3.3 Implement `generate_cascade_graph(n_leaves=8, n_benign=15, noise_level=0.0, node_feature_dim=30)` generating a star-like subgraph with hub distributing to leaves plus return edges; verify returned PyG `Data` object has correct structure
- [ ] 3.4 Implement `generate_money_mule_graph(chain_length=5, n_chains=2, n_benign=20, noise_level=0.0, node_feature_dim=30)` generating chain-like subgraphs with sequential flow; verify returned PyG `Data` object has chain paths
- [ ] 3.5 Implement `generate_benign_graph(n_nodes=25, noise_level=0.0, node_feature_dim=30)` generating a random Erdős–Rényi graph with no fraud patterns; verify returned PyG `Data` object has graph_label == "benign"
- [ ] 3.6 Implement `generate_pattern_dataset(pattern_type, n_samples, noise_levels=[0.0, 0.1, 0.2, 0.3])` generating a dataset of labeled graphs across noise levels; verify dataset is a list of PyG `Data` objects with correct labels

## 4. Graph-level classifier training

- [ ] 4.1 Create `phases/phase05_pattern_detection/src/phase05_pattern_detection/classifier.py` with module docstring, imports (torch, torch_geometric.nn, torch_geometric.data), and `build_graph_gcn(input_dim, hidden_dim, n_classes)` function returning a GCN model with global mean pooling; verify model instantiates correctly with ast.parse
- [ ] 4.2 Implement `build_graph_gat(input_dim, hidden_dim, n_classes, n_heads=4)` returning a GAT model with global mean pooling for graph-level classification; verify model instantiates correctly
- [ ] 4.3 Implement `train_graph_model(model, data, epochs=100, lr=0.01)` training a graph classifier using cross-entropy loss; verify training prints loss per 10 epochs
- [ ] 4.4 Implement `evaluate_graph_model(model, data_list)` computing accuracy, Recall, Precision@K for each pattern class; verify output dict with per-class metrics
- [ ] 4.5 Implement `train_and_evaluate(pattern_type, n_train=2000, n_test=500, n_epochs=100)` training GCN and GAT on a pattern type and returning metrics for both models; verify output contains per-model and per-class metrics

## 5. Baseline comparison — LightGBM on pattern data

- [ ] 5.1 Implement `extract_graph_features_for_lightgbm(data_list, node_feature_dim)` extracting node-level features (degree, edge count, community membership, RWSE) from graph datasets for LightGBM input; verify output is numpy array compatible with LightGBM
- [ ] 5.2 Implement `train_lightgbm_baseline(X_train, y_train)` training a LightGBM classifier on extracted graph features; verify model trains without errors
- [ ] 5.3 Implement `evaluate_lightgbm_for_patterns(model, X_test, y_test)` evaluating LightGBM on test set with per-class Recall and Precision@K; verify output matches Graph Classifier output structure
- [ ] 5.4 Implement `compare_patterns_gcn_vs_lightgbm(pattern_type)` running full comparison for a single pattern type; verify output contains both model's metrics side by side

## 6. Pattern template generation

- [ ] 6.1 Create `phases/phase05_pattern_detection/src/phase05_pattern_detection/templates.py` with module docstring and imports; implement `generate_mitme_template(suspicious_pairs, confidence_score)` producing a structured template dict with keys: pattern_type, suspicious_nodes, suspicious_edges, structural_signature, recommended_actions, confidence; verify all keys present with ast.parse
- [ ] 6.2 Implement `generate_cascade_template(hub_node, leaf_nodes, confidence_score)` producing a template describing the star topology; verify all keys present
- [ ] 6.3 Implement `generate_money_mule_template(chain_nodes, chain_edges, confidence_score)` producing a template describing the sequential chain; verify all keys present
- [ ] 6.4 Implement `generate_pattern_template(pattern_type, detection_result)` dispatching to the correct generator based on pattern_type; verify output is a valid template dict
- [ ] 6.5 Implement `templates_to_json(templates)` serializing a list of templates to JSON string; verify output is valid JSON

## 7. Pattern template validation

- [ ] 7.1 Create `phases/phase05_pattern_detection/tests/test_templates.py` with tests for template generation — verify each template has all required keys (pattern_type, suspicious_nodes, suspicious_edges, structural_signature, recommended_actions, confidence); verify with pytest
- [ ] 7.2 Test `templates_to_json` produces valid JSON parseable by `json.loads`; verify with pytest

## 8. End-to-end pattern detection pipeline

- [ ] 8.1 Create `phases/phase05_pattern_detection/run_patterns.py` that generates synthetic data for all three pattern types, trains GCN/GAT, evaluates against LightGBM, generates templates, and prints summary; verify script runs without errors
- [ ] 8.2 Print summary table: pattern_type × model (GCN/GAT/LightGBM) → metrics (Recall, Precision@K); verify output is human-readable
- [ ] 8.3 Add stop decision check: if any pattern type has Recall < 0.50, print "⚠ Structured pattern detection requires higher-capacity models or better feature engineering"
- [ ] 8.4 Generate and save sample templates to `outputs/pattern_templates.json` for a test run; verify file exists and is valid JSON

## 9. Interactive exercises (exercises/)

- [ ] 9.1 Create `phases/phase05_pattern_detection/exercises/` directory
- [ ] 9.2 Create `01-graph-motifs.ipynb` with task to identify motifs in sample graphs
- [ ] 9.3 Create `02-subgraph-patterns.ipynb` with task to match patterns
- [ ] 9.4 Create `03-anomaly-detection.ipynb` with task to detect anomalies
- [ ] 9.5 Create `04-pattern-mining.ipynb` with task to mine frequent patterns
- [ ] 9.6 Create `05-fraud-patterns-in-practice.ipynb` with task to apply patterns to fraud data

## 10. Unit tests

- [ ] 10.1 Create `phases/phase05_pattern_detection/tests/test_generation.py` with tests for all three pattern generators — verify correct graph topology, correct labels, noise injection increases random edges; verify with `pytest phases/phase05_pattern_detection/tests/test_generation.py -v`
- [ ] 10.2 Create `phases/phase05_pattern_detection/tests/test_classifier.py` with tests for model building and training — verify GCN/GAT instantiate, train runs without errors, evaluate returns dict with required keys; verify with pytest
- [ ] 10.3 Create `phases/phase05_pattern_detection/tests/test_lightgbm.py` with tests for LightGBM baseline on graph features — verify feature extraction produces correct shapes, evaluation returns per-class metrics; verify with pytest
- [ ] 10.4 Create `phases/phase05_pattern_detection/tests/test_integration.py` with end-to-end test: generate small dataset (50 samples per pattern), train GCN, verify model trains and returns metrics dict; verify with pytest

## 11. Educational content — Lessons

- [ ] 11.1 Create `phases/phase05_pattern_detection/lessons/01-graph-motifs.md` covering graph motif definitions, counting algorithms, statistical significance, and motif roles in fraud detection; follow 5-part structure (Explain, Code, Visualize, Practice, Solution); link Practice section to `exercises/01-graph-motifs.ipynb`
- [ ] 11.2 Create `phases/phase05_pattern_detection/lessons/02-subgraph-patterns.md` covering subgraph template matching, VF2 algorithm, and pattern similarity metrics; follow 5-part structure; link Practice section to `exercises/02-subgraph-patterns.ipynb`
- [ ] 11.3 Create `phases/phase05_pattern_detection/lessons/03-anomaly-detection.md` covering structural anomalies, node anomalies, and community-based anomaly detection in graphs; follow 5-part structure; link Practice section to `exercises/03-anomaly-detection.ipynb`
- [ ] 11.4 Create `phases/phase05_pattern_detection/lessons/04-pattern-mining.md` covering frequent subgraph mining, gSpan algorithm, and discriminative pattern mining; follow 5-part structure; link Practice section to `exercises/04-pattern-mining.ipynb`
- [ ] 11.5 Create `phases/phase05_pattern_detection/lessons/05-fraud-patterns-in-practice.md` covering Mitme/Cascade/Mule detection workflows, investigation templates, and practical deployment considerations; follow 5-part structure; link Practice section to `exercises/05-fraud-patterns-in-practice.ipynb`

## 12. Educational content — Notebooks

- [ ] 12.1 Create `phases/phase05_pattern_detection/notebooks/01-motif-discovery.ipynb` with interactive motif discovery on synthetic fraud graphs — generate Mitme/Cascade/Mule patterns, count motifs, compare against random baselines, include visualizations
- [ ] 12.2 Create `phases/phase05_pattern_detection/notebooks/02-pattern-analysis.ipynb` with end-to-end pattern analysis — train GCN/GAT classifiers, generate investigation templates, visualize results

## 13. Integration and verification

- [ ] 13.1 Run full test suite `pytest phases/phase05_pattern_detection/tests/ -v` and verify all tests pass including new pattern detection tests
- [ ] 13.2 Run `phases/phase05_pattern_detection/run_patterns.py` end-to-end with synthetic data and verify: all three patterns generated, GCN and GAT train, metrics reported, templates generated, summary printed, stop decision check passes; confirm exit code 0
