## 1. Environment scaffold with uv

- [x] 1.1 Create `phases/phase01_lightgbm_baseline/` directory with `pyproject.toml` specifying Python version and all dependencies (lightgbm, networkx, community-louvain, scikit-learn, pandas, numpy, scipy, jupyter, matplotlib, seaborn); verify file syntax via `python -c "import tomllib; tomllib.load(open('phases/phase01_lightgbm_baseline/pyproject.toml', 'rb'))"`
- [x] 1.2 Run `uv lock` from `phases/phase01_lightgbm_baseline/` to generate `uv.lock`; verify lockfile exists and is valid via `uv pip compile phases/phase01_lightgbm_baseline/pyproject.toml --python <version>` (dry-run check)

## 2. Module scaffold and imports

- [x] 2.1 Create `phases/phase01_lightgbm_baseline/src/phase01_lightgbm_baseline/` directory with `__init__.py`, verify directory and file exist
- [x] 2.2 Create `phases/phase01_lightgbm_baseline/src/phase01_lightgbm_baseline/data_loader.py` with module docstring, imports (pandas, os, urllib.request, zipfile), and `DataLoader` class stub with methods `check_data_exists()`, `download_data()`, `load_data()`; verify Python syntax via `python -c "import ast; ast.parse(open('phases/phase01_lightgbm_baseline/src/phase01_lightgbm_baseline/data_loader.py').read())"`
- [x] 2.3 Implement `DataLoader.check_data_exists()` checking if `data/ieee-cis/` directory exists and contains required files; verify returns True if data present, False otherwise
- [x] 2.4 Implement `DataLoader.download_data()` downloading IEEE-CIS dataset from source (or extracting from archive if provided), creating `data/ieee-cis/` directory structure; verify data directory created with expected files
- [x] 2.5 Implement `DataLoader.load_data()` loading IEEE-CIS transaction and identity data into pandas DataFrames, applying `prepare_baseline_features()` to generate 3-feature baseline; verify returns tuple of (train_df, test_df) with expected columns
- [x] 2.6 Create `phases/phase01_lightgbm_baseline/src/phase01_lightgbm_baseline/graph_features.py` with module docstring, imports (networkx, pandas, numpy, scipy), and empty `GraphFeatureExtractor` class stub; verify Python syntax via `python -c "import ast; ast.parse(open('phases/phase01_lightgbm_baseline/src/phase01_lightgbm_baseline/graph_features.py').read())"`

## 3. LightGBM wrapper — training and prediction

- [x] 3.1 Implement `LightGBMWrapper.__init__` accepting `scale_pos_weight` config, `n_estimators`, `max_depth`, `learning_rate`, `random_state` with defaults; verify by instantiating and checking attributes
- [x] 3.2 Implement `LightGBMWrapper.train(X_train, y_train, X_val, y_val)` building `lgb.Dataset` objects, calling `lgb.train()` with early stopping (patience=10, metric='binary_logloss'), storing best model and best iteration; verify by training on synthetic data and checking `best_iteration` attribute
- [x] 3.3 Implement `LightGBMWrapper.predict(X)` returning raw probability predictions via `model.predict(X)`; verify predictions are in [0, 1] range

## 4. Calibration integration

- [x] 4.1 Implement `LightGBMWrapper.calibrate(raw_probs, val_labels)` using `src.utils.calibration.IsotonicCalibrator`, returning calibrated probabilities; verify by checking calibrated output differs from raw on synthetic data
- [x] 4.2 Implement `LightGBMWrapper.full_pipeline(X_train, y_train, X_val, y_val, X_test, y_test)` orchestrating train → calibrate → predict → return metrics dict with `src.utils.metrics.compute_metrics`; verify metrics dict has keys: pr_auc, roc_auc, brier_score, log_loss, precision_at_k

## 5. Graph feature extraction

- [x] 5.1 Implement `GraphFeatureExtractor.__init__` storing user_col, merchant_col, txn_col defaults; verify instantiation
- [x] 5.2 Implement `GraphFeatureExtractor.build_bipartite_graph(txn_df, user_col, merchant_col)` creating NetworkX bipartite graph, adding nodes (users/merchants) and edges (transactions); verify graph has nodes and edges, check degree distribution
- [x] 5.3 Implement `GraphFeatureExtractor.compute_node_features(graph)` returning per-node features: degree, clustering coefficient, node type (user/merchant), and degree centrality; verify features dict has expected keys and values
- [x] 5.4 Implement `GraphFeatureExtractor.detect_communities(graph, node_type='user')` running Louvain community detection via `community_louvain`, returning dict of node→community_label; verify labels are integer labels in range [0, n_communities)
- [x] 5.5 Implement `GraphFeatureExtractor.compute_rwse_features(graph, n_walks=10, walk_length=8)` computing random walk statistics: mean shortest path, visit frequency, degree-weighted centrality from walks; verify output is numeric array with expected shape

## 6. Graph-aware model training

- [x] 6.1 Implement `GraphFeatureExtractor.append_graph_features(X_train, X_test, graph_df)` joining graph-derived node features onto transaction DataFrames by user_id and merchant_id; verify feature count increases from 3 to 3 + n_graph_features
- [x] 6.2 Create function `train_graph_aware(data, graph_extractor)` that builds graph → extracts features → joins to transaction data → trains `LightGBMWrapper` → returns model and metrics; verify model trains on extended feature matrix

## 7. Comparison and statistical significance

- [x] 7.1 Implement `compute_comparison_report(minimal_metrics, graph_metrics)` using `src.utils.metrics.compute_delta_metrics` to compute deltas for all common metrics; verify delta dict has delta_pr_auc, delta_brier_score, etc.
- [x] 7.2 Implement `test_significance(raw_probs_minimal, raw_probs_graph)` using `src.utils.metrics.paired_permutation_test` to compute p-value; verify p-value in [0, 1] range
- [x] 7.3 Implement `generate_stop_decision(delta_pr_auc, p_value)` returning "graph adds minimal value" if delta_pr_auc < 0.01 or p_value > 0.05, otherwise "graph adds value"; verify with boundary conditions (delta=0.01, p=0.05)

## 8. End-to-end run script

- [x] 8.1 Create `phases/phase01_lightgbm_baseline/run_lightgbm_baseline.py` script inside the phase with CLI args: `--data-path` (default `../data`), `--sample-limit`, `--random-state`; verify script runs with `--help`
- [x] 8.2 In run script: load data via `phases.phase01_lightgbm_baseline.src.phase01_lightgbm_baseline.data_loader.DataLoader`, prepare baseline features, train minimal model in `phases.phase01_lightgbm_baseline.src.phase01_lightgbm_baseline.lightgbm_baseline`, calibrate, evaluate, build graph via `phases.phase01_lightgbm_baseline.src.phase01_lightgbm_baseline.graph_features`, train graph-aware, compute comparison, print summary table; verify script produces printed output with metrics for both models
- [x] 8.3 Add stop decision output to run script showing whether graphs add value based on comparison criteria

## 9. Mini-lessons (lessons/)

- [x] 9.1 Create `phases/phase01_lightgbm_baseline/lessons/` directory with `01-introduction-to-lightgbm.md`, `02-baseline-features.md`, `03-calibration.md`, `04-graph-construction.md`, `05-graph-features.md`, `06-comparison-and-stop-decision.md`; verify all files exist with valid markdown syntax
- [x] 9.2 Write lesson `01-introduction-to-lightgbm.md`: Explain (what is LightGBM, why gradient boosting, why it suits fraud detection) + Code (minimal example with synthetic data) + Visualize (feature importance plot) + Practice (predict what happens when you change n_estimators, link to `../exercises/01-n-estimators.ipynb`) + Solution (completed exercise with output)
- [x] 9.3 Write lesson `02-baseline-features.md`: Explain (why 3 features: amount, time since creation, n_prior_transactions) + Code (feature engineering from raw data) + Visualize (feature distributions, fraud rate per feature) + Practice (add a 4th feature, measure impact, link to `../exercises/02-add-4th-feature.ipynb`) + Solution
- [x] 9.4 Write lesson `03-calibration.md`: Explain (why calibration matters in fraud detection, what is isotonic regression) + Code (fit IsotonicCalibrator, compare calibrated vs raw) + Visualize (calibration curve, reliability diagram) + Practice (try smoothing parameter, link to `../exercises/03-smoothing-parameter.ipynb`) + Solution
- [x] 9.5 Write lesson `04-graph-construction.md`: Explain (bipartite user-merchant graph, why it models fraud patterns) + Code (build graph with NetworkX, compute degrees) + Visualize (graph visualization with matplotlib, degree distribution) + Practice (subsample graph, compare statistics, link to `../exercises/04-subsampling-graph.ipynb`) + Solution
- [x] 9.6 Write lesson `05-graph-features.md`: Explain (node degree, clustering coefficient, Louvain communities, RWSE) + Code (compute each feature, append to transaction DataFrame) + Visualize (correlation matrix of graph features, community size distribution) + Practice (compare features by community, link to `../exercises/05-features-by-community.ipynb`) + Solution
- [x] 9.7 Write lesson `06-comparison-and-stop-decision.md`: Explain (how to compare models, PR-AUC delta, statistical significance) + Code (run paired permutation test, compute deltas) + Visualize (comparison bar chart, p-value distribution) + Practice (change delta threshold, see how stop decision changes, link to `../exercises/06-delta-threshold.ipynb`) + Solution

## 10. Interactive exercises (exercises/)

- [x] 10.1 Create `phases/phase01_lightgbm_baseline/exercises/` directory
- [x] 10.2 Create `01-n-estimators.ipynb` with setup code and task to compare n_estimators=50 vs 200
- [x] 10.3 Create `02-add-4th-feature.ipynb` with task to add 4th feature and measure impact
- [x] 10.4 Create `03-smoothing-parameter.ipynb` with task to try different smoothing parameters
- [x] 10.5 Create `04-subsampling-graph.ipynb` with task to subsample graph and compare statistics
- [x] 10.6 Create `05-features-by-community.ipynb` with task to compare features by community
- [x] 10.7 Create `06-delta-threshold.ipynb` with task to change delta threshold and observe stop decision

## 11. Notebooks

- [x] 11.1 Create `phases/phase01_lightgbm_baseline/notebooks/` directory with `01-explore-baseline.ipynb`, `02-experiment-with-calibration.ipynb`, `03-play-with-graph-features.ipynb`; verify files are valid IPYNB JSON
- [x] 11.2 Write notebook `01-explore-baseline.ipynb`: load synthetic data, create 3 features, train LightGBM, evaluate with metrics — all cells should run in <3 minutes using synthetic data only
- [x] 11.3 Write notebook `02-experiment-with-calibration.ipynb`: load minimal model predictions, apply isotonic calibration with different parameters, visualize calibration curves, compare Brier scores
- [x] 11.4 Write notebook `03-play-with-graph-features.ipynb`: build bipartite graph, compute degree/clustering/community/RWSE features, append to transactions, train graph-aware model, compare metrics

## 12. Tests

- [x] 12.1 Create `phases/phase01_lightgbm_baseline/tests/test_lightgbm_baseline.py` with tests for `LightGBMWrapper` — init defaults, train on synthetic data, predict returns probabilities in [0,1], calibrate changes scores, full_pipeline returns metrics dict with all required keys; verify with `pytest phases/phase01_lightgbm_baseline/tests/test_lightgbm_baseline.py -v`
- [x] 12.2 Create `phases/phase01_lightgbm_baseline/tests/test_graph_features.py` with tests for `GraphFeatureExtractor` — build graph adds correct edges, compute_node_features returns expected keys, detect_communities returns integer labels, append_graph_features increases feature count; verify with `pytest phases/phase01_lightgbm_baseline/tests/test_graph_features.py -v`
- [x] 12.3 Create `phases/phase01_lightgbm_baseline/tests/test_comparison.py` with tests for `compute_comparison_report` (delta values match difference), `test_significance` (p-value between 0 and 1), `generate_stop_decision` (correct flag for delta < 0.01, p > 0.05); verify with `pytest phases/phase01_lightgbm_baseline/tests/test_comparison.py -v`

## 13. Integration and verification

- [x] 13.1 Run full test suite `pytest phases/phase01_lightgbm_baseline/tests/ -v` and verify all tests pass; confirm no test failures
- [x] 13.2 Run `phases/phase01_lightgbm_baseline/run_lightgbm_baseline.py --data-path ./tests/data/ieee_cis_test --sample-limit 20` from the phase directory end-to-end with synthetic data and verify output shows minimal model metrics, graph model metrics, comparison deltas, and stop decision; confirm script exits with code 0
- [x] 13.3 Verify `uv sync` works in `phases/phase01_lightgbm_baseline/` and all lesson code blocks execute without errors (run `uv run python` with each lesson's code block as input)
- [x] 13.4 Verify all 3 notebooks execute end-to-end using `uv run jupyter execute` (or nbconvert) from the phase directory
