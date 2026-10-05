## 1. Environment scaffold with uv

- [x] 1.1 Create `phases/phase01_lightgbm_baseline/` directory with `pyproject.toml` specifying Python version and all dependencies (lightgbm, networkx, python-louvain, scikit-learn, pandas, numpy, scipy, jupyter, matplotlib, seaborn); verify file syntax via `python -c "import tomllib; tomllib.load(open('phases/phase01_lightgbm_baseline/pyproject.toml', 'rb'))"`
- [x] 1.2 Run `uv lock` from `phases/phase01_lightgbm_baseline/` to generate `uv.lock`; verify lockfile exists and is valid via `uv pip compile phases/phase01_lightgbm_baseline/pyproject.toml --python <version>` (dry-run check)
- [x] 1.3 Create `phases/phase01_lightgbm_baseline/README.md` describing the phase, how to `uv sync`, how to run tests, and how to run the pipeline

## 2. Module scaffold and imports

- [x] 2.1 Create `phases/phase01_lightgbm_baseline/src/phase01_lightgbm_baseline/` directory with `__init__.py`, verify directory and file exist
- [x] 2.2 Create `phases/phase01_lightgbm_baseline/src/phase01_lightgbm_baseline/data_loader.py` with module docstring, imports (pandas, os, json, pathlib, subprocess), and `DataLoader` class stub with methods `check_data_exists()`, `download_data()`, `load_data()`; verify Python syntax via `python -c "import ast; ast.parse(open('phases/phase01_lightgbm_baseline/src/phase01_lightgbm_baseline/data_loader.py').read())"`
- [x] 2.3 Implement `DataLoader.check_data_exists()` checking if `data/ieee-cis/` directory exists and contains the required CSV files; verify returns True if data present, False otherwise
- [x] 2.4 Implement `DataLoader.download_data()` to validate that `data/ieee-cis/` contains the required CSVs and, when missing, print download instructions and an optional Kaggle CLI command (no synthetic data fabrication); verify it reports present/missing correctly
- [x] 2.5 Implement `DataLoader.load_data()` loading IEEE-CIS transaction and identity data into pandas DataFrames and returning merged **raw** frames (no baseline features); baseline features are prepared after the split by the pipeline (tasks 14.1–14.2); verify returns tuple of (train_df, test_df) with expected raw columns
- [x] 2.6 Create `phases/phase01_lightgbm_baseline/src/phase01_lightgbm_baseline/graph_features.py` with module docstring, imports (networkx, pandas, numpy, scipy), and empty `GraphFeatureExtractor` class stub; verify Python syntax via `python -c "import ast; ast.parse(open('phases/phase01_lightgbm_baseline/src/phase01_lightgbm_baseline/graph_features.py').read())"`
- [x] 2.7 Create shared `src/utils/features.py` with `prepare_baseline_features()` building the 3-feature baseline table (transaction amount, time since account creation, number of prior transactions); verify Python syntax via `python -c "import ast; ast.parse(open('src/utils/features.py').read())"`

## 3. LightGBM wrapper — training and prediction

- [x] 3.1 Implement `LightGBMWrapper.__init__` accepting `scale_pos_weight` config, `n_estimators`, `max_depth`, `learning_rate`, `random_state` with defaults; verify by instantiating and checking attributes
- [x] 3.2 Implement `LightGBMWrapper.train(X_train, y_train, X_val, y_val)` building `lgb.Dataset` objects, calling `lgb.train()` with early stopping (patience=10, metric='binary_logloss'), storing best model and best iteration; verify by training on synthetic data and checking `best_iteration` attribute
- [x] 3.3 Implement `LightGBMWrapper.predict(X)` returning raw probability predictions via `model.predict(X)`; verify predictions are in [0, 1] range

## 4. Calibration integration

- [x] 4.1 Implement `LightGBMWrapper.calibrate(raw_probs, val_labels)` using `src.utils.calibration.IsotonicCalibrator`, returning calibrated probabilities; verify by checking calibrated output differs from raw on synthetic data
- [x] 4.2 Implement `LightGBMWrapper.full_pipeline(X_train, y_train, X_val, y_val, X_test, y_test)` orchestrating train → calibrate → predict → return metrics dict with `src.utils.metrics.compute_metrics`; verify metrics dict has keys: pr_auc, roc_auc, brier_score, log_loss, precision_at_k
- [x] 4.3 Report the calibration impact: compute metrics on raw and calibrated probabilities and include the Brier-score and log-loss deltas (before vs after calibration) in the returned metrics

## 5. Graph feature extraction

- [x] 5.1 Implement `GraphFeatureExtractor.__init__` storing user_col, merchant_col, txn_col defaults; verify instantiation
- [x] 5.2 Implement `GraphFeatureExtractor.build_bipartite_graph(txn_df, user_col, merchant_col)` creating NetworkX bipartite graph, adding nodes (users/merchants) and edges (transactions); verify graph has nodes and edges, check degree distribution
- [x] 5.3 Implement `GraphFeatureExtractor.compute_node_features(graph)` returning per-node features: degree (also exposed as `edge_count`, the number of incident transaction edges), clustering coefficient, node type (user/merchant), and degree centrality; verify features dict has expected keys and values
- [x] 5.4 Implement `GraphFeatureExtractor.detect_communities(graph, node_type='user')` running Louvain community detection via `community_louvain` on the user projection (two users are connected when they share a merchant), then mapping labels back to user nodes; returning dict of node→community_label; verify labels are integer labels in range [0, n_communities) for user nodes only
- [x] 5.5 Implement `GraphFeatureExtractor.compute_rwse_features(graph, k_max=8)` computing, per node, the diagonal of the k-step random-walk return probability matrix for k = 1..8 (a K-dimensional vector); verify output is a numeric array with shape (n_nodes, 8)

## 6. Graph-aware model training

- [x] 6.1 Implement `GraphFeatureExtractor.append_graph_features(X_train, X_test, graph_features)` where `graph_features` is the node-feature table from `compute_node_features`; join onto transaction DataFrames by `user_id` and `merchant_id` and return `(X_train, X_test)`; verify feature count increases from 3 to 3 + n_graph_features
- [x] 6.2 Create `phases/phase01_lightgbm_baseline/src/phase01_lightgbm_baseline/pipeline.py` with `train_graph_aware(data, graph_extractor)` (where `data` is the `(train_df, test_df)` loader tuple) that builds the graph → extracts features → joins to transaction data → trains `LightGBMWrapper` → returns `(model, metrics_dict)`; verify the model trains on the extended feature matrix

## 7. Comparison and statistical significance

- [x] 7.1 Implement `compute_comparison_report(minimal_metrics, graph_metrics)` using `src.utils.metrics.compute_delta_metrics` to compute deltas for all common metrics; verify delta dict has delta_pr_auc, delta_brier_score, etc.
- [x] 7.2 Implement `test_significance(raw_probs_minimal, raw_probs_graph)` using `src.utils.metrics.paired_permutation_test` to compute p-value; verify p-value in [0, 1] range
- [x] 7.3 Implement `generate_stop_decision(delta_pr_auc, p_value)` returning "graph adds minimal value" if delta_pr_auc < 0.01 or p_value > 0.05, otherwise "graph adds value"; verify with boundary conditions (delta=0.01, p=0.05)

## 8. End-to-end run script

- [x] 8.1 Create `phases/phase01_lightgbm_baseline/run_lightgbm_baseline.py` script inside the phase with CLI args: `--data-path` (default `../../data`), `--sample-limit` (stratified), `--random-state`, `--output-dir` (default `results`); verify script runs with `--help`
- [x] 8.2 In run script: import from the frozen package name `phase01_lightgbm_baseline` (`data_loader.DataLoader`, `lightgbm_baseline.LightGBMWrapper`, `graph_features.GraphFeatureExtractor`, `pipeline.train_graph_aware`), prepare baseline features, train the minimal model, calibrate, evaluate, build the graph, train graph-aware, compute comparison, and print a summary table; verify the script produces printed output with metrics for both models
- [x] 8.3 Add stop decision output to run script showing whether graphs add value based on comparison criteria
- [x] 8.4 Write run artifacts to `--output-dir`: `metrics.json` (metrics, deltas, p-value, stop decision), `model.txt`, `calibrator.pkl`, and a comparison table/plot; always compute Precision@K with `k = ceil(0.01 * len(test_set))`

## 9. Mini-lessons (lessons/)

- [x] 9.1 Create `phases/phase01_lightgbm_baseline/lessons/` directory with `00-evaluation-metrics-and-imbalance.md`, `01-introduction-to-lightgbm.md`, `02-baseline-features.md`, `03-calibration.md`, `04-graph-construction.md`, `05-graph-features.md`, `06-comparison-and-stop-decision.md`, `07-data-leakage-and-splits.md`; verify all files exist with valid markdown syntax
- [x] 9.2 Write lesson `00-evaluation-metrics-and-imbalance.md`: Explain (why accuracy misleads on the ~3.5% fraud rate; precision/recall trade-off; PR-AUC vs ROC-AUC; Precision@K; class weighting via `scale_pos_weight`) + Code (compute PR-AUC, ROC-AUC, and Precision@K on synthetic imbalanced data) + Visualize (PR curve, ROC curve, confusion matrix) + Practice (compare PR-AUC vs ROC-AUC under a shifted class balance, link to `../exercises/00-metrics-and-imbalance.ipynb`) + Solution
- [x] 9.3 Write lesson `01-introduction-to-lightgbm.md`: Explain (what is LightGBM, why gradient boosting, why it suits fraud detection) + Code (minimal example with synthetic data) + Visualize (feature importance plot) + Practice (predict what happens when you change n_estimators, link to `../exercises/01-n-estimators.ipynb`) + Solution (completed exercise with output)
- [x] 9.4 Write lesson `02-baseline-features.md`: Explain (why 3 features: amount, time since creation, n_prior_transactions) + Code (feature engineering from raw data) + Visualize (feature distributions, fraud rate per feature) + Practice (add a 4th feature, measure impact, link to `../exercises/02-add-4th-feature.ipynb`) + Solution
- [x] 9.5 Write lesson `03-calibration.md`: Explain (why calibration matters in fraud detection, what is isotonic regression) + Code (fit IsotonicCalibrator, compare calibrated vs raw) + Visualize (calibration curve, reliability diagram) + Practice (compare raw vs isotonic on Brier and log-loss, link to `../exercises/03-raw-vs-isotonic.ipynb`) + Solution
- [x] 9.6 Write lesson `04-graph-construction.md`: Explain (bipartite user-merchant graph, why it models fraud patterns) + Code (build graph with NetworkX, compute degrees) + Visualize (graph visualization with matplotlib, degree distribution) + Practice (subsample graph, compare statistics, link to `../exercises/04-subsampling-graph.ipynb`) + Solution
- [x] 9.7 Write lesson `05-graph-features.md`: Explain (node degree, clustering coefficient, Louvain communities, RWSE) + Code (compute each feature, append to transaction DataFrame) + Visualize (correlation matrix of graph features, community size distribution) + Practice (compare features by community, link to `../exercises/05-features-by-community.ipynb`) + Solution
- [x] 9.8 Write lesson `06-comparison-and-stop-decision.md`: Explain (how to compare models, PR-AUC delta, statistical significance) + Code (run paired permutation test, compute deltas) + Visualize (comparison bar chart, p-value distribution) + Practice (change delta threshold, see how stop decision changes, link to `../exercises/06-delta-threshold.ipynb`) + Solution
- [x] 9.9 Write lesson `07-data-leakage-and-splits.md`: Explain (what data leakage is; why graph features must be built from the training split only; time-based vs random splits and the role of `TransactionDT`) + Code (contrast a leaking vs leak-free graph-feature join) + Visualize (effect of leakage on held-out metrics) + Practice (compare random vs time-ordered split, link to `../exercises/07-leakage-and-splits.ipynb`) + Solution

## 10. Interactive exercises (exercises/)

- [x] 10.1 Create `phases/phase01_lightgbm_baseline/exercises/` directory
- [x] 10.2 Create `00-metrics-and-imbalance.ipynb` with task to compare PR-AUC and ROC-AUC under different class balances
- [x] 10.3 Create `01-n-estimators.ipynb` with setup code and task to compare n_estimators=50 vs 200
- [x] 10.4 Create `02-add-4th-feature.ipynb` with task to add 4th feature and measure impact
- [x] 10.5 Create `03-raw-vs-isotonic.ipynb` with task to compare raw and isotonic-calibrated scores (Brier, log-loss)
- [x] 10.6 Create `04-subsampling-graph.ipynb` with task to subsample graph and compare statistics
- [x] 10.7 Create `05-features-by-community.ipynb` with task to compare features by community
- [x] 10.8 Create `06-delta-threshold.ipynb` with task to change delta threshold and observe stop decision
- [x] 10.9 Create `07-leakage-and-splits.ipynb` with task to compare random vs time-ordered split and observe the effect of leakage on held-out metrics

## 11. Notebooks

- [x] 11.1 Create `phases/phase01_lightgbm_baseline/notebooks/` directory with `01-explore-baseline.ipynb`, `02-experiment-with-calibration.ipynb`, `03-play-with-graph-features.ipynb`; verify files are valid IPYNB JSON
- [x] 11.2 Write notebook `01-explore-baseline.ipynb`: load synthetic data, create 3 features, train LightGBM, evaluate with metrics — all cells should run in <3 minutes using synthetic data only
- [x] 11.3 Write notebook `02-experiment-with-calibration.ipynb`: load minimal model predictions, apply isotonic calibration, visualize calibration curves, and compare raw vs calibrated Brier score and log-loss
- [x] 11.4 Write notebook `03-play-with-graph-features.ipynb`: build bipartite graph, compute degree/clustering/community/RWSE features, append to transactions, train graph-aware model, compare metrics

## 12. Tests

- [x] 12.1 Create `phases/phase01_lightgbm_baseline/tests/test_lightgbm_baseline.py` with tests for `LightGBMWrapper` — init defaults, train on synthetic data, predict returns probabilities in [0,1], calibrate changes scores, full_pipeline returns metrics dict with all required keys; verify with `pytest phases/phase01_lightgbm_baseline/tests/test_lightgbm_baseline.py -v`
- [x] 12.2 Create `phases/phase01_lightgbm_baseline/tests/test_graph_features.py` with tests for `GraphFeatureExtractor` — build graph adds correct edges, compute_node_features returns expected keys, detect_communities returns integer labels, append_graph_features increases feature count; verify with `pytest phases/phase01_lightgbm_baseline/tests/test_graph_features.py -v`
- [x] 12.3 Create `phases/phase01_lightgbm_baseline/tests/test_comparison.py` with tests for `compute_comparison_report` (delta values match difference), `test_significance` (p-value between 0 and 1), `generate_stop_decision` (correct flag for delta < 0.01, p > 0.05); verify with `pytest phases/phase01_lightgbm_baseline/tests/test_comparison.py -v`
- [x] 12.4 Create `phases/phase01_lightgbm_baseline/tests/conftest.py` and the synthetic fixture `phases/phase01_lightgbm_baseline/tests/data/ieee_cis_test/` (small train/test transaction + identity CSVs) used by the unit tests and the e2e run; verify the fixture loads
- [x] 12.5 Add a test for `pipeline.train_graph_aware` (trains and returns model + metrics) and update the `append_graph_features` test for the canonical signature

## 13. Integration and verification

- [x] 13.1 Run full test suite `pytest phases/phase01_lightgbm_baseline/tests/ -v` and verify all tests pass; confirm no test failures
- [x] 13.2 Run `phases/phase01_lightgbm_baseline/run_lightgbm_baseline.py --data-path ./tests/data/ieee_cis_test --sample-limit 20` from the phase directory end-to-end with synthetic data and verify output shows minimal model metrics, graph model metrics, comparison deltas, and stop decision; confirm script exits with code 0
- [x] 13.3 Verify `uv sync` works in `phases/phase01_lightgbm_baseline/` and all lesson code blocks execute without errors (run `uv run python` with each lesson's code block as input)
- [x] 13.4 Verify all 3 notebooks execute end-to-end using `uv run jupyter execute` (or nbconvert) from the phase directory
- [x] 13.5 Verify `results/` contains `metrics.json`, `model.txt`, and `calibrator.pkl` after a run, that `precision_at_k` is present, and that `phases/phase01_lightgbm_baseline/README.md` documents setup and usage

## 14. Leakage-free baseline feature preparation

- [x] 14.1 Refactor `src/utils/features.py` to a fit/transform API: add `fit_baseline_feature_params(train_df)` returning the `TransactionDT` min/max (and any other fitted parameters), and `prepare_baseline_features(df, params)` applying them so `n_prior_transactions` is counted within the passed frame; verify syntax and that params fit on one frame apply to another
- [x] 14.2 Update `pipeline.train_graph_aware` and `run_lightgbm_baseline.py` to split first, fit feature params on the training split only, and apply the same params to validation/test for both the minimal and graph-aware paths; add a test asserting the fitted parameters and prior-count features do not depend on validation/test rows
- [x] 14.3 Update affected lessons (`02-baseline-features.md`, `07-data-leakage-and-splits.md`), exercises (`02-add-4th-feature.ipynb`, `07-leakage-and-splits.ipynb`), notebooks, and tests for the new fit/transform API; verify lesson code blocks and the 3 notebooks still execute

## 15. Community detection on the user projection

- [x] 15.1 Update `GraphFeatureExtractor.detect_communities` to run Louvain on the user projection (two users are connected when they share a merchant) and map labels back to user nodes, instead of partitioning the full bipartite graph; verify only user nodes receive labels
- [x] 15.2 Update `tests/test_graph_features.py` to assert community labels are produced for user nodes from the user projection and cover the new behavior
