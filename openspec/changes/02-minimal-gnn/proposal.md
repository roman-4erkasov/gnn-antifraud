## Why

Phase 1 of the fraud research established LightGBM baselines (minimal 3-feature and graph-aware). Now we need to answer the fundamental question: does a GNN model add predictive value over tree-based models when trained on the same graph structure? This is the first step in evaluating the GNN investment — if a simple GCN can't beat graph-aware LightGBM, the complexity of deeper graph models must be justified.

## What Changes

- Implement a single-layer GCN classifier using PyTorch Geometric (`GCNConv`) for node-level fraud detection on IEEE-CIS data
- Train GCN on the same graph structure used by graph-aware LightGBM (bipartite user-merchant-transaction graph)
- Apply isotonic calibration (existing `IsotonicCalibrator`) to GCN predictions
- Evaluate GCN with all standard metrics (PR-AUC, ROC-AUC, Brier score, log-loss, Precision@K)
- Compare GCN against both LightGBM baselines (minimal and graph-aware) with statistical significance testing
- **Stop decision:** If GCN ΔPR-AUC vs graph-aware LightGBM < 0.01 (or not statistically significant, p > 0.05), flag that GNN adds minimal value

## Capabilities

### New Capabilities

- `phase02-minimal-gnn`: Single-layer GCN classifier for fraud detection, calibration, and comparison against LightGBM baselines

### Modified Capabilities

- `phase01-lightgbm-baseline`: Adds GCN as a new model type in the comparison pipeline

## Impact

- Self-contained phase in `phases/phase02_minimal_gnn/`:
  - `pyproject.toml` and `uv.lock` with pinned dependencies
  - src/phase02_minimal_gnn/
    ├── __init__.py
    ├── minimal_gcn.py
    ├── data_prep.py
    └── data_loader.py
  - `tests/`
  - `lessons/`
    - `01-introduction-to-gcn.md`
    - `02-graph-construction-for-gnn.md`
    - `03-training-loop.md`
    - `04-evaluation-metrics.md`
    - `05-comparison-with-lightgbm.md`
  - `notebooks/`
    - `01-exploring-gcn-behavior.ipynb`
    - `02-hyperparameter-sweep.ipynb`
  - `exercises/`: interactive Jupyter notebooks for Practice sections (linked from lessons)
- Uses existing: `src/utils/calibration.py`, `src/utils/metrics.py`, `phases/phase01_lightgbm_baseline/src/phase01_lightgbm_baseline/graph_features.py`
- Dependency: PyTorch + PyTorch Geometric
- No breaking changes to existing codebase
