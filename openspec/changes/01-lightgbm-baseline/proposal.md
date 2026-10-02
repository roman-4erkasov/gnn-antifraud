## Why

Fraud detection is a classic classification problem where LightGBM is a strong baseline. Before building GNNs, we need to establish the simplest possible baseline (3 features) and a graph-aware version (with node degrees, community labels, RWSE values). This establishes the floor -- if graph features don't add value over tabular data, the entire GNN investment is unnecessary.

Beyond code, we want to deliver this as a **learning experience**: mini-lessons that explain each concept step by step, plus a sandbox where the reader can experiment freely with their own parameters and data. This way the project teaches antifraud concepts while building a real baseline.

## What Changes

- Create a set of mini-lessons on LightGBM and graph-derived features: each lesson explains a concept, provides code, visualizations, and a practical exercise
- Build a sandbox `phases/phase01_lightgbm_baseline/` with lessons, notebooks, and isolated environment for hands-on experimentation
- Each phase is self-contained with its own `pyproject.toml` and dependencies
- Each phase = a self-contained lesson with code, lessons, notebooks, and isolated environment (explain -> code -> visualize -> practice -> solution)
- Implement LightGBM baseline with exactly 3 features: transaction amount, time since account creation, number of prior transactions
- Apply isotonic calibration to baseline predictions (leveraging existing `src/utils/calibration.py`)
- Evaluate baseline with all metrics (PR-AUC, ROC-AUC, Brier score, log-loss, Precision@K) on IEEE-CIS
- Implement graph-aware LightGBM by injecting graph-derived features (node degree, edge count, community membership, RWSE values)
- Compare tabular vs graph-aware LightGBM and compute deltas
- **Stop decision:** If delta PR-AUC < 0.01 (or not statistically significant), flag that graphs add minimal value
- Extend IEEE-CIS loader to extract graph structure (transactions as edges, users as nodes) for graph feature computation

## Capabilities

### New Capabilities
- `phase01_lightgbm_baseline`: LightGBM-based fraud detection with minimal features (3) and graph-aware variants, calibration, and stop-early evaluation criteria

### Modified Capabilities
<!-- None yet -- this is the first change, the scoring capability will be created by the spec -->

## Impact

- Self-contained phase in `phases/phase01_lightgbm_baseline/`:
  - `pyproject.toml` and `uv.lock` with pinned dependencies
  - `src/phase01_lightgbm_baseline/`: lightgbm_baseline.py, graph_features.py, data_loader.py
  - `run_lightgbm_baseline.py`: end-to-end run script (in-phase, consistent with phases 02–10)
  - `tests/`: test_lightgbm_baseline.py, test_graph_features.py, test_comparison.py
  - `lessons/`: markdown files with code, explanations, visualizations, exercises and solutions
  - `notebooks/`: interactive Jupyter notebooks for hands-on experimentation
  - `exercises/`: interactive Jupyter notebooks for Practice sections (linked from lessons)
- Uses existing utilities: `src/utils/calibration.py`, `src/utils/metrics.py`
- Data stored in `data/ieee-cis/` (shared across phases); `data_loader.py` checks existence and downloads if missing
- No breaking changes to existing codebase
