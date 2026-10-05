## Why

Phases 1–6 demonstrated fraud detection on static graph representations. However, real-world fraud is inherently temporal — transaction sequences, timing patterns, and evolving network structures carry critical signals that static graphs cannot capture. The IEEE-CIS dataset, for example, has explicit timestamps. Pay-At-Pump and Sungkyunkwan datasets were designed specifically for temporal anomaly detection. Without temporal modeling, we risk missing time-dependent fraud patterns (e.g., accounts that appear normal for weeks then suddenly engage in coordinated fraud). This phase addresses that gap by introducing temporal graph models and comparing them against static baselines.

## What Changes

- Implement TGN (Temporal Graph Network) with node memory and relation encoder for processing time-stamped edge events
- Implement GRN (Graph Recurrent Network) with snapshot-sequence processing for temporal graph analysis
- Implement rolling-window GCN: split temporal graph into fixed time windows, build static snapshots, train GCN per window
- Validate all temporal models on three temporal datasets: Pay-At-Pump, Sungkyunkwan, Naver Plus Bank
- Compare temporal models (TGN, GRN, rolling-GCN) against static GCN and LightGBM baselines on PR-AUC, Brier score, and inference latency
- Implement temporal leakage checker to verify no test-period edges leak into training graph
- **Stop decision:** If temporal models do not improve PR-AUC by at least 0.02 over static GCN on ANY temporal dataset, flag that temporal modeling adds minimal value for these datasets

## Capabilities

### New Capabilities

- `phase06-temporal-graphs`: TGN with node memory, GRN with snapshot sequence processing, rolling-window GCN, temporal leakage detection, temporal dataset loaders (Pay-At-Pump, Sungkyunkwan, Naver Plus Bank), and temporal model comparison

### Modified Capabilities

- `phase02-minimal-gnn`: Adds temporal evaluation — static GCN now evaluated on temporal snapshots as an additional comparison point
- `phase01-lightgbm-baseline`: Adds temporal-aware baseline (features computed within rolling windows)
- `phase05b-xai-validation`: Adds temporal dimension to explanation validation (XAI on time-stamped predictions)

## Impact

- Self-contained phase in `phases/phase06_temporal_graphs/`:
  - `pyproject.toml` and `uv.lock` with pinned dependencies
  - `src/phase06_temporal_graphs/`: tgn.py, grn.py, rolling_gcn.py, leakage_checker.py, data_loader.py
  - `tests/`: test suite for all temporal components
  - `lessons/`:
    - `01-introduction-to-temporal-graphs.md`
    - `02-temporal-graph-networks.md`
    - `03-rolling-window-training.md`
    - `04-temporal-leakage.md`
    - `05-grn-and-attention.md`
    - `06-temporal-evaluation.md`
  - `notebooks/`:
    - `01-temporal-dynamics.ipynb`
    - `02-rolling-training.ipynb`
    - `03-leakage-analysis.ipynb`
   - `exercises/`: interactive Jupyter notebooks for Practice sections (linked from lessons)
- Data stored in `data/pay-at-pump/`, `data/sungkyunkwan/`, `data/naver-plus-bank/` (shared across phases); `data_loader.py` checks existence and downloads if missing
- Uses existing: `src/utils/calibration.py`, `src/utils/metrics.py`, `phases/phase02_minimal_gnn/`
- No breaking changes to existing codebase
