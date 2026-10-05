## Why

The project can detect and explain known fraud patterns, but it cannot proactively rank which **new** links are likely to appear. A forward-looking capability is needed that, for a given node at a given time, ranks candidate counterparties that were never connected before by the probability they connect within a forecasting horizon `H`. This powers link recommendation and early-warning, and no existing phase covers candidate generation or link ranking — the closest phases only explain existing edges (04a), list edges of already-detected patterns (05), or rank transactions by fraud probability (06).

## What Changes

- Introduce a new self-contained phase `11-temporal-link-prediction` implementing **temporal novel-link prediction**: for a source node `u` at time `t`, rank candidate nodes `v` that have never been connected to `u` before `t` by `P(edge u–v appears within (t, t+H])`. Appearances are counted even if transient (the link need not survive to `t+H`).
- Add a common **dataset interface** covering three datasets: a synthetic generator (unit tests and controlled horizon sweeps), a generic real dynamic graph (`sx-mathoverflow`), and a transaction dynamic graph (`tgbl-coin`, subsampled, via `py-tgb`).
- Add a pluggable **retrieval stage** (candidate generation) combining embedding-ANN, structural (PPR / common neighbors / Adamic-Adar), I2I co-occurrence, and sequential retrievers (TGN memory or SASRec), with a separate `Candidate Recall@M` metric.
- Add **GNN pretraining** producing frozen node embeddings without leakage (trained only on data `≤ t`): GraphSAGE (primary, inductive), GAE, node2vec/DeepWalk, RWSE/spectral, with optional TGN memory and optional SASRec/GRU4Rec sequence embeddings.
- Add a **ranker** whose target model is CatBoost with a listwise `YetiRank` objective, with pointwise `Logloss` and pairwise `PairLogit` as comparative objectives, plus a full comparison set (LightGBM `lambdarank`, XGBoost `rank:pairwise`, training-free heuristics, retrieval-cosine, LogisticRegression, and one end-to-end GNN).
- Add **evaluation**: `Hits@K`, `Recall@K`, `MRR`, `MAP`, `NDCG@K`, `Candidate Recall@M`, horizon sweeps over `H`, feature/objective ablations, generic-vs-transaction comparison, and an optional cross-check against TGB's `Evaluator` (MRR).
- Add the phase's standard anatomy (uv-installable package, run script, `src/` package, `tests/`, `lessons/`, `exercises/`, `notebooks/`, `results/`).

## Capabilities

### New Capabilities
- `phase11-temporal-link-prediction`: temporal novel-link prediction — dataset loading, temporal train/val/test splitting, candidate generation and retrieval, GNN embeddings, structural and interaction features, GBDT/listwise ranking, and ranking evaluation of links expected within a forecasting horizon.

### Modified Capabilities
<!-- None: this change introduces a new self-contained capability and does not alter requirements of existing capabilities. -->

## Impact

- New self-contained phase package `phases/phase11_temporal_link_prediction/` (uv project), following the same anatomy as phases 01–10.
- Reuses shared infrastructure only: `src/utils/` (`metrics.py`, `calibration.py`, `features.py`) and the `data/` download area. It declares no dependency on phases 02–10.
- New runtime dependencies: `catboost`, `py-tgb` (with the torch extra for GNN embeddings), `torch` / PyG, optional `faiss-cpu` for ANN, plus existing `lightgbm`, `xgboost`, `networkx`, `pandas`, `numpy`, `scipy`, `scikit-learn`.
- New dataset artifacts: `sx-mathoverflow` (SNAP) and `tgbl-coin` (TGB; license **CC BY-NC**, subsampled).
- Outputs written to `phases/phase11_temporal_link_prediction/results/`.
