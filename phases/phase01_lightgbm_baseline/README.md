# Phase 01 — LightGBM Baseline with Graph Features

A self-contained phase that establishes the simplest possible fraud-detection
baseline (a 3-feature LightGBM model) and a graph-aware variant that adds
features derived from a bipartite user–merchant transaction graph.

## What this phase does

1. Loads IEEE-CIS transaction + identity data from the shared `data/ieee-cis/`
   directory and builds 3 baseline features:
   `TransactionAmt`, `time_since_creation`, `n_prior_transactions`.
2. Trains a LightGBM classifier with early stopping and applies isotonic
   calibration to its probabilities.
3. Builds a bipartite graph (users = `card1`, merchants = `ProductCD`) and
   extracts node degree/edge count, clustering coefficient, degree centrality,
   Louvain community labels, and RWSE features (diagonal of the k-step
   random-walk return probabilities, k = 1..8).
4. Compares the minimal and graph-aware models with delta metrics, a paired
   permutation test, and a stop decision.
5. Persists run artifacts (`metrics.json`, `model.txt`, `calibrator.pkl`,
   comparison table/plot) under `results/`.

## Setup

This phase uses [`uv`](https://github.com/astral-sh/uv) with an isolated
environment. From the phase directory:

```bash
uv sync
```

## Data

Data is shared at the project root. From this phase the default path is
`../../data`, i.e. `data/ieee-cis/` at the repository root. The loader validates
that `train_transaction.csv`, `test_transaction.csv`, `train_identity.csv`, and
`test_identity.csv` exist; if they are missing it prints download instructions
(and an optional Kaggle CLI command) instead of fabricating data.

## Run tests

```bash
uv run pytest tests/ -v
```

## Run the pipeline

```bash
uv run python run_lightgbm_baseline.py --sample-limit 5000
```

Useful flags:

| Flag | Default | Description |
|------|---------|-------------|
| `--data-path` | `../../data` | Shared data directory |
| `--sample-limit` | none | Stratified cap on rows (fast prototyping) |
| `--random-state` | `42` | Random seed |
| `--output-dir` | `results` | Where run artifacts are written |

For a fast smoke run against the bundled synthetic fixture:

```bash
uv run python run_lightgbm_baseline.py --data-path ./tests/data/ieee_cis_test --sample-limit 20
```

## Layout

```
phase01_lightgbm_baseline/
├── src/phase01_lightgbm_baseline/
│   ├── data_loader.py        # load + baseline features
│   ├── lightgbm_baseline.py  # LightGBMWrapper (train/predict/calibrate)
│   ├── graph_features.py     # GraphFeatureExtractor
│   ├── pipeline.py           # train_graph_aware + split helper
│   └── comparison.py         # deltas, significance, stop decision
├── run_lightgbm_baseline.py  # end-to-end runner
├── lessons/                  # 8 mini-lessons (Explain/Code/Visualize/Practice/Solution)
├── exercises/                # practice notebooks linked from lessons
├── notebooks/                # 3 exploratory notebooks
├── tests/                    # unit tests + synthetic fixture
└── results/                  # persisted run artifacts
```
