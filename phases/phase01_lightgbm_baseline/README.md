# Phase 01: LightGBM Baseline

LightGBM baseline for fraud detection with graph-derived features.

## Overview

This phase establishes a minimal baseline for fraud detection using:
- 3 baseline features: transaction amount, time since creation, prior transactions
- Graph-derived features: node degree, clustering, community labels, RWSE

## Structure

```
phase01_lightgbm_baseline/
├── src/phase01_lightgbm_baseline/
│   ├── data_loader.py          # Data loading utilities
│   ├── lightgbm_baseline.py    # LightGBM wrapper with calibration
│   ├── graph_features.py       # Graph feature extraction
│   └── comparison.py           # Model comparison utilities
├── tests/                      # Test suite
├── lessons/                    # Educational content
├── notebooks/                  # Interactive notebooks
└── pyproject.toml             # Dependencies
```

## Usage

```bash
# Install dependencies
uv sync

# Run baseline pipeline
uv run python ../../run_lightgbm_baseline.py --sample-limit 500

# Run tests
uv run pytest tests/ -v
```

## Lessons

1. Introduction to LightGBM
2. Baseline Features
3. Calibration
4. Graph Construction
5. Graph Features
6. Comparison and Stop Decision

## Notebooks

1. Explore Baseline
2. Experiment with Calibration
3. Play with Graph Features
