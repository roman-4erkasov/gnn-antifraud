## Why

Phases 1–6 have established baselines (LightGBM, GCN, GAT), temporal models (TGN, GRN, rolling-GCN), XAI validation, pattern detection, and temporal graph modeling on individual datasets. However, we have not yet addressed whether findings generalize across datasets, models, and graph/non-graph representations. The research has no statistical rigor (no confidence intervals, no significance testing) and no cross-dataset comparison framework. Without cross-dataset generalization analysis, we cannot answer: do models trained on one dataset transfer to another? Is the observed pattern consistent across IEEE-CIS, Pay-At-Pump, Sungkyunkwan, and Naver Plus Bank?

## What Changes

- Implement cross-dataset model comparison framework: train models on one dataset, evaluate on others, measure transfer quality
- Implement statistical significance testing: confidence intervals via bootstrapping for PR-AUC, Brier score comparisons between models
- Implement cross-dataset feature compatibility analysis: identify which features are portable vs dataset-specific
- Implement model-agnostic evaluation report: structured comparison table across all models × all datasets
- **Stop decision:** If no model achieves PR-AUC >= 0.30 on ANY held-out (cross-dataset) test set, flag that cross-dataset transfer is not viable with current approaches

## Capabilities

### New Capabilities

- `phase07_cross_dataset`: Cross-dataset model comparison, bootstrapping-based confidence intervals, significance testing, transfer learning evaluation, cross-dataset feature compatibility analysis

### Modified Capabilities

- `phase01_lightgbm_baseline`: Adds cross-dataset evaluation — LightGBM trained on one dataset, tested on others
- `phase02_minimal_gnn`: Adds cross-dataset evaluation — GCN/GAT trained on one dataset, tested on others
- `phase06_temporal_graphs`: Adds cross-dataset evaluation — temporal models trained on one dataset, tested on others
- `phase04_xai_fraud`: Adds cross-dataset XAI comparison — explainability quality measured across datasets

## Impact

- Self-contained phase under `phases/phase07_cross_dataset/`:
  - `src/phase07_cross_dataset/data_loader.py`: Self-contained data loading for all fraud datasets
  - `src/phase07_cross_dataset/compare.py`: Cross-dataset model comparison framework
  - `src/phase07_cross_dataset/statistics.py`: Bootstrapping, confidence intervals, significance testing
  - `src/phase07_cross_dataset/feature_compat.py`: Feature compatibility analysis across datasets
  - `src/phase07_cross_dataset/transfer.py`: Transfer learning evaluation (train on A, test on B)
  - `tests/`: Phase-local test suite
  - `lessons/`: Educational content
    - `lessons/01-cross-dataset-challenges.md`
    - `lessons/02-transfer-learning.md`
    - `lessons/03-domain-adaptation.md`
    - `lessons/04-dataset-comparison.md`
    - `lessons/05-generalization-strategies.md`
  - `notebooks/`: Interactive tutorials
    - `notebooks/01-transfer-analysis.ipynb`
    - `notebooks/02-cross-dataset-evaluation.ipynb`
  - `run_cross_dataset.py`: End-to-end cross-dataset evaluation script
- Extends existing: `src/utils/calibration.py`, `src/utils/metrics.py`
- No breaking changes to existing codebase
