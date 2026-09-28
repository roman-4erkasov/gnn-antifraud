## Why

Phase 4 (`xai-fraud`) produced validated XAI explanations for fraud predictions, but we still don't know whether GNN models can actually *detect* structured fraud patterns in graph data compared to tabular baselines. Fraud investigators need to know not just *why* a transaction was flagged, but whether the model can identify *organized fraud patterns* — Mitme collusion clusters, Cascades, and Money Mule chains — at all. Without this capability, the system is a detector of individual anomalous transactions rather than an investigator of coordinated fraud rings.

## What Changes

- Generate synthetic datasets for three fraud patterns: Mitme (colluding merchant-buyer pairs with circular transactions), Cascade (hub-and-spoke fund distribution), and Money Mule (sequential chain-like fund flow through intermediate accounts)
- Train graph classifiers (GCN/GAT) on each synthetic pattern type and evaluate detection quality (Recall, Precision@K) against LightGBM baseline
- Run XAI (GNNExplainer, PGExplainer) on detected pattern instances to verify explanations identify correct structural roles (hub node, chain edges, circular pairs)
- Implement pattern template generator producing structured output: pattern type, suspicious nodes/edges, structural signatures, recommended investigation actions
- Compare XAI explanation quality across pattern types to identify which patterns are easiest/hardest to explain

**Stop decision:** If graph classifiers achieve Recall < 0.50 on any synthetic pattern type AND XAI explanation quality against ground truth is below 0.25, flag that structured pattern detection requires higher-capacity models or better feature engineering.

## Capabilities

### New Capabilities

- `phase05_pattern_detection`: Synthetic fraud pattern generation (Mitme, Cascade, Money Mule), graph-based pattern classification (GCN/GAT), pattern template generation, and cross-pattern XAI explanation quality comparison

### Modified Capabilities

- `phase01_lightgbm_baseline`: Adds pattern-specific evaluation metrics (pattern-level Recall, Precision@K per pattern type)
- `phase02_minimal_gnn`: Adds pattern-level evaluation on synthetic data (detection quality per pattern type)
- `phase03_structural_encoding`: Adds pattern-specific evaluation to structural encoding experiments
- `phase04_xai_fraud`: Adds validation of XAI outputs against detected pattern instances (not just synthetic ground truth)

## Impact

- `phases/phase05_pattern_detection/src/phase05_pattern_detection/data_loader.py`: Self-contained data loader generating synthetic fraud patterns into shared `data/` directory
- `phases/phase05_pattern_detection/src/phase05_pattern_detection/generation.py`: Synthetic data generators for Mitme, Cascade, Money Mule patterns
- `phases/phase05_pattern_detection/src/phase05_pattern_detection/classifier.py`: Graph classifier training and evaluation per pattern type
- `phases/phase05_pattern_detection/src/phase05_pattern_detection/xai_validation.py`: XAI explanation validation against pattern ground truth
- `phases/phase05_pattern_detection/src/phase05_pattern_detection/templates.py`: Pattern template generation for investigation leads
- `phases/phase05_pattern_detection/tests/test_generation.py`: Tests for synthetic data generation
- `phases/phase05_pattern_detection/tests/test_classifier.py`: Tests for graph classifier training and evaluation
- `phases/phase05_pattern_detection/tests/test_lightgbm.py`: Tests for LightGBM baseline on graph features
- `phases/phase05_pattern_detection/tests/test_xai_validation.py`: Tests for XAI validation functions
- `phases/phase05_pattern_detection/tests/test_integration.py`: End-to-end integration tests
- `phases/phase05_pattern_detection/tests/test_templates.py`: Tests for pattern template generation
- `phases/phase05_pattern_detection/run_patterns.py`: End-to-end pattern detection pipeline script
- `phases/phase05_pattern_detection/lessons/01-graph-motifs.md`: Lesson on graph motifs and their role in fraud detection
- `phases/phase05_pattern_detection/lessons/02-subgraph-patterns.md`: Lesson on subgraph pattern matching techniques
- `phases/phase05_pattern_detection/lessons/03-anomaly-detection.md`: Lesson on anomaly detection in graphs
- `phases/phase05_pattern_detection/lessons/04-pattern-mining.md`: Lesson on pattern mining algorithms
- `phases/phase05_pattern_detection/lessons/05-fraud-patterns-in-practice.md`: Lesson on applying fraud pattern detection in practice
- `phases/phase05_pattern_detection/notebooks/01-motif-discovery.ipynb`: Interactive notebook for motif discovery experiments
- `phases/phase05_pattern_detection/notebooks/02-pattern-analysis.ipynb`: Interactive notebook for pattern analysis and visualization
- Uses existing: `src/utils/calibration.py`, `src/utils/metrics.py`, `phases/phase04_xai_fraud/src/phase04_xai_fraud/` (GNNExplainer, PGExplainer), `phases/phase02_minimal_gnn/src/phase02_minimal_gnn/` (GCN models)
- No breaking changes to existing codebase

## Educational Content

Phase 05 includes structured educational content covering pattern detection concepts:

```
├── lessons/
│   ├── 01-graph-motifs.md
│   ├── 02-subgraph-patterns.md
│   ├── 03-anomaly-detection.md
│   ├── 04-pattern-mining.md
│   └── 05-fraud-patterns-in-practice.md
└── notebooks/
    ├── 01-motif-discovery.ipynb
    └── 02-pattern-analysis.ipynb
```
