## Why

Phase 3 (`structural-encoding`) established which structural encoding maximizes GCN fraud detection performance. But a model that predicts fraud without explaining why is useless for investigators — fraud teams need to understand *why* a transaction was flagged to justify investigations, comply with regulations, and prevent false positives that damage customer experience. XAI transforms opaque model outputs into actionable intelligence. This phase integrates GNNExplainer, PGExplainer, and GAT attention weights to produce validated, human-readable explanations tied to known fraud signals (chargeback labels).

## What Changes

- Integrate GNNExplainer (PyG) to produce node-level masks over local neighborhoods and features responsible for each GCN prediction
- Integrate PGExplainer for edge-level explanation model training and inference on transaction edges
- Extract and analyze multi-head attention weights from GAT models as an alternative, self-explanatory mechanism
- Validate XAI explanations against IEEE-CIS chargeback labels to ensure alignment with domain-relevant fraud signals
- Produce investigation leads — human-readable, structured output describing key nodes/edges, their roles, and recommended investigation steps
- Compare explanation overlap between GNNExplainer and GAT attention via Jaccard similarity on top-k edges
- **Stop decision:** If XAI-identified edges/nodes show no alignment with chargeback labels AND cross-method consistency < 0.30, flag that current GNN explanations lack trustworthiness

## Capabilities

### New Capabilities

- `phase04a-xai-basics`: GNNExplainer node-level explanation, PGExplainer edge-level explanation, GAT attention weight extraction, explanation validation against chargeback ground truth, investigation lead generation, cross-method explanation comparison

### Modified Capabilities

- `phase02-minimal-gnn`: Adds XAI output (GNNExplainer masks, PGExplainer scores) to GCN prediction pipeline
- `phase03-structural-encoding`: Adds explanation validation as an evaluation criterion for structural encodings

## Impact

- Self-contained phase under `phases/phase04a_xai_basics/`:
  - `src/phase04a_xai_basics/data_loader.py`: Data loading and graph construction for XAI analysis
  - `src/phase04a_xai_basics/gnnexplainer.py`: GNNExplainer integration, node mask extraction
  - `src/phase04a_xai_basics/pgexplainer.py`: PGExplainer edge explanation model
  - `src/phase04a_xai_basics/gat_attention.py`: Attention weight extraction and aggregation
  - `src/phase04a_xai_basics/validation.py`: Explanation validation against chargeback labels
  - `src/phase04a_xai_basics/leads.py`: Investigation lead generation from explanation outputs
  - `src/phase04a_xai_basics/comparison.py`: Cross-method explanation comparison (Jaccard, overlap)
  - `tests/`: Test suite for all XAI components
  - `run_xai.py`: End-to-end XAI pipeline script
  - `lessons/`
    - `01-introduction-to-xai.md`
    - `02-gnnexplainer.md`
    - `03-feature-importance.md`
    - `04-calibration-deep-dive.md`
    - `05-interpreting-fraud-predictions.md`
    - `06-trust-and-actionability.md`
  - `notebooks/`
    - `01-explaining-predictions.ipynb`
    - `02-feature-importance-analysis.ipynb`
    - `03-calibration-curves.ipynb`
  - `exercises/`: interactive Jupyter notebooks for Practice sections (linked from lessons)
- Uses existing: `src/utils/metrics.py`, `phases/phase02_minimal_gnn/src/phase02_minimal_gnn/` (GCN models)
- Dependency: torch-geometric-explain (PyG >= 3.0)
- No breaking changes to existing codebase
