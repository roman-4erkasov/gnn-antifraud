## Why

Phase 2 (`minimal-gnn`) established whether a single-layer GCN adds value over LightGBM. But a GCN's performance depends heavily on how node features are constructed — specifically, what structural information is encoded in node representations. Without structural encodings, GCNs only see the graph topology through message passing, which can be insufficient for sparse or noisy graphs. This phase systematically evaluates multiple structural encoding techniques (ReCWE, RWSE, Spectral, Community) to determine which encoding strategy maximizes fraud detection performance.

## What Changes

- Implement 4 structural encoding techniques: ReCWE (Relative Custom Walk Encoding), RWSE (Random Walk Structural Encoding), Spectral encoding (graph Laplacian eigenvectors), and Community detection labels (Louvain)
- Train separate GCN models with each encoding technique on the same fraud detection task
- Evaluate each encoding's impact on PR-AUC, Brier score, and other metrics
- Compute SHAP values for graph-aware LightGBM with each encoding set and compare feature importance rankings
- Produce a summary table identifying which encoding drives fraud detection most effectively

## Capabilities

### New Capabilities

- `phase03-structural-encoding`: Multiple structural encoding techniques (ReCWE, RWSE, Spectral, Community) for GNN node classification, SHAP-based feature importance comparison

### Modified Capabilities

- `phase02-minimal-gnn`: Adds structural encoding as an option to GCN training pipeline

## Impact

- Self-contained phase in `phases/phase03_structural_encoding/`:
  - `pyproject.toml` and `uv.lock` with pinned dependencies
  - `src/phase03_structural_encoding/`: data_loader.py, structural_encodings.py, train_gcn.py
  - `tests/`: test_structural_encodings.py
  - `lessons/`
    - `01-spectral-graph-theory.md`
    - `02-laplacian-eigenvectors.md`
    - `03-random-walk-features.md`
    - `04-community-detection.md`
    - `05-structural-features-for-fraud.md`
  - `notebooks/`
    - `01-exploring-eigenvectors.ipynb`
    - `02-community-analysis.ipynb`
  - `exercises/`: interactive Jupyter notebooks for Practice sections (linked from lessons)
- Uses existing: `src/utils/metrics.py`, `phases/phase02_minimal_gnn/src/phase02_minimal_gnn/`
- Dependency: numpy, scipy, community_louvain
- No breaking changes
