## Context

Phase 1 (`lightgbm-baseline`) established both minimal (3-feature) and graph-aware LightGBM baselines. The graph structure (bipartite user-merchant graph) and graph features (degree, community, RWSE) are already computed in `phases/phase01_lightgbm_baseline/src/phase01_lightgbm_baseline/graph_features.py`. PyTorch and PyG are listed as dependencies in the project but may not be installed yet. The existing calibration and metrics utilities will be reused without modification.

## Goals / Non-Goals

**Goals:**
- Implement a single-layer GCN model (`GCNConv` → ReLU → `GCNConv` → Sigmoid) in PyTorch Geometric.
- Train GCN on the same bipartite graph used by graph-aware LightGBM.
- Apply calibration to GCN predictions using existing `IsotonicCalibrator`.
- Compare GCN against LightGBM minimal and graph-aware baselines with statistical significance.
- Produce a stop decision: does GNN add value over tree models?

**Non-Goals:**
- Multi-layer GCN (deeper GNNs are out of scope for Phase 2).
- Graph pooling / graph-level classification.
- Node embedding extraction (t-SNE visualization is deferred).
- Hyperparameter optimization for GCN.

## Decisions

### Decision 1: Single-layer GCN (GCNConv) with ReLU
**Choice:** Use a single `GCNConv` → ReLU → `GCNConv` → Sigmoid architecture.
**Rationale:** A single-layer GCN is the simplest GNN — it's the minimal GNN that can be meaningfully compared to tree models. Deeper GNNs introduce over-smoothing and are reserved for later phases if single-layer proves insufficient.
**Alternatives considered:**
- Multi-layer GCN (2-3 layers) — overkill for minimal GNN evaluation; risks over-smoothing.
- GAT (attention-based) — introduces different inductive bias; reserved for later phases.

### Decision 2: Node classification task on bipartite graph
**Choice:** Predict fraud label at the user node level using the bipartite user-merchant graph. Transactions are edges; users are nodes with features.
**Rationale:** The IEEE-CIS data labels transactions as fraud, but GNNs operate on nodes. We assign fraud labels to the user node when any of their transactions is fraudulent (weak supervision), and train GCN as a node classifier. This is the standard approach for fraud GNNs on transaction data.
**Alternatives considered:**
- Transaction-level node classification — would require adding transaction nodes; increases graph size 5-10x.
- Graph-level pooling — unsuitable since we need per-transaction predictions for Precision@K.

### Decision 3: Node feature matrix construction
**Choice:** For user nodes: use the graph-derived features from Phase 1 (degree, community label, RWSE values) plus basic user features (avg transaction amount, n_transactions). For merchant nodes: use merchant-level aggregates (avg amount, n_transactions).
**Rationale:** The graph feature extraction from Phase 1 already computes these. We aggregate transaction-level data to node-level features. For the GCN, we need a fixed-dimension feature matrix.
**Alternatives considered:**
- Use raw transaction features (high-dimensional IEEE-CIS columns) — would require creating one node per transaction; memory intensive.
- Use only graph-derived features (degree, RWSE) — too few features for GCN to learn; but this could be a valid baseline.

### Decision 4: Training via PyG NeighborLoader not needed (full-batch)
**Choice:** Train GCN on the full graph (mini-batch not needed for Phase 2 dataset sizes).
**Rationale:** Phase 2 focuses on accuracy comparison, not scalability. Full-batch training is simpler and avoids sampling bias. Sampling is addressed in Phase 8.
**Alternatives considered:**
- Mini-batch training via `NeighborLoader` — introduces sampling decisions not needed for minimal evaluation.
- Semi-supervised split (PyG default) — IEEE-CIS has explicit train/test, not semi-supervised.

### Decision 5: Calibration split 60/20/20 (same as LightGBM)
**Choice:** Use the same 60/20/20 train/val/test split for GCN as used for LightGBM to ensure fair comparison.
**Rationale:** Fair comparison requires identical data splits. Calibration on held-out validation prevents overfitting GCN probabilities.

### Decision 6: Educational content structure
**Choice:** Each lesson follows a 5-part pattern: Explain (concept description) -> Code (working example) -> Visualize (plots/charts) -> Practice (exercise for the reader) -> Solution (completed exercise with results).
**Rationale:** This pattern is proven in technical education (see "Explain-Do-Review" model). It keeps learners engaged by alternating theory and practice. Lessons are written as markdown files that can be rendered in GitHub/GitLab and read directly in any text editor. Phase 02 focuses on GCN fundamentals, so lessons cover: introduction to GCN, graph construction for GNNs, training loops, evaluation metrics, and comparison with LightGBM.
**Alternatives considered:**
- Video lectures — harder to update, not searchable, not version-controlled.
- Interactive course platforms (Coursera, DataCamp) — external dependency, not integrated with the repo.
- Single monolithic document — hard to navigate, no modular exercises.

### Decision 7: Interactive notebooks for experimentation
**Choice:** Create Jupyter notebooks under `phases/phase02_minimal_gnn/notebooks/` for hands-on experimentation with GCN behavior and hyperparameter tuning.
**Rationale:** Notebooks allow learners to modify code, visualize results, and experiment with different configurations. Two notebooks cover: (1) exploring GCN behavior (message passing, node embeddings, over-smoothing), and (2) hyperparameter sweep (learning rate, hidden dimensions, number of epochs). This complements the lessons with interactive exploration.
**Alternatives considered:**
- Command-line scripts only — less interactive, harder to visualize intermediate results.
- Web-based dashboard — overkill for educational purposes, adds infrastructure complexity.

### Decision 8: Data storage and loading strategy

**Choice:** Each phase implements its own `data_loader.py` that checks for data in the shared `data/` directory at project root, downloads if missing, and loads phase-specific data.

**Rationale:** Phase 02 uses the IEEE-CIS Fraud Detection dataset (shared with phases 01, 03, 04). Storing data in a shared `data/` directory avoids duplication. Each phase's `data_loader.py` encapsulates its specific data needs (graph construction from transactions, feature extraction) while reusing the same raw data files.

**Alternatives considered:**
- Each phase downloads its own copy — wastes disk space
- Centralized `src/data/` loaders — breaks phase isolation
- Embed data in phase directory — doesn't scale for large datasets

### Decision 9: Interactive exercises with linked notebooks
**Choice:** Each lesson's Practice section links to an interactive Jupyter notebook in `exercises/` directory.
**Rationale:** Practice exercises in markdown are static text. Linking to notebooks allows users to open, run, and modify code directly. Minimal template: setup code, task description, empty code cell for user solution, solution in markdown code block.
**Alternatives considered:**
- All exercises in one notebook — harder to navigate, no clear mapping to lessons
- Exercises embedded in lesson notebooks — mixes demonstration and practice
- Practice only in markdown — requires copy-paste, less interactive

## Risks / Trade-offs

| Risk | Mitigation |
|------|-----------|
| GCN underfits due to single layer on sparse graph | Use feature normalization (batch norm); try adding residual connections if underfitting. |
| Node label assignment ambiguous (user has multiple transactions) | Use max label across user's transactions (if any tx is fraud, user is fraud); report label assignment strategy in results. |
| PyTorch / PyG not installed in project environment | Handle ImportError gracefully with clear installation instructions in run script. |
| GCN training much slower than LightGBM | Limit epochs to reasonable number (e.g., 200); use early stopping on validation loss. |

## Open Questions

- Should we train GCN on user nodes only, or on both user and merchant nodes? (User-only is simpler; merchant-augmented could capture merchant-level fraud rings.)
- What learning rate and weight decay are appropriate for this graph? (Default: LR=0.01, weight_decay=5e-4 — standard GCN hyperparameters from Kipf & Widt.)
