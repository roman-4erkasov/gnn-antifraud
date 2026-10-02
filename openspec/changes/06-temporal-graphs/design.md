## Context

Phases 1–6 (`lightgbm-baseline`, `minimal-gnn`, `structural-encoding`, `xai-basics`, `pattern-detection`, `xai-validation`) have established: (a) LightGBM baselines with and without graph features, (b) GCN/GAT models with various structural encodings, (c) XAI methods (GNNExplainer, PGExplainer, GAT attention) with validation pipelines, (d) Synthetic pattern detection. All existing code in `phases/phase06_temporal_graphs/src/phase06_temporal_graphs/` is empty. The project has `src/utils/calibration.py` and `src/utils/metrics.py` available. IEEE-CIS has explicit timestamps. Pay-At-Pump, Sungkyunkwan, and Naver Plus Bank are temporal anomaly datasets designed specifically for temporal fraud detection.

## Goals / Non-Goals

**Goals:**
- Implement TGN (Temporal Graph Network) with node memory and relation encoder for time-stamped edge events.
- Implement GRN (Graph Recurrent Network) with snapshot-sequence processing for temporal graph analysis.
- Implement rolling-window GCN: split temporal graph into fixed windows, build static snapshots, train GCN per window.
- Validate all temporal models on three temporal datasets: Pay-At-Pump, Sungkyunkwan, Naver Plus Bank.
- Compare temporal models against static GCN and LightGBM baselines on PR-AUC, Brier score, and inference latency.
- Implement temporal leakage checker to verify no test-period edges leak into training graph.

**Non-Goals:**
- Production-grade temporal fraud detection system — this is a research prototype.
- Real-time temporal inference — batch processing is acceptable.
- Temporal edge-level prediction (reserved for future phases).
- Cross-dataset generalization of temporal models (reserved for Phase 7).

## Decisions

### Decision 1: TGN implementation from scratch vs. existing library

**Choice:** Implement TGN from scratch using PyTorch, following the original paper's architecture (node memory with RNN, relation encoder, attention-based memory update).

**Rationale:** Full control over model behavior, enables ablation studies (e.g., swapping memory types), avoids external dependency on `pytorch-TGN` which may have version conflicts with the project's PyTorch/PyG versions.

**Alternatives considered:**
- Use `pytorch-TGN` library — faster to implement but may have incompatible dependencies.
- Use PyG's Temporal GNN module — limited flexibility, may not support all datasets.

### Decision 2: GRN implementation approach

**Choice:** Implement GRN by wrapping a standard GCN encoder (from `phases/phase02_minimal_gnn/src/phase02_minimal_gnn/`) with an LSTM/GRU layer that processes hidden states across snapshot sequences.

**Rationale:** Reuses existing GCN encoder, minimal new GNN layer code needed. LSTM/GRU from PyTorch nn module. Matches the original GRN paper's approach of gating between consecutive hidden states.

**Alternatives considered:**
- Implement custom recurrent GNN message passing — significantly more complex.
- Use DGI (Dynamic Graph Imputation) approach — different architecture, not directly comparable.

### Decision 3: Rolling window GCN — fixed-size vs. event-count windows

**Choice:** Use fixed-size windows (equal number of edges per window) rather than fixed time intervals, as the temporal datasets have irregular event frequencies.

**Rationale:** Equal edge counts ensure each snapshot has sufficient structure for GNN message passing. Fixed time intervals may produce empty or sparse snapshots at certain times.

**Alternatives considered:**
- Fixed time windows (e.g., 1-day windows) — may produce very uneven snapshot sizes.
- Adaptive windows (minimum edge threshold) — introduces complexity in window boundary selection.

### Decision 4: Temporal leakage detection approach

**Choice:** Implement a simple time-based check: verify that all test edges have timestamps strictly greater than the maximum training edge timestamp.

**Rationale:** Simple, deterministic, no false positives. Directly addresses the core leakage concern (future edges in training graph).

**Alternatives considered:**
- Node-overlap check (test nodes not in training nodes) — not sufficient since test nodes may have been in training.
- Graph reachability analysis — complex, not always meaningful for fraud graphs.

### Decision 5: Temporal dataset loaders — phase-local data_loader.py

**Choice:** Implement `data_loader.py` in `phases/phase06_temporal_graphs/src/phase06_temporal_graphs/` that loads all three temporal datasets (Pay-At-Pump, Sungkyunkwan, Naver Plus Bank) from shared `data/` directory. Checks if data exists, downloads if missing.

**Rationale:** Phase-local loader maintains phase isolation while storing large datasets in shared `data/` directory. Single `data_loader.py` with multiple load methods provides consistent interface for all three datasets. Checking existence before download enables offline work and prevents redundant downloads.

**Alternatives considered:**
- Extend centralized `src/data/` loaders — violates phase isolation principle
- Separate loader files per dataset — adds complexity, harder to maintain consistent interface
- Download on every run — slow, requires network, not reproducible

### Decision 6: Model evaluation metrics for temporal comparison

**Choice:** Use PR-AUC (primary), Brier score (calibration quality), and inference latency (speed) for all temporal vs. static comparisons.

**Rationale:** PR-AUC is robust for imbalanced fraud detection. Brier score evaluates calibration quality (critical for deploying models). Inference latency ensures temporal models don't add prohibitive compute costs.

**Alternatives considered:**
- AUC-ROC — sensitive to class imbalance, less informative for fraud.
- F1-score at fixed threshold — depends on arbitrary threshold choice.
- Log-loss — similar to Brier score but less interpretable.

### Decision 7: Data storage and loading strategy

**Choice:** Store temporal datasets in shared `data/` directory at project root (`data/pay-at-pump/`, `data/sungkyunkwan/`, `data/naver-plus-bank/`). Phase 06 implements `data_loader.py` that checks if data exists and downloads/extracts if missing.

**Rationale:** Temporal datasets are large and may be reused by future phases (e.g., Phase 07 cross-dataset evaluation). Storing once in `data/` avoids duplication. The phase's `data_loader.py` encapsulates loading logic for all three temporal datasets with consistent interface.

**Alternatives considered:**
- Each phase stores its own copy — wastes disk space
- Centralized `src/data/` — violates phase isolation principle
- Download on every run — slow, requires network, not reproducible

### Decision 8: Interactive exercises with linked notebooks

**Choice:** Each lesson's Practice section links to an interactive Jupyter notebook in `exercises/` directory.

**Rationale:** Practice exercises in markdown are static text. Linking to notebooks allows users to open, run, and modify code directly. Minimal template: setup code, task description, empty code cell for user solution, solution in markdown code block.

**Alternatives considered:**
- All exercises in one notebook — harder to navigate, no clear mapping to lessons
- Exercises embedded in lesson notebooks — mixes demonstration and practice
- Practice only in markdown — requires copy-paste, less interactive

## Risks / Trade-offs

| Risk | Mitigation |
|------|-----------|
| TGN/GRN have many hyperparameters (memory size, relation dimensions, RNN hidden size) — risk of poor convergence | Use default hyperparameters from original papers; perform limited grid search (3–5 configurations) on validation set |
| Temporal datasets may be too small for deep temporal models | Use early stopping; reduce model capacity; compare against simpler baselines (rolling-window GCN) |
| Pay-At-Pump labels are edge-level, not node-level — may require adaptation of node-level GCN/TGN | Use node-level prediction by assigning labels from incident edges; also implement edge-level prediction as an option |
| Sungkyunkwan has node-level labels per snapshot — GRN naturally fits but TGN needs adaptation | Convert snapshot-level labels to temporal node labels for TGN training |
| Naver Plus Bank has large-scale bipartite graph — TGN/GRN may not fit in memory | Use mini-batch training with neighbor sampling (PyG's `NeighborLoader`); limit graph sub-sampling |
| Temporal models may not outperform static GCN — research may show marginal gains | Document this as a valid finding; focus on explainability of temporal vs. static behavior; analyze when temporal modeling adds value |

## Migration Plan

1. Create `phases/phase06_temporal_graphs/src/phase06_temporal_graphs/tgn.py` with TGN model (node memory, relation encoder, attention update).
2. Create `phases/phase06_temporal_graphs/src/phase06_temporal_graphs/grn.py` with GRN model (GCN encoder + LSTM/GRU layer).
3. Create `phases/phase06_temporal_graphs/src/phase06_temporal_graphs/rolling_gcn.py` with rolling-window GCN implementation.
4. Create `phases/phase06_temporal_graphs/src/phase06_temporal_graphs/leakage_checker.py` with temporal leakage detection.
5. Create `phases/phase06_temporal_graphs/src/phase06_temporal_graphs/data_loader.py` with methods to check data existence, download datasets, and load Pay-At-Pump, Sungkyunkwan, Naver Plus Bank from `data/` directory.
6. Create `phases/phase06_temporal_graphs/tests/` test suite.
7. Create `phases/phase06_temporal_graphs/run_temporal.py` for end-to-end temporal model comparison pipeline.
8. No migration from existing code needed — this is a greenfield module.

## Educational Content

### Lessons (`lessons/`)

Six lessons covering temporal graph concepts, each following a 5-part structure:

1. **Explain** — Conceptual explanation of the topic with motivation and intuition
2. **Code** — Annotated code snippets demonstrating the key implementation
3. **Visualize** — Diagrams or plots illustrating the concept (e.g., temporal snapshots, memory updates, leakage scenarios)
4. **Practice** — Exercises or guided questions for the learner to work through
5. **Solution** — Complete solutions with explanations

| Lesson | Topic |
|--------|-------|
| `01-introduction-to-temporal-graphs.md` | Why temporal modeling matters; static vs. temporal graphs; time-stamped edges and events |
| `02-temporal-graph-networks.md` | TGN architecture: node memory, relation encoder, attention-based memory update |
| `03-rolling-window-training.md` | Rolling-window GCN: window splitting, per-snapshot training, aggregation strategies |
| `04-temporal-leakage.md` | Temporal leakage: definition, detection, prevention; time-based train/test splits |
| `05-grn-and-attention.md` | GRN architecture: GCN encoder + RNN; temporal attention mechanisms |
| `06-temporal-evaluation.md` | Evaluating temporal models: PR-AUC over time, Brier score, latency comparison |

### Notebooks (`notebooks/`)

Three interactive Jupyter notebooks for hands-on exploration:

| Notebook | Content |
|----------|---------|
| `01-temporal-dynamics.ipynb` | Visualize temporal graph dynamics: plot event timelines, build temporal snapshots, observe how graph structure evolves over time |
| `02-rolling-training.ipynb` | Train rolling-window GCN step-by-step: split data into windows, train per-window models, compare predictions across windows |
| `03-leakage-analysis.ipynb` | Demonstrate temporal leakage: create a leaking split, measure inflated metrics, then apply proper temporal split and compare |

## Open Questions

- What window sizes should be used for rolling-window GCN? (Assumption: K=10 equal-sized windows, adjustable per dataset size.)
- Should TGN process events in chronological order only, or allow shuffled mini-batches? (Assumption: chronological order for TGN training, shuffled mini-batches within time windows for efficiency.)
- How to handle node features that change over time? (Assumption: use features from the closest prior time window; if a node appears for the first time, use zero or mean features.)
- What is the stop decision for this phase? (Assumption: if temporal models do not improve PR-AUC by at least 0.02 over static GCN on any temporal dataset, flag that temporal modeling adds minimal value for these datasets.)
