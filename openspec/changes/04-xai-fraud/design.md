## Context

Phase 3 (`structural-encoding`) determined which structural encoding best serves GCN fraud detection. The models (GCN, and potentially GAT for attention-based explanations) now produce predictions, but without explanations these predictions are opaque to fraud investigators. The project has no XAI infrastructure yet. PyG provides GNNExplainer natively, and PGExplainer is a separate PyG package. GAT models (which can be built on top of `MinimalGCN` with GATConv) provide self-explanatory attention weights.

## Goals / Non-Goals

**Goals:**
- Integrate GNNExplainer for node-level subgraph and feature explanation in GCN.
- Integrate PGExplainer for edge-level explanation model training and inference.
- Implement GAT attention weight extraction for edge-level importance.
- Validate explanations against IEEE-CIS chargeback labels for domain-relevant fraud signal alignment.
- Generate human-readable investigation leads from XAI outputs.
- Compare explanation consistency across methods (GNNExplainer vs PGExplainer vs GAT attention).

**Non-Goals:**
- Building a UI/dashboard for explanation visualization.
- Implementing custom XAI algorithms (only integrating existing PyG tools).
- Explainability for temporal models (reserved for Phase 6).
- Real-time explanation generation (batch processing is acceptable).

## Decisions

### Decision 1: GNNExplainer for node-level explanation

**Choice:** Use GNNExplainer from `torch_geometric.explain` for GCN node-level explanations.

**Rationale:** GNNExplainer is the most mature and widely-used GNN explanation method. It produces both edge masks (which neighborhood edges matter) and feature masks (which node features matter). It works directly with any GCN model without architectural changes.

**Alternatives considered:**
- Integrated Gradients — works on node features but not edge structure.
- Gradient-based saliency — simpler but less reliable for GNNs (gradient saturation).
- Propagation-based (GNNXplain, GraphLIME) — newer methods but less mature integration with PyG.

### Decision 2: PGExplainer for edge-level explanation

**Choice:** Use PGExplainer from `torch_geometric.explain` for edge-level explanation model.

**Rationale:** PGExplainer learns a parametric explanation model during training that generalizes to unseen edges. Unlike post-hoc methods, it produces explanations during forward pass, making it efficient at inference. It is well-suited for transaction-level fraud prediction (edge prediction task).

**Alternatives considered:**
- Edge-specific GNNExplainer — post-hoc, slow for large number of edges.
- Attention weights from GAT — self-explanatory but only available if GAT is used.

### Decision 3: GAT with multi-head attention for self-explanation

**Choice:** Build GAT on top of existing model infrastructure using `torch_geometric.nn.conv.GATConv` with multi-head attention. Extract attention weights as edge importance scores.

**Rationale:** GAT provides built-in explanation through attention weights — each head assigns importance to neighboring edges. Multi-head attention captures diverse aspects of node/edge importance. This approach requires no post-hoc explanation tool and works alongside GNNExplainer/PGExplainer for comparison.

**Alternatives considered:**
- Single-head GAT — simpler but captures only one aspect of importance.
- SAGPool — focuses on node selection, not edge-level explanation.

### Decision 4: Explanation validation against chargeback labels

**Choice:** Validate explanations by checking alignment with chargeback labels on IEEE-CIS — whether top-importance edges connect to the transaction's merchant or buyer node.

**Rationale:** Chargeback labels on real data provide domain-relevant validation of explanation quality. This ensures XAI outputs align with actual fraud signals recognized by the payment system.

**Alternatives considered:**
- Validation against synthetic patterns with known structure — deferred to Phase 04b after synthetic patterns are generated in Phase 05.
- Human evaluation by fraud analysts — expensive and not reproducible.
- Permutation importance — measures feature importance but not structural explanation quality.

### Decision 5: JSON-based explanation export

**Choice:** All XAI outputs are exported as structured JSON for downstream use (investigation leads, comparison, archival).

**Rationale:** JSON is machine-readable, universally parsable, and supports nested structures needed for subgraphs, edge weights, and feature importances. It can be ingested by investigation tools or stored for audit.

**Alternatives considered:**
- CSV — flat structure cannot represent subgraph relationships.
- Pickle — Python-only, not portable.
- Parquet — better for tabular data but subgraph data is inherently nested.

### Decision 6: Investigation lead generation as structured text

**Choice:** Generate investigation leads as structured text templates populated with XAI output values.

**Rationale:** Fraud analysts need natural language that they can read and act on. Structured templates ensure consistency and completeness while remaining human-readable. Templates can be refined later without changing code.

**Alternatives considered:**
- Free-form text generation via LLM — introduces variability and external dependencies.
- Purely numeric output — not actionable for non-technical analysts.

### Decision 7: Self-contained data loading from shared data directory

**Choice:** Phase 04 uses the IEEE-CIS Fraud Detection dataset (shared with phases 01, 02, 03). Each phase implements its own `data_loader.py` that checks for data in the shared `data/` directory at project root, downloads if missing, and loads phase-specific data. The loader handles graph construction suitable for XAI analysis.

**Rationale:** Each phase should be self-contained and reproducible without depending on other phases' data pipelines. Checking a shared `data/` directory avoids redundant downloads while keeping each phase independent. The loader constructs graphs optimized for XAI analysis (preserving edge structure and node features needed for explanation methods).

**Alternatives considered:**
- Importing data from other phases' loaders — creates tight coupling between phases.
- Phase-specific data directories — leads to duplicate data on disk.
- No download capability — requires manual data setup, reducing reproducibility.

## Risks / Trade-offs

| Risk | Mitigation |
|------|-----------|
| GNNExplainer is slow on large graphs (per-node optimization) | Limit explanation generation to a sample of predictions (e.g., top-100 flagged transactions) rather than explaining every node. |
| PGExplainer requires separate training and may not generalize well | Train on a representative edge subset; monitor explanation loss during training; fall back to GNNExplainer if PGExplainer quality is poor. |
| GAT may not train stably on fraud graphs (sparse, imbalanced) | Start with GAT on structural encoding data (Phase 3) where features are richer; use fewer heads; use gradient clipping. |
| XAI explanations may not align with ground truth | This is a valid result, not a failure. Report the misalignment and investigate causes (model error vs explanation error). |
| Investigation leads may be too generic to be useful | Use domain-specific templates (merchant review, device fingerprint check) and tie lead content directly to XAI-highlighted nodes/edges. |

## Migration Plan

This is a greenfield XAI module. No migration from existing code is needed.

1. Create `phases/phase04_xai_fraud/src/phase04_xai_fraud/` module with `gnnexplainer.py`, `pgexplainer.py`, `gat_attention.py`, `validation.py`, `leads.py`, `comparison.py`.
2. Integrate with existing GCN model from Phase 2 / structural encodings from Phase 3.
3. Add test suite under `phases/phase04_xai_fraud/tests/`.
4. Create run script `run_xai.py` for end-to-end explanation pipeline.

## Educational Content

### Lesson Structure

Each lesson in `lessons/` follows a consistent 5-part pedagogical structure:

1. **Explain** — Conceptual overview of the topic (what it is, why it matters for fraud detection, key intuitions).
2. **Code** — Annotated code walkthrough showing how the concept is implemented in this phase's modules.
3. **Visualize** — Diagrams, plots, or graph visualizations that make the concept concrete (e.g., subgraph masks, attention heatmaps, calibration curves).
4. **Practice** — Hands-on exercise for the learner to complete (e.g., run GNNExplainer on a synthetic pattern, interpret a feature importance ranking).
5. **Solution** — Reference solution with explanation of expected results and common pitfalls.

### Lesson Catalog

| Lesson | Topic | Key Concepts |
|--------|-------|-------------|
| `01-introduction-to-xai.md` | Why XAI matters for fraud | Opacity problem, regulatory requirements, investigator workflows, trust |
| `02-gnnexplainer.md` | GNNExplainer deep dive | Node-level masks, edge masks, feature masks, local subgraph extraction |
| `03-feature-importance.md` | Feature importance for GNNs | Feature mask interpretation, ranking features, linking to fraud signals |
| `04-calibration-deep-dive.md` | Model calibration | ECE, reliability diagrams, Platt scaling, temperature scaling, fraud threshold selection |
| `05-interpreting-fraud-predictions.md` | From explanations to investigation | Reading investigation leads, mapping XAI output to fraud patterns, false positive analysis |
| `06-trust-and-actionability.md` | Trust, limitations, and actionability | When to trust explanations, cross-method agreement, actionability criteria, human-in-the-loop |

### Notebook Catalog

| Notebook | Topic | Key Activities |
|----------|-------|---------------|
| `01-explaining-predictions.ipynb` | Explaining fraud predictions with GNNExplainer | Load trained GCN, explain top fraud predictions, visualize subgraph masks, interpret edge/feature importance |
| `02-feature-importance-analysis.ipynb` | Feature importance across fraud types | Extract feature masks, rank features by importance, compare importance profiles across fraud patterns (Mitme vs Cascade) |
| `03-calibration-curves.ipynb` | Calibration analysis and improvement | Plot reliability diagrams, compute ECE, apply Platt scaling, compare pre/post calibration, select fraud thresholds |

## Open Questions

- What sample size of nodes should be explained? (Assumption: explain top-100 highest fraud-score predictions per dataset to balance quality and computation.)
- Should PGExplainer be trained on the same graph used for node classification, or on an edge-level prediction task? (Assumption: trained on the bipartite user-merchant-transaction graph with edges labeled by whether the transaction is fraudulent.)
- How should "ground truth" be defined for IEEE-CIS chargeback labels? Are they node-level or edge-level? (Assumption: chargeback labels are treated as edge-level ground truth — the transaction edge itself is the fraud signal.)
