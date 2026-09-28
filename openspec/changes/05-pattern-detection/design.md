## Context

Phases 1–4 (`lightgbm-baseline`, `minimal-gnn`, `structural-encoding`, `xai-fraud`) have established: (a) LightGBM baselines with and without graph features, (b) GCN models with various structural encodings, (c) XAI methods (GNNExplainer, PGExplainer, GAT attention) with validation pipelines. All existing code in `phases/phase05_pattern_detection/src/phase05_pattern_detection/` is empty. Synthetic data doesn't exist yet for Mitme, Cascade, or Money Mule patterns. The project has `src/utils/calibration.py` and `src/utils/metrics.py` available. The existing data loaders (`ieee_cis_loader.py`, `eev_mitme_loader.py`, `pay_at_pump_loader.py`) provide graph construction patterns that can be adapted for synthetic data generation.

## Goals / Non-Goals

**Goals:**
- Implement synthetic data generators for Mitme, Cascade, and Money Mule fraud patterns with configurable parameters (graph size, pattern density, noise level).
- Train GCN and GAT classifiers on each synthetic pattern type and evaluate detection quality.
- Integrate XAI (GNNExplainer, PGExplainer) on detected pattern instances to validate explanation quality against ground truth.
- Produce structured investigation templates for each detected pattern type.
- Compare explanation quality across pattern types.

**Non-Goals:**
- Production-grade fraud detection system — this is a research prototype.
- Real-time pattern detection — batch processing is acceptable.
- Pattern detection on temporal graphs (reserved for Phase 6).
- Cross-dataset generalization of pattern detection (reserved for Phase 7).

## Decisions

### Decision 1: Graph construction for synthetic patterns

**Choice:** Represent each synthetic pattern as an undirected graph where nodes have IEEE-CIS-like tabular features and edges represent transactions. Use PyTorch Geometric `Data` objects.

**Rationale:** Consistent with existing GCN/GAT infrastructure. PyG `Data` objects work directly with GCNConv and GATConv layers. Undirected edges simplify graph construction and allow bidirectional message passing.

**Alternatives considered:**
- Directed graphs — more realistic but complicate GNN training (requires directed GNN layers).
- Heterogeneous graphs (PyG HeteroData) — more expressive but significantly more complex to implement.

### Decision 2: Synthetic data generation approach

**Choice:** Use networkx to generate pattern topologies, then convert to PyG `Data` objects with node features derived from IEEE-CIS feature distribution. Inject noise (random edges, non-fraudulent transactions) at configurable levels.

**Rationale:** Networkx provides straightforward utilities for generating star, chain, and clustered graphs. Converting to PyG format ensures compatibility with existing GNN training code. Noise injection enables robustness testing.

**Alternatives considered:**
- Fully random graphs with injected patterns — too much noise, harder to isolate signal.
- Pure NumPy graph generation — no network topology utilities available.

### Decision 3: Classification approach — node-level vs graph-level

**Choice:** Use graph-level classification (whole-graph label: "Mitme", "Cascade", "Money Mule", or "benign") as the primary task, with node-level classification as a secondary analysis.

**Rationale:** Graph-level classification directly tests whether the model can identify the presence of a pattern anywhere in the graph. Node-level classification tests whether the model can identify specific fraudulent nodes within the graph. Both are valuable but graph-level is the primary research question.

**Alternatives considered:**
- Pure node-level classification — doesn't address whether the model can detect patterns.
- Edge-level classification — PGExplainer already covers this; pattern detection is structural.

### Decision 4: GNN architecture for pattern detection

**Choice:** Use a single-layer GCN (matching `minimal_gcn.py` architecture) and a single-layer GAT (4 heads) for each pattern type. Add global mean pooling for graph-level classification.

**Rationale:** Consistency with existing models. Global mean pooling is standard for graph-level classification. Multi-head GAT captures diverse structural signals useful for pattern detection.

**Alternatives considered:**
- Deeper GNNs (3+ layers) — risk of over-smoothing on small graphs (synthetic patterns are small).
- GraphSAGE — requires sampling, unnecessary for synthetic graphs.

### Decision 5: Pattern template format

**Choice:** Structured JSON templates with fields: pattern_type, confidence, suspicious_nodes, suspicious_edges, structural_signature, recommended_actions.

**Rationale:** JSON is machine-readable and integrates with existing XAI export pipeline (which already uses JSON for explanation outputs). Structured templates enable downstream automated investigation workflows.

**Alternatives considered:**
- Natural language reports — not machine-parseable.
- CSV — insufficient for nested structure (nodes, edges, signatures).

### Decision 6: Data storage and loading

**Choice:** Each phase implements its own `data_loader.py` that generates synthetic data into the shared `data/` directory at project root. Phase 05's loader generates graphs with known fraud patterns (motifs, rings, stars) for pattern detection testing.

**Rationale:** Self-contained data loading ensures each phase can independently generate and verify its test data. Writing to the shared `data/` directory maintains consistency with the project's data management convention while keeping generation logic co-located with the phase that needs it.

**Alternatives considered:**
- Centralized data generation in a shared utility — couples phases to a single data pipeline and makes independent testing harder.
- In-memory data generation without persistence — loses ability to inspect, cache, and share generated datasets across runs.

## Risks / Trade-offs

| Risk | Mitigation |
|------|-----------|
| Synthetic patterns may be too easy (100% detection) — providing no meaningful signal | Inject increasing levels of noise and obfuscation; compare detection quality against progressively harder variants |
| Small synthetic graphs (10–50 nodes) may not generalize to real-world patterns | Explicitly document graph size limitations; use synthetic patterns as proof-of-concept only |
| GNN models may fail to learn patterns due to noise or insufficient training | Use balanced training sets; increase training epochs; report convergence status |
| XAI explanations may not align with pattern structure due to model overfitting | Use hold-out validation sets; report misalignment as a valid finding, not a failure |
| Pattern detection may require more features than available in IEEE-CIS schema | Augment synthetic features with hand-crafted structural indicators (degree centrality, betweenness) for ablation study |

## Migration Plan

1. Create `phases/phase05_pattern_detection/src/phase05_pattern_detection/generation.py` with synthetic data generators.
2. Create `phases/phase05_pattern_detection/src/phase05_pattern_detection/classifier.py` with GCN/GAT training and evaluation.
3. Create `phases/phase05_pattern_detection/src/phase05_pattern_detection/templates.py` with pattern template generation.
4. Create `phases/phase05_pattern_detection/tests/` test suite.
5. Create `run_patterns.py` for end-to-end pattern detection pipeline.
6. No migration from existing code needed — this is a greenfield module.

## Educational Content

Each lesson follows a consistent 5-part structure:

1. **Explain** — Conceptual introduction with motivation and real-world fraud context
2. **Code** — Implementation walkthrough with annotated Python examples using phase 05 modules
3. **Visualize** — Graph visualizations of patterns (Mitme, Cascade, Money Mule) using networkx/matplotlib
4. **Practice** — Hands-on exercises where learners apply pattern detection to synthetic data
5. **Solution** — Complete solutions with discussion of results and interpretation

### Lessons

| Lesson | Topic | Key Concepts |
|--------|-------|-------------|
| `01-graph-motifs.md` | Graph Motifs | Motif definitions, counting, statistical significance, motif roles |
| `02-subgraph-patterns.md` | Subgraph Pattern Matching | Template matching, VF2 algorithm, pattern similarity metrics |
| `03-anomaly-detection.md` | Anomaly Detection in Graphs | Structural anomalies, node anomalies, community-based detection |
| `04-pattern-mining.md` | Pattern Mining Algorithms | Frequent subgraph mining, gSpan, discriminative pattern mining |
| `05-fraud-patterns-in-practice.md` | Fraud Patterns in Practice | Mitme/Cascade/Mule detection, investigation workflows, template interpretation |

### Notebooks

| Notebook | Focus |
|----------|-------|
| `01-motif-discovery.ipynb` | Interactive motif discovery on synthetic fraud graphs — generate patterns, count motifs, compare against random baselines |
| `02-pattern-analysis.ipynb` | End-to-end pattern analysis — train classifiers, run XAI, generate templates, visualize results |

## Open Questions

- What noise level should be used for synthetic data? (Assumption: start with 0% noise, then test 10%, 20%, 30% progressive noise injection.)
- Should synthetic node features exactly match IEEE-CIS distribution, or simplified features? (Assumption: simplified features matching IEEE-CIS column types and value ranges, but reduced dimensionality for faster prototyping.)
- How many samples per pattern type for balanced training? (Assumption: 2000 samples per pattern type = 8000 total for balanced training.)
