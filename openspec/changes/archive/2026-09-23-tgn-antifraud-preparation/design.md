## Context

Research project to empirically answer whether GNNs add value for anti-fraud ranking at scale (300M nodes, ~3 node features). Two GPUs available (one usually busy), Hadoop/PySpark platform for preprocessing. The research phase targets open datasets and synthetic data; CPU-first development, GPU later.

Two framework implementations (PyG and DGL) are required for every model to enable direct comparison.

## Goals / Non-Goals

**Goals:**
- Empirical comparison: tabular baseline vs GNN vs TGN
- Structural encoding impact (ReCWE, RWSE) with sparse node features
- Graph sampler speed/quality benchmarking
- XAI explanation quality validated against known fraud patterns
- Dual PyG/DGL implementation with shared interface

**Non-Goals:**
- Production-ready pipeline or deployment
- Distributed training across multiple GPUs/nodes
- Real-time inference
- 300M-node training in the research phase (prototype scale only)
- Novel algorithm design (this is evaluation, not research contribution)

## Decisions

### D1: Dual Framework with Abstraction Layer

```
┌─────────────────────────────────────────────┐
│              Experiment Runner              │  ← shared orchestrator
│  (dataset load → train → evaluate → explain)│
├─────────────────────────────────────────────┤
│  Framework Abstraction Layer                │
│  ┌──────────────┐    ┌──────────────┐       │
│  │  PyG Backend │    │   DGL Backend│       │
│  │  GCN / GAT /  │    │  GCN / GAT / │       │
│  │  TGN models   │    │  TGN models  │       │
│  └──────────────┘    └──────────────┘       │
│     Shared:                                     │
│     - metrics.py                               │
│     - explainers.py                            │
│     - samplers.py (interface)                  │
│     - data_io.py (dataset loading)             │
└─────────────────────────────────────────────┘
```

**Rationale**: One interface, two backends. Every model, sampler, and explainer is implemented twice (PyG and DGL) but through the same abstract API. This avoids code duplication in evaluation and ensures apples-to-apples comparison.

**Alternatives considered**:
- PyG only: cheaper, but no framework comparison
- Shared tensor library (e.g., both on CuPy): too much friction, framework-specific ops differ

### D2: Data Format and Storage

```
Graph representation:
┌─────────────────────────────────────────────┐
│  Edge list (timestamped):                   │
│    src (int32) → dst (int32) → ts (int64)  │
│                                             │
│  Node features:                             │
│    X (n_nodes, n_features) = float32       │
│                                             │
│  Labels:                                    │
│    y (n_nodes,) = int64                    │
│    y_edge (n_edges,) = int64 (for edge-level)│
│                                             │
│  Sparse adjacency:                          │
│    scipy.sparse.csr_matrix                 │
│                                             │
│  Temporal graph:                            │
│    sorted edge list by timestamp            │
│    + node embedding cache (for TGN)        │
└─────────────────────────────────────────────┘
```

**Rationale**: Sparse matrices and sorted edge lists are minimal, portable, and work on CPU. PySpark can produce these formats directly. TGN needs temporal ordering and node memory, which the sorted edge list supports.

**Alternatives considered**:
- Dense adjacency: too memory-intensive for large graphs
- NetworkX: convenient but slow for large graphs

### D3: CPU-First with Optional GPU

Every experiment defaults to CPU. A `device='cuda'` flag enables GPU. Numeric equivalence is verified (max diff < 1e-4). This enables:
- Debugging on any machine
- Iteration speed (CPU iteration is faster for small datasets)
- Consistent results (no non-deterministic GPU ops during dev)

### D4: Synthetic Data Generator

The system SHALL include a synthetic data generator with three fraud pattern modes:

```
Synthetic data patterns:
┌─────────────────────────────────────────────┐
│ 1. Mitme-Fraud pattern                     │
│    [User A] → [Merchant] → [User A]         │
│    [User B] → [Merchant] → [User B]         │
│    → GNN should detect coordinated cycle    │
│                                             │
│ 2. Credit Card pattern                     │
│    [Stolen] → [M-1] → [M-2] → ... → [M-N]  │
│    → GNN should detect cascade structure    │
│                                             │
│ 3. Money Mule pattern                      │
│    [Sources] → [Mules] → [Destinations]     │
│    → GNN should detect flow-through nodes   │
└─────────────────────────────────────────────┘

Controls: graph size, node features, fraud ratio, temporal noise
```

**Rationale**: Synthetic data with known ground truth enables XAI validation (do explanations recover the planted fraud patterns?) and controlled ablation (change one variable at a time).

### D5: Sampler Implementation Strategy

```
Sampler hierarchy:
┌─────────────────────────────────────────────────────┐
│  Sampler Interface (abstract)                       │
│  ├── NeighborSampler (PyG / DGL)                   │
│  │   ├── RandomNeighbor                             │
│  │   └── MultiLayerNeighbor                         │
│  │                                                  │
│  ├── GraphSAINTSampler (PyG / DGL)                  │
│  │   ├── Uniform                                    │
│  │   ├── BFS                                        │
│  │   ├── RandomWalk                                 │
│  │   └── Degree                                     │
│  │                                                  │
│  └── GraphBoltIndex (PyG only, reference)           │
│      └── Large-scale optimized indexing             │
│                                                  │
│  Benchmark suite for each sampler:                  │
│  - samples/sec (throughput)                        │
│  - peak memory (GB)                               │
│  - model accuracy delta vs full graph             │
└─────────────────────────────────────────────────────┘
```

**Rationale**: Neighbor sampling is the baseline, GraphSAINT is the established alternative, GraphBolt is the new SOTA for PyG. DGL has no direct GraphBolt equivalent, so it is marked as PyG-only reference.

## Risks / Trade-offs

| Risk | Impact | Mitigation |
|------|--------|------------|
| Dual implementation doubles development time | Schedule slip | Shared interface, start with PyG then mirror to DGL |
| Open datasets are static, TGN requires temporal data | TGN evaluation limited | Use synthetic temporal data or Naver Plus Bank dataset |
| XAI quality is subjective | Hard to measure success | Validate against synthetic data ground truth (objective metric) |
| 300M-scale never reached in research | Misses real-world validation | Out of scope; focus on scalability *design* patterns |
| PyG/DGL non-determinism makes comparison unfair | Confounded results | Same random seeds, disable cudnn benchmark, document versions |
| GNNExplainer is slow on large graphs | Evaluation bottleneck | Use PGExplainer for large graphs, GNNExplainer for small/medium |

## Migration Plan

N/A — this is a new research project, not a migration.

## Open Questions

1. **Temporal data availability**: Open static datasets (IEEE-CIS, EEV) lack timestamps. Should we use Naver Plus Bank (temporal, Korean) or generate temporal structure synthetically?
2. **GraphBolt for DGL**: DGL has no direct GraphBolt equivalent. Should we implement a simplified version or skip it for DGL?
3. **XAI evaluation metric**: 70% overlap threshold for synthetic validation — is this reasonable or arbitrary? Need domain input.
4. **Hardware constraints for research phase**: What GPU model will be available? Affects sampler size limits.
