## Context

Phases 1–7 (`lightgbm-baseline`, `minimal-gnn`, `structural-encoding`, `xai-fraud`, `pattern-detection`, `temporal-graphs`, `cross-dataset`) have established model effectiveness on datasets ranging from tens of thousands to a few million nodes. However, the research has not addressed scalability. The project's end goal is a production system for a 300M-node fraud graph. Full-graph GNN training on such a graph is computationally infeasible — it requires sampling (Neighbor Sampling, GraphSAINT, GraphBolt) and partitioning (METIS/K-way). Without these, the research cannot validate whether GNN-based fraud detection is viable at scale.

## Goals / Non-Goals

**Goals:**
- Implement Neighbor Sampling with configurable depth and fanout, benchmark speed/quality tradeoffs.
- Implement GraphSAINT sampling (vertex, edge, subgraph variants), benchmark against Neighbor Sampling.
- Integrate GraphBolt (PyTorch GNN Library) for efficient sampling; provide reference fallback.
- Implement K-way graph partitioning (METIS-based) for distributed training.
- Create scaling benchmark measuring throughput, memory, and quality across graph sizes.
- Extrapolate requirements for 300M-node graph and recommend optimal configuration.

**Non-Goals:**
- Distributed training implementation — only partitioning strategy, not the distributed training loop.
- Model architecture optimization for sampling — assume GCN/GAT architectures from phases 2–3.
- GPU-specific optimization — focus on CPU-first design with GPU as optional.
- Graph compression or sparsification — out of scope.

## Decisions

### Decision 1: Neighbor Sampling implementation — PyG vs custom

**Choice:** Use PyG's `NeighborLoader` for Neighbor Sampling when available; implement custom Neighbor sampling as a fallback for compatibility.

**Rationale:** PyG's `NeighborLoader` is well-optimized and handles edge_index conversion automatically. Custom implementation ensures the code works even without the latest PyG version.

**Alternatives considered:**
- Custom Neighbor Sampling only — more control but reinvents PyG's solution.
- PyG NeighborLoader only — may break with older PyG versions.

### Decision 2: GraphSAINT implementation

**Choice:** Implement GraphSAINT from scratch following the original paper's algorithms (Vertex, Edge, Subgraph sampling with uniform random selection).

**Rationale:** Full control over implementation enables ablation studies (e.g., weighted sampling by degree). GraphSAINT is straightforward to implement and has no major external dependencies.

**Alternatives considered:**
- Use PyG's GraphSAINT — limited configurability.
- Use DGL's GraphSAINT — requires DGL integration, which conflicts with PyG-first design.

### Decision 3: GraphBolt integration approach

**Choice:** Attempt to import GraphBolt; if available, use it for sampling. If not, fall back to custom Neighbor Sampling without errors.

**Rationale:** GraphBolt is the official PyTorch sampling library optimized for large graphs. Using it when available ensures best performance. Fallback ensures the code runs regardless of installation.

**Alternatives considered:**
- Require GraphBolt installation — breaks reproducibility on systems without it.
- Ignore GraphBolt entirely — misses opportunity for optimized sampling.

### Decision 4: Graph partitioning — METIS vs custom

**Choice:** Use `scikit-learn`'s spectral partitioning or `networkx`'s spectral_bisection as the primary partitioning method, with METIS (via `pymetis` or `igraph`) as an optional higher-quality alternative.

**Rationale:** Spectral partitioning via networkx/scikit-learn requires no external C libraries and works on CPU. METIS is higher quality but requires additional system dependencies. This approach prioritizes reproducibility.

**Alternatives considered:**
- METIS only — best quality but harder to install on all systems.
- K-means on node features — ignores graph structure, produces poor partitions.
- Label propagation — complex to tune.

### Decision 5: Benchmark methodology

**Choice:** Use a fixed test set from each dataset, measure PR-AUC at each epoch, track training time and memory. Use a synthetic scaling dataset (increasingly larger random graphs) for the scaling benchmark.

**Rationale:** Fixed test set ensures comparable quality measurements. Synthetic scaling dataset allows controlled size extrapolation to 300M nodes.

**Alternatives considered:**
- Real data at multiple sizes — few datasets exist at different scales.
- Full-dataset benchmark only — doesn't provide scaling trajectory.

### Decision 6: Memory measurement approach

**Choice:** Use Python's `tracemalloc` for Python-side memory tracking and `subprocess`-based `nvidia-smi` for GPU memory (if GPU available).

**Rationale:** tracemalloc captures PyTorch tensor allocation patterns. nvidia-smi provides accurate GPU memory usage. Both are standard tools with no special dependencies.

**Alternatives considered:**
- PyTorch memory profiling — detailed but complex to parse.
- System-level profiling (valgrind) — not portable, adds overhead.

### Decision 7: Data storage and loading

**Choice:** Each phase implements its own `data_loader.py` that generates synthetic data of varying sizes (100K, 1M, 10M nodes) into the shared `data/` directory at project root.

**Rationale:** Phase 08 generates synthetic scaling datasets for benchmarking mini-batch training and sampling strategies. Self-contained data loading ensures each phase can independently generate and manage its datasets without external dependencies, while the shared `data/` directory provides a consistent location for all generated data.

**Alternatives considered:**
- Centralized data generation script — single point of failure, harder to maintain per-phase datasets.
- In-memory data generation — doesn't allow reuse across runs or sharing between phases.

### Decision 8: Wrapper/Adapter pattern for sampling integration

**Choice:** Implement a `SamplingWrapper` class that wraps any PyG model (GCN from Phase 02, TGN/GRN from Phase 06) to add sampling-based training without modifying the original models.

**Rationale:** The wrapper pattern preserves the self-contained nature of each phase. Phase 02 and Phase 06 remain unchanged, while Phase 08 adds sampling capability through composition. This approach:
- Avoids tight coupling between phases
- Allows Phase 08 to work with any PyG model that follows the standard interface
- Makes it easy to test sampling independently
- Follows the Open/Closed Principle (open for extension, closed for modification)

**Alternatives considered:**
- Direct modification of Phase 02/06 models — violates self-contained principle, creates tight coupling.
- Interface/Protocol approach — requires Phase 02/06 to implement a specific interface, which still modifies them.
- Inheritance — less flexible than composition, harder to combine multiple sampling strategies.

## Risks / Trade-offs

| Risk | Mitigation |
|------|-----------|
| Neighbor Sampling with large fanouts may not reduce memory significantly | Cap fanout at practical values (≤ 25); measure memory at each fanout to find knee point |
| GraphSAINT may produce biased estimates (vertex vs edge sampling differ in coverage) | Use importance-weighted loss corrections; compare multiple variants |
| GraphBolt may not be available on all systems | Provide robust fallback to custom Neighbor Sampling; document GraphBolt installation requirements |
| METIS partitioning may be slow on very large graphs (100K+ nodes) | Precompute partitions and cache; partitioning is a one-time cost, not per-training |
| Synthetic scaling graphs may not capture real-world graph structure (power-law degree distribution) | Generate synthetic graphs with power-law degree distribution using networkx configuration model |
| Sampling may degrade PR-AUC significantly — research may show sampling is not viable | Document this as a valid finding; report the PR-AUC degradation as a function of speedup |
| 300M-node extrapolation may be inaccurate — training time/memory don't scale linearly | Use multiple extrapolation models (linear, sublinear, superlinear); report range of estimates |

## Migration Plan

1. Create `phases/phase08_scaling_sampling/src/phase08_scaling_sampling/` package with `__init__.py`, `neighbor.py`, `graphsaint.py`, `graphbolt.py`, `partition.py`, `scaling_benchmark.py`, `sampling_wrapper.py`.
2. Implement `SamplingWrapper` that wraps any PyG model to add sampling-based training.
3. Create `phases/phase08_scaling_sampling/tests/` test suite.
4. Create `phases/phase08_scaling_sampling/run_scaling.py` for end-to-end scaling benchmark.
5. No migration from existing code needed — this is a greenfield module.

## Educational Content

Phase 08 includes educational content under `lessons/` and `notebooks/` to make scaling and sampling concepts accessible.

### Lessons (`lessons/`)

Each lesson follows a 5-part structure:

1. **Explain** — Conceptual overview with motivation and intuition (e.g., why full-graph training fails at scale, what mini-batch training solves).
2. **Code** — Annotated code snippets from the phase implementation, showing how the concept is realized in practice.
3. **Visualize** — Diagrams, figures, or plots illustrating the concept (e.g., sampling diagrams, memory usage curves, scaling trajectories).
4. **Practice** — Guided exercises or questions for the reader to work through (e.g., "Calculate the memory savings from fanout=10 vs fanout=25").
5. **Solution** — Worked solutions to the practice exercises with explanation.

Lesson topics:

| # | File | Topic |
|---|------|-------|
| 1 | `01-scaling-challenges.md` | Why GNNs don't scale naively — O(N) memory, message-passing cost, and the motivation for sampling |
| 2 | `02-mini-batch-training.md` | Neighbor Sampling and GraphSAINT mini-batch training — how to train on subgraphs instead of the full graph |
| 3 | `03-negative-sampling.md` | Negative sampling strategies for fraud detection — how to sample non-fraud nodes efficiently in imbalanced graphs |
| 4 | `04-graph-sampling.md` | Graph partitioning and distributed training — METIS/K-way partitioning for splitting a 300M-node graph |
| 5 | `05-memory-optimization.md` | Memory optimization techniques — fanout tuning, feature caching, and gradient checkpointing for large graphs |
| 6 | `06-production-considerations.md` | Production deployment — latency requirements, model serving at scale, and monitoring sampling quality |

### Notebooks (`notebooks/`)

Interactive Jupyter notebooks for hands-on experimentation:

| # | File | Topic |
|---|------|-------|
| 1 | `01-scaling-experiments.ipynb` | Run scaling benchmarks across graph sizes, compare sampling strategies, and visualize speed/quality tradeoffs |
| 2 | `02-sampling-strategies.ipynb` | Compare Neighbor Sampling, GraphSAINT, and GraphBolt on the same graph — tune fanout, depth, and batch size |
| 3 | `03-performance-benchmarks.ipynb` | Profile throughput, memory, and PR-AUC; extrapolate to 300M nodes; reproduce the stop decision analysis |

## Open Questions

- What fanout sizes should be benchmarked? (Assumption: 5, 10, 15, 20, 25 — covering small to large fanouts.)
- What sampling depths should be benchmarked? (Assumption: 1, 2, 3, 4 hops — up to the over-smoothing threshold.)
- How to handle feature aggregation during sampling (mean, sum, LSTM)? (Assumption: default to mean aggregation; test LSTM aggregation as an optional comparison.)
- What is the stop decision? (Assumption: if no sampling strategy achieves PR-AUC >= 0.85 × full-graph PR-AUC while being 2× faster than full-graph training at scale, flag that sampling quality is insufficient for the target graph size.)
- How to generate synthetic scaling graphs that match real-world structure? (Assumption: use networkx's powerlaw_cluster_model or configuration model with power-law degree distribution and clustering.)
