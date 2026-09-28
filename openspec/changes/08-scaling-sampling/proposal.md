## Why

Phases 1–7 have established model effectiveness on datasets of varying sizes. However, the research has not addressed scalability — how models perform as graph size grows from thousands to millions to hundreds of millions of nodes. The project's end goal is a production system for a 300M-node graph. Without sampling optimization, GNN training on such a graph is computationally infeasible. We need to empirically compare sampling strategies (Neighbor Sampling, GraphSAINT, GraphBolt), understand the speed/quality tradeoffs, and design a partitioning strategy for large-scale distributed training.

## What Changes

- Implement Neighbor Sampling benchmark: compare different sampling depths (2-hop, 3-hop) and fanout sizes on model quality and training speed
- Implement GraphSAINT benchmark: compare vertex, edge, and subgraph sampling variants
- Implement GraphBolt-style sampling (PyTorch GNN Library): compare with Neighbor Sampling and GraphSAINT
- Implement graph partitioning strategy: METIS/K-way partitioning for distributed training on 300M-node graphs
- Create scaling benchmark: measure throughput (nodes/sec), memory usage, and PR-AUC as function of graph size
- **Stop decision:** If no sampling strategy can achieve PR-AUC >= 0.85 × full-graph PR-AUC while being 2× faster than full-graph training at scale, flag that sampling quality is insufficient for the target graph size

## Capabilities

### New Capabilities

- `phase08_scaling_sampling`: Neighbor Sampling with configurable depth/fanout, GraphSAINT (vertex/edge/subgraph), GraphBolt integration, graph partitioning (METIS/K-way), scaling benchmark (speed/quality/memory)

### Modified Capabilities

None — Phase 08 uses a wrapper/adapter pattern to add sampling to existing models without modifying them directly.

## Impact

- New self-contained phase under `phases/phase08_scaling_sampling/`:
  - `src/phase08_scaling_sampling/data_loader.py`: Synthetic scaling dataset generation and loading
  - `src/phase08_scaling_sampling/neighbor.py`: Neighbor Sampling implementation with configurable depth and fanout
  - `src/phase08_scaling_sampling/graphsaint.py`: GraphSAINT (vertex, edge, subgraph) sampling implementations
  - `src/phase08_scaling_sampling/graphbolt.py`: GraphBolt integration (if available) or reference implementation
  - `src/phase08_scaling_sampling/partition.py`: METIS/K-way graph partitioning for distributed training
  - `src/phase08_scaling_sampling/scaling_benchmark.py`: Scaling benchmark pipeline
  - `tests/`: Test suite for all sampling and scaling components
  - `run_scaling.py`: End-to-end scaling benchmark script
  - `lessons/`
    - `01-scaling-challenges.md`
    - `02-mini-batch-training.md`
    - `03-negative-sampling.md`
    - `04-graph-sampling.md`
    - `05-memory-optimization.md`
    - `06-production-considerations.md`
  - `notebooks/`
    - `01-scaling-experiments.ipynb`
    - `02-sampling-strategies.ipynb`
    - `03-performance-benchmarks.ipynb`
- Extends existing: Uses models from `phases/phase02_minimal_gnn/src/phase02_minimal_gnn/minimal_gcn.py` and `phases/phase06_temporal_graphs/src/phase06_temporal_graphs/` via wrapper/adapter pattern (no direct modification)
- No breaking changes to existing codebase
