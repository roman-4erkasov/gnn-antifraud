## Why

Phase 09 implements custom distributed GNN inference on PySpark, but several production-ready frameworks already solve this problem: DGL Distributed (native distributed sampling), AliGraph (industrial fraud detection), and Quiver (GPU-accelerated PyG integration). Before committing to a custom implementation, we need to benchmark these frameworks against our specific use case (300M-node fraud graph, PySpark infrastructure, PyTorch models) to determine whether building custom is justified or if adopting an existing solution is more efficient.

## What Changes

- Install and configure DGL Distributed with PySpark integration
- Install and configure AliGraph for fraud detection workloads
- Install and configure Quiver for GPU-accelerated PyG sampling
- Create unified benchmark harness that runs identical workloads across all three frameworks
- Benchmark throughput (nodes/sec), memory usage, scalability (1M → 100M nodes), and ease of integration
- Compare API compatibility with existing code from phases 02-08
- Generate recommendation report: when to use each framework vs custom PySpark implementation

## Capabilities

### New Capabilities

- `phase10-distributed-gnn-frameworks`: Comparative benchmark of DGL Distributed, AliGraph, and Quiver — installation guides, configuration templates, unified benchmark harness, performance comparison (throughput/memory/scalability), API compatibility analysis, recommendation report

### Modified Capabilities

None. This phase evaluates external frameworks without modifying existing code.

## Impact

- Self-contained phase in `phases/phase10_distributed_gnn_frameworks/`:
  - `pyproject.toml` and `uv.lock` with dependencies (dgl, aligraph, quiver, torch, torch_geometric, pyspark)
  - `src/phase10_distributed_gnn_frameworks/`: data_loader.py, dgl_benchmark.py, aligraph_benchmark.py, quiver_benchmark.py, comparison_harness.py, recommendation.py
  - `tests/`: test suite for benchmark harness
  - `benchmarks/`: configuration files and scripts for each framework
  - `run_framework_comparison.py`: End-to-end comparison script
  - `lessons/`: 5 markdown lessons covering each framework + comparison methodology
  - `notebooks/`: 2 interactive notebooks for benchmark exploration
- Requires multi-GPU cluster for Quiver benchmarks
- Requires DGL, AliGraph, and Quiver installations (may need separate environments)
- No breaking changes to existing codebase
