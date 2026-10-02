## Context

Phases 01-08 have established GNN models for fraud detection, but production deployment requires inference on 300M-node graphs using PySpark. Current approaches sample neighbors on the driver, creating bottlenecks and OOM risks. This phase enables distributed sampling on executors with proper graph partitioning and halo management.

## Goals / Non-Goals

**Goals:**
- Partition graphs across Spark executors using METIS or spectral partitioning
- Manage cross-partition neighbor access through configurable-depth halos
- Perform neighbor sampling on executors via UDFs
- Load trained PyTorch models into executors for inference
- Orchestrate end-to-end distributed inference pipeline
- Benchmark executor-based vs driver-based sampling performance

**Non-Goals:**
- Distributed training (this phase focuses on inference only)
- Model architecture changes (models are loaded as-is from phases 02-08)
- GPU-accelerated distributed inference (CPU-only for this phase)
- Real-time streaming inference (batch processing only)

## Decisions

### Decision 1: Cluster setup — Docker Compose

**Choice:** Use Docker Compose with 1 Spark master + 2 workers for local development and testing.

**Rationale:** Docker provides reproducible environment across cloud servers. Docker Compose allows easy scaling (add more workers) and isolated dependencies (PySpark + PyTorch + PyG). No need for external Hadoop cluster. Single command `docker-compose up` starts the full cluster.

**Alternatives considered:**
- External Hadoop cluster — requires infrastructure setup, not portable
- Local Spark (single machine) — cannot test distributed sampling
- Kubernetes — overkill for development, adds complexity

### Decision 2: Partitioning strategy — METIS vs spectral

**Choice:** Use METIS when available, fall back to spectral partitioning via scipy/networkx.

**Rationale:** METIS produces higher-quality partitions with better balance and lower edge cuts, but requires external C library installation. Spectral partitioning is pure Python and always available, though with slightly worse partition quality. Fallback ensures the system works in any environment.

**Alternatives considered:**
- METIS only — breaks on systems without METIS installed
- Hash partitioning — ignores graph structure, produces poor partitions with high edge cuts
- Random partitioning — no structural awareness, worst performance

### Decision 3: Halo management — precompute vs iterative loading

**Choice:** Precompute halos during partitioning phase and store them alongside partition data.

**Rationale:** Precomputation avoids repeated cross-partition communication during inference. For a 3-hop GCN, each partition needs 1-hop, 2-hop, and 3-hop neighbors from other partitions. Precomputing once is more efficient than loading iteratively during inference.

**Alternatives considered:**
- Iterative halo loading per hop — reduces storage but adds latency during inference
- On-demand halo fetching via Spark shuffle — too slow for large-scale inference
- No halo (truncate sampling paths) — loses accuracy for nodes near partition boundaries

### Decision 4: Sampling UDF implementation — PySpark mapPartitions vs Pandas UDF

**Choice:** Use PySpark mapPartitions with custom Python function for sampling UDF.

**Rationale:** mapPartitions gives full control over partition loading, halo access, and PyG Data construction. Pandas UDFs are optimized for vectorized operations but less flexible for complex graph operations. Sampling requires building PyG Data objects with edge_index and features, which is easier in mapPartitions.

**Alternatives considered:**
- Pandas UDF — less flexible for graph operations
- Spark SQL UDF — too restrictive, cannot handle complex data structures
- Custom Spark connector for PyG — requires significant development effort

### Decision 5: Model serialization — PyTorch save/load vs Spark broadcast

**Choice:** Use PyTorch's native save/load for model serialization, then broadcast to executors via Spark's broadcast mechanism.

**Rationale:** PyTorch save/load preserves model architecture and weights. Spark broadcast efficiently distributes the serialized model to all executors without repeated network transfers. This is simpler than custom serialization and works with any PyTorch model.

**Alternatives considered:**
- Custom serialization format — unnecessary complexity
- Load model from disk on each executor — slower, requires shared filesystem
- TorchScript compilation — adds complexity, not all models are TorchScript-compatible

### Decision 6: Memory management — partition size vs executor memory

**Choice:** Target partition sizes that fit within executor memory with 20% headroom for halo and sampling overhead. Default to 100K-500K nodes per partition, configurable based on executor memory.

**Rationale:** Executor memory is typically 4-16GB. Partition data + halo + sampling buffers can exceed 1GB per 100K nodes depending on feature dimensionality. Conservative partition sizing prevents OOM errors.

**Alternatives considered:**
- Dynamic partition sizing based on runtime memory — adds complexity
- Larger partitions with spill-to-disk — slower due to disk I/O
- Smaller partitions — increases overhead from more tasks

### Decision 7: Batch processing — target node batching vs full partition

**Choice:** Process target nodes in batches within each partition, with batch size configurable based on model complexity and executor memory.

**Rationale:** For very large partitions, processing all target nodes at once may exceed executor memory. Batching allows processing partitions with millions of target nodes without OOM.

**Alternatives considered:**
- Process all target nodes at once — risks OOM for large partitions
- Process one node at a time — too slow, loses batch processing benefits

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
| Halo size exceeds partition size for high-degree graphs | Monitor halo-to-partition ratio during partitioning; if ratio > 2.0, warn user and suggest increasing partition count |
| METIS partitioning fails on very large graphs (100M+ nodes) | Fall back to spectral partitioning; for extremely large graphs, use hierarchical partitioning (partition partitions) |
| Model loading fails due to architecture mismatch | Validate model compatibility during initialization; provide clear error messages for unsupported architectures |
| Sampling UDF performance is slower than expected due to Python overhead | Profile UDF execution; if bottleneck is in Python, consider Cython optimization or move critical path to C++ |
| Cross-partition edges cause inconsistent predictions | Ensure halo includes all k-hop neighbors; validate prediction consistency at partition boundaries |
| Executor memory fragmentation after multiple batches | Restart executor JVM after N batches; use Spark's dynamic allocation to recycle executors |

## Migration Plan

1. Create `phases/phase09_distributed_inference/` directory structure with pyproject.toml, src/, tests/, lessons/, notebooks/, docker/
2. Setup Docker Compose cluster (1 master + 2 workers) with PySpark, PyTorch, PyG
3. Implement partition.py with METIS and spectral partitioning
4. Implement halo.py for halo precomputation and loading
5. Implement sampling_udf.py for executor-based neighbor sampling
6. Implement model_loader.py for PyTorch model serialization and broadcast
7. Implement inference_pipeline.py for end-to-end orchestration
8. Implement benchmark.py for performance comparison
9. Create run_distributed_inference.py script
10. Create 6 lessons and 3 notebooks following 5-part structure
11. Test on synthetic graphs (1M, 10M, 100M nodes) and validate against single-machine inference

## Open Questions

- What is the optimal partition size for different executor memory configurations? (Assumption: 100K-500K nodes per partition for 4-16GB executors)
- How does halo depth affect prediction accuracy at partition boundaries? (Assumption: 3-hop halo matches GCN depth, sufficient for most cases)
- What is the speedup threshold for production deployment? (Assumption: 2× speedup over driver-based sampling is minimum viable)
- How to handle dynamic graphs where edges change between inference runs? (Assumption: static graph for this phase; dynamic graphs require re-partitioning)
