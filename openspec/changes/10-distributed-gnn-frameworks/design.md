## Context

Phase 09 implements custom distributed GNN inference on PySpark, but production-ready frameworks (DGL Distributed, AliGraph, Quiver) already exist. Before committing to custom implementation, we need to evaluate whether these frameworks meet our requirements for 300M-node fraud graphs. This phase benchmarks all three frameworks against identical workloads to provide data-driven recommendations.

## Goals / Non-Goals

**Goals:**
- Install and configure DGL Distributed, AliGraph, and Quiver in isolated environments
- Benchmark each framework on graphs of increasing size (1M, 10M, 100M nodes)
- Measure throughput (nodes/sec), memory usage, scalability, and integration complexity
- Compare fraud-specific metrics (PR-AUC on imbalanced data)
- Generate recommendation report with decision criteria

**Non-Goals:**
- Modify existing models from phases 02-08 (use as-is for benchmarking)
- Implement custom distributed solutions (that's phase 09)
- Production deployment of any framework (evaluation only)
- Benchmark on 300M-node graph (extrapolate from 100M results)

## Decisions

### Decision 1: Framework selection criteria

**Choice:** Evaluate DGL Distributed, AliGraph, and Quiver based on three criteria: (1) production readiness (stability, documentation, community), (2) PySpark/PyG compatibility, (3) fraud detection suitability (imbalanced data handling, temporal features).

**Rationale:** These three frameworks represent the main approaches: DGL (academic/research standard), AliGraph (industrial fraud focus), Quiver (GPU acceleration). Evaluating all three covers the solution space.

**Alternatives considered:**
- Evaluate only one framework — insufficient comparison
- Evaluate five+ frameworks — too broad, dilutes focus
- Skip benchmark and adopt one framework — no data-driven decision

### Decision 2: Benchmark methodology

**Choice:** Use identical workloads across all frameworks: same graph data (synthetic 1M/10M/100M nodes with power-law distribution), same GCN model architecture (2 layers, 64 hidden dim), same evaluation metrics (PR-AUC, throughput, memory).

**Rationale:** Fair comparison requires identical inputs. Synthetic data ensures reproducibility and controlled scaling. Using the same model architecture isolates framework performance from model differences.

**Alternatives considered:**
- Use real fraud data — harder to reproduce, data availability issues
- Use different models per framework — conflates model and framework performance
- Benchmark only throughput — ignores fraud-specific metrics

### Decision 3: Installation approach

**Choice:** Use separate virtual environments for each framework to avoid dependency conflicts. Document installation steps for reproducibility.

**Rationale:** DGL, AliGraph, and Quiver have different PyTorch/PyG version requirements. Separate environments prevent conflicts and ensure clean benchmarks.

**Alternatives considered:**
- Single environment with all frameworks — dependency conflicts likely
- Docker containers per framework — adds complexity, harder to debug
- Conda environments — works, but uv is already used in project

### Decision 4: Integration complexity measurement

**Choice:** Measure integration complexity by counting: (1) lines of code changed to adapt phase 02 model, (2) new dependencies added, (3) configuration files required, (4) setup time in hours.

**Rationale:** Quantitative metrics enable objective comparison. Code changes and setup time directly impact development effort.

**Alternatives considered:**
- Subjective rating (easy/medium/hard) — not reproducible
- Only count dependencies — ignores code complexity
- Skip integration measurement — incomplete comparison

### Decision 5: Scalability extrapolation

**Choice:** Benchmark up to 100M nodes, then extrapolate to 300M using linear scaling model. Validate extrapolation by comparing predicted vs actual at 100M (using 10M and 1M data points).

**Rationale:** 300M-node benchmark requires production cluster resources not available during research phase. Linear extrapolation from 100M is reasonable for throughput/memory (validated by checking linearity at smaller scales).

**Alternatives considered:**
- Benchmark directly on 300M nodes — requires production cluster
- Exponential extrapolation — overestimates resource needs
- No extrapolation — cannot answer 300M question

### Decision 6: Data storage and loading

**Choice:** Each phase implements its own `data_loader.py` that generates synthetic benchmark datasets of varying sizes (1M, 10M, 100M nodes) into the shared `data/` directory at project root. The loader creates standardized benchmark graphs with power-law degree distributions for fair framework comparison.

**Rationale:** Phase 10 compares three frameworks on identical workloads. A self-contained data loader ensures each phase can independently generate reproducible benchmark data without external dependencies. Writing to the shared `data/` directory allows cross-phase data reuse while keeping each phase self-contained.

**Alternatives considered:**
- Download real-world graph datasets — not reproducible, availability issues
- Generate data inline in each benchmark script — duplication, inconsistent graphs
- Use a shared data generation module across phases — couples phases, breaks self-containment

### Decision 7: Interactive exercises with linked notebooks

**Choice:** Each lesson's Practice section links to an interactive Jupyter notebook in `exercises/` directory.

**Rationale:** Practice exercises in markdown are static text. Linking to notebooks allows users to open, run, and modify code directly. Minimal template: setup code, task description, empty code cell for user solution, solution in markdown code block.

**Alternatives considered:**
- All exercises in one notebook — harder to navigate, no clear mapping to lessons
- Exercises embedded in lesson notebooks — mixes demonstration and practice
- Practice only in markdown — requires copy-paste, less interactive

## Risks / Trade-offs

| Risk | Mitigation |
|------|-----------|
| Framework installation fails or is unstable | Document installation issues; use fallback versions; note in recommendation |
| AliGraph is not publicly available or requires special access | Use public AliGraph papers/documentation; note limitation in report |
| Quiver requires GPU resources not available | Benchmark CPU-only mode; note GPU requirement in recommendation |
| Frameworks produce different predictions than phase 02 models | Investigate architecture differences; document prediction divergence; note in comparison |
| Benchmark results are inconclusive (no clear winner) | Provide scenario-based recommendations (e.g., "use DGL for CPU, Quiver for GPU") |
| Extrapolation to 300M is inaccurate | Validate extrapolation model at 100M; provide confidence intervals |

## Migration Plan

1. Create `phases/phase10_distributed_gnn_frameworks/` directory structure
2. Install DGL Distributed in separate environment; document installation
3. Install AliGraph (or document if unavailable); note access requirements
4. Install Quiver in separate environment; document installation
5. Create unified benchmark harness with common data loading and evaluation
6. Run DGL benchmark on 1M, 10M, 100M graphs; collect metrics
7. Run AliGraph benchmark on same graphs; collect metrics
8. Run Quiver benchmark on same graphs; collect metrics
9. Compare results; generate recommendation report
10. Create 5 lessons and 2 notebooks documenting findings

## Open Questions

- Is AliGraph publicly available for benchmarking, or is it internal to Ant Financial? (Assumption: public version exists; if not, document limitation)
- Does Quiver support CPU-only benchmarking, or is GPU required? (Assumption: CPU mode exists but is slower; benchmark both if possible)
- What is the acceptable prediction divergence threshold when adapting phase 02 models? (Assumption: <1% PR-AUC difference)
- Should we benchmark training or only inference? (Assumption: inference only, since phases 02-08 already trained models)
