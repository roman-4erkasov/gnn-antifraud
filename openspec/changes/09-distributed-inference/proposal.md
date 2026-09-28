## Why

Phases 01-08 have established GNN models for fraud detection on graphs ranging from thousands to millions of nodes. However, production deployment requires inference on a 300M-node graph using a PySpark cluster. Current approaches perform neighbor sampling on the Spark driver, which becomes a bottleneck and risks OOM errors. We need distributed sampling on executors to enable efficient, scalable GNN inference at production scale.

## What Changes

- Implement distributed GNN inference pipeline on PySpark with neighbor sampling on executors
- Implement graph partitioning (METIS/K-way) to split the graph across executors
- Implement halo management to handle cross-partition neighbor access
- Implement sampling UDF that runs on executors using local partition + halo
- Implement model loading abstraction to load trained models from phases 02-08 into Spark executors
- Create benchmark comparing driver-based sampling vs executor-based sampling
- **Stop decision:** If executor-based sampling cannot achieve at least 2× speedup over driver-based sampling while maintaining PR-AUC within 95% of full-graph inference, flag that distributed inference overhead is too high for the target graph size

## Capabilities

### New Capabilities

- `phase09-distributed-inference`: Distributed GNN inference on PySpark — graph partitioning (METIS/K-way), halo management, executor-based neighbor sampling UDF, model loading abstraction, distributed inference pipeline, performance benchmark

### Modified Capabilities

None. This phase uses trained models from phases 02-08 but does not modify their specifications.

## Impact

- Self-contained phase in `phases/phase09_distributed_inference/`:
  - `pyproject.toml` and `uv.lock` with pinned dependencies (pyspark, torch, torch_geometric, networkx, pymetis)
  - `docker/`: Dockerfile, docker-compose.yml (1 master + 2 workers), spark-defaults.conf, setup.sh
  - `src/phase09_distributed_inference/`: partition.py, halo.py, sampling_udf.py, model_loader.py, inference_pipeline.py, benchmark.py
  - `tests/`: test suite for all distributed inference components
  - `run_distributed_inference.py`: End-to-end distributed inference script
  - `lessons/`: 6 markdown lessons following 5-part structure (Explain, Code, Visualize, Practice, Solution)
  - `notebooks/`: 3 interactive Jupyter notebooks for experimentation
- Uses trained models from phases 02-08 (soft dependency — can be developed with mock models)
- Requires Docker for local Spark cluster (1 master + 2 workers); no external Hadoop needed
- No breaking changes to existing codebase
