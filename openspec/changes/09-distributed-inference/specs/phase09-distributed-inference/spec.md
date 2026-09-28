## Purpose

Enables distributed GNN inference on PySpark clusters by partitioning graphs across executors, managing cross-partition neighbor access through halos, and performing neighbor sampling on executors rather than the driver, allowing efficient inference on graphs with hundreds of millions of nodes.

## ADDED Requirements

### Requirement: Graph partitioning across executors
The system SHALL partition a large graph into N partitions suitable for distribution across Spark executors, where each partition contains a subset of nodes and their internal edges. The partitioning MUST support both METIS-based partitioning (when available) and spectral partitioning as a fallback.

#### Scenario: Successful METIS partitioning
- **WHEN** METIS library is available and a graph with 1M nodes is partitioned into 10 partitions
- **THEN** the system produces 10 partition files, each containing a subset of nodes and edges, with partition balance (node count standard deviation) within 20% of ideal uniform distribution

#### Scenario: Fallback to spectral partitioning
- **WHEN** METIS library is unavailable and a graph is partitioned
- **THEN** the system uses spectral partitioning and produces valid partitions without errors

### Requirement: Halo management for cross-partition neighbors
The system SHALL manage halo data containing k-hop neighbors from other partitions, enabling each executor to perform neighbor sampling without cross-partition communication during inference. The halo depth MUST be configurable (default 3 hops to match GCN depth).

#### Scenario: Halo precomputation
- **WHEN** a graph is partitioned with halo depth 3
- **THEN** each partition includes a halo containing all 1-hop, 2-hop, and 3-hop neighbors from other partitions, and halo data is stored separately from partition data

#### Scenario: Halo loading during inference
- **WHEN** an executor loads a partition for inference
- **THEN** the executor loads both the partition data and halo data, making cross-partition neighbors accessible for sampling

### Requirement: Executor-based neighbor sampling
The system SHALL perform neighbor sampling on Spark executors using a UDF that operates on the local partition and halo data. Sampling MUST produce the same output format as single-machine sampling (PyG Data objects with sampled subgraphs).

#### Scenario: Sampling on executor
- **WHEN** a Spark executor receives a batch of target nodes and runs the sampling UDF
- **THEN** the executor samples neighbors from local partition + halo, produces PyG Data objects with correct edge_index and node features, and returns sampled subgraphs to the driver

#### Scenario: Sampling with missing halo nodes
- **WHEN** a target node requires neighbors that are not in the halo (beyond k-hop distance)
- **THEN** the sampling UDF either truncates the sampling path or raises an error indicating insufficient halo depth

### Requirement: Model loading abstraction
The system SHALL provide an abstraction for loading trained PyTorch GNN models into Spark executors. The model loader MUST support any PyTorch model that follows the standard PyTorch model interface (forward method).

#### Scenario: Loading trained model
- **WHEN** a trained GCN model from phase 02 is loaded via the model loader
- **THEN** the model is serialized, distributed to all executors, and deserialized on each executor for inference

#### Scenario: Model inference on executor
- **WHEN** an executor receives sampled subgraphs and a loaded model
- **THEN** the executor runs model.forward() on the sampled subgraphs and returns predictions

### Requirement: Distributed inference pipeline
The system SHALL orchestrate end-to-end distributed inference: partition graph → distribute partitions + halos → load model → sample neighbors on executors → run inference → collect predictions. The pipeline MUST handle graphs with 100M+ nodes without driver OOM.

#### Scenario: End-to-end inference on large graph
- **WHEN** a 100M-node graph is processed through the distributed inference pipeline
- **THEN** the driver memory usage stays below 10GB, all executors participate in inference, and the system produces predictions for all nodes

#### Scenario: Inference with multiple batches
- **WHEN** the graph is too large to fit in executor memory
- **THEN** the pipeline processes the graph in batches, with each executor handling a subset of target nodes per batch

### Requirement: Performance benchmark
The system SHALL provide a benchmark comparing driver-based sampling vs executor-based sampling, measuring throughput (nodes/second), memory usage, and inference quality (PR-AUC if labels available). The benchmark MUST report speedup factor and memory savings.

#### Scenario: Benchmark comparison
- **WHEN** the benchmark runs on a 10M-node graph with 10 executors
- **THEN** the benchmark reports: driver sampling time, executor sampling time, speedup factor, driver memory usage, executor memory usage, and PR-AUC for both approaches

#### Scenario: Stop decision evaluation
- **WHEN** the benchmark completes
- **THEN** the system evaluates whether executor-based sampling achieves at least 2× speedup and PR-AUC within 95% of full-graph inference, and outputs a stop decision (pass/fail)
