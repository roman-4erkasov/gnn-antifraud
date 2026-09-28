## Purpose

Optimize graph sampling for training large GNNs and design a partitioning strategy for scaling to 300M+ node graphs.

## ADDED Requirements

### Requirement: Neighbor Sampling

The system SHALL implement Neighbor Sampling (from PyG) to sample subgraphs for efficient mini-batch training of GNNs.

#### Scenario: Neighbor Sampling configuration

- **WHEN** a GNN training loop is initialized with neighbor sampling enabled
- **THEN** the system samples a configurable number of neighbors per layer (e.g., [10, 5] for 2-layer GCN)

#### Scenario: Sampling impact on quality

- **WHEN** models are trained with and without neighbor sampling
- **THEN** the system reports the tradeoff: training time reduction vs. PR-AUC/Brier score degradation

### Requirement: GraphSAINT sampling

The system SHALL implement GraphSAINT sampling (uniform, random-walk, or degree-based) as an alternative to Neighbor Sampling for node classification tasks.

#### Scenario: GraphSAINT training

- **WHEN** GraphSAINT is enabled
- **THEN** the system samples subgraphs according to the chosen sampling strategy (uniform, random-walk, or degree-based) and trains the GNN on these subgraphs

#### Scenario: GraphSAINT vs Neighbor Sampling

- **WHEN** both GraphSAINT and Neighbor Sampling are used on the same dataset
- **THEN** the system reports and compares convergence speed, training time per epoch, and final model quality

### Requirement: Partitioning strategy

The system SHALL design and document a partitioning strategy for scaling to graphs with 300M+ nodes, including: (a) structural feature precomputation (Hadoop/Spark-based), (b) graph partitioning (METIS or similar), and (c) data loading architecture.

#### Scenario: Partitioning design document

- **WHEN** the system reaches the scaling phase (Phase 8)
- **THEN** it produces a documented partitioning strategy specifying: partitioning algorithm, number of partitions, edge-cut target, and feature precomputation pipeline

#### Scenario: Precomputation with Spark

- **WHEN** structural features (e.g., ReCWE, PageRank, community labels) need precomputation at scale
- **THEN** the system uses Spark GraphX or PySpark to compute these features distributedly and writes results to a partitioned storage format (Parquet)
