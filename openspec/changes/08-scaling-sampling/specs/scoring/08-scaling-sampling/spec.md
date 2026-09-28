## Purpose

Implement sampling strategies (Neighbor Sampling, GraphSAINT, GraphBolt) and partitioning for scaling GNN training to large graphs (up to 300M nodes), benchmark speed/quality/memory tradeoffs, and design a partitioning strategy for distributed training.

## ADDED Requirements

### Requirement: Neighbor Sampling implementation

The system SHALL implement Neighbor Sampling with configurable sampling depth (number of hops) and fanout (number of neighbors sampled per layer).

#### Scenario: 2-hop neighbor sampling

- **WHEN** the system performs 2-hop neighbor sampling with fanout [10, 5]
- **THEN** it samples 10 neighbors for layer 0 and 5 neighbors for layer 1, producing a subgraph containing the target nodes and their sampled neighbors

#### Scenario: 3-hop neighbor sampling

- **WHEN** the system performs 3-hop neighbor sampling with fanout [15, 10, 5]
- **THEN** it samples 15, 10, and 5 neighbors per layer respectively, producing a deeper subgraph

#### Scenario: Neighbor sampling produces valid PyG subgraph

- **WHEN** the system performs neighbor sampling
- **THEN** it returns a PyG Data object with correct edge_index, node features, and node subset that can be passed directly to a GCN model

### Requirement: GraphSAINT sampling implementation

The system SHALL implement GraphSAINT sampling with vertex, edge, and subgraph variants.

#### Scenario: GraphSAINT vertex sampling

- **WHEN** the system performs GraphSAINT vertex sampling with target_nodes=N
- **THEN** it samples a subgraph centered on N vertices, including their induced subgraph structure (all edges between sampled vertices)

#### Scenario: GraphSAINT edge sampling

- **WHEN** the system performs GraphSAINT edge sampling with target_edges=M
- **THEN** it samples a subgraph induced by M randomly selected edges

#### Scenario: GraphSAINT subgraph sampling

- **WHEN** the system performs GraphSAINT subgraph sampling with target_nodes=N and target_edges=M
- **THEN** it samples a subgraph with both vertex and edge constraints, producing a balanced subgraph

#### Scenario: GraphSAINT batch generation

- **WHEN** the system runs GraphSAINT sampling for training
- **THEN** it generates multiple mini-batches over multiple epochs, each with a different random subgraph

### Requirement: GraphBolt integration

The system SHALL integrate with GraphBolt (PyTorch GNN Library) for efficient graph sampling and mini-batch generation.

#### Scenario: GraphBolt mini-batch creation

- **WHEN** GraphBolt is available and the system creates a mini-batch
- **THEN** it produces a sampled subgraph with target nodes, features, and edges that can be processed by a GCN model

#### Scenario: GraphBolt as fallback

- **WHEN** GraphBolt is not installed
- **THEN** the system falls back to the Neighbor Sampling reference implementation without errors

### Requirement: Graph partitioning for distributed training

The system SHALL implement graph partitioning using METIS/K-way partitioning to support distributed training on large graphs.

#### Scenario: K-way graph partitioning

- **WHEN** the system partitions a graph into K partitions
- **THEN** it produces K subgraphs with balanced node counts and minimized edge-cut (edges between partitions)

#### Scenario: Partition evaluation

- **WHEN** the system evaluates partition quality
- **THEN** it reports partition balance (node count variance), edge-cut ratio, and communication cost estimate

#### Scenario: Partition export for distributed training

- **WHEN** partitions are generated
- **THEN** the system exports partition assignments as a node → partition_id mapping file and partition subgraphs as separate Data objects

### Requirement: Sampling benchmark framework

The system SHALL implement a benchmark framework that compares Neighbor Sampling, GraphSAINT, GraphBolt, and full-graph GCN across speed, quality, and memory dimensions.

#### Scenario: Benchmark sampling strategies

- **WHEN** the system runs a benchmark across sampling strategies
- **THEN** it measures: PR-AUC (quality), training time per epoch (speed), peak memory usage (memory), and nodes processed per second (throughput)

#### Scenario: Benchmark with varying fanout sizes

- **WHEN** the system benchmarks with different fanout sizes (5, 10, 15, 20, 25)
- **THEN** it produces a quality-vs-speed curve showing the tradeoff between fanout and PR-AUC

#### Scenario: Benchmark with varying sampling depths

- **WHEN** the system benchmarks with different sampling depths (1, 2, 3, 4 hops)
- **THEN** it produces a quality-vs-over-smoothing analysis showing optimal depth

#### Scenario: Benchmark report

- **WHEN** the benchmark completes
- **THEN** the system produces a report with per-strategy metrics, recommended configuration, and stop decision evaluation

### Requirement: Scaling benchmark

The system SHALL measure how model quality and training speed scale with graph size.

#### Scenario: Scale from small to large graphs

- **WHEN** the system benchmarks on graphs of increasing size (1K, 10K, 100K, 1M nodes)
- **THEN** it measures PR-AUC, training time, and memory usage at each scale and produces a scaling curve

#### Scenario: Extrapolate to 300M nodes

- **WHEN** the system has scaling measurements up to 1M nodes
- **THEN** it extrapolates estimated training time and memory requirements for a 300M-node graph using the best sampling strategy

#### Scenario: Stop decision on scaling

- **WHEN** the benchmark completes and no sampling strategy achieves PR-AUC >= 0.85 × full-graph PR-AUC while being 2× faster than full-graph training at scale
- **THEN** the system prints "⚠ Sampling quality is insufficient for the target graph size"

## ADDED Requirements

### Requirement: Sampling-aware GCN training

The system SHALL modify the GCN training pipeline to support sampling-based mini-batch training.

#### Scenario: GCN with neighbor sampling

- **WHEN** the GCN model is trained with neighbor sampling enabled
- **THEN** it processes mini-batches via sampled subgraphs instead of the full graph

#### Scenario: GCN with GraphSAINT

- **WHEN** the GCN model is trained with GraphSAINT sampling
- **THEN** it processes GraphSAINT-generated subgraph batches

### Requirement: Sampling support for temporal models

The system SHALL support sampling for TGN and GRN training on large temporal graphs.

#### Scenario: Neighbor sampling for TGN

- **WHEN** TGN is trained with neighbor sampling on a large temporal graph
- **THEN** it samples neighbors for each time-stamped event, reducing memory footprint

#### Scenario: Snapshot sampling for GRN

- **WHEN** GRN is trained with sampling on large temporal snapshots
- **THEN** it samples subgraphs from large snapshots before processing through the RNN
