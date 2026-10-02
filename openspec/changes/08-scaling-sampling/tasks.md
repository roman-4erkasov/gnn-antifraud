## 1. Data loading

- [ ] 1.1 Create `phases/phase08_scaling_sampling/src/phase08_scaling_sampling/data_loader.py` with module docstring, imports (torch, torch_geometric.data, networkx, numpy, os, pathlib), and `DataLoader(data_dir="data")` class stub; verify Python syntax
- [ ] 1.2 Implement `DataLoader.check_data_exists()` checking if synthetic scaling datasets exist in the data directory; verify returns boolean
- [ ] 1.3 Implement `DataLoader.generate_synthetic_data(size)` generating a synthetic graph with the specified number of nodes (100K, 1M, 10M) using power-law degree distribution; verify output is a valid PyG Data object saved to data directory
- [ ] 1.4 Implement `DataLoader.load_data(size)` loading a previously generated synthetic dataset from the data directory; verify returns the correct PyG Data object
- [ ] 1.5 Implement `generate_scaling_datasets(data_dir="data", sizes=[100_000, 1_000_000, 10_000_000])` creating graphs of increasing size and saving them to the shared data directory; verify all datasets are generated and saved
- [ ] 1.6 Verify output is compatible with `mini_batch.py` and `negative_sampling.py` — ensure generated PyG Data objects have the required attributes (edge_index, x, y) and can be loaded by downstream modules; verify with integration test

## 2. Neighbor Sampling implementation

- [ ] 2.1 Create `phases/phase08_scaling_sampling/src/phase08_scaling_sampling/neighbor.py` with module docstring, imports (torch, torch_geometric.nn, torch_geometric.loader), and `NeighborSampler(fanout_sizes, device="cpu")` class stub; verify Python syntax
- [ ] 2.2 Implement `NeighborSampler.sample_subgraph(target_nodes, edge_index, node_features)` sampling neighbors up to specified depths with given fanout sizes; verify output is PyG Data with correct shape
- [ ] 2.3 Implement `NeighborSampler.forward(data, target_nodes)` producing a sampled subgraph batch for training via __iter__ protocol; verify yields PyG Data objects
- [ ] 2.4 Implement fanout_size configuration validation — verify that sum(fanout_sizes) < total_nodes and each fanout_size > 0
- [ ] 2.5 Implement `NeighborSampler.get_subgraph_stats(target_nodes)` returning stats: n_sampled_nodes, n_sampled_edges, coverage_ratio; verify stats are valid numbers
- [ ] 2.6 Implement `NeighborSampler.to_pyg_loader(dataset)` converting NeighborSampler to a PyG NeighborLoader-compatible interface; verify loader yields correct batch shapes

## 3. GraphSAINT sampling implementation

- [ ] 3.1 Create `phases/phase08_scaling_sampling/src/phase08_scaling_sampling/graphsaint.py` with module docstring, imports (torch, numpy, scipy.sparse), and `GraphSAINTSampler(sampling_mode="vertex", target_size=10000)` class stub; verify Python syntax
- [ ] 3.2 Implement `GraphSAINTSampler.vertex_sample(n_nodes)` performing vertex-induced subgraph sampling; verify output is a valid PyG Data with all edges between sampled nodes
- [ ] 3.3 Implement `GraphSAINTSampler.edge_sample(n_edges)` performing edge-induced subgraph sampling; verify output contains at least n_edges edges
- [ ] 3.4 Implement `GraphSAINTSampler.subgraph_sample(n_nodes, n_edges)` performing subgraph sampling with both constraints; verify output has both node and edge counts within bounds
- [ ] 3.5 Implement `GraphSAINTSampler.batches(n_batches, epoch)` generating multiple random subgraph batches for training; verify yields list of PyG Data objects
- [ ] 3.6 Implement `GraphSAINTSampler.importance_weight(target_nodes, sampled_subgraph)` computing importance weights for unbiased gradient estimation; verify weights are positive floats

## 4. GraphBolt integration

- [ ] 4.1 Implement `has_graphbolt()` checking if GraphBolt is installed; verify returns boolean
- [ ] 4.2 Implement `GraphBoltSampler(fanout_sizes, device="cpu")` wrapper around GraphBolt's sampling when available; verify produces sampled batches compatible with GCN models
- [ ] 4.3 Implement fallback: when GraphBolt is unavailable, delegate to `NeighborSampler` from `phases/phase08_scaling_sampling/src/phase08_scaling_sampling/neighbor.py`; verify behavior is equivalent
- [ ] 4.4 Implement `GraphBoltSampler.create_dataloader(data, batch_size, fanout_sizes)` creating a GraphBolt data loader; verify loader yields correct mini-batches

## 5. Graph partitioning

- [ ] 5.1 Create `phases/phase08_scaling_sampling/src/phase08_scaling_sampling/partition.py` with module docstring, imports (numpy, networkx, scipy.sparse), and `GraphPartitioner(n_partitions=4)` class stub; verify Python syntax
- [ ] 5.2 Implement `_spectral_partition(graph, n_partitions)` performing spectral partitioning using networkx/scipy; verify output is a node → partition_id mapping
- [ ] 5.3 Implement `_metis_partition(graph, n_partitions)` performing METIS K-way partitioning using pymetis/igraph if available; verify output matches spectral partition interface
- [ ] 5.4 Implement `GraphPartitioner.partition(edge_index, node_features, n_partitions=4)` partitioning a graph and returning partition assignments, partition subgraphs, and edge-cut count; verify all outputs are correct
- [ ] 5.5 Implement `GraphPartitioner.evaluate_partitions()` reporting partition balance (node count std), edge-cut ratio, and communication cost; verify all metrics are valid numbers
- [ ] 5.6 Implement `_export_partitions(partition_assignments, subgraphs, output_dir)` saving partition files for distributed training; verify output directory contains partition files

## 6. Sampling benchmark framework

- [ ] 6.1 Create `phases/phase08_scaling_sampling/src/phase08_scaling_sampling/scaling_benchmark.py` with module docstring, imports (torch, tracemalloc, time, numpy), and `SamplingBenchmark` class stub; verify Python syntax
- [ ] 6.2 Implement `SamplingBenchmark.run_strategy(strategy_name, data, strategy_config)` running a specific sampling strategy and measuring PR-AUC, training time, memory, throughput; verify output dict with all metrics
- [ ] 6.3 Implement `SamplingBenchmark.compare_strategies(data, strategies)` running all strategies on the same data and producing a comparison report; verify output is a structured comparison dict
- [ ] 6.4 Implement `SamplingBenchmark.fanout_curve(data, base_model, fanout_sizes=[5, 10, 15, 20, 25])` benchmarking PR-AUC and speed vs fanout size; verify output is a dict mapping fanout → (PR-AUC, time, memory)
- [ ] 6.5 Implement `SamplingBenchmark.depth_curve(data, base_model, depths=[1, 2, 3, 4])` benchmarking PR-AUC and over-smoothing vs sampling depth; verify output is a dict mapping depth → (PR-AUC, embedding_similarity)
- [ ] 6.6 Implement `SamplingBenchmark.report(best_strategy, all_results)` generating a human-readable benchmark report; verify output includes strategy name, metrics, and recommended configuration

## 7. Scaling benchmark

- [ ] 7.1 Implement `generate_scaling_graphs(sizes=[1_000, 10_000, 100_000, 1_000_000], edge_density=0.001, feature_dim=30)` generating synthetic graphs with power-law degree distribution for scaling tests; verify each graph has correct size and realistic structure
- [ ] 7.2 Implement `run_scaling_benchmark(graph_sizes, strategies)` benchmarking all strategies across all graph sizes; verify output is a matrix (size × strategy → metrics)
- [ ] 7.3 Implement `measure_throughput(data, strategy, nodes_per_second)` computing nodes/second for a strategy; verify output is a positive float
- [ ] 7.4 Implement `measure_memory(data, strategy)` measuring peak memory usage for a strategy; verify output is bytes (convertible to GB)
- [ ] 7.5 Implement `extrapolate_to_300m(scaling_results, best_strategy)` extrapolating training time and memory for a 300M-node graph; verify output includes estimated_time_seconds, estimated_memory_gb, confidence_range
- [ ] 7.6 Implement `evaluate_stop_decision(full_graph_pr_auc, sampling_pr_auc, full_graph_time, sampling_time)` checking if sampling achieves >= 0.85 × full-graph PR-AUC and >= 2× speedup; verify output is a boolean with detailed finding

## 8. Sampling wrapper for existing models

- [ ] 8.1 Create `phases/phase08_scaling_sampling/src/phase08_scaling_sampling/sampling_wrapper.py` with module docstring, imports (torch, torch_geometric.nn), and `SamplingWrapper(model, sampler)` class stub; verify Python syntax
- [ ] 8.2 Implement `SamplingWrapper.__init__(model, sampler)` storing the wrapped model and sampler; verify wrapper can be instantiated with any PyG model
- [ ] 8.3 Implement `SamplingWrapper.train_with_sampling(data, epochs=50, lr=0.001)` training the wrapped model using mini-batches from the sampler; verify training loop uses sampled batches and produces a trained model
- [ ] 8.4 Implement `SamplingWrapper.forward(data)` delegating to the wrapped model's forward pass; verify output matches the wrapped model's output
- [ ] 8.5 Implement `train_with_sampling(model, data, sampler, epochs=50, lr=0.001)` function training any PyG model with the specified sampler; verify works with GCN from Phase 02 and TGN/GRN from Phase 06
- [ ] 8.6 Implement `compare_sampling_vs_full_graph(model, data, sampler, epochs=50, lr=0.001)` comparing sampling-based and full-graph training; verify output includes PR-AUC, time, memory for both

## 9. Interactive exercises (exercises/)

- [ ] 9.1 Create `phases/phase08_scaling_sampling/exercises/` directory
- [ ] 9.2 Create `01-scaling-challenges.ipynb` with task to measure memory usage
- [ ] 9.3 Create `02-mini-batch-training.ipynb` with task to implement mini-batch training
- [ ] 9.4 Create `03-negative-sampling.ipynb` with task to compare sampling strategies
- [ ] 9.5 Create `04-graph-sampling.ipynb` with task to sample subgraphs
- [ ] 9.6 Create `05-memory-optimization.ipynb` with task to optimize memory usage
- [ ] 9.7 Create `06-production-considerations.ipynb` with task to design production pipeline

## 10. Unit tests

- [ ] 10.1 Create `phases/phase08_scaling_sampling/tests/test_neighbor.py` with tests for Neighbor Sampling — verify correct subgraph sizes, fanout enforcement, valid PyG output; verify with pytest
- [ ] 10.2 Create `phases/phase08_scaling_sampling/tests/test_graphsaint.py` with tests for GraphSAINT — verify vertex/edge/subgraph sampling, importance weights, batch generation; verify with pytest
- [ ] 10.3 Create `phases/phase08_scaling_sampling/tests/test_graphbolt.py` with tests for GraphBolt integration — verify fallback to NeighborSampler when GraphBolt unavailable; verify with pytest
- [ ] 10.4 Create `phases/phase08_scaling_sampling/tests/test_partition.py` with tests for graph partitioning — verify partition balance, edge-cut ratio, partition export; verify with pytest
- [ ] 10.5 Create `phases/phase08_scaling_sampling/tests/test_scaling_benchmark.py` with tests for benchmark framework — verify metric computation, fanout curve, depth curve; verify with pytest
- [ ] 10.6 Create `phases/phase08_scaling_sampling/tests/test_scaling.py` with tests for scaling benchmark — verify synthetic graph generation, throughput measurement, extrapolation; verify with pytest
- [ ] 10.7 Create `phases/phase08_scaling_sampling/tests/test_integration.py` with end-to-end test: generate small graph, train GCN with neighbor sampling, verify metrics are comparable to full-graph; verify with pytest

## 11. Educational content — lessons

- [ ] 11.1 Create `phases/phase08_scaling_sampling/lessons/01-scaling-challenges.md` following 5-part structure (Explain, Code, Visualize, Practice, Solution) covering why GNNs don't scale naively — O(N) memory, message-passing cost, and motivation for sampling; link Practice section to `exercises/01-scaling-challenges.ipynb`; verify file exists and contains all 5 sections
- [ ] 11.2 Create `phases/phase08_scaling_sampling/lessons/02-mini-batch-training.md` following 5-part structure covering Neighbor Sampling and GraphSAINT mini-batch training — how to train on subgraphs instead of the full graph; link Practice section to `exercises/02-mini-batch-training.ipynb`; verify file exists and contains all 5 sections
- [ ] 11.3 Create `phases/phase08_scaling_sampling/lessons/03-negative-sampling.md` following 5-part structure covering negative sampling strategies for fraud detection — how to sample non-fraud nodes efficiently in imbalanced graphs; link Practice section to `exercises/03-negative-sampling.ipynb`; verify file exists and contains all 5 sections
- [ ] 11.4 Create `phases/phase08_scaling_sampling/lessons/04-graph-sampling.md` following 5-part structure covering graph partitioning and distributed training — METIS/K-way partitioning for splitting a 300M-node graph; link Practice section to `exercises/04-graph-sampling.ipynb`; verify file exists and contains all 5 sections
- [ ] 11.5 Create `phases/phase08_scaling_sampling/lessons/05-memory-optimization.md` following 5-part structure covering memory optimization techniques — fanout tuning, feature caching, and gradient checkpointing for large graphs; link Practice section to `exercises/05-memory-optimization.ipynb`; verify file exists and contains all 5 sections
- [ ] 11.6 Create `phases/phase08_scaling_sampling/lessons/06-production-considerations.md` following 5-part structure covering production deployment — latency requirements, model serving at scale, and monitoring sampling quality; link Practice section to `exercises/06-production-considerations.ipynb`; verify file exists and contains all 5 sections

## 12. Educational content — notebooks

- [ ] 12.1 Create `phases/phase08_scaling_sampling/notebooks/01-scaling-experiments.ipynb` with cells that run scaling benchmarks across graph sizes, compare sampling strategies, and visualize speed/quality tradeoffs; verify notebook executes without errors
- [ ] 12.2 Create `phases/phase08_scaling_sampling/notebooks/02-sampling-strategies.ipynb` with cells that compare Neighbor Sampling, GraphSAINT, and GraphBolt on the same graph — tune fanout, depth, and batch size; verify notebook executes without errors
- [ ] 12.3 Create `phases/phase08_scaling_sampling/notebooks/03-performance-benchmarks.ipynb` with cells that profile throughput, memory, and PR-AUC; extrapolate to 300M nodes; reproduce the stop decision analysis; verify notebook executes without errors

## 13. End-to-end scaling pipeline

- [ ] 13.1 Create `phases/phase08_scaling_sampling/run_scaling.py` that generates scaling graphs, runs all sampling strategies, benchmarks, partitions, extrapolates to 300M, and generates report; verify script runs without errors
- [ ] 13.2 Print summary: best strategy, best fanout, best depth, extrapolated 300M requirements, stop decision check; verify output is human-readable
- [ ] 13.3 Implement stop decision check: if no sampling strategy achieves PR-AUC >= 0.85 × full-graph PR-AUC while being 2× faster, print "⚠ Sampling quality is insufficient for the target graph size"
- [ ] 13.4 Save benchmark results to `outputs/scaling_benchmark.json`; verify file exists and is valid JSON
- [ ] 13.5 Run full test suite `pytest phases/phase08_scaling_sampling/tests/ -v` and verify all tests pass including new sampling tests
- [ ] 13.6 Run `phases/phase08_scaling_sampling/run_scaling.py` end-to-end with synthetic data and verify: all strategies benchmarked, scaling curve generated, partition evaluated, 300M extrapolation produced, stop decision check passes; confirm exit code 0
