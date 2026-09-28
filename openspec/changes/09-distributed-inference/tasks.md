## 1. Setup

- [ ] 1.1 Create `phases/phase09_distributed_inference/` directory structure with `pyproject.toml`, `src/phase09_distributed_inference/`, `tests/`, `lessons/`, `notebooks/`; verify all directories exist
- [ ] 1.2 Configure `pyproject.toml` with dependencies (pyspark>=3.4, torch, torch_geometric, networkx, scipy, numpy, pymetis optional); verify `uv sync` succeeds
- [ ] 1.3 Create `src/phase09_distributed_inference/__init__.py` with module docstring and version; verify Python import works

## 2. Docker setup

- [ ] 2.1 Create `phases/phase09_distributed_inference/docker/` directory; verify directory exists
- [ ] 2.2 Create `phases/phase09_distributed_inference/docker/Dockerfile` with Python 3.10, PySpark 3.4, PyTorch 2.x, PyG, networkx, pymetis; verify Docker image builds successfully
- [ ] 2.3 Create `phases/phase09_distributed_inference/docker/docker-compose.yml` with 1 spark-master + 2 spark-workers; verify `docker-compose up` starts cluster
- [ ] 2.4 Create `phases/phase09_distributed_inference/docker/spark-defaults.conf` with executor memory (4g), cores (2), and PySpark worker settings; verify Spark UI accessible at localhost:8080
- [ ] 2.5 Add `docker/setup.sh` script that builds images and starts cluster; verify script completes without errors
- [ ] 2.6 Test cluster connectivity: submit simple PySpark job to master, verify it runs on workers; verify job completes successfully

## 3. Graph partitioning

- [ ] 3.1 Create `phases/phase09_distributed_inference/src/phase09_distributed_inference/partition.py` with module docstring, imports (networkx, scipy.sparse, numpy), and `GraphPartitioner(n_partitions, method="metis")` class stub; verify Python syntax
- [ ] 3.2 Implement `GraphPartitioner._metis_partition(graph, n_partitions)` using pymetis if available; verify output is node → partition_id mapping
- [ ] 3.3 Implement `GraphPartitioner._spectral_partition(graph, n_partitions)` using scipy sparse eigenvectors; verify output matches METIS interface
- [ ] 3.4 Implement `GraphPartitioner.partition(edge_index, node_features, n_partitions)` returning partition assignments, partition subgraphs, and edge-cut count; verify all outputs are correct
- [ ] 3.5 Implement `GraphPartitioner.evaluate_partitions()` reporting partition balance (node count std), edge-cut ratio; verify metrics are valid numbers
- [ ] 3.6 Implement `GraphPartitioner.export_partitions(partition_assignments, output_dir)` saving partition files for Spark; verify output directory contains partition files

## 4. Halo management

- [ ] 4.1 Create `phases/phase09_distributed_inference/src/phase09_distributed_inference/halo.py` with module docstring, imports (networkx, numpy), and `HaloManager(halo_depth=3)` class stub; verify Python syntax
- [ ] 4.2 Implement `HaloManager.compute_halo(partition_subgraph, full_graph, halo_depth)` finding all k-hop neighbors from other partitions; verify output contains correct neighbor sets
- [ ] 4.3 Implement `HaloManager.export_halo(halo_data, partition_id, output_dir)` saving halo data alongside partition; verify output file exists
- [ ] 4.4 Implement `HaloManager.load_halo(partition_id, input_dir)` loading halo data for a partition; verify loaded halo matches exported halo
- [ ] 4.5 Implement `HaloManager.validate_halo(partition_subgraph, halo_data, full_graph)` checking halo completeness; verify validation passes for correct halo

## 5. Sampling UDF

- [ ] 5.1 Create `phases/phase09_distributed_inference/src/phase09_distributed_inference/sampling_udf.py` with module docstring, imports (torch, torch_geometric), and `SamplingUDF(fanout_sizes, device="cpu")` class stub; verify Python syntax
- [ ] 5.2 Implement `SamplingUDF.sample_neighbors(target_nodes, partition_data, halo_data)` performing neighbor sampling from local partition + halo; verify output is PyG Data with correct shape
- [ ] 5.3 Implement `SamplingUDF.__call__(partition_iterator)` as mapPartitions-compatible function; verify yields (partition_id, sampled_subgraphs) tuples
- [ ] 5.4 Implement fanout validation — verify sum(fanout_sizes) < partition_size + halo_size
- [ ] 5.5 Implement `SamplingUDF.get_sampling_stats(target_nodes)` returning stats: n_sampled_nodes, n_sampled_edges, coverage_ratio; verify stats are valid numbers

## 6. Model loading

- [ ] 6.1 Create `phases/phase09_distributed_inference/src/phase09_distributed_inference/model_loader.py` with module docstring, imports (torch, pyspark), and `ModelLoader()` class stub; verify Python syntax
- [ ] 6.2 Implement `ModelLoader.serialize_model(model, output_path)` saving PyTorch model; verify output file exists
- [ ] 6.3 Implement `ModelLoader.load_model(model_path, model_class)` loading model from file; verify loaded model matches original
- [ ] 6.4 Implement `ModelLoader.broadcast_model(spark_context, model_path)` broadcasting model to executors; verify broadcast variable is accessible on executors
- [ ] 6.5 Implement `ModelLoader.load_on_executor(broadcast_model)` deserializing model on executor; verify model.forward() works

## 7. Inference pipeline

- [ ] 7.1 Create `phases/phase09_distributed_inference/src/phase09_distributed_inference/inference_pipeline.py` with module docstring, imports (pyspark, torch), and `DistributedInferencePipeline(spark_session, model, config)` class stub; verify Python syntax
- [ ] 7.2 Implement `DistributedInferencePipeline.partition_graph(edge_index, node_features, n_partitions)` partitioning graph and distributing to executors; verify partitions are distributed
- [ ] 7.3 Implement `DistributedInferencePipeline.load_halos(partition_ids)` loading halos for all partitions; verify halos are loaded on executors
- [ ] 7.4 Implement `DistributedInferencePipeline.run_inference(target_nodes)` orchestrating sampling + model inference on executors; verify predictions are collected
- [ ] 7.5 Implement `DistributedInferencePipeline.collect_predictions()` gathering predictions from all executors; verify output is complete prediction array

## 8. Benchmark

- [ ] 8.1 Create `phases/phase09_distributed_inference/src/phase09_distributed_inference/benchmark.py` with module docstring, imports (time, tracemalloc, numpy), and `DistributedBenchmark()` class stub; verify Python syntax
- [ ] 8.2 Implement `DistributedBenchmark.run_driver_sampling(data, model)` performing sampling on driver; verify output includes time, memory, predictions
- [ ] 8.3 Implement `DistributedBenchmark.run_executor_sampling(spark_session, data, model, n_executors)` performing sampling on executors; verify output includes time, memory, predictions
- [ ] 8.4 Implement `DistributedBenchmark.compare_approaches()` comparing driver vs executor sampling; verify output includes speedup factor, memory savings, PR-AUC comparison
- [ ] 8.5 Implement `DistributedBenchmark.evaluate_stop_decision(speedup, pr_auc_ratio)` checking if executor sampling achieves 2× speedup and 95% PR-AUC; verify output is boolean with detailed finding
- [ ] 8.6 Implement `DistributedBenchmark.generate_report()` producing human-readable benchmark report; verify output includes all metrics and stop decision

## 9. Unit tests

- [ ] 9.1 Create `phases/phase09_distributed_inference/tests/test_partition.py` with tests for graph partitioning — verify partition balance, edge-cut ratio, METIS/spectral fallback; verify with pytest
- [ ] 9.2 Create `phases/phase09_distributed_inference/tests/test_halo.py` with tests for halo management — verify halo completeness, export/load, validation; verify with pytest
- [ ] 9.3 Create `phases/phase09_distributed_inference/tests/test_sampling_udf.py` with tests for sampling UDF — verify neighbor sampling, fanout enforcement, PyG output; verify with pytest
- [ ] 9.4 Create `phases/phase09_distributed_inference/tests/test_model_loader.py` with tests for model loading — verify serialization, broadcast, executor loading; verify with pytest
- [ ] 9.5 Create `phases/phase09_distributed_inference/tests/test_inference_pipeline.py` with tests for inference pipeline — verify partitioning, halo loading, inference, prediction collection; verify with pytest
- [ ] 9.6 Create `phases/phase09_distributed_inference/tests/test_benchmark.py` with tests for benchmark — verify driver/executor sampling, comparison, stop decision; verify with pytest
- [ ] 9.7 Create `phases/phase09_distributed_inference/tests/test_integration.py` with end-to-end test: partition small graph, load halo, sample on executor, run inference; verify with pytest

## 10. Educational content — lessons

- [ ] 10.1 Create `phases/phase09_distributed_inference/lessons/01-distributed-systems.md` following 5-part structure (Explain, Code, Visualize, Practice, Solution) covering distributed systems basics — why single-machine inference fails at scale, Spark architecture, executors vs driver; verify file exists and contains all 5 sections
- [ ] 10.2 Create `phases/phase09_distributed_inference/lessons/02-graph-partitioning.md` following 5-part structure covering graph partitioning — METIS algorithm, spectral partitioning, partition quality metrics; verify file exists and contains all 5 sections
- [ ] 10.3 Create `phases/phase09_distributed_inference/lessons/03-halo-management.md` following 5-part structure covering halo management — what is a halo, how to compute k-hop neighbors, halo storage; verify file exists and contains all 5 sections
- [ ] 10.4 Create `phases/phase09_distributed_inference/lessons/04-sampling-udf.md` following 5-part structure covering sampling UDF — mapPartitions, neighbor sampling on executor, PyG Data construction; verify file exists and contains all 5 sections
- [ ] 10.5 Create `phases/phase09_distributed_inference/lessons/05-model-serving.md` following 5-part structure covering model serving — PyTorch serialization, Spark broadcast, executor model loading; verify file exists and contains all 5 sections
- [ ] 10.6 Create `phases/phase09_distributed_inference/lessons/06-production-deployment.md` following 5-part structure covering production deployment — memory tuning, performance monitoring, scaling to 300M nodes; verify file exists and contains all 5 sections

## 11. Educational content — notebooks

- [ ] 11.1 Create `phases/phase09_distributed_inference/notebooks/01-distributed-inference-demo.ipynb` with cells that partition a graph, load halos, run sampling on executor, and collect predictions; verify notebook executes without errors
- [ ] 11.2 Create `phases/phase09_distributed_inference/notebooks/02-performance-tuning.ipynb` with cells that benchmark driver vs executor sampling, tune partition size and halo depth, and visualize speedup; verify notebook executes without errors
- [ ] 11.3 Create `phases/phase09_distributed_inference/notebooks/03-scaling-analysis.ipynb` with cells that run distributed inference on graphs of increasing size (1M, 10M, 100M), measure throughput and memory, and extrapolate to 300M; verify notebook executes without errors

## 12. End-to-end pipeline

- [ ] 12.1 Create `phases/phase09_distributed_inference/run_distributed_inference.py` that partitions graph, loads halos, runs distributed inference, benchmarks performance, and generates report; verify script runs without errors
- [ ] 12.2 Print summary: partition stats, halo stats, inference throughput, speedup factor, stop decision; verify output is human-readable
- [ ] 12.3 Implement stop decision check: if executor sampling does not achieve 2× speedup and 95% PR-AUC, print warning; verify warning is printed when conditions not met
- [ ] 12.4 Save benchmark results to `outputs/distributed_benchmark.json`; verify file exists and is valid JSON
- [ ] 12.5 Run full test suite `pytest phases/phase09_distributed_inference/tests/ -v` and verify all tests pass; verify exit code 0
- [ ] 12.6 Run `phases/phase09_distributed_inference/run_distributed_inference.py` end-to-end with synthetic 1M-node graph and verify: graph partitioned, halos loaded, inference completed, benchmark generated, stop decision evaluated; confirm exit code 0
