## 1. Setup

- [ ] 1.1 Create `phases/phase10_distributed_gnn_frameworks/` directory structure with `pyproject.toml`, `src/phase10_distributed_gnn_frameworks/`, `tests/`, `benchmarks/`, `lessons/`, `notebooks/`; verify all directories exist
- [ ] 1.2 Configure `pyproject.toml` with base dependencies (torch, torch_geometric, numpy, pandas); verify `uv sync` succeeds
- [ ] 1.3 Create `src/phase10_distributed_gnn_frameworks/__init__.py` with module docstring and version; verify Python import works

## 2. Data loading

- [ ] 2.1 Create `src/phase10_distributed_gnn_frameworks/data_loader.py` with `DataLoader` class containing `check_data_exists()`, `generate_benchmark_data(size)`, and `load_data()` methods; verify Python syntax and class instantiation
- [ ] 2.2 Implement `DataLoader.check_data_exists()` checking for existing benchmark graphs in `data/` directory; verify returns correct boolean for present/missing data
- [ ] 2.3 Implement `DataLoader.generate_benchmark_data(size)` generating synthetic graph with power-law degree distribution, node features, and fraud labels at specified scale (1M, 10M, 100M nodes); verify output files written to `data/` directory
- [ ] 2.4 Implement `DataLoader.load_data()` loading generated benchmark graph from `data/` directory into framework-agnostic format; verify loaded graph matches generated data
- [ ] 2.5 Implement `generate_framework_benchmarks()` function creating standardized test graphs for all three frameworks (DGL, AliGraph, Quiver) at 1M, 10M, 100M scales; verify output is compatible with `dgl_benchmark.py`, `aligraph_benchmark.py`, `quiver_benchmark.py`
- [ ] 2.6 Run `generate_framework_benchmarks()` end-to-end and verify benchmark data files exist in `data/` directory for all three frameworks

## 3. DGL Distributed benchmark

- [ ] 3.1 Create `phases/phase10_distributed_gnn_frameworks/benchmarks/dgl_config.yaml` with DGL Distributed configuration (num_machines, num_gpus, ip_config); verify file exists
- [ ] 3.2 Create `phases/phase10_distributed_gnn_frameworks/src/phase10_distributed_gnn_frameworks/dgl_benchmark.py` with `DGLBenchmark()` class stub; verify Python syntax
- [ ] 3.3 Implement `DGLBenchmark.setup(graph_data, model)` initializing DGL Distributed with graph partitioning; verify setup completes without errors
- [ ] 3.4 Implement `DGLBenchmark.run_inference(target_nodes)` running distributed inference; verify predictions are collected
- [ ] 3.5 Implement `DGLBenchmark.measure_performance()` measuring throughput, memory, scalability; verify metrics are valid numbers
- [ ] 3.6 Run DGL benchmark on 1M, 10M, 100M graphs; verify benchmark completes and results are saved to `outputs/dgl_results.json`

## 4. AliGraph benchmark

- [ ] 4.1 Create `phases/phase10_distributed_gnn_frameworks/benchmarks/aligraph_config.yaml` with AliGraph configuration; verify file exists
- [ ] 4.2 Create `phases/phase10_distributed_gnn_frameworks/src/phase10_distributed_gnn_frameworks/aligraph_benchmark.py` with `AliGraphBenchmark()` class stub; verify Python syntax
- [ ] 4.3 Implement `AliGraphBenchmark.setup(graph_data, model)` initializing AliGraph; verify setup completes or document if AliGraph unavailable
- [ ] 4.4 Implement `AliGraphBenchmark.run_inference(target_nodes)` running distributed inference; verify predictions are collected or document limitation
- [ ] 4.5 Implement `AliGraphBenchmark.measure_performance()` measuring throughput, memory, fraud-specific metrics; verify metrics are valid or document limitation
- [ ] 4.6 Run AliGraph benchmark on 1M, 10M, 100M graphs; verify benchmark completes or document access limitations

## 5. Quiver benchmark

- [ ] 5.1 Create `phases/phase10_distributed_gnn_frameworks/benchmarks/quiver_config.yaml` with Quiver configuration (gpu_ids, sampling_mode); verify file exists
- [ ] 5.2 Create `phases/phase10_distributed_gnn_frameworks/src/phase10_distributed_gnn_frameworks/quiver_benchmark.py` with `QuiverBenchmark()` class stub; verify Python syntax
- [ ] 5.3 Implement `QuiverBenchmark.setup(graph_data, model, gpu_available)` initializing Quiver with GPU/CPU mode; verify setup completes
- [ ] 5.4 Implement `QuiverBenchmark.run_inference(target_nodes)` running GPU-accelerated inference; verify predictions are collected
- [ ] 5.5 Implement `QuiverBenchmark.measure_performance()` measuring GPU memory, sampling throughput, inference speedup; verify metrics are valid
- [ ] 5.6 Run Quiver benchmark on 1M, 10M, 100M graphs with GPU (if available) and CPU-only; verify benchmark completes and results are saved

## 6. Unified benchmark harness

- [ ] 6.1 Create `phases/phase10_distributed_gnn_frameworks/src/phase10_distributed_gnn_frameworks/comparison_harness.py` with `ComparisonHarness()` class stub; verify Python syntax
- [ ] 6.2 Implement `ComparisonHarness.load_graph_data(graph_size)` loading identical graph data for all frameworks; verify data is consistent
- [ ] 6.3 Implement `ComparisonHarness.load_model(model_path)` loading same GCN model for all frameworks; verify model is loaded correctly
- [ ] 6.4 Implement `ComparisonHarness.run_all_frameworks(graph_data, model)` running benchmarks for DGL, AliGraph, Quiver; verify all results are collected
- [ ] 6.5 Implement `ComparisonHarness.compare_results()` comparing throughput, memory, scalability, integration complexity; verify comparison table is generated
- [ ] 6.6 Implement `ComparisonHarness.export_results(output_path)` saving results to JSON/CSV; verify output file exists

## 7. Recommendation report

- [ ] 7.1 Create `phases/phase10_distributed_gnn_frameworks/src/phase10_distributed_gnn_frameworks/recommendation.py` with `RecommendationGenerator()` class stub; verify Python syntax
- [ ] 7.2 Implement `RecommendationGenerator.analyze_results(benchmark_results)` analyzing performance data; verify analysis is complete
- [ ] 7.3 Implement `RecommendationGenerator.generate_recommendation(target_scenario)` generating recommendation for specific scenario (300M graph, CPU-only, GPU-available); verify recommendation includes justification
- [ ] 7.4 Implement `RecommendationGenerator.generate_report()` generating full report with decision criteria, trade-offs, migration steps; verify report is human-readable
- [ ] 7.5 Save recommendation report to `outputs/framework_recommendation.md`; verify file exists

## 8. Unit tests

- [ ] 8.1 Create `phases/phase10_distributed_gnn_frameworks/tests/test_dgl_benchmark.py` with tests for DGL benchmark setup, inference, performance measurement; verify with pytest
- [ ] 8.2 Create `phases/phase10_distributed_gnn_frameworks/tests/test_aligraph_benchmark.py` with tests for AliGraph benchmark or document if unavailable; verify with pytest
- [ ] 8.3 Create `phases/phase10_distributed_gnn_frameworks/tests/test_quiver_benchmark.py` with tests for Quiver benchmark setup, inference, GPU/CPU modes; verify with pytest
- [ ] 8.4 Create `phases/phase10_distributed_gnn_frameworks/tests/test_comparison_harness.py` with tests for unified harness data loading, model loading, result comparison; verify with pytest
- [ ] 8.5 Create `phases/phase10_distributed_gnn_frameworks/tests/test_recommendation.py` with tests for recommendation generation; verify with pytest

## 9. Educational content — lessons

- [ ] 9.1 Create `phases/phase10_distributed_gnn_frameworks/lessons/01-dgl-distributed.md` following 5-part structure (Explain, Code, Visualize, Practice, Solution) covering DGL Distributed architecture, setup, and usage; verify file exists and contains all 5 sections
- [ ] 9.2 Create `phases/phase10_distributed_gnn_frameworks/lessons/02-aligraph.md` following 5-part structure covering AliGraph for fraud detection; verify file exists and contains all 5 sections
- [ ] 9.3 Create `phases/phase10_distributed_gnn_frameworks/lessons/03-quiver.md` following 5-part structure covering Quiver GPU-accelerated sampling; verify file exists and contains all 5 sections
- [ ] 9.4 Create `phases/phase10_distributed_gnn_frameworks/lessons/04-benchmark-methodology.md` following 5-part structure covering how to fairly compare distributed frameworks; verify file exists and contains all 5 sections
- [ ] 9.5 Create `phases/phase10_distributed_gnn_frameworks/lessons/05-decision-framework.md` following 5-part structure covering how to choose the right framework for your use case; verify file exists and contains all 5 sections

## 10. Educational content — notebooks

- [ ] 10.1 Create `phases/phase10_distributed_gnn_frameworks/notebooks/01-framework-comparison.ipynb` with cells that run benchmarks for all frameworks, visualize throughput/memory/scalability, and compare results; verify notebook executes without errors
- [ ] 10.2 Create `phases/phase10_distributed_gnn_frameworks/notebooks/02-recommendation-explorer.ipynb` with cells that explore different scenarios (graph size, GPU availability, model type) and see framework recommendations; verify notebook executes without errors

## 11. End-to-end comparison

- [ ] 11.1 Create `phases/phase10_distributed_gnn_frameworks/run_framework_comparison.py` that runs all benchmarks, compares results, and generates recommendation report; verify script runs without errors
- [ ] 11.2 Print summary: framework comparison table, best framework for each scenario, recommendation for 300M graph; verify output is human-readable
- [ ] 11.3 Save benchmark results to `outputs/framework_comparison.json`; verify file exists and is valid JSON
- [ ] 11.4 Run full test suite `pytest phases/phase10_distributed_gnn_frameworks/tests/ -v` and verify all tests pass; verify exit code 0
- [ ] 11.5 Run `phases/phase10_distributed_gnn_frameworks/run_framework_comparison.py` end-to-end with synthetic 1M-node graph and verify: all frameworks benchmarked, comparison generated, recommendation produced; confirm exit code 0
