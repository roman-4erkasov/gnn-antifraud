## Purpose

Research framework for empirical evaluation of GNN effectiveness, structural encoding, graph sampling, and explainable AI on anti-fraud datasets with parallel PyTorch Geometric (PyG) and DGL implementations.

## ADDED Requirements

### Requirement: Dataset Support
The system SHALL support loading and preprocessing of the following anti-fraud datasets:
- IEEE-CIS Credit Card Fraud Detection dataset (credit card fraud)
- EEV Mitme-Fraud dataset (coordinated fraud rings)
- Synthetic data generator with configurable fraud patterns and known ground truth

### Requirement: No-Graph Baseline Comparison
The system SHALL provide a no-graph baseline (LightGBM / XGBoost on tabular features only) and automatically compare its results against graph-based models with identical train/validation/test splits.

### Requirement: Graph Model Implementations
The system SHALL implement each graph model architecture in both PyG and DGL with equivalent hyperparameters:
- GCN (Graph Convolutional Network)
- GAT (Graph Attention Network)
- TGN (Temporal Graph Network) for temporal datasets

### Requirement: Structural Encoding
The system SHALL support adding structural encodings to node features without modifying message passing layers:
- ReCWE (Relative Connected Walk Encoding)
- RWSE (Random Walk Structural Encoding)
- Degree, betweenness, and clustering coefficient as fallback

### Requirement: Sampler Benchmarking
The system SHALL support benchmarking the following graph sampling strategies with comparable settings:
- Neighbor Sampling (random and multilayer)
- GraphSAINT (uniform, BFS, RW, degree variants)
- GraphBolt (PyG large-scale indexing)

For each sampler the system SHALL report: samples-per-second, memory footprint, and model accuracy impact.

### Requirement: Evaluation Metrics
The system SHALL compute and report the following metrics for every experiment:
- AUC-ROC
- PR-AUC (Precision-Recall AUC)
- Precision@K for K = {10, 50, 100}
- Training time per epoch
- Inference time per prediction batch

### Requirement: Explainable AI
The system SHALL provide explanation capabilities for graph model predictions:
- GNNExplainer: identify important edges, nodes, and features per prediction
- PGExplainer: trained explainer for batch-level explanations
- Attention weight extraction from GAT layers
- Synthetic data validation: explanations must align with known fraud patterns with at least 70% node/edge overlap

### Requirement: CPU-First Execution
All experiments SHALL execute on CPU by default with an optional GPU acceleration flag. The system SHALL validate that CPU and GPU results are numerically equivalent (max difference < 1e-4) when using the same model configuration.

### Requirement: Framework Comparison Report
The system SHALL generate a comparison report for every experiment run with both PyG and DGL, showing:
- Accuracy metrics side by side
- Training time and memory usage
- Explanation quality (GNNExplainer overlap with ground truth)
- Configuration used (hyperparameters, random seed, hardware)
