## Purpose

Evaluate multiple structural encoding techniques (ReCWE, RWSE, Spectral, Community) for GNN node classification and identify which encoding maximizes fraud detection performance.

## ADDED Requirements

### Requirement: ReCWE positional encoding

The system SHALL implement Relative Custom Walk Encoding (ReCWE) that computes walk-based structural features per node by performing random walks and encoding the relative positions of visited nodes.

#### Scenario: ReCWE computation

- **WHEN** a graph structure is provided
- **THEN** the system computes ReCWE features for each node by performing random walks (configurable number of walks and lengths) and encoding the frequency of visits to nodes at each relative distance

#### Scenario: ReCWE integration

- **WHEN** ReCWE features are computed
- **THEN** they are concatenated as additional node features to the existing node feature matrix

### Requirement: RWSE feature computation

The system SHALL implement Random Walk Structural Encoding (RWSE) that computes statistical features per node from random walk statistics.

#### Scenario: RWSE computation

- **WHEN** a graph structure is provided
- **THEN** the system computes RWSE features including: mean shortest path, visit frequency, degree-weighted centrality from random walks
- **THEN** RWSE features are concatenated as additional node features

#### Scenario: RWSE feature shape

- **WHEN** RWSE features are computed for a graph with N nodes
- **THEN** the output feature matrix has shape [N, n_rwse_features] where n_rwse_features is the configured number of RWSE dimensions (default: 8)

### Requirement: Spectral encoding

The system SHALL implement Spectral encoding using graph Laplacian eigenvectors as node features.

#### Scenario: Spectral computation

- **WHEN** a graph structure is provided
- **THEN** the system computes the normalized graph Laplacian L = I - D^(-1/2)AD^(-1/2) and extracts the k smallest eigenvectors (excluding the trivial first eigenvector)
- **THEN** the eigenvector values for each node are concatenated as spectral features

#### Scenario: Spectral truncation

- **WHEN** spectral encoding is computed with k eigenvectors and N nodes
- **THEN** the output feature matrix has shape [N, k]

### Requirement: Community detection encoding

The system SHALL implement community detection using Louvain or Label Propagation algorithms and add community labels as one-hot or ordinal node features.

#### Scenario: Community detection

- **WHEN** a graph structure is provided
- **THEN** the system runs Louvain community detection and assigns an integer community label to each node
- **THEN** community labels are one-hot encoded (or ordinal encoded) and concatenated as node features

#### Scenario: Community detection validation

- **WHEN** community detection runs on the bipartite graph
- **THEN** each node is assigned exactly one community label in range [0, n_communities)

### Requirement: GCN training with structural encodings

The system SHALL train separate GCN models, each using a different structural encoding technique, on the same fraud detection task.

#### Scenario: Per-encoding GCN training

- **WHEN** each structural encoding (ReCWE, RWSE, Spectral, Community) has been computed
- **THEN** the system trains a separate GCN model using each encoding's node features with identical training hyperparameters (same LR, epochs, optimizer settings)

#### Scenario: Identical training conditions

- **WHEN** comparing encodings
- **THEN** all GCN models use the same edge index, same train/val/test split, and the same hyperparameters — only the node feature matrix differs

### Requirement: Evaluation and comparison

The system SHALL evaluate each encoding's GCN model with all metrics and produce a comparison summary.

#### Scenario: Per-encoding metrics

- **WHEN** each GCN model has been trained
- **THEN** the system computes PR-AUC, ROC-AUC, Brier score, log-loss, and Precision@K for each encoding variant

#### Scenario: Summary table

- **WHEN** all encoding results are computed
- **THEN** the system produces a summary table: encoding × PR-AUC × ROC-AUC × Brier score × log-loss

### Requirement: SHAP-based feature importance comparison

The system SHALL compute SHAP values for graph-aware LightGBM with each encoding set and compare feature importance rankings.

#### Scenario: SHAP computation

- **WHEN** graph-aware LightGBM models are trained with each encoding
- **THEN** the system computes SHAP values and extracts per-feature importance (mean absolute SHAP value)

#### Scenario: Feature importance ranking

- **WHEN** SHAP importance scores are computed for each encoding
- **THEN** the system produces a ranking of features (structural encoding features vs. base features) for each encoding variant

### Requirement: Encoding effectiveness identification

The system SHALL identify which structural encoding achieves the best performance across metrics and recommend it.

#### Scenario: Best encoding selection

- **WHEN** all encoding metrics are available
- **THEN** the system selects the encoding with the highest PR-AUC and reports it as the best encoding, with ties broken by lowest Brier score
