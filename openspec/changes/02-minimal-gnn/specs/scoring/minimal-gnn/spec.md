## Purpose

Implement a single-layer GCN classifier for node-level fraud detection and compare it against LightGBM baselines to determine whether graph neural networks add predictive value.

## ADDED Requirements

### Requirement: Single-layer GCN model

The system SHALL implement a single-layer GCN classifier using PyTorch Geometric's `GCNConv` with a classification head for binary fraud detection on graph nodes.

#### Scenario: GCN model construction

- **WHEN** the system is given node features and an edge index (bipartite graph structure)
- **THEN** it constructs a single-layer GCN model with GCNConv(input_dim, hidden_dim) → ReLU → GCNConv(hidden_dim, 1) → Sigmoid

#### Scenario: GCN forward pass

- **WHEN** the GCN model receives node features and edge index
- **THEN** it returns probability scores for each node through graph convolution and classification head

### Requirement: GCN training loop

The system SHALL train the GCN model using binary cross-entropy loss with standard training loop (forward → loss → backward → optimizer step).

#### Scenario: GCN training

- **WHEN** the system is initialized with training node labels, features, and edge index
- **THEN** it performs gradient descent using Adam optimizer with configurable learning rate and number of epochs, reporting training loss per epoch

#### Scenario: Training convergence

- **WHEN** GCN training runs for the specified number of epochs
- **THEN** the training loss decreases over epochs (monotonically or with overall downward trend)

### Requirement: GCN calibration

The system SHALL apply isotonic regression calibration to GCN probability outputs using the existing `IsotonicCalibrator`.

#### Scenario: GCN calibration

- **WHEN** the GCN produces raw probability outputs on a validation set
- **THEN** isotonic regression is fitted on validation predictions and applied to transform raw scores into calibrated probabilities

#### Scenario: Calibration delta

- **WHEN** GCN calibration is applied
- **THEN** the system reports the change in Brier score and log-loss before and after calibration

### Requirement: GCN evaluation

The system SHALL evaluate the GCN model with all standard classification metrics on a held-out test set.

#### Scenario: GCN metrics computation

- **WHEN** the GCN model produces predictions on the test set
- **THEN** the system computes and reports PR-AUC, ROC-AUC, Brier score, log-loss, and Precision@K for both raw and calibrated predictions

### Requirement: Cross-model comparison

The system SHALL compare GCN performance against both LightGBM minimal and graph-aware baselines using delta metrics and statistical significance testing.

#### Scenario: GCN vs LightGBM comparison

- **WHEN** GCN, minimal LightGBM, and graph-aware LightGBM are all evaluated on the same test set
- **THEN** the system reports delta PR-AUC, delta Brier score, and delta Precision@K between GCN and each LightGBM variant

#### Scenario: Statistical significance

- **WHEN** GCN and LightGBM predictions are compared on the same test set
- **THEN** the system performs a paired permutation test and reports the p-value for each model pair

### Requirement: Stop-early decision for GCN

The system SHALL flag Phase 2 as "GNN adds minimal value" if the GCN ΔPR-AUC vs graph-aware LightGBM is less than 0.01 or not statistically significant (p-value > 0.05).

#### Scenario: GNN minimal value detected

- **WHEN** ΔPR-AUC(GCN, graph-aware-LightGBM) < 0.01 or paired permutation test p-value > 0.05
- **THEN** the system flags Phase 2 as "GNN adds minimal value" and outputs a stop decision recommendation
