## Purpose

Классифицировать транзакции на мошеннические и легитимные, обеспечивая сравнение графовых и неграфовых подходов с обязательной калибровкой вероятностей.

## ADDED Requirements

### Requirement: LightGBM baseline with minimal features

The system SHALL implement a LightGBM baseline classifier trained on a feature matrix with no more than 3 features per transaction, providing a minimal-performance reference point.

#### Scenario: Baseline training with 3 features

- **WHEN** the system is initialized with the baseline configuration
- **THEN** it trains a LightGBM model using at most 3 features (e.g., transaction amount, time since account creation, number of prior transactions)

#### Scenario: Baseline evaluation

- **WHEN** baseline model is trained
- **THEN** the system evaluates and reports PR-AUC, ROC-AUC, Brier score, and Precision@K on a held-out test set

### Requirement: Graph-aware LightGBM

The system SHALL extend the LightGBM baseline by incorporating graph-derived features (node degrees, edge counts, community membership, structural encoding values), enabling a comparison between tabular and graph-aware approaches.

#### Scenario: Graph features injection

- **WHEN** graph-aware LightGBM is trained
- **THEN** features derived from the graph structure (e.g., node degree, RWSE values, ReCWE embeddings) are included in the feature matrix

#### Scenario: Tabular vs graph comparison

- **WHEN** both baselines are trained and evaluated
- **THEN** the system reports the delta in PR-AUC, Brier score, and Precision@K between tabular and graph-aware LightGBM

### Requirement: GCN classifier

The system SHALL implement a Graph Convolutional Network (GCN) using PyTorch Geometric that learns node representations directly from the graph structure and node features.

#### Scenario: GCN model construction

- **WHEN** the GCN model is initialized
- **THEN** it creates a single-layer GCN layer (PyG `GCNConv`) followed by a classification head

#### Scenario: GCN training and evaluation

- **WHEN** the GCN is trained on node classification task
- **THEN** it reports PR-AUC, ROC-AUC, Brier score, and compares against LightGBM baselines

### Requirement: Isotonic calibration

The system SHALL apply isotonic regression calibration to the probability outputs of every trained classifier, ensuring well-calibrated scores suitable for thresholding and business decisions.

#### Scenario: Calibration application

- **WHEN** a classifier produces probability outputs
- **THEN** isotonic regression is fit on a validation set and applied to transform raw scores into calibrated probabilities

#### Scenario: Calibration impact measurement

- **WHEN** calibration is applied
- **THEN** the system reports the change in Brier score and log-loss before and after calibration

### Requirement: Standard evaluation metrics

The system SHALL compute and report PR-AUC, ROC-AUC, Brier score, log-loss, Precision@K, and statistical significance tests (paired permutation test or McNemar's test) for all model comparisons.

#### Scenario: Full metric computation

- **WHEN** model predictions and ground truth labels are available
- **THEN** the system computes PR-AUC, ROC-AUC, Brier score, log-loss, and Precision@K

#### Scenario: Statistical significance

- **WHEN** two models are compared on the same test set
- **THEN** the system performs a statistical significance test (permutation test or McNemar's test) and reports the p-value
