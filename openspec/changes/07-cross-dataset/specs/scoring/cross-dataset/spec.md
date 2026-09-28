## Purpose

Implement cross-dataset model comparison framework with statistical rigor: bootstrapping-based confidence intervals, significance testing, transfer learning evaluation, and feature compatibility analysis across all datasets (IEEE-CIS, Pay-At-Pump, Sungkyunkwan, Naver Plus Bank).

## ADDED Requirements

### Requirement: Cross-dataset model comparison framework

The system SHALL implement a framework to train models on one dataset and evaluate on others, measuring transfer quality across all dataset combinations.

#### Scenario: Train model on IEEE-CIS, test on Pay-At-Pump

- **WHEN** a model is trained on IEEE-CIS dataset and evaluated on Pay-At-Pump
- **THEN** the system reports PR-AUC, Brier score, and inference latency, noting the cross-dataset transfer context

#### Scenario: Train model on Sungkyunkwan, test on Naver Plus Bank

- **WHEN** a model is trained on Sungkyunkwan dataset and evaluated on Naver Plus Bank
- **THEN** the system reports PR-AUC, Brier score, and inference latency for the transfer evaluation

#### Scenario: Full cross-dataset evaluation matrix

- **WHEN** all models are evaluated across all dataset combinations
- **THEN** the system produces a matrix where rows are source datasets, columns are target datasets, and cells contain PR-AUC/Brier score for each model type

### Requirement: Bootstrapping-based confidence intervals

The system SHALL compute confidence intervals for all metrics using bootstrapping to provide statistical rigor.

#### Scenario: Compute PR-AUC confidence intervals

- **WHEN** the system computes bootstrapped confidence intervals for PR-AUC
- **THEN** it produces 95% confidence intervals via 1000 bootstrap resamples with stratified sampling per class

#### Scenario: Compute Brier score confidence intervals

- **WHEN** the system computes bootstrapped confidence intervals for Brier score
- **THEN** it produces 95% confidence intervals via 1000 bootstrap resamples

#### Scenario: Confidence intervals for all metrics

- **WHEN** the system computes confidence intervals for a set of metrics (PR-AUC, Brier score, log-loss, Precision@K)
- **THEN** it produces a dict with metric name → (lower_bound, upper_bound, estimate) tuples

### Requirement: Statistical significance testing

The system SHALL perform statistical tests to determine whether differences between model performances are statistically significant.

#### Scenario: Wilcoxon signed-rank test on PR-AUC

- **WHEN** comparing PR-AUC scores between two models across bootstrap samples
- **THEN** the system reports a p-value from Wilcoxon signed-rank test and a boolean indicating significance at alpha=0.05

#### Scenario: Mann-Whitney U test on PR-AUC

- **WHEN** comparing PR-AUC scores between two independent models across bootstrap samples
- **THEN** the system reports a p-value from Mann-Whitney U test and a boolean indicating significance at alpha=0.05

#### Scenario: Significance report

- **WHEN** the system performs significance tests across all model pairs
- **THEN** it produces a significance matrix (model_A × model_B → p-value, significant flag)

### Requirement: Transfer learning evaluation

The system SHALL evaluate transfer learning approaches: fine-tuning models trained on one dataset for a target dataset.

#### Scenario: Fine-tune pretrained GCN

- **WHEN** a GCN model pretrained on dataset A is fine-tuned on dataset B with limited labeled data
- **THEN** the system reports the improvement over training from scratch and compares against the pretrained-on-B baseline

#### Scenario: Fine-tune pretrained LightGBM

- **WHEN** a LightGBM model trained on dataset A features is adapted to dataset B
- **THEN** the system reports transfer quality and identifies which shared features contribute most

### Requirement: Cross-dataset feature compatibility analysis

The system SHALL analyze which features are portable across datasets and which are dataset-specific.

#### Scenario: Feature overlap analysis

- **WHEN** the system compares feature schemas across datasets
- **THEN** it reports shared features, dataset-specific features, and recommended feature transformations

#### Scenario: Feature importance cross-dataset comparison

- **WHEN** the system computes feature importance for a model trained on dataset A and applies it to dataset B
- **THEN** it reports whether top features remain important in the target domain

### Requirement: Model-agnostic evaluation report

The system SHALL produce a structured evaluation report comparing all models across all datasets.

#### Scenario: Generate cross-dataset report

- **WHEN** all cross-dataset evaluations are complete
- **THEN** the system produces a report with: per-dataset model rankings, transfer quality summary, statistical significance flags, and overall best model

#### Scenario: Export report as JSON

- **WHEN** the system exports the cross-dataset report
- **THEN** it writes a structured JSON file with all metrics, confidence intervals, and significance results

## ADDED Requirements

### Requirement: Static model cross-dataset evaluation

The system SHALL evaluate static models (LightGBM, GCN, GAT) across all dataset combinations as a baseline for transfer learning.

#### Scenario: Static model transfer evaluation

- **WHEN** static models are evaluated across all dataset pairs
- **THEN** the system reports baseline transfer quality that transfer learning should improve upon

### Requirement: Temporal model cross-dataset evaluation

The system SHALL evaluate temporal models (TGN, GRN, rolling-GCN) across temporal dataset pairs.

#### Scenario: Temporal model transfer evaluation

- **WHEN** temporal models are evaluated on transfers between temporal datasets
- **THEN** the system reports whether temporal models generalize better than static models
