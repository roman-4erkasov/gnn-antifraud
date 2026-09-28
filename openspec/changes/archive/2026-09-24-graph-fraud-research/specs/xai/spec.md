## Purpose

Explain model predictions using XAI techniques, validate explanations against known patterns, and generate investigation leads for fraud analysts.

## ADDED Requirements

### Requirement: GNNExplainer integration

The system SHALL integrate GNNExplainer (from PyG) to identify the subgraph structure and node features most responsible for each model prediction.

#### Scenario: Generate GNN explanation

- **WHEN** a GCN model predicts a node as fraudulent
- **THEN** GNNExplainer produces a mask over the local neighborhood and features that contributed most to that prediction

#### Scenario: Explanation visualization support

- **WHEN** explanations are generated
- **THEN** the system exports them in a machine-readable format (JSON) containing subgraph indices, edge weights, and feature importances

### Requirement: PGExplainer integration

The system SHALL integrate PGExplainer for learning a global edge-prediction model that can explain edge-level predictions (e.g., whether a transaction edge is fraudulent).

#### Scenario: PGExplainer training

- **WHEN** edge-level predictions are needed
- **THEN** PGExplainer is trained on a subset of edges to produce a generalizable explanation model

#### Scenario: PGExplainer inference

- **WHEN** PGExplainer inference runs on an unseen edge
- **THEN** it produces an explanation score for each input feature contributing to the prediction

### Requirement: Attention weight extraction

The system SHALL extract and analyze attention weights from GAT models as an alternative, self-explanatory mechanism.

#### Scenario: Extract attention weights

- **WHEN** a GAT model is trained and a prediction is made
- **THEN** the system extracts multi-head attention weights for each edge and aggregates them to produce an edge-level importance score

#### Scenario: Attention weight comparison

- **WHEN** attention weights are extracted from GAT and GNNExplainer masks from GCN
- **THEN** the system reports the overlap (e.g., Jaccard similarity of top-k edges) between the two explanation methods

### Requirement: Explanation validation against ground truth

The system SHALL validate XAI explanations by comparing them against known patterns in synthetic datasets and labeled chargeback data from IEEE-CIS.

#### Scenario: Synthetic pattern validation

- **WHEN** synthetic fraud patterns (e.g., Mitme, Cascade) with known graph structure are generated
- **THEN** the system measures the precision and recall of XAI-identified edges/nodes against the known fraud structure

#### Scenario: Chargeback label validation

- **WHEN** XAI explanations are generated for IEEE-CIS chargeback-labeled transactions
- **THEN** the system checks whether high-importance edges align with merchant-customer relationships or other known chargeback indicators

### Requirement: Investigation lead generation

The system SHALL produce human-readable investigation leads from XAI outputs, structured as actionable items for fraud analysts.

#### Scenario: Lead from subgraph explanation

- **WHEN** GNNExplainer identifies a high-importance subgraph for a flagged transaction
- **THEN** the system generates a lead describing: (1) the key nodes/edges, (2) their roles (merchant, buyer, intermediary), and (3) recommended investigation steps
