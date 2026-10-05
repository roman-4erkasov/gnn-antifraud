## Purpose

Explain model predictions using XAI techniques, validate explanations against chargeback labels on real data, and generate investigation leads for fraud analysts.

## ADDED Requirements

### Requirement: GNNExplainer node-level explanation

The system SHALL integrate GNNExplainer (PyTorch Geometric) to identify the subgraph structure and node features most responsible for each GCN prediction.

#### Scenario: Generate node explanation for GCN

- **WHEN** a GCN model predicts a node as fraudulent
- **THEN** GNNExplainer produces a mask over the local neighborhood edges and a mask over node features that contributed most to that prediction

#### Scenario: Export explanation as machine-readable format

- **WHEN** explanations are generated
- **THEN** the system exports them as a JSON object containing: subgraph node indices, edge indices, edge importance scores (0-1 normalized), and feature importance scores (0-1 normalized)

### Requirement: PGExplainer edge-level explanation

The system SHALL integrate PGExplainer for learning a global edge-prediction model that explains edge-level predictions (e.g., whether a transaction edge is fraudulent).

#### Scenario: Train PGExplainer model

- **WHEN** edge-level explanations are requested for GCN predictions
- **THEN** PGExplainer is trained on a subset of edges from the training graph to produce an explanation model that predicts edge importance from node embeddings

#### Scenario: PGExplainer inference on unseen edges

- **WHEN** PGExplainer inference runs on an unseen transaction edge
- **THEN** it produces: (1) an edge importance score, and (2) an explanation vector over node features contributing to the prediction

### Requirement: GAT attention weight extraction

The system SHALL extract and analyze attention weights from GAT models as an alternative, self-explanatory mechanism.

#### Scenario: Extract multi-head attention weights

- **WHEN** a GAT model is trained and a prediction is made
- **THEN** the system extracts multi-head attention weights for each edge and aggregates them across heads to produce an edge-level importance score

#### Scenario: Aggregate attention weights across heads

- **WHEN** attention weights are extracted from multiple GAT heads
- **THEN** the system produces a weighted average across heads (weighted by mean attention score) and a per-head importance ranking

### Requirement: Explanation validation against chargeback labels

The system SHALL validate XAI explanations on IEEE-CIS data by checking alignment with chargeback-labeled transactions.

#### Scenario: Check merchant-customer relationship alignment

- **WHEN** XAI explanations are generated for an IEEE-CIS chargeback-labeled transaction
- **THEN** the system checks whether the top-importance edges connect the transaction node to the merchant node or buyer node (binary match indicator)

#### Scenario: Check community membership alignment

- **WHEN** XAI explanations highlight nodes within a specific community
- **THEN** the system reports the fraction of highlighted nodes that belong to the same Louvain community as the target node

### Requirement: Investigation lead generation

The system SHALL produce human-readable investigation leads from XAI outputs, structured as actionable items for fraud analysts.

#### Scenario: Generate lead from GNNExplainer subgraph

- **WHEN** GNNExplainer identifies a high-importance subgraph for a flagged transaction
- **THEN** the system generates a lead containing: (1) key nodes/edges identified, (2) their inferred roles (merchant, buyer, intermediary), (3) their risk scores, and (4) recommended investigation steps (e.g., "review merchant transaction history", "check for shared device fingerprints")

#### Scenario: Generate lead from PGExplainer edge score

- **WHEN** PGExplainer identifies a transaction edge with high importance and high fraud score
- **THEN** the system generates a lead describing the edge attributes, the contributing features, and whether the edge connects to a previously flagged entity

### Requirement: Cross-method explanation comparison

The system SHALL compare explanations from different methods (GNNExplainer vs PGExplainer vs GAT attention) to assess consistency.

#### Scenario: Jaccard similarity of top-k edges

- **WHEN** explanations from GNNExplainer and GAT attention are both available
- **THEN** the system computes Jaccard similarity between the top-k edges from each method (for k ∈ {5, 10, 20}) and reports the results

#### Scenario: Consistency scoring across all three methods

- **WHEN** GNNExplainer, PGExplainer, and GAT attention all produce explanations for the same node/edge
- **THEN** the system produces a consistency score (fraction of methods agreeing on top-5 edges) and flags cases where methods disagree significantly (consistency < 0.33)
