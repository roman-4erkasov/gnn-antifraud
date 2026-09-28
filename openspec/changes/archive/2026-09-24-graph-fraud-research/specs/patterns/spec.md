## Purpose

Detect and characterize specific fraud patterns (Mitme, Cascade, Money Mule) in graph-structured data, and validate detection quality using XAI outputs.

## ADDED Requirements

### Requirement: Mitme fraud pattern generation and detection

The system SHALL generate synthetic Mitme fraud patterns (colluding merchant-buyer pairs with circular transactions) and evaluate whether graph-based models can identify them.

#### Scenario: Synthetic Mitme generation

- **WHEN** the system generates Mitme synthetic data
- **THEN** it creates pairs of colluding nodes with transactions flowing in both directions at above-normal rates

#### Scenario: Mitme detection

- **WHEN** a graph classifier (GCN/GAT) is trained on data containing Mitme patterns
- **THEN** the system reports Recall and Precision@K for the Mitme class compared against the LightGBM baseline

### Requirement: Cascade fraud pattern generation and detection

The system SHALL generate synthetic Cascade fraud patterns (a central node distributing fraudulent transactions to multiple leaf nodes that then consolidate funds).

#### Scenario: Synthetic Cascade generation

- **WHEN** the system generates Cascade synthetic data
- **THEN** it creates a star-like subgraph with one center node distributing transactions to many leaves, followed by consolidated returns

#### Scenario: Cascade detection

- **WHEN** a graph classifier is trained on data containing Cascade patterns
- **THEN** the system reports Recall and Precision@K for the Cascade class and whether XAI correctly identifies the central hub node

### Requirement: Money mule detection

The system SHALL attempt to detect Money mule patterns (sequential chains of transactions passing through intermediate accounts) using graph structure and temporal features.

#### Scenario: Money mule pattern generation

- **WHEN** the system generates Money mule synthetic data
- **THEN** it creates chain-like subgraphs where funds flow sequentially through several intermediate nodes before reaching a final destination

#### Scenario: Money mule detection

- **WHEN** graph models process money mule chains
- **THEN** the system reports detection quality and whether PGExplainer/GNNExplainer can identify chain edges

### Requirement: Pattern template generation

The system SHALL produce investigation templates describing detected patterns in a structured format, including: pattern type, key structural features, recommended next steps, and confidence score.

#### Scenario: Generate pattern template

- **WHEN** a pattern (Mitme, Cascade, or Money Mule) is detected
- **THEN** the system outputs a template containing: pattern type, list of suspicious nodes/edges, structural signatures, and recommended investigation actions
