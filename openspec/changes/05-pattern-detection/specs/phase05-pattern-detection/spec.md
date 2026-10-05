## Purpose

Generate synthetic fraud patterns (Mitme, Cascade, Money Mule), train graph classifiers to detect them, and produce structured investigation templates for pattern-level detection.

## ADDED Requirements

### Requirement: Synthetic Mitme fraud pattern generation

The system SHALL generate synthetic Mitme fraud patterns consisting of colluding merchant-buyer pairs with circular transactions flowing in both directions.

#### Scenario: Generate single Mitme pattern

- **WHEN** the system generates a Mitme pattern sample
- **THEN** it creates two or more node pairs connected by transactions in both directions at above-normal rates, with node features consistent with the IEEE-CIS feature schema

#### Scenario: Generate Mitme dataset

- **WHEN** the system generates a Mitme dataset for training
- **THEN** it produces 1000–5000 labeled graph samples, each with known colluding pairs and corresponding ground-truth labels

### Requirement: Synthetic Cascade fraud pattern generation

The system SHALL generate synthetic Cascade fraud patterns consisting of a central hub node distributing fraudulent transactions to multiple leaf nodes that then consolidate funds.

#### Scenario: Generate single Cascade pattern

- **WHEN** the system generates a Cascade pattern sample
- **THEN** it creates a star-like subgraph with one center node connected to N leaf nodes (N ≥ 3) with outgoing transactions, followed by consolidated return transactions to a final destination node

#### Scenario: Generate Cascade dataset

- **WHEN** the system generates a Cascade dataset for training
- **THEN** it produces 1000–5000 labeled graph samples with varying fan-out degrees and ground-truth labels identifying the hub and leaf nodes

### Requirement: Synthetic Money Mule pattern generation

The system SHALL generate synthetic Money Mule patterns consisting of sequential chain-like transactions flowing through intermediate accounts.

#### Scenario: Generate single Money Mule pattern

- **WHEN** the system generates a Money Mule pattern sample
- **THEN** it creates a chain subgraph where funds flow sequentially from a source node through K ≥ 2 intermediate nodes to a final destination node, with each edge labeled as part of the mule chain

#### Scenario: Generate Money Mule dataset

- **WHEN** the system generates a Money Mule dataset for training
- **THEN** it produces 1000–5000 labeled graph samples with varying chain lengths and ground-truth labels identifying chain edges and nodes

### Requirement: Graph-based pattern classification

The system SHALL train graph classifiers (GCN and GAT) on each synthetic pattern type and evaluate detection quality.

#### Scenario: Train graph classifier on Mitme patterns

- **WHEN** a GCN or GAT model is trained on Mitme synthetic data
- **THEN** the system reports per-class Recall and Precision@K for the Mitme class, compared against the LightGBM tabular baseline on the same data

#### Scenario: Train graph classifier on Cascade patterns

- **WHEN** a GCN or GAT model is trained on Cascade synthetic data
- **THEN** the system reports per-class Recall and Precision@K for the Cascade class

#### Scenario: Train graph classifier on Money Mule patterns

- **WHEN** a GCN or GAT model is trained on Money Mule synthetic data
- **THEN** the system reports per-class Recall and Precision@K for the Money Mule class

### Requirement: Pattern template generation

The system SHALL produce investigation templates for detected fraud patterns in a structured format.

#### Scenario: Generate Mitme pattern template

- **WHEN** a Mitme pattern is detected by a graph classifier
- **THEN** the system outputs a template containing: pattern type ("Mitme"), list of suspicious node pairs, structural signature (circular bidirectional edges), confidence score, and recommended investigation actions

#### Scenario: Generate Cascade pattern template

- **WHEN** a Cascade pattern is detected by a graph classifier
- **THEN** the system outputs a template containing: pattern type ("Cascade"), list of hub node and leaf nodes, structural signature (star topology with fund distribution), confidence score, and recommended investigation actions

#### Scenario: Generate Money Mule pattern template

- **WHEN** a Money Mule pattern is detected by a graph classifier
- **THEN** the system outputs a template containing: pattern type ("Money Mule"), list of chain nodes and edges, structural signature (sequential chain path), confidence score, and recommended investigation actions

### Requirement: Pattern evaluation in LightGBM baseline

The system SHALL report pattern-specific evaluation metrics (per-pattern Recall and Precision@K) when evaluating fraud detection models.

#### Scenario: Pattern-aware baseline evaluation

- **WHEN** the LightGBM baseline is evaluated on synthetic pattern data
- **THEN** the system reports per-pattern Recall and Precision@K (not only overall metrics) for each of Mitme, Cascade, and Money Mule

### Requirement: Pattern evaluation in GNN models

The system SHALL report pattern-specific detection quality when evaluating graph-based models.

#### Scenario: Pattern-aware GNN evaluation

- **WHEN** GCN or GAT models are evaluated on synthetic pattern data
- **THEN** the system reports per-pattern Recall and Precision@K for each fraud pattern type, enabling comparison across pattern types and models
