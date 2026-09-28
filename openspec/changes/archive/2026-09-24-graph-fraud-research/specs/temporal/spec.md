## Purpose

Model temporal graph dynamics using temporal GNN architectures (TGN, GRN) on temporal anomaly datasets where fraud labels may not be available, focusing on understanding whether temporal information adds value.

## ADDED Requirements

### Requirement: TGN model for temporal graphs

The system SHALL implement a Temporal Graph Network (TGN) that processes time-stamped edge events and maintains node memory over time for node-level prediction tasks.

#### Scenario: TGN training on temporal events

- **WHEN** time-stamped edge events are provided (e.g., from Pay-At-Pump dataset)
- **THEN** TGN processes events in temporal order, updates node memories, and produces node representations for classification

#### Scenario: TGN evaluation

- **WHEN** TGN produces predictions on a held-out temporal test set
- **THEN** the system reports PR-AUC, Brier score, and compares against non-temporal baselines (GCN, LightGBM) trained on static snapshots

### Requirement: GRN model for temporal graphs

The system SHALL implement a Graph Recurrent Network (GRN) that applies recurrent gating over graph structure changes over time.

#### Scenario: GRN training

- **WHEN** a sequence of graph snapshots is available (e.g., from Sungkyunkwan or Naver Plus Bank)
- **THEN** GRN processes each snapshot in sequence, maintaining hidden states that capture temporal evolution

#### Scenario: GRN evaluation

- **WHEN** GRN produces predictions
- **THEN** the system reports PR-AUC, Brier score, and compares against static GCN and TGN

### Requirement: Rolling-window GCN baseline

The system SHALL implement a rolling-window GCN that trains separate GCN models on fixed-length temporal windows and compares against TGN/GRN.

#### Scenario: Rolling window processing

- **WHEN** a temporal graph is provided
- **THEN** the system splits events into fixed-length windows, builds static graph snapshots, and trains a GCN per window

#### Scenario: Window GCN vs temporal models

- **WHEN** rolling-window GCN, TGN, and GRN predictions are available on the same test set
- **THEN** the system compares PR-AUC, Brier score, and inference latency across all three approaches

### Requirement: Temporal leakage check

The system SHALL verify that no information from the test period leaks into the training set through graph structure (e.g., future edges used during training).

#### Scenario: Temporal split validation

- **WHEN** the dataset is split by time
- **THEN** the system verifies that no test edges or node relationships exist in the training graph

#### Scenario: Leakage report

- **WHEN** temporal split validation is performed
- **THEN** the system reports a boolean flag indicating whether leakage was detected, along with the number of leaking edges
