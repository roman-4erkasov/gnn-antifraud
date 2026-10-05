## Purpose

Implement temporal graph models (TGN, GRN, rolling-window GCN) for fraud detection on time-stamped transaction graphs, validate on temporal anomaly datasets (Pay-At-Pump, Sungkyunkwan, Naver Plus Bank), and compare against static baselines.

## ADDED Requirements

### Requirement: TGN (Temporal Graph Network) implementation

The system SHALL implement a Temporal Graph Network with node memory and relation encoder for processing time-stamped edge events.

#### Scenario: TGN processes time-stamped events

- **WHEN** the system receives time-stamped edge events (src, dst, timestamp, features)
- **THEN** TGN updates node memory states based on received messages, applies relation encoding, and produces node embeddings that incorporate temporal dynamics

#### Scenario: TGN generates predictions

- **WHEN** TGN has processed edge events up to a specific time
- **THEN** the system produces fraud prediction scores for nodes that have received events after that time

### Requirement: GRN (Graph Recurrent Network) implementation

The system SHALL implement a Graph Recurrent Network that processes temporal graphs as sequences of snapshots.

#### Scenario: GRN processes snapshot sequences

- **WHEN** the system provides a sequence of static graph snapshots (one per time window)
- **THEN** GRN processes each snapshot with a GNN encoder and passes hidden states between consecutive snapshots using an RNN layer (LSTM or GRU)

#### Scenario: GRN generates predictions

- **WHEN** GRN has processed a sequence of snapshots through its RNN
- **THEN** the system produces fraud prediction scores for nodes in the latest snapshot based on temporal patterns captured by the RNN

### Requirement: Rolling-window GCN implementation

The system SHALL implement a rolling-window GCN that splits a temporal graph into fixed time windows, builds static snapshots, and trains a GCN per window.

#### Scenario: Build temporal snapshots

- **WHEN** the system receives a temporal graph with time-stamped edges
- **THEN** it splits edges into K consecutive time windows (e.g., K=10 equal-sized windows), builds a static graph for each window, and preserves temporal ordering

#### Scenario: Train and evaluate rolling-window GCN

- **WHEN** the system trains a GCN model on consecutive temporal snapshots
- **THEN** it produces node-level predictions for the latest snapshot using the GCN trained on that or adjacent snapshots, and reports PR-AUC, Brier score

### Requirement: Temporal leakage detection

The system SHALL detect temporal leakage where edges from the test time period leak into the training graph.

#### Scenario: Check for temporal leakage

- **WHEN** the system receives train/test edge splits with timestamps
- **THEN** it reports whether any test-period edges occurred before the latest training edge and provides the count of leaking edges

#### Scenario: Verify no leakage after mitigation

- **WHEN** the system applies temporal leakage mitigation (e.g., strict time-based split)
- **THEN** it confirms that no test-period edges appear in the training graph (n_leaking_edges == 0)

### Requirement: Temporal dataset loaders

The system SHALL load and preprocess temporal anomaly datasets: Pay-At-Pump, Sungkyunkwan, and Naver Plus Bank.

#### Scenario: Load Pay-At-Pump temporal data

- **WHEN** the system loads the Pay-At-Pump dataset
- **THEN** it returns time-stamped edge events, node metadata (if available), anomaly labels, and a temporal train/test split

#### Scenario: Load Sungkyunkwan temporal data

- **WHEN** the system loads the Sungkyunkwan temporal dataset
- **THEN** it returns a sequence of temporal snapshots with bipartite user-merchant structure and anomaly labels per snapshot

#### Scenario: Load Naver Plus Bank temporal data

- **WHEN** the system loads the Naver Plus Bank temporal dataset
- **THEN** it returns time-stamped financial transactions with anomaly labels and appropriate temporal split

### Requirement: Temporal model comparison

The system SHALL compare temporal models (TGN, GRN, rolling-GCN) against static models (GCN, LightGBM) on the same temporal datasets.

#### Scenario: Compare models on Pay-At-Pump

- **WHEN** the system evaluates all models on Pay-At-Pump
- **THEN** it reports PR-AUC, Brier score, and inference latency for each model (TGN, GRN, rolling-GCN, static-GCN, LightGBM) and identifies the best-performing model

#### Scenario: Compare models on Sungkyunkwan

- **WHEN** the system evaluates all models on Sungkyunkwan
- **THEN** it reports PR-AUC, Brier score, and inference latency for each model and identifies the best-performing model

#### Scenario: Compare models on Naver Plus Bank

- **WHEN** the system evaluates all models on Naver Plus Bank
- **THEN** it reports PR-AUC, Brier score, and inference latency for each model and identifies the best-performing model

## ADDED Requirements

### Requirement: Static GCN evaluation on temporal data

The system SHALL evaluate static GCN models on temporal snapshots as an additional comparison point against temporal models.

#### Scenario: Static GCN on temporal snapshot

- **WHEN** the system evaluates static GCN on a temporal dataset
- **THEN** it treats each temporal snapshot as a separate graph, trains GCN on the training snapshot(s), and evaluates on the test snapshot(s)

### Requirement: Temporal-aware LightGBM baseline

The system SHALL include a temporal-aware LightGBM baseline that uses features computed within rolling windows.

#### Scenario: Compute temporal LightGBM features

- **WHEN** the system prepares features for LightGBM on a temporal dataset
- **THEN** it computes rolling-window statistics (e.g., 7-day transaction count, 30-day average amount) in addition to static features

### Requirement: XAI validation on temporal predictions

The system SHALL validate XAI explanations on time-stamped predictions to assess whether explanations capture temporal patterns.

#### Scenario: XAI on temporal prediction

- **WHEN** GNNExplainer or PGExplainer explains a prediction on a temporal snapshot
- **THEN** the system reports whether highlighted edges/nodes are concentrated in recent time periods or are distributed across the temporal window
