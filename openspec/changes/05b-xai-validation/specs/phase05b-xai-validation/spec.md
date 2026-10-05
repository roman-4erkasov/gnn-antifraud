## Purpose

Validate XAI explanations against synthetic fraud patterns with known ground-truth structure to measure explanation quality and determine trustworthiness for fraud investigation.

## ADDED Requirements

### Requirement: XAI validation against synthetic patterns

The system SHALL validate GNNExplainer, PGExplainer, and GAT attention explanations against synthetic fraud patterns (Mitme, Cascade, Money Mule) from Phase 05 with known ground-truth structure.

#### Scenario: Validate against Mitme pattern

- **WHEN** XAI explanations are generated for a node in a synthetic Mitme fraud pattern (cluster of colluding accounts)
- **THEN** the system measures precision and recall of XAI-identified edges/nodes against the known colluding subgraph (top-k edges compared to ground truth community)

#### Scenario: Validate against Cascade pattern

- **WHEN** XAI explanations are generated for a synthetic Cascade pattern (sequential fund transfers)
- **THEN** the system measures the extent to which high-importance edges form a path-like structure matching the known cascade chain

#### Scenario: Validate against Money Mule pattern

- **WHEN** XAI explanations are generated for a synthetic Money Mule pattern (intermediary accounts)
- **THEN** the system measures whether XAI correctly identifies the intermediary nodes and their connecting edges

### Requirement: Precision/recall metrics for XAI explanations

The system SHALL compute precision, recall, and F1-score for XAI-identified subgraphs against ground-truth fraud structure.

#### Scenario: Compute precision at top-k threshold

- **WHEN** XAI produces edge importance scores for a pattern graph
- **THEN** the system computes precision as the fraction of top-k highlighted edges that are in the ground-truth fraud subgraph (for k ∈ {5, 10, 20})

#### Scenario: Compute recall at top-k threshold

- **WHEN** XAI produces edge importance scores for a pattern graph
- **THEN** the system computes recall as the fraction of ground-truth fraud edges that appear in the top-k highlighted edges

#### Scenario: Compute F1-score

- **WHEN** precision and recall are computed
- **THEN** the system computes F1-score as the harmonic mean of precision and recall

### Requirement: Cross-pattern XAI performance comparison

The system SHALL compare XAI performance across different pattern types to identify which patterns are harder to explain.

#### Scenario: Compare XAI across pattern types

- **WHEN** XAI validation is run on Mitme, Cascade, and Money Mule patterns
- **THEN** the system produces a comparison report showing precision, recall, and F1-score for each pattern type

#### Scenario: Identify best pattern for XAI

- **WHEN** cross-pattern comparison is complete
- **THEN** the system identifies the pattern type with highest F1-score and reports which XAI method performs best on that pattern

### Requirement: Stop decision for XAI trustworthiness

The system SHALL generate a stop decision based on XAI precision on synthetic patterns.

#### Scenario: XAI precision below threshold

- **WHEN** XAI precision < 0.20 on all synthetic patterns
- **THEN** the system prints a warning: "XAI explanations are not reliable for investigation — consider using different XAI parameters or methods"

#### Scenario: XAI precision above threshold

- **WHEN** XAI precision ≥ 0.20 on at least one synthetic pattern
- **THEN** the system reports that XAI explanations are trustworthy for investigation and provides pattern-specific guidance

### Requirement: Validation on detected pattern instances

The system SHALL validate XAI explanations on pattern instances detected by the graph classifiers (not only on synthetic ground-truth samples), verifying that explanations identify correct structural roles.

#### Scenario: Validate XAI on detected Mitme

- **WHEN** GNNExplainer or PGExplainer explains a detected Mitme pattern instance
- **THEN** the system measures whether highlighted edges correspond to the known colluding pairs (circular bidirectional transactions)

#### Scenario: Validate XAI on detected Cascade

- **WHEN** GNNExplainer explains a detected Cascade pattern instance
- **THEN** the system measures whether highlighted nodes include the central hub node of the star topology

#### Scenario: Validate XAI on detected Money Mule

- **WHEN** PGExplainer or GNNExplainer explains a detected Money Mule pattern instance
- **THEN** the system measures whether highlighted edges correspond to chain edges in the sequential money flow
