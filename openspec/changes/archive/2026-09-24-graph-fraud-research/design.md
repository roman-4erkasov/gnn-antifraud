## Context

The project has no existing implementation (`src/` is empty). The goal is an empirical research study answering whether graph-based models (GNNs) provide value over tabular baselines (LightGBM) for fraud detection, with mandatory calibration, XAI, and rigorous cross-model/cross-dataset comparison. The study uses open-source datasets (IEEE-CIS, EEV Mitme-Fraud, Pay-At-Pump, Sungkyunkwan, Naver Plus Bank) and synthetic data generation.

## Goals / Non-Goals

**Goals:**
- Build an end-to-end research pipeline: data → LightGBM → GNN → calibration → XAI → comparison
- Establish whether graphs add predictive value at each stage (stop if no gain)
- Produce reproducible, well-calibrated, explainable models
- Validate XAI outputs against known patterns (synthetic + chargebacks)
- Understand temporal dynamics on graphs where fraud labels may not exist
- Design a partitioning strategy for 300M+ node graphs

**Non-Goals:**
- Production deployment of any model
- Dual implementation in both PyG and DGL (PyG only)
- Real-time inference optimization
- Building a user-facing product or dashboard

## Decisions

### Decision 1: PyG only, not DGL

**Choice:** Use PyTorch Geometric (PyG) as the sole GNN framework.

**Rationale:**
- PyG has more mature temporal graph support (TGN integration), better XAI tooling (GNNExplainer natively integrated), and broader community adoption.
- DGL is a valid alternative, but maintaining two implementations doubles effort for research purposes where the framework itself is not the question.
- GraphBolt is referenced for optimization concepts but not required for implementation.

**Alternatives considered:**
- DGL only: less mature temporal support, no native GNNExplainer.
- Both PyG + DGL: too expensive for research scope.

### Decision 2: Top-down approach (LightGBM → GNN)

**Choice:** Always start with a minimal LightGBM baseline before attempting any GNN.

**Rationale:**
- Fraud datasets are often imbalanced and noisy; a simple model establishes the floor.
- If LightGBM with graph features matches GNN performance, graphs are not worth the complexity.
- This is a "stop early" strategy: each phase must justify the previous one's investment.

### Decision 3: Mandatory calibration for all probabilistic outputs

**Choice:** Apply isotonic regression calibration to every classifier's probability outputs.

**Rationale:**
- Fraud detection is a business-critical domain where probability thresholds directly drive investigations.
- Uncalibrated models produce misleading confidence scores, making threshold tuning and business decisions unreliable.
- Brier score (proper scoring rule) is the primary calibration metric, alongside log-loss.

**Alternatives considered:**
- Platt scaling (logistic regression): assumes isotonic relationship, less flexible.
- Temperature scaling: simpler but less accurate for imbalanced data.
- No calibration: unacceptable for a research study claiming business relevance.

### Decision 4: XAI as first-class output, not an afterthought

**Choice:** Integrate GNNExplainer, PGExplainer, and Attention weight extraction from Phase 4 onward, with validation against known patterns.

**Rationale:**
- XAI is the primary bridge between model predictions and fraud investigators' workflows.
- Without validation (do explanations match ground truth?), XAI is just pretty pictures.
- Investigation leads (structured output) are the final deliverable of XAI.

### Decision 5: Phase 6 uses temporal graphs from any domain

**Choice:** Abandon the requirement for temporal fraud datasets. Instead, evaluate temporal GNNs on datasets where temporal dynamics are well-established but fraud labels may not exist.

**Rationale:**
- No public dataset with temporal graph structure + fraud labels exists (verified through community research).
- Pay-At-Pump, Sungkyunkwan, Naver Plus Bank have rich temporal structure and can test TGN/GRN/rolling-GCN capabilities.
- The research question for Phase 6 shifts from "does time help fraud detection?" to "does time add value to graph modeling at all, and under what conditions?"

**Alternatives considered:**
- Generate synthetic temporal fraud data: possible but may not generalize.
- Wait for a temporal fraud dataset: impractical (none exists).

### Decision 6: Statistical significance testing in comparison phase

**Choice:** Use paired permutation tests (or McNemar's test for classification outcomes) to validate that observed improvements are statistically significant.

**Rationale:**
- Small PR-AUC improvements (e.g., 0.02) can be noise.
- Statistical significance separates real signal from random variation.
- Essential for the "does graph add value?" conclusion.

## Risks / Trade-offs

[Risk: IEEE-CIS dataset noise] → Mitigation: Use EEV Mitme-Fraud (curated, well-labeled) as primary validation; IEEE-CIS as secondary. Apply robust loss functions (Focal Loss in LightGBM).

[Risk: GNN training instability] → Mitigation: Start with single-layer GCN; use spectral normalization; log training curves; if GCN fails, move to graph features + LightGBM only (stop).

[Risk: Calibration overfitting on small validation sets] → Mitigation: Use 80/10/10 train/val/test split; use cross-validation for calibration fitting.

[Risk: XAI explanations not aligning with ground truth] → This is a valid result, not a failure. Report it. Misaligned explanations reveal model limitations.

[Risk: Phase 6 temporal models perform poorly without fraud labels] → Mitigation: Use unsupervised anomaly scores + compare against any available label signal. The goal is understanding, not SOTA fraud detection.

[Risk: Scaling to 300M nodes is out of current scope] → Mitigation: Phase 8 produces a design document, not an implementation. Use smaller graphs (1M-10M nodes) for sampling experiments.

## Migration Plan

This is a greenfield project. No migration from existing code is needed.

1. Create the `graph-fraud-research` change (done).
2. Implement phases sequentially (LightGBM first → GNN → XAI → temporal → sampling).
3. Archive the old `tgn-antifraud-preparation` change when the new one is ready.

## Open Questions

- Which specific IEEE-CIS features are allowed for the 3-feature LightGBM baseline? (Assumption: transaction amount, time_since_account_creation, n_transactions.)
- For temporal datasets without fraud labels, what is the "anomaly" label source? (Assumption: Pay-At-Pump uses provided anomaly labels; Sungkyunkwan/Naver Plus Bank may not have labels at all — in that case, focus is on model behavior, not fraud-specific metrics.)
- How many synthetic Mitme/Cascade/Money Mule samples per dataset? (Assumption: 1000-5000 per pattern, adjustable based on graph size.)
