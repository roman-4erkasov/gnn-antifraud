## 1. Project setup and infrastructure

- [x] 1.1 Initialize project structure: create src/scoring/, src/xai/, src/patterns/, src/temporal/, src/sampling/, src/data/, src/utils/, tests/ and verify directory structure
- [x] 1.2 Add Python dependencies (PyTorch, PyG, LightGBM, scikit-learn, SHAP, pandas, numpy) and verify `pip install` succeeds
- [x] 1.3 Create shared calibration utility (isotonic regression wrapper) and verify it fits and transforms on a synthetic dataset with reduced Brier score
- [x] 1.4 Create shared metrics utility (PR-AUC, ROC-AUC, Brier score, log-loss, Precision@K, statistical significance tests) and verify against scikit-learn references

## 2. Data preparation

- [x] 2.1 Create IEEE-CIS fraud detection data loader and verify it loads training/test features, labels, and transaction tables correctly
- [x] 2.2 Create EEV Mitme-Fraud data loader and verify it loads the 10-node synthetic Mitme graphs with ground truth labels
- [ ] 2.3 Create Pay-At-Pump temporal graph loader and verify it loads time-stamped edge events with anomaly labels
- [ ] 2.4 Create Sungkyunkwan temporal graph loader and verify it loads the temporal bipartite graph snapshots
- [ ] 2.5 Create synthetic data generator for Mitme, Cascade, and Money Mule patterns and verify generated graphs have correct structural properties (e.g., Mitme has circular edges between colluding pairs)
- [ ] 2.6 Create temporal leakage checker and verify it correctly detects edges that cross train/test time boundaries

## 3. Phase 1 — LightGBM baseline (needs graphs?)

- [ ] 3.1 Implement minimal LightGBM baseline with exactly 3 features (transaction amount, time since account creation, number of prior transactions) and verify it trains on IEEE-CIS
- [ ] 3.2 Apply isotonic calibration to baseline predictions and verify Brier score improves on held-out validation set
- [ ] 3.3 Evaluate baseline with all metrics (PR-AUC, ROC-AUC, Brier score, log-loss, Precision@K) and verify output is saved to artifacts
- [ ] 3.4 Extend LightGBM with graph-derived features (node degree, edge count, community membership, RWSE values) and verify feature matrix size increases
- [ ] 3.5 Apply calibration to graph-aware LightGBM and evaluate; verify delta metrics (ΔPR-AUC, ΔBrier) vs minimal baseline are reported
- [ ] 3.6 If ΔPR-AUC < 0.01 (or not statistically significant), flag Phase 1 as "graph adds minimal value" and note stop decision

## 4. Phase 2 — Minimal GNN (does GNN add value?)

- [ ] 4.1 Implement single-layer GCN (PyG `GCNConv`) with a classification head and verify it builds on IEEE-CIS node features + edge index
- [ ] 4.2 Train GCN on node classification task (binary fraud detection) and verify training converges (loss decreases over epochs)
- [ ] 4.3 Apply isotonic calibration to GCN predictions and verify Brier score vs LightGBM baselines
- [ ] 4.4 Evaluate GCN with all metrics and compare against LightGBM minimal and graph-aware baselines; perform statistical significance test and report p-value
- [ ] 4.5 If GCN ΔPR-AUC vs graph-aware LightGBM < 0.01 (or not significant), flag Phase 2 as "GNN adds minimal value" and note stop decision

## 5. Phase 3 — Structural encoding deep dive

- [ ] 5.1 Implement ReCWE positional encoding (PyG `RPELayer` or custom) and integrate as graph attribute; verify encoding dimensionality
- [ ] 5.2 Implement Random Walk Structural Encoding (RWSE) — compute RW statistics per node and concatenate as node features; verify feature matrix shape
- [ ] 5.3 Implement Spectral encoding (graph Laplacian eigenvectors) as node features; verify eigenvector computation and truncation
- [ ] 5.4 Implement community detection (Louvain/Label Propagation via igraph/louvain) and add community label as node feature; verify community assignment
- [ ] 5.5 Train separate GCN models with each encoding (ReCWE, RWSE, Spectral, Community) and compare PR-AUC, Brier score
- [ ] 5.6 Compute SHAP values for graph-aware LightGBM with each encoding set and compare feature importance rankings across encodings
- [ ] 5.7 Produce a summary table: encoding × model × metrics; identify which encoding "drives" fraud detection most

## 6. Phase 4 — XAI

- [ ] 6.1 Integrate GNNExplainer with the Phase 2/3 GCN models; verify it produces edge masks and feature masks for sample predictions
- [ ] 6.2 Export GNNExplainer outputs as JSON (subgraph indices, edge weights, feature importances) and verify schema compliance
- [ ] 6.3 Integrate PGExplainer for edge-level predictions; verify it trains on a subset of edges and produces global explanation model
- [ ] 6.4 Integrate Attention weight extraction from GAT model; verify attention aggregation produces edge-level importance scores
- [ ] 6.5 Compare GNNExplainer masks vs GAT attention weights (Jaccard similarity of top-k edges) and report overlap metrics
- [ ] 6.6 Validate XAI on synthetic Mitme data: measure precision/recall of XAI-identified edges against known Mitme structure
- [ ] 6.7 Validate XAI on EEV Mitme-Fraud: measure precision/recall of XAI-identified nodes against ground-truth colluding pairs
- [ ] 6.8 Validate XAI on IEEE-CIS chargebacks: check if high-importance edges align with merchant-customer relationships
- [ ] 6.9 Implement investigation lead generator from XAI outputs; verify it produces structured leads (key nodes/edges, roles, recommended steps)

## 7. Phase 5 — Pattern detection

- [ ] 7.1 Generate synthetic Mitme dataset (1000-5000 samples per graph) with known colluding merchant-buyer pairs and verify structural correctness
- [ ] 7.2 Generate synthetic Cascade dataset with known star-like subgraphs (hub + leaves) and verify structure
- [ ] 7.3 Generate synthetic Money Mule dataset with known chain-like transaction paths and verify structure
- [ ] 7.4 Train graph classifiers (GCN/GAT) on each synthetic pattern type; report Recall and Precision@K per pattern vs LightGBM baseline
- [ ] 7.5 Run PGExplainer/GNNExplainer on detected pattern instances and verify explanations identify the correct structural role (hub, chain edge, circular pair)
- [ ] 7.6 Implement pattern template generator: produce structured output with pattern type, suspicious nodes/edges, structural signatures, and recommended investigation actions
- [ ] 7.7 Produce a comparison: XAI explanation quality across pattern types (which pattern is easiest/hardest to explain)

## 8. Phase 6 — Temporal graphs

- [ ] 8.1 Implement TGN model with node memory and relation encoder; verify it processes time-stamped events from Pay-At-Pump
- [ ] 8.2 Train TGN on Pay-At-Pump temporal anomaly task and report PR-AUC, Brier score vs static GCN and LightGBM on temporal snapshots
- [ ] 8.3 Implement GRN (Graph Recurrent Network) with snapshot sequence processing; verify it trains on Sungkyunkwan temporal snapshots
- [ ] 8.4 Train GRN on Sungkyunkwan and Naver Plus Bank; report metrics and compare against TGN and static GCN
- [ ] 8.5 Implement rolling-window GCN: split temporal graph into fixed windows, build static snapshots, train GCN per window; verify correct temporal ordering
- [ ] 8.6 Compare rolling-window GCN vs TGN vs GRN on PR-AUC, Brier score, and inference latency; report tradeoffs
- [ ] 8.7 Run temporal leakage check on all temporal splits and verify no test-period edges leak into training graph
- [ ] 8.8 Produce a summary: when does temporal modeling help? Which model is best for which dataset type?

## 9. Phase 7 — Cross-dataset comparison

- [ ] 9.1 Run all models (LightGBM minimal, LightGBM graph-aware, GCN, GAT, TGN, GRN, rolling-GCN) across IEEE-CIS, EEV, Pay-At-Pump, Sungkyunkwan, Naver Plus Bank; collect all metrics
- [ ] 9.2 Apply calibration to all model outputs across all datasets and report calibrated Brier scores
- [ ] 9.3 Perform statistical significance testing (paired permutation test or McNemar's test) for all model pairs per dataset; collect p-values
- [ ] 9.4 Evaluate cross-dataset generalization: train on one dataset, test on another (e.g., IEEE-CIS → EEV); report performance drop
- [ ] 9.5 Produce a summary matrix: model × dataset → PR-AUC, Brier score, Δcalibrated, significance flag
- [ ] 9.6 Answer the key research question: "Does graph modeling add value?" with evidence from all phases

## 10. Phase 8 — Sampling and scaling design

- [ ] 10.1 Implement Neighbor Sampling (PyG `NeighborLoader`) with configurable fan-out per layer; verify it produces valid subgraph mini-batches
- [ ] 10.2 Train GCN with Neighbor Sampling; measure training time reduction and ΔPR-AUC vs full-graph training
- [ ] 10.3 Implement GraphSAINT (uniform, random-walk, degree-based) sampling; verify subgraph generation and bias correction
- [ ] 10.4 Compare GraphSAINT vs Neighbor Sampling: convergence speed, training time per epoch, final quality
- [ ] 10.5 Design partitioning strategy document: partitioning algorithm (METIS target), number of partitions, edge-cut target, feature precomputation pipeline
- [ ] 10.6 Create Spark-based structural feature precomputation prototype (PageRank, community labels) on a 1M-node synthetic graph and verify distributed computation
- [ ] 10.7 Document data loading architecture for 300M+ node graphs (GPU memory constraints, multi-GPU strategy)

## 11. Final synthesis and documentation

- [ ] 11.1 Compile all results into a structured report: phase-by-phase findings, stop decisions, key insights
- [ ] 11.2 Create a comparison appendix: all models × all datasets × all metrics with calibration and significance flags
- [ ] 11.3 Create an XAI appendix: explanation quality across pattern types, investigation lead examples
- [ ] 11.4 Archive the old `tgn-antifraud-preparation` change and verify `openspec archive` succeeds
- [ ] 11.5 Final validation: run `openspec validate --strict` and verify no errors
