## 1. Project Setup

- [ ] 1.1 Create directory structure: src/datasets, src/models, src/samplers, src/explain, src/benchmark, src/synthetic, tests/ — verify all directories exist
- [ ] 1.2 Create requirements.txt (or pyproject.toml) with dependencies: PyTorch, PyG, DGL, LightGBM, scikit-learn, scipy, networkx, pandas — verify pip install succeeds
- [ ] 1.3 Create base experiment runner template (data load → train → evaluate → explain) — verify it runs with dummy data
- [ ] 1.4 Create logging and config system (YAML config for hyperparameters, seeds, device) — verify config is loaded correctly

## 2. Dataset Layer

- [ ] 2.1 Implement IEEE-CIS Credit Card Fraud dataset loader and verify it produces (X, edge_index, y) — verify with sample output
- [ ] 2.2 Implement EEV Mitme-Fraud dataset loader and verify bipartite graph structure — verify with sample output
- [ ] 2.3 Implement synthetic data generator with 3 fraud patterns (mitme, cascade, money-mule) — verify graph statistics (nodes, edges, fraud ratio)
- [ ] 2.4 Implement common graph feature extraction (degree, betweenness, clustering coeff) — verify values match NetworkX

## 3. No-Graph Baseline

- [ ] 3.1 Implement LightGBM baseline on tabular features (no graph structure) — verify training completes
- [ ] 3.2 Implement feature importance analysis and verify top features are reported — verify with synthetic data where known features matter
- [ ] 3.3 Save baseline results for comparison with graph models

## 4. Graph Models — PyG

- [ ] 4.1 Implement GCN model in PyG and verify it produces correct message passing — verify on single edge case
- [ ] 4.2 Implement GAT model in PyG with multi-head attention and verify attention weights are accessible
- [ ] 4.3 Implement TGN model in PyG (or use pytorch-geometric-tempo) and verify temporal encoding — verify on synthetic temporal data
- [ ] 4.4 Implement ReCWE structural encoding and verify it is added to node features correctly — verify on simple graph
- [ ] 4.5 Implement RWSE structural encoding and verify random walk distances are computed — verify on synthetic graph

## 5. Graph Models — DGL

- [ ] 5.1 Implement GCN model in DGL and verify equivalence with PyG GCN (same architecture, same output) — verify numerical match
- [ ] 5.2 Implement GAT model in DGL with multi-head attention and verify attention weights match PyG — verify numerical match
- [ ] 5.3 Implement TGN model in DGL and verify temporal encoding — verify on synthetic temporal data
- [ ] 5.4 Implement ReCWE structural encoding in DGL and verify equivalent to PyG version — verify numerical match
- [ ] 5.5 Implement RWSE structural encoding in DGL and verify equivalent to PyG version — verify numerical match

## 6. Graph Sampling

- [ ] 6.1 Implement Neighbor Sampling for PyG and verify neighbor extraction — verify on small graph
- [ ] 6.2 Implement Neighbor Sampling for DGL and verify equivalence with PyG — verify numerical match
- [ ] 6.3 Implement GraphSAINT (uniform variant) for PyG and verify subgraph generation — verify subgraph statistics
- [ ] 6.4 Implement GraphSAINT (uniform variant) for DGL and verify equivalence with PyG — verify numerical match
- [ ] 6.5 Implement GraphBolt indexing for PyG (large-scale reference) — verify indexing is built correctly
- [ ] 6.6 Implement sampler benchmark suite (samples/sec, memory, accuracy delta) — verify benchmark report output

## 7. Evaluation & Comparison

- [ ] 7.1 Implement metric computation (AUC-ROC, PR-AUC, Precision@K) — verify against sklearn reference
- [ ] 7.2 Implement automatic experiment runner (loop over: models × samplers × encodings × frameworks) — verify all combinations run
- [ ] 7.3 Implement side-by-side comparison report generator (PyG vs DGL) — verify report format and completeness
- [ ] 7.4 Run full evaluation on IEEE-CIS dataset with all model/sampler combinations — verify report generated
- [ ] 7.5 Run full evaluation on EEV dataset with all model/sampler combinations — verify report generated
- [ ] 7.6 Run full evaluation on synthetic data (all 3 patterns) — verify report generated

## 8. Explainable AI

- [ ] 8.1 Implement GNNExplainer interface (PyG) and verify it returns edge/node/feature importance — verify on single-node prediction
- [ ] 8.2 Implement PGExplainer interface (PyG) and verify it trains and produces explanations — verify training loss decreases
- [ ] 8.3 Extract and analyze GAT attention weights (PyG) and verify they sum to 1 per edge — verify visualization output
- [ ] 8.4 Implement synthetic data XAI validation (measure explanation overlap with ground truth) — verify 70%+ overlap on synthetic mitme pattern
- [ ] 8.5 Replicate all XAI steps (GNNExplainer, PGExplainer, attention) for DGL models — verify equivalence with PyG explanations
- [ ] 8.6 Generate XAI report per dataset showing which model produces most faithful explanations

## 9. CPU/GPU Verification

- [ ] 9.1 Add device flag (--cpu / --cuda) to experiment runner and verify it propagates to all components
- [ ] 9.2 Run identical experiment on CPU and GPU, verify numerical equivalence (max diff < 1e-4) — verify diff report
- [ ] 9.3 Measure CPU vs GPU speedup and verify results are reasonable (2-5x expected for small graphs)

## 10. Final Synthesis

- [ ] 10.1 Compile final summary: "Do we need GNN?" with evidence from all experiments — verify includes quantitative comparisons
- [ ] 10.2 Document best sampler per dataset size regime (small/medium/large) — verify with benchmark data
- [ ] 10.3 Document XAI quality assessment (which explainer works best for fraud patterns) — verify with validation scores
- [ ] 10.4 Create self-paced learning roadmap based on findings — verify actionable next steps
- [ ] 10.5 Archive all experiment configs and results for future reference — verify files are organized
