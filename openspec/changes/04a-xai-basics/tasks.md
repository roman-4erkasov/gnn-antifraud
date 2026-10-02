## 1. Data loading

- [ ] 1.1 Create `phases/phase04a_xai_basics/src/phase04a_xai_basics/data_loader.py` with module docstring, imports, and `DataLoader` class stub with `check_data_exists()`, `download_data()`, and `load_data()` methods; verify Python syntax with ast.parse
- [ ] 1.2 Implement `check_data_exists()` checking for IEEE-CIS data files in shared `data/` directory at project root; verify returns boolean
- [ ] 1.3 Implement `download_data()` downloading IEEE-CIS Fraud Detection dataset to `data/` directory if not present; verify files exist after download
- [ ] 1.4 Implement `load_data()` loading phase-specific data (transaction and identity tables) and returning raw DataFrames; verify correct column types and row counts
- [ ] 1.5 Implement `prepare_graph_data()` constructing a graph suitable for XAI analysis — building edge_index, node features, and labels compatible with GNNExplainer and PGExplainer; verify output is a `torch_geometric.data.Data` object with correct attributes
- [ ] 1.6 Verify output of `prepare_graph_data()` is compatible with `gnn_explainer.py` and `feature_importance.py` by running a smoke test: pass the graph to GNNExplainerWrapper and confirm masks are produced without errors

## 2. GNNExplainer integration

- [ ] 2.1 Create `phases/phase04a_xai_basics/src/phase04a_xai_basics/gnnexplainer.py` with module docstring, imports (torch, torch_geometric.explain, torch_geometric.nn), and `GNNExplainerWrapper` class stub; verify Python syntax with ast.parse
- [ ] 2.2 Implement `GNNExplainerWrapper.explain_node(model, data, node_idx)` returning `Explanation` object with `edge_mask` and `feature_mask` tensors; verify output masks have correct shapes (edge_mask: [n_edges], feature_mask: [n_nodes, n_features])
- [ ] 2.3 Implement `extract_mask_explanations(explanation, top_k=20)` converting masks to lists of (node_idx, score) and (edge_idx, src, dst, score) sorted by absolute mask value; verify lists are sorted descending

## 3. GNNExplainer export and usage

- [ ] 3.1 Implement `explanation_to_json(explanation, node_idx, top_k=20)` producing JSON-serializable dict with node features, edges, and importance scores; verify output is valid JSON
- [ ] 3.2 Implement `explain_top_predictions(model, data, pred_scores, labels, top_n=100)` explaining top-n highest fraud-score predictions on labeled nodes; verify returns list of JSON explanations
- [ ] 3.3 Verify GNNExplainer produces meaningful masks: on synthetic Mitme graph, explain a fraud node and confirm highlighted edges connect to other nodes in the same community

## 4. PGExplainer integration

- [ ] 4.1 Create `phases/phase04a_xai_basics/src/phase04a_xai_basics/pgexplainer.py` with imports (torch, torch_geometric.explain.algorithm.PGExplainer, torch_geometric.nn) and `PGExplainerWrapper` class stub; verify Python syntax with ast.parse
- [ ] 4.2 Implement `PGExplainerWrapper.train(teacher_model, edge_index, train_edges, epochs=50, lr=0.01)` training the PGExplainer explanation model using a teacher GCN; verify training prints explanation loss per 10 epochs
- [ ] 4.3 Implement `PGExplainerWrapper.explain_edge(model, data, edge_idx)` returning edge importance score and explanation vector for a single edge; verify output types (float score, 1D tensor)
- [ ] 4.4 Implement `batch_explain_edges(model, data, edge_indices)` explaining multiple edges at once; verify output shape [n_edges]

## 5. GAT attention weight extraction

- [ ] 5.1 Create `phases/phase04a_xai_basics/src/phase04a_xai_basics/gat_attention.py` with imports and implement `build_gat_model(input_dim, hidden_dim, output_dim, n_heads=4)` returning a GAT model using `torch_geometric.nn.GATConv` with multi-head attention; verify model instantiates correctly
- [ ] 5.2 Implement `extract_attention_weights(model, data)` using forward pass with `return_attention_weights=True` on GATConv layers, returning attention coefficient matrix [n_edges, n_heads]; verify shapes
- [ ] 5.3 Implement `aggregate_attention(attention_matrix, heads_weights=None)` producing per-edge importance score by averaging (or weighted averaging) across attention heads; verify output shape [n_edges]
- [ ] 5.4 Train a GAT model on synthetic data and verify attention weights are in [0, 1] and sum to 1 across source nodes (softmax property)

## 6. Explanation validation against chargeback labels

- [ ] 6.1 Create `phases/phase04a_xai_basics/src/phase04a_xai_basics/validation.py` with module docstring and imports; implement `validate_against_chargeback(explanations, chargeback_labels, edge_index)` checking whether top-importance edges connect to the transaction's merchant or buyer node; verify binary alignment indicator
- [ ] 6.2 Implement `batch_validate_chargeback(explanations, chargeback_labels, edge_index)` computing alignment fraction across all explanations; verify output in [0, 1]

## 7. Investigation lead generation

- [ ] 7.1 Create `phases/phase04a_xai_basics/src/phase04a_xai_basics/leads.py` with module docstring and imports; implement `generate_lead_from_gnn_explanation(explanation_json, node_idx, node_roles, risk_scores)` producing a structured lead dict with keys: description, key_nodes, key_edges, roles, risk_scores, recommended_steps; verify all keys present
- [ ] 7.2 Implement template-based lead text generation: "Node {node_idx} (role: {role}, risk: {risk:.2f}) is connected to {n_key_edges} high-importance edges. Recommended: {steps}"; verify text is human-readable
- [ ] 7.3 Implement `generate_lead_from_pg_explanation(edge_explanation, edge_attrs)` producing a lead describing the transaction edge, contributing features, and whether connected entities are flagged; verify output structure
- [ ] 7.4 Implement `batch_generate_leads(explanations, metadata)` generating leads for all explanations in batch; verify output list length equals input length

## 8. Cross-method explanation comparison

- [ ] 8.1 Create `phases/phase04a_xai_basics/src/phase04a_xai_basics/comparison.py` with module docstring; implement `jaccard_similarity(set_a, set_b)` computing Jaccard similarity between two sets of edge/node identifiers; verify output in [0, 1]
- [ ] 8.2 Implement `compare_top_k_overlap(explanation_a, explanation_b, ks=[5, 10, 20])` computing Jaccard similarity at each k; verify output dict mapping k → similarity
- [ ] 8.3 Implement `consistency_score(explanations_list)` computing fraction of methods agreeing on top-5 edges for each explanation; verify output in [0, 1]
- [ ] 8.4 Implement `run_all_comparisons(gnnexplainer_results, pgexplainer_results, gat_attention_results)` producing a comparison report with all pairwise overlaps and consistency scores; verify output is printable

## 9. End-to-end XAI pipeline

- [ ] 9.1 Create `phases/phase04a_xai_basics/run_xai.py` that loads IEEE-CIS data, builds graph, trains GCN + GAT, generates explanations via all 3 methods, validates against chargeback labels, generates leads, and runs comparisons; verify script runs without errors
- [ ] 9.2 Print summary: (a) chargeback alignment metrics, (b) cross-method consistency scores, (c) sample investigation leads (3 examples); verify output is readable
- [ ] 9.3 Add stop decision check: if chargeback alignment < 0.25 AND cross-method consistency < 0.30, print "⚠ GNN explanations lack trustworthiness — investigate model quality or XAI parameters"

## 10. Interactive exercises (exercises/)

- [ ] 10.1 Create `phases/phase04a_xai_basics/exercises/` directory
- [ ] 10.2 Create `01-introduction-to-xai.ipynb` with setup code and task to explore XAI concepts
- [ ] 10.3 Create `02-gnnexplainer.ipynb` with task to run GNNExplainer and interpret masks
- [ ] 10.4 Create `03-feature-importance.ipynb` with task to rank features by importance
- [ ] 10.5 Create `04-calibration-deep-dive.ipynb` with task to compute ECE and apply calibration
- [ ] 10.6 Create `05-interpreting-fraud-predictions.ipynb` with task to generate and read investigation leads
- [ ] 10.7 Create `06-trust-and-actionability.ipynb` with task to compare cross-method explanations

## 11. Tests

- [ ] 11.1 Create `phases/phase04a_xai_basics/tests/test_gnnexplainer.py` with tests for GNNExplainerWrapper — explain_node returns valid masks, extract_mask_explanations returns sorted list, explanation_to_json is serializable; verify with `pytest phases/phase04a_xai_basics/tests/test_gnnexplainer.py -v`
- [ ] 11.2 Create `phases/phase04a_xai_basics/tests/test_gat_attention.py` with tests for GAT model — build_gat_model returns valid model, extract_attention_weights returns correct shapes, aggregate_attention reduces [n_edges, n_heads] to [n_edges]; verify with pytest
- [ ] 11.3 Create `phases/phase04a_xai_basics/tests/test_validation.py` with tests for validation functions — validate_against_chargeback returns alignment indicator, batch_validate_chargeback returns fraction; verify with pytest
- [ ] 11.4 Create `phases/phase04a_xai_basics/tests/test_leads.py` with tests for lead generation — generate_lead_from_gnn_explanation has all required keys, batch_generate_leads returns correct count; verify with pytest
- [ ] 11.5 Create `phases/phase04a_xai_basics/tests/test_comparison.py` with tests for jaccard_similarity (identical sets → 1.0, disjoint sets → 0.0), compare_top_k_overlap, consistency_score; verify with pytest

## 12. Educational content — lessons

- [ ] 12.1 Create `phases/phase04a_xai_basics/lessons/01-introduction-to-xai.md` following 5-part structure (Explain, Code, Visualize, Practice, Solution); cover why XAI matters for fraud detection, the opacity problem, regulatory context, and investigator workflows; Practice links to `../exercises/01-introduction-to-xai.ipynb`
- [ ] 12.2 Create `phases/phase04a_xai_basics/lessons/02-gnnexplainer.md` following 5-part structure; cover GNNExplainer algorithm, node-level masks, edge masks, feature masks, local subgraph extraction, and how to read mask outputs; Practice links to `../exercises/02-gnnexplainer.ipynb`
- [ ] 12.3 Create `phases/phase04a_xai_basics/lessons/03-feature-importance.md` following 5-part structure; cover feature mask interpretation, ranking features by importance, linking top features to known fraud signals; Practice links to `../exercises/03-feature-importance.ipynb`
- [ ] 12.4 Create `phases/phase04a_xai_basics/lessons/04-calibration-deep-dive.md` following 5-part structure; cover ECE, reliability diagrams, Platt scaling, temperature scaling, and fraud threshold selection; Practice links to `../exercises/04-calibration-deep-dive.ipynb`
- [ ] 12.5 Create `phases/phase04a_xai_basics/lessons/05-interpreting-fraud-predictions.md` following 5-part structure; cover reading investigation leads, mapping XAI output to fraud patterns (Mitme, Cascade), and false positive analysis; Practice links to `../exercises/05-interpreting-fraud-predictions.ipynb`
- [ ] 12.6 Create `phases/phase04a_xai_basics/lessons/06-trust-and-actionability.md` following 5-part structure; cover when to trust explanations, cross-method agreement, actionability criteria, and human-in-the-loop considerations; Practice links to `../exercises/06-trust-and-actionability.ipynb`

## 13. Educational content — notebooks

- [ ] 13.1 Create `phases/phase04a_xai_basics/notebooks/01-explaining-predictions.ipynb`; load trained GCN, explain top fraud predictions with GNNExplainer, visualize subgraph masks, interpret edge and feature importance; verify notebook runs end-to-end
- [ ] 13.2 Create `phases/phase04a_xai_basics/notebooks/02-feature-importance-analysis.ipynb`; extract feature masks for multiple predictions, rank features by importance, analyze which features drive fraud predictions; verify notebook runs end-to-end
- [ ] 13.3 Create `phases/phase04a_xai_basics/notebooks/03-calibration-curves.ipynb`; plot reliability diagrams, compute ECE, apply Platt scaling, compare pre/post calibration, select fraud thresholds; verify notebook runs end-to-end

## 14. Integration and verification

- [ ] 14.1 Run full test suite `pytest phases/phase04a_xai_basics/tests/ -v` and verify all tests pass including new XAI tests
- [ ] 14.2 Run `phases/phase04a_xai_basics/run_xai.py` end-to-end with synthetic data and verify: GCN trains, GNNExplainer explains, PGExplainer trains and explains, GAT trains and extracts attention, validation metrics reported, leads generated, comparison table printed; confirm exit code 0
