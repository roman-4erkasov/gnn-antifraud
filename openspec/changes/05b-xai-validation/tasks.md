## 1. Data loading

- [ ] 1.1 Create `phases/phase05b_xai_validation/src/phase05b_xai_validation/data_loader.py` with module docstring, imports (os, pathlib, pandas, torch_geometric.data), and `DataLoader` class stub with `check_data_exists()`, `load_synthetic_patterns()` methods; verify Python syntax with ast.parse
- [ ] 1.2 Implement `check_data_exists()` checking for synthetic pattern files in `data/synthetic-patterns/` directory; verify returns boolean
- [ ] 1.3 Implement `load_synthetic_patterns(pattern_type)` loading synthetic patterns (Mitme, Cascade, Money Mule) from Phase 05 and returning graph Data objects with ground-truth labels; verify output is compatible with XAI methods from Phase 04a
- [ ] 1.4 Verify loaded patterns have correct ground-truth structure: fraud edges are labeled, node roles are marked

## 2. Pattern validation

- [ ] 2.1 Create `phases/phase05b_xai_validation/src/phase05b_xai_validation/pattern_validator.py` with module docstring, imports (torch, numpy), and `PatternValidator` class stub; verify Python syntax with ast.parse
- [ ] 2.2 Implement `PatternValidator.validate_explanation(explanation, ground_truth_edges, top_k=10)` computing precision/recall of XAI-highlighted edges against ground-truth fraud edges; verify output dict with keys: precision, recall, f1
- [ ] 2.3 Implement `batch_validate(explanations, ground_truth_list, top_k=10)` validating multiple explanations; verify output list of metrics dicts
- [ ] 2.4 Verify validator works with GNNExplainer output: explain a synthetic Mitme pattern node, compute precision/recall against known fraud edges

## 3. Metrics

- [ ] 3.1 Create `phases/phase05b_xai_validation/src/phase05b_xai_validation/metrics.py` with module docstring, imports (numpy), and `compute_precision_recall(predicted_edges, ground_truth_edges)` function stub; verify Python syntax
- [ ] 3.2 Implement `compute_precision_recall(predicted_edges, ground_truth_edges)` returning precision, recall, F1-score; verify output values in [0, 1]
- [ ] 3.3 Implement `compute_metrics_at_thresholds(explanation_scores, ground_truth_edges, thresholds=[5, 10, 20])` computing metrics at multiple top-k thresholds; verify output dict mapping threshold → metrics
- [ ] 3.4 Implement `aggregate_metrics(metrics_list)` computing mean and std across multiple validations; verify output dict with mean_precision, std_precision, etc.

## 4. Cross-pattern comparison

- [ ] 4.1 Create `phases/phase05b_xai_validation/src/phase05b_xai_validation/comparison.py` with module docstring, imports (pandas, numpy), and `PatternComparison` class stub; verify Python syntax
- [ ] 4.2 Implement `PatternComparison.compare_across_patterns(xai_method, pattern_types=["mitme", "cascade", "money_mule"])` running XAI validation on each pattern type and comparing metrics; verify output DataFrame with pattern_type, precision, recall, f1 columns
- [ ] 4.3 Implement `generate_comparison_report(comparison_df)` producing a formatted report showing XAI performance across patterns; verify output is printable
- [ ] 4.4 Implement `identify_best_pattern(comparison_df)` returning the pattern type with highest F1-score; verify output is a string

## 5. End-to-end validation pipeline

- [ ] 5.1 Create `phases/phase05b_xai_validation/run_xai_validation.py` that loads synthetic patterns from Phase 05, loads XAI methods from Phase 04a, runs validation on each pattern type, computes metrics, generates comparison report; verify script runs without errors
- [ ] 5.2 Print summary: (a) precision/recall per pattern type, (b) best/worst pattern for each XAI method, (c) overall recommendation
- [ ] 5.3 Add stop decision check: if XAI precision < 0.20 on all patterns, print "XAI explanations are not reliable for investigation — consider using different XAI parameters or methods"

## 6. Interactive exercises (exercises/)

- [ ] 6.1 Create `phases/phase05b_xai_validation/exercises/` directory
- [ ] 6.2 Create `01-xai-validation-overview.ipynb` with setup code and task to explore XAI validation concepts
- [ ] 6.3 Create `02-pattern-ground-truth.ipynb` with task to load and visualize synthetic patterns
- [ ] 6.4 Create `03-precision-recall-for-xai.ipynb` with task to compute precision/recall for XAI explanations
- [ ] 6.5 Create `04-cross-pattern-comparison.ipynb` with task to compare XAI across pattern types
- [ ] 6.6 Create `05-stop-decision.ipynb` with task to apply stop decision criteria

## 7. Tests

- [ ] 7.1 Create `phases/phase05b_xai_validation/tests/test_pattern_validator.py` with tests for PatternValidator — validate_explanation returns correct precision/recall, batch_validate returns list of correct length; verify with pytest
- [ ] 7.2 Create `phases/phase05b_xai_validation/tests/test_metrics.py` with tests for metrics — compute_precision_recall returns values in [0, 1], compute_metrics_at_thresholds returns dict with correct keys; verify with pytest
- [ ] 7.3 Create `phases/phase05b_xai_validation/tests/test_comparison.py` with tests for PatternComparison — compare_across_patterns returns DataFrame with correct columns, identify_best_pattern returns string; verify with pytest
- [ ] 7.4 Run full test suite `pytest phases/phase05b_xai_validation/tests/ -v` and verify all tests pass

## 8. Educational content — lessons

- [ ] 8.1 Create `phases/phase05b_xai_validation/lessons/01-xai-validation-overview.md` following 5-part structure (Explain, Code, Visualize, Practice, Solution); cover why validate XAI on synthetic patterns, ground truth, precision/recall, trustworthiness; Practice links to `../exercises/01-xai-validation-overview.ipynb`
- [ ] 8.2 Create `phases/phase05b_xai_validation/lessons/02-pattern-ground-truth.md` following 5-part structure; cover synthetic pattern structure (Mitme, Cascade, Money Mule), ground-truth labels; Practice links to `../exercises/02-pattern-ground-truth.ipynb`
- [ ] 8.3 Create `phases/phase05b_xai_validation/lessons/03-precision-recall-for-xai.md` following 5-part structure; cover precision, recall, F1-score, threshold selection; Practice links to `../exercises/03-precision-recall-for-xai.ipynb`
- [ ] 8.4 Create `phases/phase05b_xai_validation/lessons/04-cross-pattern-comparison.md` following 5-part structure; cover pattern-specific performance, structural complexity; Practice links to `../exercises/04-cross-pattern-comparison.ipynb`
- [ ] 8.5 Create `phases/phase05b_xai_validation/lessons/05-stop-decision.md` following 5-part structure; cover stop criteria, actionable guidance, limitations; Practice links to `../exercises/05-stop-decision.ipynb`

## 9. Educational content — notebooks

- [ ] 9.1 Create `phases/phase05b_xai_validation/notebooks/01-xai-validation-analysis.ipynb`; load synthetic patterns, run XAI, compute precision/recall, visualize results; verify notebook runs end-to-end
- [ ] 9.2 Create `phases/phase05b_xai_validation/notebooks/02-pattern-specific-xai.ipynb`; compare XAI across Mitme/Cascade/Money Mule, identify strengths/weaknesses; verify notebook runs end-to-end

## 10. Integration and verification

- [ ] 10.1 Run full test suite `pytest phases/phase05b_xai_validation/tests/ -v` and verify all tests pass
- [ ] 10.2 Run `phases/phase05b_xai_validation/run_xai_validation.py` end-to-end with synthetic data and verify: patterns load, XAI runs, validation metrics computed, comparison report generated, stop decision check passes; confirm exit code 0
