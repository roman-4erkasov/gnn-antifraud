## 1. Setup and data loading

- [ ] 1.1 Create `phases/phase07_cross_dataset/src/phase07_cross_dataset/data_loader.py` with module docstring, imports (os, pathlib, pandas, torch, torch_geometric), and `DataLoader` class stub with `check_data_exists()`, `download_data()`, `load_dataset(name)` method stubs; verify Python syntax
- [ ] 1.2 Implement `DataLoader.check_data_exists()` checking for all required datasets (IEEE-CIS, Pay-At-Pump, Sungkyunkwan, Naver Plus Bank) in the shared `data/` directory at project root; verify output is a dict mapping dataset name → bool
- [ ] 1.3 Implement `DataLoader.download_data()` downloading any missing datasets to the shared `data/` directory; verify all datasets present after call
- [ ] 1.4 Implement `DataLoader.load_dataset(name)` loading a single named dataset and returning it in a standard format (features, labels, graph structure if applicable); verify output is valid for each supported dataset name
- [ ] 1.5 Implement `load_all_datasets()` that loads all available fraud datasets (IEEE-CIS, Pay-At-Pump, Sungkyunkwan, Naver Plus Bank) and returns a dict mapping dataset name → loaded data; verify all datasets loaded successfully
- [ ] 1.6 Implement `prepare_comparable_data(datasets)` that normalizes datasets for cross-dataset comparison by aligning feature schemas, normalizing value ranges, and standardizing label formats; verify output datasets have compatible structures
- [ ] 1.7 Verify output of `prepare_comparable_data()` is compatible with `transfer_learning.py` and `domain_adaptation.py` input expectations; verify with a small integration test

## 2. Cross-dataset model comparison framework

- [ ] 2.1 Create `phases/phase07_cross_dataset/src/phase07_cross_dataset/compare.py` with module docstring, imports (torch, torch_geometric.nn, lightgbm, scikit-learn), and `CrossDatasetComparator` class stub; verify Python syntax
- [ ] 2.2 Implement `CrossDatasetComparator.add_model(model_name, model_fn, model_kwargs)` registering a model for cross-dataset evaluation; verify model can be retrieved by name
- [ ] 2.3 Implement `CrossDatasetComparator.train_on_dataset(dataset_name, dataset_dict)` training all registered models on a specified dataset; verify training completes and model states are saved
- [ ] 2.4 Implement `CrossDatasetComparator.evaluate_on_dataset(dataset_name, dataset_dict)` evaluating all trained models on a specified dataset; verify output dict with model_name → metrics mapping
- [ ] 2.5 Implement `CrossDatasetComparator.evaluate_all_pairwise(source_datasets, target_datasets)` evaluating all source → target combinations; verify output is a matrix (source × target × model → metrics)
- [ ] 2.6 Implement `CrossDatasetComparator.get_comparison_matrix(metric="PR-AUC")` producing a summary matrix of metrics across all evaluations; verify output shape matches evaluation dimensions

## 3. Bootstrapping-based confidence intervals

- [ ] 3.1 Create `phases/phase07_cross_dataset/src/phase07_cross_dataset/statistics.py` with module docstring, imports (numpy, scipy, sklearn), and `bootstrap_ci(scores, labels, n_bootstrap=1000, ci=0.95, stratified=True)` function stub; verify Python syntax
- [ ] 3.2 Implement `bootstrap_ci` computing confidence intervals for PR-AUC via 1000 bootstrap resamples with optional stratified sampling; verify output (lower, upper, estimate) in valid range
- [ ] 3.3 Implement `bootstrap_brier_ci(probs, labels, n_bootstrap=1000)` computing confidence intervals for Brier score; verify output is valid CI
- [ ] 3.4 Implement `bootstrap_logloss_ci(probs, labels, n_bootstrap=1000)` computing confidence intervals for log-loss; verify output is valid CI
- [ ] 3.5 Implement `bootstrap_precision_at_k_ci(probs, labels, k=100, n_bootstrap=1000)` computing confidence intervals for Precision@K; verify output is valid CI
- [ ] 3.6 Implement `compute_all_metric_cis(model_predictions, labels, metrics=["PR-AUC", "Brier", "log-loss", "Precision@K"])` computing CIs for all specified metrics; verify output dict with all metric CIs

## 4. Statistical significance testing

- [ ] 4.1 Implement `wilcoxon_test(scores_a, scores_b)` performing Wilcoxon signed-rank test on paired score lists; verify output (statistic, p-value) with p-value in [0, 1]
- [ ] 4.2 Implement `mann_whitney_test(scores_a, scores_b)` performing Mann-Whitney U test on independent score lists; verify output (statistic, p-value)
- [ ] 4.3 Implement `benjamini_hochberg_fdr(p_values, alpha=0.05)` applying Benjamini-Hochberg FDR correction to a list of p-values; verify output is list of corrected p-values in [0, 1]
- [ ] 4.4 Implement `compare_models_pairwise(bootstrap_scores_a, bootstrap_scores_b, alpha=0.05)` comparing two models across bootstrap samples with both Wilcoxon and Mann-Whitney tests; verify output dict with test results, p-values, significance flags
- [ ] 4.5 Implement `generate_significance_matrix(model_names, all_bootstrap_scores, alpha=0.05)` generating a pairwise significance matrix across all models; verify output is square matrix with p-values and significance flags

## 5. Transfer learning evaluation

- [ ] 5.1 Create `phases/phase07_cross_dataset/src/phase07_cross_dataset/transfer.py` with module docstring, imports (torch, torch_geometric.nn, lightgbm), and `TransferEvaluator(pretrained_model, target_dataset, target_labels)` class stub; verify Python syntax
- [ ] 5.2 Implement `TransferEvaluator.fine_tune(fraction=0.1, epochs=20, lr=0.001)` fine-tuning pretrained model on a fraction of target dataset labels; verify training completes and returns evaluation metrics
- [ ] 5.3 Implement `TransferEvaluator.compare_with_scratch(fraction=0.1)` comparing fine-tuning performance against training from scratch on the same fraction; verify output contains both models' metrics
- [ ] 5.4 Implement `TransferEvaluator.transfer_curve(fractions=[0.1, 0.25, 0.5, 0.75, 1.0])` evaluating transfer quality across multiple label fractions; verify output is a dict with fraction → metrics mapping
- [ ] 5.5 Implement `TransferEvaluator.evaluate_shared_features(pretrained_features, target_features)` analyzing which shared features transfer best; verify output includes feature importance overlap metric

## 6. Cross-dataset feature compatibility analysis

- [ ] 6.1 Implement `analyze_feature_overlap(dataset_a_features, dataset_b_features)` comparing feature schemas between two datasets; verify output dict with shared_features, dataset_a_only, dataset_b_only
- [ ] 6.2 Implement `analyze_feature_distributions(dataset_a_features, dataset_b_features, metrics=["wasserstein", "KL"])` comparing value distributions for shared features; verify output dict with per-feature distance metrics
- [ ] 6.3 Implement `generate_feature_recommendations(overlap_result, distribution_result)` producing recommended feature transformations for cross-dataset transfer; verify output dict with feature → transformation mapping
- [ ] 6.4 Implement `cross_dataset_feature_report(all_datasets)` generating a comprehensive feature compatibility report across all datasets; verify output is a structured dict suitable for JSON export
- [ ] 6.5 Implement `_save_feature_report(report, output_path="outputs/cross_dataset_features.json")` serializing the feature report to JSON; verify output file exists and is valid JSON

## 7. Model-agnostic evaluation report

- [ ] 7.1 Implement `generate_evaluation_report(all_evaluations, all_cis, all_significance)` producing a structured evaluation report with per-dataset model rankings, transfer quality summary, and statistical significance flags; verify output dict with all required sections
- [ ] 7.2 Implement `rank_models_per_dataset(dataset_name, model_metrics)` ranking models by PR-AUC for a specific dataset; verify output is ordered list of model_name → metric tuples
- [ ] 7.3 Implement `summarize_transfer_quality(transfer_results)` summarizing transfer quality across all dataset pairs; verify output includes average transfer PR-AUC and best/worst transfers
- [ ] 7.4 Implement `identify_overall_best_model(all_metrics)` identifying the model with best overall performance across all datasets and transfer scenarios; verify output is a model name with supporting metrics
- [ ] 7.5 Implement `report_to_json(report, output_path="outputs/cross_dataset_report.json")` serializing the report to JSON; verify output file exists and is valid JSON
- [ ] 7.6 Implement `report_to_markdown(report, output_path="outputs/cross_dataset_report.md")` serializing the report to human-readable markdown; verify output file exists and is well-formatted

## 8. Educational content — lessons

- [ ] 8.1 Create `phases/phase07_cross_dataset/lessons/01-cross-dataset-challenges.md` covering why models fail across datasets: distribution shift, feature incompatibility, label definition differences; follow 5-part structure (Explain, Code, Visualize, Practice, Solution)
- [ ] 8.2 Create `phases/phase07_cross_dataset/lessons/02-transfer-learning.md` covering fine-tuning pretrained models on target datasets, transfer curves across label fractions; follow 5-part structure
- [ ] 8.3 Create `phases/phase07_cross_dataset/lessons/03-domain-adaptation.md` covering techniques for reducing domain shift: feature alignment, distribution matching; follow 5-part structure
- [ ] 8.4 Create `phases/phase07_cross_dataset/lessons/04-dataset-comparison.md` covering systematic comparison of dataset characteristics: schema overlap, distribution distances; follow 5-part structure
- [ ] 8.5 Create `phases/phase07_cross_dataset/lessons/05-generalization-strategies.md` covering strategies for improving cross-dataset generalization: ensemble methods, feature selection, regularization; follow 5-part structure

## 9. Educational content — notebooks

- [ ] 9.1 Create `phases/phase07_cross_dataset/notebooks/01-transfer-analysis.ipynb` with hands-on transfer learning walkthrough: train on source dataset, fine-tune on target, visualize transfer curves across label fractions; include code cells, markdown explanations, and output visualizations
- [ ] 9.2 Create `phases/phase07_cross_dataset/notebooks/02-cross-dataset-evaluation.ipynb` with full cross-dataset evaluation pipeline: pairwise model comparison, bootstrapping confidence intervals, significance testing, report generation; include code cells, markdown explanations, and output visualizations

## 10. Integration and verification

- [ ] 10.1 Create `phases/phase07_cross_dataset/run_cross_dataset.py` that runs full cross-dataset evaluation on all datasets, models, computes CIs, significance tests, transfer curves, feature report, and generates final report; verify script runs without errors
- [ ] 10.2 Print summary: per-dataset model rankings, overall best model, best/worst transfers, stop decision check; verify output is human-readable
- [ ] 10.3 Implement stop decision check: if no model achieves PR-AUC >= 0.30 on ANY cross-dataset test set, print "⚠ Cross-dataset transfer is not viable with current approaches"
- [ ] 10.4 Create `phases/phase07_cross_dataset/tests/test_compare.py` with tests for cross-dataset comparison framework — verify model registration, pairwise evaluation, comparison matrix; verify with pytest
- [ ] 10.5 Create `phases/phase07_cross_dataset/tests/test_statistics.py` with tests for bootstrapping and significance tests — verify CIs are valid, p-values in [0, 1], BH correction works; verify with pytest
- [ ] 10.6 Create `phases/phase07_cross_dataset/tests/test_transfer.py` with tests for transfer evaluation — verify fine-tuning runs, compare_with_scratch works, transfer_curve produces multiple points; verify with pytest
- [ ] 10.7 Create `phases/phase07_cross_dataset/tests/test_feature_compat.py` with tests for feature analysis — verify overlap detection, distribution comparison, recommendations; verify with pytest
- [ ] 10.8 Create `phases/phase07_cross_dataset/tests/test_integration.py` with end-to-end test: small synthetic datasets, run full comparison, verify report generates; verify with pytest
- [ ] 10.9 Run full test suite `pytest phases/phase07_cross_dataset/tests/ -v` and verify all tests pass including new cross-dataset tests
- [ ] 10.10 Run `phases/phase07_cross_dataset/run_cross_dataset.py` end-to-end with synthetic data and verify: all evaluations complete, CIs computed, significance tested, transfer curves generated, report exported, stop decision check passes; confirm exit code 0
