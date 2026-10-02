## Context

Phases 1–7 (`lightgbm-baseline`, `minimal-gnn`, `structural-encoding`, `xai-basics`, `pattern-detection`, `xai-validation`, `temporal-graphs`) have established baselines, GNN models, temporal models, XAI, and pattern detection on individual datasets. Each dataset has been treated in isolation. We have no statistical rigor (no confidence intervals, no significance testing) and no cross-dataset comparison framework. The research cannot answer whether findings generalize across IEEE-CIS, Pay-At-Pump, Sungkyunkwan, and Naver Plus Bank, or whether models trained on one dataset transfer to another.

## Goals / Non-Goals

**Goals:**
- Implement cross-dataset model comparison framework: train models on one dataset, evaluate on others.
- Implement bootstrapping-based confidence intervals for all metrics (PR-AUC, Brier score, etc.).
- Implement statistical significance testing between model pairs (Wilcoxon signed-rank, Mann-Whitney U).
- Implement transfer learning evaluation: fine-tuning models pretrained on one dataset.
- Implement cross-dataset feature compatibility analysis.
- Produce a structured evaluation report comparing all models across all datasets.

**Non-Goals:**
- Multi-task learning or adversarial domain adaptation — these are reserved for future work.
- Real-world deployment of cross-dataset models — research only.
- Hyperparameter optimization for cross-dataset transfer — assume best hyperparameters from individual dataset evaluations.

## Decisions

### Decision 1: Cross-dataset evaluation protocol

**Choice:** Use a full leave-one-dataset-out evaluation: for each dataset D, train models on D and evaluate on all other datasets. Also evaluate transfer from each dataset to each other dataset.

**Rationale:** Maximizes coverage of transfer scenarios. Leave-one-out is computationally tractable with the number of datasets (4), and provides clear signals about which datasets produce robust models.

**Alternatives considered:**
- Train on all except one and evaluate on the held-out — loses dataset-specific insights.
- Random dataset splits — not reproducible, harder to interpret.

### Decision 2: Bootstrapping methodology

**Choice:** Use 1000 bootstrap resamples with stratified sampling per class for all confidence interval computations.

**Rationale:** Stratified sampling preserves class distribution, which is critical for imbalanced fraud detection. 1000 resamples is standard for stable confidence intervals.

**Alternatives considered:**
- 500 resamples — faster but less stable CIs.
- 5000 resamples — more precise but 5× slower, diminishing returns.
- Analytic confidence intervals — rely on normality assumptions that don't hold for PR-AUC.

### Decision 3: Statistical tests

**Choice:** Use Wilcoxon signed-rank test for paired comparisons (same model, different datasets or vice versa) and Mann-Whitney U test for independent comparisons (different models).

**Rationale:** Non-parametric tests that don't assume normal distribution of metrics. Wilcoxon handles paired bootstrap samples; Mann-Whitney handles independent model comparisons.

**Alternatives considered:**
- t-test — assumes normality, inappropriate for PR-AUC distributions.
- Permutation test — valid but computationally heavier.
- Bonferroni correction — too conservative for many pairwise tests; use Benjamini-Hochberg FDR instead.

### Decision 4: Transfer learning approach

**Choice:** Fine-tune pretrained models on target dataset with limited labeled data (10%, 25%, 50% of labeled samples), comparing against training from scratch.

**Rationale:** Demonstrates practical value of transfer learning: when labeled data is scarce in the target domain, pretrained models can provide significant improvement.

**Alternatives considered:**
- Feature extraction (freeze pretrained model, train linear classifier) — simpler but may not capture domain shift well.
- Adversarial domain adaptation — significantly more complex, outside scope.
- Multi-task learning — requires training all datasets jointly, different paradigm.

### Decision 5: Feature compatibility analysis

**Choice:** Compare feature schemas across datasets by column name, data type, and value distribution. Identify shared features that can be directly reused and features requiring transformation.

**Rationale:** Direct column comparison is simple, transparent, and actionable. Value distribution comparison (via Wasserstein distance or KL divergence) identifies features that need normalization or transformation.

**Alternatives considered:**
- Embedding-based feature alignment — too complex for research prototype.
- Automated feature mapping via LLM — unreliable and introduces external dependency.

### Decision 6: Report format

**Choice:** Produce both a structured JSON report and a human-readable markdown summary.

**Rationale:** JSON enables programmatic consumption by downstream tools; markdown enables human review and archival.

**Alternatives considered:**
- Only JSON — not human-friendly.
- Only markdown — not machine-parseable.
- HTML report — adds CSS/JS dependencies unnecessarily.

### Decision 7: Data storage and loading

**Choice:** Phase 07 evaluates cross-dataset generalization using multiple fraud datasets (IEEE-CIS, Pay-At-Pump, Sungkyunkwan, Naver Plus Bank). Each phase implements its own `data_loader.py` that checks for data in the shared `data/` directory at project root, downloads if missing, and loads multiple datasets for cross-dataset evaluation.

**Rationale:** Self-contained data loading ensures reproducibility and eliminates external dependencies on manually prepared data. Checking the shared `data/` directory avoids redundant downloads across phases. Downloading if missing ensures the phase can run end-to-end without manual intervention.

**Alternatives considered:**
- Rely on each dataset's own phase loader — duplicates logic, no single entry point for cross-dataset loading.
- Require manual data download — breaks reproducibility and adds friction for new users.
- Store data in phase-local directory — wastes disk space when multiple phases use the same datasets.

### Decision 8: Interactive exercises with linked notebooks

**Choice:** Each lesson's Practice section links to an interactive Jupyter notebook in `exercises/` directory.

**Rationale:** Practice exercises in markdown are static text. Linking to notebooks allows users to open, run, and modify code directly. Minimal template: setup code, task description, empty code cell for user solution, solution in markdown code block.

**Alternatives considered:**
- All exercises in one notebook — harder to navigate, no clear mapping to lessons
- Exercises embedded in lesson notebooks — mixes demonstration and practice
- Practice only in markdown — requires copy-paste, less interactive

## Educational Content

### Structure

Each lesson follows a 5-part structure:

1. **Explain** — Conceptual overview with motivation and real-world context
2. **Code** — Annotated code examples implementing the core technique
3. **Visualize** — Plots and diagrams illustrating key results and patterns
4. **Practice** — Guided exercises for the reader to apply the technique
5. **Solution** — Reference solutions with discussion of trade-offs

### Lessons

| Lesson | Topic |
|--------|-------|
| `01-cross-dataset-challenges.md` | Why models fail when transferred across datasets; distribution shift, feature incompatibility, label definition differences |
| `02-transfer-learning.md` | Fine-tuning pretrained models on target datasets; transfer curves across label fractions |
| `03-domain-adaptation.md` | Techniques for reducing domain shift; feature alignment, distribution matching |
| `04-dataset-comparison.md` | Systematic comparison of dataset characteristics; schema overlap, distribution distances |
| `05-generalization-strategies.md` | Strategies for improving cross-dataset generalization; ensemble methods, feature selection, regularization |

### Notebooks

| Notebook | Topic |
|----------|-------|
| `01-transfer-analysis.ipynb` | Hands-on transfer learning: train on source, fine-tune on target, visualize transfer curves |
| `02-cross-dataset-evaluation.ipynb` | Full cross-dataset evaluation pipeline: pairwise comparison, significance testing, report generation |

## Risks / Trade-offs

| Risk | Mitigation |
|------|-----------|
| Some dataset pairs may have completely incompatible features, making transfer impossible | Report incompatibility clearly; do not force transfer where it is not feasible |
| Bootstrapping on small test sets may produce unstable confidence intervals | Use stratified sampling; report CI width as an indicator of stability; warn when CIs are too wide (> 0.1) |
| Transfer learning may not outperform training from scratch — research may show negative results | Document this as a valid finding; negative transfer is itself a scientific insight |
| Multi-comparison problem: many pairwise tests increase false positive rate | Use Benjamini-Hochberg FDR correction for p-values; report both raw and corrected p-values |
| Computational cost: full cross-dataset matrix requires training K×K models per model type | Prioritize CPU-efficient models (LightGBM) first; run GNN transfers on a subset; cache pretrained model weights |
| Kaggle datasets (IEEE-CIS, Sungkyunkwan, Naver Plus Bank) may have restricted usage terms | Use datasets only for research evaluation; do not redistribute; document data usage restrictions in report |

## Migration Plan

1. Create `phases/phase07_cross_dataset/src/phase07_cross_dataset/` package with `compare.py`, `statistics.py`, `feature_compat.py`, `transfer.py`.
2. Implement cross-dataset evaluation pipeline.
3. Create `phases/phase07_cross_dataset/tests/` test suite.
4. Create `phases/phase07_cross_dataset/run_cross_dataset.py` for end-to-end cross-dataset evaluation.
5. No migration from existing code needed — this is a greenfield module.

## Open Questions

- What transfer ratios should be used for fine-tuning? (Assumption: 10%, 25%, 50% of labeled target data.)
- Should cross-dataset evaluation include temporal models? (Assumption: yes, but only for temporal datasets that share compatible structures.)
- What is the stop decision? (Assumption: if no model achieves PR-AUC >= 0.30 on ANY cross-dataset test set, flag that cross-dataset transfer is not viable with current approaches.)
