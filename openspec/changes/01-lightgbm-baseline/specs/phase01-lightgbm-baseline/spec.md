## Purpose

Classify transactions as fraudulent or legitimate using LightGBM with minimal features (3) and graph-derived features, with mandatory calibration and stop-early evaluation criteria.

## ADDED Requirements

### Requirement: LightGBM baseline with minimal features

The system SHALL implement a LightGBM baseline classifier trained on a feature matrix with no more than 3 features per transaction, providing a minimal-performance reference point.

#### Scenario: Baseline training with 3 features

- **WHEN** the system is initialized with the baseline configuration
- **THEN** it trains a LightGBM model using at most 3 features (transaction amount, time since account creation, number of prior transactions)

#### Scenario: Baseline evaluation

- **WHEN** baseline model is trained
- **THEN** the system evaluates and reports PR-AUC, ROC-AUC, Brier score, and Precision@K on a held-out test set

### Requirement: Isotonic calibration for LightGBM

The system SHALL apply isotonic regression calibration to the probability outputs of the LightGBM baseline, ensuring well-calibrated scores.

#### Scenario: Calibration application

- **WHEN** the LightGBM baseline produces probability outputs
- **THEN** isotonic regression is fit on a validation set and applied to transform raw scores into calibrated probabilities

#### Scenario: Calibration impact measurement

- **WHEN** calibration is applied to LightGBM
- **THEN** the system reports the change in Brier score and log-loss before and after calibration

### Requirement: Graph-derived feature extraction

The system SHALL extract graph-derived features from the transaction graph including node degree (equivalently, the count of incident transaction edges), community membership labels, and random walk structural encoding (RWSE) values.

#### Scenario: Graph structure construction

- **WHEN** IEEE-CIS transaction data is loaded
- **THEN** the system builds a bipartite graph with users and merchants as nodes and transactions as edges, computing node degree for each node

#### Scenario: RWSE feature computation

- **WHEN** graph structure is available
- **THEN** the system computes, per node, the diagonal of the k-step random-walk return probability matrix for k = 1..8 and appends this 8-dimensional vector to the node feature matrix

### Requirement: Graph-aware LightGBM

The system SHALL extend the LightGBM baseline by incorporating graph-derived features, enabling a comparison between tabular and graph-aware approaches.

#### Scenario: Graph features injection

- **WHEN** graph-aware LightGBM is trained
- **THEN** features derived from the graph structure (node degree — equivalently incident edge count — community membership, RWSE values) are included in the feature matrix

#### Scenario: Tabular vs graph comparison

- **WHEN** both minimal LightGBM and graph-aware LightGBM are trained and evaluated
- **THEN** the system reports the delta in PR-AUC, Brier score, and Precision@K between the two versions

### Requirement: Stop-early decision criteria

The system SHALL flag the graph feature investigation as "minimal value" if the delta PR-AUC between graph-aware and minimal LightGBM is less than 0.01 or not statistically significant (p-value > 0.05).

#### Scenario: Minimal graph value detected

- **WHEN** ΔPR-AUC < 0.01 between graph-aware and minimal LightGBM on the same test set
- **THEN** the system flags Phase 1 as "graph adds minimal value" and outputs a stop decision recommendation

#### Scenario: Statistical significance check

- **WHEN** two LightGBM models are compared on the same test set
- **THEN** the system performs a statistical significance test (paired permutation test or McNemar's test) and reports the p-value

### Requirement: Mini-lesson delivery

The system SHALL deliver a set of mini-lessons under `phases/phase01_lightgbm_baseline/lessons/` that teach each concept step by step. Each lesson must follow the 5-part structure: Explain (concept description), Code (working example), Visualize (plots or charts), Practice (exercise for the reader), and Solution (completed exercise with results).

#### Scenario: Lesson structure validation

- **WHEN** a lesson file is authored
- **THEN** it contains all five sections (Explain, Code, Visualize, Practice, Solution) in that order

#### Scenario: Lesson code is runnable

- **WHEN** code blocks in a lesson are executed
- **THEN** they run successfully without errors using the environment defined in `phases/phase01_lightgbm_baseline/`

#### Scenario: Evaluation metrics and class imbalance lesson

- **WHEN** the lesson set is delivered
- **THEN** it includes a foundational lesson on imbalanced-classification evaluation covering the precision/recall trade-off, PR-AUC vs ROC-AUC, Precision@K, and class weighting (`scale_pos_weight`)

#### Scenario: Data leakage and split strategy lesson

- **WHEN** the lesson set is delivered
- **THEN** it includes a lesson on data leakage and train/validation/test splitting, covering why graph features are built from the training split only and why time-based splits matter for `TransactionDT`

### Requirement: Phase directory with interactive notebooks

The system SHALL provide a self-contained phase directory at `phases/phase01_lightgbm_baseline/` containing Jupyter notebooks under `notebooks/` that reference code from `src/phase01_lightgbm_baseline/` via imports. The phase must include a small synthetic dataset for fast experimentation (<5 minutes to run end-to-end).

#### Scenario: Notebook imports from phase source

- **WHEN** a phase notebook imports from the project
- **THEN** it imports the `LightGBMWrapper` and `GraphFeatureExtractor` classes from the installed `phase01_lightgbm_baseline` package (not duplicate copies)

#### Scenario: Synthetic sample for phase

- **WHEN** the phase data is loaded
- **THEN** it is a stratified sample small enough to run end-to-end in under 5 minutes on a standard laptop

### Requirement: Isolated environment with uv

The system SHALL provision an isolated Python environment using `uv` at `phases/phase01_lightgbm_baseline/`. The environment must be defined by a `pyproject.toml` with pinned dependencies and a `uv.lock` file for reproducible installs. Users must be able to activate the environment with `uv sync` and run notebooks with `uv run`.

#### Scenario: Environment reproducibility

- **WHEN** a user runs `uv sync` in `phases/phase01_lightgbm_baseline/`
- **THEN** the exact same package versions specified in `uv.lock` are installed

#### Scenario: Notebooks run in the isolated environment

- **WHEN** a phase notebook is launched via `uv run jupyter notebook` from the phase directory
- **THEN** the notebook executes using the pinned dependencies without needing to install anything globally

### Requirement: Persisted evaluation artifacts

The system SHALL persist the phase's evaluation results to `phases/phase01_lightgbm_baseline/results/` so that later phases can consume them without re-running training. It SHALL always compute Precision@K using K equal to the top 1% of the test set (`ceil(0.01 * len(test_set))`).

#### Scenario: Artifacts written after a run

- **WHEN** the end-to-end run script completes
- **THEN** it writes `results/metrics.json` (both models' metrics, deltas, p-value, and stop decision), `results/model.txt`, `results/calibrator.pkl`, and a comparison table or plot

#### Scenario: Precision@K always reported

- **WHEN** metrics are computed for either model
- **THEN** `compute_metrics` is called with `k = ceil(0.01 * len(test_set))` and `precision_at_k` is present in the persisted results
