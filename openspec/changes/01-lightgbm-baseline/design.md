## Context

The project has a running skeleton with shared utilities already implemented:
- `src/utils/calibration.py` — `IsotonicCalibrator` with `fit`/`predict`/`fit_predict` and conformal prediction intervals.
- `src/utils/metrics.py` — `compute_metrics` (PR-AUC, ROC-AUC, Brier, log-loss, Precision@K), `compute_delta_metrics`, and `paired_permutation_test`.

This change adds a third shared utility, `src/utils/features.py`, providing `prepare_baseline_features()` for the 3-feature baseline table.

Phase 01 is a self-contained phase under `phases/phase01_lightgbm_baseline/`. No existing scoring modules need to be modified. The phase package is installed and imported as `phase01_lightgbm_baseline` (hatch `packages = ["src/phase01_lightgbm_baseline"]`), consistent with phases 02+.

## Goals / Non-Goals

**Goals:**
- Create a working LightGBM baseline pipeline under `phases/phase01_lightgbm_baseline/src/phase01_lightgbm_baseline/lightgbm_baseline.py` that consumes the baseline feature table produced by `src/utils/features.py:prepare_baseline_features()`.
- Create a graph feature extraction module under `phases/phase01_lightgbm_baseline/src/phase01_lightgbm_baseline/graph_features.py` that builds a bipartite user-merchant graph from IEEE-CIS data and computes degree, community labels, and RWSE features.
- Wire calibration (using existing `IsotonicCalibrator`) and metrics (using existing `compute_metrics`, `paired_permutation_test`) into an end-to-end runnable script.
- Produce a comparison report: minimal LightGBM vs graph-aware LightGBM with delta metrics and statistical significance.
- Deliver a self-contained phase under `phases/phase01_lightgbm_baseline/` with lessons, notebooks, and isolated environment.
- Build interactive notebooks under `phases/phase01_lightgbm_baseline/notebooks/` for hands-on experimentation.
- Provision an isolated environment using `uv` with `pyproject.toml` and `uv.lock` at the phase root.

**Non-Goals:**
- Hyperparameter optimization or automated tuning.
- Building an actual GNN model (reserved for later phases).
- Production-grade serving infrastructure.
- Data preprocessing beyond what's needed for the 3-feature baseline and graph features.
- Distributed computing or cloud infrastructure.

## Decisions

### Decision 1: Self-contained phase structure
**Choice:** Code lives in `phases/phase01_lightgbm_baseline/src/phase01_lightgbm_baseline/lightgbm_baseline.py` and `phases/phase01_lightgbm_baseline/src/phase01_lightgbm_baseline/graph_features.py`.
**Rationale:** Each phase is self-contained with its own dependencies, source code, tests, lessons, and notebooks. The wrapper consumes DataFrames from the loader, handles the calibration pipeline, and returns calibrated probabilities. Graph features live in the same phase directory, keeping data loading separate from graph computation.
**Alternatives considered:**
- Shared `src/scoring/` directory — violates phase isolation principle.
- Adding graph extraction to `data_loader.py` — would mix data loading and graph computation concerns.
- Centralized `src/data/graph_extractor.py` — would duplicate functionality across phases.

### Decision 2: Graph construction uses NetworkX, not PyTorch Geometric
**Choice:** Use `networkx` for graph construction, degree computation, community detection (Louvain via `community_louvain`), and RWSE feature computation.
**Rationale:** Phase 1 is about establishing a tabular baseline with graph-derived features, not about training GNNs. NetworkX is lighter, easier to debug, and sufficient for computing node features (degree — equivalently incident edge count — clustering coefficient, Louvain community labels, and RWSE, defined as the diagonal of k-step random-walk return probabilities for k = 1..8). PyG will be introduced in later GNN phases.
**Alternatives considered:**
- PyG `Data` objects — overkill for feature extraction without GNN training.
- `igraph` — comparable, but NetworkX has broader ecosystem integration.

### Decision 3: Model training via a thin wrapper around LightGBM
**Choice:** Create `LightGBMWrapper` class in `phases/phase01_lightgbm_baseline/src/phase01_lightgbm_baseline/lightgbm_baseline.py` with `train`, `predict`, `calibrate` methods, using default LightGBM parameters with class-weight adjustment (`scale_pos_weight` computed from training data).
**Rationale:** A thin wrapper keeps the code simple, testable, and focused on the experiment (comparing features, not tuning). The wrapper consumes DataFrames from the loader, handles the calibration pipeline, and returns calibrated probabilities. It also computes metrics on both raw and calibrated probabilities so the calibration impact (ΔBrier, Δlog-loss) can be reported.
**Alternatives considered:**
- Direct `lightgbm.train()` calls — less reusable, harder to test.
- `sklearn`-compatible `LGBMClassifier` — fine alternative, but `lgb.train()` gives more control over early stopping and custom metrics.

### Decision 4: Calibration split — 60/20/20 (train/val/test)
**Choice:** Split data into 60% train, 20% validation (shared by early stopping and calibration), 20% test (for final evaluation).
**Rationale:** Calibration requires held-out data to avoid overfitting, and the existing `IsotonicCalibrator` needs a validation set separate from training. The same validation split is used both for LightGBM early stopping (`binary_logloss`) and for fitting the calibrator; the test set remains untouched until final evaluation. Reusing one validation split keeps the Phase 1 pipeline simple and its data usage easy to reason about.
**Alternatives considered:**
- 80/20 (train/test) with calibration on test — risks overfitting calibration.
- 50/25/25 — less training data, not worth the marginal calibration improvement.

### Decision 5: Statistical significance via paired permutation test (existing)
**Choice:** Use the existing `paired_permutation_test` from `src/utils/metrics.py` for comparing model predictions.
**Rationale:** The function already implements the required paired permutation test with configurable permutations and seed. No need to reimplement. It works on per-sample score differences, making it appropriate for comparing model predictions on the same test set.

### Decision 6: Mini-lesson structure
**Choice:** Each lesson follows a 5-part pattern: Explain (concept description) -> Code (working example) -> Visualize (plots/charts) -> Practice (exercise for the reader) -> Solution (completed exercise with results).
**Rationale:** This pattern is proven in technical education (see "Explain-Do-Review" model). It keeps learners engaged by alternating theory and practice. Lessons are written as markdown files that can be rendered in GitHub/GitLab and read directly in any text editor.
**Alternatives considered:**
- Video lectures — harder to update, not searchable, not version-controlled.
- Interactive course platforms (Coursera, DataCamp) — external dependency, not integrated with the repo.
- Single monolithic document — hard to navigate, no modular exercises.

### Decision 7: Phase architecture
**Choice:** Each phase is self-contained at `phases/phase01_lightgbm_baseline/` with `pyproject.toml` at the phase root, `src/`, `tests/`, `lessons/`, and `notebooks/` subdirectories.
**Rationale:** Self-contained phases are easier to manage, copy, or remove independently. Each phase has its own dependencies (some phases may need Hadoop, others may not). Using `pyproject.toml` at the phase root follows standard Python project conventions. Lessons and notebooks live with the code they teach, making the phase a complete learning unit.
**Alternatives considered:**
- Separate `sandbox/` and `environments/` directories — scatters phase artifacts across the project, harder to manage.
- Single shared environment for all phases — mixes dependencies (e.g., Hadoop for some phases), harder to isolate.
- `environment/` subdirectory — non-standard, adds unnecessary nesting.

### Decision 8: Environment isolation with uv
**Choice:** Use `uv` (Astral) for environment management. Create `phases/phase01_lightgbm_baseline/pyproject.toml` with pinned dependencies, then generate `uv.lock` via `uv lock`. Users activate the environment with `uv sync` from the phase directory.
**Rationale:** `uv` is significantly faster than conda/pip for dependency resolution and environment creation. It's compatible with standard pyproject.toml, works on Linux/Mac/Windows, and produces deterministic builds via `uv.lock`. Each phase has its own environment, allowing different dependencies per phase (e.g., Hadoop for some phases).
**Alternatives considered:**
- conda — slower, heavier, but better for non-Python system dependencies. Not needed here since all dependencies are pure Python.
- pip + venv — works but lacks fast dependency resolution and lockfile generation.
- Docker — overkill for a local learning sandbox, adds unnecessary complexity.
- `poetry` — similar to uv but slower and heavier; uv is the newer, faster alternative.

### Decision 9: Data storage and loading strategy
**Choice:** Store raw datasets in shared `data/` directory at project root (e.g., `data/ieee-cis/`). Each phase implements its own `data_loader.py` that validates the dataset is present in `data/` and prints download instructions if missing.
**Rationale:** Datasets are large and shared across multiple phases (IEEE-CIS used by phases 01-04). Storing once in `data/` avoids duplication. Each phase's `data_loader.py` encapsulates loading logic specific to that phase's needs. Validating presence before use enables offline work and prevents redundant or fabricated data.
**Alternatives considered:**
- Each phase stores its own copy of data — wastes disk space, harder to keep in sync.
- Centralized `src/data/` with all loaders — violates phase isolation principle.
- Download data on every run — slow, requires network, not reproducible.
- Generating synthetic data as a stand-in for the real dataset — masks missing data and makes results non-comparable across phases.

### Decision 10: Interactive exercises with linked notebooks
**Choice:** Each lesson's Practice section links to an interactive Jupyter notebook in `exercises/` directory.
**Rationale:** Practice exercises in markdown are static text. Linking to notebooks allows users to open, run, and modify code directly. Minimal template: setup code, task description, empty code cell for user solution, solution in markdown code block.
**Alternatives considered:**
- All exercises in one notebook — harder to navigate, no clear mapping to lessons
- Exercises embedded in lesson notebooks — mixes demonstration and practice
- Practice only in markdown — requires copy-paste, less interactive

### Decision 11: Graph-aware training entrypoint
**Choice:** Provide `train_graph_aware(data, graph_extractor)` in `phases/phase01_lightgbm_baseline/src/phase01_lightgbm_baseline/pipeline.py`. `data` is the `(train_df, test_df)` tuple returned by the loader and `graph_extractor` is a configured `GraphFeatureExtractor`. The function builds the graph, extracts and joins features, trains a `LightGBMWrapper`, and returns `(model, metrics_dict)`.
**Rationale:** Keeps the run script thin and makes graph-aware training testable and reusable by later phases, instead of inlining the whole pipeline in `run_lightgbm_baseline.py`.
**Alternatives considered:**
- Inline in the run script — not importable, untestable, duplicated across phases.
- Method on `GraphFeatureExtractor` — would make the feature module depend on the LightGBM wrapper.
- Method on `LightGBMWrapper` — would force the baseline model module to know about graphs.

### Decision 12: Persisted evaluation artifacts
**Choice:** The run script writes `results/metrics.json` (both models' metrics, deltas, p-value, stop decision), `results/model.txt` (LightGBM model dump), `results/calibrator.pkl` (fitted `IsotonicCalibrator`), and a comparison table/plot under `phases/phase01_lightgbm_baseline/results/`. `compute_metrics` is always called with `k = ceil(0.01 * len(test_set))`, so Precision@K is always present.
**Rationale:** Later phases (Phase 02 consumes `lgb_minimal_metrics` and `lgb_graph_metrics`) need Phase 1's results without re-running training. Persisting artifacts makes phases composable and keeps the 3-feature baseline a fixed reference point.

### Decision 13: Foundational lessons on evaluation metrics and leakage
**Choice:** Add two foundational lessons beyond the per-artifact walkthrough: `00-evaluation-metrics-and-imbalance.md` (precision/recall trade-off, PR-AUC vs ROC-AUC, Precision@K, class weighting via `scale_pos_weight`) and `07-data-leakage-and-splits.md` (what leakage is, why graph features are built from the training split only, time-based vs random splits and the role of `TransactionDT`). Each has a matching exercise notebook (`00-metrics-and-imbalance.ipynb`, `07-leakage-and-splits.ipynb`).
**Rationale:** IEEE-CIS is heavily imbalanced (~3.5% fraud) and Phase 1's stop decision hinges on PR-AUC and Precision@K, which are otherwise used but never explained. Likewise, the no-leakage property of graph construction (graph built from the training split only) and the choice of a random 60/20/20 split over a time-based split are central methodology decisions that deserve explicit teaching rather than being implicit.
**Alternatives considered:**
- Fold the material into existing lessons — keeps the file count fixed but buries foundational concepts and overloads lessons 02/06.
- Add more notebooks instead of lessons — no guided Explain/Visualize flow.

## Risks / Trade-offs

| Risk | Mitigation |
|------|-----------|
| IEEE-CIS dataset not available or too large to load | Support `sample_limit` parameter in loader for prototyping; use stratified subsampling for fraud minority class. |
| Graph features computation slow on full dataset | Pre-compute and cache graph features to disk (parquet); use community detection only on the user subgraph, not the full bipartite graph. |
| Calibration overfits on small validation set | Require a minimum validation-set size and enough positive examples; fall back to identity (raw probabilities) or blend the isotonic output with raw scores when the validation set is too small. (`IsotonicCalibrator` intentionally has no smoothing parameter.) |
| Memory issues with dense graph representation | Use sparse adjacency matrices via `scipy.sparse` if memory becomes a constraint; use iterative Louvain for large graphs. |

## Migration Plan

This is a greenfield self-contained phase. No migration needed.

**Deployment order:**
1. Create `src/utils/features.py` with `prepare_baseline_features()` (shared 3-feature baseline engineering).
2. Create `phases/phase01_lightgbm_baseline/pyproject.toml` and run `uv lock` to generate `uv.lock`.
3. Create `phases/phase01_lightgbm_baseline/src/phase01_lightgbm_baseline/` directory with `__init__.py`.
4. Implement `phases/phase01_lightgbm_baseline/src/phase01_lightgbm_baseline/lightgbm_baseline.py` (baseline model + calibration + evaluation).
5. Implement `phases/phase01_lightgbm_baseline/src/phase01_lightgbm_baseline/graph_features.py` (graph construction + feature extraction).
6. Implement `phases/phase01_lightgbm_baseline/src/phase01_lightgbm_baseline/pipeline.py` (`train_graph_aware`) and `phases/phase01_lightgbm_baseline/src/phase01_lightgbm_baseline/comparison.py` (report, significance, stop decision).
7. Create phase structure: `lessons/`, `notebooks/`, `exercises/`, and `tests/`.
8. Add `phases/phase01_lightgbm_baseline/tests/conftest.py` and the synthetic fixture `phases/phase01_lightgbm_baseline/tests/data/ieee_cis_test/`.
9. Write lessons (markdown), notebooks, and exercises (IPYNB) that reference the implementation code.
10. Add tests under `phases/phase01_lightgbm_baseline/tests/`.
11. Create a phase-level run script `phases/phase01_lightgbm_baseline/run_lightgbm_baseline.py` that orchestrates the full pipeline and writes artifacts to `results/` (consistent with the in-phase runner convention used by phases 02, 03, 04a–10).
12. Add `phases/phase01_lightgbm_baseline/README.md` and `results/.gitkeep`.

## Open Questions

- Resolved: graph features are recomputed on each run in Phase 1; caching to parquet is deferred until dataset size makes it necessary.
- Resolved: Precision@K uses K = ceil(0.01 * len(test_set)) (top 1% of the test set) and is always computed and persisted.
- Resolved: RWSE is the diagonal of k-step random-walk return probabilities for k = 1..8.
