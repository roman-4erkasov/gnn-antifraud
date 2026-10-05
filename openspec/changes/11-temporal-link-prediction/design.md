## Context

See `proposal.md` - Why. This design covers the *how*.

Current state that shapes the approach:

- Phases 01-10 are fraud-detection oriented and mostly implemented as self-contained, uv-installable packages under `phases/phaseNN_<name>/`, each with `run_<name>.py`, a `src/<pkg>/` package, `tests/`, `lessons/`, `exercises/`, `notebooks/`, and `results/`. Only phase 01 currently has code.
- Shared infrastructure already exists at the repository root: `src/utils/metrics.py` (`compute_metrics`, `compute_delta_metrics`, `paired_permutation_test`), `src/utils/calibration.py`, and `src/utils/features.py`; data lives under `data/`.
- There are no durable specs (`openspec list --specs` is empty) and no existing link-prediction or candidate-ranking capability. Phases 04a/05/05b/06 explain or classify existing structures; none rank future links.
- This phase is intentionally **independent** of phases 02-10. It depends only on shared infrastructure (`src/utils/`, `data/`), which is not a phase.

## Goals / Non-Goals

**Goals:**

- Implement the target defined in `specs/phase11-temporal-link-prediction/spec.md`: temporal novel-link ranking within a horizon `H`.
- Compare the CatBoost listwise target against pointwise/pairwise GBDT objectives and a broad baseline set on two real dynamic graphs plus a synthetic generator.
- Keep retrieval, embeddings, features, ranking, and evaluation separable and independently measurable.

**Non-Goals:**

- No dependency on, or modification of, phases 02-10.
- No fraud-label use: datasets may carry fraud labels, but they are ignored for this task.
- No distributed/large-scale serving work (that is phases 08-10); `tgbl-coin` is subsampled.
- No requirement to reproduce a specific published leaderboard number.

## Decisions

### Decision 1: Self-contained phase that reuses only shared infrastructure
**Choice:** New package `phases/phase11_temporal_link_prediction/`, importable as `phase11_temporal_link_prediction` (src-layout, hatch `packages=["src/phase11_temporal_link_prediction"]`), reusing root `src/utils/` and `data/` only.
**Rationale:** Keeps the phase independent and consistent with the project's per-phase anatomy and the phase-01 convention.
**Alternatives considered:** Adding to a later phase (violates independence); a new top-level `src/` tree (breaks per-phase convention).

### Decision 2: Target is novel-link appearance in a horizon
**Choice:** Target = pairs `(u,v)` never observed before `t`, ranked by `P(link appears in (t, t+H])`; transient appearances are positive; historical neighbors excluded.
**Rationale:** Matches the user's stated need and standard new-edge link-prediction protocols (OGB/TGB), giving a clean, testable label.
**Alternatives considered:** "Any interaction in the window" (A) degenerates into retention; "separate repeat vs new" (C) adds scope without a stated need.

### Decision 3: Common dataset interface over three datasets
**Choice:** A `datasets.py` interface yields time-stamped directed edges `(src, dst, t[, attrs])` plus a horizon labeler. Adapters: synthetic generator (deterministic), SNAP `sx-mathoverflow`, TGB `tgbl-coin` via `py-tgb` (subsampled, license recorded). Missing files trigger explicit download/instructions rather than silent synthetic substitution for the real datasets.
**Rationale:** One pipeline across domains, and `tgbl-coin` is the applied transaction case while `sx-mathoverflow` gives a larger but dependency-light generic graph.
**Alternatives considered:** Two TGB datasets for a shared evaluator (extra `py-tgb` coupling and a second large download); IEEE-CIS-derived transaction graph (weak structure).

### Decision 4: Temporal, leakage-free splitting
**Choice:** Split by event time into disjoint train < validation < test intervals. Query reference times `t` come from train/validation; final metrics on test. All features, embeddings, and graph structure for `t` use only interactions `≤ t`.
**Rationale:** Prevents future leakage, which is the dominant failure mode in temporal link prediction.
**Alternatives considered:** Random edge split (leaks future); snapshot-at-`t+H` labels (miss transient edges; contradicts target).

### Decision 5: Pluggable two-stage retrieval with its own metric
**Choice:** A `Retriever` protocol produces top-`M` candidates; the default union combines (a) embedding ANN (dot/cosine, `faiss` if available, NumPy fallback), (b) structural scoring (preferential attachment / common neighbors / Adamic-Adar / personalized PageRank), (c) I2I co-occurrence, and (d) a sequential retriever that scores candidates with a sequence model (TGN memory or SASRec) trained on each node's interaction history. Historical neighbors and non-novel pairs are removed after the union. Metrics report `Candidate Recall@M` (ceiling) separately from reranker quality.
**Rationale:** Retrieval bounds end-to-end recall; heuristics are often strong for link prediction and must be first-class.
**Alternatives considered:** Embedding-only retrieval (loses Recall@M of structural candidates); full-pair scoring (infeasible on `tgbl-coin`).

### Decision 6: Frozen, inductive GNN embeddings
**Choice:** Primary GraphSAGE with a self-supervised link-prediction objective (inductive); also support GAE (transductive), node2vec/DeepWalk (shallow), and RWSE/spectral features; optional TGN memory; and optional SASRec/GRU4Rec sequence embeddings backing the sequential retriever. Embeddings are trained per reference interval and frozen.
**Rationale:** Inductive encoding handles cold-start nodes that appear after training; frozen embeddings double as retrieval vectors and ranker features.
**Alternatives considered:** Only transductive embeddings (break on new nodes); joint end-to-end GNN (kept as a separate comparison model, not the target).

### Decision 7: Configurable negative sampling
**Choice:** Training candidates are sampled: random negatives, popularity-weighted negatives, and iteratively mined hard negatives, with optional `logQ` popularity correction. Pos:neg ratio is configurable.
**Rationale:** Random negatives make ranking too easy on power-law graphs; hard negatives + `logQ` mitigate popularity bias (option B).
**Alternatives considered:** All-pair training (infeasible); random-only (optimistic metrics).

### Decision 8: Pair features are embedding + structural + temporal
**Choice:** Feature groups: embedding interactions (endpoint embeddings, cosine, dot, Hadamard, absolute difference, L2), structural (degrees, common neighbors, Jaccard, Adamic-Adar, resource allocation, preferential attachment, personalized PageRank, short path counts), temporal/activity (recency-weighted degree, activity, time since last interaction), and retrieval-meta (candidate rank/score and contributing retriever). All computed from data `≤ t`.
**Rationale:** Gives the GBDT both learned and training-free signals; retrieval-meta is often predictive.
**Alternatives considered:** Embeddings-only features (weaker, and duplicates the end-to-end GNN comparison).

### Decision 9: CatBoost listwise target with pointwise/pairwise comparatives
**Choice:** Target = CatBoost `YetiRank` (listwise), grouped by query `(u,t)`. Comparatives = CatBoost `Logloss` (pointwise, calibrated) and `PairLogit` (pairwise), same features and grouping. Full comparison set: LightGBM `lambdarank`, XGBoost `rank:pairwise`, heuristic rankers (popularity, common neighbors, Adamic-Adar, resource allocation), retrieval-similarity-only, logistic regression, and one end-to-end GNN. Feature importances are persisted.
**Rationale:** Listwise directly optimizes per-query ordering; the pointwise variant provides calibrated probabilities; the breadth checks that the GBDT actually beats heuristics (a leakage/bug sanity check).
**Alternatives considered:** Pointwise-only target (weaker top-K); a single objective (loses the target/comparison structure the user asked for).

### Decision 10: Ranking evaluation with significance and optional TGB cross-check
**Choice:** Per query, compute `Hits@K`, `Recall@K`, `MRR`, `MAP`, `NDCG@K`, plus `Candidate Recall@M`; report conditional (reranker-on-shortlist) and end-to-end. Use `paired_permutation_test` from `src/utils/metrics.py` for significance; optionally cross-check MRR with TGB's `Evaluator` on `tgbl-coin`.
**Rationale:** Reuses existing project utilities and separates retrieval loss from ranking loss.
**Alternatives considered:** A single aggregate metric (hides where loss occurs); skipping significance (no robust model comparison).

### Decision 11: Horizon sweep, ablations, and per-domain reporting
**Choice:** `H` is configurable and swept; ablations over feature groups and objectives; results reported separately per dataset.
**Rationale:** Directly answers "how does the set of probable neighbors change as the period changes" and shows domain transfer.
**Alternatives considered:** Single fixed horizon (does not address the user's variant).

### Decision 12: Persisted artifacts and CLI
**Choice:** `run_link_prediction.py` with flags `--dataset {synthetic,sx-mathoverflow,tgbl-coin}`, `--horizon`, `--retrieval-size` (top-M), `--subsample`, `--seed`, `--output-dir`, `--objectives`. Persist metrics, feature importances, run config, and comparison tables/plots to `results/`.
**Rationale:** Reproducibility and inspectability, consistent with phase-01's `results/` pattern.
**Alternatives considered:** Notebook-only execution (not reproducible/CI-friendly).

### Decision 13: Phase package layout
**Choice:** `src/phase11_temporal_link_prediction/{__init__,datasets,data_loader,embeddings,candidates,features,ranker,evaluation}.py`, with `tests/`, `lessons/`, `exercises/`, `notebooks/`, `results/`.
**Rationale:** One module per responsibility, matching the seam lines of the pipeline.
**Alternatives considered:** A single monolithic module (harder to test and extend).

### Decision 14: Optional ANN backend with a pure-NumPy fallback
**Choice:** Embedding retrieval uses `faiss` when it is importable and falls back to a NumPy/`scikit-learn` brute-force nearest-neighbor path otherwise; `faiss-cpu` remains an optional extra, not a required dependency.
**Rationale:** Keeps installs light and CI portable while allowing fast ANN on larger graphs; both paths return equivalent top-M candidates.
**Alternatives considered:** A hard `faiss-cpu` dependency (heavier installs, platform friction); NumPy-only (too slow on `tgbl-coin`).

### Decision 15: XGBoost pairwise objective fallback
**Choice:** The XGBoost comparison model uses `objective="rank:pairwise"` when supported by the installed version, otherwise `objective="rank:ndcg"`; if neither is available it reports a clear "unavailable" status. The pairwise comparison remains covered by LightGBM `lambdarank`.
**Rationale:** `rank:pairwise` naming and availability vary across XGBoost releases; the comparison intent is pairwise/listwise ranking, which `rank:ndcg` and LightGBM already provide.
**Alternatives considered:** Pinning a specific XGBoost version (maintenance burden); dropping XGBoost (loses a sibling GBDT comparison).

## Risks / Trade-offs

- [Very sparse novel links / low recall] → report `Candidate Recall@M`, use retrieval union, and evaluate on a sample of query nodes when the graph is large.
- [Popularity bias from retrieval and negatives] → include popularity-weighted and hard negatives, optional `logQ` correction, and always show heuristic baselines.
- [`tgbl-coin` is large (~22.8M edges)] → subsample by time/active-nodes/edges; record parameters; default to the synthetic dataset for tests and CI.
- [Dataset downloads require network and a CC BY-NC license] → surface download instructions, cache under `data/`, and record license/version in run outputs.
- [Future leakage through embeddings/features] → enforce `≤ t` computation in `data_loader`/`features`/`embeddings`, and add explicit leakage tests.
- [Groupwise objectives can be unstable or need tuning] → keep the pointwise and pairwise variants and heuristic baselines; early-stop on validation NDCG.
- [Cold-start nodes have no common neighbors] → GraphSAGE inductive encoding plus union retrieval (ANN/I2I) compensates.
- [Sequential retriever adds training/inference cost] → keep it optional and off by default on large datasets, guard by model availability, and record whether it was enabled per run.

## Migration Plan

New, additive phase - no migration or rollback of existing behavior required.

Implementation order (carried into `tasks.md`):

1. Package skeleton, pyproject/uv.lock, README, CLI stub.
2. `datasets.py` + `data_loader.py`: synthetic generator, SNAP and TGB adapters, temporal split, horizon labeler.
3. `embeddings.py`: GraphSAGE primary (+ GAE/node2vec/RWSE, optional TGN/SASRec), inductive, leakage-guarded.
4. `candidates.py`: `Retriever` protocol, union retrieval (embedding/structural/I2I/sequential), negative sampling, `Candidate Recall@M`.
5. `features.py`: embedding/structural/temporal/retrieval-meta features.
6. `ranker.py`: CatBoost listwise target + pointwise/pairwise comparatives; sibling and heuristic baselines.
7. `evaluation.py`: metrics, conditional vs end-to-end, significance, per-domain; horizon sweep and ablations.
8. `run_link_prediction.py`: orchestration, CLI flags, persisted artifacts.
9. Tests, mini-lessons, exercises, notebooks.

## Open Questions

- Final subsample size for `tgbl-coin` that keeps runs tractable; decided empirically during implementation and recorded per run.
- Pinned versions for `catboost`, `torch`, PyG, and `py-tgb[torch]` at lock time.
