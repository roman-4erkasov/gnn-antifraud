## 1. Phase skeleton and environment

- [ ] 1.1 Create the phase tree `phases/phase11_temporal_link_prediction/{src/phase11_temporal_link_prediction,tests,lessons,exercises,notebooks,results}` and verify all directories exist
- [ ] 1.2 Write `pyproject.toml` (src-layout, hatch `packages=["src/phase11_temporal_link_prediction"]`, deps `catboost`, `py-tgb[torch]`, `torch`, PyG, `lightgbm`, `xgboost`, `networkx`, `pandas`, `numpy`, `scipy`, `scikit-learn`, optional `faiss-cpu`, dev `pytest`/`pytest-cov`) and verify `uv lock` then `uv sync` succeed
- [ ] 1.3 Add `__init__.py` with re-exports and verify `uv run python -c "import phase11_temporal_link_prediction"` succeeds
- [ ] 1.4 Add `[tool.pytest.ini_options]` (`testpaths`, `pythonpath=src`) and stub `tests/conftest.py`, then verify `uv run pytest -q` collects without errors
- [ ] 1.5 Write phase `README.md` (setup, data, run flags, layout) and verify it documents `uv sync` and the run command

## 2. Dataset interface and loaders

- [ ] 2.1 Implement `datasets.py` common interface yielding time-stamped directed edges `(src, dst, t[, attrs])` with schema validation; verify a unit test rejects malformed edges and accepts valid ones
- [ ] 2.2 Implement the deterministic synthetic generator (configurable node count, horizon density, transient edges) and verify two runs with the same seed produce identical edges
- [ ] 2.3 Implement the `sx-mathoverflow` adapter (download/cache under `data/`, parse `src dst timestamp`) and verify parsing against a small fixture edge list
- [ ] 2.4 Implement the `tgbl-coin` adapter via `py-tgb` with subsampling and license/version recording, and verify it loads a tiny cached sample or skips cleanly when the dataset is unavailable
- [ ] 2.5 Implement the dataset registry/loader dispatch for `{synthetic, sx-mathoverflow, tgbl-coin}` with explicit download instructions and a `FileNotFoundError` on missing real data (no silent synthetic substitution); verify the missing-file error path with a test
- [ ] 2.6 Support optional edge-attribute passthrough and verify attributes survive loading when present and are absent when not provided

## 3. Temporal splitting, query building, and labeling

- [ ] 3.1 Implement chronological train < validation < test splitting and verify every train event precedes every validation event precedes every test event
- [ ] 3.2 Implement the horizon labeler for novel pairs (positive if any appearance in `(t, t+H]`, transient counts, historical pairs excluded); verify with tests for transient positive, historical exclusion, and absent negative
- [ ] 3.3 Implement query building from train/validation reference times, excluding all historical neighbors of the query node; verify no positive/negative candidate is a prior neighbor
- [ ] 3.4 Implement and use a leakage guard asserting all inputs for time `t` have time `≤ t`; verify a test triggers it when a future edge is injected

## 4. GNN embeddings (leakage-free, inductive)

- [ ] 4.1 Implement GraphSAGE with an unsupervised link-prediction objective (inductive) and verify embedding shapes and that a cold-start node receives an embedding
- [ ] 4.2 Implement the GAE variant (GCN encoder + inner-product decoder) and verify training produces finite embeddings
- [ ] 4.3 Implement node2vec/DeepWalk embeddings and verify output shapes
- [ ] 4.4 Implement RWSE/spectral structural features and verify the per-node vector has the configured dimension
- [ ] 4.5 (Optional) Implement a TGN-memory embedder guarded by availability and verify a smoke test
- [ ] 4.6 (Optional) Implement a SASRec/GRU4Rec sequence embedder guarded by availability and verify a smoke test
- [ ] 4.7 Implement embedding persistence/loading keyed by reference interval and verify round-trip equality
- [ ] 4.8 Add a leakage test showing embeddings for `t` are unchanged when only post-`t` edges are modified

## 5. Retrieval and candidate generation

- [ ] 5.1 Define the `Retriever` protocol (`top_m(query) -> ranked candidates`) and verify a dummy retriever conforms
- [ ] 5.2 Implement the embedding-similarity retriever (dot/cosine) with `faiss` when importable and a NumPy fallback; verify both paths return top-M
- [ ] 5.3 Implement the structural retriever (preferential attachment / common neighbors / Adamic-Adar / personalized PageRank) and verify scores on a known toy graph
- [ ] 5.4 Implement the I2I co-occurrence retriever (candidates = neighbors of recent neighbors) and verify it excludes the query node itself
- [ ] 5.5 Implement the sequential retriever (TGN memory or SASRec) trained on per-node interaction history and verify it returns top-M candidates on a toy sequence
- [ ] 5.6 Implement union retrieval with historical-neighbor and non-novel exclusion; verify no returned candidate was a prior neighbor
- [ ] 5.7 Implement negative sampling (random, popularity-weighted, iteratively mined hard negatives) with optional `logQ` correction; verify each mode produces the configured ratio and no false negatives
- [ ] 5.8 Implement `Candidate Recall@M` and verify on synthetic data with planted new links that full recall is 1.0 when M covers all candidates

## 6. Interaction features

- [ ] 6.1 Implement embedding-interaction features (endpoint embeddings, cosine, dot, Hadamard, absolute difference, L2) and verify shapes and finite values
- [ ] 6.2 Implement structural features (degrees, common neighbors, Jaccard, Adamic-Adar, resource allocation, preferential attachment, personalized PageRank, short path counts) and verify against a hand-computed toy graph
- [ ] 6.3 Implement temporal/activity features (recency-weighted degree, activity rates, time since last interaction) and verify they use only `≤ t` data
- [ ] 6.4 Implement retrieval-meta features (candidate rank/score, contributing retriever) and verify they join correctly onto candidates
- [ ] 6.5 Implement feature-table assembly for a query set with a leakage assertion and verify a test that mutating future edges leaves features for `t` unchanged

## 7. Ranking models

- [ ] 7.1 Define the `Ranker` protocol (`fit`, `predict`, `feature_importances`) and verify a dummy ranker conforms
- [ ] 7.2 Implement the target CatBoost listwise ranker (`YetiRank`, `group_id=(u,t)`) and verify it trains and beats a random ordering on synthetic data
- [ ] 7.3 Implement the CatBoost pointwise `Logloss` variant and verify it produces probabilities in `[0,1]`
- [ ] 7.4 Implement the CatBoost pairwise `PairLogit` variant and verify it trains with the same features and grouping
- [ ] 7.5 Implement the LightGBM `lambdarank` ranker and verify it trains and ranks
- [ ] 7.6 Implement the XGBoost `rank:pairwise` ranker with a fallback if the parameter name differs across versions; verify it trains or reports a clear unavailable message
- [ ] 7.7 Implement training-free heuristic rankers (popularity, common neighbors, Adamic-Adar, resource allocation) and verify they produce rankings without fitting
- [ ] 7.8 Implement the retrieval-similarity-only ranker (no reranking) and verify it returns the retriever order
- [ ] 7.9 Implement the logistic-regression ranker and verify it trains on the numeric features
- [ ] 7.10 Implement one end-to-end GNN ranker (GraphSAGE encoder + dot-product decoder trained on the link objective) and verify it trains and scores pairs
- [ ] 7.11 Implement feature-importance extraction for the GBDT rankers and verify the importances sum to positive values

## 8. Evaluation

- [ ] 8.1 Implement `Hits@K`, `Recall@K`, `MRR`, `MAP`, and `NDCG@K` per query and verify against hand-computed rankings
- [ ] 8.2 Report both conditional (reranker-on-shortlist) and end-to-end metrics and verify a test distinguishes them when retrieval drops a positive
- [ ] 8.3 Integrate `paired_permutation_test` from `src/utils/metrics.py` for model comparison and verify it returns a p-value on synthetic score arrays
- [ ] 8.4 Implement the horizon sweep over configured `H` values and verify metrics are produced per horizon
- [ ] 8.5 Implement ablations over feature groups and objectives and verify each configuration produces a result row
- [ ] 8.6 Implement per-domain result reporting (generic vs transaction) and verify datasets are reported separately
- [ ] 8.7 (Optional) Add a cross-check of MRR against TGB's `Evaluator` on `tgbl-coin` and verify it runs or is cleanly skipped

## 9. Orchestration, CLI, and persistence

- [ ] 9.1 Implement `run_link_prediction.py` with flags `--dataset {synthetic,sx-mathoverflow,tgbl-coin}`, `--horizon`, `--retrieval-size`, `--subsample`, `--seed`, `--output-dir`, `--objectives`; verify `--help` and a tiny synthetic run complete
- [ ] 9.2 Persist ranking metrics, feature importances, run configuration, and comparison tables/plots to the output directory; verify all artifact files exist after a run
- [ ] 9.3 Verify determinism: two runs with the same seed and config produce identical metrics artifacts

## 10. Tests and fixtures

- [ ] 10.1 Add `conftest.py` fixtures (tiny synthetic edge set, planted new links, temporary data dir) and verify fixtures load
- [ ] 10.2 Add leakage regression tests across loader, features, and embeddings and verify they pass
- [ ] 10.3 Run the full test suite with `uv run pytest -q` and verify all tests pass

## 11. Mini-lessons (lessons/)

- [ ] 11.1 Write lesson `01-introduction-to-temporal-link-prediction.md` (Explain/Code/Visualize/Practice/Solution) covering the novel-link horizon task; verify the 5-part headings exist
- [ ] 11.2 Write lesson `02-ranking-evaluation-metrics.md` covering Hits@K, Recall@K, MRR, MAP, NDCG, and Candidate Recall@M; verify the 5-part headings and a runnable code block
- [ ] 11.3 Write lesson `03-retrieval-and-candidates.md` covering two-stage retrieval (embedding, structural, I2I, sequential) and negative sampling; verify the 5-part headings
- [ ] 11.4 Write lesson `04-embeddings-without-leakage.md` covering graph embeddings and temporal leakage; verify the 5-part headings
- [ ] 11.5 Write lesson `05-interaction-features.md` covering embedding/structural/temporal features; verify the 5-part headings
- [ ] 11.6 Write lesson `06-rankers-pointwise-pairwise-listwise.md` comparing pointwise, pairwise, and listwise GBDT objectives; verify the 5-part headings
- [ ] 11.7 Write lesson `07-horizon-and-temporal-drift.md` covering horizon sweeps and changing neighbor sets; verify the 5-part headings

## 12. Interactive exercises (exercises/)

- [ ] 12.1 Create `01-target-and-horizon.ipynb` with a task to build labels for two horizons; verify it executes
- [ ] 12.2 Create `02-metric-implementation.ipynb` with a task to implement and check a ranking metric; verify it executes
- [ ] 12.3 Create `03-retriever-comparison.ipynb` with a task to compare retrievers via Candidate Recall@M; verify it executes
- [ ] 12.4 Create `04-negative-sampling.ipynb` with a task to compare random vs popularity vs hard negatives; verify it executes
- [ ] 12.5 Create `05-objective-comparison.ipynb` with a task to compare pointwise/pairwise/listwise rankers; verify it executes
- [ ] 12.6 Create `06-feature-ablation.ipynb` with a task to ablate feature groups; verify it executes

## 13. Notebooks (notebooks/)

- [ ] 13.1 Create `01-explore-dataset.ipynb` visualizing degree/time distributions for a loaded dataset; verify it executes
- [ ] 13.2 Create `02-embeddings-demo.ipynb` visualizing embeddings and their similarity; verify it executes
- [ ] 13.3 Create `03-ranking-results.ipynb` loading a run's `results/` and plotting model comparison; verify it executes

## 14. Final verification

- [ ] 14.1 Run the end-to-end pipeline on the synthetic dataset with a small horizon and verify metrics and artifacts are produced
- [ ] 14.2 Run on a small `sx-mathoverflow` subsample and verify generic-domain metrics and artifacts
- [ ] 14.3 (Optional, requires network) Run on a `tgbl-coin` subsample and verify transaction-domain metrics with license/version recorded
- [ ] 14.4 Run `openspec validate 11-temporal-link-prediction --type change` and verify the change validates
