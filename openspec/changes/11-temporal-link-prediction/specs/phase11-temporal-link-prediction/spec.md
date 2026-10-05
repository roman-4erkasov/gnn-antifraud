## Purpose

Rank novel links: for a source node at a chosen reference time, rank candidate nodes that have never been connected before by the probability that a link appears within a forecasting horizon, using graph embeddings, structural signals, and gradient-boosted ranking over a pluggable retrieval stage.

## ADDED Requirements

### Requirement: Temporal novel-link prediction target
The phase SHALL define its prediction target as pairs `(u, v)` that have never appeared before reference time `t`, and SHALL rank them by the probability that a link between `u` and `v` appears at least once within the horizon window `(t, t+H]`. A link that appears and disappears inside the window SHALL count as a positive. Pairs already observed before `t` SHALL be excluded from candidate ranking.

#### Scenario: Novel pair appearing transiently
- **WHEN** a pair `(u, v)` has no prior interaction before `t` and has at least one interaction inside `(t, t+H]` that ends before `t+H`
- **THEN** the pair is labeled positive and included in the ranking targets

#### Scenario: Historical pair excluded
- **WHEN** a pair `(u, v)` already had an interaction before `t`
- **THEN** the pair is excluded from the candidate set and never ranked

#### Scenario: Pair absent in the horizon is negative
- **WHEN** a novel pair has no interaction anywhere in `(t, t+H]`
- **THEN** it is treated as a negative

### Requirement: Temporal data splitting
The phase SHALL split data by event time into disjoint train, validation, and test intervals ordered chronologically, with no future information leaking into an earlier interval. Reference times `t` used to build ranking queries SHALL come from the train and validation intervals, and final metrics SHALL be reported on the test interval.

#### Scenario: Chronologically disjoint splits
- **WHEN** the data is split
- **THEN** every training event precedes every validation event, and every validation event precedes every test event

#### Scenario: No future leakage
- **WHEN** any feature, embedding, or graph structure is computed for a reference time `t`
- **THEN** it uses only interactions with time `≤ t`

### Requirement: Supported datasets through a common interface
The phase SHALL support a synthetic edge generator, at least one generic real dynamic graph, and one real transaction dynamic graph, all exposed through a common interface that yields timestamped directed edges with optional edge attributes. It SHALL support `sx-mathoverflow` as the generic dataset and `tgbl-coin` as the transaction dataset. For `tgbl-coin` the phase SHALL support subsampling and SHALL record the dataset license.

#### Scenario: Load dataset as timestamped edges
- **WHEN** any supported dataset is loaded
- **THEN** it is available as time-ordered directed edges with source, destination, timestamp, and any edge attributes

#### Scenario: Synthetic generator is deterministic
- **WHEN** the synthetic generator runs with a fixed seed and parameters
- **THEN** it produces identical edges on repeated runs

#### Scenario: Transaction dataset subsampled
- **WHEN** `tgbl-coin` is used
- **THEN** a subsample can be selected and the applied subsample parameters are recorded in the run outputs

### Requirement: Candidate generation and retrieval
The phase SHALL generate ranking candidates for each query node through a pluggable retrieval stage combining multiple retrievers, including an embedding-similarity retriever, a structural retriever (preferential attachment, common neighbors, Adamic-Adar, or personalized PageRank), a co-occurrence (I2I) retriever, and a sequential retriever scored by a sequence model (TGN memory or SASRec). It SHALL exclude all historical neighbors of the query node and SHALL evaluate retrieval quality independently with `Candidate Recall@M`.

#### Scenario: Union of retrievers
- **WHEN** candidates are generated for a query
- **THEN** candidates are the union of the configured retrievers' top results, with historical neighbors removed
#### Scenario: Retrieval ceiling measured

- **WHEN** end-to-end ranking metrics are reported
- **THEN** `Candidate Recall@M` is reported as the retrieval coverage ceiling

#### Scenario: Sequential retriever included

- **WHEN** the retrieval union is configured with the sequential retriever
- **THEN** candidates from a sequence model (TGN memory or SASRec) trained on interaction history are included in the union

### Requirement: GNN embeddings without leakage
The phase SHALL produce frozen node embeddings from a GNN trained on the graph induced by data up to a reference time, using a self-supervised objective suitable for link prediction. The phase SHALL support inductive embeddings so that nodes unseen during training can be encoded. The phase MAY additionally provide optional TGN-memory and sequence-based (SASRec/GRU4Rec) embeddings. Embeddings SHALL NOT be computed using any interaction after the reference time.

#### Scenario: Embeddings trained up to reference time
- **WHEN** embeddings are produced for reference time `t`
- **THEN** only interactions with time `≤ t` participate in training
#### Scenario: Cold-start nodes encoded

- **WHEN** a candidate node first appears after the embedding training interval
- **THEN** an embedding for it is produced without using future interactions

#### Scenario: Optional sequence embeddings

- **WHEN** sequence-based embeddings (SASRec/GRU4Rec) or TGN-memory embeddings are enabled
- **THEN** they are trained only on interactions with time `≤ t` and obey the same leakage rule

### Requirement: Interaction feature construction
The phase SHALL construct numeric features for each candidate pair, including embedding-derived features (both endpoints' embeddings and their similarity and difference), structural features (degrees, common neighbors, Adamic-Adar, resource allocation, preferential attachment, personalized PageRank), and temporal or activity features. Feature computation SHALL use only data with time `≤ t`.

#### Scenario: Pair feature vector produced
- **WHEN** a candidate pair is scored
- **THEN** a numeric feature vector combining embedding, structural, and temporal features is produced

#### Scenario: Features respect the reference time
- **WHEN** pair features are constructed for reference time `t`
- **THEN** no feature uses an interaction with time `> t`

### Requirement: Ranking models
The phase SHALL train and compare ranking models, with the target model being a gradient-boosted decision tree using a listwise learning-to-rank objective. The phase SHALL additionally train a pointwise classification objective and a pairwise learning-to-rank objective on the same features and query grouping. The comparison set SHALL include training-free heuristic rankers (popularity, common neighbors, Adamic-Adar, resource allocation), a retrieval-similarity-only ranker, a logistic-regression ranker, a LightGBM `lambdarank` model, an XGBoost `rank:pairwise` model, and one end-to-end GNN model.

#### Scenario: Target objective is listwise
- **WHEN** the target ranking model is trained
- **THEN** it optimizes a listwise learning-to-rank objective and ranks candidates per query node

#### Scenario: Comparative objectives trained
- **WHEN** the target model is evaluated
- **THEN** pointwise and pairwise variants are trained on identical features and query grouping and reported alongside it

#### Scenario: Heuristic and sibling baselines present
- **WHEN** ranking results are reported
- **THEN** training-free heuristic baselines, the retrieval-similarity-only baseline, logistic regression, LightGBM `lambdarank`, XGBoost `rank:pairwise`, and one end-to-end GNN are included

### Requirement: Ranking evaluation metrics
The phase SHALL evaluate ranking quality per query with `Hits@K`, `Recall@K`, `MRR`, `MAP`, and `NDCG@K`, and SHALL report results both for the reranker on the retrieval shortlist and end-to-end (retrieval plus reranking). It SHALL also report `Candidate Recall@M`.

#### Scenario: Ranking metrics computed
- **WHEN** a model ranks candidates for each query
- **THEN** `Hits@K`, `Recall@K`, `MRR`, `MAP`, and `NDCG@K` are computed and reported

#### Scenario: End-to-end versus reranker reported
- **WHEN** results are summarized
- **THEN** both the end-to-end pipeline and the reranker-on-shortlist metrics are reported

### Requirement: Horizon sweep
The phase SHALL allow configuring the forecasting horizon `H` and SHALL produce a sweep over multiple horizons, reporting ranking metrics as a function of `H`.

#### Scenario: Horizon varied
- **WHEN** the horizon sweep runs
- **THEN** ranking metrics are produced for each configured horizon value

### Requirement: Ablations and domain comparison
The phase SHALL support ablations over feature groups (with and without embeddings, structural features, and temporal features) and over objectives, and SHALL report results separately for the generic dataset and the transaction dataset.

#### Scenario: Embedding ablation
- **WHEN** the ranker is trained without embedding features
- **THEN** its metrics are reported alongside the full-feature model

#### Scenario: Per-domain reporting
- **WHEN** results are summarized
- **THEN** metrics are reported separately per dataset

### Requirement: Persisted evaluation artifacts
The phase SHALL persist run outputs, including per-model ranking metrics, the feature-importance report, the run configuration (dataset, horizon, subsample, seed), and comparison tables or plots, to a results directory.

#### Scenario: Artifacts written
- **WHEN** a run completes
- **THEN** metrics, configuration, and comparison outputs are written to the results directory

### Requirement: Reproducible environment and phase layout
The phase SHALL be a self-contained, installable Python package with a pinned dependency lockfile and a command-line entry point accepting at least dataset, horizon, retrieval size, subsample, and output-directory options. It SHALL include tests, mini-lessons, interactive exercises, and notebooks organized like the other phases, and its learning material SHALL be runnable inside the phase environment.

#### Scenario: Reproducible run
- **WHEN** the environment is installed from the lockfile and a run is executed with a fixed seed and configuration
- **THEN** the produced artifacts are reproducible

#### Scenario: CLI configures a run
- **WHEN** the command-line entry point is invoked with dataset, horizon, retrieval size, subsample, and output directory
- **THEN** the run uses those values and writes outputs accordingly

#### Scenario: Learning materials runnable
- **WHEN** a mini-lesson or exercise is executed in the phase environment
- **THEN** it runs without errors
