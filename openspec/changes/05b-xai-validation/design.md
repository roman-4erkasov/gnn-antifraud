## Context

Phase 04a established XAI methods (GNNExplainer, PGExplainer, GAT attention) and validated them against IEEE-CIS chargeback labels. Phase 05 generated synthetic fraud patterns (Mitme, Cascade, Money Mule) with known ground-truth structure. This phase validates XAI explanations against these synthetic patterns to answer: **Can XAI methods correctly identify known fraud structures?**

This is the final validation step — if XAI cannot identify known patterns, its explanations are not trustworthy for real fraud investigation.

## Goals / Non-Goals

**Goals:**
- Validate GNNExplainer, PGExplainer, and GAT attention explanations against synthetic patterns from Phase 05
- Measure explanation quality: precision/recall of XAI-identified subgraphs against ground-truth patterns
- Compare XAI performance across pattern types (Mitme vs Cascade vs Money Mule)
- Validate explanations on classifier-detected pattern instances in addition to synthetic ground-truth samples
- Generate stop decision: if XAI precision < 0.20, flag that explanations are not reliable

**Non-Goals:**
- Implementing new XAI methods (Phase 04a already provides them)
- Generating synthetic patterns (Phase 05 already provides them)
- Building UI for visualization (reserved for future work)

## Decisions

### Decision 1: Validation against synthetic patterns with known structure

**Choice:** Validate XAI explanations by comparing highlighted edges/nodes against the known ground-truth structure of synthetic patterns (Mitme, Cascade, Money Mule) from Phase 05.

**Rationale:** Synthetic patterns provide ground-truth fraud structure — we know exactly which edges/nodes are part of the fraud. This allows precise measurement of XAI quality (precision/recall). Chargeback labels (Phase 04a) provide domain-relevant validation but lack structural ground truth.

**Alternatives considered:**
- Human evaluation by fraud analysts — expensive and not reproducible
- Permutation importance — measures feature importance but not structural explanation quality
- Only chargeback validation (Phase 04a) — lacks structural ground truth

### Decision 2: Precision/recall metrics for XAI subgraphs

**Choice:** Compute precision (fraction of XAI-highlighted edges that are in ground-truth fraud) and recall (fraction of ground-truth fraud edges that are XAI-highlighted) for each explanation.

**Rationale:** Precision measures false positives (XAI highlighting non-fraud edges), recall measures false negatives (XAI missing fraud edges). Both are needed to assess explanation quality. F1-score provides a single metric for comparison.

**Alternatives considered:**
- Only precision — ignores missed fraud edges
- Only recall — ignores false positives
- IoU (Intersection over Union) — similar to F1 but less interpretable

### Decision 3: Cross-pattern comparison

**Choice:** Compare XAI performance across pattern types (Mitme, Cascade, Money Mule) to identify which patterns are harder to explain.

**Rationale:** Different patterns have different structures (linear chains vs trees vs cycles). XAI methods may perform differently on different structures. Understanding this helps investigators know when to trust explanations.

**Alternatives considered:**
- Aggregate all patterns — loses pattern-specific insights
- Only validate on one pattern type — not representative

### Decision 4: Stop decision threshold

**Choice:** If XAI precision < 0.20 on synthetic patterns, print warning: "XAI explanations are not reliable for investigation — consider using different XAI parameters or methods."

**Rationale:** Precision < 0.20 means >80% of XAI-highlighted edges are not part of the fraud. This is too noisy for investigation. The threshold is conservative — even random explanations would achieve ~5% precision on large graphs.

**Alternatives considered:**
- Higher threshold (0.50) — may be too strict for complex patterns
- Lower threshold (0.10) — too permissive, allows noisy explanations
- No threshold — loses actionable guidance

### Decision 5: Self-contained data loading from Phase 05

**Choice:** Phase 05b loads synthetic patterns from Phase 05's `data/synthetic-patterns/` directory. Implements its own `data_loader.py` that reads pattern files and constructs graphs compatible with XAI methods from Phase 04a.

**Rationale:** Each phase should be self-contained and reproducible. Loading from Phase 05's data directory avoids duplication while keeping phases independent. The loader constructs graphs optimized for XAI validation (preserving ground-truth labels).

**Alternatives considered:**
- Importing data from Phase 05's loader — creates tight coupling
- Phase-specific data directory — leads to duplicate data
- No download capability — requires manual data setup

### Decision 6: Interactive exercises with linked notebooks

**Choice:** Each lesson's Practice section links to an interactive Jupyter notebook in `exercises/` directory.

**Rationale:** Practice exercises in markdown are static text. Linking to notebooks allows users to open, run, and modify code directly. Minimal template: setup code, task description, empty code cell for user solution, solution in markdown code block.

**Alternatives considered:**
- All exercises in one notebook — harder to navigate, no clear mapping to lessons
- Exercises embedded in lesson notebooks — mixes demonstration and practice
- Practice only in markdown — requires copy-paste, less interactive

## Risks / Trade-offs

| Risk | Mitigation |
|------|-----------|
| XAI may perform poorly on synthetic patterns | This is a valid result, not a failure. Report the poor performance and investigate causes (pattern complexity, XAI parameters). |
| Synthetic patterns may not represent real fraud | Synthetic patterns are simplified. Use them as a sanity check, not as the only validation. |
| Precision/recall may be sensitive to threshold | Use multiple thresholds (top-5, top-10, top-20 edges) and report the trend. |
| XAI methods may not generalize across pattern types | This is expected. Report pattern-specific performance and provide guidance on when to trust each method. |

## Migration Plan

This is a greenfield validation module. No migration from existing code is needed.

1. Create `phases/phase05b_xai_validation/src/phase05b_xai_validation/` module with `data_loader.py`, `pattern_validator.py`, `metrics.py`, `comparison.py`.
2. Integrate with XAI methods from Phase 04a and synthetic patterns from Phase 05.
3. Add test suite under `phases/phase05b_xai_validation/tests/`.
4. Create run script `run_xai_validation.py` for end-to-end validation pipeline.

## Educational Content

### Lesson Structure

Each lesson in `lessons/` follows a consistent 5-part pedagogical structure:

1. **Explain** — Conceptual overview of the topic (what it is, why it matters for XAI validation).
2. **Code** — Annotated code walkthrough showing how the concept is implemented.
3. **Visualize** — Diagrams, plots, or graph visualizations that make the concept concrete (e.g., precision/recall curves, pattern-specific XAI performance).
4. **Practice** — Hands-on exercise for the learner to complete.
5. **Solution** — Reference solution with explanation of expected results.

### Lesson Catalog

| Lesson | Topic | Key Concepts |
|--------|-------|-------------|
| `01-xai-validation-overview.md` | Why validate XAI on synthetic patterns | Ground truth, precision/recall, trustworthiness |
| `02-pattern-ground-truth.md` | Synthetic pattern structure | Mitme, Cascade, Money Mule, ground-truth labels |
| `03-precision-recall-for-xai.md` | Measuring XAI quality | Precision, recall, F1-score, threshold selection |
| `04-cross-pattern-comparison.md` | XAI across pattern types | Pattern-specific performance, structural complexity |
| `05-stop-decision.md` | When to trust XAI | Stop criteria, actionable guidance, limitations |

### Notebook Catalog

| Notebook | Topic | Key Activities |
|----------|-------|---------------|
| `01-xai-validation-analysis.ipynb` | Validate XAI on synthetic patterns | Load patterns, run XAI, compute precision/recall, visualize results |
| `02-pattern-specific-xai.ipynb` | Pattern-specific XAI performance | Compare XAI across Mitme/Cascade/Money Mule, identify strengths/weaknesses |

## Open Questions

- What threshold should be used for "highlighted" edges? (Assumption: top-10 edges by importance score.)
- Should we validate node-level or edge-level explanations? (Assumption: edge-level, since fraud is transaction-level.)
- How should we handle patterns with different sizes? (Assumption: normalize precision/recall by pattern size.)
