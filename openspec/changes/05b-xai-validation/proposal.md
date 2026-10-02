## Why

Phase 04a established XAI methods (GNNExplainer, PGExplainer, GAT attention) and validated them against IEEE-CIS chargeback labels. Phase 05 generated synthetic fraud patterns (Mitme, Cascade, Money Mule) with known ground-truth structure. This phase validates XAI explanations against these synthetic patterns to answer: **Can XAI methods correctly identify known fraud structures?**

This is the final validation step — if XAI cannot identify known patterns, its explanations are not trustworthy for real fraud investigation.

## What Changes

- Validate GNNExplainer, PGExplainer, and GAT attention explanations against synthetic patterns from Phase 05
- Measure explanation quality: do highlighted edges/nodes match the known fraud structure?
- Compute precision/recall of XAI-identified subgraphs against ground-truth patterns
- Compare XAI performance across pattern types (Mitme vs Cascade vs Money Mule)
- **Stop decision:** If XAI precision < 0.20 on synthetic patterns, flag that explanations are not reliable for investigation

## Capabilities

### New Capabilities

- `phase05b_xai_validation`: XAI validation against synthetic fraud patterns, precision/recall metrics for explanation quality, cross-pattern XAI performance comparison

### Modified Capabilities

- `phase04a_xai_basics`: Adds synthetic pattern validation to XAI pipeline (extends chargeback validation)
- `phase05_pattern_detection`: XAI quality becomes an evaluation criterion for pattern detection

## Impact

- Self-contained phase under `phases/phase05b_xai_validation/`:
  - `src/phase05b_xai_validation/data_loader.py`: Load synthetic patterns from Phase 05
  - `src/phase05b_xai_validation/pattern_validator.py`: Validate XAI explanations against pattern ground truth
  - `src/phase05b_xai_validation/metrics.py`: Precision/recall for XAI-identified subgraphs
  - `src/phase05b_xai_validation/comparison.py`: Cross-pattern XAI performance comparison
  - `tests/`: Test suite for validation components
  - `run_xai_validation.py`: End-to-end validation pipeline
  - `lessons/`
    - `01-xai-validation-overview.md`
    - `02-pattern-ground-truth.md`
    - `03-precision-recall-for-xai.md`
    - `04-cross-pattern-comparison.md`
    - `05-stop-decision.md`
  - `notebooks/`
    - `01-xai-validation-analysis.ipynb`
    - `02-pattern-specific-xai.ipynb`
  - `exercises/`: interactive Jupyter notebooks for Practice sections (linked from lessons)
- Uses existing: `phases/phase04a_xai_basics/` (XAI methods), `phases/phase05_pattern_detection/` (synthetic patterns)
- Dependency: Phase 04a and Phase 05 must be completed first
- No breaking changes to existing codebase
