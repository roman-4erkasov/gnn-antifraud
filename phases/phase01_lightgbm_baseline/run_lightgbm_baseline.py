#!/usr/bin/env python3
"""Run the Phase 01 LightGBM baseline pipeline with graph features."""

import argparse
import json
import pickle
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

# Add this phase's ``src`` and the project root to the path so the package and
# the shared ``src/utils`` modules import without an install.
phase_dir = Path(__file__).resolve().parent / "src"
project_root = Path(__file__).resolve().parents[2]
for _path in (str(phase_dir), str(project_root)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from phase01_lightgbm_baseline.comparison import (  # noqa: E402
    compute_comparison_report,
    generate_stop_decision,
    test_significance,
)
from phase01_lightgbm_baseline.data_loader import DataLoader  # noqa: E402
from phase01_lightgbm_baseline.pipeline import (  # noqa: E402
    split_and_prepare,
    train_graph_aware,
)
from phase01_lightgbm_baseline.lightgbm_baseline import (  # noqa: E402
    LightGBMWrapper,
)
from phase01_lightgbm_baseline.graph_features import (  # noqa: E402
    GraphFeatureExtractor,
)
from src.utils.features import BASELINE_FEATURES  # noqa: E402


def _print_metrics(title, metrics):
    print(f"  {title}:")
    for metric, value in metrics.items():
        print(f"    {metric}: {value:+.4f}")


def _save_artifacts(output_dir, minimal_model, graph_model, payload, deltas):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    with open(out / "metrics.json", "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)

    if minimal_model.model is not None:
        minimal_model.model.save_model(str(out / "model.txt"))
    if graph_model.model is not None:
        graph_model.model.save_model(str(out / "model_graph.txt"))
    if minimal_model.calibrator is not None:
        with open(out / "calibrator.pkl", "wb") as fh:
            pickle.dump(minimal_model.calibrator, fh)
    if graph_model.calibrator is not None:
        with open(out / "calibrator_graph.pkl", "wb") as fh:
            pickle.dump(graph_model.calibrator, fh)

    pd.Series(deltas).to_csv(out / "comparison.csv")

    labels = list(deltas.keys())
    values = list(deltas.values())
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(labels, values, color="steelblue")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title("Graph-aware minus minimal (delta metrics)")
    fig.tight_layout()
    fig.savefig(out / "comparison.png", dpi=120)
    plt.close(fig)

    return out


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run LightGBM baseline with graph features"
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="../../data",
        help="Shared data directory (relative to the phase root)",
    )
    parser.add_argument(
        "--sample-limit",
        type=int,
        default=None,
        help="Stratified row cap for fast prototyping",
    )
    parser.add_argument("--random-state", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Directory for persisted run artifacts",
    )
    args = parser.parse_args()

    print("=" * 80)
    print("LightGBM Baseline with Graph Features")
    print("=" * 80)

    print("Loading data...")
    loader = DataLoader(data_dir=args.data_path)
    train_df, test_df = loader.load_data(
        sample_limit=args.sample_limit, random_state=args.random_state
    )
    print(f"  Train samples: {len(train_df)}")
    print(f"  Test samples: {len(test_df)}")
    print(f"  Fraud rate (train): {train_df['isFraud'].mean():.4f}")

    train_feat, val_feat, test_feat, feature_params = split_and_prepare(
        (train_df, test_df), random_state=args.random_state
    )
    print(f"  Baseline feature params (fit on train split): {feature_params}")

    print("Training minimal model (3 baseline features)...")
    minimal_model = LightGBMWrapper(random_state=args.random_state)
    k = max(1, int(np.ceil(0.01 * len(test_feat))))
    minimal_metrics = minimal_model.full_pipeline(
        train_feat[BASELINE_FEATURES].to_numpy(dtype=float),
        train_feat["isFraud"].to_numpy(dtype=int),
        val_feat[BASELINE_FEATURES].to_numpy(dtype=float),
        val_feat["isFraud"].to_numpy(dtype=int),
        test_feat[BASELINE_FEATURES].to_numpy(dtype=float),
        test_feat["isFraud"].to_numpy(dtype=int),
        k=k,
    )
    _print_metrics("Minimal model metrics", minimal_metrics)

    print("Training graph-aware model (baseline + graph features)...")
    graph_extractor = GraphFeatureExtractor(user_col="card1", merchant_col="ProductCD")
    graph_model, graph_metrics = train_graph_aware(
        (train_df, test_df),
        graph_extractor,
        k=k,
        wrapper=LightGBMWrapper(random_state=args.random_state),
    )
    _print_metrics("Graph-aware model metrics", graph_metrics)

    print("Comparing models...")
    deltas = compute_comparison_report(minimal_metrics, graph_metrics)
    print("  Delta metrics (graph - minimal):")
    for metric, value in deltas.items():
        print(f"    {metric}: {value:+.4f}")

    print("Testing statistical significance...")
    p_value = test_significance(
        minimal_model.raw_test_probs_,
        graph_model.raw_test_probs_,
        seed=args.random_state,
    )
    print(f"  P-value: {p_value:.4f}")

    delta_pr_auc = deltas.get("delta_pr_auc", 0.0)
    decision = generate_stop_decision(delta_pr_auc, p_value)

    payload = {
        "lgb_minimal_metrics": minimal_metrics,
        "lgb_graph_metrics": graph_metrics,
        "delta_metrics": deltas,
        "p_value": p_value,
        "delta_pr_auc": delta_pr_auc,
        "stop_decision": decision,
        "config": {
            "data_path": args.data_path,
            "sample_limit": args.sample_limit,
            "random_state": args.random_state,
            "precision_at_k": k,
            "feature_params": feature_params,
        },
    }
    out = _save_artifacts(args.output_dir, minimal_model, graph_model, payload, deltas)

    print("=" * 80)
    print("STOP DECISION")
    print("=" * 80)
    print(f"  Delta PR-AUC: {delta_pr_auc:+.4f}")
    print(f"  P-value: {p_value:.4f}")
    print(f"  Decision: {decision}")
    print(f"  Artifacts written to: {out}")
    print("=" * 80)

    return 0


if __name__ == "__main__":
    sys.exit(main())
