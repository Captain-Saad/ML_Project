"""
End-to-end orchestration for the intelligent housing analytics system.

Running this script executes preprocessing, model training, tiering, anomaly
scoring, visualization, and SHAP explainability in order while surfacing concise
progress logs suitable for demos and grading checkpoints.
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

import pandas as pd

# Ensure sibling modules import correctly when invoked as ``python src/pipeline.py``.
SRC_DIR = Path(__file__).resolve().parent
ROOT_DIR = SRC_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import anomaly  # noqa: E402
import classify  # noqa: E402
import evaluate  # noqa: E402
import explain  # noqa: E402
import preprocess  # noqa: E402
import tier_classifier  # noqa: E402
import train  # noqa: E402
import visualize  # noqa: E402


def _run_step(title: str, func, *, critical: bool = True) -> None:
    """Execute a pipeline stage with friendly console progress and soft errors."""
    print(f"\n{'=' * 20} {title} {'=' * 20}")
    try:
        func()
    except Exception as exc:  # pragma: no cover - runtime safety for coursework demos
        print(f"[pipeline] ERROR in {title}: {exc}")
        traceback.print_exc()
        if critical:
            raise
        print(f"[pipeline] Continuing despite error in {title}...")


def run_full_pipeline() -> None:
    """Run all project stages sequentially."""
    print(f"[pipeline] Project root: {ROOT_DIR}")
    _run_step("PREPROCESSING", preprocess.run_preprocess)
    _run_step("MODEL TRAINING", lambda: train.run_training())
    _run_step("TIERING & HOUSESCORE (value ranking)", classify.run_classification)
    _run_step("PRICE-TIER CLASSIFICATION (genuine, held-out)", tier_classifier.run_all)
    _run_step("MODEL EVALUATION", evaluate.run_evaluation)
    _run_step("ANOMALY & OPPORTUNITIES", anomaly.run_anomaly_detection)
    _run_step("VISUALIZATION", visualize.run_visualizations)
    _run_step("EXPLAINABILITY", explain.run_explainability, critical=False)

    # Console highlights requested for evaluation / demos.
    metrics = pd.read_csv(ROOT_DIR / "reports" / "model_metrics.csv")
    train.print_comparison_table(metrics)
    best = metrics.sort_values("rmse").iloc[0]["model"]
    print(f"[pipeline] Best regression model (lowest RMSE on hold-out): {best}")

    tier_metrics = pd.read_csv(ROOT_DIR / "reports" / "tier_classifier_metrics.csv")
    best_tier = tier_metrics.sort_values("test_accuracy", ascending=False).iloc[0]
    print(
        f"[pipeline] Best tier classifier: {best_tier['model']} "
        f"(test acc={best_tier['test_accuracy']:.4f}, "
        f"overfit gap={best_tier['overfit_gap']:.4f})"
    )

    df = pd.read_csv(ROOT_DIR / "data" / "processed_dataset.csv")
    classify.print_top_houses(df, n=10)
    anomaly.print_undervalued_opportunities(df, n=10)
    print("\n[pipeline] All stages completed successfully.\n")


def main() -> None:
    run_full_pipeline()


if __name__ == "__main__":
    main()
