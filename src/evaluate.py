"""
Extended model evaluation: regression diagnostics + genuine tier-classification
metrics.

═══════════════════════════════════════════════════════════════════════════
WHY THIS FILE CHANGED
═══════════════════════════════════════════════════════════════════════════
The previous version of this module compared ``house_tier`` (a composite
score built 40% from ``predicted_price`` and 60% from luxury_score /
geo_desirability / amenities_score / lot size / new-construction) against
tiers derived from the *same composite formula* but swapping in actual
``price_value``. Because 60% of the composite's weight was identical in both
versions, the resulting "98.7% accuracy" was structurally guaranteed to be
high — a constant (zero-signal) ``predicted_price`` still scored 91% on this
metric. It measured agreement between two derived scores, not classification
skill, and it was computed in-sample (no held-out split).

This module now reports on the **genuine** tier classifier
(``src/tier_classifier.py``), which predicts ``price_tier`` (1/2/3, defined
from train-split price tertiles) from the same 40 `_rs` features used by the
price regressor, with a real train/test split and 5-fold CV. Run
``python src/tier_classifier.py`` before this module to populate
``reports/tier_classifier_metrics.csv`` and ``results/<model>/*``.
═══════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

warnings.filterwarnings("ignore", category=UserWarning)

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
RESULTS_DIR = ROOT / "results"
PLOTS_EVAL = ROOT / "plots" / "evaluation"

RANDOM_STATE = 42
CLASS_LABELS = ["Tier 1 (Budget)", "Tier 2 (Mid-range)", "Tier 3 (Premium)"]


def _evaluate_regression_summary() -> list[dict]:
    """Pass through reports/model_metrics.csv as the regression section."""
    metrics_path = REPORTS_DIR / "model_metrics.csv"
    rows: list[dict] = []
    if not metrics_path.exists():
        print("[evaluate] No reports/model_metrics.csv found - skipping regression section.")
        return rows
    reg = pd.read_csv(metrics_path)
    for _, r in reg.iterrows():
        row = {"section": "regression_holdout", "model": r.get("model", "")}
        for c in reg.columns:
            if c != "model":
                row[c] = r[c]
        rows.append(row)
    return rows


def _evaluate_tier_classifiers():
    """Pull genuine classifier results from tier_classifier.py's output."""
    tier_path = REPORTS_DIR / "tier_classifier_metrics.csv"
    rows: list[dict] = []
    if not tier_path.exists():
        print(
            "[evaluate] No reports/tier_classifier_metrics.csv found - "
            "run `python src/tier_classifier.py` first."
        )
        return rows, None
    tiers = pd.read_csv(tier_path)
    for _, r in tiers.iterrows():
        rows.append(
            {
                "section": "price_tier_classification",
                "model": r["model"],
                "train_accuracy": r["train_accuracy"],
                "test_accuracy": r["test_accuracy"],
                "overfit_gap": r["overfit_gap"],
                "cv_accuracy_mean": r["cv_accuracy_mean"],
                "cv_accuracy_std": r["cv_accuracy_std"],
                "test_f1_macro": r["test_f1_macro"],
            }
        )
    return rows, tiers


def _plot_best_tier_classifier(tiers: pd.DataFrame) -> None:
    """Confusion matrix + per-class PRF + tier distribution for the BEST classifier
    (highest test_accuracy), loaded from disk and evaluated on the real test split."""
    import joblib
    import sys

    sys.path.insert(0, str(ROOT / "src"))
    from tier_classifier import load_tier_data  # noqa: E402

    best_row = tiers.sort_values("test_accuracy", ascending=False).iloc[0]
    key = best_row["key"]
    model_path = MODELS_DIR / f"tier_{key}.pkl"
    if not model_path.exists():
        print(f"[evaluate] Best tier model file missing ({model_path}); skipping plots.")
        return

    model = joblib.load(model_path)
    X_train, X_test, y_train, y_test, feature_names = load_tier_data()

    # XGBoost is trained on 0-indexed labels
    y_eval = y_test - 1 if key == "05_xgboost" else y_test
    y_pred = model.predict(X_test)

    labels_idx = sorted(y_eval.unique())
    cm = confusion_matrix(y_eval, y_pred, labels=labels_idx)
    report_dict = classification_report(y_eval, y_pred, labels=labels_idx,
                                          target_names=CLASS_LABELS, output_dict=True,
                                          zero_division=0)
    report_txt = classification_report(y_eval, y_pred, labels=labels_idx,
                                         target_names=CLASS_LABELS, zero_division=0)

    PLOTS_EVAL.mkdir(parents=True, exist_ok=True)

    # Plotly confusion matrix
    fig_cm = go.Figure(
        data=go.Heatmap(
            z=cm,
            x=[f"Pred {l}" for l in CLASS_LABELS],
            y=[f"Actual {l}" for l in CLASS_LABELS],
            colorscale="Blues",
            text=cm,
            texttemplate="%{text}",
        )
    )
    fig_cm.update_layout(
        title=f"Price-Tier Confusion Matrix — {best_row['model']} (test set)",
        xaxis_title="Predicted tier",
        yaxis_title="Actual tier (price_value tertile)",
    )
    fig_cm.write_html(PLOTS_EVAL / "confusion_matrix.html")

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASS_LABELS, yticklabels=CLASS_LABELS)
    plt.xlabel("Predicted tier")
    plt.ylabel("Actual tier (price_value tertile)")
    plt.title(f"Price-Tier Confusion Matrix — {best_row['model']}")
    plt.tight_layout()
    plt.savefig(PLOTS_EVAL / "confusion_matrix.png", dpi=140)
    plt.close()

    # Per-class precision/recall/F1
    per_class = []
    for lab in CLASS_LABELS:
        row = report_dict.get(lab, {})
        per_class.append({
            "tier": lab,
            "precision": row.get("precision", 0.0),
            "recall": row.get("recall", 0.0),
            "f1": row.get("f1-score", 0.0),
            "support": row.get("support", 0.0),
        })
    pc_df = pd.DataFrame(per_class)
    fig_bar = px.bar(
        pc_df, x="tier", y=["precision", "recall", "f1"], barmode="group",
        title=f"Per-tier precision / recall / F1 — {best_row['model']} (test set)",
    )
    fig_bar.write_html(PLOTS_EVAL / "per_class_prf.html")

    # Actual vs predicted tier distribution
    actual_counts = y_eval.value_counts().sort_index()
    pred_counts = pd.Series(y_pred).value_counts().sort_index()
    dist = pd.DataFrame({
        "tier": CLASS_LABELS,
        "actual_count": [int(actual_counts.get(i, 0)) for i in labels_idx],
        "predicted_count": [int(pred_counts.get(i, 0)) for i in labels_idx],
    })
    fig_dist = px.bar(
        dist.melt(id_vars="tier", var_name="source", value_name="count"),
        x="tier", y="count", color="source", barmode="group",
        title=f"Tier distribution: actual vs predicted — {best_row['model']} (test set)",
    )
    fig_dist.write_html(PLOTS_EVAL / "tier_distribution_compare.html")

    report_path = REPORTS_DIR / "classification_report_tiers.txt"
    header = (
        f"Genuine price_tier classification report - best model: {best_row['model']}\n"
        f"Test accuracy: {best_row['test_accuracy']:.4f} | "
        f"Train accuracy: {best_row['train_accuracy']:.4f} | "
        f"Overfit gap: {best_row['overfit_gap']:.4f}\n"
        f"{'='*60}\n\n"
    )
    report_path.write_text(header + report_txt, encoding="utf-8")
    print(f"[evaluate] Wrote classification report -> {report_path}")
    print(f"[evaluate] Saved evaluation plots under {PLOTS_EVAL}")


def run_evaluation() -> None:
    """Build evaluation artifacts for the dashboard and reports."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    print("[evaluate] Building regression summary...")
    summary_rows = _evaluate_regression_summary()

    print("[evaluate] Loading genuine tier-classifier results...")
    tier_rows, tiers_df = _evaluate_tier_classifiers()
    summary_rows.extend(tier_rows)
    if tiers_df is not None:
        _plot_best_tier_classifier(tiers_df)

    summary_df = pd.DataFrame(summary_rows)
    out_csv = REPORTS_DIR / "model_evaluation_summary.csv"
    summary_df.to_csv(out_csv, index=False)
    print(f"[evaluate] Wrote summary -> {out_csv}")


def main() -> None:
    run_evaluation()


if __name__ == "__main__":
    main()
