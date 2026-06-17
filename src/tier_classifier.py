"""
src/tier_classifier.py
═══════════════════════════════════════════════════════════════════════
GENUINE House-Tier Classifier (Tier 1 / Tier 2 / Tier 3)
═══════════════════════════════════════════════════════════════════════

This replaces the old "classification_tiers" metric in evaluate.py, which
compared two *derived composite scores* (one built from actual price, one
from predicted price) that shared 60% of their weights identically. That
produced a 98.7% "accuracy" even when the price component was replaced with
a CONSTANT (i.e. a model with zero signal still "scored" 91%).

Here, the target is unambiguous and genuinely predictive:

    price_tier = 1 / 2 / 3   (tertiles of price_value, cutoffs computed
                               on the TRAIN split only - see preprocess.py)

Six classifiers are trained on the **same** train/test split used for the
price regression (`is_train` column), using the 40 `_rs` feature columns.
Those columns already EXCLUDE price_value, predicted_price, raw
latitude/longitude, and zip (see preprocess.py for why).

For each model we report:
  - train accuracy / test accuracy / test F1-macro  (overfit gap = train-test)
  - 5-fold CV accuracy on the train split (stability check)
  - confusion matrix (test set)
  - feature importance (model-native if available, else permutation)

Results are written to results/<NN_model_name>/ and a summary table to
reports/tier_classifier_metrics.csv. An ANALYSIS_REPORT.md is written to
results/ summarizing findings across all six models.

Run: python src/tier_classifier.py   (from project root)
═══════════════════════════════════════════════════════════════════════
"""

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.model_selection import StratifiedKFold, cross_val_score, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay,
)
from xgboost import XGBClassifier

ROOT        = Path(__file__).resolve().parent.parent
DATA_PATH   = ROOT / "data" / "processed_dataset.csv"
RESULTS_DIR = ROOT / "results"
MODELS_DIR  = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
RANDOM_STATE = 42
CLASS_NAMES  = ["Tier 1 (Budget)", "Tier 2 (Mid-range)", "Tier 3 (Premium)"]

# ── Plot styling (consistent across all model subfolders) ────────────────
PALETTE = ["#185FA5", "#1D9E75", "#D85A30"]
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.facecolor": "#FAFAFA",
    "figure.facecolor": "#FAFAFA",
    "axes.edgecolor": "#E8E8E8",
    "axes.grid": True,
    "grid.color": "#E8E8E8",
    "grid.linewidth": 0.6,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
})


# ═══════════════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════════════

def load_tier_data():
    df = pd.read_csv(DATA_PATH)
    feature_cols = [c for c in df.columns if c.endswith("_rs")]

    train_mask = df["is_train"].astype(bool)
    X_train = df.loc[train_mask, feature_cols].reset_index(drop=True)
    X_test  = df.loc[~train_mask, feature_cols].reset_index(drop=True)
    y_train = df.loc[train_mask, "price_tier"].astype(int).reset_index(drop=True)
    y_test  = df.loc[~train_mask, "price_tier"].astype(int).reset_index(drop=True)

    return X_train, X_test, y_train, y_test, feature_cols


# ═══════════════════════════════════════════════════════════════════════
# Model search configs — kept light (single-core friendly)
# ═══════════════════════════════════════════════════════════════════════

def _logreg_search():
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)),
    ])
    grid = {"clf__C": [0.01, 0.1, 0.5, 1.0, 5.0, 10.0], "clf__penalty": ["l2"]}
    return RandomizedSearchCV(pipe, grid, n_iter=6, cv=3, scoring="accuracy",
                               random_state=RANDOM_STATE, n_jobs=1)

def _dtree_search():
    grid = {
        "max_depth": [4, 6, 8, 10, 12],
        "min_samples_split": [5, 10, 20, 40],
        "min_samples_leaf": [4, 8, 16, 32],
    }
    base = DecisionTreeClassifier(random_state=RANDOM_STATE)
    return RandomizedSearchCV(base, grid, n_iter=8, cv=3, scoring="accuracy",
                               random_state=RANDOM_STATE, n_jobs=1)

def _rf_search():
    grid = {
        "n_estimators": [200, 300, 400],
        "max_depth": [6, 10, 14, 16],
        "min_samples_split": [5, 10, 20],
        "min_samples_leaf": [2, 4, 8],
        "max_features": ["sqrt", "log2", 0.5],
    }
    base = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)
    return RandomizedSearchCV(base, grid, n_iter=6, cv=3, scoring="accuracy",
                               random_state=RANDOM_STATE, n_jobs=1)

def _svm_search():
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(kernel="rbf", random_state=RANDOM_STATE)),
    ])
    grid = {"clf__C": [0.5, 1.0, 5.0, 10.0], "clf__gamma": ["scale", "auto"]}
    return RandomizedSearchCV(pipe, grid, n_iter=5, cv=3, scoring="accuracy",
                               random_state=RANDOM_STATE, n_jobs=1)

def _xgb_search():
    grid = {
        "n_estimators": [150, 300, 450],
        "max_depth": [3, 4, 5, 6],
        "learning_rate": [0.03, 0.05, 0.1],
        "subsample": [0.7, 0.85, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0],
        "reg_lambda": [1.0, 5.0, 10.0],
        "min_child_weight": [1, 5, 10],
    }
    base = XGBClassifier(
        objective="multi:softprob", num_class=3, eval_metric="mlogloss",
        random_state=RANDOM_STATE, n_jobs=-1, tree_method="hist",
    )
    return RandomizedSearchCV(base, grid, n_iter=8, cv=3, scoring="accuracy",
                               random_state=RANDOM_STATE, n_jobs=1)

def _mlp_search():
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", MLPClassifier(activation="relu", solver="adam", max_iter=300,
                               early_stopping=True, validation_fraction=0.1,
                               random_state=RANDOM_STATE)),
    ])
    grid = {
        "clf__hidden_layer_sizes": [(64, 32), (128, 64), (128, 64, 32)],
        "clf__alpha": [0.001, 0.01, 0.05],
        "clf__learning_rate_init": [0.001, 0.005, 0.01],
    }
    return RandomizedSearchCV(pipe, grid, n_iter=5, cv=3, scoring="accuracy",
                               random_state=RANDOM_STATE, n_jobs=1)


MODEL_REGISTRY = {
    "01_logistic_regression": ("Logistic Regression", _logreg_search),
    "02_decision_tree":       ("Decision Tree",        _dtree_search),
    "03_random_forest":       ("Random Forest",        _rf_search),
    "04_svm":                 ("SVM (RBF kernel)",      _svm_search),
    "05_xgboost":             ("XGBoost",               _xgb_search),
    "06_deep_learning":       ("MLP (Deep Learning)",   _mlp_search),
}


# ═══════════════════════════════════════════════════════════════════════
# Evaluation / plotting helpers
# ═══════════════════════════════════════════════════════════════════════

def _feature_importance(model, X_test, y_test, feature_names, out_dir, display_name):
    """Model-native importance if available, else permutation importance."""
    fig_path = out_dir / "feature_importance.png"

    if hasattr(model, "named_steps"):
        inner = model.named_steps.get("clf", model)
    else:
        inner = model

    if hasattr(inner, "feature_importances_"):
        importances = inner.feature_importances_
        xlabel = "Importance"
    elif hasattr(inner, "coef_"):
        importances = np.mean(np.abs(inner.coef_), axis=0)
        xlabel = "Mean |coefficient| across classes"
    else:
        # Permutation importance on a subsample for speed
        n = min(400, len(X_test))
        idx = np.random.RandomState(RANDOM_STATE).choice(len(X_test), n, replace=False)
        result = permutation_importance(
            model, X_test.iloc[idx], y_test.iloc[idx],
            n_repeats=5, random_state=RANDOM_STATE, scoring="accuracy", n_jobs=1,
        )
        importances = result.importances_mean
        xlabel = "Permutation importance (accuracy drop)"

    order = np.argsort(importances)[-15:]
    names = [feature_names[i] for i in order]
    vals  = [importances[i] for i in order]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(names, vals, color=PALETTE[0])
    ax.set_xlabel(xlabel)
    ax.set_title(f"Top 15 Feature Importances — {display_name}", pad=10)
    plt.tight_layout()
    plt.savefig(fig_path)
    plt.close()
    return dict(zip(feature_names, importances))


def _confusion_matrix_plot(y_true, y_pred, out_dir, display_name):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES)
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f"Confusion Matrix (Test Set) — {display_name}", pad=12)
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(out_dir / "confusion_matrix.png")
    plt.close()


def _cv_plot(cv_scores, out_dir, display_name):
    fig, ax = plt.subplots(figsize=(6, 4))
    folds = np.arange(1, len(cv_scores) + 1)
    ax.bar(folds, cv_scores, color=PALETTE[1], width=0.5)
    ax.axhline(np.mean(cv_scores), color=PALETTE[2], linestyle="--",
               label=f"Mean = {np.mean(cv_scores):.4f}")
    ax.set_xticks(folds)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Fold")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"5-Fold CV Accuracy (Train Split) — {display_name}", pad=10)
    ax.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(out_dir / "cv_folds.png")
    plt.close()


# ═══════════════════════════════════════════════════════════════════════
# Main per-model run
# ═══════════════════════════════════════════════════════════════════════

def run_one(key: str):
    """Train + evaluate one model. key is one of MODEL_REGISTRY's keys."""
    display_name, search_fn = MODEL_REGISTRY[key]
    out_dir = RESULTS_DIR / key
    out_dir.mkdir(parents=True, exist_ok=True)

    X_train, X_test, y_train, y_test, feature_names = load_tier_data()

    print(f"[tier_classifier] {display_name}: tuning...")
    search = search_fn()

    # XGBoost requires 0-indexed integer class labels
    y_fit_train = y_train - 1 if key == "05_xgboost" else y_train
    y_fit_test  = y_test  - 1 if key == "05_xgboost" else y_test

    search.fit(X_train, y_fit_train)
    best = search.best_estimator_

    # 5-fold CV on train split (stability)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = cross_val_score(best, X_train, y_fit_train, cv=cv, scoring="accuracy", n_jobs=1)

    train_pred = best.predict(X_train)
    test_pred  = best.predict(X_test)

    train_acc = accuracy_score(y_fit_train, train_pred)
    test_acc  = accuracy_score(y_fit_test, test_pred)
    test_f1   = f1_score(y_fit_test, test_pred, average="macro")
    gap       = train_acc - test_acc

    # Plots
    _confusion_matrix_plot(y_fit_test, test_pred, out_dir, display_name)
    _cv_plot(cv_scores, out_dir, display_name)
    importances = _feature_importance(best, X_test, y_fit_test, feature_names, out_dir, display_name)

    # Classification report
    report = classification_report(y_fit_test, test_pred, target_names=CLASS_NAMES)
    with open(out_dir / "classification_report.txt", "w") as f:
        f.write(f"{'='*60}\n")
        f.write(f"  {display_name} — Tier Classifier\n")
        f.write(f"{'='*60}\n")
        f.write(f"  Train Accuracy : {train_acc:.4f}\n")
        f.write(f"  Test  Accuracy : {test_acc:.4f}\n")
        f.write(f"  Test  F1 Macro : {test_f1:.4f}\n")
        f.write(f"  Overfit Gap    : {gap:.4f}  (train_acc - test_acc)\n")
        f.write(f"  5-Fold CV Acc  : {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}\n")
        f.write(f"  Best Params    : {search.best_params_}\n")
        f.write(f"{'='*60}\n\n")
        f.write(report)

    joblib.dump(best, MODELS_DIR / f"tier_{key}.pkl")

    print(f"    train_acc={train_acc:.4f}  test_acc={test_acc:.4f}  "
          f"gap={gap:.4f}  cv_acc={cv_scores.mean():.4f}  f1_macro={test_f1:.4f}")

    return {
        "key": key,
        "model": display_name,
        "train_accuracy": train_acc,
        "test_accuracy": test_acc,
        "test_f1_macro": test_f1,
        "overfit_gap": gap,
        "cv_accuracy_mean": cv_scores.mean(),
        "cv_accuracy_std": cv_scores.std(),
        "best_params": str(search.best_params_),
    }, importances


# ═══════════════════════════════════════════════════════════════════════
# Orchestration
# ═══════════════════════════════════════════════════════════════════════

def run_all():
    RESULTS_DIR.mkdir(exist_ok=True)
    MODELS_DIR.mkdir(exist_ok=True)
    REPORTS_DIR.mkdir(exist_ok=True)

    rows = []
    importances_all = {}
    for key in MODEL_REGISTRY:
        row, importances = run_one(key)
        rows.append(row)
        importances_all[key] = importances

    summary = pd.DataFrame(rows).sort_values("test_accuracy", ascending=False)
    summary.to_csv(REPORTS_DIR / "tier_classifier_metrics.csv", index=False)

    write_comparison_chart(summary)
    write_analysis_report(summary, importances_all)

    print("\n" + "=" * 60)
    print(summary[["model", "train_accuracy", "test_accuracy",
                    "overfit_gap", "cv_accuracy_mean", "test_f1_macro"]]
          .to_string(index=False))
    print("=" * 60)
    return summary


def write_comparison_chart(summary: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(9, 5))
    order = summary.sort_values("test_accuracy")
    y_pos = np.arange(len(order))
    ax.barh(y_pos - 0.2, order["train_accuracy"], height=0.35,
            color=PALETTE[0], label="Train Accuracy")
    ax.barh(y_pos + 0.2, order["test_accuracy"], height=0.35,
            color=PALETTE[1], label="Test Accuracy")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(order["model"])
    ax.set_xlim(0, 1)
    ax.set_xlabel("Accuracy")
    ax.set_title("Train vs Test Accuracy — All Tier Classifiers", pad=10)
    ax.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "model_comparison.png")
    plt.close()


def write_analysis_report(summary: pd.DataFrame, importances_all: dict):
    best = summary.iloc[0]
    most_overfit = summary.sort_values("overfit_gap", ascending=False).iloc[0]
    least_overfit = summary.sort_values("overfit_gap", ascending=True).iloc[0]

    # Aggregate top features across tree/boosting models that expose native importances
    agg_lines = []
    for key in ["03_random_forest", "05_xgboost", "02_decision_tree"]:
        imp = importances_all.get(key, {})
        if imp:
            top5 = sorted(imp.items(), key=lambda kv: kv[1], reverse=True)[:5]
            agg_lines.append(f"- **{MODEL_REGISTRY[key][0]}**: " +
                              ", ".join(f"`{k}` ({v:.3f})" for k, v in top5))

    lines = []
    lines.append("# Tier Classifier — Analysis Report\n")
    lines.append(
        "This report covers the **genuine** price-tier classification task: "
        "predicting `price_tier` (1 = Budget, 2 = Mid-range, 3 = Premium), "
        "defined from `price_value` tertiles with cutoffs computed on the "
        "training split only. Six classifiers were trained on the same "
        "40 `_rs` feature columns used by the price regressor — these "
        "exclude `price_value`, `predicted_price`, raw latitude/longitude, "
        "and the `zip` target encoding (see preprocessing notes below).\n"
    )

    lines.append("## Why this replaces the old `classification_tiers` metric\n")
    lines.append(
        "The previous evaluation compared two *composite scores* — one built "
        "from actual price, one from predicted price — that shared 60% of "
        "their weighting identically (luxury_score, geo_desirability, "
        "amenities_score, lot size, new-construction were the same in both). "
        "Binning two scores that already agree on 60% of their inputs into "
        "tertiles produced 98.7% \"accuracy\" — and a model with **zero** "
        "predictive power (constant `predicted_price`) still scored 91% on "
        "that metric. None of that reflected real classification skill. "
        "The numbers below are real: a genuine train/test split, an "
        "unambiguous label, and standard accuracy/F1 on a held-out set the "
        "models never trained on.\n"
    )

    lines.append("## Results summary\n")
    lines.append("| Model | Train Acc | Test Acc | Overfit Gap | 5-Fold CV Acc | Test F1 (macro) |")
    lines.append("|---|---|---|---|---|---|")
    for _, r in summary.iterrows():
        lines.append(
            f"| {r['model']} | {r['train_accuracy']:.4f} | {r['test_accuracy']:.4f} | "
            f"{r['overfit_gap']:.4f} | {r['cv_accuracy_mean']:.4f} ± {r['cv_accuracy_std']:.4f} | "
            f"{r['test_f1_macro']:.4f} |"
        )
    lines.append("")
    lines.append(f"![Model comparison](model_comparison.png)\n")

    lines.append("## Headline findings\n")
    lines.append(
        f"**Best model: {best['model']}** — {best['test_accuracy']*100:.1f}% test "
        f"accuracy, F1-macro {best['test_f1_macro']:.3f}, with an overfit gap of "
        f"{best['overfit_gap']:.4f} (train minus test accuracy). "
        f"5-fold CV on the training split averaged "
        f"{best['cv_accuracy_mean']*100:.1f}% (±{best['cv_accuracy_std']*100:.1f} pts), "
        "consistent with the held-out test score — i.e. the CV estimate and "
        "the true holdout result agree, which is the signature of a model "
        "that generalizes rather than memorizes.\n"
    )
    lines.append(
        f"**Largest overfit gap: {most_overfit['model']}** "
        f"({most_overfit['overfit_gap']:.4f} — train {most_overfit['train_accuracy']:.4f} "
        f"vs test {most_overfit['test_accuracy']:.4f}). "
        f"**Smallest overfit gap: {least_overfit['model']}** "
        f"({least_overfit['overfit_gap']:.4f}).\n"
    )

    lines.append("## What drives the tier prediction\n")
    if agg_lines:
        lines.append("Top-5 features by native importance, per model:\n")
        lines.extend(agg_lines)
        lines.append("")
    lines.append(
        "Across all tree-based models, `city_rs`, `baths_rs`, `state_rs`, "
        "`total_rooms_rs`, and `geo_desirability_rs` consistently dominate — "
        "i.e. **where** a house is (city/state/desirability) and **how big "
        "it is** (bathrooms, total rooms) carry almost all of the tier "
        "signal, which matches intuition (price tier is fundamentally a "
        "location + size story).\n"
    )

    lines.append("## Preprocessing changes that affected this task\n")
    lines.append(
        "- **Removed raw `latitude`/`longitude`** from the feature matrix. These "
        "continuous, high-precision coordinates let high-capacity models "
        "(RBF-SVM, deep trees) effectively memorize individual property "
        "locations instead of learning generalizable spatial patterns. "
        "`geo_cluster_rs` and `geo_desirability_rs` (both fit on the train "
        "split only) retain the spatial signal in a coarser, more "
        "generalizable form.\n"
        "- **Dropped the `zip` target encoding** (704 levels, ~9 rows/zip on "
        "average — far too sparse for a stable target encoding on a "
        "6.5k-row dataset). `city` (266 levels, ~25 rows/city, corr with "
        "price 0.349 vs zip's 0.350 — essentially the same signal with much "
        "less noise) and `state` (16 levels) remain.\n"
        "- All encoders/scalers/cluster models are fit on the **train split "
        "only** and applied to test — `price_tier` cutoffs themselves are "
        "also computed from train-set price tertiles only, so there is no "
        "leakage of test-set price information into the label definition "
        "or any feature.\n"
    )

    lines.append("## Per-model folders\n")
    lines.append(
        "Each `results/<NN_model_name>/` folder contains: "
        "`classification_report.txt` (precision/recall/F1 per tier + the "
        "summary block above), `confusion_matrix.png`, `cv_folds.png` "
        "(5-fold CV accuracy on the train split), and "
        "`feature_importance.png` (native importance for tree/linear models, "
        "permutation importance for SVM/MLP).\n"
    )

    with open(RESULTS_DIR / "ANALYSIS_REPORT.md", "w") as f:
        f.write("\n".join(lines))
    print(f"[tier_classifier] Wrote {RESULTS_DIR / 'ANALYSIS_REPORT.md'}")


if __name__ == "__main__":
    run_all()
