"""
Model explainability via SHAP.

Builds an explainer aligned with the trained regressor, renders summary, bar,
and waterfall diagnostics, and prints a concise narrative of the top drivers.
Supports Pipeline-wrapped models (SVR, MLP) via KernelExplainer fallback.
"""
from __future__ import annotations
import warnings
from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

warnings.filterwarnings("ignore", category=UserWarning)
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
PLOTS_DIR = ROOT / "plots" / "shap"
RANDOM_STATE = 42

def _load_artifacts():
    df = pd.read_csv(DATA_DIR / "processed_dataset.csv")
    model = joblib.load(MODELS_DIR / "best_model.pkl")
    rs_cols = [c for c in df.columns if c.endswith("_rs")]
    X = df[rs_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return df, model, X

def _sample_background(X: pd.DataFrame, max_rows: int = 400) -> pd.DataFrame:
    if len(X) <= max_rows:
        return X
    return X.sample(max_rows, random_state=RANDOM_STATE)

def _get_inner_estimator(model):
    """Unwrap Pipeline to get the final estimator."""
    if hasattr(model, "named_steps"):
        return list(model.named_steps.values())[-1]
    return model

def run_explainability() -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    print("[explain] Building SHAP explainers...")

    df, model, X = _load_artifacts()
    background = _sample_background(X, max_rows=300)

    inner = _get_inner_estimator(model)
    model_name = type(inner).__name__
    tree_models = {"RandomForestRegressor", "GradientBoostingRegressor", "XGBRegressor", "DecisionTreeRegressor"}

    if model_name in tree_models:
        explainer = shap.TreeExplainer(inner if not hasattr(model, "named_steps") else model)
        shap_values = explainer.shap_values(background)
        if isinstance(shap_values, list):
            shap_values = shap_values[0]
    elif model_name in {"LinearRegression", "Ridge"}:
        explainer = shap.LinearExplainer(model, background)
        shap_values = explainer.shap_values(background)
    else:
        # Fallback for SVR, MLP, etc.
        # Wrap predict in a plain function so SHAP can't set attrs on the Pipeline
        # NOTE: KernelExplainer is O(n_samples * 2^n_features) in the worst case;
        # for SVR specifically each explained row took ~37s on a single core
        # with the previous settings (100 rows -> ~1hr). Reduced background
        # k-means summary to 20 points and explained-sample count to 25, and
        # cap nsamples explicitly so this finishes in a few minutes.
        predict_fn = lambda x: model.predict(x)  # noqa: E731
        bg_summary = shap.kmeans(background, 20)
        explainer = shap.KernelExplainer(predict_fn, bg_summary)
        n_explain = min(25, len(background))
        shap_values = explainer.shap_values(background.iloc[:n_explain], nsamples=100)
        background = background.iloc[:n_explain]

    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, background, show=False)
    plt.gcf().suptitle("SHAP summary — contributions on log1p(price) model output", fontsize=12)
    plt.tight_layout(); plt.savefig(PLOTS_DIR / "shap_summary_beeswarm.png"); plt.close()

    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, background, plot_type="bar", show=False)
    plt.gcf().suptitle("SHAP bar importance — log-scale contributions", fontsize=12)
    plt.tight_layout(); plt.savefig(PLOTS_DIR / "shap_bar_importance.png"); plt.close()

    if "house_score" in df.columns:
        try:
            top_idx = int(df["house_score"].idxmax())
            row = X.loc[[top_idx]]
            row_sv = explainer.shap_values(row)
            vec = np.ravel(np.asarray(row_sv))
            base_val = explainer.expected_value
            base_scalar = float(np.ravel(np.asarray(base_val))[0])
            exp = shap.Explanation(values=vec, base_values=base_scalar,
                data=row.iloc[0].to_numpy(), feature_names=row.columns.tolist())
            shap.plots.waterfall(exp, max_display=20, show=False)
            plt.gcf().suptitle("SHAP waterfall — top HouseScore row", fontsize=11)
            plt.tight_layout(); plt.savefig(PLOTS_DIR / "shap_waterfall_top_house.png"); plt.close()
        except Exception as exc:
            print(f"[explain] Waterfall plot skipped: {exc}")

    mean_abs = np.mean(np.abs(shap_values), axis=0)
    ranking = (pd.DataFrame({"feature": background.columns, "mean_abs_shap": mean_abs})
        .sort_values("mean_abs_shap", ascending=False).head(12))
    print("\n=== Top SHAP drivers (mean |log-scale SHAP| on the background sample) ===")
    print(ranking.to_string(index=False))
    print("========================================================================\n")
    print(f"[explain] SHAP figures saved under {PLOTS_DIR}")

def main() -> None:
    run_explainability()

if __name__ == "__main__":
    main()
