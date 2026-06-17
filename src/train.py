"""
Model training, hyperparameter search, and persistence for price prediction.

This module loads the engineered dataset, builds the modeling matrix from the
robust-scaled feature columns produced during preprocessing, runs
``RandomizedSearchCV`` for **seven** regressors on ``log1p(price_value)``,
evaluates with **inverse-transformed USD metrics** (RMSE / MAE / R² on real
prices), persists the best model, and writes real-dollar ``predicted_price``
back onto ``processed_dataset.csv`` for downstream ranking and analytics.

Models trained:
  1. Linear Regression (baseline)
  2. Ridge Regression (L2 regularisation)
  3. Decision Tree Regressor
  4. Random Forest Regressor
  5. Gradient Boosting Regressor
  6. XGBoost Regressor
  7. Support Vector Regressor (SVR)
  8. MLP Neural Network (Deep Learning comparison)
"""

from __future__ import annotations

import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import (
    explained_variance_score,
    make_scorer,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*ill-conditioned.*")
warnings.filterwarnings("ignore", category=DeprecationWarning)

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"

RANDOM_STATE = 42

# Align inverse-transform clipping with preprocessing price bounds (USD).
_LOG_Y_MIN = float(np.log1p(50_000))
_LOG_Y_MAX = float(np.log1p(50_000_000))


def _log_to_price(log_vals: np.ndarray) -> np.ndarray:
    """Map model outputs on log1p(price) back to dollars with numerical stability."""
    clipped = np.clip(np.asarray(log_vals, dtype=float), _LOG_Y_MIN, _LOG_Y_MAX)
    return np.expm1(clipped)


def _neg_rmse_real_prices(y_true_log: np.ndarray, y_pred_log: np.ndarray) -> float:
    """Scorer helper: negative RMSE on real (expm1) prices for honest CV."""
    y_true_real = _log_to_price(y_true_log)
    y_pred_real = _log_to_price(y_pred_log)
    rmse = float(np.sqrt(mean_squared_error(y_true_real, y_pred_real)))
    return -rmse


REAL_RMSE_SCORER = make_scorer(_neg_rmse_real_prices, greater_is_better=True)


def _ensure_dirs() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)


def _load_processed() -> pd.DataFrame:
    path = DATA_DIR / "processed_dataset.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing processed dataset at {path}. Run preprocess first.")
    return pd.read_csv(path)


def _feature_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """Return (X, y) where ``X`` uses only robust-scaled columns."""
    rs_cols = [c for c in df.columns if c.endswith("_rs")]
    if not rs_cols:
        raise ValueError("No ``_rs`` columns found; preprocessing may have failed.")
    X = df[rs_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y = df["price_value"].to_numpy()
    return X, y


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def _mape_percent(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean absolute percentage error; ignores near-zero denominators."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.where(np.abs(y_true) < 1.0, np.nan, np.abs(y_true))
    return float(np.nanmean(np.abs((y_true - y_pred) / denom)) * 100.0)


def _adjusted_r2(r2: float, n: int, p: int) -> float:
    """Adjusted R² for ``p`` features and ``n`` samples."""
    if n - p - 1 <= 0:
        return float("nan")
    return float(1.0 - (1.0 - r2) * (n - 1) / (n - p - 1))


def _evaluate(model, X: pd.DataFrame, y_log: np.ndarray) -> dict[str, float]:
    """Evaluate on real-dollar prices after reversing the log1p transform."""
    preds_log = model.predict(X)
    preds_real = _log_to_price(preds_log)
    y_real = _log_to_price(y_log)
    n, p = len(y_real), X.shape[1]
    r2 = float(r2_score(y_real, preds_real))
    return {
        "rmse": _rmse(y_real, preds_real),
        "mae": float(mean_absolute_error(y_real, preds_real)),
        "r2": r2,
        "mape": _mape_percent(y_real, preds_real),
        "adjusted_r2": _adjusted_r2(r2, n, p),
        "explained_variance": float(explained_variance_score(y_real, preds_real)),
    }


# ── Model search configurations ──────────────────────────────────────────────

def _linear_search(X_train, y_train) -> RandomizedSearchCV:
    param_dist = {
        "fit_intercept": [True, False],
        "positive": [False],
    }
    base = LinearRegression(n_jobs=-1)
    return RandomizedSearchCV(
        base,
        param_distributions=param_dist,
        n_iter=2,
        cv=5,
        scoring=REAL_RMSE_SCORER,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=0,
    )


def _ridge_search(X_train, y_train) -> RandomizedSearchCV:
    param_dist = {
        "alpha": [0.01, 0.1, 1.0, 10.0, 100.0, 500.0],
        "fit_intercept": [True],
    }
    base = Ridge(random_state=RANDOM_STATE)
    return RandomizedSearchCV(
        base,
        param_distributions=param_dist,
        n_iter=6,
        cv=5,
        scoring=REAL_RMSE_SCORER,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=0,
    )


def _decision_tree_search(X_train, y_train) -> RandomizedSearchCV:
    # NOTE: 'None' (unlimited depth) removed from the grid - on ~5.2k training
    # rows an unconstrained tree can grow a leaf per handful of samples,
    # memorizing noise. Capping at 12 plus larger min_samples_leaf options
    # forces the tree to generalize across more rows per split.
    param_dist = {
        "max_depth": [4, 6, 8, 10, 12],
        "min_samples_split": [5, 10, 20, 40],
        "min_samples_leaf": [4, 8, 16, 32],
        "max_features": ["sqrt", "log2", 0.5, 0.8],
    }
    base = DecisionTreeRegressor(random_state=RANDOM_STATE)
    return RandomizedSearchCV(
        base,
        param_distributions=param_dist,
        n_iter=20,
        cv=5,
        scoring=REAL_RMSE_SCORER,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=0,
    )


def _random_forest_search(X_train, y_train) -> RandomizedSearchCV:
    # NOTE: previous grid's max_depth=24/None plus min_samples_leaf=1 lets
    # individual trees grow until they isolate single rows. Capping depth at
    # 16 and requiring at least 2-8 samples per leaf trades a little training
    # fit for materially better generalization.
    param_dist = {
        "n_estimators": [200, 300, 400],
        "max_depth": [6, 10, 14, 16],
        "min_samples_split": [5, 10, 20],
        "min_samples_leaf": [2, 4, 8],
        "max_features": ["sqrt", "log2", 0.3, 0.5],
    }
    base = RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1)
    return RandomizedSearchCV(
        base,
        param_distributions=param_dist,
        n_iter=8,
        cv=3,
        scoring=REAL_RMSE_SCORER,
        random_state=RANDOM_STATE,
        n_jobs=1,
        verbose=0,
    )


def _gbr_search(X_train, y_train) -> RandomizedSearchCV:
    param_dist = {
        "learning_rate": [0.01, 0.03, 0.05, 0.1],
        "n_estimators": [150, 300, 450],
        "max_depth": [2, 3, 4],
        "subsample": [0.7, 0.85, 1.0],
        "min_samples_split": [5, 10, 20],
        "min_samples_leaf": [4, 8, 16],
    }
    base = GradientBoostingRegressor(random_state=RANDOM_STATE)
    return RandomizedSearchCV(
        base,
        param_distributions=param_dist,
        n_iter=8,
        cv=3,
        scoring=REAL_RMSE_SCORER,
        random_state=RANDOM_STATE,
        n_jobs=1,
        verbose=0,
    )


def _xgb_search(X_train, y_train) -> RandomizedSearchCV:
    param_dist = {
        "n_estimators": [200, 400, 600],
        "max_depth": [3, 4, 5, 6],
        "learning_rate": [0.01, 0.03, 0.05, 0.1],
        "subsample": [0.7, 0.85, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0],
        "reg_lambda": [1.0, 5.0, 10.0],
        "min_child_weight": [1, 5, 10],
    }
    base = XGBRegressor(
        random_state=RANDOM_STATE,
        n_jobs=-1,
        tree_method="hist",
        objective="reg:squarederror",
    )
    return RandomizedSearchCV(
        base,
        param_distributions=param_dist,
        n_iter=10,
        cv=3,
        scoring=REAL_RMSE_SCORER,
        random_state=RANDOM_STATE,
        n_jobs=1,
        verbose=0,
    )


def _svr_search(X_train, y_train) -> RandomizedSearchCV:
    """SVR with RBF kernel — wrapped in a Pipeline with StandardScaler for best results.

    NOTE: the previous grid allowed C up to 100 with gamma='auto' and a tiny
    epsilon=0.01. On ~47 features that combination lets the RBF kernel fit a
    decision surface that wiggles around almost every training point
    (observed: train R2=0.9998 vs test R2=0.56 - textbook overfitting). The
    grid below caps C much lower, drops 'auto' gamma (which scales as
    1/n_features and was effectively too sharp here), and requires a larger
    epsilon so the model tolerates a margin of error instead of memorizing.
    """
    param_dist = {
        "svr__C": [0.05, 0.1, 0.5, 1.0, 5.0],
        "svr__epsilon": [0.05, 0.1, 0.2, 0.3],
        "svr__gamma": ["scale"],
    }
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("svr", SVR(kernel="rbf")),
    ])
    return RandomizedSearchCV(
        pipe,
        param_distributions=param_dist,
        n_iter=10,
        cv=3,
        scoring=REAL_RMSE_SCORER,
        random_state=RANDOM_STATE,
        n_jobs=1,
        verbose=0,
    )


def _mlp_search(X_train, y_train) -> RandomizedSearchCV:
    """MLP Neural Network (Deep Learning comparison) — StandardScaler + MLPRegressor."""
    param_dist = {
        "mlp__hidden_layer_sizes": [
            (64, 32),
            (128, 64),
            (128, 64, 32),
        ],
        "mlp__learning_rate_init": [0.001, 0.005, 0.01],
        "mlp__alpha": [0.001, 0.01, 0.05],
        "mlp__batch_size": [32, 64],
    }
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", MLPRegressor(
            activation="relu",
            solver="adam",
            max_iter=300,
            early_stopping=True,
            validation_fraction=0.1,
            random_state=RANDOM_STATE,
        )),
    ])
    return RandomizedSearchCV(
        pipe,
        param_distributions=param_dist,
        n_iter=6,
        cv=3,
        scoring=REAL_RMSE_SCORER,
        random_state=RANDOM_STATE,
        n_jobs=1,
        verbose=0,
    )


# ── Feature importance extraction ─────────────────────────────────────────────

def _model_feature_importance(
    best_name: str, model, feature_names: list[str],
    X_test: pd.DataFrame = None, y_test: np.ndarray = None,
) -> pd.DataFrame:
    """Extract importances for tree models, coefficients for linear, or permutation importance for others."""
    # If the model is a Pipeline, try to get the final estimator
    estimator = model
    if hasattr(model, "named_steps"):
        last_step_name = list(model.named_steps.keys())[-1]
        estimator = model.named_steps[last_step_name]

    if hasattr(estimator, "feature_importances_"):
        vals = estimator.feature_importances_
    elif hasattr(estimator, "coef_"):
        coef = estimator.coef_
        vals = np.abs(coef).ravel()
    elif X_test is not None and y_test is not None:
        # Permutation importance fallback (SVR, MLP, etc.)
        from sklearn.inspection import permutation_importance
        print(f"[train] Computing permutation importance for {best_name} (no native importances)...")
        perm = permutation_importance(model, X_test, y_test, n_repeats=10,
                                       random_state=RANDOM_STATE, n_jobs=-1,
                                       scoring=REAL_RMSE_SCORER)
        vals = perm.importances_mean
    else:
        vals = np.zeros(len(feature_names))

    # Handle shape mismatch
    if len(vals) != len(feature_names):
        vals = np.zeros(len(feature_names))

    out = pd.DataFrame({"feature": feature_names, "importance": vals})
    out = out.sort_values("importance", ascending=False)
    out.insert(0, "model", best_name)
    return out


def run_training() -> tuple[str, object, pd.DataFrame]:
    """
    Train all candidate models and persist the best performer.

    Returns
    -------
    tuple[str, object, pd.DataFrame]
        Name of the best model, the fitted estimator, and the updated dataframe.
    """
    _ensure_dirs()
    print("[train] Loading processed dataset...")
    df = _load_processed()
    X, y = _feature_matrix(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
    )
    y_train_log = np.log1p(y_train)
    y_test_log = np.log1p(y_test)
    print("[train] Training regressors on log1p(price); metrics reported on real-dollar prices.")

    searches = {
        "LinearRegression": _linear_search(X_train, y_train_log),
        "Ridge": _ridge_search(X_train, y_train_log),
        "DecisionTree": _decision_tree_search(X_train, y_train_log),
        "RandomForest": _random_forest_search(X_train, y_train_log),
        "GradientBoosting": _gbr_search(X_train, y_train_log),
        "XGBoost": _xgb_search(X_train, y_train_log),
        "SVR": _svr_search(X_train, y_train_log),
        "MLP_DeepLearning": _mlp_search(X_train, y_train_log),
    }

    rows = []
    trained = {}
    print("[train] Running RandomizedSearchCV for each regressor (this may take several minutes)...")
    for name, search in searches.items():
        print(f"[train] Tuning {name}...")
        search.fit(X_train, y_train_log)
        best = search.best_estimator_
        trained[name] = best
        metrics = _evaluate(best, X_test, y_test_log)
        train_metrics = _evaluate(best, X_train, y_train_log)
        row = {
            "model": name,
            **metrics,
            "train_r2": train_metrics["r2"],
            "train_rmse": train_metrics["rmse"],
            "overfit_gap_r2": train_metrics["r2"] - metrics["r2"],
            "best_params": str(search.best_params_),
        }
        rows.append(row)
        print(
            f"    -> val RMSE={metrics['rmse']:.2f} | MAE={metrics['mae']:.2f} | R2={metrics['r2']:.4f} | "
            f"MAPE={metrics['mape']:.2f}% | adjR2={metrics['adjusted_r2']:.4f} | "
            f"train R2={train_metrics['r2']:.4f} | overfit gap={row['overfit_gap_r2']:.4f}"
        )

    metrics_df = pd.DataFrame(rows).sort_values("rmse", ascending=True)
    metrics_path = REPORTS_DIR / "model_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    print(f"[train] Saved metrics -> {metrics_path}")

    best_name = metrics_df.iloc[0]["model"]
    best_model = trained[best_name]
    print(f"[train] Best model selected: {best_name}")

    preds_real = _log_to_price(best_model.predict(X))
    df["predicted_price"] = preds_real
    # NOTE: predicted_price is computed in-sample (full X, train+test) purely
    # for the downstream `value_score`/`value_tier` RANKING tool in
    # classify.py - it is a portfolio-ranking convenience, not a metric. Do
    # not use predicted_price (or anything derived from it) as a measure of
    # model accuracy; use reports/model_metrics.csv (test-only) for that, and
    # reports/tier_classifier_metrics.csv for the genuine tier classifier.

    processed_path = DATA_DIR / "processed_dataset.csv"
    df.to_csv(processed_path, index=False)
    print(f"[train] Wrote in-sample predictions to {processed_path}")

    joblib.dump(best_model, MODELS_DIR / "best_model.pkl")

    feature_names = list(X.columns)
    importance = _model_feature_importance(best_name, best_model, feature_names,
                                           X_test=X_test, y_test=y_test_log)
    importance_path = REPORTS_DIR / "feature_importance.csv"
    importance.to_csv(importance_path, index=False)
    print(f"[train] Saved feature importance -> {importance_path}")

    return best_name, best_model, df


def print_comparison_table(df_metrics: pd.DataFrame) -> None:
    """Pretty-print the model comparison table for console output."""
    printable = df_metrics.drop(columns=["best_params"], errors="ignore")
    print("\n=== Model comparison (hold-out test set, real USD prices) ===")
    print(printable.to_string(index=False, float_format=lambda x: f"{x:,.4f}"))
    print("============================================\n")


def main() -> None:
    best_name, _, df = run_training()
    metrics = pd.read_csv(REPORTS_DIR / "model_metrics.csv")
    print_comparison_table(metrics)
    print(f"[train] Best model on RMSE: {best_name}")
    _ = df  # noqa: F841


if __name__ == "__main__":
    main()
