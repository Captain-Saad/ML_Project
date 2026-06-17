"""
Pricing anomaly detection and investment opportunity scoring.

This module compares model predictions to observed list prices, derives an
``investment_opportunity_score`` that emphasizes undervalued listings, and exports
a curated CSV of high-signal opportunities for further review.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.preprocessing import MinMaxScaler

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
REPORTS_DIR = ROOT / "reports"


def _load_processed() -> pd.DataFrame:
    path = DATA_DIR / "processed_dataset.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing processed dataset at {path}.")
    return pd.read_csv(path)


def run_anomaly_detection() -> pd.DataFrame:
    """
    Annotate listings with pricing gaps and opportunity scores.

    Returns
    -------
    pd.DataFrame
        Updated dataframe persisted to ``processed_dataset.csv``.
    """
    print("[anomaly] Scoring pricing gaps and investment opportunities...")
    df = _load_processed()
    if "predicted_price" not in df.columns:
        raise RuntimeError("predicted_price missing; train the regressor first.")

    df["pricing_gap_pct"] = (df["predicted_price"] - df["price_value"]) / df["price_value"]

    undervalued = df[df["pricing_gap_pct"] > 0].copy()
    scaler = MinMaxScaler()
    if not undervalued.empty:
        undervalued["investment_opportunity_score"] = (
            scaler.fit_transform(undervalued[["pricing_gap_pct"]]) * 100.0
        )
    df["investment_opportunity_score"] = 0.0
    if not undervalued.empty:
        df.loc[undervalued.index, "investment_opportunity_score"] = undervalued[
            "investment_opportunity_score"
        ]

    df["undervalued_flag"] = (df["pricing_gap_pct"] > 0).astype(int)
    df["overpriced_flag"] = (df["pricing_gap_pct"] < 0).astype(int)

    processed_path = DATA_DIR / "processed_dataset.csv"
    df.to_csv(processed_path, index=False)
    print(f"[anomaly] Saved enriched dataset -> {processed_path}")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    export_cols = [
        c
        for c in [
            "house_score",
            "value_tier",
            "price_value",
            "predicted_price",
            "pricing_gap_pct",
            "investment_opportunity_score",
            "undervalued_flag",
            "beds",
            "baths",
            "lot_size_sqft",
            "latitude",
            "longitude",
            "luxury_score",
            "amenities_score",
        ]
        if c in df.columns
    ]
    ranked = df.sort_values(["investment_opportunity_score", "house_score"], ascending=False).head(
        200
    )
    ranked_path = REPORTS_DIR / "top_ranked_houses.csv"
    ranked[export_cols].to_csv(ranked_path, index=False)
    print(f"[anomaly] Wrote opportunity leaderboard -> {ranked_path}")
    return df


def print_undervalued_opportunities(df: pd.DataFrame, n: int = 10) -> None:
    """Print undervalued listings ranked by percentage pricing gap."""
    und = df[df["pricing_gap_pct"] > 0].sort_values("pricing_gap_pct", ascending=False).head(n)
    cols = [
        "pricing_gap_pct",
        "investment_opportunity_score",
        "price_value",
        "predicted_price",
        "house_score",
        "beds",
        "baths",
    ]
    cols = [c for c in cols if c in und.columns]
    print(f"\n=== Top {n} undervalued opportunities (by pricing_gap_pct) ===")
    if und.empty:
        print("No undervalued rows detected with the current thresholds.")
    else:
        print(und[cols].to_string(index=False))
    print("==========================================================\n")


def main() -> None:
    df = run_anomaly_detection()
    print_undervalued_opportunities(df)


if __name__ == "__main__":
    main()
