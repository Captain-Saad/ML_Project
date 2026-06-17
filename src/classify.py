"""
Value-ranking score construction.

This module builds a weighted ``value_score`` from normalized signals (model
price, luxury cues, geo desirability, amenities, lot size, and new construction),
maps scores into **3 balanced bands** (Budget / Mid-Range / Premium) using 33rd
and 67th percentile cutoffs, and materializes a 0-100 ``house_score`` for
portfolio-ready ranking tables.

NOTE: ``value_tier`` (this module's output) is a *ranking/desirability* label,
not a prediction. For the genuine, held-out-evaluated price-tier
**classifier** (Tier 1/2/3 from price_value tertiles, predicted from
structural/location features), see ``src/tier_classifier.py`` and
``results/ANALYSIS_REPORT.md``. The old ``house_tier`` column has been renamed
to ``value_tier`` here specifically so it can't be mistaken for that
classifier's ``price_tier`` target.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"


def _load_processed() -> pd.DataFrame:
    path = DATA_DIR / "processed_dataset.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing processed dataset at {path}.")
    return pd.read_csv(path)


def run_classification() -> pd.DataFrame:
    """
    Compute composite scores, tiers, and house scores.

    Returns
    -------
    pd.DataFrame
        Updated listings dataframe written back to disk.
    """
    print("[classify] Loading processed dataset with predictions...")
    df = _load_processed()
    if "predicted_price" not in df.columns:
        raise RuntimeError("predicted_price is missing. Run training before classification.")

    if "lotSize_value" not in df.columns and "lot_size_sqft" in df.columns:
        df["lotSize_value"] = df["lot_size_sqft"]
    if "geo_desirability" not in df.columns:
        raise RuntimeError("geo_desirability missing; rerun preprocessing.")

    components = [
        "predicted_price",
        "luxury_score",
        "geo_desirability",
        "amenities_score",
        "lotSize_value",
    ]
    missing = [c for c in components if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing composite inputs: {missing}")

    scaler = MinMaxScaler()
    norm_matrix = scaler.fit_transform(df[components].astype(float))
    norm_cols = [f"{c}_norm" for c in components]
    df[norm_cols] = norm_matrix

    df["value_score"] = (
        0.40 * df["predicted_price_norm"]
        + 0.20 * df["luxury_score_norm"]
        + 0.15 * df["geo_desirability_norm"]
        + 0.10 * df["amenities_score_norm"]
        + 0.10 * df["lotSize_value_norm"]
        + 0.05 * df["isNewConstruction"].astype(float)
    )
    df["house_score"] = (df["value_score"] * 100.0).round(2)

    # --- 3-tier balanced split using 33rd and 67th percentile ---
    t33 = df["value_score"].quantile(0.333)
    t67 = df["value_score"].quantile(0.667)
    cuts = sorted({float(t33), float(t67)})
    strict_cuts: list[float] = []
    for val in cuts:
        if not strict_cuts or val > strict_cuts[-1]:
            strict_cuts.append(val)
        else:
            strict_cuts.append(float(np.nextafter(strict_cuts[-1], np.inf)))
    bins = [-np.inf] + strict_cuts + [np.inf]

    df["value_tier"] = pd.cut(
        df["value_score"],
        bins=bins,
        labels=["Budget", "Mid-Range", "Premium"],
    ).astype(str)

    out_path = DATA_DIR / "processed_dataset.csv"
    df.to_csv(out_path, index=False)
    print(f"[classify] Updated dataset with tiers/scores -> {out_path}")
    return df


def print_top_houses(df: pd.DataFrame, n: int = 10) -> None:
    """Print the top ``n`` listings ranked by ``house_score``."""
    cols = [
        "house_score",
        "value_tier",
        "price_value",
        "predicted_price",
        "beds",
        "baths",
        "latitude",
        "longitude",
    ]
    cols = [c for c in cols if c in df.columns]
    top = df.sort_values("house_score", ascending=False).head(n)
    print(f"\n=== Top {n} houses by HouseScore ===")
    print(top[cols].to_string(index=False))
    print("====================================\n")


def main() -> None:
    df = run_classification()
    print_top_houses(df)


if __name__ == "__main__":
    main()
