"""
src/enrich_with_kaggle.py
═══════════════════════════════════════════════════════════════════════
Enriches the existing Redfin dataset (6.8k rows, 16 US states) with
house size / sqft statistics from the Kaggle USA Real Estate Dataset.

WHY THIS APPROACH (join on city+state rather than replacing the dataset):
  - Our existing data has valuable NLP features from listingRemarks and
    keyFacts_combined (sentiment, luxury signals, amenities, etc.) that
    the Kaggle dataset does not have. Replacing the dataset entirely would
    lose all of that signal.
  - The Kaggle dataset has 2.2M listings, giving us statistically robust
    city-level sqft distributions even for smaller cities.
  - We compute median / p25 / p75 house_size and median price_per_sqft per
    (city, state), then join to our data. For cities not in Kaggle we fall
    back to the state-level median.

NEW FEATURES ADDED (all saved to data/cleaned_dataset_enriched.xlsx):
  - median_sqft        : median living area sqft for that city+state
  - p25_sqft           : 25th pct sqft (smaller-home benchmark)
  - p75_sqft           : 75th pct sqft (larger-home benchmark)
  - sqft_range         : p75 - p25 (how spread out sizes are in that city)
  - median_ppsqft      : median price-per-sqft for that city+state
  - kaggle_n           : # Kaggle listings used to compute the stats
                         (lower = less reliable; use as a confidence weight)
  - sqft_join_level    : "city" or "state" (tells you which fallback was used)

EXPECTED IMPROVEMENT:
  median_ppsqft alone should have corr ~0.4-0.6 with price_value
  (vs. our current best feature city_enc at 0.35). Combined with the
  existing NLP + structural features, expect test R² to improve from
  ~0.61 → ~0.72-0.78 and overfit gap to shrink below 0.15.

USAGE:
  # After downloading realtor-com-real-estate-listings-usa.csv from Kaggle:
  python src/enrich_with_kaggle.py --kaggle data/realtor-data.zip.csv

  # Or if you renamed it:
  python src/enrich_with_kaggle.py --kaggle data/kaggle_realestate.csv

OUTPUT:
  data/cleaned_dataset_enriched.xlsx   ← use this going forward
  data/kaggle_city_stats.csv           ← the computed join table (inspect it)

Run this ONCE, then update DATA_PATH in preprocess.py to point at the
enriched file and run the full pipeline.
═══════════════════════════════════════════════════════════════════════
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"

# ── State name → abbreviation mapping ───────────────────────────────────────
# Kaggle uses full state names; our dataset uses 2-letter abbreviations.
STATE_MAP = {
    "Alabama": "AL", "Alaska": "AK", "Arizona": "AZ", "Arkansas": "AR",
    "California": "CA", "Colorado": "CO", "Connecticut": "CT",
    "Delaware": "DE", "District of Columbia": "DC", "Florida": "FL",
    "Georgia": "GA", "Hawaii": "HI", "Idaho": "ID", "Illinois": "IL",
    "Indiana": "IN", "Iowa": "IA", "Kansas": "KS", "Kentucky": "KY",
    "Louisiana": "LA", "Maine": "ME", "Maryland": "MD",
    "Massachusetts": "MA", "Michigan": "MI", "Minnesota": "MN",
    "Mississippi": "MS", "Missouri": "MO", "Montana": "MT",
    "Nebraska": "NE", "Nevada": "NV", "New Hampshire": "NH",
    "New Jersey": "NJ", "New Mexico": "NM", "New York": "NY",
    "North Carolina": "NC", "North Dakota": "ND", "Ohio": "OH",
    "Oklahoma": "OK", "Oregon": "OR", "Pennsylvania": "PA",
    "Rhode Island": "RI", "South Carolina": "SC", "South Dakota": "SD",
    "Tennessee": "TN", "Texas": "TX", "Utah": "UT", "Vermont": "VT",
    "Virginia": "VA", "Washington": "WA", "West Virginia": "WV",
    "Wisconsin": "WI", "Wyoming": "WY",
    # Kaggle sometimes uses abbreviations directly — keep them too
    "AZ": "AZ", "CA": "CA", "CO": "CO", "DC": "DC", "FL": "FL",
    "HI": "HI", "KY": "KY", "MD": "MD", "MN": "MN", "NC": "NC",
    "NY": "NY", "OH": "OH", "OR": "OR", "PA": "PA", "SC": "SC",
    "VA": "VA",
}

# States present in our dataset
OUR_STATES = {
    "AZ", "CA", "CO", "DC", "FL", "HI",
    "KY", "MD", "MN", "NC", "NY", "OH",
    "OR", "PA", "SC", "VA",
}


def _normalise_city(s: str) -> str:
    return str(s).strip().lower()


def load_kaggle(path: str) -> pd.DataFrame:
    """Load Kaggle CSV, normalise columns, filter to our 16 states."""
    print(f"[enrich] Loading Kaggle dataset from {path} ...")
    try:
        df = pd.read_csv(path, low_memory=False)
    except Exception as e:
        sys.exit(f"[enrich] ERROR reading Kaggle CSV: {e}")

    print(f"[enrich] Raw Kaggle rows: {len(df):,}  cols: {df.columns.tolist()}")

    # ── Normalise column names ───────────────────────────────────────────
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    # Accept both naming conventions that appear in Kaggle versions
    rename = {
        "bed": "beds",
        "bath": "baths",
        "acre_lot": "lot_acres",
        "house_size": "house_sqft",     # ← the key feature
        "zip_code": "zip",
        "prev_sold_date": "sold_date",
    }
    df.rename(columns={k: v for k, v in rename.items() if k in df.columns}, inplace=True)

    # ── State normalisation ──────────────────────────────────────────────
    df["state_abbr"] = df["state"].map(STATE_MAP)
    missing_states = df["state_abbr"].isna().sum()
    if missing_states:
        unmapped = df.loc[df["state_abbr"].isna(), "state"].value_counts().head(5)
        print(f"[enrich] WARNING: {missing_states} rows have unmapped state values:")
        print(unmapped.to_string())

    df = df[df["state_abbr"].isin(OUR_STATES)].copy()
    print(f"[enrich] After filtering to {len(OUR_STATES)} states: {len(df):,} rows")

    # ── house_sqft cleaning ──────────────────────────────────────────────
    df["house_sqft"] = pd.to_numeric(df["house_sqft"], errors="coerce")
    # Remove physically impossible values
    df = df[(df["house_sqft"] >= 100) & (df["house_sqft"] <= 30_000)]
    df = df.dropna(subset=["house_sqft", "city", "state_abbr", "price"])

    # ── price cleaning ───────────────────────────────────────────────────
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df = df[(df["price"] >= 10_000) & (df["price"] <= 30_000_000)]
    df = df.dropna(subset=["price"])
    df["price_per_sqft"] = df["price"] / df["house_sqft"]
    # Remove nonsensical ppsqft
    df = df[(df["price_per_sqft"] >= 5) & (df["price_per_sqft"] <= 10_000)]

    df["city_norm"] = df["city"].apply(_normalise_city)

    # Filter to active "for_sale" listings only — our Redfin dataset is
    # active listings, so computing city stats from sold listings (which
    # skew toward older, smaller homes) would introduce a systematic bias.
    if "status" in df.columns:
        for_sale_mask = df["status"].str.lower().str.strip().isin(
            {"for_sale", "for sale", "active", "active under contract"}
        )
        n_before = len(df)
        df = df[for_sale_mask].copy()
        print(f"[enrich] Filtered to 'for_sale' status: {len(df):,} / {n_before:,} rows")
    else:
        print("[enrich] No 'status' column found — using all listings")

    print(f"[enrich] After cleaning: {len(df):,} usable rows")
    return df


def compute_city_stats(kaggle: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate per (city_norm, state_abbr):
      median / p25 / p75 sqft, median ppsqft, listing count.
    """
    print("[enrich] Computing city-level sqft statistics ...")
    grp = kaggle.groupby(["state_abbr", "city_norm"])

    stats = grp["house_sqft"].agg(
        median_sqft="median",
        p25_sqft=lambda x: x.quantile(0.25),
        p75_sqft=lambda x: x.quantile(0.75),
        kaggle_n="count",
    ).reset_index()

    ppsqft = grp["price_per_sqft"].median().reset_index()
    ppsqft.columns = ["state_abbr", "city_norm", "median_ppsqft"]

    stats = stats.merge(ppsqft, on=["state_abbr", "city_norm"], how="left")
    stats["sqft_range"] = stats["p75_sqft"] - stats["p25_sqft"]

    print(f"[enrich] City-level stats: {len(stats):,} (state, city) combinations")
    return stats


def compute_state_stats(kaggle: pd.DataFrame) -> pd.DataFrame:
    """Fallback: state-level medians for cities not in Kaggle."""
    grp = kaggle.groupby("state_abbr")
    stats = grp["house_sqft"].agg(
        median_sqft="median",
        p25_sqft=lambda x: x.quantile(0.25),
        p75_sqft=lambda x: x.quantile(0.75),
        kaggle_n="count",
    ).reset_index()
    ppsqft = grp["price_per_sqft"].median().reset_index()
    ppsqft.columns = ["state_abbr", "median_ppsqft"]
    stats = stats.merge(ppsqft, on="state_abbr", how="left")
    stats["sqft_range"] = stats["p75_sqft"] - stats["p25_sqft"]
    print(f"[enrich] State-level fallback stats: {len(stats)} states")
    return stats


def enrich(our_df: pd.DataFrame, city_stats: pd.DataFrame,
           state_stats: pd.DataFrame) -> pd.DataFrame:
    """
    Left-join city stats onto our dataset.
    Rows that don't match a city fall back to the state median.
    """
    our_df = our_df.copy()
    our_df["city_norm"] = our_df["city"].apply(_normalise_city)

    # ── City-level join ──────────────────────────────────────────────────
    stat_cols = ["state_abbr", "city_norm", "median_sqft", "p25_sqft",
                 "p75_sqft", "sqft_range", "median_ppsqft", "kaggle_n"]
    merged = our_df.merge(
        city_stats[stat_cols],
        left_on=["state", "city_norm"],
        right_on=["state_abbr", "city_norm"],
        how="left",
    )
    merged.drop(columns=["state_abbr"], errors="ignore", inplace=True)
    city_matched = merged["median_sqft"].notna().sum()
    print(f"[enrich] City-level match: {city_matched:,} / {len(merged):,} rows "
          f"({city_matched / len(merged) * 100:.1f}%)")

    # ── State-level fallback ─────────────────────────────────────────────
    state_stat_cols = ["state_abbr", "median_sqft", "p25_sqft", "p75_sqft",
                       "sqft_range", "median_ppsqft", "kaggle_n"]
    state_lkp = state_stats[state_stat_cols].rename(
        columns={c: c + "_state" for c in state_stat_cols if c != "state_abbr"}
    )
    merged = merged.merge(state_lkp, left_on="state", right_on="state_abbr", how="left")
    merged.drop(columns=["state_abbr"], errors="ignore", inplace=True)

    mask_fallback = merged["median_sqft"].isna()
    for col in ["median_sqft", "p25_sqft", "p75_sqft", "sqft_range",
                "median_ppsqft", "kaggle_n"]:
        merged.loc[mask_fallback, col] = merged.loc[mask_fallback, col + "_state"]

    merged["sqft_join_level"] = "city"
    merged.loc[mask_fallback, "sqft_join_level"] = "state"

    # Drop the state-fallback helper columns
    drop_cols = [c for c in merged.columns if c.endswith("_state")]
    merged.drop(columns=drop_cols, inplace=True)

    state_matched = mask_fallback.sum()
    print(f"[enrich] State-level fallback applied to {state_matched} rows")
    still_missing = merged["median_sqft"].isna().sum()
    if still_missing:
        print(f"[enrich] WARNING: {still_missing} rows have no sqft stats "
              f"(state not in Kaggle); filling with dataset-wide median.")
        global_median = merged["median_sqft"].median()
        merged["median_sqft"].fillna(global_median, inplace=True)
        merged["median_ppsqft"].fillna(merged["median_ppsqft"].median(), inplace=True)
        merged[["p25_sqft","p75_sqft","sqft_range","kaggle_n"]].fillna(
            merged[["p25_sqft","p75_sqft","sqft_range","kaggle_n"]].median(),
            inplace=True
        )
        merged["sqft_join_level"].fillna("global_fallback", inplace=True)

    merged.drop(columns=["city_norm"], inplace=True)
    return merged


def validate_enrichment(df: pd.DataFrame) -> None:
    """Print correlation diagnostics for the new features."""
    print("\n[enrich] === Enrichment validation ===")
    new_cols = ["median_sqft", "p25_sqft", "p75_sqft",
                "sqft_range", "median_ppsqft", "kaggle_n"]
    existing_cols = ["beds", "baths"]

    print("Correlations with price_value:")
    for col in new_cols + existing_cols:
        if col in df.columns:
            corr = df[col].corr(df["price_value"])
            bar = "█" * int(abs(corr) * 30)
            print(f"  {col:<22} {corr:+.3f}  {bar}")

    print(f"\nRows per sqft_join_level:")
    print(df["sqft_join_level"].value_counts().to_string())
    print(f"\nnew median_sqft stats:")
    print(df["median_sqft"].describe().round(0).to_string())


def main():
    parser = argparse.ArgumentParser(description="Enrich dataset with Kaggle sqft stats")
    parser.add_argument(
        "--kaggle",
        default="data/realtor-data.zip.csv",
        help="Path to Kaggle realtor CSV (default: data/realtor-data.zip.csv)",
    )
    parser.add_argument(
        "--our-data",
        default="data/cleaned_dataset.xlsx",
        help="Path to our existing dataset (default: data/cleaned_dataset.xlsx)",
    )
    parser.add_argument(
        "--out",
        default="data/cleaned_dataset_enriched.xlsx",
        help="Output path (default: data/cleaned_dataset_enriched.xlsx)",
    )
    args = parser.parse_args()

    kaggle_path = ROOT / args.kaggle
    our_path    = ROOT / args.our_data
    out_path    = ROOT / args.out

    if not kaggle_path.exists():
        sys.exit(
            f"\n[enrich] ERROR: Kaggle CSV not found at {kaggle_path}\n\n"
            "DOWNLOAD STEPS:\n"
            "  1. Go to: https://www.kaggle.com/datasets/ahmedshahriarsakib/usa-real-estate-dataset\n"
            "  2. Click the blue [Download] button (top right)\n"
            "  3. Sign in to Kaggle if prompted (free account)\n"
            "  4. You'll get: realtor-data.zip.csv  (~90 MB)\n"
            "  5. Place it in your housing_project/data/ folder\n"
            "  6. Re-run: python src/enrich_with_kaggle.py\n\n"
            "  OR via Kaggle API (if you have ~/.kaggle/kaggle.json):\n"
            "    pip install kaggle\n"
            "    cd data && kaggle datasets download -d ahmedshahriarsakib/usa-real-estate-dataset --unzip\n"
        )

    print("[enrich] Loading our dataset ...")
    if str(our_path).endswith(".xlsx"):
        our_df = pd.read_excel(our_path, engine="openpyxl")
    else:
        our_df = pd.read_csv(our_path)
    print(f"[enrich] Our data: {len(our_df):,} rows")

    kaggle = load_kaggle(str(kaggle_path))

    city_stats  = compute_city_stats(kaggle)
    state_stats = compute_state_stats(kaggle)

    # Save the join table for inspection
    city_stats_path = DATA_DIR / "kaggle_city_stats.csv"
    city_stats.to_csv(city_stats_path, index=False)
    print(f"[enrich] Saved city stats -> {city_stats_path}")

    enriched = enrich(our_df, city_stats, state_stats)
    validate_enrichment(enriched)

    enriched.to_excel(out_path, index=False, engine="openpyxl")
    print(f"\n[enrich] Saved enriched dataset -> {out_path}")
    print(f"[enrich] Shape: {enriched.shape}")
    print(f"[enrich] New columns added: median_sqft, p25_sqft, p75_sqft, "
          f"sqft_range, median_ppsqft, kaggle_n, sqft_join_level")
    print("\n[enrich] NEXT STEP: open src/preprocess.py and change DATA_PATH to:")
    print(f'  DATA_PATH = "data/cleaned_dataset_enriched.xlsx"')
    print("  Then run: python src/pipeline.py")


if __name__ == "__main__":
    main()
