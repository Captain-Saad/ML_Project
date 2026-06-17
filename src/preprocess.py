"""
Data preprocessing and feature engineering for the housing analytics pipeline.

This module loads the cleaned listing dataset, applies validity/outlier filters,
imputes missing values, engineers numeric and geospatial features without using
``price_value`` as a direct model input, extracts NLP-derived signals, fits
TF-IDF + SVD on training text only, fits target encoding **only on training rows**
(using ``log1p(price)`` as the encoding target to align with the log-price
regressor), fits a robust scaler on the training split, and writes
``processed_dataset.csv`` plus serialized preprocessors for downstream training.
"""

from __future__ import annotations

import re
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from category_encoders import TargetEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from textblob import TextBlob

warnings.filterwarnings("ignore", category=UserWarning)

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"

# Reproducibility for every stochastic step in this module.
RANDOM_STATE = 42

# Columns that must be present and finite after outlier filtering (programmatic).
_CANDIDATE_STRUCTURE_COLS = [
    "beds",
    "baths",
    "fullBaths",
    "lotSize_value",
    "latitude",
    "longitude",
    "city",
    "state",
    "zip",
    "listingRemarks",
    "keyFacts_combined",
]

# Keyword lexicons for interpretable NLP-derived features.
LUXURY_KWS = [
    "luxury",
    "penthouse",
    "marble",
    "concierge",
    "doorman",
    "white-glove",
    "white glove",
    "high-end",
    "high end",
    "designer",
    "prestigious",
    "exclusive",
    "custom-built",
    "custom built",
]
RENOVATION_KWS = [
    "renovated",
    "updated",
    "remodeled",
    "modern",
    "new kitchen",
    "upgraded",
]
VIEW_KWS = [
    "waterfront",
    "ocean view",
    "skyline",
    "river view",
    "lake view",
    "city view",
]
AMENITY_KWS = [
    "pool",
    "gym",
    "garage",
    "parking",
    "rooftop",
    "terrace",
    "balcony",
    "spa",
    "fireplace",
    "elevator",
    "garden",
]
CONDITION_KWS = [
    "move-in ready",
    "move in ready",
    "pristine",
    "mint condition",
    "turnkey",
]


def _project_paths() -> None:
    """Ensure expected folders exist before writing artifacts."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)


def _load_raw_frame() -> pd.DataFrame:
    """
    Load the raw dataset from CSV (preferred) or XLSX under ``data/``.

    Returns
    -------
    pd.DataFrame
        Raw listings matching the project schema.
    """
    csv_path = DATA_DIR / "cleaned_dataset.csv"
    xlsx_path = DATA_DIR / "cleaned_dataset.xlsx"
    enriched_path = DATA_DIR / "cleaned_dataset_enriched.xlsx"

    # Prefer enriched dataset (has Kaggle sqft features) if it exists
    if enriched_path.exists():
        print(f"[preprocess] Using enriched dataset (with Kaggle sqft): {enriched_path.name}")
        return pd.read_excel(enriched_path, engine="openpyxl")
    if csv_path.exists():
        print(f"[preprocess] Using: {csv_path.name}")
        return pd.read_csv(csv_path)
    if xlsx_path.exists():
        print(f"[preprocess] Using: {xlsx_path.name}")
        return pd.read_excel(xlsx_path, engine="openpyxl")
    raise FileNotFoundError(
        "Place ``cleaned_dataset.csv`` or ``cleaned_dataset.xlsx`` inside the "
        f"``data`` folder: {DATA_DIR}"
    )


def _safe_str(value: object) -> str:
    """Convert arbitrary cell values to a lowercase string for NLP heuristics."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    return str(value).lower()


def _count_keyword_hits(text: str, keywords: list[str]) -> int:
    """Count keyword hits; phrases are matched as substrings."""
    if not text:
        return 0
    return sum(1 for kw in keywords if kw in text)


def _flag_keyword_hits(text: str, keywords: list[str]) -> int:
    """Binary flag (0/1) if any keyword from the list appears."""
    return int(any(kw in text for kw in keywords) if text else 0)


def _extract_hoa_value(text: str) -> float:
    """
    Extract a numeric HOA fee when present in ``keyFacts_combined``.

    The heuristic focuses on common US listing phrasing like ``HOA: $450/mo``.
    """
    if not text:
        return 0.0
    m = re.search(r"hoa[^$]{0,40}\$\s*([\d,]+)", text, flags=re.IGNORECASE)
    if not m:
        m = re.search(r"hoa\s*[:\-]\s*([\d,]+)", text, flags=re.IGNORECASE)
    if m:
        try:
            return float(m.group(1).replace(",", ""))
        except ValueError:
            return 0.0
    return 0.0


def _lot_size_sqft(row: pd.Series) -> float:
    """
    Normalize lot size to square feet using ``keyFacts_combined`` cues.

    If the facts mention acres without a conflicting explicit sqft cue, values are
    treated as acres and converted using the standard 43,560 sqft per acre factor.
    """
    raw = row.get("lotSize_value", np.nan)
    facts = _safe_str(row.get("keyFacts_combined", ""))
    if pd.isna(raw):
        return 0.0
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return 0.0
    if value == 0:
        return 0.0

    acre_hit = bool(re.search(r"\bacres?\b", facts))
    sqft_hit = bool(re.search(r"(sq\.?\s*ft\.?|square\s*feet|sqft)\b", facts))

    if acre_hit and not sqft_hit:
        return value * 43560.0
    if acre_hit and sqft_hit:
        # Ambiguous marketing copy: prefer the explicit sqft cue if both appear.
        return value
    return value


def _sentiment_polarity(text: str) -> float:
    """Return TextBlob polarity for ``listingRemarks`` in [-1, 1]."""
    if not text.strip():
        return 0.0
    try:
        return float(TextBlob(text).sentiment.polarity)
    except Exception:
        return 0.0


def _add_nlp_keyword_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create interpretable keyword-based features from free text fields."""
    remarks = df["listingRemarks"].map(_safe_str)
    facts = df["keyFacts_combined"].map(_safe_str)
    combined = remarks + " " + facts

    df["luxury_score"] = combined.map(lambda t: _count_keyword_hits(t, LUXURY_KWS))
    df["renovation_flag"] = combined.map(lambda t: _flag_keyword_hits(t, RENOVATION_KWS))
    df["view_flag"] = combined.map(lambda t: _flag_keyword_hits(t, VIEW_KWS))
    df["amenities_score"] = combined.map(lambda t: _count_keyword_hits(t, AMENITY_KWS))
    df["condition_flag"] = combined.map(lambda t: _flag_keyword_hits(t, CONDITION_KWS))
    df["hoa_value"] = df["keyFacts_combined"].map(lambda x: _extract_hoa_value(_safe_str(x)))
    df["sentiment_score"] = df["listingRemarks"].map(lambda x: _sentiment_polarity(str(x)))
    return df


def _impute_basic(df: pd.DataFrame) -> pd.DataFrame:
    """Median imputation for numerics, mode imputation for categoricals."""
    out = df.copy()
    numeric_cols = out.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        med = out[col].median()
        out[col] = out[col].fillna(med)

    cat_cols = out.select_dtypes(include=["object", "category"]).columns
    for col in cat_cols:
        mode = out[col].mode(dropna=True)
        fill = mode.iloc[0] if len(mode) else ""
        out[col] = out[col].fillna(fill)
    return out


def _coerce_bools(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize boolean columns to integer {0,1} indicators."""
    for col in ["isNewConstruction", "isHot"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.lower().isin(["true", "1", "yes"]).astype(int)
    return df


def _fill_text_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Replace missing text with empty strings to keep downstream NLP stable."""
    for col in ["listingRemarks", "keyFacts_combined"]:
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str)
    return df


def run_preprocess() -> pd.DataFrame:
    """
    Execute the full preprocessing stack.

    Returns
    -------
    pd.DataFrame
        The engineered dataset (without model predictions).
    """
    _project_paths()
    print("[preprocess] Loading raw dataset...")
    df = _load_raw_frame()

    if "url" in df.columns:
        df = df.drop(columns=["url"])

    print("[preprocess] Coercing types and cleaning raw text fields...")
    df = _fill_text_columns(df)
    df = _coerce_bools(df)

    print("[preprocess] Applying outlier / validity filters (before train/test split)...")
    initial_count = len(df)
    df = df.replace([np.inf, -np.inf], np.nan)

    feature_cols_exist = [c for c in _CANDIDATE_STRUCTURE_COLS if c in df.columns]
    dropna_subset = feature_cols_exist + ["price_value"]

    df = df[
        (df["beds"] > 0)
        & (df["beds"] <= 15)
        & (df["baths"] > 0)
        & (df["baths"] <= 15)
        & (df["price_value"] >= 50_000)
        & (df["price_value"] <= 3_000_000)   # capped from $50M — removes 119 luxury
        & (df["lotSize_value"] > 0)           # outliers (1.8% of rows) that cause linear
    ]                                         # models to collapse and inflate overfit gaps
    df = df.dropna(subset=dropna_subset)
    removed = initial_count - len(df)
    print(f"[preprocess] Removed {removed} outlier/invalid rows. Remaining: {len(df)}")

    print("[preprocess] Imputing missing values on the filtered frame...")
    df = _impute_basic(df)

    # Target encoding expects categorical-compatible dtypes.
    for col in ["city", "state", "zip"]:
        if col in df.columns:
            df[col] = df[col].astype(str)

    # Safe structural ratios (avoid division by zero on beds). Intentionally avoid
    # any feature derived from ``price_value`` (e.g., price-per-bed proxies).
    beds_safe = df["beds"].replace(0, np.nan).fillna(1.0)
    df["bath_bed_ratio"] = df["baths"] / beds_safe
    df["total_rooms"] = df["beds"] + df["baths"]

    print("[preprocess] Normalizing lot sizes using key facts...")
    df["lot_size_sqft"] = df.apply(_lot_size_sqft, axis=1)

    print("[preprocess] Deriving NLP keyword and sentiment features...")
    df = _add_nlp_keyword_features(df)

    # Train/test indices are reused downstream for modeling and reporting.
    train_idx, test_idx = train_test_split(
        np.arange(len(df)),
        test_size=0.2,
        random_state=RANDOM_STATE,
    )
    train_mask = np.zeros(len(df), dtype=bool)
    train_mask[train_idx] = True

    print("[preprocess] Fitting geospatial clustering on the training split...")
    geo = df.loc[train_mask, ["latitude", "longitude"]].to_numpy()
    kmeans = KMeans(n_clusters=20, random_state=RANDOM_STATE, n_init="auto")
    kmeans.fit(geo)
    df["geo_cluster"] = kmeans.predict(df[["latitude", "longitude"]].to_numpy())

    # Geo desirability proxy without using ``price_value``: local listing density.
    df["geo_desirability"] = df.groupby("geo_cluster")["geo_cluster"].transform("count")

    # Structural density on lot size (does not use list price).
    lot_safe = df["lot_size_sqft"].replace(0, np.nan).fillna(1.0)
    df["room_density_score"] = df["total_rooms"] / lot_safe

    print("[preprocess] Fitting TF-IDF + TruncatedSVD on training text only...")
    text_train = (
        df.loc[train_mask, "listingRemarks"].astype(str)
        + " "
        + df.loc[train_mask, "keyFacts_combined"].astype(str)
    )
    text_all = df["listingRemarks"].astype(str) + " " + df["keyFacts_combined"].astype(str)

    tfidf = TfidfVectorizer(
        max_features=100,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2,
    )
    tfidf.fit(text_train)
    X_tfidf = tfidf.transform(text_all)

    svd = TruncatedSVD(n_components=20, random_state=RANDOM_STATE)
    svd.fit(tfidf.transform(text_train))
    X_svd = svd.transform(X_tfidf)

    for i in range(1, 21):
        df[f"nlp_dim_{i}"] = X_svd[:, i - 1]

    print("[preprocess] Fitting target encoders (city/state) on training rows only...")
    # NOTE: ``zip`` (704 unique values, ~9 rows/zip on average) and raw
    # ``latitude``/``longitude`` are intentionally excluded from the model
    # feature matrix below. With ~6.5k rows, a 704-level target encoding is
    # extremely noisy (high-variance per-zip means) and raw lat/long let
    # high-capacity models (RBF-SVR, deep trees) memorize individual
    # property locations instead of learning generalizable spatial patterns.
    # ``city`` (266 levels, ~25 rows/city) carries essentially the same
    # correlation with price as ``zip`` (0.35 vs 0.349) with far less noise,
    # and ``geo_cluster``/``geo_desirability`` (below) already summarize the
    # lat/long signal in a coarser, train-fit, generalizable form.
    te_cols = ["city", "state"]
    target_encoder = TargetEncoder(cols=te_cols, smoothing=10.0)
    y_train_te = np.log1p(df.loc[train_mask, "price_value"].astype(float).to_numpy())
    target_encoder.fit(df.loc[train_mask, te_cols], y_train_te)
    df[te_cols] = target_encoder.transform(df[te_cols])
    print("[preprocess] Target encoder fit on training data only - no leakage.")

    print("[preprocess] Building genuine price-tier classification target...")
    # Ground-truth Tier label for the classification task (Tier 1/2/3),
    # independent of any model prediction. Cutoffs are computed on the
    # TRAIN split only and then applied to the full frame, so the test
    # split's tier boundaries can't leak from test-set price values.
    train_prices = df.loc[train_mask, "price_value"].astype(float)
    tier_cut_low = float(train_prices.quantile(1 / 3))
    tier_cut_high = float(train_prices.quantile(2 / 3))
    df["price_tier"] = pd.cut(
        df["price_value"].astype(float),
        bins=[-np.inf, tier_cut_low, tier_cut_high, np.inf],
        labels=[1, 2, 3],
    ).astype(int)
    df["is_train"] = train_mask
    print(
        f"[preprocess] price_tier cutoffs (train-derived): "
        f"Tier1 <= ${tier_cut_low:,.0f} < Tier2 <= ${tier_cut_high:,.0f} < Tier3"
    )

    print("[preprocess] Fitting robust scaler on training feature matrix...")
    nlp_cols = [f"nlp_dim_{i}" for i in range(1, 21)]
    # ``latitude``/``longitude``/``zip`` deliberately excluded - see note above
    # the target-encoder block. geo_cluster + geo_desirability + city + state
    # already cover the spatial signal with far less redundancy/overfitting risk.
    #
    # NEW (post-Kaggle enrichment): median_sqft, p25_sqft, p75_sqft, sqft_range,
    # median_ppsqft are included when present (enriched dataset only). They are
    # the single biggest missing signal in the original dataset — living area sqft
    # explains 40-60% of price variance in most US real estate datasets.
    structural_cols = [
        c
        for c in [
            "baths",
            "beds",
            "fullBaths",
            "lot_size_sqft",
            "isNewConstruction",
            "isHot",
            "bath_bed_ratio",
            "total_rooms",
            "geo_cluster",
            "geo_desirability",
            "room_density_score",
            "luxury_score",
            "renovation_flag",
            "view_flag",
            "amenities_score",
            "condition_flag",
            "hoa_value",
            "sentiment_score",
            "city",
            "state",
            # ── Kaggle-enriched sqft features (present only after running
            #    src/enrich_with_kaggle.py and switching to enriched dataset) ──
            "median_sqft",
            "p25_sqft",
            "p75_sqft",
            "sqft_range",
            "median_ppsqft",
        ]
        if c in df.columns
    ]
    feature_cols = structural_cols + [c for c in nlp_cols if c in df.columns]

    scaler = RobustScaler()
    scaler.fit(df.loc[train_mask, feature_cols].to_numpy())
    scaled = scaler.transform(df[feature_cols].to_numpy())
    scaled_df = pd.DataFrame(scaled, columns=[f"{c}_rs" for c in feature_cols], index=df.index)
    df = pd.concat([df, scaled_df], axis=1)

    processed_path = DATA_DIR / "processed_dataset.csv"
    print(f"[preprocess] Writing processed dataset -> {processed_path}")
    df.to_csv(processed_path, index=False)

    joblib.dump(scaler, MODELS_DIR / "scaler.pkl")
    joblib.dump(target_encoder, MODELS_DIR / "target_encoder.pkl")
    joblib.dump(tfidf, MODELS_DIR / "tfidf.pkl")
    joblib.dump(svd, MODELS_DIR / "svd.pkl")
    joblib.dump(kmeans, MODELS_DIR / "geo_kmeans.pkl")
    # Column order for ``RobustScaler`` / model matrix (no ``feature_names_in_`` on array fit).
    joblib.dump(feature_cols, MODELS_DIR / "scaler_feature_names.pkl")
    joblib.dump(feature_cols, MODELS_DIR / "feature_columns.pkl")

    print("[preprocess] Done.")
    return df


def main() -> None:
    run_preprocess()


if __name__ == "__main__":
    main()