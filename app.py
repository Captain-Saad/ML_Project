"""
Intelligent Housing Analysis System — Streamlit dashboard.

Loads trained pipeline artifacts (no retraining), supports single-property
prediction with feature alignment via ``models/feature_columns.pkl``, and
surfaces analytics, maps, and evaluation views.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
from sklearn.preprocessing import MinMaxScaler
from textblob import TextBlob

# ---------------------------------------------------------------------------
# Paths (``streamlit run app.py`` from ``housing_project/``)
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
PLOTS_DIR = BASE_DIR / "plots"
REPORTS_DIR = BASE_DIR / "reports"
PLOTS_SHAP = PLOTS_DIR / "shap"
PLOTS_EVAL = PLOTS_DIR / "evaluation"

RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# NLP / text heuristics — must match ``src/preprocess.py`` lexicons exactly.
# ---------------------------------------------------------------------------
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


def _safe_str(value: object) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    return str(value).lower()


def _count_keyword_hits(text: str, keywords: list[str]) -> int:
    if not text:
        return 0
    return sum(1 for kw in keywords if kw in text)


def _flag_keyword_hits(text: str, keywords: list[str]) -> int:
    return int(any(kw in text for kw in keywords) if text else 0)


def _extract_hoa_value(text: str) -> float:
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


def _lot_size_sqft_scalar(lot_val: float, key_facts: str) -> float:
    """Normalize lot size to sqft using the same rules as ``preprocess._lot_size_sqft``."""
    facts = _safe_str(key_facts)
    try:
        value = float(lot_val)
    except (TypeError, ValueError):
        return 0.0
    if value == 0 or np.isnan(value):
        return 0.0
    acre_hit = bool(re.search(r"\bacres?\b", facts))
    sqft_hit = bool(re.search(r"(sq\.?\s*ft\.?|square\s*feet|sqft)\b", facts))
    if acre_hit and not sqft_hit:
        return value * 43560.0
    return value


def _sentiment_polarity(text: str) -> float:
    if not str(text).strip():
        return 0.0
    try:
        return float(TextBlob(str(text)).sentiment.polarity)
    except Exception:
        return 0.0


def nlp_feature_dict(listing_remarks: str, key_facts: str) -> dict[str, float | int]:
    remarks = _safe_str(listing_remarks)
    facts = _safe_str(key_facts)
    combined = remarks + " " + facts
    return {
        "luxury_score": float(_count_keyword_hits(combined, LUXURY_KWS)),
        "renovation_flag": int(_flag_keyword_hits(combined, RENOVATION_KWS)),
        "view_flag": int(_flag_keyword_hits(combined, VIEW_KWS)),
        "amenities_score": float(_count_keyword_hits(combined, AMENITY_KWS)),
        "condition_flag": int(_flag_keyword_hits(combined, CONDITION_KWS)),
        "hoa_value": float(_extract_hoa_value(facts)),
        "sentiment_score": float(_sentiment_polarity(str(listing_remarks))),
    }


# ---------------------------------------------------------------------------
# Log-price inverse (matches ``src/train.py`` clipping)
# ---------------------------------------------------------------------------
_LOG_Y_MIN = float(np.log1p(50_000))
_LOG_Y_MAX = float(np.log1p(50_000_000))


def log_to_price(log_vals: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(log_vals, dtype=float), _LOG_Y_MIN, _LOG_Y_MAX)
    return np.expm1(clipped)


# ---------------------------------------------------------------------------
# Cached artifact loading
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=True)
def load_artifacts() -> dict[str, Any]:
    """Load models, encoders, reference dataframe, and cluster counts."""
    out: dict[str, Any] = {}
    out["model"] = joblib.load(MODELS_DIR / "best_model.pkl")
    out["scaler"] = joblib.load(MODELS_DIR / "scaler.pkl")
    out["target_encoder"] = joblib.load(MODELS_DIR / "target_encoder.pkl")
    out["tfidf"] = joblib.load(MODELS_DIR / "tfidf.pkl")
    out["svd"] = joblib.load(MODELS_DIR / "svd.pkl")
    fc_path = MODELS_DIR / "feature_columns.pkl"
    if not fc_path.exists():
        raise FileNotFoundError("Missing models/feature_columns.pkl — run src/preprocess.py (or the full pipeline).")
    out["feature_columns"] = list(joblib.load(fc_path))
    k_path = MODELS_DIR / "geo_kmeans.pkl"
    if k_path.exists():
        out["kmeans"] = joblib.load(k_path)
    else:
        out["kmeans"] = None

    proc = DATA_DIR / "processed_dataset.csv"
    if proc.exists():
        out["ref_df"] = pd.read_csv(proc)
    else:
        out["ref_df"] = pd.DataFrame()

    if len(out["ref_df"]) and "geo_cluster" in out["ref_df"].columns:
        out["cluster_counts"] = out["ref_df"].groupby("geo_cluster").size().to_dict()
    else:
        out["cluster_counts"] = {}

    mpath = REPORTS_DIR / "model_metrics.csv"
    if mpath.exists():
        out["model_metrics"] = pd.read_csv(mpath)
        best = out["model_metrics"].sort_values("rmse").iloc[0]["model"]
        out["best_model_name"] = best
        r2 = float(out["model_metrics"].sort_values("rmse").iloc[0].get("r2", np.nan))
        out["best_r2"] = r2
    else:
        out["model_metrics"] = pd.DataFrame()
        out["best_model_name"] = "N/A"
        out["best_r2"] = float("nan")

    return out


def scaler_transform_row(scaler, feature_columns: list[str], raw_values: dict[str, Any]) -> pd.DataFrame:
    row = np.zeros((1, len(feature_columns)), dtype=float)
    for j, col in enumerate(feature_columns):
        v = raw_values.get(col, 0.0)
        try:
            vec = 0.0 if v is None or (isinstance(v, float) and np.isnan(v)) else float(v)
        except (TypeError, ValueError):
            vec = 0.0
        row[0, j] = vec
    scaled = scaler.transform(row)
    rs_names = [f"{c}_rs" for c in feature_columns]
    return pd.DataFrame(scaled, columns=rs_names)


def build_raw_feature_dict(
    artifacts: dict[str, Any],
    city: str,
    state: str,
    zip_code: str,
    beds: float,
    baths: float,
    full_baths: float,
    lot_size: float,
    lat: float,
    lon: float,
    is_new: int,
    listing_remarks: str,
    key_facts: str,
) -> dict[str, Any]:
    """Replicate preprocessing feature construction for one listing (no leakage from target)."""
    beds_safe = max(float(beds), 1.0)
    bath_bed_ratio = float(baths) / beds_safe
    total_rooms = float(beds) + float(baths)
    lot_sq = _lot_size_sqft_scalar(float(lot_size), key_facts)
    lot_safe = max(lot_sq, 1e-9)

    nlp = nlp_feature_dict(listing_remarks, key_facts)

    kmeans = artifacts.get("kmeans")
    if kmeans is not None:
        geo_cluster = int(kmeans.predict(np.array([[lat, lon]], dtype=float))[0])
    else:
        geo_cluster = 0

    cluster_counts: dict[int, int] = artifacts.get("cluster_counts", {})
    geo_desirability = float(cluster_counts.get(geo_cluster, 1))

    room_density_score = total_rooms / lot_safe

    te = artifacts["target_encoder"]
    # NOTE: the target encoder is now fit on ["city", "state"] only (see
    # preprocess.py) - zip was dropped (704 levels, ~9 rows/zip on average,
    # too sparse to encode reliably on a 6.5k-row dataset). zip_code is kept
    # as a UI input for context but no longer feeds the model.
    te_df = pd.DataFrame({"city": [str(city)], "state": [str(state)]})
    te_out = te.transform(te_df)
    city_te = float(te_out["city"].iloc[0])
    state_te = float(te_out["state"].iloc[0])

    combined_text = f"{listing_remarks} {key_facts}"
    x_tfidf = artifacts["tfidf"].transform([combined_text])
    nlp_svd = artifacts["svd"].transform(x_tfidf).ravel()

    feats: dict[str, Any] = {
        "baths": float(baths),
        "beds": float(beds),
        "fullBaths": float(full_baths),
        "lot_size_sqft": float(lot_sq),
        "isNewConstruction": int(is_new),
        "bath_bed_ratio": float(bath_bed_ratio),
        "total_rooms": float(total_rooms),
        "geo_cluster": float(geo_cluster),
        "geo_desirability": geo_desirability,
        "room_density_score": float(room_density_score),
        "luxury_score": nlp["luxury_score"],
        "renovation_flag": nlp["renovation_flag"],
        "view_flag": nlp["view_flag"],
        "amenities_score": nlp["amenities_score"],
        "condition_flag": nlp["condition_flag"],
        "hoa_value": nlp["hoa_value"],
        "sentiment_score": nlp["sentiment_score"],
        "city": city_te,
        "state": state_te,
        # latitude/longitude/zip intentionally excluded from model features -
        # see preprocess.py notes. lat/lon are still used above for
        # geo_cluster/geo_desirability.
    }
    for i in range(1, 21):
        feats[f"nlp_dim_{i}"] = float(nlp_svd[i - 1]) if i - 1 < len(nlp_svd) else 0.0
    return feats


def value_score_and_tier(ref_df: pd.DataFrame, pred_price: float, feats: dict[str, Any]) -> tuple[float, str, float]:
    """Apply the same value-score + tier logic as ``classify.py`` using reference + one new row.

    NOTE: this produces a *ranking* tier (``value_tier``: Budget/Mid-Range/
    Premium based on a weighted desirability composite), not the genuine
    price-tier **classifier** prediction from ``src/tier_classifier.py``.
    """
    if "lotSize_value" not in ref_df.columns and "lot_size_sqft" in ref_df.columns:
        ref_work = ref_df.copy()
        ref_work["lotSize_value"] = ref_work["lot_size_sqft"]
    else:
        ref_work = ref_df.copy()

    row = {
        "predicted_price": float(pred_price),
        "luxury_score": float(feats.get("luxury_score", 0.0)),
        "geo_desirability": float(feats.get("geo_desirability", 0.0)),
        "amenities_score": float(feats.get("amenities_score", 0.0)),
        "lotSize_value": float(feats.get("lot_size_sqft", 0.0)),
        "isNewConstruction": float(feats.get("isNewConstruction", 0.0)),
    }
    tail = pd.DataFrame([row])
    cols = ["predicted_price", "luxury_score", "geo_desirability", "amenities_score", "lotSize_value"]
    base = ref_work[cols + ["isNewConstruction"]].copy()
    comb = pd.concat([base, tail], ignore_index=True)

    scaler = MinMaxScaler()
    norm = scaler.fit_transform(comb[cols].astype(float))
    tmp = pd.DataFrame(norm, columns=[f"{c}_norm" for c in cols], index=comb.index)
    composite = (
        0.40 * tmp["predicted_price_norm"]
        + 0.20 * tmp["luxury_score_norm"]
        + 0.15 * tmp["geo_desirability_norm"]
        + 0.10 * tmp["amenities_score_norm"]
        + 0.10 * tmp["lotSize_value_norm"]
        + 0.05 * comb["isNewConstruction"].astype(float)
    )
    score_tail = float(composite.iloc[-1] * 100.0)
    t33 = composite.quantile(0.333)
    t67 = composite.quantile(0.667)
    cuts = sorted({float(t33), float(t67)})
    strict_cuts: list[float] = []
    for val in cuts:
        if not strict_cuts or val > strict_cuts[-1]:
            strict_cuts.append(val)
        else:
            strict_cuts.append(float(np.nextafter(strict_cuts[-1], np.inf)))
    bins = [-np.inf] + strict_cuts + [np.inf]
    tier_ser = pd.cut(composite, bins=bins, labels=["Budget", "Mid-Range", "Premium"]).astype(str)
    tier_tail = str(tier_ser.iloc[-1])
    return float(composite.iloc[-1]), tier_tail, round(score_tail, 2)


def tier_badge_color(tier: str) -> str:
    if tier == "Premium":
        return "#D4AF37"
    if tier == "Mid-Range":
        return "#4A90D9"
    return "#888888"


def show_plot_if_exists(path: Path, caption: str | None = None) -> None:
    if path.exists():
        st.image(str(path), use_container_width=True)
        if caption:
            st.caption(caption)
    else:
        st.warning(f"Missing plot: `{path}` — run the pipeline to generate assets.")


def init_session_defaults() -> None:
    defaults = {
        "pf_city": "New York",
        "pf_state": "NY",
        "pf_zip": "10001",
        "pf_beds": 2,
        "pf_baths": 2.0,
        "pf_fullbaths": 2.0,
        "pf_lot": 1200.0,
        "pf_lat": 40.75,
        "pf_lon": -73.98,
        "pf_new": 0,
        "pf_hot": 0,
        "pf_remarks": "Bright corner unit with city views.",
        "pf_facts": "HOA: $850 / month. 1200 sq ft. Updated kitchen.",
        "pf_listprice": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def apply_preset(name: str) -> None:
    presets = {
        "Premium": {
            "pf_city": "New York",
            "pf_state": "NY",
            "pf_zip": "10019",
            "pf_beds": 4,
            "pf_baths": 5.0,
            "pf_fullbaths": 4.0,
            "pf_lot": 3500.0,
            "pf_lat": 40.764,
            "pf_lon": -73.97,
            "pf_new": 0,
            "pf_hot": 1,
            "pf_remarks": "Luxury penthouse with marble finishes, concierge, and skyline views.",
            "pf_facts": "HOA: $2500 / month. Designer kitchen. Private terrace and elevator.",
        },
        "Mid-Range": {
            "pf_city": "Austin",
            "pf_state": "TX",
            "pf_zip": "78701",
            "pf_beds": 3,
            "pf_baths": 2.5,
            "pf_fullbaths": 2.0,
            "pf_lot": 7200.0,
            "pf_lat": 30.27,
            "pf_lon": -97.74,
            "pf_new": 0,
            "pf_hot": 0,
            "pf_remarks": "Move-in ready home near downtown with upgraded appliances.",
            "pf_facts": "HOA: $120 / month. Garage parking. Modern kitchen.",
        },
        "Budget": {
            "pf_city": "Detroit",
            "pf_state": "MI",
            "pf_zip": "48201",
            "pf_beds": 2,
            "pf_baths": 1.0,
            "pf_fullbaths": 1.0,
            "pf_lot": 5200.0,
            "pf_lat": 42.35,
            "pf_lon": -83.06,
            "pf_new": 0,
            "pf_hot": 0,
            "pf_remarks": "Cozy starter home, turnkey condition.",
            "pf_facts": "HOA: $0. Fenced garden. 5200 sq ft lot.",
        },
    }
    data = presets.get(name, {})
    for k, v in data.items():
        st.session_state[k] = v


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Intelligent Housing Analysis",
    layout="wide",
    initial_sidebar_state="collapsed",
)

init_session_defaults()

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    /* ── Global Theme ── */
    .stApp { font-family: 'Inter', sans-serif; }
    .block-container { padding-top: 1rem; max-width: 1200px; }

    /* ── Header Hero ── */
    .hero-container {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
        border-radius: 16px; padding: 2rem 2.5rem; margin-bottom: 1.5rem;
        box-shadow: 0 8px 32px rgba(0,0,0,0.25);
    }
    .hero-title {
        font-size: 2.2rem; font-weight: 800; color: #ffffff;
        margin: 0 0 0.3rem 0; letter-spacing: -0.5px;
    }
    .hero-subtitle {
        font-size: 0.95rem; color: #a8a3c7; font-weight: 400;
        margin: 0 0 1.2rem 0;
    }
    .hero-metrics { display: flex; gap: 1.5rem; flex-wrap: wrap; }
    .hero-metric {
        background: rgba(255,255,255,0.08); backdrop-filter: blur(12px);
        border: 1px solid rgba(255,255,255,0.12); border-radius: 12px;
        padding: 1rem 1.4rem; min-width: 150px; flex: 1;
    }
    .hero-metric-label { font-size: 0.75rem; color: #9b97b8; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 4px; }
    .hero-metric-value { font-size: 1.5rem; font-weight: 700; color: #ffffff; }
    .hero-metric-value.accent { color: #7c6aef; }
    .hero-metric-value.gold { color: #D4AF37; }
    .hero-metric-value.green { color: #4ade80; }

    /* ── Tier Badges ── */
    .tier-pill {
        display: inline-block; padding: 0.4rem 1rem; border-radius: 20px;
        font-weight: 700; font-size: 0.9rem; letter-spacing: 0.5px;
        text-transform: uppercase;
    }
    .tier-premium { background: linear-gradient(135deg, #D4AF37, #f5d76e); color: #1a1a2e; }
    .tier-midrange { background: linear-gradient(135deg, #4A90D9, #63b3ed); color: #ffffff; }
    .tier-budget { background: linear-gradient(135deg, #6b7280, #9ca3af); color: #ffffff; }

    /* ── Cards ── */
    .glass-card {
        background: rgba(255,255,255,0.04); backdrop-filter: blur(10px);
        border: 1px solid rgba(255,255,255,0.08); border-radius: 14px;
        padding: 1.5rem; margin-bottom: 1rem;
    }
    .card-title { font-size: 1.1rem; font-weight: 700; color: #e2e0ea; margin-bottom: 0.8rem; }

    /* ── Metric Cards ── */
    div[data-testid="stMetricValue"] { font-size: 1.5rem; font-weight: 700; }
    div[data-testid="stMetricLabel"] { font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.5px; }

    /* ── Tabs ── */
    button[data-baseweb="tab"] { font-weight: 600; font-size: 0.85rem; }

    /* ── Section Headers ── */
    .section-header {
        font-size: 1.3rem; font-weight: 700; margin: 1.5rem 0 0.8rem 0;
        padding-bottom: 0.5rem; border-bottom: 2px solid rgba(124,106,239,0.3);
    }

    /* ── Model Ranking Badge ── */
    .rank-badge {
        display: inline-flex; align-items: center; justify-content: center;
        width: 28px; height: 28px; border-radius: 50%; font-weight: 700;
        font-size: 0.8rem; margin-right: 8px;
    }
    .rank-1 { background: linear-gradient(135deg, #D4AF37, #f5d76e); color: #1a1a2e; }
    .rank-2 { background: linear-gradient(135deg, #C0C0C0, #e0e0e0); color: #1a1a2e; }
    .rank-3 { background: linear-gradient(135deg, #cd7f32, #e8a954); color: #1a1a2e; }
    .rank-other { background: rgba(255,255,255,0.1); color: #9b97b8; }

    /* ── Streamlit overrides ── */
    .stTabs [data-baseweb="tab-list"] { gap: 4px; }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0; padding: 0.5rem 1.2rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

try:
    artifacts = load_artifacts()
except Exception as exc:
    st.error(f"Failed to load models or data: {exc}")
    st.stop()

ref_df = artifacts["ref_df"]
n_props = len(ref_df)
best_name = str(artifacts.get("best_model_name", "N/A"))
best_r2 = artifacts.get("best_r2", float("nan"))

# --- Premium Header Hero ---
r2_display = f"{best_r2:.3f}" if best_r2 == best_r2 else "—"
st.markdown(
    f"""
    <div class="hero-container">
        <div class="hero-title">🏠 Intelligent Housing Analysis</div>
        <div class="hero-subtitle">
            ML-driven price intelligence · NLP listing signals · Geospatial clustering · HouseScore ranking · SHAP explainability
        </div>
        <div class="hero-metrics">
            <div class="hero-metric">
                <div class="hero-metric-label">Properties</div>
                <div class="hero-metric-value accent">{n_props:,}</div>
            </div>
            <div class="hero-metric">
                <div class="hero-metric-label">Best Model</div>
                <div class="hero-metric-value gold">{best_name}</div>
            </div>
            <div class="hero-metric">
                <div class="hero-metric-label">Hold-out R²</div>
                <div class="hero-metric-value green">{r2_display}</div>
            </div>
            <div class="hero-metric">
                <div class="hero-metric-label">Models Compared</div>
                <div class="hero-metric-value">8</div>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

tabs = st.tabs(
    [
        "🔮 Prediction",
        "📊 Analytics",
        "🗺️ Map View",
        "🏆 Model Performance",
        "🧠 Explainability",
        "📂 Dataset Insights",
    ]
)

# ----- Prediction tab -----
with tabs[0]:
    st.markdown('<div class="section-header">🔮 Single-Property Analysis</div>', unsafe_allow_html=True)
    st.caption(
        "Submit structured listing fields. The app aligns features using ``feature_columns.pkl``, "
        "applies the saved scaler and regressors (log-price with USD inverse), then scores the listing."
    )

    pc1, pc2, pc3 = st.columns(3)
    with pc1:
        if st.button("Sample: Premium"):
            apply_preset("Premium")
            st.rerun()
    with pc2:
        if st.button("Sample: Mid-Range"):
            apply_preset("Mid-Range")
            st.rerun()
    with pc3:
        if st.button("Sample: Budget"):
            apply_preset("Budget")
            st.rerun()

    with st.form("prediction_form"):
        r1c1, r1c2, r1c3 = st.columns(3)
        with r1c1:
            city = st.text_input("City", value=st.session_state["pf_city"])
        with r1c2:
            state = st.text_input("State", value=st.session_state["pf_state"])
        with r1c3:
            zip_code = st.text_input("ZIP", value=st.session_state["pf_zip"])

        r2c1, r2c2, r2c3 = st.columns(3)
        with r2c1:
            beds = st.number_input("Beds", min_value=1, max_value=15, value=int(st.session_state["pf_beds"]))
        with r2c2:
            baths = st.number_input("Baths", min_value=0.5, max_value=15.0, value=float(st.session_state["pf_baths"]))
        with r2c3:
            full_baths = st.number_input(
                "Full baths", min_value=0.0, max_value=15.0, value=float(st.session_state["pf_fullbaths"])
            )

        r3c1, r3c2, r3c3 = st.columns(3)
        with r3c1:
            lot_size = st.number_input("Lot size (raw)", min_value=1.0, value=float(st.session_state["pf_lot"]))
        with r3c2:
            lat = st.number_input("Latitude", value=float(st.session_state["pf_lat"]), format="%.5f")
        with r3c3:
            lon = st.number_input("Longitude", value=float(st.session_state["pf_lon"]), format="%.5f")

        r4c1, r4c2, r4c3 = st.columns(3)
        with r4c1:
            is_new = st.checkbox("New construction", value=bool(int(st.session_state["pf_new"])))
        with r4c2:
            _ = st.checkbox("Hot listing (display only; not in model)", value=bool(int(st.session_state["pf_hot"])))
        with r4c3:
            list_price = st.number_input(
                "List price (optional, for investment % gap)",
                min_value=0.0,
                value=0.0,
                help="Leave 0 to skip investment gap / rating.",
            )

        listing_remarks = st.text_area("Listing remarks", value=st.session_state["pf_remarks"], height=100)
        key_facts = st.text_area("Key facts combined", value=st.session_state["pf_facts"], height=100)

        submitted = st.form_submit_button("Analyze property")

    if submitted:
        for k, v in [
            ("pf_city", city),
            ("pf_state", state),
            ("pf_zip", zip_code),
            ("pf_beds", beds),
            ("pf_baths", baths),
            ("pf_fullbaths", full_baths),
            ("pf_lot", lot_size),
            ("pf_lat", lat),
            ("pf_lon", lon),
            ("pf_new", int(is_new)),
            ("pf_remarks", listing_remarks),
            ("pf_facts", key_facts),
        ]:
            st.session_state[k] = v

        if not len(ref_df):
            st.error("Reference dataset missing. Run ``python src/pipeline.py`` first.")
        else:
            try:
                raw_feats = build_raw_feature_dict(
                    artifacts,
                    city,
                    state,
                    zip_code,
                    float(beds),
                    float(baths),
                    float(full_baths),
                    float(lot_size),
                    float(lat),
                    float(lon),
                    int(is_new),
                    listing_remarks,
                    key_facts,
                )
                fc = artifacts["feature_columns"]
                missing = [c for c in fc if c not in raw_feats]
                for c in missing:
                    raw_feats[c] = 0.0
                X_rs = scaler_transform_row(artifacts["scaler"], fc, raw_feats)
                pred_log = artifacts["model"].predict(X_rs)
                pred_price = float(log_to_price(pred_log)[0])

                _, tier, house_score = value_score_and_tier(ref_df, pred_price, raw_feats)

                if list_price and list_price > 0:
                    gap_pct = (pred_price - list_price) / list_price
                    if gap_pct > 0:
                        inv_rating = "Undervalued vs model (positive gap %)"
                    elif gap_pct < 0:
                        inv_rating = "Premium vs model (negative gap %)"
                    else:
                        inv_rating = "Aligned with model"
                    gap_pct_disp = f"{100 * gap_pct:.2f}%"
                else:
                    gap_pct = None
                    inv_rating = "Add list price for investment gap"
                    gap_pct_disp = "—"

                m1, m2, m3, m4 = st.columns(4)
                with m1:
                    st.metric("Predicted price (USD)", f"${pred_price:,.0f}")
                with m2:
                    st.metric("HouseScore", f"{house_score:.2f}")
                with m3:
                    tier_css = {"Premium": "tier-premium", "Mid-Range": "tier-midrange"}.get(tier, "tier-budget")
                    st.markdown(
                        f'<span class="tier-pill {tier_css}">{tier}</span>',
                        unsafe_allow_html=True,
                    )
                    st.caption("Tier (composite vs reference cohort)")
                with m4:
                    st.metric("Pricing gap %", gap_pct_disp)
                    st.caption(inv_rating)

                st.markdown('<div class="section-header">📊 Property Intelligence</div>', unsafe_allow_html=True)
                ic1, ic2 = st.columns(2)
                with ic1:
                    lux = float(raw_feats.get("luxury_score", 0) or 0)
                    ame = float(raw_feats.get("amenities_score", 0) or 0)
                    sent = float(raw_feats.get("sentiment_score", 0) or 0)
                    st.progress(min(1.0, lux / 10.0))
                    st.caption(f"Luxury score: {lux:.0f}")
                    st.progress(min(1.0, ame / 15.0))
                    st.caption(f"Amenities: {ame:.0f}")
                    st.progress(max(0.0, min(1.0, (sent + 1.0) / 2.0)))
                    st.caption(f"Sentiment: {sent:.2f}")
                with ic2:
                    st.metric("HOA (extracted)", f"${raw_feats.get('hoa_value', 0):,.0f}")
                    st.metric("Geo desirability (density)", f"{raw_feats.get('geo_desirability', 0):.0f}")
                    st.write(f"Renovation flag: **{int(raw_feats.get('renovation_flag', 0))}** · Condition flag: **{int(raw_feats.get('condition_flag', 0))}**")

                st.markdown('<div class="section-header">📈 Visual Analytics</div>', unsafe_allow_html=True)
                fi_path = REPORTS_DIR / "feature_importance.csv"
                g1, g2 = st.columns(2)
                with g1:
                    fig_g = go.Figure(
                        go.Indicator(
                            mode="gauge+number",
                            value=min(100, max(0, house_score)),
                            title={"text": "HouseScore"},
                            gauge={"axis": {"range": [0, 100]}, "bar": {"color": "#2E86AB"}},
                        )
                    )
                    st.plotly_chart(fig_g, use_container_width=True)
                with g2:
                    if fi_path.exists():
                        fi = pd.read_csv(fi_path)
                        if "model" in fi.columns and best_name != "N/A":
                            fi = fi[fi["model"] == best_name]
                        top = fi.head(10)
                        fig_b = px.bar(top, x="importance", y="feature", orientation="h", title="Top 10 feature contributions (global)")
                        st.plotly_chart(fig_b, use_container_width=True)
                    else:
                        st.info("Feature importance file not found.")

                if len(ref_df) and "value_tier" in ref_df.columns:
                    tier_counts = ref_df["value_tier"].value_counts().reindex(["Premium", "Mid-Range", "Budget"]).fillna(0)
                    fig_pie = px.pie(
                        values=tier_counts.values,
                        names=tier_counts.index,
                        title="Tier distribution (reference dataset)",
                        color_discrete_map={
                            "Premium": "#D4AF37",
                            "Mid-Range": "#4A90D9",
                            "Budget": "#888888",
                        },
                    )
                    st.plotly_chart(fig_pie, use_container_width=True)

                if "price_value" in ref_df.columns:
                    fig_hist = px.histogram(ref_df, x="price_value", nbins=60, title="Reference price distribution (USD)")
                    st.plotly_chart(fig_hist, use_container_width=True)

                st.markdown('<div class="section-header">🧠 Explainability</div>', unsafe_allow_html=True)
                top_feats = ""
                fi = pd.DataFrame()
                if fi_path.exists():
                    fi = pd.read_csv(fi_path)
                    if "model" in fi.columns and best_name != "N/A":
                        fi = fi[fi["model"] == best_name]
                    if len(fi):
                        top_feats = ", ".join(fi.head(5)["feature"].astype(str).tolist())
                st.info(
                    f"This property ranks as **{tier}** with HouseScore **{house_score:.2f}** mainly because the model "
                    f"priced it near **${pred_price:,.0f}** given structure, location encoding, and NLP cues. "
                    f"Global drivers include: {top_feats or 'see SHAP tab'}."
                )

                st.markdown('<div class="section-header">🗺️ Nearby Listings</div>', unsafe_allow_html=True)
                if {"latitude", "longitude"}.issubset(ref_df.columns):
                    sample = ref_df.dropna(subset=["latitude", "longitude"]).sample(min(400, len(ref_df)), random_state=RANDOM_STATE)
                    plot_df = pd.concat(
                        [
                            sample,
                            pd.DataFrame(
                                [{"latitude": lat, "longitude": lon, "value_tier": "NEW QUERY", "price_value": pred_price}]
                            ),
                        ],
                        ignore_index=True,
                    )
                    fig_map = px.scatter_mapbox(
                        plot_df,
                        lat="latitude",
                        lon="longitude",
                        color="value_tier" if "value_tier" in plot_df.columns else None,
                        hover_data=["price_value"] if "price_value" in plot_df.columns else None,
                        zoom=9,
                        height=420,
                        mapbox_style="open-street-map",
                    )
                    st.plotly_chart(fig_map, use_container_width=True)
                else:
                    st.map(pd.DataFrame({"lat": [lat], "lon": [lon]}))

            except Exception as exc:
                st.error(f"Prediction failed: {exc}")

# ----- Analytics tab -----
with tabs[1]:
    st.markdown('<div class="section-header">📊 Precomputed Analytics</div>', unsafe_allow_html=True)
    plot_files = [
        ("📉 Price Distribution (Log Scale)", PLOTS_DIR / "01_price_distribution_log.png"),
        ("🔗 Correlation Heatmap", PLOTS_DIR / "02_correlation_heatmap.png"),
        ("🎯 Predicted vs Actual", PLOTS_DIR / "03_predicted_vs_actual.png"),
        ("⭐ Feature Importance (Top 20)", PLOTS_DIR / "04_feature_importance_top20.png"),
        ("🗺️ Geospatial Tier Map", PLOTS_DIR / "07_geo_scatter_tier.png"),
        ("📈 HouseScore Distribution", PLOTS_DIR / "06_housescore_distribution.png"),
    ]
    for title, path in plot_files:
        st.markdown(f"##### {title}")
        show_plot_if_exists(path)

# ----- Map view tab -----
with tabs[2]:
    st.markdown('<div class="section-header">🗺️ Geospatial Explorer</div>', unsafe_allow_html=True)
    if len(ref_df) and {"latitude", "longitude"}.issubset(ref_df.columns):
        color_col = "value_tier" if "value_tier" in ref_df.columns else None
        fig_all = px.scatter_mapbox(
            ref_df.dropna(subset=["latitude", "longitude"]).sample(min(2500, len(ref_df)), random_state=RANDOM_STATE),
            lat="latitude",
            lon="longitude",
            color=color_col,
            hover_data=["price_value", "predicted_price"] if "predicted_price" in ref_df.columns else ["price_value"],
            zoom=3,
            height=600,
            mapbox_style="carto-positron",
        )
        st.plotly_chart(fig_all, use_container_width=True)
    else:
        st.warning("No coordinates available in processed dataset.")

# ----- Model performance tab -----
with tabs[3]:
    st.markdown('<div class="section-header">🏆 Model Comparison & Performance</div>', unsafe_allow_html=True)
    mm = artifacts.get("model_metrics", pd.DataFrame())
    if len(mm):
        # Visual R² comparison chart
        chart_df = mm[["model", "r2", "rmse", "mae", "mape"]].copy()
        chart_df["r2_display"] = chart_df["r2"].clip(lower=0)  # clip negatives for chart
        fig_compare = go.Figure()
        colors = ['#D4AF37' if i == 0 else '#C0C0C0' if i == 1 else '#cd7f32' if i == 2 else '#4A90D9'
                  for i in range(len(chart_df))]
        fig_compare.add_trace(go.Bar(
            x=chart_df["model"], y=chart_df["r2_display"],
            marker_color=colors,
            text=[f"R²={v:.3f}" for v in chart_df["r2"]],
            textposition="outside",
        ))
        fig_compare.update_layout(
            title="Model R² Comparison (Hold-out Test Set)",
            yaxis_title="R² Score", xaxis_title="",
            height=400, showlegend=False,
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_compare, use_container_width=True)

        st.markdown("##### 📋 Detailed Metrics Table")
        show_cols = [c for c in mm.columns if c != "best_params"]
        st.dataframe(mm[show_cols], use_container_width=True, hide_index=True)
    else:
        st.warning("``reports/model_metrics.csv`` not found.")

    summ_path = REPORTS_DIR / "model_evaluation_summary.csv"
    if summ_path.exists():
        st.markdown("##### 📊 Combined Evaluation Summary")
        st.dataframe(pd.read_csv(summ_path), use_container_width=True, hide_index=True)
    else:
        st.info("Run the full pipeline to build ``model_evaluation_summary.csv``.")

    st.markdown('<div class="section-header">🎯 Tier Classification</div>', unsafe_allow_html=True)
    cm_png = PLOTS_EVAL / "confusion_matrix.png"
    show_plot_if_exists(cm_png, "Confusion matrix: actual price-based tier vs predicted tier")
    prf_html = PLOTS_EVAL / "per_class_prf.html"
    if prf_html.exists():
        components.html(prf_html.read_text(encoding="utf-8"), height=520, scrolling=True)
    else:
        st.caption("Open ``plots/evaluation/per_class_prf.html`` after running evaluation.")

# ----- Explainability tab -----
with tabs[4]:
    st.markdown('<div class="section-header">🧠 SHAP Explainability & Global Importance</div>', unsafe_allow_html=True)
    show_plot_if_exists(PLOTS_SHAP / "shap_summary_beeswarm.png", "SHAP summary (log-scale model output)")
    show_plot_if_exists(PLOTS_SHAP / "shap_bar_importance.png", "SHAP bar plot")
    fi_csv = REPORTS_DIR / "feature_importance.csv"
    if fi_csv.exists():
        st.markdown("##### Feature importance table")
        st.dataframe(pd.read_csv(fi_csv).head(25), use_container_width=True, hide_index=True)
    st.markdown(
        "SHAP values are on **log1p(price)** model outputs; USD predictions use clipped inverse transform "
        "as in training."
    )

# ----- Dataset insights tab -----
with tabs[5]:
    st.markdown('<div class="section-header">📂 Dataset Insights</div>', unsafe_allow_html=True)
    if not len(ref_df):
        st.warning("Load ``data/processed_dataset.csv`` via the pipeline.")
    else:
        st.write(f"**Shape:** {ref_df.shape[0]:,} rows × {ref_df.shape[1]} columns")
        if "price_value" in ref_df.columns:
            pv = ref_df["price_value"]
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Mean price", f"${pv.mean():,.0f}")
            c2.metric("Median price", f"${pv.median():,.0f}")
            c3.metric("Min price", f"${pv.min():,.0f}")
            c4.metric("Max price", f"${pv.max():,.0f}")

        if "city" in ref_df.columns:
            topc = ref_df["city"].value_counts().head(10).reset_index()
            topc.columns = ["city", "count"]
            st.markdown("##### Top 10 cities by listing count")
            st.bar_chart(topc.set_index("city"))

        if "value_tier" in ref_df.columns:
            st.markdown("##### Tier distribution")
            st.bar_chart(ref_df["value_tier"].value_counts())

        if "luxury_score" in ref_df.columns:
            st.markdown("##### Luxury score distribution")
            st.plotly_chart(px.histogram(ref_df, x="luxury_score", nbins=30), use_container_width=True)

        if "hoa_value" in ref_df.columns:
            st.markdown("##### HOA (extracted) distribution")
            st.plotly_chart(px.histogram(ref_df[ref_df["hoa_value"] > 0], x="hoa_value", nbins=40), use_container_width=True)

        if "isHot" in ref_df.columns and "isNewConstruction" in ref_df.columns:
            st.markdown("##### Flags")
            c1, c2 = st.columns(2)
            c1.write(ref_df["isHot"].value_counts())
            c2.write(ref_df["isNewConstruction"].value_counts())
