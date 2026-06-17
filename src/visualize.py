"""
Visualization utilities for exploratory and model-driven analytics.
Uses the **3-tier** system: Budget / Mid-Range / Premium.
"""
from __future__ import annotations
import warnings
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
from matplotlib import ticker

warnings.filterwarnings("ignore", category=UserWarning)
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
PLOTS_DIR = ROOT / "plots"

def _load_processed() -> pd.DataFrame:
    path = DATA_DIR / "processed_dataset.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing processed dataset at {path}.")
    return pd.read_csv(path)

def _style_matplotlib() -> None:
    sns.set_theme(style="whitegrid", context="talk", font_scale=0.8)
    plt.rcParams["figure.dpi"] = 140

def run_visualizations() -> None:
    _style_matplotlib()
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    print("[visualize] Loading processed dataset for plotting...")
    df = _load_processed()

    # 1) Price distribution
    fig, ax = plt.subplots(figsize=(10, 6))
    pos = df.loc[df["price_value"] > 0, "price_value"]
    sns.histplot(np.log10(pos), kde=True, ax=ax, color="#2E86AB")
    ax.set_title("Log10 Price Distribution"); ax.set_xlabel("log10(price)")
    fig.tight_layout(); fig.savefig(PLOTS_DIR / "01_price_distribution_log.png"); plt.close(fig)

    # 2) Correlation heatmap
    ncols = [c for c in ["price_value","predicted_price","beds","baths","lot_size_sqft",
        "latitude","longitude","luxury_score","amenities_score","sentiment_score",
        "geo_desirability","house_score","pricing_gap_pct"] if c in df.columns]
    corr = df[ncols].corr()
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr, cmap="RdBu_r", center=0, ax=ax)
    ax.set_title("Correlation Heatmap"); fig.tight_layout()
    fig.savefig(PLOTS_DIR / "02_correlation_heatmap.png"); plt.close(fig)

    # 3) Predicted vs actual
    if {"predicted_price", "price_value"}.issubset(df.columns):
        fig, ax = plt.subplots(figsize=(8, 8))
        s = df.sample(min(4000, len(df)), random_state=42)
        ax.scatter(s["price_value"], s["predicted_price"], alpha=0.35, s=12, color="#A23B72")
        lims = [min(s["price_value"].min(), s["predicted_price"].min()),
                max(s["price_value"].max(), s["predicted_price"].max())]
        ax.plot(lims, lims, ls="--", c="black", lw=1)
        ax.set_xlabel("Actual price"); ax.set_ylabel("Predicted price")
        ax.set_title("Predicted vs Actual Prices")
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f"{x/1e6:.1f}M"))
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f"{x/1e6:.1f}M"))
        fig.tight_layout(); fig.savefig(PLOTS_DIR / "03_predicted_vs_actual.png"); plt.close(fig)

    # 4) Feature importance
    fi_path = ROOT / "reports" / "feature_importance.csv"
    if fi_path.exists():
        fi = pd.read_csv(fi_path).head(20)
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.barplot(data=fi, y="feature", x="importance", hue="feature", palette="viridis", legend=False, ax=ax)
        ax.set_title("Top 20 Feature Importances"); fig.tight_layout()
        fig.savefig(PLOTS_DIR / "04_feature_importance_top20.png"); plt.close(fig)

    # 5) Tier distribution — 3-tier
    if "value_tier" in df.columns:
        order = ["Budget", "Mid-Range", "Premium"]
        counts = df["value_tier"].value_counts().reindex(order).fillna(0).reset_index()
        counts.columns = ["value_tier", "count"]
        fig, ax = plt.subplots(figsize=(8, 5))
        pal = {"Budget": "#888888", "Mid-Range": "#4A90D9", "Premium": "#D4AF37"}
        sns.barplot(data=counts, x="value_tier", y="count", hue="value_tier", palette=pal, legend=False, ax=ax)
        ax.set_title("House Tier Distribution (3-Tier)"); ax.set_ylabel("Count")
        fig.tight_layout(); fig.savefig(PLOTS_DIR / "05_tier_distribution.png"); plt.close(fig)

    # 6) HouseScore distribution
    if "house_score" in df.columns:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(df["house_score"], kde=True, ax=ax, color="#F18F01")
        ax.set_title("HouseScore Distribution"); fig.tight_layout()
        fig.savefig(PLOTS_DIR / "06_housescore_distribution.png"); plt.close(fig)

    # 7) Geo scatter
    if {"latitude", "longitude", "value_tier"}.issubset(df.columns):
        pal = {"Budget": "#e7298a", "Mid-Range": "#7570b3", "Premium": "#1b9e77"}
        fig, ax = plt.subplots(figsize=(9, 7))
        for tier, sub in df.groupby("value_tier"):
            ax.scatter(sub["longitude"], sub["latitude"], s=8, alpha=0.35, label=tier, color=pal.get(tier, "#666"))
        ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude"); ax.set_title("Geospatial Tier Map")
        ax.legend(markerscale=2, frameon=True); fig.tight_layout()
        fig.savefig(PLOTS_DIR / "07_geo_scatter_tier.png"); plt.close(fig)

        fig_px = px.scatter_geo(df, lat="latitude", lon="longitude", color="value_tier",
            hover_data=["price_value","house_score"], scope="usa", title="Interactive Geo Map by Tier")
        fig_px.write_html(PLOTS_DIR / "07_geo_scatter_tier_interactive.html")

    # 8) Geo predicted price
    if {"latitude", "longitude", "predicted_price"}.issubset(df.columns):
        fig, ax = plt.subplots(figsize=(9, 7))
        sc = ax.scatter(df["longitude"], df["latitude"], c=df["predicted_price"], cmap="plasma", s=8, alpha=0.45)
        plt.colorbar(sc, ax=ax, label="Predicted price")
        ax.set_title("Geospatial Heat (Predicted Price)"); ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
        fig.tight_layout(); fig.savefig(PLOTS_DIR / "08_geo_scatter_predicted_price.png"); plt.close(fig)

        fig_px = px.scatter_geo(df, lat="latitude", lon="longitude", color="predicted_price",
            hover_data=["price_value","house_score"], scope="usa", color_continuous_scale="Viridis",
            title="Interactive Geo Map by Predicted Price")
        fig_px.write_html(PLOTS_DIR / "08_geo_scatter_predicted_price_interactive.html")

    # 9) Pricing gap scatter
    if {"pricing_gap_pct", "price_value"}.issubset(df.columns):
        fig, ax = plt.subplots(figsize=(9, 6))
        colors = np.where(df["pricing_gap_pct"] >= 0, "#2ca02c", "#d62728")
        ax.scatter(df["price_value"], df["pricing_gap_pct"], c=colors, alpha=0.35, s=12)
        ax.axhline(0, color="black", lw=1)
        ax.set_xlabel("Actual list price"); ax.set_ylabel("Pricing gap %")
        ax.set_title("Undervalued (green) vs Overpriced (red)")
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f"{x/1e6:.1f}M"))
        fig.tight_layout(); fig.savefig(PLOTS_DIR / "09_pricing_gap_scatter.png"); plt.close(fig)

    # 10) Cluster pricing
    if {"geo_cluster", "price_value"}.issubset(df.columns):
        cs = df.groupby("geo_cluster")["price_value"].mean().reset_index()
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.barplot(data=cs, x="geo_cluster", y="price_value", hue="geo_cluster", palette="mako", legend=False, ax=ax)
        ax.set_title("Average Observed Price by Geo Cluster"); fig.tight_layout()
        fig.savefig(PLOTS_DIR / "10_cluster_average_price.png"); plt.close(fig)

        cs2 = cs.rename(columns={"price_value": "cluster_mean_price"})
        pdf = df.merge(cs2, on="geo_cluster", how="left")
        fig_px = px.scatter(pdf, x="longitude", y="latitude", color="geo_cluster",
            size="cluster_mean_price", hover_data=["price_value","predicted_price"],
            title="Clusters Sized by Mean Observed Price (EDA)")
        fig_px.write_html(PLOTS_DIR / "10_cluster_pricing_interactive.html")

    print(f"[visualize] Saved charts under {PLOTS_DIR}")

def main() -> None:
    run_visualizations()

if __name__ == "__main__":
    main()
