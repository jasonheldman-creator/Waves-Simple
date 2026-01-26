# app_min.py
# WAVES Intelligence — Minimal Stable Console (Institutional)
# PURPOSE: Portfolio + Alpha Source Attribution with Sidebar
# AUTHOR: Stabilized rebuild (FINAL)

import streamlit as st
import pandas as pd
from pathlib import Path

# -----------------------------
# App Config
# -----------------------------
st.set_page_config(
    page_title="WAVES — Intelligence Console",
    layout="wide",
)

# -----------------------------
# Paths
# -----------------------------
DATA_DIR = Path("data")
ALPHA_ATTR_PATH = DATA_DIR / "alpha_attribution_summary.csv"
LIVE_SNAPSHOT_PATH = DATA_DIR / "live_snapshot.csv"

# -----------------------------
# Helpers
# -----------------------------
def safe_read_csv(path: Path) -> pd.DataFrame | None:
    try:
        if not path.exists():
            return None
        df = pd.read_csv(path)
        if df.empty:
            return pd.DataFrame()
        return df
    except Exception as e:
        st.error(f"Failed to read {path.name}: {e}")
        return None


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
    )
    return df


# -----------------------------
# Sidebar (RESTORED)
# -----------------------------
with st.sidebar:
    st.header("WAVES Console")
    st.caption("Navigation & Diagnostics")

    st.markdown("**Data Status**")
    st.write("Live Snapshot:", LIVE_SNAPSHOT_PATH.exists())
    st.write("Alpha Attribution:", ALPHA_ATTR_PATH.exists())

# -----------------------------
# Header
# -----------------------------
st.title("Intelligence Console")
st.caption("Returns • Alpha • Attribution • Adaptive Intelligence • Operations")

tabs = st.tabs([
    "Overview",
    "Alpha Attribution",
    "Adaptive Intelligence",
    "Operations",
])

# -----------------------------
# Overview Tab
# -----------------------------
with tabs[0]:
    st.subheader("Portfolio Snapshot")

    snapshot_df = safe_read_csv(LIVE_SNAPSHOT_PATH)

    if snapshot_df is None:
        st.warning("Live snapshot not found.")
    elif snapshot_df.empty:
        st.info("Live snapshot exists but contains no rows.")
    else:
        snapshot_df = normalize_columns(snapshot_df)
        st.dataframe(snapshot_df, use_container_width=True)

# -----------------------------
# Alpha Attribution Tab
# -----------------------------
with tabs[1]:
    st.subheader("⚡ Alpha Attribution — Source Breakdown")

    alpha_df = safe_read_csv(ALPHA_ATTR_PATH)

    st.caption(
        f"DEBUG — alpha_attribution_summary.csv exists: {ALPHA_ATTR_PATH.exists()}"
    )

    if alpha_df is None or alpha_df.empty:
        st.warning("Alpha attribution data not available yet.")
        st.stop()

    alpha_df = normalize_columns(alpha_df)

    required_cols = {
        "wave",
        "horizon",
        "total_alpha",
        "selection_alpha",
        "momentum_alpha",
        "volatility_alpha",
        "regime_alpha",
        "exposure_alpha",
        "residual_alpha",
    }

    missing = required_cols - set(alpha_df.columns)
    if missing:
        st.error("Alpha attribution file is missing required columns:")
        st.code(sorted(missing))
        st.stop()

    # -----------------------------
    # Selectors (PROMINENT)
    # -----------------------------
    waves = ["Portfolio"] + sorted(alpha_df["wave"].unique().tolist())

    selected_wave = st.selectbox(
        "🔎 Select Wave",
        waves,
        index=0,
    )

    selected_horizon = st.selectbox(
        "📅 Select Horizon (Days)",
        sorted(alpha_df["horizon"].unique()),
        index=sorted(alpha_df["horizon"].unique()).index(365),
    )

    # -----------------------------
    # Filter
    # -----------------------------
    if selected_wave == "Portfolio":
        filtered = alpha_df[alpha_df["horizon"] == selected_horizon]
        title = f"Portfolio — {selected_horizon} Day Alpha Sources"
    else:
        filtered = alpha_df[
            (alpha_df["wave"] == selected_wave)
            & (alpha_df["horizon"] == selected_horizon)
        ]
        title = f"{selected_wave} — {selected_horizon} Day Alpha Sources"

    st.subheader(title)

    if filtered.empty:
        st.info("No data available for this selection.")
    else:
        display_cols = [
            "selection_alpha",
            "momentum_alpha",
            "volatility_alpha",
            "regime_alpha",
            "exposure_alpha",
            "residual_alpha",
        ]

        st.dataframe(
            filtered[display_cols],
            use_container_width=True,
        )

# -----------------------------
# Adaptive Intelligence Tab
# -----------------------------
with tabs[2]:
    st.subheader("Adaptive Intelligence")
    st.info("Adaptive intelligence metrics will appear here.")

# -----------------------------
# Operations Tab
# -----------------------------
with tabs[3]:
    st.subheader("Operations")
    st.info("Operational controls coming soon.")