# app_min.py
# WAVES Intelligence — Stable Console
# PURPOSE: Portfolio + Alpha Attribution + Alpha Source Breakdown

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

    if alpha_df is None:
        st.warning("Alpha attribution file not found.")
        st.stop()

    if alpha_df.empty:
        st.info("Alpha attribution file exists but has no data yet.")
        st.stop()

    alpha_df = normalize_columns(alpha_df)

    # REQUIRED COLUMNS (NEW SCHEMA)
    required_cols = {
        "wave",
        "horizon",
        "selection_alpha",
        "momentum_alpha",
        "vix_alpha",
        "volatility_alpha",
        "exposure_alpha",
        "residual_alpha",
    }

    missing = required_cols - set(alpha_df.columns)

    if missing:
        st.error("Alpha attribution file is missing required columns:")
        st.code(sorted(missing))
        st.stop()

    # -----------------------------
    # Controls (PRONOUNCED UI)
    # -----------------------------
    col1, col2 = st.columns([2, 1])

    with col1:
        selected_wave = st.selectbox(
            "🔍 Select Wave",
            sorted(alpha_df["wave"].unique()),
        )

    with col2:
        selected_horizon = st.selectbox(
            "📅 Select Horizon",
            [30, 60, 365],
            index=2,
        )

    st.markdown(
        f"### **{selected_wave} — {selected_horizon} Day Alpha Sources**"
    )

    # -----------------------------
    # Filter + Display
    # -----------------------------
    view_df = alpha_df[
        (alpha_df["wave"] == selected_wave)
        & (alpha_df["horizon"] == selected_horizon)
    ]

    if view_df.empty:
        st.warning("No data available for this selection.")
        st.stop()

    display_df = view_df[[
        "selection_alpha",
        "momentum_alpha",
        "vix_alpha",
        "volatility_alpha",
        "exposure_alpha",
        "residual_alpha",
    ]].T

    display_df.columns = ["Alpha Contribution"]
    display_df.index = display_df.index.str.replace("_", " ").str.title()

    st.dataframe(display_df, use_container_width=True)

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