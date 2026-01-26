# app_min.py
# WAVES Intelligence — Alpha Attribution Console (Stable)
# PURPOSE: Institutional-grade Portfolio + Wave Alpha Source Breakdown
# AUTHOR: Locked stable rewrite (Portfolio-first, horizon-safe)

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

# =====================================================
# Overview Tab
# =====================================================
with tabs[0]:
    st.subheader("📊 Portfolio Snapshot")

    snapshot_df = safe_read_csv(LIVE_SNAPSHOT_PATH)

    if snapshot_df is None:
        st.warning("Live snapshot not found.")
    elif snapshot_df.empty:
        st.info("Live snapshot exists but contains no rows.")
    else:
        snapshot_df = normalize_columns(snapshot_df)
        st.dataframe(snapshot_df, use_container_width=True)

# =====================================================
# Alpha Attribution Tab
# =====================================================
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

    # -----------------------------
    # Required schema (LONG FORMAT)
    # -----------------------------
    required_cols = {
        "wave",
        "horizon",
        "selection_alpha",
        "momentum_alpha",
        "volatility_alpha",
        "regime_alpha",
        "exposure_alpha",
        "residual_alpha",
        "total_alpha",
    }

    missing = required_cols - set(alpha_df.columns)
    if missing:
        st.error("Alpha attribution file is missing required columns:")
        st.code(sorted(missing))
        st.stop()

    # -----------------------------
    # Normalize horizon to INT
    # -----------------------------
    alpha_df["horizon"] = alpha_df["horizon"].astype(int)

    # -----------------------------
    # Wave Selector (Portfolio FIRST)
    # -----------------------------
    waves = sorted(alpha_df["wave"].unique().tolist())

    if "Portfolio" in waves:
        waves.remove("Portfolio")
        waves.insert(0, "Portfolio")

    selected_wave = st.selectbox(
        "🔍 Select Wave",
        waves,
        index=0
    )

    # -----------------------------
    # Horizon Selector
    # -----------------------------
    horizons = sorted(alpha_df["horizon"].unique().tolist())

    selected_horizon = st.selectbox(
        "📅 Select Horizon (Days)",
        horizons,
        index=horizons.index(365) if 365 in horizons else 0
    )

    # -----------------------------
    # Filter
    # -----------------------------
    filtered = alpha_df[
        (alpha_df["wave"] == selected_wave) &
        (alpha_df["horizon"] == selected_horizon)
    ]

    # -----------------------------
    # Display Header (Pronounced)
    # -----------------------------
    st.markdown("---")
    st.markdown(
        f"## **{selected_wave} — {selected_horizon} Day Alpha Sources**"
    )

    if filtered.empty:
        st.warning(
            "No alpha source data available for this Wave/Horizon combination.\n\n"
            "This usually means attribution logic has not yet populated this slice."
        )

        st.markdown("#### 🔎 Diagnostic Preview")
        st.dataframe(
            alpha_df[
                alpha_df["wave"] == selected_wave
            ][["wave", "horizon", "total_alpha"]],
            use_container_width=True
        )
        st.stop()

    # -----------------------------
    # Alpha Source Breakdown Table
    # -----------------------------
    source_cols = [
        "selection_alpha",
        "momentum_alpha",
        "volatility_alpha",
        "regime_alpha",
        "exposure_alpha",
        "residual_alpha",
        "total_alpha",
    ]

    display_df = filtered[source_cols].T.reset_index()
    display_df.columns = ["Alpha Source", "Contribution"]

    st.dataframe(display_df, use_container_width=True)

# =====================================================
# Adaptive Intelligence Tab
# =====================================================
with tabs[2]:
    st.subheader("🧠 Adaptive Intelligence")
    st.info("Adaptive intelligence diagnostics will appear here.")

# =====================================================
# Operations Tab
# =====================================================
with tabs[3]:
    st.subheader("⚙️ Operations")
    st.info("Operational controls coming soon.")