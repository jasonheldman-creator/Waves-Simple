# app_min.py
# WAVES Intelligence — Minimal Stable Console
# PURPOSE: Bulletproof read + display of Portfolio + Alpha Attribution data
# AUTHOR: Stabilized rebuild

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
    """Safely read CSV, return None on any failure."""
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
    """Normalize column names for safety."""
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
    st.subheader("⚡ Alpha Attribution Breakdown (365D)")

    alpha_df = safe_read_csv(ALPHA_ATTR_PATH)

    # Explicit debug visibility (kept intentionally)
    st.caption(
        f"DEBUG — alpha_attribution_summary.csv exists: {ALPHA_ATTR_PATH.exists()}"
    )

    if alpha_df is None:
        st.warning("Alpha attribution file not found. Awaiting next build.")
        st.stop()

    if alpha_df.empty:
        st.info("Alpha attribution file exists but has no data yet.")
        st.stop()

    alpha_df = normalize_columns(alpha_df)

    # Required columns (non-negotiable)
    required_cols = {
        "wave",
        "alpha_30d",
        "alpha_60d",
        "alpha_365d",
    }

    missing = required_cols - set(alpha_df.columns)

    if missing:
        st.error(
            "Alpha attribution file is present but missing required columns:"
        )
        st.code(sorted(missing))
        st.stop()

    # Display
    display_cols = [
        "wave",
        "alpha_30d",
        "alpha_60d",
        "alpha_365d",
    ]

    st.dataframe(
        alpha_df[display_cols].sort_values("alpha_365d", ascending=False),
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