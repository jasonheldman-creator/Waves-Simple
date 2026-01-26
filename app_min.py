# app_min.py
# WAVES Intelligence — Minimal Stable Console
# PURPOSE: Stable rendering of Overview + Alpha Attribution + Sidebar
# AUTHOR: Institutional-safe rebuild (FULL FILE)

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
LIVE_SNAPSHOT_PATH = DATA_DIR / "live_snapshot.csv"
ALPHA_ATTR_PATH = DATA_DIR / "alpha_attribution_summary.csv"

# -----------------------------
# Helpers
# -----------------------------
def safe_read_csv(path: Path) -> pd.DataFrame | None:
    try:
        if not path.exists():
            return None
        df = pd.read_csv(path)
        return df if not df.empty else pd.DataFrame()
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
    st.markdown("### WAVES Console")
    st.caption("Navigation & Diagnostics")

    snapshot_exists = LIVE_SNAPSHOT_PATH.exists()
    alpha_exists = ALPHA_ATTR_PATH.exists()

    st.markdown("**Data Status**")
    st.write(f"Live Snapshot: {'True' if snapshot_exists else 'False'}")
    st.write(f"Alpha Attribution: {'True' if alpha_exists else 'False'}")

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

# =========================================================
# OVERVIEW TAB — Portfolio & Wave Performance Snapshot
# =========================================================
with tabs[0]:
    st.subheader("Portfolio & Wave Performance Snapshot")

    snapshot_df = safe_read_csv(LIVE_SNAPSHOT_PATH)

    if snapshot_df is None:
        st.warning("Live snapshot not found.")
    elif snapshot_df.empty:
        st.warning("Live snapshot exists but contains no rows.")
    else:
        snapshot_df = normalize_columns(snapshot_df)

        required_returns = {
            "display_name",
            "return_30d",
            "return_60d",
            "return_365d",
        }

        missing = required_returns - set(snapshot_df.columns)

        if missing:
            st.error("Live snapshot missing required return columns.")
            st.code(sorted(missing))
        else:
            st.dataframe(
                snapshot_df[
                    [
                        "display_name",
                        "return_30d",
                        "return_60d",
                        "return_365d",
                    ]
                ].sort_values("return_365d", ascending=False),
                use_container_width=True,
            )

# =========================================================
# ALPHA ATTRIBUTION TAB
# =========================================================
with tabs[1]:
    st.subheader("⚡ Alpha Attribution — Source Breakdown")

    st.caption(
        f"DEBUG — alpha_attribution_summary.csv exists: {ALPHA_ATTR_PATH.exists()}"
    )

    alpha_df = safe_read_csv(ALPHA_ATTR_PATH)

    if alpha_df is None:
        st.warning("Alpha attribution file not found.")
        st.stop()

    if alpha_df.empty:
        st.info("Alpha attribution file exists but has no data yet.")
        st.stop()

    alpha_df = normalize_columns(alpha_df)

    required_cols = {
        "wave",
        "horizon",
        "total_alpha",
        "selection_alpha",
        "momentum_alpha",
        "regime_alpha",
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
    # Controls
    # -----------------------------
    wave_options = ["Portfolio"] + sorted(alpha_df["wave"].unique().tolist())
    selected_wave = st.selectbox("🔍 Select Wave", wave_options)

    horizon_options = sorted(alpha_df["horizon"].unique().tolist())
    selected_horizon = st.selectbox("📅 Select Horizon (Days)", horizon_options)

    # -----------------------------
    # Filtered View
    # -----------------------------
    if selected_wave == "Portfolio":
        view_df = alpha_df[alpha_df["horizon"] == selected_horizon]
    else:
        view_df = alpha_df[
            (alpha_df["wave"] == selected_wave)
            & (alpha_df["horizon"] == selected_horizon)
        ]

    st.markdown(
        f"### {selected_wave} — {selected_horizon} Day Alpha Sources"
    )

    if view_df.empty:
        st.warning("No data available for this selection.")
    else:
        display_cols = [
            "total_alpha",
            "selection_alpha",
            "momentum_alpha",
            "regime_alpha",
            "volatility_alpha",
            "exposure_alpha",
            "residual_alpha",
        ]

        st.dataframe(
            view_df[display_cols],
            use_container_width=True,
        )

    # -----------------------------
    # Alpha History (Placeholder)
    # -----------------------------
    st.markdown("### 📈 Alpha History")
    st.info("Alpha history not yet available.")

# =========================================================
# ADAPTIVE INTELLIGENCE TAB
# =========================================================
with tabs[2]:
    st.subheader("Adaptive Intelligence")
    st.info("Adaptive intelligence metrics will appear here.")

# =========================================================
# OPERATIONS TAB
# =========================================================
with tabs[3]:
    st.subheader("Operations")
    st.info("Operational controls coming soon.")
    