# app_min.py
# WAVES Intelligence — Minimal Stable Console
# PURPOSE: Overview + Alpha Attribution + Sidebar (Institutional-safe)
# NOTE: Full-file replacement. No truncation.

import streamlit as st
import pandas as pd
from pathlib import Path

# =====================================================
# App Config
# =====================================================
st.set_page_config(
    page_title="WAVES — Intelligence Console",
    layout="wide",
)

# =====================================================
# Paths
# =====================================================
DATA_DIR = Path("data")
LIVE_SNAPSHOT_PATH = DATA_DIR / "live_snapshot.csv"
ALPHA_ATTR_PATH = DATA_DIR / "alpha_attribution_summary.csv"

# =====================================================
# Helpers
# =====================================================
def safe_read_csv(path: Path) -> pd.DataFrame | None:
    try:
        if not path.exists():
            return None
        df = pd.read_csv(path)
        if df.empty:
            return None
        return df
    except Exception as e:
        st.error(f"Failed to read {path.name}: {e}")
        return None


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = (
        df.columns
        .astype(str)
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
    )
    return df


# =====================================================
# Sidebar (RESTORED & STABLE)
# =====================================================
with st.sidebar:
    st.markdown("### WAVES Console")
    st.caption("Navigation & Diagnostics")

    snapshot_exists = LIVE_SNAPSHOT_PATH.exists()
    alpha_exists = ALPHA_ATTR_PATH.exists()

    st.markdown("**Data Status**")
    st.write(f"Live Snapshot: {'True' if snapshot_exists else 'False'}")
    st.write(f"Alpha Attribution: {'True' if alpha_exists else 'False'}")


# =====================================================
# Header + Tabs
# =====================================================
st.title("Intelligence Console")
st.caption("Returns • Alpha • Attribution • Adaptive Intelligence • Operations")

tabs = st.tabs([
    "Overview",
    "Alpha Attribution",
    "Adaptive Intelligence",
    "Operations",
])

# =====================================================
# OVERVIEW TAB — Portfolio & Wave Snapshot
# =====================================================
with tabs[0]:
    st.subheader("Portfolio & Wave Performance Snapshot")

    snapshot_df = safe_read_csv(LIVE_SNAPSHOT_PATH)

    if snapshot_df is None:
        st.error("Live snapshot file not found or empty.")
    else:
        snapshot_df = normalize_columns(snapshot_df)

        required_cols = {
            "display_name",
            "return_30d",
            "return_60d",
            "return_365d",
        }

        missing = required_cols - set(snapshot_df.columns)

        if missing:
            st.error("Live snapshot missing required return columns.")
            st.code(sorted(list(missing)))
        else:
            display_df = snapshot_df[
                ["display_name", "return_30d", "return_60d", "return_365d"]
            ].copy()

            display_df = display_df.rename(columns={
                "display_name": "Wave",
                "return_30d": "30D Return",
                "return_60d": "60D Return",
                "return_365d": "365D Return",
            })

            st.dataframe(
                display_df,
                use_container_width=True,
                hide_index=True,
            )

# =====================================================
# ALPHA ATTRIBUTION TAB
# =====================================================
with tabs[1]:
    st.subheader("⚡ Alpha Attribution — Source Breakdown")

    st.caption(
        f"DEBUG — alpha_attribution_summary.csv exists: "
        f"{'True' if ALPHA_ATTR_PATH.exists() else 'False'}"
    )

    alpha_df = safe_read_csv(ALPHA_ATTR_PATH)

    if alpha_df is None:
        st.error("Alpha attribution file not found or empty.")
    else:
        alpha_df = normalize_columns(alpha_df)

        required_cols = {"display_name", "horizon"}
        missing = required_cols - set(alpha_df.columns)

        if missing:
            st.error("Alpha attribution file is missing required columns:")
            st.code(sorted(list(missing)))
        else:
            # Dropdowns
            wave = st.selectbox(
                "🔎 Select Wave",
                sorted(alpha_df["display_name"].unique()),
            )

            horizon = st.selectbox(
                "📅 Select Horizon (Days)",
                sorted(alpha_df["horizon"].unique()),
            )

            filtered = alpha_df[
                (alpha_df["display_name"] == wave)
                & (alpha_df["horizon"] == horizon)
            ]

            st.markdown(
                f"### {wave} — {horizon} Day Alpha Sources"
            )

            if filtered.empty:
                st.warning("No data available for this selection.")
            else:
                # Show only alpha columns
                alpha_cols = [
                    c for c in filtered.columns
                    if c.endswith("_alpha")
                ]

                if not alpha_cols:
                    st.warning("No alpha source columns found.")
                else:
                    st.dataframe(
                        filtered[alpha_cols],
                        use_container_width=True,
                        hide_index=True,
                    )

            # -------------------------
            # Alpha History (Placeholder)
            # -------------------------
            st.markdown("### 📈 Alpha History")
            st.info("Alpha history not yet available.")

# =====================================================
# Adaptive Intelligence (Placeholder)
# =====================================================
with tabs[2]:
    st.info("Adaptive Intelligence module coming soon.")

# =====================================================
# Operations (Placeholder)
# =====================================================
with tabs[3]:
    st.info("Operations & diagnostics coming soon.")