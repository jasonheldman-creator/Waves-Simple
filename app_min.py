# app_min.py
# WAVES Intelligence — Minimal Stable Console (RESTORED)
# PURPOSE:
# 1) Portfolio + Wave Returns & Alpha Summary (TOP)
# 2) Alpha Source Attribution (MIDDLE)
# 3) Alpha History Chart (BOTTOM)
# 4) Sidebar restored
#
# AUTHOR: Stabilized institutional rebuild

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
WAVE_HISTORY_PATH = DATA_DIR / "wave_history.csv"  # optional, handled safely

# -----------------------------
# Helpers
# -----------------------------
def safe_read_csv(path: Path):
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
# Sidebar
# -----------------------------
with st.sidebar:
    st.markdown("## WAVES Console")
    st.caption("Navigation & Diagnostics")

    live_ok = LIVE_SNAPSHOT_PATH.exists()
    alpha_ok = ALPHA_ATTR_PATH.exists()

    st.markdown("### Data Status")
    st.markdown(f"**Live Snapshot:** {'✅ True' if live_ok else '❌ False'}")
    st.markdown(f"**Alpha Attribution:** {'✅ True' if alpha_ok else '❌ False'}")

# -----------------------------
# Header + Tabs
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
# OVERVIEW TAB — RETURNS + ALPHA SUMMARY (TOP SECTION)
# =====================================================
with tabs[0]:
    st.subheader("Portfolio & Wave Performance Snapshot")

    snapshot_df = safe_read_csv(LIVE_SNAPSHOT_PATH)

    if snapshot_df is None:
        st.warning("Live snapshot not found.")
    elif snapshot_df.empty:
        st.info("Live snapshot exists but contains no rows.")
    else:
        snapshot_df = normalize_columns(snapshot_df)

        required_cols = {
            "display_name",
            "return_30d",
            "return_60d",
            "return_365d",
        }

        if not required_cols.issubset(snapshot_df.columns):
            st.error("Live snapshot missing required return columns.")
        else:
            # Portfolio row (simple average for display stability)
            portfolio_row = {
                "display_name": "Portfolio",
                "return_30d": snapshot_df["return_30d"].mean(),
                "return_60d": snapshot_df["return_60d"].mean(),
                "return_365d": snapshot_df["return_365d"].mean(),
            }

            summary_df = pd.concat(
                [
                    pd.DataFrame([portfolio_row]),
                    snapshot_df[[
                        "display_name",
                        "return_30d",
                        "return_60d",
                        "return_365d",
                    ]]
                ],
                ignore_index=True
            )

            st.dataframe(
                summary_df.rename(columns={
                    "display_name": "Wave",
                    "return_30d": "Return 30D",
                    "return_60d": "Return 60D",
                    "return_365d": "Return 365D",
                }),
                use_container_width=True,
            )

# =====================================================
# ALPHA ATTRIBUTION TAB — SOURCE BREAKDOWN (MIDDLE)
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

    required_cols = {
        "wave",
        "horizon",
        "regime_alpha",
        "exposure_alpha",
        "residual_alpha",
    }

    missing = required_cols - set(alpha_df.columns)
    if missing:
        st.error("Alpha attribution file is missing required columns:")
        st.code(sorted(missing))
        st.stop()

    # Portfolio option (aggregate)
    portfolio_rows = (
        alpha_df
        .groupby("horizon", as_index=False)
        .mean(numeric_only=True)
    )
    portfolio_rows["wave"] = "Portfolio"

    alpha_df = pd.concat([portfolio_rows, alpha_df], ignore_index=True)

    # Pronounced dropdowns
    st.markdown("### 🔎 Select Wave")
    wave = st.selectbox(
        "",
        sorted(alpha_df["wave"].unique()),
        index=0,
    )

    st.markdown("### 📅 Select Horizon (Days)")
    horizon = st.selectbox(
        "",
        sorted(alpha_df["horizon"].unique()),
        index=len(sorted(alpha_df["horizon"].unique())) - 1,
    )

    filtered = alpha_df[
        (alpha_df["wave"] == wave) &
        (alpha_df["horizon"] == horizon)
    ]

    st.markdown(f"## {wave} — {horizon} Day Alpha Sources")

    if filtered.empty:
        st.info("No data available for this selection.")
    else:
        display_cols = [
            c for c in filtered.columns
            if c.endswith("_alpha")
        ]

        st.dataframe(
            filtered[display_cols],
            use_container_width=True,
        )

# =====================================================
# ALPHA HISTORY TAB — TIME SERIES (BOTTOM SECTION)
# =====================================================
with tabs[1]:
    st.markdown("---")
    st.subheader("📈 Alpha History")

    history_df = safe_read_csv(WAVE_HISTORY_PATH)

    if history_df is None or history_df.empty:
        st.info("Alpha history not yet available.")
    else:
        history_df = normalize_columns(history_df)

        if {"date", "wave", "alpha"}.issubset(history_df.columns):
            wave_hist = history_df[history_df["wave"] == wave]
            if wave_hist.empty:
                st.info("No alpha history for this selection.")
            else:
                wave_hist["date"] = pd.to_datetime(wave_hist["date"])
                wave_hist = wave_hist.sort_values("date")

                st.line_chart(
                    wave_hist.set_index("date")["alpha"],
                    use_container_width=True,
                )
        else:
            st.info("Alpha history schema not ready yet.")

# =====================================================
# OTHER TABS
# =====================================================
with tabs[2]:
    st.subheader("Adaptive Intelligence")
    st.info("Adaptive intelligence metrics will appear here.")

with tabs[3]:
    st.subheader("Operations")
    st.info("Operational controls coming soon.")