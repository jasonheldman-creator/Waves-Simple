# app_min.py
# WAVES Intelligence™ Console (Minimal)
# IC-GRADE OVERVIEW + ATTRIBUTION + ADAPTIVE INTELLIGENCE

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

from intelligence.adaptive_intelligence import (
    render_alpha_quality_and_confidence,
    render_adaptive_intelligence_preview,
)

# ===========================
# Page Config
# ===========================
st.set_page_config(
    page_title="WAVES Intelligence™ Console",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ===========================
# Constants
# ===========================
DATA_DIR = Path("data")
LIVE_SNAPSHOT_PATH = DATA_DIR / "live_snapshot.csv"

RETURN_COLS = {
    "1D": "return_1d",
    "30D": "return_30d",
    "60D": "return_60d",
    "365D": "return_365d",
}

ALPHA_COLS = {
    "1D": "alpha_1d",
    "30D": "alpha_30d",
    "60D": "alpha_60d",
    "365D": "alpha_365d",
}

BENCHMARK_COLS = {
    "30D": "benchmark_return_30d",
    "60D": "benchmark_return_60d",
    "365D": "benchmark_return_365d",
}

# ===========================
# Load Snapshot
# ===========================
def load_snapshot():
    if not LIVE_SNAPSHOT_PATH.exists():
        return None, "Live snapshot file not found"

    df = pd.read_csv(LIVE_SNAPSHOT_PATH)
    df.columns = [c.strip().lower() for c in df.columns]

    if "display_name" not in df.columns:
        if "wave_name" in df.columns:
            df["display_name"] = df["wave_name"]
        elif "wave_id" in df.columns:
            df["display_name"] = df["wave_id"]
        else:
            df["display_name"] = "Unnamed Wave"

    for col in list(RETURN_COLS.values()) + list(ALPHA_COLS.values()):
        if col not in df.columns:
            df[col] = np.nan

    return df, None


snapshot_df, snapshot_error = load_snapshot()

# ===========================
# Sidebar
# ===========================
st.sidebar.title("System Status")
st.sidebar.markdown(
    f"""
    **Live Snapshot:** {'✅ Loaded' if snapshot_error is None else '❌ Missing'}  
    **Alpha Attribution:** ✅ Active  
    **Adaptive Intelligence:** 🟡 Preview  
    """
)
st.sidebar.divider()

# Global wave selector (single source of truth)
if snapshot_error:
    selected_wave = None
else:
    st.sidebar.subheader("Wave Selection")
    selected_wave = st.sidebar.selectbox(
        "Select Wave",
        snapshot_df["display_name"].tolist(),
        key="global_wave_selector",
    )

# ===========================
# Tabs
# ===========================
tabs = st.tabs(
    [
        "Overview",
        "Alpha Attribution",
        "Adaptive Intelligence",
        "Operations",
    ]
)

# ===========================
# OVERVIEW TAB (LOCKED)
# ===========================
with tabs[0]:
    st.header("Portfolio Overview")

    if snapshot_error:
        st.error(snapshot_error)
    else:
        df = snapshot_df.copy()

        # ---------------------------
        # Portfolio Snapshot (Equal-Weighted)
        # ---------------------------
        portfolio_returns = {
            k: df[v].mean(skipna=True)
            for k, v in RETURN_COLS.items()
        }
        portfolio_alpha = {
            k: df[v].mean(skipna=True)
            for k, v in ALPHA_COLS.items()
        }

        with st.container():
            st.subheader("🏛️ Portfolio Snapshot — Equal-Weighted")

            # RETURNS ROW
            cols = st.columns(4)
            for i, (label, value) in enumerate(portfolio_returns.items()):
                cols[i].metric(
                    label=f"{label} Return",
                    value="—" if pd.isna(value) else f"{value:.2%}",
                )

            # ALPHA ROW
            cols = st.columns(4)
            for i, (label, value) in enumerate(portfolio_alpha.items()):
                cols[i].metric(
                    label=f"{label} Alpha",
                    value="—" if pd.isna(value) else f"{value:.2%}",
                )

        st.divider()

        # ---------------------------
        # Selected Wave Snapshot
        # ---------------------------
        if selected_wave is not None:
            wave_row = df[df["display_name"] == selected_wave].iloc[0]

            wave_returns = {
                k: wave_row[v] for k, v in RETURN_COLS.items()
            }
            wave_alpha = {
                k: wave_row[v] for k, v in ALPHA_COLS.items()
            }

            with st.container():
                st.subheader(f"📊 Wave Snapshot — {selected_wave}")

                # RETURNS ROW
                cols = st.columns(4)
                for i, (label, value) in enumerate(wave_returns.items()):
                    cols[i].metric(
                        label=f"{label} Return",
                        value="—" if pd.isna(value) else f"{value:.2%}",
                    )

                # ALPHA ROW
                cols = st.columns(4)
                for i, (label, value) in enumerate(wave_alpha.items()):
                    cols[i].metric(
                        label=f"{label} Alpha",
                        value="—" if pd.isna(value) else f"{value:.2%}",
                    )

# ===========================
# ALPHA ATTRIBUTION
# ===========================
with tabs[1]:
    st.header("Alpha Attribution")
    st.caption("Explains where returns came from")

    if snapshot_error or selected_wave is None:
        st.info("Select a wave to view attribution.")
    else:
        source_df = pd.DataFrame(
            {
                "Alpha Source": [
                    "Selection Alpha",
                    "Momentum Alpha",
                    "Regime Alpha",
                    "Exposure Alpha",
                    "Residual Alpha",
                ],
                "Contribution": [
                    snapshot_df.loc[
                        snapshot_df["display_name"] == selected_wave,
                        "selection_alpha",
                    ].values[0]
                    if "selection_alpha" in snapshot_df.columns
                    else np.nan,
                    snapshot_df.loc[
                        snapshot_df["display_name"] == selected_wave,
                        "momentum_alpha",
                    ].values[0]
                    if "momentum_alpha" in snapshot_df.columns
                    else np.nan,
                    snapshot_df.loc[
                        snapshot_df["display_name"] == selected_wave,
                        "regime_alpha",
                    ].values[0]
                    if "regime_alpha" in snapshot_df.columns
                    else np.nan,
                    snapshot_df.loc[
                        snapshot_df["display_name"] == selected_wave,
                        "exposure_alpha",
                    ].values[0]
                    if "exposure_alpha" in snapshot_df.columns
                    else np.nan,
                    snapshot_df.loc[
                        snapshot_df["display_name"] == selected_wave,
                        "residual_alpha",
                    ].values[0]
                    if "residual_alpha" in snapshot_df.columns
                    else np.nan,
                ],
            }
        )

        st.subheader("Source Breakdown")
        st.dataframe(source_df, use_container_width=True, hide_index=True)

        render_alpha_quality_and_confidence(
            snapshot_df,
            source_df,
            selected_wave,
            RETURN_COLS,
            BENCHMARK_COLS,
        )

# ===========================
# ADAPTIVE INTELLIGENCE
# ===========================
with tabs[2]:
    st.header("Adaptive Intelligence")
    st.caption("Interpretive, forward-looking (read-only)")

    if snapshot_error or selected_wave is None:
        st.info("Select a wave to view adaptive intelligence.")
    else:
        render_adaptive_intelligence_preview(
            snapshot_df,
            None,
            selected_wave,
            RETURN_COLS,
            BENCHMARK_COLS,
        )

# ===========================
# OPERATIONS
# ===========================
with tabs[3]:
    st.header("Operations")
    st.info("Human-in-the-loop execution & overrides coming next.")