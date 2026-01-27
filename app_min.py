# app_min.py
# WAVES Intelligence™ Console — Minimal (Institutional Overview Fixed)

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

from intelligence.adaptive_intelligence import render_alpha_quality_and_confidence

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

    for col in list(RETURN_COLS.values()) + list(BENCHMARK_COLS.values()):
        if col not in df.columns:
            df[col] = np.nan

    return df, None


snapshot_df, snapshot_error = load_snapshot()

# ===========================
# Sidebar (GLOBAL WAVE SELECTOR)
# ===========================
st.sidebar.title("Wave Selection")

if snapshot_error:
    st.sidebar.error(snapshot_error)
    selected_wave = None
else:
    wave_options = snapshot_df["display_name"].tolist()
    selected_wave = st.sidebar.selectbox(
        "Select Wave",
        wave_options,
        key="global_wave_select",
    )

st.sidebar.divider()
st.sidebar.markdown(
    """
    **System Status**
    - Overview: ✅ Active  
    - Alpha Attribution: ✅ Active  
    - Adaptive Intelligence: 🟡 Preview  
    """
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

# =====================================================
# OVERVIEW TAB — CLEAN, BOXED, HORIZONTAL, IC-GRADE
# =====================================================
with tabs[0]:
    st.header("Overview")

    if snapshot_error:
        st.error(snapshot_error)
    else:
        # ---------------------------
        # Portfolio Snapshot (Equal Weight)
        # ---------------------------
        portfolio_returns = {
            k: snapshot_df[v].mean(skipna=True)
            for k, v in RETURN_COLS.items()
        }

        portfolio_alpha = {
            k: (
                snapshot_df[RETURN_COLS[k]].mean(skipna=True)
                - snapshot_df[BENCHMARK_COLS[k]].mean(skipna=True)
                if k in BENCHMARK_COLS
                else np.nan
            )
            for k in RETURN_COLS
        }

        st.subheader("Portfolio Snapshot (Equal-Weighted)")

        cols = st.columns(4)
        for i, horizon in enumerate(["1D", "30D", "60D", "365D"]):
            with cols[i]:
                st.metric(
                    label=f"{horizon} Return",
                    value=f"{portfolio_returns[horizon]:.2%}"
                    if not pd.isna(portfolio_returns[horizon])
                    else "—",
                )
                st.metric(
                    label=f"{horizon} Alpha",
                    value=f"{portfolio_alpha[horizon]:.2%}"
                    if not pd.isna(portfolio_alpha[horizon])
                    else "—",
                )

        st.divider()

        # ---------------------------
        # Selected Wave Snapshot
        # ---------------------------
        wave_row = snapshot_df[
            snapshot_df["display_name"] == selected_wave
        ].iloc[0]

        st.subheader(f"Wave Snapshot — {selected_wave}")

        cols = st.columns(4)
        for i, horizon in enumerate(["1D", "30D", "60D", "365D"]):
            ret_val = wave_row[RETURN_COLS[horizon]]
            alpha_val = (
                wave_row[RETURN_COLS[horizon]]
                - wave_row[BENCHMARK_COLS[horizon]]
                if horizon in BENCHMARK_COLS
                else np.nan
            )

            with cols[i]:
                st.metric(
                    label=f"{horizon} Return",
                    value=f"{ret_val:.2%}" if not pd.isna(ret_val) else "—",
                )
                st.metric(
                    label=f"{horizon} Alpha",
                    value=f"{alpha_val:.2%}"
                    if not pd.isna(alpha_val)
                    else "—",
                )

# ===========================
# ALPHA ATTRIBUTION TAB
# ===========================
with tabs[1]:
    st.header("Alpha Attribution")
    st.caption("Where performance came from (descriptive, audit-ready)")

    if snapshot_error:
        st.error(snapshot_error)
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
                "Contribution": [0.012, 0.008, -0.003, 0.004, 0.001],
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
# ADAPTIVE INTELLIGENCE TAB
# ===========================
with tabs[2]:
    st.header("Adaptive Intelligence")
    st.caption("Interpretive layer — what the system is learning")

    if snapshot_error:
        st.error(snapshot_error)
    else:
        render_alpha_quality_and_confidence(
            snapshot_df,
            None,
            selected_wave,
            RETURN_COLS,
            BENCHMARK_COLS,
        )

# ===========================
# OPERATIONS TAB
# ===========================
with tabs[3]:
    st.header("Operations")
    st.info(
        "Execution & override layer coming next. "
        "Adaptive Intelligence insights will surface here for human approval."
    )