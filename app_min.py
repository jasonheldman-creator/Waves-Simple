# ============================================================
# app_min.py
# WAVES Intelligence™ Console (Minimal)
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

from intelligence.adaptive_intelligence import (
    render_alpha_quality_and_confidence,
    render_alpha_attribution_drivers,
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
    "INTRADAY": "return_intraday",
    "30D": "return_30d",
    "60D": "return_60d",
    "365D": "return_365d",
}

BENCHMARK_COLS = {
    "30D": "benchmark_return_30d",
    "60D": "benchmark_return_60d",
    "365D": "benchmark_return_365d",
}

ALPHA_COLS = {
    "INTRADAY": "alpha_intraday",
    "30D": "alpha_30d",
    "60D": "alpha_60d",
    "365D": "alpha_365d",
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

    for col in (
        list(RETURN_COLS.values())
        + list(BENCHMARK_COLS.values())
        + list(ALPHA_COLS.values())
    ):
        if col not in df.columns:
            df[col] = np.nan

    if "intraday_label" not in df.columns:
        df["intraday_label"] = None

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
    **Adaptive Intelligence:** 🟡 Interpretive  
    """
)

st.sidebar.divider()

if snapshot_df is not None:
    selected_wave = st.sidebar.selectbox(
        "Select Wave",
        snapshot_df["display_name"].tolist(),
        key="global_wave_select",
    )
else:
    selected_wave = None

# ===========================
# Tabs
# ===========================
tabs = st.tabs(
    ["Overview", "Alpha Attribution", "Adaptive Intelligence", "Operations"]
)

# ============================================================
# OVERVIEW TAB
# ============================================================
with tabs[0]:
    st.header("Portfolio Overview")

    if snapshot_error:
        st.error(snapshot_error)
    else:
        df = snapshot_df.copy()

        def format_percentage(value: float) -> str:
            if pd.isna(value):
                return "—"
            if abs(value) < 1e-10:
                return "0.00%"
            pct = value * 100
            return f"+{pct:.2f}%" if pct > 0 else f"{pct:.2f}%"

        portfolio_returns = {k: df[v].mean(skipna=True) for k, v in RETURN_COLS.items()}
        portfolio_alpha = {k: df[v].mean(skipna=True) for k, v in ALPHA_COLS.items()}

        with st.container():
            st.markdown("### 🏛️ Portfolio Snapshot — Equal-Weighted")
            st.divider()

            st.markdown("**Returns**")
            cols = st.columns(4)
            for i, (label, value) in enumerate(portfolio_returns.items()):
                cols[i].metric(label, format_percentage(value))

            st.markdown("**Alpha**")
            cols = st.columns(4)
            for i, (label, value) in enumerate(portfolio_alpha.items()):
                cols[i].metric(label, format_percentage(value))

# ============================================================
# ALPHA ATTRIBUTION TAB (CANONICAL HOME)
# ============================================================
with tabs[1]:
    st.header("Alpha Attribution")
    st.caption("Governance-native view of realized alpha and its drivers.")
    st.divider()

    if snapshot_error:
        st.error(snapshot_error)
    else:
        # ---- Alpha Quality & Confidence (ONLY PLACE IT APPEARS)
        render_alpha_quality_and_confidence(
            snapshot_df,
            None,
            selected_wave,
            RETURN_COLS,
            BENCHMARK_COLS,
        )

        st.divider()

        # ---- Alpha Attribution Drivers
        st.subheader("Alpha Attribution Drivers")
        st.caption("Explanatory decomposition of realized alpha across strategy components.")
        st.divider()

        render_alpha_attribution_drivers(
            snapshot_df,
            selected_wave,
            RETURN_COLS,
            BENCHMARK_COLS,
        )

# ============================================================
# ADAPTIVE INTELLIGENCE TAB (NO DUPLICATES)
# ============================================================
with tabs[2]:
    st.header("Adaptive Intelligence")
    st.caption("Interpretive layer — forward-looking, read-only.")
    st.divider()

    if selected_wave is None:
        st.info("Select a wave to view adaptive intelligence.")
    else:
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Adaptive Bias", "Neutral")

        with col2:
            st.metric("Regime Read", "Undetermined")

        with col3:
            st.metric("Signal Strength", "—")

        st.info(
            "Adaptive Intelligence is interpretive only. "
            "No trading logic or automated actions are executed."
        )

# ============================================================
# OPERATIONS TAB
# ============================================================
with tabs[3]:
    st.header("Operations")
    st.info("Execution & override layer coming next.")