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
)

# ===========================
# Page Config
# ===========================
st.set_page_config(
    page_title="WAVES Intelligence™ Console",
    layout="wide",
    initial_sidebar_state="expanded"
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

ALPHA_COLS = {
    "1D": "alpha_1d",
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
tabs = st.tabs([
    "Overview",
    "Alpha Attribution",
    "Adaptive Intelligence",
    "Operations",
])

# ============================================================
# OVERVIEW TAB — ALLOWED MODIFICATION ZONE
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
            if pct > 0:
                return f"+{pct:.2f}%"
            return f"{pct:.2f}%"

        # ============================================================
        # PORTFOLIO SNAPSHOT CARD
        # ============================================================
        portfolio_returns = {
            k: df[v].mean(skipna=True)
            for k, v in RETURN_COLS.items()
        }

        portfolio_alpha = {
            k: df[v].mean(skipna=True)
            for k, v in ALPHA_COLS.items()
        }

        with st.container():
            st.markdown("### 🏛️ Portfolio Snapshot — Equal-Weighted")
            st.caption("Equal-weighted performance across all active waves.")
            st.divider()

            st.markdown("**Returns**")
            ret_cols = st.columns(4)
            for i, (label, value) in enumerate(portfolio_returns.items()):
                ret_cols[i].markdown(
                    f"<div style='line-height:1.4'><span style='font-size:0.85rem; font-weight:500; color:#666;'>{label}:</span><br>"
                    f"<span style='font-size:1.25rem; font-weight:700;'>{format_percentage(value)}</span></div>",
                    unsafe_allow_html=True,
                )

            st.markdown("**Alpha**")
            alpha_cols = st.columns(4)
            for i, (label, value) in enumerate(portfolio_alpha.items()):
                alpha_cols[i].markdown(
                    f"<div style='line-height:1.4'><span style='font-size:0.85rem; font-weight:500; color:#666;'>{label}:</span><br>"
                    f"<span style='font-size:1.25rem; font-weight:700;'>{format_percentage(value)}</span></div>",
                    unsafe_allow_html=True,
                )

        st.divider()

        # ============================================================
        # SELECTED WAVE SNAPSHOT CARD
        # ============================================================
        if selected_wave is not None:
            wave_row = df[df["display_name"] == selected_wave].iloc[0]

            wave_returns = {
                k: wave_row[v]
                for k, v in RETURN_COLS.items()
            }

            wave_alpha = {
                k: wave_row[v]
                for k, v in ALPHA_COLS.items()
            }

            with st.container():
                st.markdown(f"### 📊 Selected Wave Snapshot — {selected_wave}")
                st.caption("Performance profile for the selected wave.")
                st.divider()

                st.markdown("**Returns**")
                wret_cols = st.columns(4)
                for i, (label, value) in enumerate(wave_returns.items()):
                    wret_cols[i].markdown(
                        f"<div style='line-height:1.4'><span style='font-size:0.85rem; font-weight:500; color:#666;'>{label}:</span><br>"
                        f"<span style='font-size:1.25rem; font-weight:700;'>{format_percentage(value)}</span></div>",
                        unsafe_allow_html=True,
                    )

                st.markdown("**Alpha**")
                walpha_cols = st.columns(4)
                for i, (label, value) in enumerate(wave_alpha.items()):
                    walpha_cols[i].markdown(
                        f"<div style='line-height:1.4'><span style='font-size:0.85rem; font-weight:500; color:#666;'>{label}:</span><br>"
                        f"<span style='font-size:1.25rem; font-weight:700;'>{format_percentage(value)}</span></div>",
                        unsafe_allow_html=True,
                    )

# ============================================================
# ALPHA ATTRIBUTION TAB — DO NOT MODIFY
# ============================================================
with tabs[1]:
    st.header("Alpha Attribution")

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

# ============================================================
# ADAPTIVE INTELLIGENCE TAB — DO NOT MODIFY
# ============================================================
with tabs[2]:
    st.header("Adaptive Intelligence")
    st.caption("Interpretive layer derived from attribution")

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

# ============================================================
# OPERATIONS TAB — DO NOT MODIFY
# ============================================================
with tabs[3]:
    st.header("Operations")
    st.info("Execution & override layer coming next.")
