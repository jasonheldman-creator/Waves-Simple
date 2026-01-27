# app_min.py
# WAVES Intelligence™ Console (Minimal)
# OPTION A — Alpha Attribution + Adaptive Intelligence (Preview)

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

from intelligence.adaptive_intelligence import (
    render_alpha_quality_and_confidence,
)

# ---------------------------
# Page Config
# ---------------------------
st.set_page_config(
    page_title="WAVES Intelligence™ Console",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------
# Constants
# ---------------------------
DATA_DIR = Path("data")
LIVE_SNAPSHOT_PATH = DATA_DIR / "live_snapshot.csv"

RETURN_COLS = {
    "intraday": "return_1d",
    "30d": "return_30d",
    "60d": "return_60d",
    "365d": "return_365d",
}

BENCHMARK_COLS = {
    "30d": "benchmark_return_30d",
    "60d": "benchmark_return_60d",
    "365d": "benchmark_return_365d",
}

# ---------------------------
# Load Snapshot
# ---------------------------
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

# ---------------------------
# Sidebar
# ---------------------------
st.sidebar.title("System Status")

st.sidebar.markdown(
    f"""
**Live Snapshot:** {'✅ Loaded' if snapshot_error is None else '❌ Missing'}  
**Alpha Attribution Engine:** ✅ Enabled  
**Adaptive Intelligence:** 🟡 Preview Mode  
"""
)

st.sidebar.divider()

# ---------------------------
# Tabs
# ---------------------------
tabs = st.tabs(
    [
        "Overview",
        "Alpha Attribution",
        "Adaptive Intelligence",
        "Operations",
    ]
)

# ===========================
# OVERVIEW
# ===========================
with tabs[0]:
    st.header("Portfolio & Wave Performance Snapshot")

    if snapshot_error:
        st.error(snapshot_error)
    else:
        df = snapshot_df.copy()

        portfolio_row = {"display_name": "TOTAL PORTFOLIO"}
        for col in RETURN_COLS.values():
            portfolio_row[col] = df[col].mean(skipna=True)

        df = pd.concat([pd.DataFrame([portfolio_row]), df], ignore_index=True)

        view = df[
            ["display_name"] + list(RETURN_COLS.values())
        ].rename(
            columns={
                "display_name": "Wave",
                "return_1d": "Intraday",
                "return_30d": "30D Return",
                "return_60d": "60D Return",
                "return_365d": "365D Return",
            }
        )

        view = view.replace({np.nan: "—"})
        st.dataframe(view, use_container_width=True, hide_index=True)

# ===========================
# ALPHA ATTRIBUTION
# ===========================
with tabs[1]:
    st.header("Alpha Attribution")

    if snapshot_error:
        st.error("Alpha Attribution requires a valid snapshot.")
    else:
        waves = snapshot_df["display_name"].tolist()

        selected_wave = st.selectbox(
            "Select Wave",
            waves,
            key="alpha_attr_wave_select",
        )

        # Placeholder attribution sources (Option A)
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

        # Core Alpha Quality & Confidence
        render_alpha_quality_and_confidence(
            snapshot_df,
            source_df,
            selected_wave,
            RETURN_COLS,
            BENCHMARK_COLS,
        )

# ===========================
# ADAPTIVE INTELLIGENCE (OPTION A)
# ===========================
with tabs[2]:
    st.header("Adaptive Intelligence")
    st.caption("Read-only preview derived from Alpha Attribution")

    if snapshot_error:
        st.warning("Adaptive Intelligence requires a valid snapshot.")
    else:
        waves = snapshot_df["display_name"].tolist()

        selected_wave = st.selectbox(
            "Select Wave",
            waves,
            key="adaptive_intel_wave_select",
        )

        # Interpretive preview layer
        st.subheader("Adaptive Signal Preview")

        st.info(
            "This is a **preview-only interpretive layer**. "
            "No adaptive actions, rebalancing, or execution is performed."
        )

        # Reuse quality & confidence as signal input
        render_alpha_quality_and_confidence(
            snapshot_df,
            None,
            selected_wave,
            RETURN_COLS,
            BENCHMARK_COLS,
        )

        st.divider()

        # Simple heuristic preview (Option A)
        st.metric("Adaptive Bias", "Neutral")
        st.metric("Confidence Regime", "Moderate")
        st.metric("Recommended Action", "Observe")

# ===========================
# OPERATIONS
# ===========================
with tabs[3]:
    st.header("Operations")
    st.info("Operations control center coming next.")