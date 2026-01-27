# app_min.py
# WAVES Intelligence™ Console — Minimal (Executive Overview v1)
# Overview • Alpha Attribution • Adaptive Intelligence (Preview) • Operations

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

from intelligence.adaptive_intelligence import (
    render_alpha_quality_and_confidence,
    render_adaptive_intelligence_preview,
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

ALPHA_COLS = {
    "30d": "alpha_30d",
    "60d": "alpha_60d",
    "365d": "alpha_365d",
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

    for col in list(RETURN_COLS.values()) + list(ALPHA_COLS.values()):
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
    **Alpha Attribution:** ✅ Active  
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
# OVERVIEW — EXECUTIVE (A)
# ===========================
with tabs[0]:
    st.header("Portfolio Snapshot")

    if snapshot_error:
        st.error(snapshot_error)
    else:
        df = snapshot_df.copy()

        # ---------------------------
        # Portfolio Aggregates
        # ---------------------------
        portfolio_metrics = {}

        for label, col in RETURN_COLS.items():
            portfolio_metrics[col] = df[col].mean(skipna=True)

        for label, col in ALPHA_COLS.items():
            portfolio_metrics[col] = df[col].mean(skipna=True)

        # ---------------------------
        # HERO PANEL
        # ---------------------------
        st.markdown("### 🏛 TOTAL PORTFOLIO")
        st.caption("Executive summary — current portfolio state")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Intraday Return", f"{portfolio_metrics['return_1d']:.2%}")
        c2.metric("30D Return", f"{portfolio_metrics['return_30d']:.2%}")
        c3.metric("60D Return", f"{portfolio_metrics['return_60d']:.2%}")
        c4.metric("365D Return", f"{portfolio_metrics['return_365d']:.2%}")

        c5, c6, c7 = st.columns(3)
        c5.metric("Alpha 30D", f"{portfolio_metrics['alpha_30d']:.2%}")
        c6.metric("Alpha 60D", f"{portfolio_metrics['alpha_60d']:.2%}")
        c7.metric("Alpha 365D", f"{portfolio_metrics['alpha_365d']:.2%}")

        st.divider()

        # ---------------------------
        # EXECUTIVE POSTURE READ
        # ---------------------------
        st.subheader("Portfolio Posture")
        p1, p2, p3 = st.columns(3)

        p1.metric("Risk Posture", "Neutral")
        p2.metric("Breadth", "Balanced")
        p3.metric("Volatility Context", "Normal")

        st.divider()

        # ---------------------------
        # WAVE PERFORMANCE (DETAIL)
        # ---------------------------
        st.subheader("Wave Performance — Detail")

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
                "return_30d": "30D",
                "return_60d": "60D",
                "return_365d": "365D",
            }
        )

        view = view.replace({np.nan: "—"})
        st.dataframe(view, use_container_width=True, hide_index=True)

# ===========================
# ALPHA ATTRIBUTION
# ===========================
with tabs[1]:
    st.header("Alpha Attribution")
    st.caption("Explains where performance came from")

    if snapshot_error:
        st.error(snapshot_error)
    else:
        waves = snapshot_df["display_name"].tolist()

        selected_wave = st.selectbox(
            "Select Wave",
            waves,
            key="alpha_attr_wave_select",
        )

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
                    snapshot_df.loc[snapshot_df["display_name"] == selected_wave, col].values[0]
                    if col in snapshot_df.columns
                    else None
                    for col in [
                        "selection_alpha",
                        "momentum_alpha",
                        "regime_alpha",
                        "exposure_alpha",
                        "residual_alpha",
                    ]
                ],
            }
        )

        st.subheader("Source Breakdown")
        st.dataframe(source_df, use_container_width=True, hide_index=True)

        render_alpha_quality_and_confidence(
            snapshot_df,
            None,
            selected_wave,
            RETURN_COLS,
            BENCHMARK_COLS,
        )

# ===========================
# ADAPTIVE INTELLIGENCE (Preview)
# ===========================
with tabs[2]:
    st.header("Adaptive Intelligence")
    st.caption("Read-only interpretive layer derived from Alpha Attribution")

    if snapshot_error:
        st.error(snapshot_error)
    else:
        waves = snapshot_df["display_name"].tolist()

        selected_wave = st.selectbox(
            "Select Wave",
            waves,
            key="adaptive_intel_wave_select",
        )

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
    st.info("Human-in-the-loop controls and overrides will live here.")