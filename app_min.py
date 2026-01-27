# app_min.py
# WAVES Intelligence™ Console — Minimal (B2)
# Overview (Institutional) + Alpha Attribution + Adaptive Intelligence

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
# Styling (SAFE – no HTML rendering)
# ---------------------------
st.markdown(
    """
    <style>
    .snapshot-box {
        border-radius: 18px;
        padding: 22px;
        border: 1px solid rgba(0, 255, 255, 0.35);
        box-shadow: 0 0 18px rgba(0, 255, 255, 0.15);
        margin-bottom: 26px;
        background: linear-gradient(180deg, rgba(15,25,40,0.85), rgba(10,15,25,0.95));
    }
    .snapshot-title {
        font-size: 1.4rem;
        font-weight: 700;
        margin-bottom: 4px;
    }
    .snapshot-subtitle {
        color: #9aa4b2;
        margin-bottom: 18px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------
# Constants
# ---------------------------
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

# ---------------------------
# Load Snapshot
# ---------------------------
def load_snapshot():
    if not LIVE_SNAPSHOT_PATH.exists():
        return None, "Live snapshot file not found"

    df = pd.read_csv(LIVE_SNAPSHOT_PATH)
    df.columns = [c.strip().lower() for c in df.columns]

    if "display_name" not in df.columns:
        df["display_name"] = df.get("wave_name", df.get("wave_id", "Unnamed Wave"))

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
    ["Overview", "Alpha Attribution", "Adaptive Intelligence", "Operations"]
)

# ===========================
# OVERVIEW (FIXED)
# ===========================
with tabs[0]:
    st.header("Portfolio Overview")

    if snapshot_error:
        st.error(snapshot_error)
    else:
        df = snapshot_df.copy()

        # ---------- Portfolio Snapshot (Equal Weight)
        st.markdown('<div class="snapshot-box">', unsafe_allow_html=True)
        st.markdown('<div class="snapshot-title">🏛️ Portfolio Snapshot</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="snapshot-subtitle">Equal-Weighted Diagnostic Portfolio · Live Data</div>',
            unsafe_allow_html=True,
        )

        ret_cols = st.columns(4)
        for i, (label, col) in enumerate(RETURN_COLS.items()):
            val = df[col].mean(skipna=True)
            ret_cols[i].metric(label, f"{val:.2%}" if pd.notna(val) else "—")

        alpha_cols = st.columns(4)
        for i, (label, col) in enumerate(ALPHA_COLS.items()):
            val = df[col].mean(skipna=True)
            alpha_cols[i].metric(f"Alpha {label}", f"{val:.2%}" if pd.notna(val) else "—")

        st.markdown("</div>", unsafe_allow_html=True)

        # ---------- Wave-Level Snapshots
        for _, row in df.iterrows():
            st.markdown('<div class="snapshot-box">', unsafe_allow_html=True)
            st.markdown(
                f'<div class="snapshot-title">🏛️ {row["display_name"]}</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                '<div class="snapshot-subtitle">Wave-Level Diagnostic Snapshot</div>',
                unsafe_allow_html=True,
            )

            ret_cols = st.columns(4)
            for i, (label, col) in enumerate(RETURN_COLS.items()):
                val = row[col]
                ret_cols[i].metric(label, f"{val:.2%}" if pd.notna(val) else "—")

            alpha_cols = st.columns(4)
            for i, (label, col) in enumerate(ALPHA_COLS.items()):
                val = row[col]
                alpha_cols[i].metric(f"Alpha {label}", f"{val:.2%}" if pd.notna(val) else "—")

            st.markdown("</div>", unsafe_allow_html=True)

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
        selected_wave = st.selectbox("Select Wave", waves)

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
                    snapshot_df.loc[snapshot_df["display_name"] == selected_wave].get(c, pd.Series([None])).values[0]
                    for c in [
                        "selection_alpha",
                        "momentum_alpha",
                        "regime_alpha",
                        "exposure_alpha",
                        "residual_alpha",
                    ]
                ],
            }
        )

        st.dataframe(source_df, use_container_width=True, hide_index=True)

        render_alpha_quality_and_confidence(
            snapshot_df,
            None,
            selected_wave,
            RETURN_COLS,
            {},
        )

# ===========================
# ADAPTIVE INTELLIGENCE
# ===========================
with tabs[2]:
    st.header("Adaptive Intelligence")
    st.caption("Read-only interpretive layer")

    if snapshot_error:
        st.error(snapshot_error)
    else:
        waves = snapshot_df["display_name"].tolist()
        selected_wave = st.selectbox("Select Wave", waves)

        render_adaptive_intelligence_preview(
            snapshot_df,
            None,
            selected_wave,
            RETURN_COLS,
            {},
        )

# ===========================
# OPERATIONS
# ===========================
with tabs[3]:
    st.header("Operations")
    st.info("Execution & override controls will live here.")