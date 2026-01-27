# app_min.py
# WAVES Intelligence™ Console — Minimal (Institutional B2)
# Overview • Alpha Attribution • Adaptive Intelligence • Operations

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

from intelligence.adaptive_intelligence import (
    render_alpha_quality_and_confidence,
    render_adaptive_intelligence_preview,
)

# =========================================================
# Page Config
# =========================================================
st.set_page_config(
    page_title="WAVES Intelligence™ Console",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================================================
# Constants
# =========================================================
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

# =========================================================
# Load Data
# =========================================================
def load_snapshot():
    if not LIVE_SNAPSHOT_PATH.exists():
        return None, "Live snapshot file not found"

    df = pd.read_csv(LIVE_SNAPSHOT_PATH)
    df.columns = [c.strip().lower() for c in df.columns]

    if "display_name" not in df.columns:
        if "wave_name" in df.columns:
            df["display_name"] = df["wave_name"]
        else:
            df["display_name"] = df["wave_id"]

    return df, None


snapshot_df, snapshot_error = load_snapshot()

# =========================================================
# Styling
# =========================================================
st.markdown(
    """
<style>
.metric-box {
    background: linear-gradient(135deg, #0b1220, #0e1628);
    border-radius: 18px;
    padding: 26px;
    border: 1px solid rgba(0,255,255,0.18);
    box-shadow: 0 0 28px rgba(0,255,255,0.15);
    margin-bottom: 28px;
}

.metric-title {
    font-size: 26px;
    font-weight: 700;
    margin-bottom: 6px;
}

.metric-sub {
    color: #9aa4b2;
    font-size: 14px;
    margin-bottom: 22px;
}

.metric-row {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 16px;
    margin-bottom: 14px;
}

.metric-cell {
    background: rgba(255,255,255,0.04);
    border-radius: 14px;
    padding: 14px;
    text-align: center;
    border: 1px solid rgba(255,255,255,0.08);
}

.metric-label {
    font-size: 12px;
    letter-spacing: 1px;
    color: #9aa4b2;
}

.metric-value {
    font-size: 22px;
    font-weight: 700;
    margin-top: 4px;
}

.positive { color: #00ff9c; }
.negative { color: #ff6b6b; }
.neutral  { color: #d0d4dc; }
</style>
""",
    unsafe_allow_html=True,
)

# =========================================================
# Sidebar
# =========================================================
st.sidebar.title("System Status")
st.sidebar.markdown(
    f"""
**Live Snapshot:** {'✅ Loaded' if snapshot_error is None else '❌ Missing'}  
**Alpha Attribution:** ✅ Active  
**Adaptive Intelligence:** 🟡 Preview  
"""
)

# =========================================================
# Tabs
# =========================================================
tabs = st.tabs(["Overview", "Alpha Attribution", "Adaptive Intelligence", "Operations"])

# =========================================================
# OVERVIEW TAB
# =========================================================
with tabs[0]:
    st.header("Portfolio Overview")

    if snapshot_error:
        st.error(snapshot_error)
    else:
        df = snapshot_df.copy()

        # Equal-weight portfolio math
        returns = {
            k: df[v].mean(skipna=True)
            for k, v in RETURN_COLS.items()
            if v in df.columns
        }

        alphas = {
            k: df[v].mean(skipna=True)
            for k, v in ALPHA_COLS.items()
            if v in df.columns
        }

        def fmt(val):
            if pd.isna(val):
                return "—", "neutral"
            return f"{val*100:.2f}%", "positive" if val >= 0 else "negative"

        # Build HTML
        html = """
        <div class="metric-box">
            <div class="metric-title">🏛 Portfolio Snapshot</div>
            <div class="metric-sub">Equal-Weighted Diagnostic Portfolio · Live Data</div>

            <div class="metric-row">
        """

        for k in RETURN_COLS.keys():
            val, cls = fmt(returns.get(k))
            html += f"""
            <div class="metric-cell">
                <div class="metric-label">{k} RETURN</div>
                <div class="metric-value {cls}">{val}</div>
            </div>
            """

        html += "</div><div class='metric-row'>"

        for k in ALPHA_COLS.keys():
            val, cls = fmt(alphas.get(k))
            html += f"""
            <div class="metric-cell">
                <div class="metric-label">{k} ALPHA</div>
                <div class="metric-value {cls}">{val}</div>
            </div>
            """

        html += "</div></div>"

        st.markdown(html, unsafe_allow_html=True)

# =========================================================
# ALPHA ATTRIBUTION TAB
# =========================================================
with tabs[1]:
    st.header("Alpha Attribution")
    st.caption("Explains where performance came from")

    if snapshot_error:
        st.error(snapshot_error)
    else:
        waves = snapshot_df["display_name"].tolist()
        selected_wave = st.selectbox("Select Wave", waves, key="alpha_wave")

        rows = []
        for src in ["selection", "momentum", "regime", "exposure", "residual"]:
            col = f"{src}_alpha"
            rows.append(
                {
                    "Source": src.title(),
                    "Contribution": snapshot_df.loc[
                        snapshot_df["display_name"] == selected_wave, col
                    ].values[0]
                    if col in snapshot_df.columns
                    else None,
                }
            )

        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        render_alpha_quality_and_confidence(
            snapshot_df,
            None,
            selected_wave,
            RETURN_COLS,
            {},
        )

# =========================================================
# ADAPTIVE INTELLIGENCE TAB
# =========================================================
with tabs[2]:
    st.header("Adaptive Intelligence")
    st.caption("Interpretive layer · No execution")

    if snapshot_error:
        st.error(snapshot_error)
    else:
        waves = snapshot_df["display_name"].tolist()
        selected_wave = st.selectbox("Select Wave", waves, key="adaptive_wave")

        render_adaptive_intelligence_preview(
            snapshot_df,
            None,
            selected_wave,
            RETURN_COLS,
            {},
        )

# =========================================================
# OPERATIONS TAB
# =========================================================
with tabs[3]:
    st.header("Operations")
    st.info("Execution & override controls will live here.")