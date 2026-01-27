# app_min.py
# WAVES Intelligence™ Console (Minimal)
# IC-GRADE POLISH — Institutional Overview Rewrite

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

# ===========================
# CSS — Institutional Cards
# ===========================
st.markdown("""
<style>
.metric-card {
    background: linear-gradient(135deg, #0b132b, #111827);
    border: 1px solid rgba(0,255,255,0.25);
    border-radius: 18px;
    padding: 22px;
    margin-bottom: 24px;
    box-shadow: 0 0 18px rgba(0,255,255,0.15);
}
.metric-title {
    font-size: 22px;
    font-weight: 700;
    margin-bottom: 6px;
}
.metric-sub {
    color: #9ca3af;
    font-size: 13px;
    margin-bottom: 14px;
}
.metric-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 14px;
}
.metric-cell {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 12px;
    padding: 12px;
    text-align: center;
}
.metric-label {
    font-size: 11px;
    color: #9ca3af;
    margin-bottom: 6px;
}
.metric-value {
    font-size: 18px;
    font-weight: 700;
}
.pos { color: #22c55e; }
.neg { color: #ef4444; }
.na  { color: #9ca3af; }
.divider {
    height: 1px;
    background: rgba(255,255,255,0.08);
    margin: 16px 0;
}
</style>
""", unsafe_allow_html=True)

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
    """
)
st.sidebar.divider()

if snapshot_error is None:
    wave_list = snapshot_df["display_name"].tolist()
    selected_wave = st.sidebar.selectbox(
        "Wave Diagnostic View",
        wave_list,
        index=0
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
    "Operations"
])

# ===========================
# Helper: Metric Grid
# ===========================
def render_metric_grid(values: dict):
    html = '<div class="metric-grid">'
    for label, val in values.items():
        if val is None or pd.isna(val):
            cls = "na"
            txt = "—"
        else:
            cls = "pos" if val >= 0 else "neg"
            txt = f"{val*100:.2f}%"
        html += f"""
        <div class="metric-cell">
            <div class="metric-label">{label}</div>
            <div class="metric-value {cls}">{txt}</div>
        </div>
        """
    html += "</div>"
    return html

# ===========================
# OVERVIEW TAB
# ===========================
with tabs[0]:
    st.header("Portfolio Overview")

    if snapshot_error:
        st.error(snapshot_error)
    else:
        df = snapshot_df.copy()

        # ---- Portfolio (Equal-Weighted Diagnostic)
        portfolio_vals = {}
        portfolio_alpha = {}

        for k, col in RETURN_COLS.items():
            portfolio_vals[k] = df[col].mean(skipna=True)

        for k, bcol in BENCHMARK_COLS.items():
            rcol = RETURN_COLS[k]
            portfolio_alpha[f"α {k}"] = (
                df[rcol].mean(skipna=True) - df[bcol].mean(skipna=True)
                if bcol in df.columns else None
            )

        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">🏛 Portfolio Snapshot</div>
            <div class="metric-sub">Equal-Weighted Diagnostic Portfolio · Live Data</div>
            {render_metric_grid(portfolio_vals)}
            <div class="divider"></div>
            {render_metric_grid(portfolio_alpha)}
        </div>
        """, unsafe_allow_html=True)

        # ---- Wave Snapshot
        wave_df = df[df["display_name"] == selected_wave].iloc[0]

        wave_vals = {k: wave_df[v] for k, v in RETURN_COLS.items()}
        wave_alpha = {
            f"α {k}": (
                wave_df[RETURN_COLS[k]] - wave_df[BENCHMARK_COLS[k]]
                if k in BENCHMARK_COLS and BENCHMARK_COLS[k] in df.columns
                else None
            )
            for k in RETURN_COLS
        }

        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">🏛 {selected_wave}</div>
            <div class="metric-sub">Wave-Level Diagnostic Snapshot</div>
            {render_metric_grid(wave_vals)}
            <div class="divider"></div>
            {render_metric_grid(wave_alpha)}
        </div>
        """, unsafe_allow_html=True)

# ===========================
# ALPHA ATTRIBUTION TAB
# ===========================
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

# ===========================
# ADAPTIVE INTELLIGENCE TAB
# ===========================
with tabs[2]:
    st.header("Adaptive Intelligence")
    st.caption("Interpretive layer derived from Alpha Attribution")

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
    st.info("Execution & override controls will live here.")