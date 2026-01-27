# app_min.py
# WAVES Intelligence™ Console — Minimal (Institutional Overview Upgrade)

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
    "intraday": "alpha_1d",
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

if snapshot_error is None:
    wave_options = ["TOTAL PORTFOLIO (Equal-Weighted)"] + snapshot_df["display_name"].tolist()
    selected_view = st.sidebar.selectbox(
        "View",
        wave_options,
        key="overview_view_selector",
    )
else:
    selected_view = None

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
# OVERVIEW (INSTITUTIONAL)
# ===========================
with tabs[0]:
    st.header("Portfolio Overview")

    if snapshot_error:
        st.error(snapshot_error)
    else:
        df = snapshot_df.copy()

        if selected_view == "TOTAL PORTFOLIO (Equal-Weighted)":
            scope_label = "Equal-Weighted Diagnostic Portfolio"
            agg = df.mean(numeric_only=True)
        else:
            scope_label = selected_view
            agg = df[df["display_name"] == selected_view].iloc[0]

        def fmt(x):
            return "—" if pd.isna(x) else f"{x*100:.2f}%"

        # ---- CSS ----
        st.markdown(
            """
<style>
.snapshot-card {
    background: linear-gradient(145deg, #0f172a, #020617);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 18px;
    padding: 28px;
    box-shadow: 0 0 40px rgba(0,255,255,0.12);
}
.metric-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 18px;
    margin-top: 22px;
}
.metric-tile {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 12px;
    padding: 16px;
    text-align: center;
}
.metric-label {
    font-size: 12px;
    color: #9ca3af;
    letter-spacing: 0.08em;
}
.metric-value {
    font-size: 20px;
    font-weight: 600;
    margin-top: 6px;
    color: #e5e7eb;
}
.snapshot-sub {
    color: #94a3b8;
    font-size: 13px;
    margin-top: 6px;
}
</style>
""",
            unsafe_allow_html=True,
        )

        # ---- Snapshot Card ----
        st.markdown(
            f"""
<div class="snapshot-card">
    <h3>🏛 Portfolio Snapshot</h3>
    <div class="snapshot-sub">{scope_label} · Live Data</div>

    <div class="metric-grid">
        <div class="metric-tile"><div class="metric-label">INTRADAY RETURN</div><div class="metric-value">{fmt(agg[RETURN_COLS['intraday']])}</div></div>
        <div class="metric-tile"><div class="metric-label">30D RETURN</div><div class="metric-value">{fmt(agg[RETURN_COLS['30d']])}</div></div>
        <div class="metric-tile"><div class="metric-label">60D RETURN</div><div class="metric-value">{fmt(agg[RETURN_COLS['60d']])}</div></div>
        <div class="metric-tile"><div class="metric-label">365D RETURN</div><div class="metric-value">{fmt(agg[RETURN_COLS['365d']])}</div></div>

        <div class="metric-tile"><div class="metric-label">INTRADAY ALPHA</div><div class="metric-value">{fmt(agg[ALPHA_COLS['intraday']])}</div></div>
        <div class="metric-tile"><div class="metric-label">30D ALPHA</div><div class="metric-value">{fmt(agg[ALPHA_COLS['30d']])}</div></div>
        <div class="metric-tile"><div class="metric-label">60D ALPHA</div><div class="metric-value">{fmt(agg[ALPHA_COLS['60d']])}</div></div>
        <div class="metric-tile"><div class="metric-label">365D ALPHA</div><div class="metric-value">{fmt(agg[ALPHA_COLS['365d']])}</div></div>
    </div>
</div>
""",
            unsafe_allow_html=True,
        )

        st.divider()

        # ---- Vertical Audit View ----
        st.subheader("Diagnostic Metrics")
        for label, col in RETURN_COLS.items():
            st.metric(f"{label.upper()} RETURN", fmt(agg[col]))
        for label, col in ALPHA_COLS.items():
            st.metric(f"{label.upper()} ALPHA", fmt(agg[col]))

# ===========================
# ALPHA ATTRIBUTION (UNCHANGED)
# ===========================
with tabs[1]:
    st.header("Alpha Attribution")
    st.caption("Explains where performance came from")

    if snapshot_error:
        st.error(snapshot_error)
    else:
        waves = snapshot_df["display_name"].tolist()
        selected_wave = st.selectbox("Select Wave", waves, key="alpha_attr_wave_select")

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
                    snapshot_df.loc[snapshot_df["display_name"] == selected_wave, c].values[0]
                    if c in snapshot_df.columns else None
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
# ADAPTIVE INTELLIGENCE (UNCHANGED)
# ===========================
with tabs[2]:
    st.header("Adaptive Intelligence")
    st.caption("Read-only interpretive layer derived from Alpha Attribution")

    if snapshot_error:
        st.error(snapshot_error)
    else:
        waves = snapshot_df["display_name"].tolist()
        selected_wave = st.selectbox("Select Wave", waves, key="adaptive_intel_wave_select")

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