# app_min.py
# WAVES Intelligence™ Console — Minimal (B2)
# Overview + Alpha Attribution + Adaptive Intelligence (Preview)

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

from intelligence.adaptive_intelligence import (
    render_alpha_quality_and_confidence,
    render_adaptive_intelligence_preview,
)

# ===========================
# PAGE CONFIG
# ===========================
st.set_page_config(
    page_title="WAVES Intelligence™ Console",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ===========================
# GLOBAL STYLES (SAFE)
# ===========================
st.markdown("""
<style>
.snapshot-card {
    background: radial-gradient(circle at top left, #0f1b2d, #070c16);
    border-radius: 18px;
    padding: 28px;
    border: 1px solid rgba(255,255,255,0.08);
    box-shadow: 0 0 40px rgba(0,255,255,0.08);
    margin-bottom: 36px;
}

.snapshot-title {
    font-size: 26px;
    font-weight: 700;
    margin-bottom: 6px;
}

.snapshot-sub {
    font-size: 14px;
    opacity: 0.75;
    margin-bottom: 22px;
}

.metric-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 16px;
}

.metric-tile {
    background: rgba(255,255,255,0.05);
    border-radius: 12px;
    padding: 16px;
    border: 1px solid rgba(255,255,255,0.08);
    text-align: center;
}

.metric-label {
    font-size: 12px;
    opacity: 0.65;
    margin-bottom: 6px;
}

.metric-value {
    font-size: 22px;
    font-weight: 700;
}
</style>
""", unsafe_allow_html=True)

# ===========================
# CONSTANTS
# ===========================
DATA_DIR = Path("data")
LIVE_SNAPSHOT_PATH = DATA_DIR / "live_snapshot.csv"

RETURN_COLS = {
    "1d": "return_1d",
    "30d": "return_30d",
    "60d": "return_60d",
    "365d": "return_365d",
}

ALPHA_COLS = {
    "1d": "alpha_1d",
    "30d": "alpha_30d",
    "60d": "alpha_60d",
    "365d": "alpha_365d",
}

# ===========================
# LOAD SNAPSHOT
# ===========================
def load_snapshot():
    if not LIVE_SNAPSHOT_PATH.exists():
        return None, "Live snapshot file not found"

    df = pd.read_csv(LIVE_SNAPSHOT_PATH)
    df.columns = [c.strip().lower() for c in df.columns]

    if "display_name" not in df.columns:
        df["display_name"] = df.get("wave_name", df.get("wave_id", "Unnamed Wave"))

    return df, None


snapshot_df, snapshot_error = load_snapshot()

# ===========================
# SIDEBAR
# ===========================
st.sidebar.title("System Status")
st.sidebar.markdown(
    f"""
    **Live Snapshot:** {'✅ Loaded' if snapshot_error is None else '❌ Missing'}  
    **Alpha Attribution:** ✅ Active  
    **Adaptive Intelligence:** 🟡 Preview Mode  
    """
)
st.sidebar.divider()

# ===========================
# TABS
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
# OVERVIEW TAB
# ===========================
with tabs[0]:
    st.header("Portfolio Overview")

    if snapshot_error:
        st.error(snapshot_error)
    else:
        df = snapshot_df.copy()

        # Equal-weighted diagnostics
        ew = {}
        for k, col in RETURN_COLS.items():
            ew[col] = df[col].mean(skipna=True) if col in df.columns else np.nan

        for k, col in ALPHA_COLS.items():
            ew[col] = df[col].mean(skipna=True) if col in df.columns else np.nan

        # -------- Snapshot Card (HTML ONLY) --------
        st.markdown(f"""
        <div class="snapshot-card">
            <div class="snapshot-title">🏛 Portfolio Snapshot</div>
            <div class="snapshot-sub">
                Equal-Weighted Diagnostic Portfolio · Live Data
            </div>

            <div class="metric-grid">
                <div class="metric-tile">
                    <div class="metric-label">1D Return</div>
                    <div class="metric-value">{ew['return_1d']:.2%}</div>
                </div>
                <div class="metric-tile">
                    <div class="metric-label">30D Return</div>
                    <div class="metric-value">{ew['return_30d']:.2%}</div>
                </div>
                <div class="metric-tile">
                    <div class="metric-label">60D Return</div>
                    <div class="metric-value">{ew['return_60d']:.2%}</div>
                </div>
                <div class="metric-tile">
                    <div class="metric-label">365D Return</div>
                    <div class="metric-value">{ew['return_365d']:.2%}</div>
                </div>

                <div class="metric-tile">
                    <div class="metric-label">Alpha 1D</div>
                    <div class="metric-value">{ew.get('alpha_1d', np.nan):.2%}</div>
                </div>
                <div class="metric-tile">
                    <div class="metric-label">Alpha 30D</div>
                    <div class="metric-value">{ew.get('alpha_30d', np.nan):.2%}</div>
                </div>
                <div class="metric-tile">
                    <div class="metric-label">Alpha 60D</div>
                    <div class="metric-value">{ew.get('alpha_60d', np.nan):.2%}</div>
                </div>
                <div class="metric-tile">
                    <div class="metric-label">Alpha 365D</div>
                    <div class="metric-value">{ew.get('alpha_365d', np.nan):.2%}</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # -------- Wave Table --------
        table_cols = ["display_name"] + list(RETURN_COLS.values())
        view = df[table_cols].rename(columns={
            "display_name": "Wave",
            "return_1d": "1D",
            "return_30d": "30D",
            "return_60d": "60D",
            "return_365d": "365D",
        })

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

        source_df = pd.DataFrame({
            "Alpha Source": [
                "Selection Alpha",
                "Momentum Alpha",
                "Regime Alpha",
                "Exposure Alpha",
                "Residual Alpha",
            ],
            "Contribution": [
                snapshot_df.loc[snapshot_df["display_name"] == selected_wave].get("selection_alpha", pd.Series([np.nan])).values[0],
                snapshot_df.loc[snapshot_df["display_name"] == selected_wave].get("momentum_alpha", pd.Series([np.nan])).values[0],
                snapshot_df.loc[snapshot_df["display_name"] == selected_wave].get("regime_alpha", pd.Series([np.nan])).values[0],
                snapshot_df.loc[snapshot_df["display_name"] == selected_wave].get("exposure_alpha", pd.Series([np.nan])).values[0],
                snapshot_df.loc[snapshot_df["display_name"] == selected_wave].get("residual_alpha", pd.Series([np.nan])).values[0],
            ]
        })

        st.subheader("Source Breakdown")
        st.dataframe(source_df, use_container_width=True, hide_index=True)

        render_alpha_quality_and_confidence(
            snapshot_df,
            None,
            selected_wave,
            RETURN_COLS,
            {},
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
            {},
        )

# ===========================
# OPERATIONS
# ===========================
with tabs[3]:
    st.header("Operations")
    st.info("Execution & override controls will live here.")