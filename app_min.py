# app_min.py
# WAVES Intelligence™ Console — Minimal (B2)
# Institutional Overview + Alpha Attribution + Adaptive Intelligence (Preview)

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
# Global CSS (INSTITUTIONAL)
# ---------------------------
st.markdown(
    """
    <style>
    .snapshot-card {
        background: linear-gradient(135deg, #0b1220 0%, #0f1c33 100%);
        border-radius: 18px;
        padding: 28px 32px;
        box-shadow: 0 0 0 1px rgba(255,255,255,0.04),
                    0 0 40px rgba(0, 200, 255, 0.15);
        margin-bottom: 32px;
    }
    .snapshot-title {
        font-size: 28px;
        font-weight: 700;
        letter-spacing: 0.3px;
        margin-bottom: 4px;
    }
    .snapshot-subtitle {
        color: #9fb3c8;
        font-size: 14px;
        margin-bottom: 22px;
    }
    .metric-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
        gap: 16px;
    }
    .metric-tile {
        background: rgba(255,255,255,0.04);
        border-radius: 14px;
        padding: 14px 16px;
        border: 1px solid rgba(255,255,255,0.06);
    }
    .metric-label {
        font-size: 12px;
        color: #8aa0b8;
        margin-bottom: 6px;
    }
    .metric-value {
        font-size: 22px;
        font-weight: 600;
        color: #ffffff;
    }
    .metric-positive { color: #3dffb2; }
    .metric-negative { color: #ff6b6b; }
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
    "Alpha 1D": "alpha_1d",
    "Alpha 30D": "alpha_30d",
    "Alpha 60D": "alpha_60d",
    "Alpha 365D": "alpha_365d",
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

    return df, None


snapshot_df, snapshot_error = load_snapshot()

# ---------------------------
# Sidebar
# ---------------------------
st.sidebar.title("System Status")
st.sidebar.markdown(
    """
    **Alpha Attribution:** ✅ Active  
    **Adaptive Intelligence:** 🟡 Preview  
    **Execution Layer:** 🔒 Manual Override  
    """
)
st.sidebar.divider()

# Sidebar Wave Selector (B)
selected_wave_sidebar = None
if snapshot_df is not None:
    selected_wave_sidebar = st.sidebar.selectbox(
        "Wave Drill-Down",
        snapshot_df["display_name"].tolist(),
    )

# ---------------------------
# Tabs
# ---------------------------
tabs = st.tabs(
    ["Overview", "Alpha Attribution", "Adaptive Intelligence", "Operations"]
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

        # Equal-weighted aggregation
        portfolio_metrics = {}
        for label, col in {**RETURN_COLS, **ALPHA_COLS}.items():
            if col in df.columns:
                portfolio_metrics[label] = df[col].mean(skipna=True)
            else:
                portfolio_metrics[label] = np.nan

        def render_snapshot(title, subtitle, metrics):
            st.markdown(
                f"""
                <div class="snapshot-card">
                    <div class="snapshot-title">🏛️ {title}</div>
                    <div class="snapshot-subtitle">{subtitle}</div>
                    <div class="metric-grid">
                """,
                unsafe_allow_html=True,
            )

            for label, value in metrics.items():
                if pd.isna(value):
                    display = "—"
                    cls = ""
                else:
                    display = f"{value*100:.2f}%"
                    cls = "metric-positive" if value >= 0 else "metric-negative"

                st.markdown(
                    f"""
                    <div class="metric-tile">
                        <div class="metric-label">{label}</div>
                        <div class="metric-value {cls}">{display}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            st.markdown("</div></div>", unsafe_allow_html=True)

        # Portfolio Snapshot
        render_snapshot(
            title="Portfolio Snapshot",
            subtitle="Equal-Weighted Diagnostic Portfolio · Live Data",
            metrics=portfolio_metrics,
        )

        # Wave Drill-Down Snapshot
        if selected_wave_sidebar:
            wave_row = df[df["display_name"] == selected_wave_sidebar].iloc[0]
            wave_metrics = {}
            for label, col in {**RETURN_COLS, **ALPHA_COLS}.items():
                wave_metrics[label] = wave_row[col] if col in wave_row else np.nan

            render_snapshot(
                title=f"{selected_wave_sidebar}",
                subtitle="Wave-Level Diagnostic Snapshot",
                metrics=wave_metrics,
            )

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
        selected_wave = st.selectbox("Select Wave", waves, key="alpha_attr_wave")

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
                    if c in snapshot_df.columns
                    else None
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
            snapshot_df, None, selected_wave, RETURN_COLS, {}
        )

# ===========================
# ADAPTIVE INTELLIGENCE
# ===========================
with tabs[2]:
    st.header("Adaptive Intelligence")
    st.caption("Read-only interpretive layer (no execution)")

    if snapshot_error:
        st.error(snapshot_error)
    else:
        waves = snapshot_df["display_name"].tolist()
        selected_wave = st.selectbox("Select Wave", waves, key="adaptive_wave")

        render_adaptive_intelligence_preview(
            snapshot_df, None, selected_wave, RETURN_COLS, {}
        )

# ===========================
# OPERATIONS
# ===========================
with tabs[3]:
    st.header("Operations")
    st.info("Execution & override controls will live here.")