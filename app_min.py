# app_min.py
# WAVES Intelligence™ Console — Minimal (B2)
# Institutional Overview + Alpha Attribution (Stable)

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

from intelligence.adaptive_intelligence import (
    render_alpha_quality_and_confidence,
    render_adaptive_intelligence_preview,
)

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="WAVES Intelligence™ Console",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================================================
# GLOBAL CSS — INSTITUTIONAL THEME
# =========================================================
st.markdown(
    """
    <style>
    .glow-box {
        background: linear-gradient(135deg, #0b1220, #0e1628);
        border-radius: 16px;
        padding: 22px;
        border: 1px solid rgba(0, 255, 255, 0.25);
        box-shadow: 0 0 24px rgba(0, 255, 255, 0.18);
        margin-bottom: 22px;
    }

    .section-title {
        font-size: 26px;
        font-weight: 700;
        margin-bottom: 6px;
    }

    .section-subtitle {
        font-size: 14px;
        opacity: 0.75;
        margin-bottom: 18px;
    }

    .metric-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 14px;
        margin-top: 10px;
    }

    .metric-cell {
        background: rgba(255,255,255,0.04);
        border-radius: 12px;
        padding: 14px;
        border: 1px solid rgba(255,255,255,0.08);
        text-align: center;
    }

    .metric-label {
        font-size: 12px;
        letter-spacing: 1px;
        text-transform: uppercase;
        opacity: 0.7;
        margin-bottom: 6px;
    }

    .metric-value {
        font-size: 20px;
        font-weight: 700;
    }

    .positive { color: #3CFFB1; }
    .negative { color: #FF5C5C; }
    .neutral  { color: #BBBBBB; }

    .divider {
        height: 1px;
        background: linear-gradient(to right, transparent, rgba(255,255,255,0.2), transparent);
        margin: 18px 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================================================
# CONSTANTS
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
# LOAD SNAPSHOT
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
            df["display_name"] = df.index.astype(str)

    return df, None


snapshot_df, snapshot_error = load_snapshot()

# =========================================================
# HELPERS
# =========================================================
def color_class(val):
    if pd.isna(val):
        return "neutral"
    if val > 0:
        return "positive"
    if val < 0:
        return "negative"
    return "neutral"


def render_horizontal_metrics(title, subtitle, returns, alphas):
    html = f"""
    <div class="glow-box">
        <div class="section-title">🏛️ {title}</div>
        <div class="section-subtitle">{subtitle}</div>

        <div class="metric-grid">
    """

    for k, v in returns.items():
        cls = color_class(v)
        display = "—" if pd.isna(v) else f"{v*100:.2f}%"
        html += f"""
        <div class="metric-cell">
            <div class="metric-label">{k}</div>
            <div class="metric-value {cls}">{display}</div>
        </div>
        """

    html += "</div><div class='divider'></div><div class='metric-grid'>"

    for k, v in alphas.items():
        cls = color_class(v)
        display = "—" if pd.isna(v) else f"{v*100:.2f}%"
        html += f"""
        <div class="metric-cell">
            <div class="metric-label">α {k}</div>
            <div class="metric-value {cls}">{display}</div>
        </div>
        """

    html += "</div></div>"
    st.markdown(html, unsafe_allow_html=True)

# =========================================================
# TABS
# =========================================================
tabs = st.tabs(
    ["Overview", "Alpha Attribution", "Adaptive Intelligence", "Operations"]
)

# =========================================================
# OVERVIEW TAB — REBUILT
# =========================================================
with tabs[0]:
    st.header("Portfolio Overview")

    if snapshot_error:
        st.error(snapshot_error)
    else:
        df = snapshot_df.copy()

        # Equal-weight portfolio diagnostic
        portfolio_returns = {
            k: df[v].mean(skipna=True) if v in df.columns else np.nan
            for k, v in RETURN_COLS.items()
        }

        portfolio_alpha = {
            k: df.get(ALPHA_COLS[k], pd.Series(dtype=float)).mean(skipna=True)
            for k in ALPHA_COLS
        }

        render_horizontal_metrics(
            "Portfolio Snapshot",
            "Equal-Weighted Diagnostic Portfolio · Live Data",
            portfolio_returns,
            portfolio_alpha,
        )

        # Wave-level snapshots
        for _, row in df.iterrows():
            wave_returns = {
                k: row.get(v, np.nan) for k, v in RETURN_COLS.items()
            }
            wave_alpha = {
                k: row.get(ALPHA_COLS[k], np.nan) for k in ALPHA_COLS
            }

            render_horizontal_metrics(
                row["display_name"],
                "Wave-Level Diagnostic Snapshot",
                wave_returns,
                wave_alpha,
            )

# =========================================================
# ALPHA ATTRIBUTION (UNCHANGED)
# =========================================================
with tabs[1]:
    st.header("Alpha Attribution")
    st.caption("Explains where performance came from")

    if snapshot_error:
        st.error(snapshot_error)
    else:
        waves = snapshot_df["display_name"].tolist()
        selected_wave = st.selectbox("Select Wave", waves)

        render_alpha_quality_and_confidence(
            snapshot_df,
            None,
            selected_wave,
            RETURN_COLS,
            {},
        )

# =========================================================
# ADAPTIVE INTELLIGENCE (UNCHANGED)
# =========================================================
with tabs[2]:
    st.header("Adaptive Intelligence")
    st.caption("Read-only interpretive layer derived from Alpha Attribution")

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

# =========================================================
# OPERATIONS
# =========================================================
with tabs[3]:
    st.header("Operations")
    st.info("Execution & override controls will live here.")