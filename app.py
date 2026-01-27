# app_min.py
# WAVES Intelligence™ Console — Minimal IC-Grade Foundation
# Authoritative rewrite — Overview fixed structurally

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

# ===========================
# Page Config
# ===========================
st.set_page_config(
    page_title="WAVES Intelligence™ Console",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ===========================
# Paths & Columns
# ===========================
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

# ===========================
# CSS — ONE injection, global
# ===========================
st.markdown(
    """
    <style>
    .snapshot-box {
        background: linear-gradient(135deg, #0b1220, #0f172a);
        border: 1px solid rgba(56,189,248,0.35);
        border-radius: 16px;
        padding: 20px 24px;
        margin-bottom: 24px;
        box-shadow: 0 0 24px rgba(56,189,248,0.15);
    }

    .snapshot-title {
        font-size: 1.3rem;
        font-weight: 700;
        margin-bottom: 4px;
    }

    .snapshot-subtitle {
        font-size: 0.85rem;
        color: #94a3b8;
        margin-bottom: 16px;
    }

    .metric-row {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 12px;
        margin-bottom: 12px;
    }

    .metric-cell {
        background: rgba(255,255,255,0.03);
        border-radius: 10px;
        padding: 12px;
        text-align: center;
    }

    .metric-label {
        font-size: 0.75rem;
        color: #94a3b8;
        margin-bottom: 4px;
        letter-spacing: 0.04em;
    }

    .metric-value {
        font-size: 1.2rem;
        font-weight: 700;
    }

    .positive { color: #22c55e; }
    .negative { color: #ef4444; }
    .neutral  { color: #e5e7eb; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ===========================
# Data Loader
# ===========================
def load_snapshot():
    if not LIVE_SNAPSHOT_PATH.exists():
        return None, "Live snapshot not found"

    df = pd.read_csv(LIVE_SNAPSHOT_PATH)
    df.columns = [c.strip().lower() for c in df.columns]

    if "display_name" not in df.columns:
        if "wave_name" in df.columns:
            df["display_name"] = df["wave_name"]
        elif "wave_id" in df.columns:
            df["display_name"] = df["wave_id"]
        else:
            df["display_name"] = "Unknown Wave"

    # Ensure required columns exist
    for col in list(RETURN_COLS.values()) + list(ALPHA_COLS.values()):
        if col not in df.columns:
            df[col] = np.nan

    return df, None


snapshot_df, snapshot_error = load_snapshot()

# ===========================
# Sidebar
# ===========================
st.sidebar.title("Console Controls")

if snapshot_error:
    st.sidebar.error(snapshot_error)
    st.stop()

waves = snapshot_df["display_name"].tolist()

selected_wave = st.sidebar.selectbox(
    "Selected Wave",
    waves,
    index=0,
)

st.sidebar.divider()
st.sidebar.caption("Equal-weighted diagnostics · Read-only")

# ===========================
# Tabs
# ===========================
tabs = st.tabs([
    "Overview",
    "Alpha Attribution",
    "Adaptive Intelligence",
    "Operations",
])

# ===========================
# Helper Renderers
# ===========================
def render_metric_row(row, cols):
    html = '<div class="metric-row">'
    for label, col in cols.items():
        val = row[col]
        if pd.isna(val):
            display = "—"
            cls = "neutral"
        else:
            display = f"{val*100:.2f}%"
            cls = "positive" if val >= 0 else "negative"

        html += f"""
        <div class="metric-cell">
            <div class="metric-label">{label}</div>
            <div class="metric-value {cls}">{display}</div>
        </div>
        """
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)

# ===========================
# OVERVIEW TAB — FIXED
# ===========================
with tabs[0]:
    st.header("Portfolio Overview")

    # ---------- Portfolio Snapshot ----------
    portfolio = snapshot_df.copy()

    portfolio_row = {}
    for col in RETURN_COLS.values():
        portfolio_row[col] = portfolio[col].mean(skipna=True)
    for col in ALPHA_COLS.values():
        portfolio_row[col] = portfolio[col].mean(skipna=True)

    st.markdown('<div class="snapshot-box">', unsafe_allow_html=True)
    st.markdown('<div class="snapshot-title">🏛 Portfolio Snapshot</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="snapshot-subtitle">Equal-weighted diagnostic portfolio · Live data</div>',
        unsafe_allow_html=True,
    )

    render_metric_row(portfolio_row, RETURN_COLS)
    render_metric_row(portfolio_row, ALPHA_COLS)

    st.markdown("</div>", unsafe_allow_html=True)

    # ---------- Wave Snapshot ----------
    wave_row = snapshot_df[snapshot_df["display_name"] == selected_wave].iloc[0]

    st.markdown('<div class="snapshot-box">', unsafe_allow_html=True)
    st.markdown(
        f'<div class="snapshot-title">🏛 {selected_wave}</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="snapshot-subtitle">Wave-level diagnostic snapshot</div>',
        unsafe_allow_html=True,
    )

    render_metric_row(wave_row, RETURN_COLS)
    render_metric_row(wave_row, ALPHA_COLS)

    st.markdown("</div>", unsafe_allow_html=True)

# ===========================
# OTHER TABS — PLACEHOLDERS
# ===========================
with tabs[1]:
    st.header("Alpha Attribution")
    st.info("Alpha attribution layer remains unchanged.")

with tabs[2]:
    st.header("Adaptive Intelligence")
    st.info("Interpretive layer — execution logic comes next.")

with tabs[3]:
    st.header("Operations")
    st.info("Human override & execution controls will live here.")