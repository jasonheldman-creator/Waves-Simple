# ============================================================
# app_min.py
# WAVES Intelligence™ Console (Minimal)
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

# UPDATED IMPORTS — match current adaptive_intelligence.py
from intelligence.adaptive_intelligence import (
    render_alpha_attribution_intraday,
)

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
ATTRIBUTION_INTRADAY_PATH = DATA_DIR / "live_snapshot_attribution.csv"

RETURN_COLS = {
    "INTRADAY": "return_intraday",
    "30D": "return_30d",
    "60D": "return_60d",
    "365D": "return_365d",
}

BENCHMARK_COLS = {
    "30D": "benchmark_return_30d",
    "60D": "benchmark_return_60d",
    "365D": "benchmark_return_365d",
}

ALPHA_COLS = {
    "INTRADAY": "alpha_intraday",
    "30D": "alpha_30d",
    "60D": "alpha_60d",
    "365D": "alpha_365d",
}

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

    for col in (
        list(RETURN_COLS.values())
        + list(BENCHMARK_COLS.values())
        + list(ALPHA_COLS.values())
    ):
        if col not in df.columns:
            df[col] = np.nan

    if "intraday_label" not in df.columns:
        df["intraday_label"] = None

    return df, None


def load_intraday_attribution():
    if not ATTRIBUTION_INTRADAY_PATH.exists():
        return None
    df = pd.read_csv(ATTRIBUTION_INTRADAY_PATH)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


snapshot_df, snapshot_error = load_snapshot()
attribution_intraday_df = load_intraday_attribution()

# ===========================
# Sidebar
# ===========================
st.sidebar.title("System Status")

st.sidebar.markdown(
    f"""
**Live Snapshot:** {'✅ Loaded' if snapshot_error is None else '❌ Missing'}  
**Alpha Attribution:** {'✅ Active' if attribution_intraday_df is not None else '🟡 Pending'}  
**Adaptive Intelligence:** 🟡 Interpretive  
"""
)

st.sidebar.divider()

if snapshot_df is not None:
    selected_wave = st.sidebar.selectbox(
        "Select Wave",
        snapshot_df["display_name"].tolist(),
        key="global_wave_select",
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
    "Operations",
])

# ============================================================
# OVERVIEW TAB
# ============================================================
with tabs[0]:
    st.header("Portfolio Overview")

    if snapshot_error:
        st.error(snapshot_error)
    else:
        df = snapshot_df.copy()

        def format_percentage(value: float) -> str:
            if pd.isna(value):
                return "—"
            if abs(value) < 1e-10:
                return "0.00%"
            pct = value * 100
            return f"+{pct:.2f}%" if pct > 0 else f"{pct:.2f}%"

        portfolio_returns = {
            k: df[v].mean(skipna=True)
            for k, v in RETURN_COLS.items()
        }

        portfolio_alpha = {
            k: df[v].mean(skipna=True)
            for k, v in ALPHA_COLS.items()
        }

        intraday_labels = df["intraday_label"].dropna().unique().tolist()
        portfolio_intraday_label = intraday_labels[0] if len(intraday_labels) == 1 else None

        with st.container():
            st.markdown("### 🏛️ Portfolio Snapshot — Equal-Weighted")
            if portfolio_intraday_label:
                st.caption(
                    f"Equal-weighted performance across all active waves · {portfolio_intraday_label}"
                )
            else:
                st.caption("Equal-weighted performance across all active waves.")
            st.divider()

            st.markdown("**Returns**")
            ret_cols = st.columns(4)
            for i, (label, value) in enumerate(portfolio_returns.items()):
                ret_cols[i].markdown(
                    f"<div style='line-height:1.4'>"
                    f"<span style='font-size:0.85rem; font-weight:500; color:#666;'>{label} Return</span><br>"
                    f"<span style='font-size:1.25rem; font-weight:700;'>{format_percentage(value)}</span>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

            st.markdown("**Alpha**")
            alpha_cols = st.columns(4)
            for i, (label, value) in enumerate(portfolio_alpha.items()):
                alpha_cols[i].markdown(
                    f"<div style='line-height:1.4'>"
                    f"<span style='font-size:0.85rem; font-weight:500; color:#666;'>{label} Alpha</span><br>"
                    f"<span style='font-size:1.25rem; font-weight:700;'>{format_percentage(value)}</span>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

# ============================================================
# ALPHA ATTRIBUTION TAB — GOVERNANCE-NATIVE
# ============================================================
with tabs[1]:
    st.header("Alpha Attribution")
    st.caption("Portfolio-first attribution with wave-level drill-down.")
    st.divider()

    if attribution_intraday_df is None:
        st.info("Intraday attribution stream not yet available.")
    else:
        render_alpha_attribution_intraday(attribution_intraday_df)

# ============================================================
# ADAPTIVE INTELLIGENCE TAB
# ============================================================
with tabs[2]:
    st.header("Adaptive Intelligence")
    st.caption(
        "Interpretive layer derived from Alpha Attribution — read-only, governance-first."
    )
    st.divider()

    st.info(
        "Adaptive Intelligence signals (confidence, regime awareness, signal decay) "
        "will activate once attribution history accumulation thresholds are met."
    )

# ============================================================
# OPERATIONS TAB
# ============================================================
with tabs[3]:
    st.header("Operations")
    st.info("Execution & override layer coming next.")