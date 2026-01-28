# ============================================================
# app_min.py
# WAVES Intelligence™ Console (Minimal)
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

from intelligence.adaptive_intelligence import (
    render_alpha_attribution_drivers,
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

RETURN_COLS = {
    "INTRADAY": "return_intraday",
    "1D": "return_intraday",
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
    "1D": "alpha_intraday",
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

    for col in list(RETURN_COLS.values()) + list(ALPHA_COLS.values()):
        if col not in df.columns:
            df[col] = np.nan

    if "intraday_label" not in df.columns:
        df["intraday_label"] = None

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
            if v in df.columns
        }

        portfolio_alpha = {
            k: df[v].mean(skipna=True)
            for k, v in ALPHA_COLS.items()
            if v in df.columns
        }

        with st.container():
            st.markdown("### 🏛️ Portfolio Snapshot — Equal-Weighted")
            st.caption("Equal-weighted performance across all active waves.")
            st.divider()

            st.markdown("**Returns**")
            ret_cols = st.columns(len(portfolio_returns))
            for i, (label, value) in enumerate(portfolio_returns.items()):
                ret_cols[i].markdown(
                    f"<div style='line-height:1.4'>"
                    f"<span style='font-size:0.85rem; color:#666;'>{label}</span><br>"
                    f"<span style='font-size:1.25rem; font-weight:700;'>{format_percentage(value)}</span>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

            st.markdown("**Alpha**")
            alpha_cols = st.columns(len(portfolio_alpha))
            for i, (label, value) in enumerate(portfolio_alpha.items()):
                if label == "365D":
                    if (
                        "alpha_momentum_365d" not in df.columns
                        or "alpha_volatility_365d" not in df.columns
                    ):
                        display_value = "—"
                    else:
                        mom_series = df["alpha_momentum_365d"]
                        vol_series = df["alpha_volatility_365d"]
                        overlay_series = mom_series + vol_series
                        valid_overlay = overlay_series.dropna()

                        if valid_overlay.empty:
                            display_value = "—"
                        else:
                            portfolio_overlay_alpha = valid_overlay.mean() * 100
                            display_value = f"{portfolio_overlay_alpha:.1f}%"

                    label_text = "Overlay Alpha (Momentum + Volatility)"
                    alpha_cols[i].markdown(
                        f"<div style='line-height:1.4'>"
                        f"<span style='font-size:0.85rem; color:#666;'>{label_text}</span><br>"
                        f"<span style='font-size:1.25rem; font-weight:700;'>{display_value}</span>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )
                else:
                    alpha_cols[i].markdown(
                        f"<div style='line-height:1.4'>"
                        f"<span style='font-size:0.85rem; color:#666;'>{label}</span><br>"
                        f"<span style='font-size:1.25rem; font-weight:700;'>{format_percentage(value)}</span>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )

# ============================================================
# ALPHA ATTRIBUTION TAB (WITH HORIZON DROPDOWN)
# ============================================================
with tabs[1]:
    st.header("Alpha Attribution")
    st.caption("Governance-native attribution with horizon control.")
    st.divider()

    if snapshot_error or snapshot_df is None:
        st.info("Attribution requires a valid live snapshot.")
    else:
        horizon = st.selectbox(
            "Attribution Horizon",
            ["30D", "INTRADAY", "1D", "60D", "365D"],
            index=0,
            key="alpha_attribution_horizon_select",
        )

        st.session_state["alpha_attribution_horizon"] = horizon

        st.divider()

        render_alpha_attribution_drivers(
            snapshot_df,
            selected_wave,
            RETURN_COLS,
            BENCHMARK_COLS,
        )

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
        "Adaptive Intelligence signals activate once attribution history "
        "accumulation thresholds are met."
    )

# ============================================================
# OPERATIONS TAB
# ============================================================
with tabs[3]:
    st.header("Operations")
    st.info("Execution & override layer coming next.")
