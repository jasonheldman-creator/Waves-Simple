# app_min.py
# WAVES Intelligence™ Console (Minimal, CLEAN OVERVIEW)

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

ALPHA_COLS = {
    "1D": "alpha_1d",
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
        else:
            df["display_name"] = df["wave_id"]

    for col in list(RETURN_COLS.values()) + list(ALPHA_COLS.values()):
        if col not in df.columns:
            df[col] = np.nan

    return df, None

snapshot_df, snapshot_error = load_snapshot()

# ===========================
# Sidebar
# ===========================
st.sidebar.title("Controls")

if snapshot_error:
    st.sidebar.error(snapshot_error)
    selected_wave = None
else:
    wave_options = snapshot_df["display_name"].tolist()
    selected_wave = st.sidebar.selectbox(
        "Select Wave",
        wave_options,
        key="overview_wave_select"
    )

st.sidebar.divider()
st.sidebar.caption("WAVES Intelligence™")

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
# OVERVIEW TAB
# ===========================
with tabs[0]:
    st.header("Portfolio Overview")

    if snapshot_error:
        st.error(snapshot_error)
    else:
        df = snapshot_df.copy()

        # ---------------------------
        # Portfolio Snapshot (Equal-Weighted)
        # ---------------------------
        portfolio_returns = {
            k: df[v].mean(skipna=True) for k, v in RETURN_COLS.items()
        }
        portfolio_alpha = {
            k: df[v].mean(skipna=True) for k, v in ALPHA_COLS.items()
        }

        with st.container(border=True):
            st.subheader("🏛️ Portfolio Snapshot — Equal-Weighted")

            ret_cols = st.columns(4)
            for i, horizon in enumerate(RETURN_COLS.keys()):
                ret_cols[i].metric(
                    label=f"{horizon} Return",
                    value=f"{portfolio_returns[horizon]:.2%}"
                )

            st.divider()

            alpha_cols = st.columns(4)
            for i, horizon in enumerate(ALPHA_COLS.keys()):
                val = portfolio_alpha[horizon]
                alpha_cols[i].metric(
                    label=f"{horizon} Alpha",
                    value="—" if np.isnan(val) else f"{val:.2%}"
                )

        st.markdown("")

        # ---------------------------
        # Wave Snapshot (Selected)
        # ---------------------------
        wave_row = df[df["display_name"] == selected_wave].iloc[0]

        with st.container(border=True):
            st.subheader(f"📈 Wave Snapshot — {selected_wave}")

            ret_cols = st.columns(4)
            for i, (horizon, col) in enumerate(RETURN_COLS.items()):
                val = wave_row[col]
                ret_cols[i].metric(
                    label=f"{horizon} Return",
                    value="—" if np.isnan(val) else f"{val:.2%}"
                )

            st.divider()

            alpha_cols = st.columns(4)
            for i, (horizon, col) in enumerate(ALPHA_COLS.items()):
                val = wave_row[col]
                alpha_cols[i].metric(
                    label=f"{horizon} Alpha",
                    value="—" if np.isnan(val) else f"{val:.2%}"
                )

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
            {},
        )

# ===========================
# ADAPTIVE INTELLIGENCE TAB
# ===========================
with tabs[2]:
    st.header("Adaptive Intelligence")
    st.caption("Read-only interpretive layer")

    if snapshot_error:
        st.error(snapshot_error)
    else:
        render_alpha_quality_and_confidence(
            snapshot_df,
            None,
            selected_wave,
            RETURN_COLS,
            {},
        )

# ===========================
# OPERATIONS TAB
# ===========================
with tabs[3]:
    st.header("Operations")
    st.info("Execution & override controls coming next.")