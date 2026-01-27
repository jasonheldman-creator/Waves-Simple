# app_min.py
# WAVES Intelligence™ Console (Minimal)
# LOCKED OVERVIEW LAYOUT — Streamlit-Native, Mobile-Safe

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
            df["display_name"] = df.get("wave_id", "Unknown Wave")

    # Ensure required columns exist
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
    st.sidebar.error("Live data unavailable")
else:
    wave_options = snapshot_df["display_name"].tolist()
    selected_wave = st.sidebar.selectbox(
        "Select Wave",
        wave_options,
        key="overview_wave_select"
    )

st.sidebar.divider()

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
# OVERVIEW TAB (LOCKED)
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
            k: df[v].mean(skipna=True)
            for k, v in RETURN_COLS.items()
        }
        portfolio_alpha = {
            k: df[v].mean(skipna=True)
            for k, v in ALPHA_COLS.items()
        }

        with st.container():
            st.subheader("🏛️ Portfolio Snapshot — Equal-Weighted")

            # RETURNS ROW
            cols = st.columns(4)
            for i, (label, value) in enumerate(portfolio_returns.items()):
                cols[i].metric(
                    label=f"{label} Return",
                    value="—" if pd.isna(value) else f"{value:.2%}"
                )

            # ALPHA ROW
            cols = st.columns(4)
            for i, (label, value) in enumerate(portfolio_alpha.items()):
                cols[i].metric(
                    label=f"{label} Alpha",
                    value="—" if pd.isna(value) else f"{value:.2%}"
                )

        st.divider()

        # ---------------------------
        # Selected Wave Snapshot
        # ---------------------------
        wave_row = df[df["display_name"] == selected_wave].iloc[0]

        wave_returns = {
            k: wave_row[v]
            for k, v in RETURN_COLS.items()
        }
        wave_alpha = {
            k: wave_row[v]
            for k, v in ALPHA_COLS.items()
        }

        with st.container():
            st.subheader(f"📊 Wave Snapshot — {selected_wave}")

            # RETURNS ROW
            cols = st.columns(4)
            for i, (label, value) in enumerate(wave_returns.items()):
                cols[i].metric(
                    label=f"{label} Return",
                    value="—" if pd.isna(value) else f"{value:.2%}"
                )

            # ALPHA ROW
            cols = st.columns(4)
            for i, (label, value) in enumerate(wave_alpha.items()):
                cols[i].metric(
                    label=f"{label} Alpha",
                    value="—" if pd.isna(value) else f"{value:.2%}"
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
            None,
        )

# ===========================
# ADAPTIVE INTELLIGENCE TAB
# ===========================
with tabs[2]:
    st.header("Adaptive Intelligence")
    st.caption("Read-only intelligence layer derived from attribution")

    if snapshot_error:
        st.error(snapshot_error)
    else:
        render_alpha_quality_and_confidence(
            snapshot_df,
            None,
            selected_wave,
            RETURN_COLS,
            None,
        )

# ===========================
# OPERATIONS TAB
# ===========================
with tabs[3]:
    st.header("Operations")
    st.info("Operations control center coming next.")
    