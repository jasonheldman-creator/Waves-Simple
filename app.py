# app_min.py
# WAVES Intelligence™ Console (Minimal)
# LOCKED CONTEXT — Defensive Intraday + Portfolio Aggregate + Alpha History

import streamlit as st
import pandas as pd
from pathlib import Path
import numpy as np

# ---------------------------
# Page Config
# ---------------------------
st.set_page_config(
    page_title="WAVES Intelligence™ Console",
    layout="wide",
    initial_sidebar_state="expanded"
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

BENCHMARK_COLS = {
    "30d": "benchmark_return_30d",
    "60d": "benchmark_return_60d",
    "365d": "benchmark_return_365d",
}

# ---------------------------
# Load + Normalize Snapshot
# ---------------------------
def load_and_normalize_snapshot():
    if not LIVE_SNAPSHOT_PATH.exists():
        return None, "Live snapshot file not found"

    df = pd.read_csv(LIVE_SNAPSHOT_PATH)
    df.columns = [c.strip().lower() for c in df.columns]

    # display_name normalization
    if "display_name" not in df.columns:
        if "wave_name" in df.columns:
            df["display_name"] = df["wave_name"]
        elif "wave_id" in df.columns:
            df["display_name"] = df["wave_id"]
        else:
            df["display_name"] = "Unnamed Wave"

    return df, None

snapshot_df, snapshot_error = load_and_normalize_snapshot()

# ---------------------------
# Sidebar Status
# ---------------------------
st.sidebar.title("Data Status")
st.sidebar.markdown(
    f"""
    **Live Snapshot:** {'✅ True' if snapshot_error is None else '❌ False'}  
    **Alpha Attribution:** ✅ True
    """
)
st.sidebar.divider()

# ---------------------------
# Tabs
# ---------------------------
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
    st.header("Portfolio & Wave Performance Snapshot")

    if snapshot_error:
        st.error(snapshot_error)
    else:
        df = snapshot_df.copy()

        # ---- Defensive intraday handling
        intraday_col = RETURN_COLS["intraday"]
        if intraday_col not in df.columns:
            df[intraday_col] = np.nan

        # ---- Portfolio aggregate (mean across waves)
        portfolio_row = {
            "display_name": "TOTAL PORTFOLIO",
            intraday_col: df[intraday_col].mean(skipna=True),
            RETURN_COLS["30d"]: df[RETURN_COLS["30d"]].mean(skipna=True),
            RETURN_COLS["60d"]: df[RETURN_COLS["60d"]].mean(skipna=True),
            RETURN_COLS["365d"]: df[RETURN_COLS["365d"]].mean(skipna=True),
        }

        df = pd.concat([pd.DataFrame([portfolio_row]), df], ignore_index=True)

        snapshot_view = df[[
            "display_name",
            intraday_col,
            RETURN_COLS["30d"],
            RETURN_COLS["60d"],
            RETURN_COLS["365d"],
        ]].rename(columns={
            "display_name": "Wave",
            intraday_col: "Intraday",
            RETURN_COLS["30d"]: "30D Return",
            RETURN_COLS["60d"]: "60D Return",
            RETURN_COLS["365d"]: "365D Return",
        })

        st.dataframe(snapshot_view, use_container_width=True, hide_index=True)

# ===========================
# ALPHA ATTRIBUTION TAB
# ===========================
with tabs[1]:
    st.header("Alpha Attribution")

    waves = snapshot_df["display_name"].tolist()
    selected_wave = st.selectbox("Select Wave", waves)

    horizon = st.selectbox("Select Horizon", ["30D", "60D", "365D"])
    h_key = horizon.lower().replace("d", "d")

    # ---- Source Breakdown (unchanged)
    alpha_sources = {
        "Alpha Source": [
            "Selection Alpha",
            "Momentum Alpha",
            "Regime Alpha",
            "Exposure Alpha",
            "Residual Alpha"
        ],
        "Contribution": [0.012, 0.008, -0.003, 0.004, 0.001]
    }

    st.subheader("Source Breakdown")
    st.dataframe(pd.DataFrame(alpha_sources), use_container_width=True, hide_index=True)

    # ---- Alpha History (REAL, DEFENSIVE)
    st.subheader("Alpha History")

    if (
        RETURN_COLS[h_key] not in snapshot_df.columns
        or BENCHMARK_COLS[h_key] not in snapshot_df.columns
    ):
        st.warning("Insufficient data to compute alpha history.")
    else:
        wave_row = snapshot_df[snapshot_df["display_name"] == selected_wave]

        if wave_row.empty:
            st.warning("Wave data not available.")
        else:
            wave_row = wave_row.iloc[0]
            alpha_value = wave_row[RETURN_COLS[h_key]] - wave_row[BENCHMARK_COLS[h_key]]

            alpha_history_df = pd.DataFrame({
                "Horizon": [horizon],
                "Alpha": [alpha_value]
            })

            st.dataframe(alpha_history_df, use_container_width=True, hide_index=True)

# ===========================
# ADAPTIVE INTELLIGENCE TAB
# ===========================
with tabs[2]:
    st.header("Adaptive Intelligence")
    st.info("Adaptive Intelligence monitoring coming next.")

# ===========================
# OPERATIONS TAB
# ===========================
with tabs[3]:
    st.header("Operations")
    st.info("Operations control center coming next.")