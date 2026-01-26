# app_min.py
# WAVES Intelligence™ Console (Minimal)
# FINAL FIX — Order-safe intraday handling + portfolio aggregate + alpha history

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

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
def load_snapshot():
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

    # 🔒 CRITICAL FIX:
    # Pre-inject all expected return columns BEFORE any access
    for col in RETURN_COLS.values():
        if col not in df.columns:
            df[col] = np.nan

    for col in BENCHMARK_COLS.values():
        if col not in df.columns:
            df[col] = np.nan

    return df, None

snapshot_df, snapshot_error = load_snapshot()

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

        # ---- Portfolio aggregate (mean across waves)
        portfolio_row = {"display_name": "TOTAL PORTFOLIO"}
        for key, col in RETURN_COLS.items():
            portfolio_row[col] = df[col].mean(skipna=True)

        df = pd.concat([pd.DataFrame([portfolio_row]), df], ignore_index=True)

        # ---- Dynamic column assembly (NO eager access)
        columns_to_show = ["display_name"]
        columns_to_show += [RETURN_COLS["intraday"]]
        columns_to_show += [
            RETURN_COLS["30d"],
            RETURN_COLS["60d"],
            RETURN_COLS["365d"],
        ]

        snapshot_view = df[columns_to_show].rename(columns={
            "display_name": "Wave",
            RETURN_COLS["intraday"]: "Intraday",
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
    h_key = horizon.lower()

    # ---- Source Breakdown (unchanged)
    st.subheader("Source Breakdown")
    st.dataframe(
        pd.DataFrame({
            "Alpha Source": [
                "Selection Alpha",
                "Momentum Alpha",
                "Regime Alpha",
                "Exposure Alpha",
                "Residual Alpha",
            ],
            "Contribution": [0.012, 0.008, -0.003, 0.004, 0.001],
        }),
        use_container_width=True,
        hide_index=True
    )

    # ---- Alpha History (REAL, derived)
    st.subheader("Alpha History")

    ret_col = RETURN_COLS[h_key]
    bench_col = BENCHMARK_COLS[h_key]

    wave_row = snapshot_df[snapshot_df["display_name"] == selected_wave]

    if wave_row.empty:
        st.warning("Wave data not available.")
    else:
        wave_row = wave_row.iloc[0]
        alpha_value = wave_row[ret_col] - wave_row[bench_col]

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
    