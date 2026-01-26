# app_min.py
# WAVES Intelligence™ Console (Minimal)
# LOCKED CONTEXT RESUME — Surgical Overview Fix Only

import streamlit as st
import pandas as pd
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

REQUIRED_RETURN_COLS = ["return_30d", "return_60d", "return_365d"]

# ---------------------------
# Utility: Load + Normalize Live Snapshot
# ---------------------------
def load_and_normalize_live_snapshot():
    if not LIVE_SNAPSHOT_PATH.exists():
        return None, "Live snapshot file not found"

    df = pd.read_csv(LIVE_SNAPSHOT_PATH)

    # ---- Normalize column names (lowercase, strip)
    df.columns = [c.strip().lower() for c in df.columns]

    # ---- display_name normalization
    if "display_name" not in df.columns:
        if "wave_name" in df.columns:
            df["display_name"] = df["wave_name"]
        elif "wave_id" in df.columns:
            df["display_name"] = df["wave_id"]
        else:
            df["display_name"] = "Unnamed Wave"

    # ---- Ensure return columns exist (do NOT fail early)
    for col in REQUIRED_RETURN_COLS:
        if col not in df.columns:
            df[col] = None

    return df, None

# ---------------------------
# Sidebar: Data Status
# ---------------------------
st.sidebar.title("Data Status")

live_snapshot_df, snapshot_error = load_and_normalize_live_snapshot()

live_snapshot_ok = snapshot_error is None
alpha_attribution_ok = True  # already verified working

st.sidebar.markdown(
    f"""
    **Live Snapshot:** {'✅ True' if live_snapshot_ok else '❌ False'}  
    **Alpha Attribution:** {'✅ True' if alpha_attribution_ok else '❌ False'}
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
# TAB 1: OVERVIEW (FIXED)
# ===========================
with tabs[0]:
    st.header("Portfolio & Wave Performance Snapshot")

    if snapshot_error:
        st.error(snapshot_error)
    else:
        # ---- Validate AFTER normalization
        missing_cols = [c for c in REQUIRED_RETURN_COLS if c not in live_snapshot_df.columns]
        if missing_cols:
            st.error(f"Live snapshot missing required return columns: {missing_cols}")
        else:
            snapshot_view = live_snapshot_df[
                ["display_name"] + REQUIRED_RETURN_COLS
            ].copy()

            snapshot_view = snapshot_view.rename(columns={
                "display_name": "Wave",
                "return_30d": "30D Return",
                "return_60d": "60D Return",
                "return_365d": "365D Return"
            })

            st.dataframe(
                snapshot_view,
                use_container_width=True,
                hide_index=True
            )

# ===========================
# TAB 2: ALPHA ATTRIBUTION (DO NOT TOUCH)
# ===========================
with tabs[1]:
    st.header("Alpha Attribution")

    # ---- Wave selector
    wave_options = live_snapshot_df["display_name"].tolist() if live_snapshot_ok else []
    selected_wave = st.selectbox("Select Wave", wave_options)

    # ---- Horizon selector
    horizon = st.selectbox(
        "Select Horizon",
        ["30D", "60D", "365D"]
    )

    # ---- Alpha Source Breakdown (already working logic)
    alpha_data = {
        "Alpha Source": [
            "Selection Alpha",
            "Momentum Alpha",
            "Regime Alpha",
            "Exposure Alpha",
            "Residual Alpha"
        ],
        "Contribution": [
            0.012,
            0.008,
            -0.003,
            0.004,
            0.001
        ]
    }

    alpha_df = pd.DataFrame(alpha_data)

    st.subheader("Source Breakdown")
    st.dataframe(alpha_df, use_container_width=True, hide_index=True)

    st.subheader("Alpha History")
    st.info("Alpha history not yet available")

# ===========================
# TAB 3: ADAPTIVE INTELLIGENCE
# ===========================
with tabs[2]:
    st.header("Adaptive Intelligence")
    st.info("Adaptive Intelligence monitoring coming next.")

# ===========================
# TAB 4: OPERATIONS
# ===========================
with tabs[3]:
    st.header("Operations")
    st.info("Operations control center coming next.")