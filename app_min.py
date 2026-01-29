# ============================================================
# app_min.py
# WAVES Intelligence™ Console (Minimal)
# ============================================================

import streamlit as st

# Temporary cache invalidation for diagnostic purposes
st.cache_data.clear()
st.cache_resource.clear()

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
ALPHA_ATTRIBUTION_PATH = DATA_DIR / "alpha_attribution_snapshot.csv"

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

    # -------------------------------------------------------
    # Merge alpha attribution snapshot (data plumbing only)
    # -------------------------------------------------------
    # This brings in:
    #   alpha_market
    #   alpha_momentum
    #   alpha_volatility
    #   alpha_rotation
    #   alpha_stock_selection
    # and any suffixed variants, without changing math or schemas.
    if ALPHA_ATTRIBUTION_PATH.exists():
        try:
            attrib_df = pd.read_csv(ALPHA_ATTRIBUTION_PATH)
            attrib_df.columns = [c.strip().lower() for c in attrib_df.columns]

            if "wave_id" in df.columns and "wave_id" in attrib_df.columns:
                df = df.merge(
                    attrib_df,
                    on="wave_id",
                    how="left",
                )
        except Exception:
            # Governance-safe: if attribution snapshot is unreadable,
            # continue with base snapshot only.
            pass

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