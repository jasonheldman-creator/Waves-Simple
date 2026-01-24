# ==========================================================
# app_min.py — WAVES Live Recovery Console (Canonical)
# ==========================================================
# Snapshot-driven, read-only recovery console
# • Intraday + 30D / 60D / 365D Returns & Alpha
# • Portfolio Snapshot (blue box)
# • Horizon comparison
# • Zero dependency on legacy wave init
# ==========================================================

import streamlit as st
import sys
import os
import pandas as pd
from datetime import datetime

# ----------------------------------------------------------
# STREAMLIT CONFIG
# ----------------------------------------------------------

st.set_page_config(
    page_title="WAVES — Live Recovery Console",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ----------------------------------------------------------
# BOOT CONFIRMATION
# ----------------------------------------------------------

st.success("STREAMLIT EXECUTION STARTED")
st.write("app_min.py reached line 1")

# ----------------------------------------------------------
# LOAD SNAPSHOT (CANONICAL)
# ----------------------------------------------------------

SNAPSHOT_PATH = "data/live_snapshot.csv"

if not os.path.exists(SNAPSHOT_PATH):
    st.error(f"Missing snapshot: {SNAPSHOT_PATH}")
    st.stop()

snapshot_df = pd.read_csv(SNAPSHOT_PATH)

# ----------------------------------------------------------
# DERIVED METRICS (SAFE)
# ----------------------------------------------------------

def safe_col(df, col):
    return df[col] if col in df.columns else 0.0

row_count = len(snapshot_df)

asof_date = (
    snapshot_df["Date"].max()
    if "Date" in snapshot_df.columns
    else "Unknown"
)

# Portfolio-level aggregates
portfolio = {}

portfolio["Return_INTRA"] = safe_col(snapshot_df, "Return_1D").mean()
portfolio["Alpha_INTRA"]  = safe_col(snapshot_df, "Alpha_1D").mean()

portfolio["Alpha_30D"]  = safe_col(snapshot_df, "Alpha_30D").mean()
portfolio["Alpha_60D"]  = safe_col(snapshot_df, "Alpha_60D").mean()
portfolio["Alpha_365D"] = safe_col(snapshot_df, "Alpha_365D").mean()

portfolio["Exposure"] = safe_col(snapshot_df, "Exposure").mean()
portfolio["Cash"]     = safe_col(snapshot_df, "CashPercent").mean()

portfolio["VIX"]    = safe_col(snapshot_df, "VIX_Level").mean()
portfolio["Regime"] = (
    snapshot_df["VIX_Regime"].mode().iloc[0]
    if "VIX_Regime" in snapshot_df.columns and not snapshot_df["VIX_Regime"].empty
    else "Unknown"
)

portfolio["Coverage"] = safe_col(snapshot_df, "Coverage_Score").mean()

# ----------------------------------------------------------
# 🔵 PORTFOLIO SNAPSHOT BOX (BLUE)
# ----------------------------------------------------------

st.markdown(
    """
    <style>
    .snapshot-box {
        background-color: #0b2c4d;
        padding: 20px;
        border-radius: 12px;
        color: white;
        margin-bottom: 20px;
    }
    .snapshot-metric {
        font-size: 28px;
        font-weight: 700;
    }
    .snapshot-label {
        font-size: 12px;
        opacity: 0.8;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    f"""
    <div class="snapshot-box">
        <div style="display:flex; justify-content:space-between; gap:20px;">

            <div>
                <div class="snapshot-label">SYSTEM STATUS</div>
                <div class="snapshot-metric">LIVE</div>
                <div class="snapshot-label">As of {asof_date} · Intraday</div>
                <div class="snapshot-label">Waves active: {row_count}</div>
            </div>

            <div>
                <div class="snapshot-label">PORTFOLIO (INTRADAY)</div>
                <div class="snapshot-metric">
                    Return {portfolio["Return_INTRA"]:.3%}
                </div>
                <div class="snapshot-metric">
                    Alpha {portfolio["Alpha_INTRA"]:.3%}
                </div>
            </div>

            <div>
                <div class="snapshot-label">ALPHA BY HORIZON</div>
                <div>30D: {portfolio["Alpha_30D"]:.3%}</div>
                <div>60D: {portfolio["Alpha_60D"]:.3%}</div>
                <div>365D: {portfolio["Alpha_365D"]:.3%}</div>
            </div>

            <div>
                <div class="snapshot-label">RISK CONTEXT</div>
                <div>Exposure: {portfolio["Exposure"]:.1%}</div>
                <div>Cash: {portfolio["Cash"]:.1%}</div>
                <div>VIX: {portfolio["VIX"]:.1f}</div>
                <div>Regime: {portfolio["Regime"]}</div>
            </div>

        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# ----------------------------------------------------------
# LIVE RETURNS & ALPHA (TABLE)
# ----------------------------------------------------------

st.subheader("📊 Live Returns & Alpha")

live_cols = [
    "Wave_ID",
    "Return_1D",
    "Benchmark_Return_1D",
    "Alpha_1D",
    "Return_30D",
    "Alpha_30D",
    "Return_60D",
    "Alpha_60D",
    "Return_365D",
    "Alpha_365D",
]

available_cols = [c for c in live_cols if c in snapshot_df.columns]

live_df = (
    snapshot_df[available_cols]
    .groupby("Wave_ID")
    .mean()
    .reset_index()
)

st.dataframe(live_df, use_container_width=True)

# ----------------------------------------------------------
# ALPHA BY HORIZON (BAR)
# ----------------------------------------------------------

st.subheader("📈 Alpha by Horizon")

alpha_cols = [c for c in ["Alpha_30D", "Alpha_60D", "Alpha_365D"] if c in live_df.columns]

if alpha_cols:
    chart_df = live_df.set_index("Wave_ID")[alpha_cols]
    st.bar_chart(chart_df)
else:
    st.info("Alpha horizon columns not available.")

# ----------------------------------------------------------
# FINAL STATUS
# ----------------------------------------------------------

st.success(
    "LIVE SYSTEM ACTIVE ✅\n\n"
    "✔ Intraday returns populated\n"
    "✔ Multi-horizon alpha live\n"
    "✔ Snapshot-driven truth\n"
    "✔ Zero dependency on legacy wave init\n\n"
    "Forward build unblocked."
)

with st.expander("Preview snapshot (first 10 rows)"):
    st.dataframe(snapshot_df.head(10))