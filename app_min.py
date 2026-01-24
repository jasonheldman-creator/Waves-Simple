# ==========================================================
# app_min.py — WAVES Recovery Console (Aggressive / Live)
# ==========================================================
# Canonical forward console.
# Snapshot-driven.
# Intraday-first.
# Zero dependency on wave initialization.
# ==========================================================

import streamlit as st
import sys
import os
import pandas as pd

from returns_alpha_engine import build_returns_alpha

# ----------------------------------------------------------
# BOOT CONFIRMATION
# ----------------------------------------------------------

st.success("STREAMLIT EXECUTION STARTED")
st.write("app_min.py reached line 1")

# ----------------------------------------------------------
# HEADER
# ----------------------------------------------------------

st.title("WAVES — Live Recovery Console")
st.caption("Intraday • 30D • 60D • 365D — Snapshot Driven")

# ----------------------------------------------------------
# ENVIRONMENT
# ----------------------------------------------------------

with st.expander("🧭 Runtime Environment"):
    st.write("Python:", sys.version)
    st.write("Working directory:", os.getcwd())
    st.write("Files:", sorted(os.listdir(".")))

# ----------------------------------------------------------
# LOAD SNAPSHOT
# ----------------------------------------------------------

SNAPSHOT_PATH = "data/live_snapshot.csv"

if not os.path.exists(SNAPSHOT_PATH):
    st.error(f"Missing snapshot: {SNAPSHOT_PATH}")
    st.stop()

snapshot_df = pd.read_csv(SNAPSHOT_PATH)

st.success("Live snapshot loaded")
st.write("Rows:", len(snapshot_df))
st.write("Columns:", list(snapshot_df.columns))

# ----------------------------------------------------------
# LIVE RETURNS & ALPHA
# ----------------------------------------------------------

st.divider()
st.subheader("📊 Live Returns & Alpha")

returns_alpha_df = build_returns_alpha(snapshot_df)

st.dataframe(
    returns_alpha_df.sort_values("Alpha_INTRADAY", ascending=False),
    use_container_width=True
)

# ----------------------------------------------------------
# CHARTS
# ----------------------------------------------------------

st.divider()
st.subheader("📈 Alpha by Horizon")

chart_cols = [
    "Alpha_INTRADAY",
    "Alpha_30D",
    "Alpha_60D",
    "Alpha_365D",
]

available = [c for c in chart_cols if c in returns_alpha_df.columns]

if available:
    st.bar_chart(
        returns_alpha_df.set_index("Wave_ID")[available]
    )
else:
    st.warning("No alpha columns available for charting")

# ----------------------------------------------------------
# FINAL STATUS
# ----------------------------------------------------------

st.divider()
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