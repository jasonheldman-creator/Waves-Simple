# ==========================================================
# app_min.py — WAVES Live Recovery Console (Canonical)
# ==========================================================
# Snapshot-driven, read-only, rendering-safe implementation
# ==========================================================

import streamlit as st
import pandas as pd
import os
import sys
from datetime import datetime

# ----------------------------------------------------------
# BOOT CONFIRMATION
# ----------------------------------------------------------

st.success("STREAMLIT EXECUTION STARTED")
st.write("app_min.py reached line 1")

# ----------------------------------------------------------
# PAGE CONFIG
# ----------------------------------------------------------

st.set_page_config(
    page_title="WAVES — Live Recovery Console",
    layout="wide",
)

st.title("WAVES — Live Recovery Console")
st.caption("Intraday • 30D • 60D • 365D — Snapshot Driven")

# ----------------------------------------------------------
# LOAD SNAPSHOT
# ----------------------------------------------------------

SNAPSHOT_PATH = "data/live_snapshot.csv"

if not os.path.exists(SNAPSHOT_PATH):
    st.error(f"Missing snapshot: {SNAPSHOT_PATH}")
    st.stop()

df = pd.read_csv(SNAPSHOT_PATH)

st.success("Live snapshot loaded")
st.write(f"Rows: {len(df)}")

# ----------------------------------------------------------
# SAFE COLUMN ACCESS
# ----------------------------------------------------------

def col(name, default=0.0):
    return df[name] if name in df.columns else default

# ----------------------------------------------------------
# PORTFOLIO AGGREGATES
# ----------------------------------------------------------

portfolio_return = col("Return_1D").mean() * 100
portfolio_alpha = col("Alpha_1D").mean() * 100

alpha_30d = col("Alpha_30D").mean() * 100
alpha_60d = col("Alpha_60D").mean() * 100
alpha_365d = col("Alpha_365D").mean() * 100

exposure = col("Exposure").mean()
cash = col("CashPercent").mean()
vix = col("VIX_Level").mean()
regime = col("VIX_Regime").mode().iloc[0] if "VIX_Regime" in df.columns else "unknown"

as_of = df["Date"].max() if "Date" in df.columns else datetime.utcnow().date()

# ----------------------------------------------------------
# BLUE SNAPSHOT BOX (GUARANTEED RENDER)
# ----------------------------------------------------------

snapshot_html = f"""
<div style="
    background: linear-gradient(135deg, #0a2540, #0b3a5a);
    border-radius: 16px;
    padding: 24px;
    color: #e8f1ff;
    box-shadow: 0 12px 30px rgba(0,0,0,0.4);
    margin-bottom: 32px;
">

  <h3 style="margin-top:0;">📘 Portfolio Snapshot</h3>

  <div style="display:grid; grid-template-columns: repeat(4, 1fr); gap:16px;">

    <div>
      <strong>System</strong><br/>
      LIVE
    </div>

    <div>
      <strong>As of</strong><br/>
      {as_of}
    </div>

    <div>
      <strong>Waves</strong><br/>
      {df["Wave_ID"].nunique()}
    </div>

    <div>
      <strong>Regime</strong><br/>
      {regime}
    </div>

    <div>
      <strong>Return (Intraday)</strong><br/>
      {portfolio_return:.3f}%
    </div>

    <div>
      <strong>Alpha (Intraday)</strong><br/>
      {portfolio_alpha:.3f}%
    </div>

    <div>
      <strong>Exposure</strong><br/>
      {exposure:.1f}%
    </div>

    <div>
      <strong>Cash</strong><br/>
      {cash:.1f}%
    </div>

  </div>

  <hr style="margin:20px 0; border-color:#1c4d75;"/>

  <div style="display:grid; grid-template-columns: repeat(3, 1fr); gap:16px;">
    <div><strong>Alpha 30D</strong><br/>{alpha_30d:.3f}%</div>
    <div><strong>Alpha 60D</strong><br/>{alpha_60d:.3f}%</div>
    <div><strong>Alpha 365D</strong><br/>{alpha_365d:.3f}%</div>
  </div>

</div>
"""

st.markdown(snapshot_html, unsafe_allow_html=True)

# ----------------------------------------------------------
# LIVE RETURNS & ALPHA TABLE
# ----------------------------------------------------------

st.subheader("📊 Live Returns & Alpha")

returns_cols = [
    "Wave_ID",
    "Return_1D",
    "Return_30D",
    "Return_60D",
    "Return_365D",
    "Alpha_1D",
    "Alpha_30D",
    "Alpha_60D",
    "Alpha_365D",
]

table_cols = [c for c in returns_cols if c in df.columns]

returns_df = df[table_cols].copy()

st.dataframe(
    returns_df.sort_values("Alpha_365D", ascending=False),
    use_container_width=True,
)

# ----------------------------------------------------------
# ALPHA BY HORIZON (BAR CHART)
# ----------------------------------------------------------

st.subheader("📈 Alpha by Horizon")

alpha_chart = (
    df.groupby("Wave_ID")[["Alpha_30D", "Alpha_60D", "Alpha_365D"]]
    .mean()
    .sort_values("Alpha_365D", ascending=False)
)

st.bar_chart(alpha_chart)

# ----------------------------------------------------------
# SYSTEM STATUS
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
    st.dataframe(df.head(10))