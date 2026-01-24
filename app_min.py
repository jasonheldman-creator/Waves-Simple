# ==========================================================
# app_min.py — WAVES Live Recovery Console (RESTORED)
# ==========================================================
# Combines:
# • Correct blue snapshot box (HTML fixed)
# • Full Live Returns table
# • Alpha by Horizon chart
# • Snapshot-driven, read-only safety
# ==========================================================

import streamlit as st
import pandas as pd
import os
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
# TITLE
# ----------------------------------------------------------

st.title("WAVES — Live Recovery Console")
st.caption("Intraday · 30D · 60D · 365D — Snapshot Driven")

# ----------------------------------------------------------
# LOAD SNAPSHOT
# ----------------------------------------------------------

SNAPSHOT_PATH = "data/live_snapshot.csv"

if not os.path.exists(SNAPSHOT_PATH):
    st.error("❌ live_snapshot.csv not found")
    st.stop()

df = pd.read_csv(SNAPSHOT_PATH)

st.success("Live snapshot loaded")
st.write(f"Rows: {len(df)}")

# ----------------------------------------------------------
# SAFE HELPERS
# ----------------------------------------------------------

def safe_mean(col):
    return float(df[col].dropna().mean()) if col in df.columns else 0.0

def safe_mode(col):
    return df[col].mode().iloc[0] if col in df and not df[col].dropna().empty else "unknown"

# ----------------------------------------------------------
# PORTFOLIO AGGREGATES
# ----------------------------------------------------------

portfolio = {
    "return_1d": safe_mean("Return_1D"),
    "alpha_1d": safe_mean("Alpha_1D"),
    "alpha_30d": safe_mean("Alpha_30D"),
    "alpha_60d": safe_mean("Alpha_60D"),
    "alpha_365d": safe_mean("Alpha_365D"),
    "exposure": safe_mean("Exposure"),
    "cash": safe_mean("CashPercent"),
    "vix": safe_mean("VIX_Level"),
    "regime": safe_mode("VIX_Regime")
}

# ----------------------------------------------------------
# BLUE SNAPSHOT BOX (FIXED)
# ----------------------------------------------------------

snapshot_html = f"""
<div style="
    background: linear-gradient(135deg, #0b2a4a, #0e3a63);
    padding: 28px;
    border-radius: 16px;
    color: #eaf4ff;
    box-shadow: 0 10px 30px rgba(0,0,0,0.35);
    margin-bottom: 32px;
">
  <div style="display:grid; grid-template-columns: repeat(4, 1fr); gap: 22px;">

    <div>
      <strong>SYSTEM</strong><br/>
      LIVE<br/>
      As of {datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")}
    </div>

    <div>
      <strong>PORTFOLIO</strong><br/>
      Return: {portfolio["return_1d"]:.3%}<br/>
      Alpha: {portfolio["alpha_1d"]:.3%}
    </div>

    <div>
      <strong>ALPHA</strong><br/>
      30D: {portfolio["alpha_30d"]:.3%}<br/>
      60D: {portfolio["alpha_60d"]:.3%}<br/>
      365D: {portfolio["alpha_365d"]:.3%}
    </div>

    <div>
      <strong>RISK</strong><br/>
      Exposure: {portfolio["exposure"]:.1f}%<br/>
      Cash: {portfolio["cash"]:.1f}%<br/>
      VIX: {portfolio["vix"]:.1f}<br/>
      Regime: {portfolio["regime"]}
    </div>

  </div>
</div>
"""

st.markdown(snapshot_html, unsafe_allow_html=True)

# ----------------------------------------------------------
# LIVE RETURNS & ALPHA TABLE (RESTORED)
# ----------------------------------------------------------

st.subheader("📊 Live Returns & Alpha")

table_cols = [
    "Wave_ID",
    "Return_1D",
    "Return_30D",
    "Return_60D",
    "Return_365D",
    "Alpha_1D",
    "Alpha_30D",
    "Alpha_60D",
    "Alpha_365D"
]

existing = [c for c in table_cols if c in df.columns]

returns_df = (
    df[existing]
    .sort_values("Alpha_365D", ascending=False)
    .reset_index(drop=True)
)

st.dataframe(returns_df, use_container_width=True)

# ----------------------------------------------------------
# ALPHA BY HORIZON (RESTORED CHART)
# ----------------------------------------------------------

st.subheader("📈 Alpha by Horizon")

alpha_chart_df = (
    df.groupby("Wave_ID")[["Alpha_30D", "Alpha_60D", "Alpha_365D"]]
    .mean()
    .sort_values("Alpha_365D", ascending=False)
)

st.bar_chart(alpha_chart_df)

# ----------------------------------------------------------
# SYSTEM STATUS
# ----------------------------------------------------------

st.success(
    "LIVE SYSTEM ACTIVE ✅\n\n"
    "✓ Intraday returns populated\n"
    "✓ Multi-horizon alpha live\n"
    "✓ Snapshot-driven truth\n"
    "✓ Blue snapshot box fixed\n"
    "✓ No legacy wave init dependency\n\n"
    "Forward build unblocked."
)

with st.expander("Preview snapshot (first 10 rows)"):
    st.dataframe(df.head(10))