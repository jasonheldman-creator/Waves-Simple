# ============================================================
# WAVES — Live Recovery Console (app_min.py)
# FULL REPLACEMENT — SNAPSHOT-DRIVEN, MOBILE-SAFE
# ============================================================

import streamlit as st
import pandas as pd
from datetime import datetime, timezone

# ------------------------------------------------------------
# Page Config
# ------------------------------------------------------------
st.set_page_config(
    page_title="WAVES — Live Recovery Console",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ------------------------------------------------------------
# Styling
# ------------------------------------------------------------
st.markdown("""
<style>
.blue-box {
    background: linear-gradient(135deg, #0b1f33, #102b46);
    border: 2px solid #2ecbff;
    border-radius: 18px;
    padding: 22px;
    margin-bottom: 24px;
    box-shadow: 0 0 22px rgba(46,203,255,0.25);
}
.blue-title {
    font-size: 22px;
    font-weight: 800;
    color: #e9f6ff;
    margin-bottom: 6px;
}
.mode-pill {
    display: inline-block;
    background: #1aff7c;
    color: #052814;
    font-weight: 700;
    font-size: 12px;
    padding: 4px 10px;
    border-radius: 12px;
    margin-left: 10px;
}
.metric-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
    gap: 12px;
    margin-top: 14px;
}
.metric-card {
    background: rgba(255,255,255,0.05);
    border-radius: 12px;
    padding: 10px;
    text-align: center;
}
.metric-label {
    font-size: 12px;
    color: #9ecbff;
}
.metric-value {
    font-size: 18px;
    font-weight: 800;
    color: #ffffff;
}
.metric-value.negative {
    color: #ff6b6b;
}
.footer-text {
    margin-top: 14px;
    font-size: 12px;
    color: #9bbbd8;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------
# Header
# ------------------------------------------------------------
st.markdown("## 🌊 WAVES — Live Recovery Console")
st.caption("Intraday • 30D • 60D • 365D — Snapshot Driven")

# ------------------------------------------------------------
# Load Snapshot
# ------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_snapshot():
    return pd.read_csv("data/live_snapshot.csv")

df = load_snapshot()

st.success("Live snapshot loaded")
st.caption(f"Rows: {len(df)}")

# ------------------------------------------------------------
# Helper: Safe Mean
# ------------------------------------------------------------
def mean_or_zero(series):
    if series.dropna().empty:
        return 0.0
    return float(series.mean())

# ------------------------------------------------------------
# Portfolio Aggregation (ALL WAVES)
# ------------------------------------------------------------
portfolio = {
    "Return_1D": mean_or_zero(df["Return_1D"]),
    "Return_30D": mean_or_zero(df["Return_30D"]),
    "Return_60D": mean_or_zero(df["Return_60D"]),
    "Return_365D": mean_or_zero(df["Return_365D"]),
    "Alpha_1D": mean_or_zero(df["Alpha_1D"]),
    "Alpha_30D": mean_or_zero(df["Alpha_30D"]),
    "Alpha_60D": mean_or_zero(df["Alpha_60D"]),
    "Alpha_365D": mean_or_zero(df["Alpha_365D"]),
}

def fmt(x):
    return f"{x:+.2%}"

def val_class(x):
    return "negative" if x < 0 else ""

# ------------------------------------------------------------
# BLUE PORTFOLIO SNAPSHOT BOX (FULL RESTORE)
# ------------------------------------------------------------
st.markdown(f"""
<div class="blue-box">
    <div class="blue-title">
        🏛️ Portfolio Snapshot (All Waves)
        <span class="mode-pill">STANDARD</span>
    </div>

    <div class="metric-grid">
        <div class="metric-card">
            <div class="metric-label">Intraday Return</div>
            <div class="metric-value {val_class(portfolio['Return_1D'])}">{fmt(portfolio['Return_1D'])}</div>