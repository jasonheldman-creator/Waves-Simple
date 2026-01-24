import streamlit as st
from datetime import datetime
import pandas as pd

# ----------------------------
# Page Config
# ----------------------------
st.set_page_config(
    page_title="WAVES — Live Recovery Console",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ----------------------------
# Load snapshot data
# ----------------------------
@st.cache_data
def load_snapshot():
    return pd.read_csv("data/live_snapshot.csv")

df = load_snapshot()

# ----------------------------
# Portfolio-level aggregates
# ----------------------------
def pct(x):
    return f"{x:+.2f}%"

portfolio = {
    "return_1d": df["Return_1D"].mean() * 100,
    "return_30d": df["Return_30D"].mean() * 100,
    "return_60d": df["Return_60D"].mean() * 100,
    "return_365d": df["Return_365D"].mean() * 100,
    "alpha_1d": df["Alpha_1D"].mean() * 100,
    "alpha_30d": df["Alpha_30D"].mean() * 100,
    "alpha_60d": df["Alpha_60D"].mean() * 100,
    "alpha_365d": df["Alpha_365D"].mean() * 100,
}

timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")

# ----------------------------
# Header
# ----------------------------
st.markdown("""
# WAVES — Live Recovery Console
Intraday • 30D • 60D • 365D • Snapshot-Driven
---
""")

# ----------------------------
# Portfolio Snapshot (BLUE BOX)
# ----------------------------
snapshot_html = f"""
<style>
.snapshot-box {{
    background: linear-gradient(135deg, #0b1f3a, #0e2a4f);
    border: 2px solid #36d6ff;
    border-radius: 18px;
    padding: 28px;
    margin-top: 20px;
    box-shadow: 0 0 25px rgba(54,214,255,0.35);
}}
.snapshot-title {{
    font-size: 28px;
    font-weight: 700;
    color: #ffffff;
    margin-bottom: 6px;
}}
.snapshot-sub {{
    font-size: 14px;
    color: #9bb7d4;
    margin-bottom: 22px;
}}
.metric-grid {{
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 14px;
}}
.metric {{
    background: rgba(255,255,255,0.06);
    border-radius: 12px;
    padding: 14px;
    text-align: center;
}}
.metric-label {{
    font-size: 12px;
    color: #9bb7d4;
}}
.metric-value {{
    font-size: 22px;
    font-weight: 700;
    color: #ffffff;
}}
.footer-note {{
    margin-top: 18px;
    font-size: 12px;
    color: #9bb7d4;
}}
</style>

<div class="snapshot-box">
    <div class="snapshot-title">🏛 Portfolio Snapshot (All Waves)</div>
    <div class="snapshot-sub">STANDARD MODE</div>

    <div class="metric-grid">
        <div class="metric"><div class="metric-label">Return 1D (Intraday)</div><div class="metric-value">{pct(portfolio["return_1d"])}</div></div>
        <div class="metric"><div class="metric-label">Return 30D</div><div class="metric-value">{pct(portfolio["return_30d"])}</div></div>
        <div class="metric"><div class="metric-label">Return 60D</div><div class="metric-value">{pct(portfolio["return_60d"])}</div></div>
        <div class="metric"><div class="metric-label">Return 365D</div><div class="metric-value">{pct(portfolio["return_365d"])}</div></div>

        <div class="metric"><div class="metric-label">Alpha 1D</div><div class="metric-value">{pct(portfolio["alpha_1d"])}</div></div>
        <div class="metric"><div class="metric-label">Alpha 30D</div><div class="metric-value">{pct(portfolio["alpha_30d"])}</div></div>
        <div class="metric"><div class="metric-label">Alpha 60D</div><div class="metric-value">{pct(portfolio["alpha_60d"])}</div></div>
        <div class="metric"><div class="metric-label">Alpha 365D</div><div class="metric-value">{pct(portfolio["alpha_365d"])}</div></div>
    </div>

    <div class="footer-note">
        ⚡ Computed from live snapshot | {timestamp}
    </div>
</div>
"""

st.markdown(snapshot_html, unsafe_allow_html=True)

# ----------------------------
# Live Returns Table
# ----------------------------
st.markdown("## 📊 Live Returns & Alpha")
st.dataframe(df, use_container_width=True)

# ----------------------------
# System Status
# ----------------------------
st.success("LIVE SYSTEM ACTIVE ✅ — Intraday live • Multi-horizon alpha • Snapshot truth")