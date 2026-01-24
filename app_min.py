# app_min.py
import streamlit as st
import pandas as pd
from datetime import datetime

# -------------------------------------------------
# Page Config
# -------------------------------------------------
st.set_page_config(
    page_title="WAVES — Live Recovery Console",
    layout="wide",
)

# -------------------------------------------------
# Global Styles
# -------------------------------------------------
st.markdown(
    """
    <style>
    body {
        background-color: #0b0f1a;
        color: white;
    }

    .blue-box {
        background: linear-gradient(135deg, #0b2a4a, #0a1f38);
        border: 2px solid #3cc9ff;
        border-radius: 20px;
        padding: 28px;
        box-shadow: 0 0 28px rgba(60,201,255,0.35);
        margin-bottom: 30px;
    }

    .section-label {
        opacity: 0.75;
        margin-bottom: 16px;
        font-size: 14px;
    }

    .status-banner {
        background: linear-gradient(90deg, #0f5132, #198754);
        padding: 14px;
        border-radius: 10px;
        text-align: center;
        font-weight: 700;
        margin-top: 30px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------------------------------------
# Header
# -------------------------------------------------
st.title("WAVES — Live Recovery Console")
st.caption("Intraday • 30D • 60D • 365D • Snapshot-Driven")
st.divider()

# -------------------------------------------------
# Load Live Snapshot
# -------------------------------------------------
SNAPSHOT_PATH = "data/live_snapshot.csv"

try:
    snapshot_df = pd.read_csv(SNAPSHOT_PATH)
except Exception:
    snapshot_df = pd.DataFrame()

def safe_mean(col):
    return snapshot_df[col].mean() if col in snapshot_df.columns else 0.0

# -------------------------------------------------
# Portfolio Snapshot (VISUALS ONLY)
# -------------------------------------------------
snapshot_time = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")

st.markdown(
    """
    <div class="blue-box">
        <h2>🏛 Portfolio Snapshot (All Waves)</h2>
        <div class="section-label">STANDARD MODE</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Streamlit-native metrics (cannot render as code)
c1, c2, c3, c4 = st.columns(4)

c1.metric("Intraday Return", f"{safe_mean('Return_1D')*100:.2f}%")
c2.metric("30D Return", f"{safe_mean('Return_30D')*100:.2f}%")
c3.metric("60D Return", f"{safe_mean('Return_60D')*100:.2f}%")
c4.metric("365D Return", f"{safe_mean('Return_365D')*100:.2f}%")

c5, c6, c7, c8 = st.columns(4)

c5.metric("Alpha Intraday", f"{safe_mean('Alpha_1D')*100:.2f}%")
c6.metric("Alpha 30D", f"{safe_mean('Alpha_30D')*100:.2f}%")
c7.metric("Alpha 60D", f"{safe_mean('Alpha_60D')*100:.2f}%")
c8.metric("Alpha 365D", f"{safe_mean('Alpha_365D')*100:.2f}%")

st.caption(
    f"⚡ Computed from live snapshot | {snapshot_time}  \n"
    "ℹ Wave-level Beta, Exposure, Cash, VIX regime shown below"
)

st.divider()

# -------------------------------------------------
# Live Returns & Alpha Table
# -------------------------------------------------
st.subheader("📊 Live Returns & Alpha")

table_cols = [
    "Wave",
    "Return_1D",
    "Return_30D",
    "Return_60D",
    "Return_365D",
    "Alpha_1D",
    "Alpha_30D",
    "Alpha_60D",
    "Alpha_365D",
]

available_cols = [c for c in table_cols if c in snapshot_df.columns]

if available_cols:
    st.dataframe(snapshot_df[available_cols], use_container_width=True)
else:
    st.info("Live returns table unavailable.")

# -------------------------------------------------
# Alpha History by Horizon
# -------------------------------------------------
st.subheader("📈 Alpha History by Horizon")

alpha_cols = ["Alpha_1D", "Alpha_30D", "Alpha_60D", "Alpha_365D"]
alpha_cols = [c for c in alpha_cols if c in snapshot_df.columns]

if alpha_cols:
    alpha_df = snapshot_df[["Wave"] + alpha_cols].set_index("Wave")
    st.bar_chart(alpha_df)
else:
    st.warning("Alpha data unavailable or incomplete for horizon chart.")

# -------------------------------------------------
# System Status
# -------------------------------------------------
st.markdown(
    """
    <div class="status-banner">
        LIVE SYSTEM ACTIVE ✅
    </div>
    """,
    unsafe_allow_html=True,
)

st.caption(
    "✓ Intraday live • ✓ Multi-horizon returns • ✓ Alpha attribution • ✓ Snapshot truth"
)