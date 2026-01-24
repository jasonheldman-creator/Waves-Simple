import streamlit as st
import pandas as pd
from datetime import datetime

# =============================
# PAGE CONFIG
# =============================
st.set_page_config(
    page_title="WAVES — Live Recovery Console",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =============================
# STYLES
# =============================
st.markdown("""
<style>
body {
    background-color: #0e1117;
}
.blue-box {
    background-color: #081c33;
    border: 2px solid #3fd0ff;
    border-radius: 18px;
    padding: 20px;
    margin-bottom: 30px;
    box-shadow: 0 0 25px rgba(63, 208, 255, 0.35);
}
.bold-metric-title {
    color: #3fd0ff;
    font-size: 20px;
    font-weight: bold;
    margin-top: 10px;
}
.bold-metric-value {
    color: white;
    font-size: 30px;
    font-weight: bold;
}
.metric-caption {
    color: #a0a0a0;
    font-size: 14px;
}
</style>
""", unsafe_allow_html=True)

# =============================
# HEADER
# =============================
st.title("WAVES — Live Recovery Console")
st.caption("Intraday • 30D • 60D • 365D • Snapshot-Driven")
st.divider()

# =============================
# LOAD SNAPSHOT DATA
# =============================
snapshot_df = pd.read_csv("data/live_snapshot.csv")
portfolio = snapshot_df.mean(numeric_only=True)

# =============================
# PORTFOLIO SNAPSHOT (EXECUTIVE BLUE BOX)
# =============================
with st.container():
    st.markdown('<div class="blue-box">', unsafe_allow_html=True)
    st.subheader("🏛️ Portfolio Snapshot (All Waves)")
    st.caption("STANDARD MODE")
    
    # Executive-style horizontal metrics layout
    st.text("")  # Spacer
    
    # ROW 1: Returns (1D, 30D, 60D, 365D)
    st.markdown("### 📈 Returns")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric(label="Intraday Return", value=f"{portfolio['Return_1D']*100:.2f}%", help="Performance for the past 1 day.")
    col2.metric(label="30D Return", value=f"{portfolio['Return_30D']*100:.2f}%", help="Performance over the last 30 days.")
    col3.metric(label="60D Return", value=f"{portfolio['Return_60D']*100:.2f}%", help="Performance over the last 60 days.")
    col4.metric(label="365D Return", value=f"{portfolio['Return_365D']*100:.2f}%", help="Performance over the past year.")

    # ROW 2: Alpha metrics (1D, 30D, 60D, 365D)
    st.markdown("### ⚡ Alpha Metrics")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric(label="Alpha Intraday", value=f"{portfolio['Alpha_1D']*100:.2f}%", help="Alpha generated in the past 1 day.")
    col2.metric(label="Alpha 30D", value=f"{portfolio['Alpha_30D']*100:.2f}%", help="Alpha over the last 30 days.")
    col3.metric(label="Alpha 60D", value=f"{portfolio['Alpha_60D']*100:.2f}%", help="Alpha over the last 60 days.")
    col4.metric(label="Alpha 365D", value=f"{portfolio['Alpha_365D']*100:.2f}%", help="Alpha achieved over the last year.")
    
    # Additional Executive Summary
    st.text("")  # Spacer
    st.caption(f"⚡ Computed from live snapshot | {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")
    st.caption("ℹ Includes aggregated portfolio-level views for Alpha, Beta, Exposure & Market Conditions.")
    st.markdown('</div>', unsafe_allow_html=True)

# =============================
# LIVE RETURNS & ALPHA TABLE
# =============================
st.subheader("📊 Live Returns & Alpha")
st.dataframe(
    snapshot_df[
        [
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
    ],
    use_container_width=True
)

# =============================
# ALPHA HISTORY BY HORIZON
# =============================
st.subheader("📈 Alpha History by Horizon")

alpha_cols = ["Alpha_1D", "Alpha_30D", "Alpha_365D"]
alpha_df = snapshot_df[["Wave"] + alpha_cols].set_index("Wave")

if alpha_df.dropna().empty:
    st.warning("Alpha data unavailable or incomplete for horizon chart.")
else:
    st.bar_chart(alpha_df)

# =============================
# STATUS BANNER
# =============================
st.markdown(
    """
    <div class="status-banner">
        LIVE SYSTEM ACTIVE ✅<br/>
        ✓ Intraday live • ✓ Multi-horizon returns • ✓ Alpha attribution • ✓ Snapshot truth
    </div>
    """,
    unsafe_allow_html=True
)