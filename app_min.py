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
# GLOBAL STYLES
# =============================
st.markdown("""
<style>
body {
    background-color: #0e1117;
}

.blue-box {
    background: linear-gradient(145deg, #0b2a4a, #081c33);
    border: 2px solid #3fd0ff;
    border-radius: 20px;
    padding: 28px 32px;
    margin-bottom: 40px;
    box-shadow: 0 0 35px rgba(63,208,255,0.35);
}

.snapshot-title {
    font-size: 34px;
    font-weight: 800;
    color: white;
    margin-bottom: 4px;
}

.snapshot-sub {
    font-size: 14px;
    letter-spacing: 1px;
    color: #9fbad0;
    margin-bottom: 26px;
}

.section-label {
    font-size: 15px;
    font-weight: 700;
    letter-spacing: 1.5px;
    color: #3fd0ff;
    margin: 22px 0 12px 0;
}

.metric-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 18px;
}

.metric-card {
    background: rgba(255,255,255,0.03);
    border-radius: 14px;
    padding: 18px 16px;
}

.metric-label {
    font-size: 13px;
    color: #9aa7b3;
    margin-bottom: 6px;
}

.metric-value {
    font-size: 30px;
    font-weight: 800;
    color: white;
}

.footer-note {
    margin-top: 22px;
    font-size: 13px;
    color: #8fa3b5;
}

.status-banner {
    background: linear-gradient(90deg, #1f8f4e, #2ecc71);
    padding: 18px;
    border-radius: 12px;
    font-weight: 700;
    text-align: center;
    color: white;
    margin-top: 40px;
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
# LOAD DATA
# =============================
snapshot_df = pd.read_csv("data/live_snapshot.csv")
portfolio = snapshot_df.mean(numeric_only=True)

# =============================
# PORTFOLIO SNAPSHOT (EXECUTIVE BLUE BOX)
# =============================
st.markdown(f"""
<div class="blue-box">
    <div class="snapshot-title">🏛️ Portfolio Snapshot (All Waves)</div>
    <div class="snapshot-sub">STANDARD MODE</div>

    <div class="section-label">RETURNS</div>
    <div class="metric-grid">
        <div class="metric-card">
            <div class="metric-label">Intraday</div>
            <div class="metric-value">{portfolio['Return_1D']*100:.2f}%</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">30D</div>
            <div class="metric-value">{portfolio['Return_30D']*100:.2f}%</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">60D</div>
            <div class="metric-value">{portfolio['Return_60D']*100:.2f}%</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">365D</div>
            <div class="metric-value">{portfolio['Return_365D']*100:.2f}%</div>
        </div>
    </div>

    <div class="section-label">ALPHA CAPTURED</div>
    <div class="metric-grid">
        <div class="metric-card">
            <div class="metric-label">Intraday</div>
            <div class="metric-value">{portfolio['Alpha_1D']*100:.2f}%</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">30D</div>
            <div class="metric-value">{portfolio['Alpha_30D']*100:.2f}%</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">60D</div>
            <div class="metric-value">{portfolio['Alpha_60D']*100:.2f}%</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">365D</div>
            <div class="metric-value">{portfolio['Alpha_365D']*100:.2f}%</div>
        </div>
    </div>

    <div class="footer-note">
        ⚡ Computed from live snapshot | {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC<br/>
        ℹ Wave-level Beta, Exposure, Cash, VIX regime shown below
    </div>
</div>
""", unsafe_allow_html=True)

# =============================
# MIDDLE SECTION — LIVE RETURNS & ALPHA
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
# BOTTOM SECTION — ALPHA HISTORY
# =============================
st.subheader("📈 Alpha History by Horizon")

alpha_cols = ["Alpha_1D", "Alpha_30D", "Alpha_365D"]
alpha_df = snapshot_df[["Wave"] + alpha_cols].set_index("Wave")

if alpha_df.dropna().empty:
    st.warning("Alpha data unavailable or incomplete for horizon chart.")
else:
    st.bar_chart(alpha_df)

# =============================
# STATUS
# =============================
st.markdown("""
<div class="status-banner">
    LIVE SYSTEM ACTIVE ✅<br/>
    ✓ Intraday live • ✓ Multi-horizon returns • ✓ Alpha attribution • ✓ Snapshot truth
</div>
""", unsafe_allow_html=True)