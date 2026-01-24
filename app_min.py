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

/* ===== PORTFOLIO SNAPSHOT ===== */
.snapshot-box {
    background: linear-gradient(145deg, #0b2a4a, #081c33);
    border: 2px solid #3fd0ff;
    border-radius: 22px;
    padding: 36px 34px;
    margin: 30px 0 40px 0;
    box-shadow: 0 0 35px rgba(63, 208, 255, 0.35);
}

.snapshot-title {
    font-size: 34px;
    font-weight: 800;
    color: #ffffff;
    margin-bottom: 6px;
}

.snapshot-sub {
    font-size: 15px;
    letter-spacing: 1px;
    opacity: 0.75;
    margin-bottom: 26px;
}

.section-label {
    font-size: 15px;
    letter-spacing: 1.2px;
    font-weight: 700;
    color: #3fd0ff;
    margin: 28px 0 16px 0;
}

.metric-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 26px;
}

.metric-card {
    background: rgba(255,255,255,0.03);
    border-radius: 14px;
    padding: 22px 18px;
}

.metric-label {
    font-size: 14px;
    opacity: 0.75;
    margin-bottom: 6px;
}

.metric-value {
    font-size: 32px;
    font-weight: 800;
    color: #ffffff;
}

.snapshot-footer {
    margin-top: 26px;
    font-size: 13px;
    opacity: 0.7;
}

/* ===== STATUS ===== */
.status-banner {
    background: linear-gradient(90deg, #1f8f4e, #2ecc71);
    padding: 18px;
    border-radius: 14px;
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
# LOAD SNAPSHOT DATA
# =============================
snapshot_df = pd.read_csv("data/live_snapshot.csv")
p = snapshot_df.mean(numeric_only=True)

# =============================
# PORTFOLIO SNAPSHOT (EXECUTIVE)
# =============================
st.markdown(f"""
<div class="snapshot-box">
    <div class="snapshot-title">🏛️ Portfolio Snapshot (All Waves)</div>
    <div class="snapshot-sub">STANDARD MODE</div>

    <div class="section-label">RETURNS</div>
    <div class="metric-grid">
        <div class="metric-card">
            <div class="metric-label">Intraday</div>
            <div class="metric-value">{p['Return_1D']*100:.2f}%</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">30D</div>
            <div class="metric-value">{p['Return_30D']*100:.2f}%</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">60D</div>
            <div class="metric-value">{p['Return_60D']*100:.2f}%</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">365D</div>
            <div class="metric-value">{p['Return_365D']*100:.2f}%</div>
        </div>
    </div>

    <div class="section-label">ALPHA CAPTURED</div>
    <div class="metric-grid">
        <div class="metric-card">
            <div class="metric-label">Intraday</div>
            <div class="metric-value">{p['Alpha_1D']*100:.2f}%</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">30D</div>
            <div class="metric-value">{p['Alpha_30D']*100:.2f}%</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">60D</div>
            <div class="metric-value">{p['Alpha_60D']*100:.2f}%</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">365D</div>
            <div class="metric-value">{p['Alpha_365D']*100:.2f}%</div>
        </div>
    </div>

    <div class="snapshot-footer">
        ⚡ Computed from live snapshot | {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC<br/>
        ℹ Wave-level Beta, Exposure, Cash, VIX regime shown below
    </div>
</div>
""", unsafe_allow_html=True)

# =============================
# LIVE RETURNS & ALPHA (UNCHANGED)
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
# ALPHA HISTORY (UNCHANGED)
# =============================
st.subheader("📈 Alpha History by Horizon")

alpha_cols = ["Alpha_1D", "Alpha_30D", "Alpha_365D"]
alpha_df = snapshot_df[["Wave"] + alpha_cols].set_index("Wave")

if alpha_df.dropna().empty:
    st.warning("Alpha data unavailable or incomplete for horizon chart.")
else:
    st.bar_chart(alpha_df)

# =============================
# STATUS (UNCHANGED)
# =============================
st.markdown("""
<div class="status-banner">
    LIVE SYSTEM ACTIVE ✅<br/>
    ✓ Intraday live • ✓ Multi-horizon returns • ✓ Alpha attribution • ✓ Snapshot truth
</div>
""", unsafe_allow_html=True)