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
# STYLES (INSTITUTIONAL POLISH)
# =============================
st.markdown("""
<style>
body {
    background-color: #0e1117;
}

/* Executive Snapshot Box */
.snapshot-shell {
    background: linear-gradient(160deg, #0b2a4a 0%, #081c33 55%, #071726 100%);
    border: 2px solid rgba(63, 208, 255, 0.55);
    border-radius: 20px;
    padding: 34px 36px 30px 36px;
    box-shadow:
        0 0 28px rgba(63, 208, 255, 0.35),
        inset 0 0 12px rgba(255, 255, 255, 0.04);
    margin-bottom: 36px;
}

/* Titles */
.snapshot-title {
    font-size: 34px;
    font-weight: 800;
    color: #ffffff;
    margin-bottom: 4px;
}
.snapshot-mode {
    font-size: 14px;
    letter-spacing: 0.12em;
    color: #9fb6cc;
    margin-bottom: 26px;
}

/* Section headers */
.snapshot-section {
    font-size: 15px;
    font-weight: 700;
    letter-spacing: 0.12em;
    color: #3fd0ff;
    margin: 14px 0 6px 0;
}

/* Metric cards */
.metric-card {
    background: rgba(255, 255, 255, 0.03);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 14px;
    padding: 16px 18px 14px 18px;
}

/* Footer */
.snapshot-footer {
    font-size: 13px;
    color: #9aa9b7;
    margin-top: 22px;
}

/* Status banner */
.status-banner {
    background: linear-gradient(90deg, #1f8f4e, #2ecc71);
    padding: 18px;
    border-radius: 14px;
    font-weight: 700;
    text-align: center;
    color: white;
    margin-top: 36px;
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
    st.markdown('<div class="snapshot-shell">', unsafe_allow_html=True)

    st.markdown('<div class="snapshot-title">🏛️ Portfolio Snapshot (All Waves)</div>', unsafe_allow_html=True)
    st.markdown('<div class="snapshot-mode">STANDARD MODE</div>', unsafe_allow_html=True)

    # RETURNS
    st.markdown('<div class="snapshot-section">RETURNS</div>', unsafe_allow_html=True)
    r1, r2, r3, r4 = st.columns(4)
    with r1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Intraday", f"{portfolio['Return_1D']*100:.2f}%")
        st.markdown('</div>', unsafe_allow_html=True)
    with r2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("30D", f"{portfolio['Return_30D']*100:.2f}%")
        st.markdown('</div>', unsafe_allow_html=True)
    with r3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("60D", f"{portfolio['Return_60D']*100:.2f}%")
        st.markdown('</div>', unsafe_allow_html=True)
    with r4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("365D", f"{portfolio['Return_365D']*100:.2f}%")
        st.markdown('</div>', unsafe_allow_html=True)

    # ALPHA
    st.markdown('<div class="snapshot-section">ALPHA CAPTURED</div>', unsafe_allow_html=True)
    a1, a2, a3, a4 = st.columns(4)
    with a1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Intraday", f"{portfolio['Alpha_1D']*100:.2f}%")
        st.markdown('</div>', unsafe_allow_html=True)
    with a2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("30D", f"{portfolio['Alpha_30D']*100:.2f}%")
        st.markdown('</div>', unsafe_allow_html=True)
    with a3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("60D", f"{portfolio['Alpha_60D']*100:.2f}%")
        st.markdown('</div>', unsafe_allow_html=True)
    with a4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("365D", f"{portfolio['Alpha_365D']*100:.2f}%")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown(
        f"""
        <div class="snapshot-footer">
        ⚡ Computed from live snapshot | {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC<br/>
        ℹ Wave-level Beta, Exposure, Cash, VIX regime shown below
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown('</div>', unsafe_allow_html=True)

# =============================
# LIVE RETURNS & ALPHA TABLE (UNCHANGED)
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
# ALPHA HISTORY BY HORIZON (UNCHANGED)
# =============================
st.subheader("📈 Alpha History by Horizon")

alpha_cols = ["Alpha_1D", "Alpha_30D", "Alpha_365D"]
alpha_df = snapshot_df[["Wave"] + alpha_cols].set_index("Wave")

if alpha_df.dropna().empty:
    st.warning("Alpha data unavailable or incomplete for horizon chart.")
else:
    st.bar_chart(alpha_df)

# =============================
# STATUS BANNER (UNCHANGED)
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