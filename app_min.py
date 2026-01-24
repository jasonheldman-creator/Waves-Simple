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
.snapshot-box {
    border: 2px solid #3fd0ff;
    border-radius: 18px;
    padding: 28px;
    background: linear-gradient(145deg, #0b2a4a, #081c33);
    box-shadow: 0 0 25px rgba(63, 208, 255, 0.35);
    margin-bottom: 30px;
}
.snapshot-title {
    font-size: 34px;
    font-weight: 800;
    color: white;
}
.snapshot-sub {
    font-size: 16px;
    opacity: 0.75;
    margin-bottom: 25px;
}
.metric-label {
    font-size: 14px;
    opacity: 0.7;
}
.metric-value {
    font-size: 30px;
    font-weight: 800;
}
.footer-note {
    margin-top: 20px;
    font-size: 13px;
    opacity: 0.7;
}
.status-banner {
    background: linear-gradient(90deg, #1f8f4e, #2ecc71);
    padding: 18px;
    border-radius: 12px;
    font-weight: 700;
    text-align: center;
    color: white;
    margin-top: 30px;
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
# PORTFOLIO SNAPSHOT (BLUE BOX)
# =============================
with st.container():
    st.markdown('<div class="snapshot-box">', unsafe_allow_html=True)

    st.markdown(
        '<div class="snapshot-title">🏛️ Portfolio Snapshot (All Waves)</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<div class="snapshot-sub">STANDARD MODE</div>',
        unsafe_allow_html=True
    )

    # --- RETURNS ROW ---
    r1, r2, r3, r4 = st.columns(4)
    r1.metric("Intraday Return", f"{portfolio['Return_1D']*100:.2f}%")
    r2.metric("30D Return", f"{portfolio['Return_30D']*100:.2f}%")
    r3.metric("60D Return", f"{portfolio['Return_60D']*100:.2f}%")
    r4.metric("365D Return", f"{portfolio['Return_365D']*100:.2f}%")

    st.markdown("---")

    # --- ALPHA ROW ---
    a1, a2, a3, a4 = st.columns(4)
    a1.metric("Alpha Intraday", f"{portfolio['Alpha_1D']*100:.2f}%")
    a2.metric("Alpha 30D", f"{portfolio['Alpha_30D']*100:.2f}%")
    a3.metric("Alpha 60D", f"{portfolio['Alpha_60D']*100:.2f}%")
    a4.metric("Alpha 365D", f"{portfolio['Alpha_365D']*100:.2f}%")

    st.markdown(
        f"""
        <div class="footer-note">
        ⚡ Computed from live snapshot | {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC<br/>
        ℹ Wave-level Beta, Exposure, Cash, VIX regime shown below
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("</div>", unsafe_allow_html=True)

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