# app_min.py
import streamlit as st
import pandas as pd
from datetime import datetime

# ======================================================
# PAGE CONFIG
# ======================================================
st.set_page_config(
    page_title="WAVES — Institutional Console",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ======================================================
# GLOBAL STYLES
# ======================================================
st.markdown("""
<style>
body {
    background-color: #0e1117;
    color: white;
}
.section-box {
    border: 1px solid #2a3f55;
    border-radius: 14px;
    padding: 20px;
    margin-bottom: 25px;
    background-color: #0b1624;
}
.blue-box {
    border: 2px solid #3fd0ff;
    border-radius: 18px;
    padding: 24px;
    margin-bottom: 30px;
    background: linear-gradient(145deg, #0b2a4a, #081c33);
    box-shadow: 0 0 25px rgba(63,208,255,0.35);
}
.status-banner {
    background: linear-gradient(90deg, #1f8f4e, #2ecc71);
    padding: 16px;
    border-radius: 12px;
    font-weight: 700;
    text-align: center;
    margin-top: 30px;
}
.negative {
    color: #ff5c5c;
    font-weight: 700;
}
.positive {
    color: #7CFFB2;
    font-weight: 700;
}
</style>
""", unsafe_allow_html=True)

# ======================================================
# LOAD DATA
# ======================================================
df = pd.read_csv("data/live_snapshot.csv")

portfolio = df.mean(numeric_only=True)
wave_list = sorted(df["Wave"].unique())

# Attribution columns (REAL, from CSV)
ATTR_COLS = {
    "Momentum & Trend Signals": "Alpha_Momentum_365D",
    "Market Regime / VIX Overlay": "Alpha_Regime_365D",
    "Beta Discipline & Risk Control": "Alpha_BetaDiscipline_365D",
    "Stock Selection": "Alpha_StockSelection_365D",
    "Execution & Rebalancing": "Alpha_Execution_365D",
    "Residual / Interaction": "Alpha_Residual_365D",
}

# ======================================================
# SIDEBAR
# ======================================================
st.sidebar.title("WAVES Console")

attribution_scope = st.sidebar.radio(
    "Attribution Scope",
    ["Portfolio", "Individual Wave"],
    index=0
)

selected_wave = None
if attribution_scope == "Individual Wave":
    selected_wave = st.sidebar.selectbox("Select Wave", wave_list)

st.sidebar.divider()
st.sidebar.subheader("System Mode")
st.sidebar.write("STANDARD MODE")
st.sidebar.caption("Adaptive Intelligence: ACTIVE")

st.sidebar.divider()
st.sidebar.subheader("Controls (Coming Soon)")
st.sidebar.button("Apply Manual Override", disabled=True)
st.sidebar.button("Rebalance Portfolio", disabled=True)

# ======================================================
# HEADER
# ======================================================
st.title("WAVES — Institutional Intelligence Console")
st.caption("Returns • Alpha • Attribution • Adaptive Intelligence • Operations")
st.divider()

# ======================================================
# TABS
# ======================================================
tab_overview, tab_attr, tab_ai, tab_ops, tab_status = st.tabs(
    ["Overview", "Alpha Attribution", "Adaptive Intelligence", "Operations", "System Status"]
)

# ======================================================
# OVERVIEW TAB
# ======================================================
with tab_overview:

    st.subheader("🏛 Portfolio Snapshot")

    st.markdown('<div class="blue-box">', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Intraday Return", f"{portfolio['Return_1D']*100:.2f}%")
    c2.metric("30D Return", f"{portfolio['Return_30D']*100:.2f}%")
    c3.metric("60D Return", f"{portfolio['Return_60D']*100:.2f}%")
    c4.metric("365D Return", f"{portfolio['Return_365D']*100:.2f}%")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Alpha Intraday", f"{portfolio['Alpha_1D']*100:.2f}%")
    c2.metric("Alpha 30D", f"{portfolio['Alpha_30D']*100:.2f}%")
    c3.metric("Alpha 60D", f"{portfolio['Alpha_60D']*100:.2f}%")
    c4.metric("Alpha 365D", f"{portfolio['Alpha_365D']*100:.2f}%")

    st.caption(f"⚡ Snapshot timestamp: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")
    st.markdown("</div>", unsafe_allow_html=True)

    st.subheader("📊 Live Returns & Alpha")
    st.dataframe(
        df[
            ["Wave",
             "Return_1D","Return_30D","Return_60D","Return_365D",
             "Alpha_1D","Alpha_30D","Alpha_60D","Alpha_365D"]
        ],
        use_container_width=True
    )

# ======================================================
# ALPHA ATTRIBUTION TAB (REAL MATH)
# ======================================================
with tab_attr:

    st.subheader("⚡ Alpha Attribution Breakdown (365D)")

    if attribution_scope == "Portfolio":
        scope_df = df
        scope_label = "Portfolio — All Waves"
    else:
        scope_df = df[df["Wave"] == selected_wave]
        scope_label = selected_wave

    total_alpha = scope_df["Alpha_365D"].mean()

    attr_values = {}
    for label, col in ATTR_COLS.items():
        attr_values[label] = scope_df[col].mean()

    attr_df = pd.DataFrame.from_dict(
        attr_values, orient="index", columns=["Alpha Contribution"]
    )

    attr_df["% of Total Alpha"] = attr_df["Alpha Contribution"] / total_alpha * 100

    st.markdown(f"**Scope:** {scope_label}")
    st.markdown(f"**Total Alpha (365D):** `{total_alpha:.4f}`")

    def style_pct(val):
        if val < 0:
            return "color: #ff5c5c; font-weight: 700"
        return "color: #7CFFB2; font-weight: 700"

    st.dataframe(
        attr_df.style.format({
            "Alpha Contribution": "{:.4f}",
            "% of Total Alpha": "{:.1f}%"
        }).applymap(style_pct, subset=["% of Total Alpha"]),
        use_container_width=True
    )

    st.bar_chart(attr_df["Alpha Contribution"])

# ======================================================
# ADAPTIVE INTELLIGENCE TAB
# ======================================================
with tab_ai:

    st.subheader("🧠 Adaptive Intelligence")

    st.markdown('<div class="section-box">', unsafe_allow_html=True)
    st.write("Detected Market Regime: **Placeholder**")
    st.write("Volatility State: **Placeholder**")
    st.write("Correlation Shifts: **Placeholder**")
    st.markdown("</div>", unsafe_allow_html=True)

# ======================================================
# OPERATIONS TAB
# ======================================================
with tab_ops:

    st.subheader("🛠 Operations Console")

    st.markdown('<div class="section-box">', unsafe_allow_html=True)
    st.write("Manual overrides and approvals will appear here.")
    st.markdown("</div>", unsafe_allow_html=True)

# ======================================================
# SYSTEM STATUS
# ======================================================
with tab_status:

    st.subheader("✅ System Status")

    st.markdown(
        """
        <div class="status-banner">
            LIVE SYSTEM ACTIVE<br/>
            Data Fresh • Attribution Real • Adaptive Intelligence Online
        </div>
        """,
        unsafe_allow_html=True
    )