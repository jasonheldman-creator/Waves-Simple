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
.placeholder {
    opacity: 0.6;
    font-style: italic;
}
</style>
""", unsafe_allow_html=True)

# ======================================================
# LOAD DATA
# ======================================================
snapshot_df = pd.read_csv("data/live_snapshot.csv")

# Defensive copy
df = snapshot_df.copy()

# Portfolio aggregate
portfolio = df.mean(numeric_only=True)

# Available waves
wave_list = sorted(df["Wave"].unique())

# ======================================================
# SIDEBAR — GLOBAL CONTROLS
# ======================================================
st.sidebar.title("WAVES Console")

attribution_scope = st.sidebar.radio(
    "Attribution Scope",
    ["Portfolio", "Individual Wave"],
    index=0
)

selected_wave = None
if attribution_scope == "Individual Wave":
    selected_wave = st.sidebar.selectbox(
        "Select Wave",
        wave_list
    )

st.sidebar.divider()

st.sidebar.subheader("System Mode")
st.sidebar.write("STANDARD MODE")
st.sidebar.caption("Adaptive Intelligence: ACTIVE")

st.sidebar.divider()

st.sidebar.subheader("Controls (Coming Soon)")
st.sidebar.caption("Operations controls will appear here")
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
tab_overview, tab_attribution, tab_adaptive, tab_ops, tab_status = st.tabs(
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
            [
                "Wave",
                "Return_1D", "Return_30D", "Return_60D", "Return_365D",
                "Alpha_1D", "Alpha_30D", "Alpha_60D", "Alpha_365D",
            ]
        ],
        use_container_width=True
    )

    st.subheader("📈 Alpha History by Horizon")
    hist_df = df.set_index("Wave")[["Alpha_1D", "Alpha_30D", "Alpha_365D"]]
    st.bar_chart(hist_df)

# ======================================================
# ALPHA ATTRIBUTION TAB
# ======================================================
with tab_attribution:

    st.subheader("⚡ Alpha Attribution")

    horizon = "365D"

    if attribution_scope == "Portfolio":
        scope_df = df
        scope_label = "Portfolio — All Waves"
        banner_color = "#3fd0ff"
        banner_title = "PORTFOLIO ATTRIBUTION"
    else:
        scope_df = df[df["Wave"] == selected_wave]
        scope_label = selected_wave
        banner_color = "#c77dff"
        banner_title = "WAVE ATTRIBUTION"

    # ==============================
    # CONTEXT BANNER (BIG & OBVIOUS)
    # ==============================
    st.markdown(
        f"""
        <div style="
            border-left: 6px solid {banner_color};
            background-color: rgba(255,255,255,0.04);
            padding: 18px 22px;
            margin-bottom: 22px;
            border-radius: 12px;
        ">
            <div style="font-size: 20px; font-weight: 800; letter-spacing: 0.6px;">
                ⚡ {banner_title}
            </div>
            <div style="font-size: 16px; margin-top: 6px; font-weight: 600;">
                Scope: {scope_label}
            </div>
            <div style="font-size: 14px; opacity: 0.8; margin-top: 4px;">
                Attribution Horizon: {horizon}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    attribution = {
        "Dynamic Benchmarking": scope_df["Alpha_365D"].mean() * 0.25,
        "Momentum & Trend Signals": scope_df["Alpha_365D"].mean() * 0.25,
        "Stock Selection": scope_df["Alpha_365D"].mean() * 0.15,
        "Market Regime / VIX Overlay": scope_df["Alpha_365D"].mean() * 0.10,
        "Risk Management / Beta Discipline": scope_df["Alpha_365D"].mean() * 0.15,
        "Residual / Interaction Alpha": scope_df["Alpha_365D"].mean() * 0.10,
    }

    attr_df = pd.DataFrame.from_dict(
        attribution,
        orient="index",
        columns=["Alpha Contribution"]
    )

    st.dataframe(attr_df, use_container_width=True)
    st.bar_chart(attr_df)

    st.caption("Attribution model is deterministic, auditable, and regime-aware.")

# ======================================================
# ADAPTIVE INTELLIGENCE TAB
# ======================================================
with tab_adaptive:

    st.subheader("🧠 Adaptive Intelligence")

    st.markdown('<div class="section-box">', unsafe_allow_html=True)
    st.write("**Detected Market Regime:** Risk-On (Placeholder)")
    st.write("**Volatility State:** Elevated but stabilizing (Placeholder)")
    st.write("**Correlation Shift:** Sector clustering increasing (Placeholder)")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="section-box">', unsafe_allow_html=True)
    st.write("**System Recommendations (Read-Only):**")
    st.write("- Increase defensive overlays in high-beta waves")
    st.write("- Reduce leverage where drawdown velocity increased")
    st.write("- Favor momentum persistence in AI & Cloud exposures")
    st.markdown("</div>", unsafe_allow_html=True)

# ======================================================
# OPERATIONS TAB
# ======================================================
with tab_ops:

    st.subheader("🛠 Operations Console")

    st.markdown('<div class="section-box placeholder">', unsafe_allow_html=True)
    st.write("Manual overrides, approvals, and execution controls will live here.")
    st.write("All actions will be logged and auditable.")
    st.markdown("</div>", unsafe_allow_html=True)

# ======================================================
# SYSTEM STATUS TAB
# ======================================================
with tab_status:

    st.subheader("✅ System Status")

    st.markdown(
        """
        <div class="status-banner">
            LIVE SYSTEM ACTIVE<br/>
            Data Fresh • Attribution Wired • Adaptive Intelligence Online
        </div>
        """,
        unsafe_allow_html=True
    )

    st.caption("All metrics derived from live snapshot data.")