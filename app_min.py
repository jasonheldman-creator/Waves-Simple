# redeploy trigger

# app_min.py
import streamlit as st
import pandas as pd
from datetime import datetime
from pathlib import Path

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
# SAFE DATA LOADERS
# ======================================================
@st.cache_data(show_spinner=False)
def safe_read_csv(path: str) -> pd.DataFrame | None:
    p = Path(path)
    if not p.exists():
        return None
    try:
        return pd.read_csv(p)
    except Exception:
        return None

# ======================================================
# LOAD DATA
# ======================================================
df = safe_read_csv("data/live_snapshot.csv")
attr_summary = safe_read_csv("data/alpha_attribution_summary.csv")

# 🔍 DIAGNOSTIC — DO NOT REMOVE YET
st.write(
    "DEBUG — alpha_attribution_summary.csv exists:",
    Path("data/alpha_attribution_summary.csv").exists()
)

if df is None:
    st.error("❌ live_snapshot.csv not found — system cannot start.")
    st.stop()

portfolio = df.mean(numeric_only=True)
wave_list = sorted(df["Wave"].unique())

# ======================================================
# ATTRIBUTION COLUMN MAP (LOCKED TO REAL ENGINE)
# ======================================================
ATTR_MAP = {
    "Exposure & Timing": (
        "exposure_timing_alpha",
        "exposure_timing_contribution_pct"
    ),
    "Regime & VIX Overlay": (
        "regime_vix_alpha",
        "regime_vix_contribution_pct"
    ),
    "Momentum & Trend": (
        "momentum_trend_alpha",
        "momentum_trend_contribution_pct"
    ),
    "Volatility & Risk Control": (
        "volatility_control_alpha",
        "volatility_control_contribution_pct"
    ),
    "Asset Selection (Residual)": (
        "asset_selection_alpha",
        "asset_selection_contribution_pct"
    ),
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
# ALPHA ATTRIBUTION TAB (SAFE + REAL)
# ======================================================
with tab_attr:

    st.subheader("⚡ Alpha Attribution Breakdown (365D)")

    if attr_summary is None:
        st.warning("⏳ Alpha attribution data not yet available. Awaiting next build.")
        st.stop()

    if attribution_scope == "Portfolio":
        scope_df = attr_summary
        scope_label = "Portfolio — All Waves"
    else:
        scope_df = attr_summary[attr_summary["wave_name"] == selected_wave]
        scope_label = selected_wave

    if scope_df.empty:
        st.warning("No attribution data available for the selected scope.")
        st.stop()

    total_alpha = scope_df["total_alpha"].mean()

    rows = []
    for label, (alpha_col, pct_col) in ATTR_MAP.items():
        if alpha_col in scope_df.columns and pct_col in scope_df.columns:
            rows.append({
                "Component": label,
                "Alpha Contribution": scope_df[alpha_col].mean(),
                "% of Total Alpha": scope_df[pct_col].mean()
            })

    attr_df = pd.DataFrame(rows).set_index("Component")

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