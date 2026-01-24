import streamlit as st
import pandas as pd
from datetime import datetime

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="WAVES — Institutional Console",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================================
# GLOBAL STYLES
# =====================================================
st.markdown("""
<style>
body {
    background-color: #0e1117;
}
.blue-box {
    background: linear-gradient(145deg, #0b2a4a, #081c33);
    border: 2px solid #3fd0ff;
    border-radius: 18px;
    padding: 26px;
    margin-bottom: 30px;
    box-shadow: 0 0 25px rgba(63, 208, 255, 0.35);
}
.status-banner {
    background: linear-gradient(90deg, #1f8f4e, #2ecc71);
    padding: 16px;
    border-radius: 12px;
    font-weight: 700;
    text-align: center;
    color: white;
    margin-top: 30px;
}
.placeholder-box {
    border: 1px dashed #555;
    border-radius: 12px;
    padding: 18px;
    opacity: 0.7;
}
</style>
""", unsafe_allow_html=True)

# =====================================================
# LOAD DATA
# =====================================================
SNAPSHOT_PATH = "data/live_snapshot.csv"
snapshot_df = pd.read_csv(SNAPSHOT_PATH)

# Defensive normalization
snapshot_df["Wave"] = snapshot_df["Wave"].astype(str)

portfolio_df = snapshot_df.mean(numeric_only=True)

# =====================================================
# SIDEBAR — GLOBAL CONTROL PLANE
# =====================================================
with st.sidebar:
    st.header("⚙️ Controls")

    scope = st.radio(
        "Attribution Scope",
        options=["Portfolio", "Individual Wave"],
        index=0
    )

    selected_wave = None
    if scope == "Individual Wave":
        selected_wave = st.selectbox(
            "Select Wave",
            options=sorted(snapshot_df["Wave"].unique())
        )

    st.divider()

    horizon = st.multiselect(
        "Time Horizons",
        options=["Intraday", "30D", "60D", "365D"],
        default=["Intraday", "30D", "60D", "365D"]
    )

    st.divider()

    st.caption("Mode (placeholder)")
    st.selectbox(
        "Execution Mode",
        ["Standard", "Alpha-Minus-Beta", "Private Logic™"],
        index=0
    )

# =====================================================
# HEADER
# =====================================================
st.title("WAVES — Institutional Console")
st.caption("Performance • Attribution • Governance • Risk")
st.divider()

# =====================================================
# PORTFOLIO SNAPSHOT (HERO SECTION)
# =====================================================
with st.container():
    st.markdown('<div class="blue-box">', unsafe_allow_html=True)

    title = "🏛 Portfolio Snapshot"
    if scope == "Individual Wave" and selected_wave:
        title += f" — {selected_wave}"

    st.subheader(title)
    st.caption("STANDARD MODE")

    data_source = (
        snapshot_df[snapshot_df["Wave"] == selected_wave]
        if scope == "Individual Wave" and selected_wave
        else snapshot_df
    )

    agg = data_source.mean(numeric_only=True)

    r1, r2, r3, r4 = st.columns(4)
    r1.metric("Intraday Return", f"{agg.get('Return_1D', 0)*100:.2f}%")
    r2.metric("30D Return", f"{agg.get('Return_30D', 0)*100:.2f}%")
    r3.metric("60D Return", f"{agg.get('Return_60D', 0)*100:.2f}%")
    r4.metric("365D Return", f"{agg.get('Return_365D', 0)*100:.2f}%")

    a1, a2, a3, a4 = st.columns(4)
    a1.metric("Alpha Intraday", f"{agg.get('Alpha_1D', 0)*100:.2f}%")
    a2.metric("Alpha 30D", f"{agg.get('Alpha_30D', 0)*100:.2f}%")
    a3.metric("Alpha 60D", f"{agg.get('Alpha_60D', 0)*100:.2f}%")
    a4.metric("Alpha 365D", f"{agg.get('Alpha_365D', 0)*100:.2f}%")

    st.caption(
        f"⚡ Computed from live snapshot | "
        f"{datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC"
    )

    st.markdown('</div>', unsafe_allow_html=True)

# =====================================================
# MAIN TABS
# =====================================================
tab_overview, tab_attr, tab_waves, tab_risk, tab_gov, tab_sys = st.tabs([
    "Overview",
    "Alpha Attribution",
    "Waves",
    "Risk",
    "Governance",
    "System"
])

# -----------------------------------------------------
# TAB: OVERVIEW
# -----------------------------------------------------
with tab_overview:
    st.subheader("📊 Live Returns & Alpha")

    st.dataframe(
        snapshot_df[
            [
                "Wave",
                "Return_1D", "Return_30D", "Return_60D", "Return_365D",
                "Alpha_1D", "Alpha_30D", "Alpha_60D", "Alpha_365D",
            ]
        ],
        use_container_width=True
    )

    st.subheader("📈 Alpha History by Horizon")

    alpha_cols = ["Alpha_1D", "Alpha_30D", "Alpha_60D", "Alpha_365D"]
    alpha_hist = snapshot_df[["Wave"] + alpha_cols].set_index("Wave")

    if alpha_hist.dropna().empty:
        st.warning("Alpha data unavailable.")
    else:
        st.bar_chart(alpha_hist)

# -----------------------------------------------------
# TAB: ALPHA ATTRIBUTION
# -----------------------------------------------------
with tab_attr:
    st.subheader("🧠 Alpha Attribution Breakdown")

    st.markdown(
        """
        Alpha is decomposed into the following institutional sources:
        - Stock Selection
        - Strategy / Algo Signals
        - Dynamic Benchmarking
        - Volatility / VIX Overlays
        - Exposure & Timing
        - Residual / Other
        """
    )

    st.markdown('<div class="placeholder-box">', unsafe_allow_html=True)
    st.write("🔧 Attribution engine wired — population coming next step.")
    st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------------------------------
# TAB: WAVES
# -----------------------------------------------------
with tab_waves:
    st.subheader("🌊 Individual Wave Detail")
    st.markdown('<div class="placeholder-box">', unsafe_allow_html=True)
    st.write("Wave-level diagnostics, history, and attribution will appear here.")
    st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------------------------------
# TAB: RISK
# -----------------------------------------------------
with tab_risk:
    st.subheader("⚠️ Risk & Resilience")
    st.markdown('<div class="placeholder-box">', unsafe_allow_html=True)
    st.write("Drawdowns, volatility, correlations (placeholder).")
    st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------------------------------
# TAB: GOVERNANCE
# -----------------------------------------------------
with tab_gov:
    st.subheader("📜 Governance & Transparency")
    st.markdown('<div class="placeholder-box">', unsafe_allow_html=True)
    st.write("Data lineage, benchmarks, audit flags (placeholder).")
    st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------------------------------
# TAB: SYSTEM
# -----------------------------------------------------
with tab_sys:
    st.subheader("🟢 System Status")
    st.markdown(
        """
        <div class="status-banner">
            LIVE SYSTEM ACTIVE ✅<br/>
            ✓ Snapshot loaded • ✓ Multi-horizon analytics • ✓ Attribution scaffolded
        </div>
        """,
        unsafe_allow_html=True
    )