import streamlit as st
import pandas as pd
import numpy as np
import os
import plotly.express as px
import datetime
from waves_engine import compute_all_waves  # must match your engine filename

st.set_page_config(
    page_title="WAVES Intelligence™ Institutional Console",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------------------------------------------------
# HELPER — Load CSV files if present
# ------------------------------------------------------------
def load_csv_if_exists(path):
    if os.path.exists(path):
        return pd.read_csv(path)
    return None

# ------------------------------------------------------------
# LOAD WAVE WEIGHTS + LIST
# ------------------------------------------------------------
weights_df = load_csv_if_exists("wave_weights.csv")
list_df = load_csv_if_exists("list.csv")

if weights_df is None or list_df is None:
    st.error("❌ Missing required files: wave_weights.csv or list.csv")
    st.stop()

# ------------------------------------------------------------
# DYNAMIC WAVE DISCOVERY
# ------------------------------------------------------------
AVAILABLE_WAVES = sorted(weights_df["wave"].unique().tolist())

MODES = {
    "Standard": "standard",
    "Alpha-Minus-Beta": "amb",
    "Private Logic™": "pl"
}

# ------------------------------------------------------------
# RUN ENGINE (SmartSafe included)
# ------------------------------------------------------------
@st.cache_data(ttl=300)
def run_engine():
    return compute_all_waves(weights_df)

all_wave_dfs = run_engine()

# ------------------------------------------------------------
# SIDEBAR CONTROLS
# ------------------------------------------------------------
st.sidebar.title("⚡ WAVES Intelligence™")
sel_wave = st.sidebar.selectbox("Select Wave", AVAILABLE_WAVES)
sel_mode_label = st.sidebar.selectbox("Select Mode", list(MODES.keys()))
sel_mode = MODES[sel_mode_label]

# Extract the selected wave’s dataframe
wave_df = all_wave_dfs.get(sel_wave)

if wave_df is None or len(wave_df) == 0:
    st.error("❌ No data available for this Wave.")
    st.stop()

# ------------------------------------------------------------
# TITLE
# ------------------------------------------------------------
st.markdown(
    f"""
    <h1 style='color:#6EE7FF;'>
        {sel_wave} — {sel_mode_label}
    </h1>
    """,
    unsafe_allow_html=True
)

# ------------------------------------------------------------
# TOP METRICS DISPLAY
# ------------------------------------------------------------
def fmt(x):
    if pd.isna(x): return "N/A"
    return f"{x*100:.2f}%"

latest = wave_df.iloc[-1]

col1, col2, col3, col4 = st.columns(4)
col1.metric("1-Day Return", fmt(latest.get("return_1d")))
col2.metric("30-Day Return", fmt(latest.get("return_30d")))
col3.metric("60-Day Return", fmt(latest.get("return_60d")))
col4.metric("1-Year Return", fmt(latest.get("return_252d")))

col5, col6, col7, col8 = st.columns(4)
col5.metric("Alpha 1-Day", fmt(latest.get("alpha_1d")))
col6.metric("Alpha 30-Day", fmt(latest.get("alpha_30d")))
col7.metric("Alpha 60-Day", fmt(latest.get("alpha_60d")))
col8.metric("Alpha 1-Year", fmt(latest.get("alpha_1y")))

# ------------------------------------------------------------
# TABS
# ------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "Wave Details",
    "Alpha Capture",
    "WaveScore",
    "System Status"
])

# ------------------------------------------------------------
# TAB 1 — WAVE DETAILS
# ------------------------------------------------------------
with tab1:
    st.subheader(f"{sel_wave} — Detailed Positions")

    # Top 10 holdings with Google links
    try:
        wdf = weights_df[weights_df["wave"] == sel_wave].copy()
        wdf = wdf.sort_values("weight", ascending=False).head(10)

        wdf["google_link"] = wdf["ticker"].apply(
            lambda t: f"https://www.google.com/finance/quote/{t}"
        )

        st.write("### Top 10 Holdings")
        st.dataframe(
            wdf[["ticker", "weight", "google_link"]],
            use_container_width=True
        )
    except Exception as e:
        st.warning(f"Could not load top holdings: {str(e)}")

    st.write("---")

    st.subheader("NAV Chart (SmartSafe Included)")
    try:
        fig = px.line(
            wave_df,
            x="date",
            y=["nav", "nav_risk", "bench_nav"],
            labels={"value": "NAV", "date": "Date"},
            title="NAV vs Benchmark"
        )
        st.plotly_chart(fig, use_container_width=True)
    except Exception:
        st.info("Not enough data for chart.")

# ------------------------------------------------------------
# TAB 2 — ALPHA CAPTURE MATRIX
# ------------------------------------------------------------
with tab2:
    st.subheader(f"Alpha Capture Matrix — {sel_mode_label}")

    try:
        matrix = wave_df[[
            "wave", "alpha_1d", "alpha_30d", "alpha_60d",
            "alpha_1y", "return_30d", "return_60d", "return_252d"
        ]].copy()

        # Latest snapshot only
        matrix = matrix.tail(1)

        st.dataframe(matrix, use_container_width=True)

    except Exception as e:
        st.warning(f"Alpha matrix unavailable: {str(e)}")

# ------------------------------------------------------------
# TAB 3 — WAVESCORE PLACEHOLDER
# ------------------------------------------------------------
with tab3:
    st.subheader("WaveScore™ (Preview)")

    st.info("WaveScore v1.0 engine plugged in soon — scoring via Return Quality, Risk Control, Consistency, Resilience, Efficiency, Governance.")

# ------------------------------------------------------------
# TAB 4 — SYSTEM STATUS
# ------------------------------------------------------------
with tab4:
    st.subheader("System Status")

    st.write(f"**Data Points Loaded:** {len(wave_df)}")
    st.write(f"**Latest Date:** {latest.get('date')}")
    st.write(f"**Benchmark:** {latest.get('benchmark_ticker')}")
    st.write("---")

    st.write("### SmartSafe Status")
    try:
        colA, colB = st.columns(2)
        colA.metric("Current SmartSafe Weight", fmt(latest.get("smartsafe_weight")))
        colB.metric("Annual SmartSafe Yield", fmt(latest.get("smartsafe_yield_annual")))
    except:
        st.info("SmartSafe values unavailable.")

    st.write("---")
    st.write("### Underlying DataFrame")
    st.dataframe(wave_df.tail(30), use_container_width=True)