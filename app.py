"""
app.py

WAVES Intelligence™ Institutional Console (Streamlit)

- Clears Streamlit cache on startup
- Uses ONLY the latest waves_engine.py logic
- Loads list.csv (universe) and wave_weights.csv (wave definitions)
- Auto-discovers Waves
- Shows intraday + 30-day return & alpha
- Displays top 10 holdings with Google Finance links
- Adds Wave Snapshot card + benchmark label + history table
"""

import streamlit as st
import pandas as pd

from waves_engine import WavesEngine


# ----------------------------------------------------------------------
# Hard cache reset on app start
# ----------------------------------------------------------------------
def clear_streamlit_cache_once():
    """Force Streamlit to forget any older cached state on first load."""
    if "cache_cleared" in st.session_state:
        return

    try:
        if hasattr(st, "cache_data"):
            st.cache_data.clear()
        if hasattr(st, "cache_resource"):
            st.cache_resource.clear()
    except Exception:
        # Fail silently; app will still run
        pass

    st.session_state["cache_cleared"] = True


clear_streamlit_cache_once()

# ----------------------------------------------------------------------
# Page configuration
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="WAVES Intelligence™ Institutional Console",
    layout="wide",
)

st.title("WAVES Intelligence™ Institutional Console")
st.caption("Live Wave Engine • Intraday + 30-Day Alpha • S&P Wave + Full Lineup")

# ----------------------------------------------------------------------
# Initialize engine
# ----------------------------------------------------------------------
try:
    engine = WavesEngine(list_path="list.csv", weights_path="wave_weights.csv")
except Exception as e:
    st.error(f"Engine failed to initialize: {e}")
    st.stop()

waves = engine.get_wave_names()
if not waves:
    st.error("No Waves detected in wave_weights.csv.")
    st.stop()

# Sidebar
st.sidebar.header("Wave Selector")
selected_wave = st.sidebar.selectbox("Select Wave", waves, index=0)

st.sidebar.markdown("---")
st.sidebar.markdown("**Files in use:**")
st.sidebar.code("list.csv\nwave_weights.csv", language="text")

# ----------------------------------------------------------------------
# Main layout
# ----------------------------------------------------------------------
top_row = st.columns([2.2, 1.2])
perf_col, snapshot_col = top_row

bottom_row = st.columns([2.0, 1.4])
chart_col, holdings_col = bottom_row

# ----------------------------------------------------------------------
# Wave Snapshot (right of title)
# ----------------------------------------------------------------------
with snapshot_col:
    st.subheader("Wave Snapshot")

    try:
        holdings_df = engine.get_wave_holdings(selected_wave)
        num_holdings = len(holdings_df)
        benchmark = engine.get_benchmark(selected_wave)
    except Exception as e:
        st.error(f"Could not load snapshot for {selected_wave}: {e}")
        holdings_df = None
        num_holdings = 0
        benchmark = "SPY"

    snapshot1, snapshot2 = st.columns(2)
    snapshot1.metric("Wave", selected_wave)
    snapshot2.metric("Benchmark", benchmark)

    st.write("")
    st.write(f"**Holdings:** {num_holdings:,}")

    if holdings_df is not None and "sector" in holdings_df.columns:
        # Top sector by total weight
        sector_weights = (
            holdings_df.groupby("sector")["weight"].sum().sort_values(ascending=False)
        )
        if not sector_weights.empty:
            top_sector = sector_weights.index[0]
            top_sector_weight = float(sector_weights.iloc[0])
            st.write(f"**Top Sector:** {top_sector} ({top_sector_weight:.1%})")

# ----------------------------------------------------------------------
# Performance Panel
# ----------------------------------------------------------------------
with perf_col:
    st.subheader(f"{selected_wave} — Performance")

    try:
        perf = engine.get_wave_performance(selected_wave, days=30, log=True)
    except Exception as e:
        st.error(f"Could not compute performance for {selected_wave}: {e}")
        perf = None

    if perf is not None:
        benchmark = perf["benchmark"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Intraday Return", f"{perf['intraday_return'] * 100:0.2f}%")
        c2.metric("Intraday Alpha", f"{perf['intraday_alpha'] * 100:0.2f}%")
        c3.metric("30-Day Return", f"{perf['return_30d'] * 100:0.2f}%")
        c4.metric("30-Day Alpha", f"{perf['alpha_30d'] * 100:0.2f}%")

        st.markdown(f"**Benchmark:** {benchmark}")

# ----------------------------------------------------------------------
# Chart + History Table
# ----------------------------------------------------------------------
with chart_col:
    if perf is not None:
        st.markdown(f"### {selected_wave} vs {perf['benchmark']} — 30-Day Curve")

        history = perf["history"]
        chart_data = history[["wave_value", "benchmark_value"]]
        st.line_chart(chart_data)

        # History table (last 15 days)
        hist_df = history.copy()
        hist_df = hist_df.reset_index().rename(columns={"index": "date"})
        hist_df["date"] = pd.to_datetime(hist_df["date"]).dt.date
        hist_df["wave_return_pct"] = hist_df["wave_return"] * 100
        hist_df["benchmark_return_pct"] = hist_df["benchmark_return"] * 100
        hist_df["alpha_pct"] = (hist_df["wave_return"] - hist_df["benchmark_return"]) * 100

        display_cols = [
            "date",
            "wave_return_pct",
            "benchmark_return_pct",
            "alpha_pct",
        ]
        hist_display = hist_df[display_cols].tail(15).iloc[::-1]  # most recent on top
        hist_display = hist_display.rename(
            columns={
                "date": "Date",
                "wave_return_pct": "Wave Return (%)",
                "benchmark_return_pct": "Benchmark Return (%)",
                "alpha_pct": "Alpha (%)",
            }
        )
        hist_display = hist_display.round(3)

        st.markdown("#### Recent Daily Returns & Alpha (Last 15 Days)")
        st.dataframe(hist_display, hide_index=True, use_container_width=True)

# ----------------------------------------------------------------------
# Holdings Panel (Top 10 + Google links)
# ----------------------------------------------------------------------
with holdings_col:
    st.subheader(f"{selected_wave} — Top 10 Holdings")

    try:
        top10 = engine.get_top_holdings(selected_wave, n=10)
    except Exception as e:
        st.error(f"Could not load holdings for {selected_wave}: {e}")
        top10 = None

    if top10 is not None and not top10.empty:
        # Build Google Finance URLs
        def google_finance_url(ticker: str) -> str:
            # You can adjust exchange suffix logic here if needed
            return f"https://www.google.com/finance/quote/{ticker}:NASDAQ"

        display_df = top10.copy()

        if "company" not in display_df.columns:
            display_df["company"] = ""

        display_df = display_df[["ticker", "company", "weight"]].copy()
        display_df["weight"] = display_df["weight"].round(4)
        display_df["Google Finance"] = display_df["ticker"].apply(google_finance_url)

        st.dataframe(display_df, hide_index=True, use_container_width=True)
    else:
        st.write("No holdings found for this Wave.")

# ----------------------------------------------------------------------
# Footer / Debug Info
# ----------------------------------------------------------------------
st.markdown("---")
st.caption(
    "Engine: WAVES Intelligence™ • list.csv = total market universe • "
    "wave_weights.csv = Wave definitions • Modes: Standard / Alpha-Minus-Beta / "
    "Private Logic handled in engine logic."
)