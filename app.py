"""
app.py

WAVES Intelligence™ Institutional Console (Streamlit)

- Clears Streamlit cache on startup
- Uses ONLY the latest waves_engine.py logic
- Loads list.csv (universe) and wave_weights.csv (wave definitions)
- Auto-discovers Waves
- Shows intraday + 30-day return & alpha
- Displays top 10 holdings with Google Finance links
"""

import streamlit as st

from waves_engine import WavesEngine


# ----------------------------------------------------------------------
# Hard cache reset on app start
# ----------------------------------------------------------------------
def clear_streamlit_cache_once():
    if "cache_cleared" in st.session_state:
        return

    try:
        # Newer Streamlit APIs
        if hasattr(st, "cache_data"):
            st.cache_data.clear()
        if hasattr(st, "cache_resource"):
            st.cache_resource.clear()
    except Exception:
        # Fall back silently if not supported
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
col_perf, col_holdings = st.columns([2.0, 1.4])

# ---------------- Performance Panel ----------------
with col_perf:
    st.subheader(f"{selected_wave} — Performance")

    try:
        perf = engine.get_wave_performance(selected_wave, days=30, log=True)
    except Exception as e:
        st.error(f"Could not compute performance for {selected_wave}: {e}")
        perf = None

    if perf is not None:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric(
            "Intraday Return",
            f"{perf['intraday_return'] * 100:0.2f}%",
        )
        c2.metric(
            "Intraday Alpha",
            f"{perf['intraday_alpha'] * 100:0.2f}%",
        )
        c3.metric(
            "30-Day Return",
            f"{perf['return_30d'] * 100:0.2f}%",
        )
        c4.metric(
            "30-Day Alpha",
            f"{perf['alpha_30d'] * 100:0.2f}%",
        )

        st.markdown("### Wave vs Benchmark — 30-Day Curve")
        history = perf["history"]
        # Only intraday + 30-day; no 60-day / 1-year as requested
        chart_data = history[["wave_value", "benchmark_value"]]
        st.line_chart(chart_data)

# ---------------- Holdings Panel ----------------
with col_holdings:
    st.subheader(f"{selected_wave} — Top 10 Holdings")

    try:
        top10 = engine.get_top_holdings(selected_wave, n=10)
    except Exception as e:
        st.error(f"Could not load holdings for {selected_wave}: {e}")
        top10 = None

    if top10 is not None and not top10.empty:
        # Build Google Finance URLs (simple default: NASDAQ; edit if needed)
        def google_finance_url(ticker: str) -> str:
            # You can adjust the suffix logic here if you want NYSE/other handling
            return f"https://www.google.com/finance/quote/{ticker}:NASDAQ"

        display_df = top10.copy()
        if "company" not in display_df.columns:
            display_df["company"] = ""

        display_df = display_df[["ticker", "company", "weight"]].copy()
        display_df["weight"] = display_df["weight"].round(4)
        display_df["Google Finance"] = display_df["ticker"].apply(google_finance_url)

        st.dataframe(
            display_df,
            hide_index=True,
            use_container_width=True,
        )
    else:
        st.write("No holdings found for this Wave.")

# ----------------------------------------------------------------------
# Footer / Debug Info
# ----------------------------------------------------------------------
st.markdown("---")
st.caption(
    "Engine: WAVES Intelligence™ • list.csv = total market universe • "
    "wave_weights.csv = Wave definitions • Modes: Standard / Alpha-Minus-Beta / Private Logic handled in engine logic."
)