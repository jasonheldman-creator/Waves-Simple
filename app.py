"""
app.py — WAVES Intelligence™ Institutional Console (Restored)

Key features
------------
- Imports compute_history_nav from waves_engine.py (Option B).
- Portfolio-Level Overview:
    • All Waves
    • Last NAV
    • 365D return
    • 30D return
- Wave Detail:
    • NAV chart
    • Metrics summary
    • Top 10 holdings with clickable Google Finance links
- Mode selection:
    • Standard
    • Alpha-Minus-Beta
    • Private Logic
"""

import math

import streamlit as st
import pandas as pd

from waves_engine import (
    get_all_waves,
    get_portfolio_overview,
    get_wave_positions,
    compute_history_nav,
)

# -----------------------------#
# Streamlit page config
# -----------------------------#

st.set_page_config(
    page_title="WAVES Intelligence™ Institutional Console",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------#
# Sidebar controls
# -----------------------------#

st.sidebar.title("WAVES Intelligence™")
st.sidebar.caption("Vector OS · Institutional Console")

# Mode selector
mode = st.sidebar.radio(
    "Risk Mode",
    options=["Standard", "Alpha-Minus-Beta", "Private Logic"],
    index=0,
)

# Lookback for charts (detail view)
lookback_days = st.sidebar.selectbox(
    "Lookback window (detail view)",
    options=[365, 730],
    index=0,
    format_func=lambda x: f"{x} days",
)

# Load Waves list
try:
    waves = get_all_waves()
except Exception as e:
    st.error(f"Error loading Waves from wave_weights.csv: {e}")
    st.stop()

if not waves:
    st.error("No Waves found in wave_weights.csv.")
    st.stop()

selected_wave = st.sidebar.selectbox("Select Wave", options=waves, index=0)

st.sidebar.markdown("---")
st.sidebar.caption("NAV & returns are approximate and for demonstration only.\nNot investment advice.")


# -----------------------------#
# Main layout: Overview + Detail
# -----------------------------#

st.title("Portfolio-Level Overview")

# --- Portfolio-Level Overview ---

try:
    overview_df = get_portfolio_overview(
        mode=mode,
        long_lookback_days=365,
        short_lookback_days=30,
    )
except Exception as e:
    st.error(f"Error computing portfolio overview: {e}")
    overview_df = None

if overview_df is not None:
    display_df = overview_df.copy()

    # Formatting for display
    display_df["NAV (last)"] = display_df["NAV_last"].map(
        lambda x: f"{x:,.3f}" if isinstance(x, (int, float)) and not math.isnan(x) else "—"
    )
    display_df["365D Return"] = display_df["Return_365D"].map(
        lambda x: f"{x*100:,.1f}%" if isinstance(x, (int, float)) and not math.isnan(x) else "—"
    )
    display_df["30D Return"] = display_df["Return_30D"].map(
        lambda x: f"{x*100:,.1f}%" if isinstance(x, (int, float)) and not math.isnan(x) else "—"
    )

    display_df = display_df[["Wave", "NAV (last)", "365D Return", "30D Return"]]

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
    )

st.markdown(
    """
NAV is normalized to 1.0 at the start of the 365D window.  
Returns are cumulative over the selected periods.
"""
)

st.markdown("---")

# -----------------------------#
# Wave Detail Section
# -----------------------------#

st.header(f"Wave Detail — {selected_wave}")

col1, col2 = st.columns([2, 1])

# --- Column 1: NAV chart ---

with col1:
    st.subheader("NAV History")

    try:
        nav_df = compute_history_nav(
            wave_name=selected_wave,
            lookback_days=lookback_days,
            mode=mode,
        )
    except Exception as e:
        st.error(f"Error computing NAV history for {selected_wave}: {e}")
        nav_df = None

    if nav_df is not None and not nav_df.empty:
        nav_display = nav_df.copy()
        nav_display = nav_display.reset_index().rename(columns={"index": "Date"})
        nav_display["Date"] = pd.to_datetime(nav_display.index)

        st.line_chart(
            nav_df["NAV"],
            use_container_width=True,
        )

        # Summary stats
        nav_last = float(nav_df["NAV"].iloc[-1])
        cum_ret = float(nav_df["CumReturn"].iloc[-1])

        st.metric(
            label=f"{lookback_days}D Cumulative Return",
            value=f"{cum_ret*100:,.1f}%",
        )
    else:
        st.info("No NAV history available for this Wave and lookback window.")

# --- Column 2: Metrics + Top 10 ---

with col2:
    st.subheader("Top 10 Holdings")

    try:
        positions_df = get_wave_positions(selected_wave)
    except Exception as e:
        st.error(f"Error loading positions for {selected_wave}: {e}")
        positions_df = None

    if positions_df is not None and not positions_df.empty:
        positions_df = positions_df.sort_values("Weight", ascending=False).reset_index(drop=True)
        top10 = positions_df.head(10).copy()

        # Build Google Finance links
        def google_link(ticker: str) -> str:
            url = f"https://www.google.com/finance/quote/{ticker}:NASDAQ"
            # Keep it simple; user can adjust exchange suffix as needed.
            return f"[{ticker}]({url})"

        top10["Ticker"] = top10["Ticker"].astype(str).str.upper()
        top10["Weight"] = top10["Weight"].astype(float)

        display_top10 = pd.DataFrame(
            {
                "Ticker": top10["Ticker"].apply(google_link),
                "Weight": top10["Weight"].map(lambda x: f"{x*100:,.2f}%"),
            }
        )

        # Use st.markdown to keep links clickable
        st.markdown("Top 10 by target weight (click ticker for Google Finance):")
        st.write(display_top10.to_markdown(index=False))
    else:
        st.info("No positions available for this Wave.")


# -----------------------------#
# Footer
# -----------------------------#

st.markdown("---")
st.caption(
    "WAVES Intelligence™ • Vector OS • For demonstration and research purposes only. "
    "Not an offer to sell or solicitation to buy any security."
)