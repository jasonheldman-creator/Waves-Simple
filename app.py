# app.py
"""
WAVES Intelligence™ Institutional Console
- Uses waves_engine.py
- Shows per-Wave performance & top holdings
"""

import streamlit as st
import pandas as pd

import waves_engine as we


# -------------------------------------------------------------------
# Page config
# -------------------------------------------------------------------

st.set_page_config(
    page_title="WAVES Intelligence™ Console",
    layout="wide",
)

st.title("WAVES Intelligence™ Institutional Console")
st.caption("Autonomous Waves • Dynamic Alpha • Real-Time Intelligence")


# -------------------------------------------------------------------
# Load weights & wave list
# -------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_all_weights() -> pd.DataFrame:
    return we.get_dynamic_wave_weights()


weights_df = load_all_weights()
wave_names = we.get_wave_names(weights_df)

if not wave_names:
    st.error("No waves found. Check your wave_weights.csv and sp500_universe.csv.")
    st.stop()

# Sidebar
st.sidebar.header("Wave Selection")
selected_wave = st.sidebar.selectbox("Select Wave", wave_names, index=0)

st.sidebar.markdown("---")
st.sidebar.write("**Tip:** S&P Wave is now fully dynamic with a 500-stock autonomous engine.")


# -------------------------------------------------------------------
# Fetch summary for selected wave
# -------------------------------------------------------------------

with st.spinner(f"Computing metrics for {selected_wave}..."):
    summary = we.compute_wave_summary(selected_wave, weights_df)

# -------------------------------------------------------------------
# Metrics display
# -------------------------------------------------------------------

def fmt_pct(x):
    if x is None or pd.isna(x):
        return "—"
    return f"{x*100:0.2f}%"

col1, col2, col3, col4, col5, col6 = st.columns(6)

col1.metric("1-Day Return", fmt_pct(summary.get("return_1d")))
col2.metric("1-Day Alpha", fmt_pct(summary.get("alpha_1d")))
col3.metric("30-Day Return", fmt_pct(summary.get("return_30d")))
col4.metric("30-Day Alpha", fmt_pct(summary.get("alpha_30d")))
col5.metric("60-Day Return", fmt_pct(summary.get("return_60d")))
col6.metric("60-Day Alpha", fmt_pct(summary.get("alpha_60d")))

bench = summary.get("benchmark")
if bench:
    st.caption(f"Benchmark for **{selected_wave}**: `{bench}`")
else:
    st.caption(f"No specific benchmark mapped yet for **{selected_wave}**.")


# -------------------------------------------------------------------
# Top 10 holdings with Google Finance links
# -------------------------------------------------------------------

st.markdown("---")
st.subheader(f"Top 10 Holdings — {selected_wave}")

top = summary.get("top_holdings")
if top is None or top.empty:
    st.write("No holdings found for this Wave.")
else:
    # Show table with weights
    top_display = top.copy()
    top_display["weight_pct"] = top_display["weight"] * 100.0
    top_display = top_display[["ticker", "weight_pct"]]
    top_display.columns = ["Ticker", "Weight (%)"]

    st.dataframe(
        top_display.style.format({"Weight (%)": "{:0.2f}"}),
        use_container_width=True,
    )

    st.markdown("**Google Finance Links**")
    for _, row in top.iterrows():
        ticker = row["ticker"]
        weight = row["weight"]
        url = f"https://www.google.com/finance?q={ticker}"
        st.markdown(
            f"- [{ticker}]({url}) — {weight*100:0.2f}% weight",
            unsafe_allow_html=True,
        )


# -------------------------------------------------------------------
# Footer
# -------------------------------------------------------------------

st.markdown("---")
st.caption("WAVES Intelligence™ • Autonomous Wealth Engine • Not investment advice.")