import pandas as pd
import streamlit as st

from waves_engine import (
    get_all_waves,
    get_wave_positions,
    compute_history_nav,
    get_portfolio_overview,
)

st.set_page_config(
    page_title="WAVES Intelligence™ Institutional Console",
    layout="wide",
)


# ----------------------------------------------------
# Utility helpers
# ----------------------------------------------------

def fmt_pct(x):
    if pd.isna(x):
        return "—"
    return f"{x * 100:0.1f}%"


def fmt_nav(x):
    if pd.isna(x):
        return "—"
    return f"{x:0.3f}"


def build_holdings_table(wave_name: str):
    df = get_wave_positions(wave_name).copy()
    df = df.sort_values("Weight", ascending=False)
    df_top = df.head(10).copy()
    df_top["Weight"] = df_top["Weight"].apply(fmt_pct)

    # Build Markdown table with clickable Google Finance links
    lines = ["| Ticker | Weight |", "|--------|--------|"]
    for _, row in df_top.iterrows():
        t = row["Ticker"]
        w = row["Weight"]
        link = f"[{t}](https://www.google.com/finance/quote/{t}:NASDAQ)"
        lines.append(f"| {link} | {w} |")
    return "\n".join(lines)


# ----------------------------------------------------
# Sidebar controls
# ----------------------------------------------------

st.sidebar.title("WAVES Intelligence™")
st.sidebar.markdown("**Mode:**")

mode = st.sidebar.radio(
    "",
    ["Standard", "Alpha-Minus-Beta", "Private Logic"],
    index=0,
)

lookback_days = 365
short_lookback_days = 30

waves = get_all_waves()
default_wave = "AI Wave" if "AI Wave" in waves else waves[0]
selected_wave = st.sidebar.selectbox("Select Wave", waves, index=waves.index(default_wave))

st.sidebar.markdown("---")
st.sidebar.caption("NAV normalized to 1.0 at start of lookback window.")
# ----------------------------------------------------
# Main Title
# ----------------------------------------------------

st.title("Portfolio-Level Overview")

st.markdown(
    f"**Current Mode:** `{mode}`  •  Lookback: **{lookback_days} days**  •  "
    f"Short window: **{short_lookback_days} days**"
)

# ----------------------------------------------------
# Overview Table (with alpha vs S&P 500)
# ----------------------------------------------------

overview_df = get_portfolio_overview(
    mode=mode,
    long_lookback_days=lookback_days,
    short_lookback_days=short_lookback_days,
).copy()

display_df = overview_df.copy()
display_df["NAV (last)"] = display_df["NAV_last"].apply(fmt_nav)
display_df["365D Return"] = display_df["Return_365D"].apply(fmt_pct)
display_df["30D Return"] = display_df["Return_30D"].apply(fmt_pct)
display_df["365D Alpha vs S&P"] = display_df["Alpha_365D"].apply(fmt_pct)
display_df["30D Alpha vs S&P"] = display_df["Alpha_30D"].apply(fmt_pct)

display_df = display_df[
    [
        "Wave",
        "NAV (last)",
        "365D Return",
        "30D Return",
        "365D Alpha vs S&P",
        "30D Alpha vs S&P",
    ]
]

st.dataframe(
    display_df,
    use_container_width=True,
    hide_index=True,
)

st.caption(
    "NAV is normalized to 1.0 at the start of the 365D window. "
    "Returns and alpha are cumulative over the selected periods."
)

st.markdown("---")
# ----------------------------------------------------
# Wave Detail Section
# ----------------------------------------------------

st.header(f"Wave Detail — {selected_wave}")

# NAV history
try:
    hist = compute_history_nav(selected_wave, lookback_days=lookback_days, mode=mode)
except Exception as e:
    st.error(f"Error computing NAV history for {selected_wave}: {e}")
    hist = None

if hist is not None and not hist.empty:
    hist = hist.copy()
    hist.index.name = "Date"
    hist_reset = hist.reset_index()

    # Chart
    st.subheader("NAV History")
    st.line_chart(
        hist_reset.set_index("Date")[["NAV"]],
        use_container_width=True,
    )

    # Summary stats
    nav_last = fmt_nav(hist["NAV"].iloc[-1])
    total_ret = fmt_pct(hist["CumReturn"].iloc[-1])

    if len(hist) > short_lookback_days:
        short_slice = hist.iloc[-short_lookback_days:]
        short_ret = fmt_pct(short_slice["NAV"].iloc[-1] / short_slice["NAV"].iloc[0] - 1.0)
    else:
        short_ret = total_ret

    col1, col2, col3 = st.columns(3)
    col1.metric("NAV (last)", nav_last)
    col2.metric(f"{lookback_days}D Return", total_ret)
    col3.metric(f"{short_lookback_days}D Return", short_ret)
else:
    st.warning("No NAV history available for this Wave and lookback window.")

st.markdown("---")

# Top 10 holdings
st.subheader("Top 10 Holdings (by target weight)")
try:
    markdown_table = build_holdings_table(selected_wave)
    st.markdown(markdown_table, unsafe_allow_html=True)
except Exception as e:
    st.error(f"Error loading holdings for {selected_wave}: {e}")

st.markdown("---")
st.caption(
    "WAVES Intelligence™ • Vector OS • For demonstration and research purposes only. "
    "Not an offer to sell or solicitation to buy any security."
)