import pandas as pd
import streamlit as st

from waves_engine import (
    get_all_waves,
    get_wave_positions,
    compute_history_nav,
    get_portfolio_overview,
    get_benchmark_wave_for,
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


def build_holdings_table(wave_name: str) -> str:
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
st.sidebar.markdown("**Risk Mode (table default):**")

mode = st.sidebar.radio(
    "",
    ["Standard", "Alpha-Minus-Beta", "Private Logic"],
    index=0,
)

# For now keep fixed windows; we can expose these later if you want sliders
lookback_days = 365
short_lookback_days = 30

waves = get_all_waves()
default_wave = "AI Wave" if "AI Wave" in waves else waves[0]
selected_wave = st.sidebar.selectbox("Select Wave", waves, index=waves.index(default_wave))

st.sidebar.markdown("---")
st.sidebar.caption("NAV normalized to 1.0 at start of lookback window.")

# ----------------------------------------------------
# Portfolio-Level Overview
# ----------------------------------------------------

st.title("Portfolio-Level Overview")

st.markdown(
    f"**Current Mode (table):** `{mode}`  •  Lookback: **{lookback_days} days**  •  "
    f"Short window: **{short_lookback_days} days**"
)

overview_df = get_portfolio_overview(
    mode=mode,
    long_lookback_days=lookback_days,
    short_lookback_days=short_lookback_days,
).copy()

display_df = overview_df.copy()
display_df["NAV (last)"] = display_df["NAV_last"].apply(fmt_nav)
display_df["365D Return"] = display_df["Return_365D"].apply(fmt_pct)
display_df["30D Return"] = display_df["Return_30D"].apply(fmt_pct)
display_df["365D Alpha vs Benchmark"] = display_df["Alpha_365D"].apply(fmt_pct)
display_df["30D Alpha vs Benchmark"] = display_df["Alpha_30D"].apply(fmt_pct)

display_df = display_df[
    [
        "Wave",
        "Benchmark",
        "NAV (last)",
        "365D Return",
        "30D Return",
        "365D Alpha vs Benchmark",
        "30D Alpha vs Benchmark",
    ]
]

st.dataframe(
    display_df,
    use_container_width=True,
    hide_index=True,
)

st.caption(
    "Each Wave is compared to its own benchmark (see Benchmark column). "
    "NAV is normalized to 1.0 at the start of the 365D window; returns and alpha "
    "are cumulative over the selected periods."
)

st.markdown("---")

# ----------------------------------------------------
# Wave Detail Section
# ----------------------------------------------------

benchmark_for_selected = get_benchmark_wave_for(selected_wave)
st.header(f"Wave Detail — {selected_wave}")
st.caption(f"Benchmark for this Wave: **{benchmark_for_selected}**")

# NAV history for currently selected sidebar mode
try:
    hist = compute_history_nav(
        selected_wave,
        lookback_days=lookback_days,
        mode=mode,
    )
except Exception as e:
    st.error(f"Error computing NAV history for {selected_wave}: {e}")
    hist = None

if hist is not None and not hist.empty:
    hist = hist.copy()
    hist.index.name = "Date"
    hist_reset = hist.reset_index()

    # Chart
    st.subheader(f"NAV History ({mode})")
    st.line_chart(
        hist_reset.set_index("Date")[["NAV"]],
        use_container_width=True,
    )

    # Summary stats
    nav_last = fmt_nav(hist["NAV"].iloc[-1])
    total_ret = fmt_pct(hist["CumReturn"].iloc[-1])

    if len(hist) > short_lookback_days:
        short_slice = hist.iloc[-short_lookback_days:]
        short_ret = fmt_pct(
            short_slice["NAV"].iloc[-1] / short_slice["NAV"].iloc[0] - 1.0
        )
    else:
        short_ret = total_ret

    col1, col2, col3 = st.columns(3)
    col1.metric("NAV (last)", nav_last)
    col2.metric(f"{lookback_days}D Return", total_ret)
    col3.metric(f"{short_lookback_days}D Return", short_ret)
else:
    st.warning("No NAV history available for this Wave and lookback window.")

st.markdown("---")

# ----------------------------------------------------
# Mode Comparison Panel for Selected Wave (vs its benchmark)
# ----------------------------------------------------

st.subheader(f"Mode Comparison — {selected_wave}")

modes = ["Standard", "Alpha-Minus-Beta", "Private Logic"]
rows = []

# Benchmark history for this Wave (always Standard mode for benchmark)
try:
    bench_hist = compute_history_nav(
        benchmark_for_selected,
        lookback_days=lookback_days,
        mode="Standard",
    )
except Exception as e:
    bench_hist = None
    st.warning(f"Benchmark history unavailable for alpha calc: {e}")

for m in modes:
    try:
        h = compute_history_nav(selected_wave, lookback_days=lookback_days, mode=m)
        nav_last_val = h["NAV"].iloc[-1]
        ret_long = h["CumReturn"].iloc[-1]

        if len(h) > short_lookback_days:
            hs = h.iloc[-short_lookback_days:]
            ret_short = hs["NAV"].iloc[-1] / hs["NAV"].iloc[0] - 1.0
        else:
            ret_short = ret_long

        alpha_long = float("nan")
        alpha_short = float("nan")

        if bench_hist is not None:
            combo = pd.concat(
                [
                    h[["NAV"]].rename(columns={"NAV": "NAV_wave"}),
                    bench_hist[["NAV"]].rename(columns={"NAV": "NAV_bench"}),
                ],
                axis=1,
                join="inner",
            ).dropna()

            if not combo.empty:
                nav_w = combo["NAV_wave"]
                nav_b = combo["NAV_bench"]
                wave_ret = nav_w.iloc[-1] / nav_w.iloc[0] - 1.0
                bench_ret = nav_b.iloc[-1] / nav_b.iloc[0] - 1.0
                alpha_long = wave_ret - bench_ret

                if len(combo) > short_lookback_days:
                    cs = combo.iloc[-short_lookback_days:]
                else:
                    cs = combo
                nav_w_s = cs["NAV_wave"]
                nav_b_s = cs["NAV_bench"]
                wave_ret_s = nav_w_s.iloc[-1] / nav_w_s.iloc[0] - 1.0
                bench_ret_s = nav_b_s.iloc[-1] / nav_b_s.iloc[0] - 1.0
                alpha_short = wave_ret_s - bench_ret_s

        rows.append(
            {
                "Mode": m,
                "NAV (last)": fmt_nav(nav_last_val),
                "365D Return": fmt_pct(ret_long),
                "30D Return": fmt_pct(ret_short),
                "365D Alpha vs Benchmark": fmt_pct(alpha_long),
                "30D Alpha vs Benchmark": fmt_pct(alpha_short),
            }
        )
    except Exception as e:
        rows.append(
            {
                "Mode": m,
                "NAV (last)": "—",
                "365D Return": "—",
                "30D Return": "—",
                "365D Alpha vs Benchmark": "—",
                "30D Alpha vs Benchmark": "—",
            }
        )
        st.warning(f"Error computing mode '{m}' for {selected_wave}: {e}")

mode_df = pd.DataFrame(rows)
st.dataframe(mode_df, use_container_width=True, hide_index=True)

st.markdown("---")

# ----------------------------------------------------
# Top 10 Holdings
# ----------------------------------------------------

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