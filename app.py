import pandas as pd
import streamlit as st

from waves_engine import (
    get_all_waves,
    get_wave_positions,
    compute_history_nav,
    get_portfolio_overview,
    get_benchmark_wave_for,
    get_benchmark_composition,
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


def compute_vol_and_maxdd(nav_series: pd.Series, return_series: pd.Series):
    """
    Compute annualized volatility and max drawdown from NAV & returns.
    """
    if nav_series is None or nav_series.empty:
        return float("nan"), float("nan")

    # Annualized vol from daily returns (252 trading days)
    vol = return_series.std() * (252 ** 0.5) if len(return_series) > 1 else float("nan")

    # Max drawdown from NAV
    nav = nav_series.values
    running_max = pd.Series(nav).cummax().values
    drawdowns = nav / running_max - 1.0
    max_dd = drawdowns.min() if len(drawdowns) > 0 else float("nan")

    return vol, max_dd


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

# History window (controls lookback_days)
history_choice = st.sidebar.selectbox(
    "History window",
    ["1Y", "3Y", "5Y"],
    index=0,
)

if history_choice == "1Y":
    lookback_days = 365
elif history_choice == "3Y":
    lookback_days = 365 * 3
else:
    lookback_days = 365 * 5

short_lookback_days = 30  # keep a 30D "tactical" window

waves = get_all_waves()
default_wave = "AI Wave" if "AI Wave" in waves else waves[0]
selected_wave = st.sidebar.selectbox("Select Wave", waves, index=waves.index(default_wave))

st.sidebar.markdown("---")
st.sidebar.caption(
    "NAV normalized to 1.0 at start of selected window. "
    "Alpha vs composite benchmark portfolios."
)

# ----------------------------------------------------
# Portfolio-Level Overview
# ----------------------------------------------------

st.title("Portfolio-Level Overview")

st.markdown(
    f"**Current Mode (table):** `{mode}`  •  Window: **{history_choice}**  •  "
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
    "Each Wave is compared to its own composite benchmark portfolio (see Benchmark column). "
    "NAV is normalized to 1.0 at the start of the selected history window; returns and alpha "
    "are cumulative over the window and the last 30 trading days."
)

st.markdown("---")

# ----------------------------------------------------
# Wave Detail Section
# ----------------------------------------------------

benchmark_for_selected = get_benchmark_wave_for(selected_wave)
benchmark_comp = get_benchmark_composition(benchmark_for_selected)

st.header(f"Wave Detail — {selected_wave}")

if benchmark_comp:
    comp_str_parts = [f"{weight * 100:0.0f}% {ticker}" for ticker, weight in benchmark_comp.items()]
    comp_str = " + ".join(comp_str_parts)
    st.caption(
        f"Benchmark for this Wave: **{benchmark_for_selected}**  "
        f"= {comp_str}"
    )
else:
    st.caption(f"Benchmark for this Wave: **{benchmark_for_selected}**")

# NAV history for currently selected sidebar mode (Wave)
try:
    hist = compute_history_nav(
        selected_wave,
        lookback_days=lookback_days,
        mode=mode,
        is_benchmark=False,
    )
except Exception as e:
    st.error(f"Error computing NAV history for {selected_wave}: {e}")
    hist = None

if hist is not None and not hist.empty:
    hist = hist.copy()
    hist.index.name = "Date"
    hist_reset = hist.reset_index()

    # Chart
    st.subheader(f"NAV History ({mode}, {history_choice})")
    st.line_chart(
        hist_reset.set_index("Date")[["NAV"]],
        use_container_width=True,
    )

    # Summary stats
    nav_last = fmt_nav(hist["NAV"].iloc[-1])
    total_ret = fmt_pct(hist["CumReturn"].iloc[-1])

    if len(hist) > short_lookback_days:
        short_slice = hist.iloc[-short_lookback_days:]
        short_ret_val = short_slice["NAV"].iloc[-1] / short_slice["NAV"].iloc[0] - 1.0
        short_ret = fmt_pct(short_ret_val)
    else:
        short_ret = total_ret

    col1, col2, col3 = st.columns(3)
    col1.metric("NAV (last)", nav_last)
    col2.metric(f"{history_choice} Return", total_ret)
    col3.metric(f"{short_lookback_days}D Return", short_ret)
else:
    st.warning("No NAV history available for this Wave and window.")

st.markdown("---")

# ----------------------------------------------------
# Mode Comparison Panel for Selected Wave (vs composite benchmark)
# ----------------------------------------------------

st.subheader(f"Mode Comparison — {selected_wave}")

modes = ["Standard", "Alpha-Minus-Beta", "Private Logic"]
rows = []

# Benchmark history for this Wave (composite benchmark, Standard mode)
try:
    bench_hist = compute_history_nav(
        benchmark_for_selected,
        lookback_days=lookback_days,
        mode="Standard",
        is_benchmark=True,
    )
except Exception as e:
    bench_hist = None
    st.warning(f"Benchmark history unavailable for alpha calc: {e}")

for m in modes:
    try:
        h = compute_history_nav(
            selected_wave,
            lookback_days=lookback_days,
            mode=m,
            is_benchmark=False,
        )
        nav_last_val = h["NAV"].iloc[-1]
        ret_long = h["CumReturn"].iloc[-1]

        if len(h) > short_lookback_days:
            hs = h.iloc[-short_lookback_days:]
            ret_short_val = hs["NAV"].iloc[-1] / hs["NAV"].iloc[0] - 1.0
        else:
            ret_short_val = ret_long

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

        # Risk analytics: vol & max drawdown (on full window)
        vol, max_dd = compute_vol_and_maxdd(
            nav_series=h["NAV"],
            return_series=h["Return"],
        )

        rows.append(
            {
                "Mode": m,
                "NAV (last)": fmt_nav(nav_last_val),
                f"{history_choice} Return": fmt_pct(ret_long),
                "30D Return": fmt_pct(ret_short_val),
                f"{history_choice} Alpha vs Benchmark": fmt_pct(alpha_long),
                "30D Alpha vs Benchmark": fmt_pct(alpha_short),
                "Ann. Volatility": fmt_pct(vol),
                "Max Drawdown": fmt_pct(max_dd),
            }
        )
    except Exception as e:
        rows.append(
            {
                "Mode": m,
                "NAV (last)": "—",
                f"{history_choice} Return": "—",
                "30D Return": "—",
                f"{history_choice} Alpha vs Benchmark": "—",
                "30D Alpha vs Benchmark": "—",
                "Ann. Volatility": "—",
                "Max Drawdown": "—",
            }
        )
        st.warning(f"Error computing mode '{m}' for {selected_wave}: {e}")

mode_df = pd.DataFrame(rows)
st.dataframe(mode_df, use_container_width=True, hide_index=True)

st.caption(
    "Mode comparison shows each risk mode's return, alpha vs composite benchmark, "
    "and basic risk stats (annualized volatility and max drawdown) over the selected window."
)

st.markdown("---")

# ----------------------------------------------------
# Benchmark Composition Table
# ----------------------------------------------------

st.subheader("Benchmark Composition (ETFs & Weights)")

if benchmark_comp:
    rows_b = []
    for ticker, weight in benchmark_comp.items():
        link = f"[{ticker}](https://www.google.com/finance/quote/{ticker}:NYSEARCA)"
        rows_b.append(
            {
                "ETF": link,
                "Weight": fmt_pct(weight),
            }
        )
    bench_df = pd.DataFrame(rows_b)
    st.markdown(
        bench_df.to_markdown(index=False),
        unsafe_allow_html=True,
    )
else:
    st.write("No benchmark composition data available for this Wave.")

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