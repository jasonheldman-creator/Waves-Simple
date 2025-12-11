# app.py — WAVES Intelligence™ • Vector OS • Institutional Console (Mobile-Friendly)
# ------------------------------------------------------------------------------
# Works with the mobile-optimized waves_engine.py (no terminal scripts required).
#
# Features
# --------
# • Sidebar:
#       - Risk Mode: Standard / Alpha-Minus-Beta / Private Logic
#       - Lookback Window (detail): 30D, 90D, 365D
#       - Wave selector
# • Portfolio-Level Overview:
#       - For all Waves:
#           Wave, Benchmark, NAV(last), 365D Return, 30D Return,
#           365D Alpha vs Benchmark, 30D Alpha vs Benchmark
#       - Shows a separate table listing each Benchmark & its ETF mix
# • Wave Detail:
#       - NAV chart: selected Wave vs its composite Benchmark
#       - Key stats: 365D / 30D returns & alpha vs benchmark
#       - Mode comparison table (Standard vs Alpha-Minus-Beta vs Private Logic)
#       - Top 10 holdings (with Google Finance links)
#
# Notes
# -----
# • All history is computed from live/simulated prices via waves_engine.py.
# • Full_Wave_History.csv is optional and currently disabled in waves_engine.py.

import math
from typing import Dict, List

import pandas as pd
import streamlit as st

from waves_engine import (
    get_all_waves,
    get_wave_positions,
    get_portfolio_overview,
    compute_history_nav,
    get_benchmark_wave_for,
    get_benchmark_composition,
)


# ------------------------------------------------------------------------------
# Utility formatting helpers
# ------------------------------------------------------------------------------

def pct(x: float) -> str:
    """Format a float as percentage with 1 decimal; handle NaN safely."""
    try:
        if x is None or (isinstance(x, float) and math.isnan(x)):
            return "—"
        return f"{x * 100:.1f}%"
    except Exception:
        return "—"


def nav_fmt(x: float) -> str:
    try:
        if x is None or (isinstance(x, float) and math.isnan(x)):
            return "—"
        return f"{x:.3f}"
    except Exception:
        return "—"


def build_benchmark_mix_string(composition: Dict[str, float]) -> str:
    """
    Turn {"QQQ": 0.5, "IGV": 0.5} into "50% QQQ + 50% IGV".
    """
    if not composition:
        return "—"
    parts: List[str] = []
    total = sum(composition.values())
    if total <= 0:
        return "—"
    for ticker, w in composition.items():
        pct_w = w / total * 100.0
        parts.append(f"{pct_w:.0f}% {ticker}")
    return " + ".join(parts)


def google_finance_url(ticker: str) -> str:
    """
    Simple Google Finance link. We don't know the exact exchange, but
    this still works well enough for demos.
    """
    t = ticker.strip().upper()
    return f"https://www.google.com/finance/quote/{t}:NASDAQ"


# ------------------------------------------------------------------------------
# Streamlit App
# ------------------------------------------------------------------------------

st.set_page_config(
    page_title="WAVES Intelligence™ • Institutional Console",
    page_icon="🌊",
    layout="wide",
)

# Sidebar branding
with st.sidebar:
    st.markdown("### 🌊 WAVES Intelligence™")
    st.markdown("##### Vector OS • Institutional Console")

    # Risk Mode
    mode = st.radio(
        "Risk Mode",
        options=["Standard", "Alpha-Minus-Beta", "Private Logic"],
        index=0,
        help=(
            "Standard = base strategy.\n"
            "Alpha-Minus-Beta = de-risked version.\n"
            "Private Logic = enhanced version."
        ),
    )

    # Lookback window for detail charts
    lookback_choice = st.selectbox(
        "Lookback window (detail view)",
        options=["30 days", "90 days", "365 days"],
        index=2,
    )
    if lookback_choice == "30 days":
        long_lookback_days = 365  # still use 1Y for overview
        detail_lookback_days = 30
    elif lookback_choice == "90 days":
        long_lookback_days = 365
        detail_lookback_days = 90
    else:
        long_lookback_days = 365
        detail_lookback_days = 365

    # Wave selector for detail pane
    all_waves = get_all_waves()
    default_wave_index = 0 if "AI Wave" not in all_waves else all_waves.index("AI Wave")
    selected_wave = st.selectbox("Select Wave", all_waves, index=default_wave_index)

    st.markdown("---")
    st.caption(
        "NAV & returns are approximate and for demonstration only. "
        "Not investment advice."
    )

st.title("Portfolio-Level Overview")

# ------------------------------------------------------------------------------
# Portfolio-Level Overview (Waves + Benchmarks + Alpha)
# ------------------------------------------------------------------------------

with st.spinner("Computing portfolio overview…"):
    overview_df = get_portfolio_overview(
        mode=mode,
        long_lookback_days=long_lookback_days,
        short_lookback_days=30,
    )

# Display the numbers in a nicely formatted table
display_df = overview_df.copy()

display_df["NAV (last)"] = display_df["NAV_last"].apply(nav_fmt)
display_df["365D Return"] = display_df["Return_365D"].apply(pct)
display_df["30D Return"] = display_df["Return_30D"].apply(pct)
display_df["365D Alpha vs Benchmark"] = display_df["Alpha_365D"].apply(pct)
display_df["30D Alpha vs Benchmark"] = display_df["Alpha_30D"].apply(pct)

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
    "NAV is normalized to 1.0 at the start of the 365D window. "
    "Returns and alpha are cumulative over the selected periods."
)

# ------------------------------------------------------------------------------
# Benchmark composition table
# ------------------------------------------------------------------------------

st.markdown("### Benchmark — ETF Mix")

bench_rows = []
for wave_name in get_all_waves():
    bench_name = get_benchmark_wave_for(wave_name)
    comp = get_benchmark_composition(bench_name)
    mix_str = build_benchmark_mix_string(comp)
    bench_rows.append(
        {"Wave": wave_name, "Benchmark (ETF mix)": f"{bench_name} ({mix_str})"}
    )

bench_df = (
    pd.DataFrame(bench_rows)
    .drop_duplicates(subset=["Benchmark (ETF mix)"])
    .sort_values("Wave")
    .reset_index(drop=True)
)

st.dataframe(
    bench_df[["Benchmark (ETF mix)"]],
    use_container_width=True,
    hide_index=True,
)

st.caption(
    "Each Wave is compared to its own composite benchmark portfolio as shown above."
)

# ------------------------------------------------------------------------------
# Wave Detail
# ------------------------------------------------------------------------------

st.markdown("---")
st.header(f"Wave Detail — {selected_wave}")

col_left, col_right = st.columns([2, 1])

benchmark_name = get_benchmark_wave_for(selected_wave)
benchmark_comp = get_benchmark_composition(benchmark_name)
benchmark_mix_str = build_benchmark_mix_string(benchmark_comp)

with col_right:
    st.subheader("Benchmark")
    st.markdown(f"**{benchmark_name}**")
    st.caption(benchmark_mix_str or "—")

with col_left:
    # NAV chart: Wave vs its benchmark (using detail_lookback_days)
    try:
        with st.spinner("Computing NAV history…"):
            hist_wave = compute_history_nav(
                selected_wave,
                lookback_days=detail_lookback_days,
                mode=mode,
                is_benchmark=False,
            )
            hist_bench = compute_history_nav(
                benchmark_name,
                lookback_days=detail_lookback_days,
                mode="Standard",
                is_benchmark=True,
            )

            # Align the two series on dates
            wave_nav = hist_wave[["NAV"]].rename(columns={"NAV": "Wave NAV"})
            bench_nav = hist_bench[["NAV"]].rename(columns={"NAV": "Benchmark NAV"})
            nav_combo = pd.concat([wave_nav, bench_nav], axis=1, join="inner").dropna()

        if nav_combo.empty:
            st.warning("No overlapping NAV history available for this Wave and benchmark.")
        else:
            # Normalize both to 1.0 at start for visual comparison
            nav_combo = nav_combo / nav_combo.iloc[0]
            st.line_chart(nav_combo, use_container_width=True)
    except Exception as e:
        st.error(f"Error computing NAV history: {e}")

# ------------------------------------------------------------------------------
# Wave vs Benchmark statistics (365D & 30D) for the selected mode
# ------------------------------------------------------------------------------

st.subheader("Performance vs Benchmark")

try:
    # Use 365D history for stats; fall back gracefully if shorter
    hist_wave_full = compute_history_nav(
        selected_wave,
        lookback_days=365,
        mode=mode,
        is_benchmark=False,
    )
    hist_bench_full = compute_history_nav(
        benchmark_name,
        lookback_days=365,
        mode="Standard",
        is_benchmark=True,
    )

    stats_rows = []

    for label, window in [("365D", 365), ("30D", 30)]:
        combo = pd.concat(
            [
                hist_wave_full[["NAV"]].rename(columns={"NAV": "NAV_wave"}),
                hist_bench_full[["NAV"]].rename(columns={"NAV": "NAV_bench"}),
            ],
            axis=1,
            join="inner",
        ).dropna()

        if combo.empty:
            stats_rows.append(
                {
                    "Window": label,
                    "Wave Return": "—",
                    "Benchmark Return": "—",
                    "Alpha": "—",
                }
            )
            continue

        if len(combo) > window:
            combo_w = combo.iloc[-window:]
        else:
            combo_w = combo

        nav_w = combo_w["NAV_wave"]
        nav_b = combo_w["NAV_bench"]
        wave_ret = nav_w.iloc[-1] / nav_w.iloc[0] - 1.0
        bench_ret = nav_b.iloc[-1] / nav_b.iloc[0] - 1.0
        alpha = wave_ret - bench_ret

        stats_rows.append(
            {
                "Window": label,
                "Wave Return": pct(wave_ret),
                "Benchmark Return": pct(bench_ret),
                "Alpha": pct(alpha),
            }
        )

    stats_df = pd.DataFrame(stats_rows)
    st.table(stats_df)

except Exception as e:
    st.error(f"Error computing performance statistics: {e}")

# ------------------------------------------------------------------------------
# Mode comparison table for the selected Wave (Standard vs AMB vs PL)
# ------------------------------------------------------------------------------

st.subheader("Mode Comparison (Wave vs Benchmark)")

modes_to_compare = ["Standard", "Alpha-Minus-Beta", "Private Logic"]
mode_rows = []

for m in modes_to_compare:
    try:
        hist_wave_m = compute_history_nav(
            selected_wave,
            lookback_days=365,
            mode=m,
            is_benchmark=False,
        )
        hist_bench_m = compute_history_nav(
            benchmark_name,
            lookback_days=365,
            mode="Standard",
            is_benchmark=True,
        )

        combo = pd.concat(
            [
                hist_wave_m[["NAV"]].rename(columns={"NAV": "NAV_wave"}),
                hist_bench_m[["NAV"]].rename(columns={"NAV": "NAV_bench"}),
            ],
            axis=1,
            join="inner",
        ).dropna()

        if combo.empty:
            mode_rows.append(
                {
                    "Mode": m,
                    "Wave 365D Return": "—",
                    "Benchmark 365D Return": "—",
                    "365D Alpha": "—",
                }
            )
            continue

        nav_w = combo["NAV_wave"]
        nav_b = combo["NAV_bench"]
        wave_ret = nav_w.iloc[-1] / nav_w.iloc[0] - 1.0
        bench_ret = nav_b.iloc[-1] / nav_b.iloc[0] - 1.0
        alpha = wave_ret - bench_ret

        mode_rows.append(
            {
                "Mode": m,
                "Wave 365D Return": pct(wave_ret),
                "Benchmark 365D Return": pct(bench_ret),
                "365D Alpha": pct(alpha),
            }
        )

    except Exception as e:
        mode_rows.append(
            {
                "Mode": m,
                "Wave 365D Return": f"Error: {e}",
                "Benchmark 365D Return": "—",
                "365D Alpha": "—",
            }
        )

mode_df = pd.DataFrame(mode_rows)
st.table(mode_df)

# ------------------------------------------------------------------------------
# Top 10 holdings with Google Finance links
# ------------------------------------------------------------------------------

st.subheader("Top 10 Holdings (by target weight)")

try:
    positions = get_wave_positions(selected_wave).copy()
    positions = positions.sort_values("Weight", ascending=False).head(10).reset_index(
        drop=True
    )

    # Build clickable link column
    link_col = []
    for _, row in positions.iterrows():
        t = row["Ticker"]
        url = google_finance_url(t)
        link_col.append(f"[{t}]({url})")

    positions_display = pd.DataFrame(
        {
            "Ticker": link_col,
            "Weight": positions["Weight"].apply(lambda w: f"{w * 100:.2f}%"),
        }
    )

    st.markdown(
        "Top 10 by target weight "
        "(click ticker for Google Finance; weights are target portfolio weights)."
    )
    st.write(positions_display.to_markdown(index=False), unsafe_allow_html=True)

except Exception as e:
    st.error(f"Error loading holdings: {e}")

# ------------------------------------------------------------------------------
# Footer
# ------------------------------------------------------------------------------

st.markdown("---")
st.caption(
    "WAVES Intelligence™ • Vector OS • For demonstration and research purposes only. "
    "Not an offer to sell or solicitation to buy any security."
)