# app.py
"""
WAVES Intelligence™ Console — Emergency Vector 1 Fallback

Simple, stable console:

- Sidebar:
    • Wave selector
    • History window selector
- Top strip:
    • 1D / 30D / 60D returns + alpha
    • WaveScore slot
    • VIX reference
- Main panel:
    • Performance chart (Wave vs Benchmark)
    • Top 10 holdings table
    • Google Finance links for holdings
- Bottom:
    • All Waves summary table (returns & alpha)

Everything is wrapped in try/except so that failures
show as messages instead of crashing or blank screens.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

import waves_engine as we


# ------------------------------------------------------------
# Small helpers
# ------------------------------------------------------------

def fmt_pct(x):
    if x is None or pd.isna(x):
        return "—"
    return f"{x * 100:0.2f}%"


def get_wave_timeseries(wave_name, days, weights_df):
    """
    Very simple Wave vs Benchmark timeseries.
    Uses waves_engine.fetch_price_history and benchmark mapping.
    """
    w = weights_df[weights_df["wave"] == wave_name].copy()
    if w.empty:
        return pd.DataFrame(), None, None

    weights = (
        w.groupby("ticker")["weight"]
        .sum()
        .reindex(w["ticker"].unique())
        .fillna(0.0)
    )
    tickers = list(weights.index)
    lookback = days + 10

    prices = we.fetch_price_history(tickers, lookback_days=lookback)
    if prices.empty:
        return pd.DataFrame(), None, None

    rets = prices.pct_change().dropna(how="all")
    weights_aligned = weights.reindex(prices.columns).fillna(0.0)
    port_rets = (rets * weights_aligned).sum(axis=1)

    port_rets = port_rets.tail(days)
    if port_rets.empty:
        return pd.DataFrame(), None, None

    cum_wave = (1.0 + port_rets).cumprod() - 1.0

    bench_ticker = we.get_benchmark_for_wave(wave_name)
    bench_rets = None
    cum_bench = None
    if bench_ticker:
        try:
            bench_prices = we.fetch_price_history([bench_ticker], lookback_days=lookback)
            if not bench_prices.empty:
                bench_rets = bench_prices.iloc[:, 0].pct_change().dropna()
                common = port_rets.index.intersection(bench_rets.index)
                port_rets = port_rets.reindex(common)
                bench_rets = bench_rets.reindex(common)
                cum_wave = (1.0 + port_rets).cumprod() - 1.0
                cum_bench = (1.0 + bench_rets).cumprod() - 1.0
        except Exception as e:
            st.warning(f"Benchmark series unavailable for {bench_ticker}: {e}")

    frames = [
        pd.DataFrame({"date": cum_wave.index, "series": "Wave", "value": cum_wave.values})
    ]
    if cum_bench is not None:
        frames.append(
            pd.DataFrame({"date": cum_bench.index, "series": "Benchmark", "value": cum_bench.values})
        )
    df_cum = pd.concat(frames, ignore_index=True)
    return df_cum, port_rets, bench_rets


def build_all_waves_table(summaries: dict) -> pd.DataFrame:
    rows = []
    for wname, summ in summaries.items():
        if not isinstance(summ, dict):
            continue
        rows.append(
            {
                "Wave": wname,
                "Return 1D": summ.get("return_1d"),
                "Alpha 1D": summ.get("alpha_1d"),
                "Return 30D": summ.get("return_30d"),
                "Alpha 30D": summ.get("alpha_30d"),
                "Return 60D": summ.get("return_60d"),
                "Alpha 60D": summ.get("alpha_60d"),
            }
        )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("Wave")


# ------------------------------------------------------------
# Main app (wrapped in try/except)
# ------------------------------------------------------------

st.set_page_config(page_title="WAVES Intelligence™ Console", layout="wide")

try:
    st.title("WAVES Intelligence™ Institutional Console")
    st.caption("Vector 1 • Emergency Stable Build • Autonomous Wealth Engine")

    # Load weights & summaries
    @st.cache_data(show_spinner=False)
    def load_all_weights():
        return we.get_dynamic_wave_weights()

    @st.cache_data(show_spinner=False)
    def preload_summaries(weights_df):
        summaries = {}
        for wname in we.get_wave_names(weights_df):
            try:
                summaries[wname] = we.compute_wave_summary(wname, weights_df)
            except Exception as e:
                print(f"[WARN] summary failed for {wname}: {e}")
        return summaries

    weights_df = load_all_weights()
    if weights_df is None or weights_df.empty:
        st.error("No Waves found. Check wave_weights.csv and sp500_universe.csv.")
        st.stop()

    wave_names = we.get_wave_names(weights_df)
    all_summaries = preload_summaries(weights_df)

    # Sidebar controls
    st.sidebar.header("Wave Controls")
    selected_wave = st.sidebar.selectbox("Select Wave", wave_names, index=0)
    history_window = st.sidebar.selectbox(
        "History Window",
        options=[30, 60, 90, 180],
        index=1,
        format_func=lambda x: f"{x} days",
    )
    st.sidebar.markdown("---")
    st.sidebar.caption("Stable Vector 1 console — minimal but robust.")

    # Selected wave summary
    summary = all_summaries.get(selected_wave, {})
    r1 = summary.get("return_1d")
    a1 = summary.get("alpha_1d")
    r30 = summary.get("return_30d")
    a30 = summary.get("alpha_30d")
    r60 = summary.get("return_60d")
    a60 = summary.get("alpha_60d")
    bench = summary.get("benchmark")
    top_holdings = summary.get("top_holdings")

    # Metrics strip
    st.markdown("### Wave Snapshot")
    m1, m2, m3, m4, m5, m6, m7, m8 = st.columns(8)
    m1.metric("1-Day Return", fmt_pct(r1))
    m2.metric("1-Day Alpha", fmt_pct(a1))
    m3.metric("30-Day Return", fmt_pct(r30))
    m4.metric("30-Day Alpha", fmt_pct(a30))
    m5.metric("60-Day Return", fmt_pct(r60))
    m6.metric("60-Day Alpha", fmt_pct(a60))
    m7.metric("WaveScore™ (slot)", "—")

    try:
        vix_level = we.get_vix_level()
        m8.metric("VIX (Ref)", f"{vix_level:0.1f}" if vix_level is not None else "—")
    except Exception:
        m8.metric("VIX (Ref)", "—")

    if bench:
        st.caption(f"Benchmark for **{selected_wave}**: `{bench}`")
    else:
        st.caption(f"No specific benchmark mapped yet for **{selected_wave}**.")

    st.markdown("---")

    # Layout: chart + holdings
    col_left, col_right = st.columns([2, 1])

    # Left: performance chart
    with col_left:
        st.subheader("Performance vs Benchmark")
        df_cum, port_rets, bench_rets = get_wave_timeseries(
            selected_wave, history_window, weights_df
        )
        if df_cum.empty:
            st.info("Not enough data to build performance chart.")
        else:
            chart = (
                alt.Chart(df_cum)
                .mark_line()
                .encode(
                    x=alt.X("date:T", title="Date"),
                    y=alt.Y("value:Q", title="Cumulative Return"),
                    color=alt.Color("series:N", title="Series"),
                    tooltip=[
                        alt.Tooltip("date:T", title="Date"),
                        alt.Tooltip("series:N", title="Series"),
                        alt.Tooltip("value:Q", title="Cumulative Return", format=".2%"),
                    ],
                )
                .properties(height=380)
            )
            st.altair_chart(chart, use_container_width=True)

    # Right: holdings
    with col_right:
        st.subheader("Top 10 Holdings")
        if top_holdings is None or top_holdings.empty:
            st.write("No holdings found for this Wave.")
        else:
            top_disp = top_holdings.copy()
            top_disp["Weight (%)"] = (top_disp["weight"] * 100.0).round(2)
            st.dataframe(top_disp[["ticker", "Weight (%)"]], use_container_width=True)

            st.markdown("##### Google Finance Links")
            for _, row in top_holdings.iterrows():
                ticker = str(row["ticker"])
                wgt = row["weight"]
                url = f"https://www.google.com/finance?q={ticker}"
                st.markdown(
                    f"- [{ticker}]({url}) — {wgt * 100:0.2f}% weight",
                    unsafe_allow_html=True,
                )

    # All Waves table
    st.markdown("---")
    st.subheader("All Waves — Returns & Alpha")

    df_all = build_all_waves_table(all_summaries)
    if df_all.empty:
        st.info("No wave summaries available.")
    else:
        df_show = df_all.copy()
        for col in [
            "Return 1D",
            "Alpha 1D",
            "Return 30D",
            "Alpha 30D",
            "Return 60D",
            "Alpha 60D",
        ]:
            if col in df_show.columns:
                df_show[col] = (df_show[col] * 100.0).round(2)
        st.dataframe(df_show, use_container_width=True)

    st.markdown("---")
    st.caption("WAVES Intelligence™ • Vector 1 Stable Console • Not investment advice.")

except Exception as e:
    # Last-resort guard so you never see a blank page
    st.error("Console encountered an error while loading.")
    st.exception(e)