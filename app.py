# app.py
"""
WAVES Intelligence™ Console — Vector 1 Final (Version C)

Features:
- Sidebar: wave selector, mode selector, history window
- Top strip: 1D / 30D / 60D returns & alpha, WaveScore slot, VIX
- Tabs:
    • All Waves — grid + alpha matrix heatmap
    • Overview — perf vs benchmark + quick holdings preview
    • Holdings — Top 10 + Google links
    • Performance — cumulative + rolling alpha + mode comparison
    • Risk — vol, drawdown, factor exposure

No pandas Styler (.style). Safe for Streamlit Cloud.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

import waves_engine as we


# -------------------------------------------------------------------
# Page config
# -------------------------------------------------------------------

st.set_page_config(
    page_title="WAVES Intelligence™ Console — Vector 1",
    layout="wide",
)

st.title("WAVES Intelligence™ Institutional Console (Vector 1)")
st.caption("Autonomous Waves • Alpha-Minus-Beta Modes • Vector-Ready")


# -------------------------------------------------------------------
# Cached loaders
# -------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_all_weights() -> pd.DataFrame:
    return we.get_dynamic_wave_weights()


@st.cache_data(show_spinner=False)
def preload_summaries_for_all_waves(weights_df: pd.DataFrame) -> dict:
    summaries: dict = {}
    for wname in we.get_wave_names(weights_df):
        try:
            summaries[wname] = we.compute_wave_summary(wname, weights_df)
        except Exception as e:
            print(f"[WARN] summary failed for {wname}: {e}")
    return summaries


weights_df = load_all_weights()
wave_names = we.get_wave_names(weights_df)
if not wave_names:
    st.error("No Waves found. Check wave_weights.csv and sp500_universe.csv.")
    st.stop()

all_summaries = preload_summaries_for_all_waves(weights_df)


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------

def fmt_pct(x: float | None) -> str:
    if x is None or pd.isna(x):
        return "—"
    return f"{x * 100:0.2f}%"


def get_wave_timeseries(
    wave_name: str,
    days: int,
    weights_df: pd.DataFrame,
):
    """
    Return tidy cumulative series + daily returns for a wave and its benchmark.
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
                bench_rets = bench_prices.pct_change().dropna().iloc[:, 0]
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


def compute_risk_metrics(
    port_rets: pd.Series | None,
    window_short: int = 30,
    window_long: int = 60,
) -> dict:
    metrics = {"vol_30d": None, "vol_60d": None, "max_dd": None}
    if port_rets is None or port_rets.empty:
        return metrics

    if len(port_rets) >= window_short:
        metrics["vol_30d"] = float(port_rets.tail(window_short).std() * np.sqrt(252))
    if len(port_rets) >= window_long:
        metrics["vol_60d"] = float(port_rets.tail(window_long).std() * np.sqrt(252))

    cum = (1.0 + port_rets).cumprod()
    running_max = cum.cummax()
    dd = cum / running_max - 1.0
    metrics["max_dd"] = float(dd.min())
    return metrics


def compute_factor_exposure_for_wave(
    wave_name: str,
    weights_df: pd.DataFrame,
    days: int = 260,
) -> pd.DataFrame | None:
    w = weights_df[weights_df["wave"] == wave_name].copy()
    if w.empty:
        return None

    weights = (
        w.groupby("ticker")["weight"]
        .sum()
        .reindex(w["ticker"].unique())
        .fillna(0.0)
    )
    tickers = list(weights.index)
    if not tickers:
        return None

    try:
        prices = we.fetch_price_history(tickers, lookback_days=days)
        factor_df = we.compute_factor_scores(prices)
    except Exception as e:
        st.warning(f"Factor data unavailable for {wave_name}: {e}")
        return None

    factor_df = factor_df.reindex(tickers)
    needed = ["momentum_score", "quality_score", "factor_score"]
    if any(c not in factor_df.columns for c in needed):
        return None

    w_vec = weights.reindex(factor_df.index).fillna(0.0).values

    def w_avg(series: pd.Series) -> float:
        arr = series.fillna(0.0).values
        if w_vec.sum() <= 0:
            return float("nan")
        return float((arr * w_vec).sum() / w_vec.sum())

    m = w_avg(factor_df["momentum_score"])
    q = w_avg(factor_df["quality_score"])
    h = w_avg(factor_df["factor_score"])

    vals = np.array([m, q, h], dtype=float)
    if np.all(np.isnan(vals)):
        return None
    mean = np.nanmean(vals)
    std = np.nanstd(vals)
    if std == 0 or np.isnan(std):
        z_vals = np.zeros_like(vals)
    else:
        z_vals = (vals - mean) / std

    return pd.DataFrame(
        {
            "factor": ["Momentum", "Quality / Low-Vol", "Hybrid Factor"],
            "exposure": z_vals,
        }
    )


def build_mode_comparison_series(
    port_rets: pd.Series | None,
    bench_rets: pd.Series | None,
) -> pd.DataFrame | None:
    if port_rets is None or port_rets.empty:
        return None
    if bench_rets is None or bench_rets.empty:
        cum_std = (1.0 + port_rets).cumprod() - 1.0
        return pd.DataFrame(
            {"date": cum_std.index, "series": "Standard", "value": cum_std.values}
        )

    common = port_rets.index.intersection(bench_rets.index)
    if common.empty:
        return None

    wave = port_rets.reindex(common)
    bench = bench_rets.reindex(common)
    alpha = wave - bench

    cum_std = (1.0 + wave).cumprod() - 1.0
    amb_rets = 0.90 * bench + alpha
    cum_amb = (1.0 + amb_rets).cumprod() - 1.0
    pl_rets = wave + 0.30 * alpha
    cum_pl = (1.0 + pl_rets).cumprod() - 1.0

    return pd.concat(
        [
            pd.DataFrame({"date": cum_std.index, "series": "Standard", "value": cum_std.values}),
            pd.DataFrame({"date": cum_amb.index, "series": "Alpha-Minus-Beta", "value": cum_amb.values}),
            pd.DataFrame({"date": cum_pl.index, "series": "Private Logic™", "value": cum_pl.values}),
        ],
        ignore_index=True,
    )


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


def build_alpha_matrix(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    melted = pd.melt(
        df,
        id_vars=["Wave"],
        value_vars=["Alpha 1D", "Alpha 30D", "Alpha 60D"],
        var_name="Window",
        value_name="Alpha",
    )
    melted["Window"] = melted["Window"].str.replace("Alpha ", "")
    return melted


# -------------------------------------------------------------------
# Sidebar controls
# -------------------------------------------------------------------

st.sidebar.header("Wave Controls")
selected_wave = st.sidebar.selectbox("Select Wave", wave_names, index=0)

mode = st.sidebar.radio(
    "Mode (for comparison curves)",
    ["Standard", "Alpha-Minus-Beta", "Private Logic™"],
    index=0,
)

history_window = st.sidebar.selectbox(
    "History Window",
    options=[30, 60, 90, 180],
    index=2,
    format_func=lambda x: f"{x} days",
)

st.sidebar.markdown("---")
st.sidebar.caption("Vector 1 Final — stable multi-tab console.")


# -------------------------------------------------------------------
# Selected wave summary
# -------------------------------------------------------------------

summary = all_summaries.get(selected_wave, {})
r1 = summary.get("return_1d")
a1 = summary.get("alpha_1d")
r30 = summary.get("return_30d")
a30 = summary.get("alpha_30d")
r60 = summary.get("return_60d")
a60 = summary.get("alpha_60d")
bench = summary.get("benchmark")
top_holdings = summary.get("top_holdings")


# -------------------------------------------------------------------
# Top metrics
# -------------------------------------------------------------------

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
    m8.metric("VIX (Ref)", f"{vix_level:0.1f}")
except Exception:
    m8.metric("VIX (Ref)", "—")

if bench:
    st.caption(f"Benchmark for **{selected_wave}**: `{bench}`")
else:
    st.caption(f"No specific benchmark mapped yet for **{selected_wave}**.")


# -------------------------------------------------------------------
# Tabs
# -------------------------------------------------------------------

tab_all, tab_overview, tab_holdings, tab_perf, tab_risk = st.tabs(
    ["All Waves", "Overview", "Holdings", "Performance", "Risk"]
)

# -------------------------------------------------------------------
# ALL WAVES TAB
# -------------------------------------------------------------------

with tab_all:
    st.subheader("All Waves — Dashboard")

    df_all = build_all_waves_table(all_summaries)
    if df_all.empty:
        st.info("No wave summaries available.")
    else:
        st.markdown("#### Wave Metrics Table")
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

        st.markdown("#### Alpha Matrix (Heatmap)")
        alpha_mat = build_alpha_matrix(df_all)
        if alpha_mat.empty:
            st.info("Not enough alpha data for heatmap.")
        else:
            chart_alpha = (
                alt.Chart(alpha_mat)
                .mark_rect()
                .encode(
                    x=alt.X("Window:N", title="Window"),
                    y=alt.Y("Wave:N", title="Wave"),
                    color=alt.Color("Alpha:Q", title="Alpha", scale=alt.Scale(scheme="redblue")),
                    tooltip=[
                        alt.Tooltip("Wave:N", title="Wave"),
                        alt.Tooltip("Window:N", title="Window"),
                        alt.Tooltip("Alpha:Q", title="Alpha", format=".2%"),
                    ],
                )
                .properties(height=400)
            )
            st.altair_chart(chart_alpha, use_container_width=True)

    st.caption(
        "All-Waves dashboard: returns and alpha across horizons, plus a simple alpha heatmap."
    )


# -------------------------------------------------------------------
# OVERVIEW TAB
# -------------------------------------------------------------------

with tab_overview:
    st.subheader(f"Overview — {selected_wave}")

    df_cum, port_rets, bench_rets = get_wave_timeseries(
        selected_wave, history_window, weights_df
    )

    col_left, col_right = st.columns([2, 1])
    with col_left:
        st.markdown("##### Performance vs Benchmark")
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
                .properties(height=350)
            )
            st.altair_chart(chart, use_container_width=True)

    with col_right:
        st.markdown("##### Wave Details")
        st.write(f"**Wave:** {selected_wave}")
        st.write(f"**Selected Mode (for comparison curves):** {mode}")
        st.write(f"**History Window:** {history_window} days")
        st.write(f"**Benchmark:** `{bench}`" if bench else "**Benchmark:** Not mapped")
        st.markdown("---")
        st.caption(
            "Performance curves use realized daily returns from the current Wave definition.\n"
            "Mode comparison logic is shown in the Performance tab."
        )

    st.markdown("---")
    st.markdown("##### Top 10 Holdings (Preview)")
    if top_holdings is None or top_holdings.empty:
        st.write("No holdings found for this Wave.")
    else:
        top_disp = top_holdings.copy()
        top_disp["Weight (%)"] = (top_disp["weight"] * 100.0).round(2)
        st.dataframe(top_disp[["ticker", "Weight (%)"]], use_container_width=True)


# -------------------------------------------------------------------
# HOLDINGS TAB
# -------------------------------------------------------------------

with tab_holdings:
    st.subheader(f"Holdings — {selected_wave}")

    if top_holdings is None or top_holdings.empty:
        st.write("No holdings for this Wave.")
    else:
        st.markdown("##### Top 10 Holdings Table")
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


# -------------------------------------------------------------------
# PERFORMANCE TAB
# -------------------------------------------------------------------

with tab_perf:
    st.subheader(f"Performance — {selected_wave}")

    df_cum, port_rets, bench_rets = get_wave_timeseries(
        selected_wave, history_window, weights_df
    )

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("##### Cumulative Performance")
        if df_cum.empty:
            st.info("Not enough data to build performance chart.")
        else:
            chart_perf = (
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
                .properties(height=350)
            )
            st.altair_chart(chart_perf, use_container_width=True)

    with col2:
        st.markdown("##### Rolling Alpha (Wave − Benchmark)")
        if df_cum.empty or bench_rets is None:
            st.info("Benchmark series not available; alpha chart skipped.")
        else:
            common = port_rets.index.intersection(bench_rets.index)
            port = port_rets.reindex(common)
            bench = bench_rets.reindex(common)
            alpha_daily = port - bench
            alpha_cum = (1.0 + alpha_daily).cumprod() - 1.0
            df_alpha = pd.DataFrame(
                {"date": alpha_cum.index, "alpha": alpha_cum.values}
            )
            chart_alpha = (
                alt.Chart(df_alpha)
                .mark_line()
                .encode(
                    x=alt.X("date:T", title="Date"),
                    y=alt.Y("alpha:Q", title="Cumulative Alpha"),
                    tooltip=[
                        alt.Tooltip("date:T", title="Date"),
                        alt.Tooltip("alpha:Q", title="Cumulative Alpha", format=".2%"),
                    ],
                )
                .properties(height=350)
            )
            st.altair_chart(chart_alpha, use_container_width=True)

    st.markdown("---")
    st.markdown("##### Mode Comparison — Standard vs A−B vs Private Logic™")

    df_modes = build_mode_comparison_series(port_rets, bench_rets)
    if df_modes is None or df_modes.empty:
        st.info("Not enough data (or no benchmark) to show mode comparison.")
    else:
        chart_modes = (
            alt.Chart(df_modes)
            .mark_line()
            .encode(
                x=alt.X("date:T", title="Date"),
                y=alt.Y("value:Q", title="Cumulative Return"),
                color=alt.Color("series:N", title="Mode"),
                tooltip=[
                    alt.Tooltip("date:T", title="Date"),
                    alt.Tooltip("series:N", title="Mode"),
                    alt.Tooltip("value:Q", title="Cumulative Return", format=".2%"),
                ],
            )
            .properties(height=350)
        )
        st.altair_chart(chart_modes, use_container_width=True)

    st.markdown("---")
    st.caption(
        "Mode comparison curves are analytical constructs derived from realized returns and benchmarks.\n"
        "Engine-level mode separation (distinct portfolios per mode) can be added later."
    )


# -------------------------------------------------------------------
# RISK TAB
# -------------------------------------------------------------------

with tab_risk:
    st.subheader(f"Risk — {selected_wave}")

    df_cum, port_rets, bench_rets = get_wave_timeseries(
        selected_wave, history_window, weights_df
    )

    risk = compute_risk_metrics(port_rets)
    c1, c2, c3 = st.columns(3)
    c1.metric("30-Day Vol (ann.)", fmt_pct(risk.get("vol_30d")))
    c2.metric("60-Day Vol (ann.)", fmt_pct(risk.get("vol_60d")))
    c3.metric("Max Drawdown", fmt_pct(risk.get("max_dd")))

    st.markdown("---")
    st.markdown("##### Drawdown Curve")
    if port_rets is None or port_rets.empty:
        st.info("Not enough data to show drawdown.")
    else:
        cum = (1.0 + port_rets).cumprod()
        running_max = cum.cummax()
        dd = cum / running_max - 1.0
        df_dd = pd.DataFrame({"date": dd.index, "drawdown": dd.values})
        chart_dd = (
            alt.Chart(df_dd)
            .mark_area()
            .encode(
                x=alt.X("date:T", title="Date"),
                y=alt.Y("drawdown:Q", title="Drawdown"),
                tooltip=[
                    alt.Tooltip("date:T", title="Date"),
                    alt.Tooltip("drawdown:Q", title="Drawdown", format=".2%"),
                ],
            )
            .properties(height=350)
        )
        st.altair_chart(chart_dd, use_container_width=True)

    st.markdown("---")
    st.markdown("##### Factor Exposure (Momentum & Quality Tilt)")
    factor_exposure = compute_factor_exposure_for_wave(
        selected_wave, weights_df, days=260
    )
    if factor_exposure is None or factor_exposure.empty:
        st.info(
            "Factor exposure could not be computed (insufficient history or data)."
        )
    else:
        chart_factor = (
            alt.Chart(factor_exposure)
            .mark_bar()
            .encode(
                x=alt.X("factor:N", title="Factor"),
                y=alt.Y("exposure:Q", title="Exposure (rel. z-score)"),
                tooltip=[
                    alt.Tooltip("factor:N", title="Factor"),
                    alt.Tooltip("exposure:Q", title="Exposure", format=".2f"),
                ],
            )
            .properties(height=300)
        )
        st.altair_chart(chart_factor, use_container_width=True)

    st.markdown("---")
    st.caption(
        "Risk metrics are based on recent realized volatility, drawdowns, and price-derived factor tilts."
    )


# -------------------------------------------------------------------
# Footer
# -------------------------------------------------------------------

st.markdown("---")
st.caption("WAVES Intelligence™ • Vector 1 Final • Not investment advice.")