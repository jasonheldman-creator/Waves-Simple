# app.py
"""
WAVES Intelligence™ Institutional Console v2.0 — Stage 2 (Advanced Analytics)

- Uses waves_engine.py (dynamic S&P 500 Wave)
- Multi-tab institutional layout
- 1D / 30D / 60D returns & alpha
- Performance vs benchmark chart
- Rolling alpha chart
- Mode comparison: Standard vs Alpha-Minus-Beta vs Private Logic™
- Top 10 holdings with Google Finance links
- Sector allocation chart (if sector_map.csv is available)
- Factor exposure panel (Momentum & Quality/Low-Vol tilt)
"""

import pandas as pd
import numpy as np
import streamlit as st
import altair as alt

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
# Cached loaders
# -------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_all_weights() -> pd.DataFrame:
    return we.get_dynamic_wave_weights()


@st.cache_data(show_spinner=False)
def load_sector_map(path: str = "sector_map.csv") -> pd.DataFrame | None:
    """
    Optional: sector_map.csv with columns: ticker,sector
    """
    try:
        df = pd.read_csv(path)
        if "ticker" not in df.columns or "sector" not in df.columns:
            return None
        df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
        df["sector"] = df["sector"].astype(str).str.strip()
        return df
    except Exception:
        return None


weights_df = load_all_weights()
wave_names = we.get_wave_names(weights_df)

if not wave_names:
    st.error("No waves found. Check wave_weights.csv and sp500_universe.csv.")
    st.stop()

sector_map_df = load_sector_map()


# -------------------------------------------------------------------
# Sidebar controls
# -------------------------------------------------------------------

st.sidebar.header("Wave Controls")
selected_wave = st.sidebar.selectbox("Select Wave", wave_names, index=0)

mode = st.sidebar.radio(
    "Mode (applied in Mode Comparison analytics)",
    ["Standard", "Alpha-Minus-Beta", "Private Logic™"],
    index=0,
)

history_window = st.sidebar.selectbox(
    "History Window",
    options=[90, 180, 365],
    index=1,
    format_func=lambda x: f"{x} days",
)

st.sidebar.markdown("---")
st.sidebar.caption(
    "Mode logic in this stage is applied in the Performance ➜ Mode Comparison section.\n"
    "Engine-level mode separation can be wired in a later code stage."
)


# -------------------------------------------------------------------
# Helper functions for metrics & charts
# -------------------------------------------------------------------

def fmt_pct(x: float | None) -> str:
    if x is None or pd.isna(x):
        return "—"
    return f"{x * 100:0.2f}%"


def get_wave_timeseries(
    wave_name: str,
    days: int,
    weights_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series | None, pd.Series | None]:
    """
    Build daily series for:
        - wave cumulative return
        - benchmark cumulative return (if exists)
        - rolling alpha (wave - benchmark)

    Returns:
        df_cum (DataFrame with cols 'date','series','value')
        port_rets (Series of daily portfolio returns)
        bench_rets (Series or None)
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
    lookback = days + 10  # pad for weekends/holidays

    prices = we.fetch_price_history(tickers, lookback_days=lookback)
    rets = prices.pct_change().dropna(how="all")
    weights_aligned = weights.reindex(prices.columns).fillna(0.0)
    port_rets = (rets * weights_aligned).sum(axis=1)

    # restrict to last N calendar days
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
            bench_rets = bench_prices.pct_change().dropna().iloc[:, 0]
            # align indices
            common_idx = port_rets.index.intersection(bench_rets.index)
            port_rets = port_rets.reindex(common_idx)
            cum_wave = (1.0 + port_rets).cumprod() - 1.0

            bench_rets = bench_rets.reindex(common_idx)
            cum_bench = (1.0 + bench_rets).cumprod() - 1.0
        except Exception as e:
            st.warning(f"Benchmark series unavailable for {bench_ticker}: {e}")

    # melt into tidy frame for Altair
    df_list = []
    df_wave = pd.DataFrame({"date": cum_wave.index, "series": "Wave", "value": cum_wave.values})
    df_list.append(df_wave)

    if cum_bench is not None:
        df_bench = pd.DataFrame({"date": cum_bench.index, "series": "Benchmark", "value": cum_bench.values})
        df_list.append(df_bench)

    df_cum = pd.concat(df_list, ignore_index=True)

    return df_cum, port_rets, bench_rets


def build_sector_allocation(
    wave_name: str,
    weights_df: pd.DataFrame,
    sector_map: pd.DataFrame | None,
) -> pd.DataFrame | None:
    """
    Returns sector allocation for the wave if a sector_map is available.
    """
    if sector_map is None:
        return None

    w = weights_df[weights_df["wave"] == wave_name].copy()
    if w.empty:
        return None

    agg = (
        w.groupby("ticker")["weight"]
        .sum()
        .reset_index()
    )
    merged = agg.merge(sector_map, how="left", on="ticker")
    merged["sector"].fillna("Other / Unknown", inplace=True)

    sector_alloc = (
        merged.groupby("sector")["weight"]
        .sum()
        .reset_index()
        .sort_values("weight", ascending=False)
    )
    return sector_alloc


def compute_risk_metrics(
    port_rets: pd.Series | None,
    window_short: int = 30,
    window_long: int = 60,
) -> dict:
    """
    Simple risk metrics for the Risk tab.
    """
    metrics: dict = {
        "vol_30d": None,
        "vol_60d": None,
        "max_dd": None,
    }
    if port_rets is None or port_rets.empty:
        return metrics

    # Annualized vol approximations (sqrt(252))
    if len(port_rets) >= window_short:
        metrics["vol_30d"] = float(
            port_rets.tail(window_short).std() * np.sqrt(252)
        )
    if len(port_rets) >= window_long:
        metrics["vol_60d"] = float(
            port_rets.tail(window_long).std() * np.sqrt(252)
        )

    # Max drawdown on available history
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
    """
    Uses waves_engine.compute_factor_scores to estimate factor tilts for a wave.
    Returns DataFrame with columns: factor, exposure
    """
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

    # Align to tickers and weights
    factor_df = factor_df.reindex(tickers)

    # We use three composite factors from the engine:
    #   momentum_score, quality_score, factor_score (hybrid)
    cols_needed = ["momentum_score", "quality_score", "factor_score"]
    for c in cols_needed:
        if c not in factor_df.columns:
            return None

    w_vec = weights.reindex(factor_df.index).fillna(0.0).values

    def w_avg(series: pd.Series) -> float:
        arr = series.fillna(0.0).values
        if w_vec.sum() <= 0:
            return float(np.nan)
        return float((arr * w_vec).sum() / w_vec.sum())

    momentum_exposure = w_avg(factor_df["momentum_score"])
    quality_exposure = w_avg(factor_df["quality_score"])
    hybrid_exposure = w_avg(factor_df["factor_score"])

    # Normalize exposures into something presentable
    vals = np.array([momentum_exposure, quality_exposure, hybrid_exposure], dtype=float)
    if np.all(np.isnan(vals)):
        return None

    # z-score the exposures so 0 = neutral
    mean = np.nanmean(vals)
    std = np.nanstd(vals)
    if std == 0 or np.isnan(std):
        z_vals = np.zeros_like(vals)
    else:
        z_vals = (vals - mean) / std

    df_out = pd.DataFrame(
        {
            "factor": ["Momentum", "Quality / Low-Vol", "Hybrid Factor"],
            "exposure": z_vals,
        }
    )
    return df_out


def build_mode_comparison_series(
    port_rets: pd.Series | None,
    bench_rets: pd.Series | None,
) -> pd.DataFrame | None:
    """
    Synthesizes Standard vs Alpha-Minus-Beta vs Private Logic™
    as three different cumulative curves, using daily rets.

    - Standard: portfolio returns as-is.
    - Alpha-Minus-Beta: 0.9 * benchmark + alpha (wave - bench).
    - Private Logic™: wave + 0.3 * alpha (a more aggressive tilt).

    Returns tidy DataFrame with columns: date, series, value
    """
    if port_rets is None or port_rets.empty:
        return None

    # If no benchmark, just show Standard as the only curve
    if bench_rets is None or bench_rets.empty:
        cum_std = (1.0 + port_rets).cumprod() - 1.0
        df_std = pd.DataFrame(
            {"date": cum_std.index, "series": "Standard", "value": cum_std.values}
        )
        return df_std

    common_idx = port_rets.index.intersection(bench_rets.index)
    if common_idx.empty:
        return None

    wave = port_rets.reindex(common_idx)
    bench = bench_rets.reindex(common_idx)
    alpha = wave - bench

    # Standard
    cum_std = (1.0 + wave).cumprod() - 1.0

    # Alpha-Minus-Beta: keep alpha, reduce beta to ~0.9
    amb_rets = 0.90 * bench + alpha
    cum_amb = (1.0 + amb_rets).cumprod() - 1.0

    # Private Logic™: increase tilt to alpha
    pl_rets = wave + 0.30 * alpha
    cum_pl = (1.0 + pl_rets).cumprod() - 1.0

    df_list = [
        pd.DataFrame({"date": cum_std.index, "series": "Standard", "value": cum_std.values}),
        pd.DataFrame({"date": cum_amb.index, "series": "Alpha-Minus-Beta", "value": cum_amb.values}),
        pd.DataFrame({"date": cum_pl.index, "series": "Private Logic™", "value": cum_pl.values}),
    ]
    df_modes = pd.concat(df_list, ignore_index=True)
    return df_modes


# -------------------------------------------------------------------
# Pull summary metrics from engine
# -------------------------------------------------------------------

with st.spinner(f"Computing summary metrics for {selected_wave}..."):
    summary = we.compute_wave_summary(selected_wave, weights_df)

# 1D / 30D / 60D metrics
r1 = summary.get("return_1d")
a1 = summary.get("alpha_1d")
r30 = summary.get("return_30d")
a30 = summary.get("alpha_30d")
r60 = summary.get("return_60d")
a60 = summary.get("alpha_60d")

bench = summary.get("benchmark")


# -------------------------------------------------------------------
# Top metrics strip
# -------------------------------------------------------------------

st.markdown("### Wave Snapshot")

m1, m2, m3, m4, m5, m6, m7, m8 = st.columns(8)

m1.metric("1-Day Return", fmt_pct(r1))
m2.metric("1-Day Alpha", fmt_pct(a1))
m3.metric("30-Day Return", fmt_pct(r30))
m4.metric("30-Day Alpha", fmt_pct(a30))
m5.metric("60-Day Return", fmt_pct(r60))
m6.metric("60-Day Alpha", fmt_pct(a60))

# Placeholder for WaveScore slot (engine scoring can be attached later)
m7.metric("WaveScore™ (slot)", "—")

# VIX tag
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
# Tabs: Overview | Holdings | Performance | Risk
# -------------------------------------------------------------------

tab_overview, tab_holdings, tab_perf, tab_risk = st.tabs(
    ["Overview", "Holdings", "Performance", "Risk"]
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
        st.markdown("##### Mode & Wave Details")
        st.write(f"**Wave:** {selected_wave}")
        st.write(f"**Selected Mode (for comparison):** {mode}")
        st.write(f"**History Window:** {history_window} days")
        if bench:
            st.write(f"**Benchmark:** `{bench}`")
        else:
            st.write("**Benchmark:** Not mapped")

        st.markdown("---")
        st.markdown("##### Quick Notes")
        st.caption(
            "- Performance curves use realized daily returns from the current Wave definition.\n"
            "- Mode comparison logic (Standard / A−B / Private Logic™) is applied analytically in Stage 2."
        )

    st.markdown("---")
    st.markdown("##### Top 10 Holdings (Preview)")
    top = summary.get("top_holdings")
    if top is None or top.empty:
        st.write("No holdings found for this Wave.")
    else:
        top_disp = top.copy()
        top_disp["weight_pct"] = top_disp["weight"] * 100.0
        top_disp = top_disp[["ticker", "weight_pct"]]
        top_disp.columns = ["Ticker", "Weight (%)"]
        st.dataframe(
            top_disp.style.format({"Weight (%)": "{:0.2f}"}),
            use_container_width=True,
        )

# -------------------------------------------------------------------
# HOLDINGS TAB
# -------------------------------------------------------------------

with tab_holdings:
    st.subheader(f"Holdings — {selected_wave}")

    top = summary.get("top_holdings")
    if top is None or top.empty:
        st.write("No holdings for this Wave.")
    else:
        st.markdown("##### Top 10 Holdings Table")

        top_disp = top.copy()
        top_disp["Weight (%)"] = top_disp["weight"] * 100.0
        top_disp = top_disp[["ticker", "Weight (%)"]]
        top_disp.columns = ["Ticker", "Weight (%)"]

        st.dataframe(
            top_disp.style.format({"Weight (%)": "{:0.2f}"}),
            use_container_width=True,
        )

        st.markdown("##### Google Finance Links")
        for _, row in top.iterrows():
            ticker = row["ticker"]
            wgt = row["weight"]
            url = f"https://www.google.com/finance?q={ticker}"
            st.markdown(
                f"- [{ticker}]({url}) — {wgt * 100:0.2f}% weight",
                unsafe_allow_html=True,
            )

        st.markdown("---")

        st.markdown("##### Sector Allocation (if mapped)")
        sector_alloc = build_sector_allocation(
            selected_wave, weights_df, sector_map_df
        )
        if sector_alloc is None or sector_alloc.empty:
            st.info(
                "No sector_map.csv found or no sectors mapped. "
                "Add a sector_map.csv (ticker,sector) to enable this chart."
            )
        else:
            df_sec = sector_alloc.copy()
            df_sec["Weight (%)"] = df_sec["weight"] * 100.0

            chart_sector = (
                alt.Chart(df_sec)
                .mark_bar()
                .encode(
                    x=alt.X("Weight (%):Q", title="Weight (%)"),
                    y=alt.Y("sector:N", sort="-x", title="Sector"),
                    tooltip=[
                        alt.Tooltip("sector:N", title="Sector"),
                        alt.Tooltip("Weight (%):Q", format=".2f"),
                    ],
                )
                .properties(height=400)
            )

            st.altair_chart(chart_sector, use_container_width=True)

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
            common_idx = port_rets.index.intersection(bench_rets.index)
            port = port_rets.reindex(common_idx)
            bench = bench_rets.reindex(common_idx)
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
        st.info(
            "Not enough data (or no benchmark) to show mode comparison. "
            "Standard curve is used elsewhere."
        )
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
        "Engine-level mode separation (distinct portfolios per mode) can be added on top later."
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
            "Factor exposure could not be computed (insufficient history or data). "
            "Ensure tickers have ~1 year of price history for best results."
        )
    else:
        chart_factor = (
            alt.Chart(factor_exposure)
            .mark_bar()
            .encode(
                x=alt.X("factor:N", title="Factor"),
                y=alt.Y("exposure:Q", title="Exposure (relative z-score)"),
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
        "Risk metrics are based on recent realized volatility, drawdowns, and price-derived factor tilts. "
        "A full WAVES Risk Engine (stress tests, scenario analysis, factor risk) can be layered in a future stage."
    )

# -------------------------------------------------------------------
# Footer
# -------------------------------------------------------------------

st.markdown("---")
st.caption("WAVES Intelligence™ • Autonomous Wealth Engine • Not investment advice.")