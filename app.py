# app.py
"""
WAVES Intelligence™ Super Console (Combined)

- Uses waves_engine.py (dynamic S&P 500 Wave)
- All Waves Dashboard (table + alpha heatmap)
- Per-Wave deep console:
    - Overview
    - Holdings (Top 10 + Google links)
    - Performance (benchmark & modes)
    - Risk (vol, drawdown, factor exposure)
- Vector Lab:
    - Wave Blender (two-wave slider blend)
    - Scenario Shocks (simple equity shocks)
    - VectorOS Prompt stub (future agent slot)
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
    page_title="WAVES Intelligence™ Super Console",
    layout="wide",
)

st.title("WAVES Intelligence™ Institutional Super Console")
st.caption("Autonomous Waves • Dynamic Alpha • Vector-Ready OS")


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


@st.cache_data(show_spinner=False)
def preload_summaries_for_all_waves(
    weights_df: pd.DataFrame,
) -> dict:
    """
    Precompute summary metrics for all Waves:
    returns dict: { wave_name: summary_dict }
    """
    summaries = {}
    wave_names_local = we.get_wave_names(weights_df)
    for wname in wave_names_local:
        try:
            summaries[wname] = we.compute_wave_summary(wname, weights_df)
        except Exception as e:
            print(f"[WARN] Failed summary for {wname}: {e}")
    return summaries


weights_df = load_all_weights()
wave_names = we.get_wave_names(weights_df)
if not wave_names:
    st.error("No waves found. Check wave_weights.csv and sp500_universe.csv.")
    st.stop()

sector_map_df = load_sector_map()
all_summaries = preload_summaries_for_all_waves(weights_df)


# -------------------------------------------------------------------
# Sidebar controls
# -------------------------------------------------------------------

st.sidebar.header("Wave Controls")
selected_wave = st.sidebar.selectbox("Select Wave", wave_names, index=0)

mode = st.sidebar.radio(
    "Mode (used in Mode Comparison analytics)",
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
    "Mode logic is applied analytically in Performance ➜ Mode Comparison.\n"
    "Engine-level mode separation can be added later."
)


# -------------------------------------------------------------------
# Helper functions
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

    factor_df = factor_df.reindex(tickers)

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

    vals = np.array([momentum_exposure, quality_exposure, hybrid_exposure], dtype=float)
    if np.all(np.isnan(vals)):
        return None

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
    """
    if port_rets is None or port_rets.empty:
        return None

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

    cum_std = (1.0 + wave).cumprod() - 1.0
    amb_rets = 0.90 * bench + alpha
    cum_amb = (1.0 + amb_rets).cumprod() - 1.0
    pl_rets = wave + 0.30 * alpha
    cum_pl = (1.0 + pl_rets).cumprod() - 1.0

    df_list = [
        pd.DataFrame({"date": cum_std.index, "series": "Standard", "value": cum_std.values}),
        pd.DataFrame({"date": cum_amb.index, "series": "Alpha-Minus-Beta", "value": cum_amb.values}),
        pd.DataFrame({"date": cum_pl.index, "series": "Private Logic™", "value": cum_pl.values}),
    ]
    df_modes = pd.concat(df_list, ignore_index=True)
    return df_modes


def build_blended_wave_timeseries(
    wave_a: str,
    wave_b: str,
    blend_a: float,
    days: int,
    weights_df: pd.DataFrame,
) -> pd.DataFrame | None:
    """
    Builds a blended series: blend_a * Wave A + (1 - blend_a) * Wave B
    Based on daily returns.
    Returns tidy DataFrame with columns: date, series, value
    """
    df_a, rets_a, _ = get_wave_timeseries(wave_a, days, weights_df)
    df_b, rets_b, _ = get_wave_timeseries(wave_b, days, weights_df)

    if rets_a is None or rets_a.empty or rets_b is None or rets_b.empty:
        return None

    common_idx = rets_a.index.intersection(rets_b.index)
    if common_idx.empty:
        return None

    rets_a = rets_a.reindex(common_idx)
    rets_b = rets_b.reindex(common_idx)

    blended_rets = blend_a * rets_a + (1.0 - blend_a) * rets_b
    cum_blend = (1.0 + blended_rets).cumprod() - 1.0

    df = pd.DataFrame({"date": cum_blend.index, "series": "Blended Wave", "value": cum_blend.values})
    return df


def compute_shock_scenarios(
    port_rets: pd.Series | None,
    shock_levels: list[float] | None = None,
) -> pd.DataFrame | None:
    """
    Simple scenario approximations:
    - shock_levels: list of instantaneous equity shocks, e.g. [-0.1, -0.2, -0.3]
    """
    if port_rets is None or port_rets.empty:
        return None

    if shock_levels is None:
        shock_levels = [-0.10, -0.20, -0.30]

    recent_vol = float(port_rets.tail(30).std() * np.sqrt(252)) if len(port_rets) >= 30 else None

    rows = []
    for s in shock_levels:
        scenario_name = f"{int(abs(s) * 100)}% Equity Shock"
        est_effect = s
        rows.append(
            {
                "scenario": scenario_name,
                "shock": s,
                "est_effect": est_effect,
                "recent_vol": recent_vol,
            }
        )
    return pd.DataFrame(rows)


# -------------------------------------------------------------------
# All-Waves top summary (super console feel)
# -------------------------------------------------------------------

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
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.sort_values("Wave")


def build_alpha_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert alpha columns into tidy matrix for heatmap:
    columns: Wave, window, alpha
    """
    if df.empty:
        return df

    melted = pd.melt(
        df,
        id_vars=["Wave"],
        value_vars=["Alpha 1D", "Alpha 30D", "Alpha 60D"],
        var_name="window",
        value_name="alpha",
    )
    # Clean window labels
    melted["window"] = melted["window"].str.replace("Alpha ", "")
    return melted


# -------------------------------------------------------------------
# Pull summary metrics for selected wave from preloaded dict
# -------------------------------------------------------------------

summary = all_summaries.get(selected_wave, {})

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

# Placeholder for WaveScore slot
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
# Tabs:
#   All Waves
#   Overview (selected)
#   Holdings
#   Performance
#   Risk
#   Vector Lab
# -------------------------------------------------------------------

tab_all, tab_overview, tab_holdings, tab_perf, tab_risk, tab_vector = st.tabs(
    ["All Waves", "Overview", "Holdings", "Performance", "Risk", "Vector Lab"]
)

# -------------------------------------------------------------------
# ALL WAVES TAB (Super console grid)
# -------------------------------------------------------------------

with tab_all:
    st.subheader("All Waves — Dashboard")

    df_all = build_all_waves_table(all_summaries)
    if df_all.empty:
        st.info("No wave summaries available.")
    else:
        df_disp = df_all.copy()
        st.markdown("#### Wave Metrics Table")
        st.dataframe(
            df_disp.style.format(
                {
                    "Return 1D": "{:.2%}",
                    "Alpha 1D": "{:.2%}",
                    "Return 30D": "{:.2%}",
                    "Alpha 30D": "{:.2%}",
                    "Return 60D": "{:.2%}",
                    "Alpha 60D": "{:.2%}",
                }
            ),
            use_container_width=True,
        )

        st.markdown("#### Alpha Matrix (Heatmap)")
        alpha_mat = build_alpha_matrix(df_all)
        if alpha_mat.empty:
            st.info("Not enough alpha data for heatmap.")
        else:
            chart_alpha = (
                alt.Chart(alpha_mat)
                .mark_rect()
                .encode(
                    x=alt.X("window:N", title="Window"),
                    y=alt.Y("Wave:N", title="Wave"),
                    color=alt.Color(
                        "alpha:Q",
                        title="Alpha",
                        scale=alt.Scale(scheme="redblue"),
                    ),
                    tooltip=[
                        alt.Tooltip("Wave:N", title="Wave"),
                        alt.Tooltip("window:N", title="Window"),
                        alt.Tooltip("alpha:Q", title="Alpha", format=".2%"),
                    ],
                )
                .properties(height=400)
            )
            st.altair_chart(chart_alpha, use_container_width=True)

    st.caption(
        "All-Wave dashboard approximates the old 'grid' console: "
        "return and alpha across multiple horizons, with a simple alpha heatmap."
    )


# -------------------------------------------------------------------
# OVERVIEW TAB (Selected wave)
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
            "- Mode comparison logic (Standard / A−B / Private Logic™) is applied analytically in Performance."
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
        "Risk metrics are based on recent realized volatility, drawdowns, and price-derived factor tilts."
    )


# -------------------------------------------------------------------
# VECTOR LAB TAB
# -------------------------------------------------------------------

with tab_vector:
    st.subheader("Vector Lab — Experiments & OS Toys")

    subtab1, subtab2, subtab3 = st.tabs(
        ["Wave Blender", "Scenario Shocks", "VectorOS Prompt"]
    )

    # ---- Wave Blender ----
    with subtab1:
        st.markdown("### Wave Blender")

        st.caption(
            "Blend your selected Wave with another Wave (e.g., SmartSafe, S&P Wave) "
            "to see how a combined allocation would have behaved."
        )

        other_waves = [w for w in wave_names if w != selected_wave]
        if not other_waves:
            st.info("Need at least 2 Waves to use the blender.")
        else:
            default_other = 0
            for i, wname in enumerate(other_waves):
                if "safe" in wname.lower() or "cash" in wname.lower():
                    default_other = i
                    break

            blend_wave = st.selectbox(
                "Blend with Wave",
                other_waves,
                index=default_other,
            )

            blend_pct = st.slider(
                "Allocation to primary Wave",
                min_value=0,
                max_value=100,
                value=60,
                step=5,
                help="The remainder goes to the secondary Wave.",
            )

            blend_a = blend_pct / 100.0

            df_blend = build_blended_wave_timeseries(
                selected_wave, blend_wave, blend_a, history_window, weights_df
            )

            if df_blend is None or df_blend.empty:
                st.info("Not enough overlapping data to build blended series.")
            else:
                st.markdown(
                    f"**Blend:** {blend_pct}% {selected_wave} + {100 - blend_pct}% {blend_wave}"
                )
                chart_blend = (
                    alt.Chart(df_blend)
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
                st.altair_chart(chart_blend, use_container_width=True)

    # ---- Scenario Shocks ----
    with subtab2:
        st.markdown("### Scenario Shocks")

        st.caption(
            "Very simple hypothetical shocks applied to the selected Wave. "
            "These are not full stress tests, but quick intuition aids."
        )

        df_cum, port_rets, bench_rets = get_wave_timeseries(
            selected_wave, history_window, weights_df
        )
        scenarios = compute_shock_scenarios(port_rets)

        if scenarios is None or scenarios.empty:
            st.info("Not enough data to compute scenarios.")
        else:
            scenarios_disp = scenarios.copy()
            scenarios_disp["Shock (%)"] = scenarios_disp["shock"] * 100.0
            scenarios_disp["Estimated Impact (%)"] = scenarios_disp["est_effect"] * 100.0
            scenarios_disp = scenarios_disp[
                ["scenario", "Shock (%)", "Estimated Impact (%)", "recent_vol"]
            ]
            scenarios_disp.rename(
                columns={
                    "scenario": "Scenario",
                    "recent_vol": "Recent Vol (ann.)",
                },
                inplace=True,
            )

            st.dataframe(
                scenarios_disp.style.format(
                    {
                        "Shock (%)": "{:0.1f}",
                        "Estimated Impact (%)": "{:0.1f}",
                        "Recent Vol (ann.)": "{:0.2%}",
                    }
                ),
                use_container_width=True,
            )

    # ---- VectorOS Prompt Stub ----
    with subtab3:
        st.markdown("### VectorOS Prompt (Stub)")

        st.caption(
            "UI-only stub for the future VectorOS agent. "
            "Use it to prototype how you'd talk to the system."
        )

        default_prompt = (
            "Reallocate 10% from S&P Wave into SmartSafe over the next 30 days, "
            "but only if VIX stays above 25. Show me the expected drawdown improvement."
        )

        user_prompt = st.text_area(
            "Tell Vector what to do:",
            value=default_prompt,
            height=120,
        )

        if st.button("Interpret with VectorOS (placeholder)"):
            st.markdown("#### VectorOS Interpretation (Prototype)")
            st.write("**Raw Instruction:**")
            st.code(user_prompt)

            st.write("**Parsed Intent (example):**")
            st.json(
                {
                    "action": "rebalance",
                    "from_wave": "S&P Wave",
                    "to_wave": "SmartSafe (or cash-like Wave)",
                    "fraction": 0.10,
                    "schedule": "over_30_days",
                    "condition": {
                        "metric": "VIX",
                        "operator": ">",
                        "threshold": 25,
                    },
                    "objective": "reduce_drawdown",
                    "outputs": [
                        "expected_drawdown_change",
                        "before_vs_after_vol",
                        "before_vs_after_allocation",
                    ],
                }
            )

# -------------------------------------------------------------------
# Footer
# -------------------------------------------------------------------

st.markdown("---")
st.caption("WAVES Intelligence™ • Autonomous Wealth Engine • Not investment advice.")