# app.py — WAVES Intelligence™ Institutional Console
# Mini-Bloomberg style dashboard for WAVES Intelligence™
#
# Tabs:
#   • Console: full WAVES dashboard
#   • Market Intel: global market overview, events, and Wave reactions
#   • Factor Decomposition: factor betas + correlation matrix
#   • Vector OS Insights: AI-style narrative / chat layer
#
# Console features:
#   • Market Regime Monitor: SPY vs VIX on one chart
#   • Portfolio-Level Overview (All Waves)
#   • Multi-Window Alpha Capture (1D, 30D, 60D, 365D)
#   • Benchmark ETF Mix table
#   • Risk & WaveScore Ingredients (All Waves)
#   • WaveScore Leaderboard (Preview)
#   • Wave Detail view:
#       – NAV chart (Wave vs benchmark)
#       – Performance vs benchmark (30D / 365D)
#       – Mode comparison (Standard, AMB, PL)
#       – Top-10 holdings with Google Finance links
#
# Market Intel features:
#   • Global Market Dashboard (SPY, QQQ, IWM, TLT, GLD, BTC, VIX, 10Y)
#   • Market Regime Monitor (SPY vs VIX) + regime label
#   • Macro & Events panel (earnings + themes)
#   • WAVES Reaction Snapshot (30D return/alpha + narrative)
#
# Factor Decomposition features:
#   • Uses daily Wave returns from compute_history_nav(...)
#   • Regresses vs daily returns of SPY, QQQ, IWM, TLT, GLD, BTC-USD
#   • Outputs factor betas for each Wave
#   • Bar chart of factor loadings for selected Wave
#   • Wave Correlation Matrix (Wave daily returns, heatmap + table)
#
# Vector OS Insights:
#   • Wave-level, horizon-specific narrative using all computed stats
#   • “Question for Vector” text box for flavor (local, rule-based)

from __future__ import annotations

import math
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import waves_engine as we  # your engine module

try:
    import yfinance as yf
except ImportError:
    yf = None


# ------------------------------------------------------------
# Streamlit config
# ------------------------------------------------------------

st.set_page_config(
    page_title="WAVES Intelligence™ Console",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ------------------------------------------------------------
# Helpers: data fetching & caching
# ------------------------------------------------------------

@st.cache_data(show_spinner=False)
def fetch_spy_vix(days: int = 365) -> pd.DataFrame:
    """
    Fetch SPY and VIX history, last N days.
    Returns DataFrame with columns ['SPY', '^VIX'] and Date index.
    """
    if yf is None:
        return pd.DataFrame()

    end = datetime.utcnow().date()
    start = end - timedelta(days=days + 10)
    tickers = ["SPY", "^VIX"]

    data = yf.download(
        tickers=tickers,
        start=start.isoformat(),
        end=end.isoformat(),
        interval="1d",
        auto_adjust=True,
        progress=False,
        group_by="column",
    )

    if isinstance(data.columns, pd.MultiIndex):
        if "Adj Close" in data.columns.get_level_values(0):
            data = data["Adj Close"]
        elif "Close" in data.columns.get_level_values(0):
            data = data["Close"]
        else:
            data = data[data.columns.levels[0][0]]

    if isinstance(data.columns, pd.MultiIndex):
        data = data.droplevel(0, axis=1)

    if isinstance(data, pd.Series):
        data = data.to_frame()

    data = data.sort_index().ffill().bfill()
    if len(data) > days:
        data = data.iloc[-days:]

    return data


@st.cache_data(show_spinner=False)
def fetch_market_assets(days: int = 365) -> pd.DataFrame:
    """
    Fetch multi-asset history for the Market Intel & Factor dashboards.
    Assets: SPY, QQQ, IWM, TLT, GLD, BTC-USD, ^VIX, ^TNX
    Returns cleaned price DataFrame.
    """
    if yf is None:
        return pd.DataFrame()

    end = datetime.utcnow().date()
    start = end - timedelta(days=days + 10)
    tickers = ["SPY", "QQQ", "IWM", "TLT", "GLD", "BTC-USD", "^VIX", "^TNX"]

    data = yf.download(
        tickers=tickers,
        start=start.isoformat(),
        end=end.isoformat(),
        interval="1d",
        auto_adjust=True,
        progress=False,
        group_by="column",
    )

    if isinstance(data.columns, pd.MultiIndex):
        if "Adj Close" in data.columns.get_level_values(0):
            data = data["Adj Close"]
        elif "Close" in data.columns.get_level_values(0):
            data = data["Close"]
        else:
            data = data[data.columns.levels[0][0]]

    if isinstance(data.columns, pd.MultiIndex):
        data = data.droplevel(0, axis=1)

    if isinstance(data, pd.Series):
        data = data.to_frame()

    data = data.sort_index().ffill().bfill()
    if len(data) > days:
        data = data.iloc[-days:]

    return data


@st.cache_data(show_spinner=False)
def compute_wave_history(
    wave_name: str,
    mode: str,
    days: int = 365,
) -> pd.DataFrame:
    """
    Wrapper around we.compute_history_nav with caching.
    Returns DataFrame with ['wave_nav', 'bm_nav', 'wave_ret', 'bm_ret'].
    """
    try:
        df = we.compute_history_nav(wave_name, mode=mode, days=days)
    except Exception:
        df = pd.DataFrame(columns=["wave_nav", "bm_nav", "wave_ret", "bm_ret"])
    return df


@st.cache_data(show_spinner=False)
def get_benchmark_mix() -> pd.DataFrame:
    try:
        return we.get_benchmark_mix_table()
    except Exception:
        return pd.DataFrame(columns=["Wave", "Ticker", "Name", "Weight"])


@st.cache_data(show_spinner=False)
def get_wave_holdings(wave_name: str) -> pd.DataFrame:
    try:
        return we.get_wave_holdings(wave_name)
    except Exception:
        return pd.DataFrame(columns=["Ticker", "Name", "Weight"])


# ------------------------------------------------------------
# Analytics helpers
# ------------------------------------------------------------

def compute_return_from_nav(nav: pd.Series, window: int) -> float:
    """
    Compute total return over the last `window` observations of a NAV series.
    """
    if nav is None or len(nav) < 2:
        return float("nan")
    if len(nav) < window:
        window = len(nav)
    if window < 2:
        return float("nan")
    sub = nav.iloc[-window:]
    start = sub.iloc[0]
    end = sub.iloc[-1]
    if start <= 0:
        return float("nan")
    return (end / start) - 1.0


def annualized_vol(daily_ret: pd.Series) -> float:
    if daily_ret is None or len(daily_ret) < 2:
        return float("nan")
    return float(daily_ret.std() * np.sqrt(252))


def max_drawdown(nav: pd.Series) -> float:
    if nav is None or len(nav) < 2:
        return float("nan")
    running_max = nav.cummax()
    dd = (nav / running_max) - 1.0
    return float(dd.min())


def tracking_error(daily_wave: pd.Series, daily_bm: pd.Series) -> float:
    if daily_wave is None or daily_bm is None:
        return float("nan")
    if len(daily_wave) != len(daily_bm) or len(daily_wave) < 2:
        return float("nan")
    diff = (daily_wave - daily_bm).dropna()
    if len(diff) < 2:
        return float("nan")
    return float(diff.std() * np.sqrt(252))


def information_ratio(
    nav_wave: pd.Series,
    nav_bm: pd.Series,
    te: float,
) -> float:
    if nav_wave is None or nav_bm is None or len(nav_wave) < 2 or len(nav_bm) < 2:
        return float("nan")
    if te is None or te <= 0:
        return float("nan")
    ret_wave = compute_return_from_nav(nav_wave, window=len(nav_wave))
    ret_bm = compute_return_from_nav(nav_bm, window=len(nav_bm))
    excess = ret_wave - ret_bm
    return float(excess / te)


def simple_ret(series: pd.Series, window: int) -> float:
    if series is None or len(series) < 2:
        return float("nan")
    if len(series) < window:
        window = len(series)
    if window < 2:
        return float("nan")
    sub = series.iloc[-window:]
    return float(sub.iloc[-1] / sub.iloc[0] - 1.0)


def regress_factors(
    wave_ret: pd.Series,
    factor_ret: pd.DataFrame,
) -> dict:
    """
    Simple OLS: wave_ret ~ factors.
    Returns dict of {factor_name: beta}.
    """
    df = pd.concat(
        [wave_ret.rename("wave"), factor_ret],
        axis=1,
    ).dropna()
    if df.shape[0] < 60 or df.shape[1] < 2:
        return {col: float("nan") for col in factor_ret.columns}

    y = df["wave"].values
    X = df[factor_ret.columns].values
    # add intercept
    X_design = np.concatenate([np.ones((X.shape[0], 1)), X], axis=1)

    try:
        beta, *_ = np.linalg.lstsq(X_design, y, rcond=None)
    except Exception:
        return {col: float("nan") for col in factor_ret.columns}

    # beta[0] is intercept; ignore
    betas = beta[1:]
    return {col: float(b) for col, b in zip(factor_ret.columns, betas)}


def compute_wave_score(
    alpha_365: float,
    vol_wave: float,
    vol_bm: float,
    mdd_wave: float,
    mdd_bm: float,
    ir: float,
) -> dict:
    """
    Lightweight, engine-based WaveScore preview (0–100).
    This is a *preview* approximation of the full WAVESCORE™ spec.
    """
    # Return Quality (0–100): based on 365D alpha
    #  - 0% alpha ~ 50
    #  - +10% alpha ~ 70
    #  - +20% alpha ~ 90–100
    #  - -10% alpha ~ 30
    if math.isnan(alpha_365):
        rq = 50.0
    else:
        rq = 50.0 + (alpha_365 * 100.0 * 2.0)  # 2 pts per 1% alpha
        rq = float(np.clip(rq, 0.0, 100.0))

    # Risk Control: favor lower vol and shallower drawdowns vs benchmark
    if math.isnan(vol_wave) or math.isnan(vol_bm) or vol_bm <= 0:
        rc_vol_score = 50.0
    else:
        vol_ratio = vol_wave / vol_bm
        # vol_ratio = 1 → 50, 0.8 → ~60, 1.2 → ~40
        rc_vol_score = 50.0 + (1.0 - vol_ratio) * 25.0
        rc_vol_score = float(np.clip(rc_vol_score, 20.0, 80.0))

    if math.isnan(mdd_wave) or math.isnan(mdd_bm):
        rc_dd_score = 50.0
    else:
        # drawdown values are negative; closer to 0 is better
        gap = abs(mdd_bm) - abs(mdd_wave)
        # if Wave drawdown is 5 pts shallower → +10; 5 pts deeper → -10
        rc_dd_score = 50.0 + gap * 200.0
        rc_dd_score = float(np.clip(rc_dd_score, 20.0, 80.0))

    rc = 0.6 * rc_vol_score + 0.4 * rc_dd_score

    # Alpha Efficiency: based on IR
    #  - IR = 0 → 50
    #  - IR = 1 → ~70
    #  - IR = 2 → ~85
    #  - IR <= -0.5 → ~35
    if math.isnan(ir):
        ae = 50.0
    else:
        ae = 50.0 + ir * 15.0
        ae = float(np.clip(ae, 20.0, 95.0))

    # Combine into 0–100
    score = 0.4 * rq + 0.3 * rc + 0.3 * ae
    score = float(np.clip(score, 0.0, 100.0))

    return {
        "WaveScore": score,
        "ReturnQuality": rq,
        "RiskControl": rc,
        "AlphaEfficiency": ae,
    }


# ------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------

all_waves = we.get_all_waves()
all_modes = we.get_modes()

with st.sidebar:
    st.title("WAVES Intelligence™")
    st.caption("Mini Bloomberg Console • Vector OS™")

    mode = st.selectbox("Mode", all_modes, index=0)

    selected_wave = st.selectbox(
        "Select Wave",
        all_waves,
        index=0,
    )

    st.markdown("---")
    st.markdown("**Display settings**")
    nav_days = st.slider("History window (days)", min_value=60, max_value=730, value=365, step=15)


# ------------------------------------------------------------
# Page header + tabs
# ------------------------------------------------------------

st.title("WAVES Intelligence™ Institutional Console")
st.caption("Live Alpha Capture • SmartSafe™ • Multi-Asset • Crypto • Gold")

tab_console, tab_market, tab_factors, tab_vector = st.tabs(
    ["Console", "Market Intel", "Factor Decomposition", "Vector OS Insights"]
)


# ============================================================
# TAB 1: Console (main dashboard)
# ============================================================

with tab_console:
    # 1) Market Regime Monitor — SPY vs VIX
    st.subheader("Market Regime Monitor — SPY vs VIX")

    spy_vix = fetch_spy_vix(days=nav_days)

    if spy_vix.empty or "SPY" not in spy_vix.columns or "^VIX" not in spy_vix.columns:
        st.warning("Unable to load SPY/VIX data at the moment.")
    else:
        spy = spy_vix["SPY"].copy()
        vix = spy_vix["^VIX"].copy()

        if len(spy) > 0:
            spy_norm = spy / spy.iloc[0] * 100.0
        else:
            spy_norm = spy

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=spy_vix.index,
                y=spy_norm,
                name="SPY (Index = 100)",
                mode="lines",
            )
        )

        fig.add_trace(
            go.Scatter(
                x=spy_vix.index,
                y=vix,
                name="VIX Level",
                mode="lines",
                yaxis="y2",
            )
        )

        fig.update_layout(
            margin=dict(l=40, r=40, t=40, b=40),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            xaxis=dict(title="Date"),
            yaxis=dict(
                title="SPY (Indexed)",
                rangemode="tozero",
            ),
            yaxis2=dict(
                title="VIX",
                overlaying="y",
                side="right",
                rangemode="tozero",
            ),
            height=400,
        )

        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # 2) Portfolio-Level Overview (All Waves)
    st.subheader("Portfolio-Level Overview (All Waves)")

    overview_rows = []
    for wave in all_waves:
        hist_365 = compute_wave_history(wave, mode=mode, days=365)
        if hist_365.empty or len(hist_365) < 2:
            overview_rows.append(
                {
                    "Wave": wave,
                    "365D Return": np.nan,
                    "365D Alpha": np.nan,
                    "30D Return": np.nan,
                    "30D Alpha": np.nan,
                }
            )
            continue

        nav_wave = hist_365["wave_nav"]
        nav_bm = hist_365["bm_nav"]

        ret_365_wave = compute_return_from_nav(nav_wave, window=len(nav_wave))
        ret_365_bm = compute_return_from_nav(nav_bm, window=len(nav_bm))
        alpha_365 = ret_365_wave - ret_365_bm

        ret_30_wave = compute_return_from_nav(nav_wave, window=min(30, len(nav_wave)))
        ret_30_bm = compute_return_from_nav(nav_bm, window=min(30, len(nav_bm)))
        alpha_30 = ret_30_wave - ret_30_bm

        overview_rows.append(
            {
                "Wave": wave,
                "365D Return": ret_365_wave,
                "365D Alpha": alpha_365,
                "30D Return": ret_30_wave,
                "30D Alpha": alpha_30,
            }
        )

    if overview_rows:
        overview_df = pd.DataFrame(overview_rows)
        fmt_overview = overview_df.copy()
        for col in ["365D Return", "365D Alpha", "30D Return", "30D Alpha"]:
            fmt_overview[col] = fmt_overview[col].apply(
                lambda x: f"{x*100:0.2f}%" if pd.notna(x) else "—"
            )

        st.dataframe(
            fmt_overview.set_index("Wave"),
            use_container_width=True,
        )
    else:
        st.info("No overview data available yet.")

    st.markdown("---")

    # 3) Multi-Window Alpha Capture (All Waves)
    st.subheader(f"Multi-Window Alpha Capture (All Waves · Mode = {mode})")

    alpha_rows = []
    for wave in all_waves:
        hist_365 = compute_wave_history(wave, mode=mode, days=365)
        if hist_365.empty or len(hist_365) < 2:
            alpha_rows.append(
                {
                    "Wave": wave,
                    "1D Ret": np.nan,
                    "1D Alpha": np.nan,
                    "30D Ret": np.nan,
                    "30D Alpha": np.nan,
                    "60D Ret": np.nan,
                    "60D Alpha": np.nan,
                    "365D Ret": np.nan,
                    "365D Alpha": np.nan,
                }
            )
            continue

        nav_wave = hist_365["wave_nav"]
        nav_bm = hist_365["bm_nav"]

        if len(nav_wave) >= 2:
            ret_1d_wave = nav_wave.iloc[-1] / nav_wave.iloc[-2] - 1.0
            ret_1d_bm = nav_bm.iloc[-1] / nav_bm.iloc[-2] - 1.0
            alpha_1d = ret_1d_wave - ret_1d_bm
        else:
            ret_1d_wave = ret_1d_bm = alpha_1d = np.nan

        ret_30_wave = compute_return_from_nav(nav_wave, window=min(30, len(nav_wave)))
        ret_30_bm = compute_return_from_nav(nav_bm, window=min(30, len(nav_bm)))
        alpha_30 = ret_30_wave - ret_30_bm

        ret_60_wave = compute_return_from_nav(nav_wave, window=min(60, len(nav_wave)))
        ret_60_bm = compute_return_from_nav(nav_bm, window=min(60, len(nav_bm)))
        alpha_60 = ret_60_wave - ret_60_bm

        ret_365_wave = compute_return_from_nav(nav_wave, window=len(nav_wave))
        ret_365_bm = compute_return_from_nav(nav_bm, window=len(nav_bm))
        alpha_365 = ret_365_wave - ret_365_bm

        alpha_rows.append(
            {
                "Wave": wave,
                "1D Ret": ret_1d_wave,
                "1D Alpha": alpha_1d,
                "30D Ret": ret_30_wave,
                "30D Alpha": alpha_30,
                "60D Ret": ret_60_wave,
                "60D Alpha": alpha_60,
                "365D Ret": ret_365_wave,
                "365D Alpha": alpha_365,
            }
        )

    if alpha_rows:
        alpha_df = pd.DataFrame(alpha_rows)
        fmt_alpha = alpha_df.copy()
        for col in [
            "1D Ret",
            "1D Alpha",
            "30D Ret",
            "30D Alpha",
            "60D Ret",
            "60D Alpha",
            "365D Ret",
            "365D Alpha",
        ]:
            fmt_alpha[col] = fmt_alpha[col].apply(
                lambda x: f"{x*100:0.2f}%" if pd.notna(x) else "—"
            )

        st.dataframe(
            fmt_alpha.set_index("Wave"),
            use_container_width=True,
        )
    else:
        st.info("No alpha capture data available yet.")

    st.markdown("---")

    # 4) Risk & WaveScore Ingredients (All Waves)
    st.subheader("Risk & WaveScore Ingredients (All Waves · 365D Window)")

    risk_rows = []
    for wave in all_waves:
        hist_365 = compute_wave_history(wave, mode=mode, days=365)
        if hist_365.empty or len(hist_365) < 2:
            risk_rows.append(
                {
                    "Wave": wave,
                    "Wave Vol (365D)": np.nan,
                    "Benchmark Vol (365D)": np.nan,
                    "Max Drawdown (Wave)": np.nan,
                    "Max Drawdown (Benchmark)": np.nan,
                    "Tracking Error": np.nan,
                    "Information Ratio": np.nan,
                }
            )
            continue

        wave_ret = hist_365["wave_ret"]
        bm_ret = hist_365["bm_ret"]
        nav_wave = hist_365["wave_nav"]
        nav_bm = hist_365["bm_nav"]

        vol_wave = annualized_vol(wave_ret)
        vol_bm = annualized_vol(bm_ret)

        mdd_wave = max_drawdown(nav_wave)
        mdd_bm = max_drawdown(nav_bm)

        te = tracking_error(wave_ret, bm_ret)
        ir = information_ratio(nav_wave, nav_bm, te)

        risk_rows.append(
            {
                "Wave": wave,
                "Wave Vol (365D)": vol_wave,
                "Benchmark Vol (365D)": vol_bm,
                "Max Drawdown (Wave)": mdd_wave,
                "Max Drawdown (Benchmark)": mdd_bm,
                "Tracking Error": te,
                "Information Ratio": ir,
            }
        )

    risk_df = pd.DataFrame(risk_rows) if risk_rows else pd.DataFrame()

    if not risk_df.empty:
        fmt_risk = risk_df.copy()

        vol_cols = ["Wave Vol (365D)", "Benchmark Vol (365D)", "Tracking Error"]
        pct_cols = ["Max Drawdown (Wave)", "Max Drawdown (Benchmark)"]

        for col in vol_cols:
            fmt_risk[col] = fmt_risk[col].apply(
                lambda x: f"{x*100:0.2f}%" if pd.notna(x) else "—"
            )
        for col in pct_cols:
            fmt_risk[col] = fmt_risk[col].apply(
                lambda x: f"{x*100:0.2f}%" if pd.notna(x) else "—"
            )
        fmt_risk["Information Ratio"] = fmt_risk["Information Ratio"].apply(
            lambda x: f"{x:0.2f}" if pd.notna(x) else "—"
        )

        st.dataframe(
            fmt_risk.set_index("Wave"),
            use_container_width=True,
        )
    else:
        st.info("No risk analytics available yet.")

    # 4b) WaveScore Leaderboard (Preview)
    st.markdown("#### WaveScore Leaderboard (Preview)")

    ws_rows = []
    for wave in all_waves:
        hist_365 = compute_wave_history(wave, mode=mode, days=365)
        if hist_365.empty or len(hist_365) < 2:
            ws_rows.append(
                {
                    "Wave": wave,
                    "WaveScore": np.nan,
                    "ReturnQuality": np.nan,
                    "RiskControl": np.nan,
                    "AlphaEfficiency": np.nan,
                    "365D Alpha": np.nan,
                }
            )
            continue

        nav_wave = hist_365["wave_nav"]
        nav_bm = hist_365["bm_nav"]
        wave_ret = hist_365["wave_ret"]
        bm_ret = hist_365["bm_ret"]

        alpha_365 = compute_return_from_nav(nav_wave, window=len(nav_wave)) - compute_return_from_nav(
            nav_bm, window=len(nav_bm)
        )
        vol_wave = annualized_vol(wave_ret)
        vol_bm = annualized_vol(bm_ret)
        mdd_wave = max_drawdown(nav_wave)
        mdd_bm = max_drawdown(nav_bm)
        te = tracking_error(wave_ret, bm_ret)
        ir = information_ratio(nav_wave, nav_bm, te)

        scores = compute_wave_score(alpha_365, vol_wave, vol_bm, mdd_wave, mdd_bm, ir)

        ws_rows.append(
            {
                "Wave": wave,
                "WaveScore": scores["WaveScore"],
                "ReturnQuality": scores["ReturnQuality"],
                "RiskControl": scores["RiskControl"],
                "AlphaEfficiency": scores["AlphaEfficiency"],
                "365D Alpha": alpha_365,
            }
        )

    if ws_rows:
        ws_df = pd.DataFrame(ws_rows)
        ws_df = ws_df.sort_values("WaveScore", ascending=False)

        fmt_ws = ws_df.copy()
        fmt_ws["WaveScore"] = fmt_ws["WaveScore"].apply(lambda x: f"{x:0.1f}" if pd.notna(x) else "—")
        fmt_ws["ReturnQuality"] = fmt_ws["ReturnQuality"].apply(lambda x: f"{x:0.1f}" if pd.notna(x) else "—")
        fmt_ws["RiskControl"] = fmt_ws["RiskControl"].apply(lambda x: f"{x:0.1f}" if pd.notna(x) else "—")
        fmt_ws["AlphaEfficiency"] = fmt_ws["AlphaEfficiency"].apply(lambda x: f"{x:0.1f}" if pd.notna(x) else "—")
        fmt_ws["365D Alpha"] = fmt_ws["365D Alpha"].apply(
            lambda x: f"{x*100:0.2f}%" if pd.notna(x) else "—"
        )

        st.dataframe(
            fmt_ws.set_index("Wave"),
            use_container_width=True,
        )
        st.caption(
            "WaveScore here is a *preview approximation* built from 365D alpha, risk profile, and Information Ratio, "
            "mapped into a 0–100 scale consistent with the WAVESCORE™ v1.0 philosophy."
        )
    else:
        st.info("No WaveScore data available yet.")

    st.markdown("---")

    # 5) Benchmark ETF Mix Table
    st.subheader("Benchmark ETF Mix (Composite Benchmarks)")

    bm_mix = get_benchmark_mix()
    if bm_mix.empty:
        st.info("No benchmark mix data available.")
    else:
        fmt_bm = bm_mix.copy()
        fmt_bm["Weight"] = fmt_bm["Weight"].apply(
            lambda x: f"{x*100:0.2f}%" if pd.notna(x) else "—"
        )
        st.dataframe(
            fmt_bm,
            use_container_width=True,
        )

    st.markdown("---")

    # 6) Wave Detail View
    st.subheader(f"Wave Detail — {selected_wave} (Mode: {mode})")

    col_chart, col_stats = st.columns([2.0, 1.0])

    with col_chart:
        hist = compute_wave_history(selected_wave, mode=mode, days=nav_days)
        if hist.empty or len(hist) < 2:
            st.warning("Not enough data to display NAV chart.")
        else:
            nav_wave = hist["wave_nav"]
            nav_bm = hist["bm_nav"]

            fig_nav = go.Figure()
            fig_nav.add_trace(
                go.Scatter(
                    x=hist.index,
                    y=nav_wave,
                    name=f"{selected_wave} NAV",
                    mode="lines",
                )
            )
            fig_nav.add_trace(
                go.Scatter(
                    x=hist.index,
                    y=nav_bm,
                    name="Benchmark NAV",
                    mode="lines",
                )
            )
            fig_nav.update_layout(
                margin=dict(l=40, r=40, t=40, b=40),
                xaxis=dict(title="Date"),
                yaxis=dict(title="NAV (Normalized)"),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1,
                ),
                height=380,
            )
            st.plotly_chart(fig_nav, use_container_width=True)

    with col_stats:
        if hist.empty or len(hist) < 2:
            st.info("No stats available.")
        else:
            nav_wave = hist["wave_nav"]
            nav_bm = hist["bm_nav"]

            ret_30_wave = compute_return_from_nav(nav_wave, window=min(30, len(nav_wave)))
            ret_30_bm = compute_return_from_nav(nav_bm, window=min(30, len(nav_bm)))
            alpha_30 = ret_30_wave - ret_30_bm

            ret_365_wave = compute_return_from_nav(nav_wave, window=len(nav_wave))
            ret_365_bm = compute_return_from_nav(nav_bm, window=len(nav_bm))
            alpha_365 = ret_365_wave - ret_365_bm

            st.markdown("**Performance vs Benchmark**")
            st.metric(
                "30D Return",
                f"{ret_30_wave*100:0.2f}%" if not math.isnan(ret_30_wave) else "—",
            )
            st.metric(
                "30D Alpha",
                f"{alpha_30*100:0.2f}%" if not math.isnan(alpha_30) else "—",
            )
            st.metric(
                "365D Return",
                f"{ret_365_wave*100:0.2f}%" if not math.isnan(ret_365_wave) else "—",
            )
            st.metric(
                "365D Alpha",
                f"{alpha_365*100:0.2f}%" if not math.isnan(alpha_365) else "—",
            )

    st.markdown("#### Mode Comparison (365D)")

    mode_rows = []
    for m in all_modes:
        hist_m = compute_wave_history(selected_wave, mode=m, days=365)
        if hist_m.empty or len(hist_m) < 2:
            mode_rows.append({"Mode": m, "365D Return": np.nan})
            continue
        nav = hist_m["wave_nav"]
        r = compute_return_from_nav(nav, window=len(nav))
        mode_rows.append({"Mode": m, "365D Return": r})

    if mode_rows:
        mode_df = pd.DataFrame(mode_rows)
        fmt_mode = mode_df.copy()
        fmt_mode["365D Return"] = fmt_mode["365D Return"].apply(
            lambda x: f"{x*100:0.2f}%" if pd.notna(x) else "—"
        )
        st.dataframe(fmt_mode.set_index("Mode"), use_container_width=True)
    else:
        st.info("No mode comparison data available.")

    st.markdown("#### Top-10 Holdings")

    holdings_df = get_wave_holdings(selected_wave)
    if holdings_df.empty:
        st.info("No holdings available for this Wave.")
    else:
        def google_link(ticker: str) -> str:
            base = "https://www.google.com/finance/quote"
            return f"[{ticker}]({base}/{ticker})"

        fmt_hold = holdings_df.copy()
        fmt_hold["Weight"] = fmt_hold["Weight"].apply(
            lambda x: f"{x*100:0.2f}%" if pd.notna(x) else "—"
        )
        fmt_hold["Google Finance"] = fmt_hold["Ticker"].apply(google_link)

        st.dataframe(
            fmt_hold[["Ticker", "Name", "Weight", "Google Finance"]],
            use_container_width=True,
        )


# ============================================================
# TAB 2: Market Intel — global markets & events
# ============================================================

with tab_market:
    st.subheader("Global Market Dashboard")

    market_df = fetch_market_assets(days=nav_days)

    if market_df.empty:
        st.warning("Unable to load multi-asset market data right now.")
    else:
        assets = {
            "SPY": "S&P 500",
            "QQQ": "NASDAQ-100",
            "IWM": "US Small Caps",
            "TLT": "US 20+Y Treasuries",
            "GLD": "Gold",
            "BTC-USD": "Bitcoin (USD)",
            "^VIX": "VIX (Implied Vol)",
            "^TNX": "US 10Y Yield",
        }

        rows = []
        for tkr, label in assets.items():
            if tkr not in market_df.columns:
                continue
            series = market_df[tkr]
            last = float(series.iloc[-1]) if len(series) else float("nan")
            r1d = simple_ret(series, 2)
            r30 = simple_ret(series, 30)
            rows.append(
                {
                    "Ticker": tkr,
                    "Asset": label,
                    "Last": last,
                    "1D Return": r1d,
                    "30D Return": r30,
                }
            )

        snap_df = pd.DataFrame(rows)
        fmt_snap = snap_df.copy()

        def fmt_ret(x: float) -> str:
            return f"{x*100:0.2f}%" if pd.notna(x) else "—"

        fmt_snap["Last"] = fmt_snap["Last"].apply(
            lambda x: f"{x:0.2f}" if pd.notna(x) else "—"
        )
        fmt_snap["1D Return"] = fmt_snap["1D Return"].apply(fmt_ret)
        fmt_snap["30D Return"] = fmt_snap["30D Return"].apply(fmt_ret)

        st.dataframe(
            fmt_snap.set_index("Ticker"),
            use_container_width=True,
        )

        st.caption(
            "Snapshot of key risk assets and hedges — equities, Treasuries, gold, Bitcoin, VIX, and the 10Y yield."
        )

    st.markdown("---")

    # Market Regime Monitor inside Market Intel tab
    st.subheader("Market Regime Monitor — SPY vs VIX")

    spy_vix_m = fetch_spy_vix(days=nav_days)

    col_left, col_right = st.columns([2.0, 1.0])

    with col_left:
        if spy_vix_m.empty or "SPY" not in spy_vix_m.columns or "^VIX" not in spy_vix_m.columns:
            st.warning("Unable to load SPY/VIX data at the moment.")
        else:
            spy = spy_vix_m["SPY"].copy()
            vix = spy_vix_m["^VIX"].copy()

            if len(spy) > 0:
                spy_norm = spy / spy.iloc[0] * 100.0
            else:
                spy_norm = spy

            fig_m = go.Figure()

            fig_m.add_trace(
                go.Scatter(
                    x=spy_vix_m.index,
                    y=spy_norm,
                    name="SPY (Index = 100)",
                    mode="lines",
                )
            )
            fig_m.add_trace(
                go.Scatter(
                    x=spy_vix_m.index,
                    y=vix,
                    name="VIX Level",
                    mode="lines",
                    yaxis="y2",
                )
            )

            fig_m.update_layout(
                margin=dict(l=40, r=40, t=40, b=40),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1,
                ),
                xaxis=dict(title="Date"),
                yaxis=dict(title="SPY (Indexed)", rangemode="tozero"),
                yaxis2=dict(
                    title="VIX",
                    overlaying="y",
                    side="right",
                    rangemode="tozero",
                ),
                height=380,
            )

            st.plotly_chart(fig_m, use_container_width=True)

    with col_right:
        if spy_vix_m.empty or "SPY" not in spy_vix_m.columns:
            st.info("No market summary available.")
        else:
            spy = spy_vix_m["SPY"].copy()
            vix = spy_vix_m.get("^VIX", pd.Series(index=spy.index, data=np.nan))

            current_spy = float(spy.iloc[-1]) if len(spy) else float("nan")
            current_vix = float(vix.iloc[-1]) if len(vix) else float("nan")

            r30 = simple_ret(spy, 30)
            r60 = simple_ret(spy, 60)

            if len(spy) > 30:
                daily_ret_spy = spy.pct_change().dropna()
                vol30 = float(daily_ret_spy.iloc[-30:].std() * np.sqrt(252))
            else:
                vol30 = float("nan")

            try:
                regime = we._regime_from_return(r60)  # type: ignore[attr-defined]
            except Exception:
                regime = "neutral"

            st.markdown("**Current Market Snapshot**")
            st.metric("SPY (last)", f"{current_spy:0.2f}" if not math.isnan(current_spy) else "—")
            st.metric("SPY 30D Return", f"{r30*100:0.2f}%" if not math.isnan(r30) else "—")
            st.metric("SPY 60D Return", f"{r60*100:0.2f}%" if not math.isnan(r60) else "—")
            st.metric("30D Realized Vol (SPY)", f"{vol30*100:0.2f}%" if not math.isnan(vol30) else "—")
            st.metric("VIX (last)", f"{current_vix:0.2f}" if not math.isnan(current_vix) else "—")
            st.metric("Engine Regime", regime.capitalize())

            st.caption(
                "Regime is computed from SPY 60D trend via WAVES engine. "
                "VIX & realized vol indicate how aggressively or defensively SmartSafe and VIX scaling behave."
            )

    st.markdown("---")

    # Macro & Events Panel
    st.subheader("Macro & Events — Earnings & Key Themes")

    col_ev_left, col_ev_right = st.columns([1.3, 1.0])

    with col_ev_left:
        st.markdown("**Selected Large-Cap Earnings (via yfinance)**")
        if yf is None:
            st.info("Earnings calendar requires yfinance and internet connectivity.")
        else:
            selected_names = [
                ("AAPL", "Apple"),
                ("MSFT", "Microsoft"),
                ("AMZN", "Amazon"),
                ("GOOGL", "Alphabet"),
                ("META", "Meta Platforms"),
                ("NVDA", "NVIDIA"),
                ("JPM", "JPMorgan"),
            ]
            events = []
            today = datetime.utcnow().date()
            for ticker, name in selected_names:
                try:
                    t_obj = yf.Ticker(ticker)
                    cal = getattr(t_obj, "calendar", None)
                    next_earn = None
                    if isinstance(cal, pd.DataFrame) and not cal.empty:
                        if "Earnings Date" in cal.index:
                            val = cal.loc["Earnings Date"].iloc[0]
                            try:
                                next_earn = pd.to_datetime(val).date()
                            except Exception:
                                next_earn = None
                        else:
                            val = cal.iloc[0, 0]
                            try:
                                next_earn = pd.to_datetime(val).date()
                            except Exception:
                                next_earn = None
                    if next_earn and next_earn >= today:
                        events.append(
                            {
                                "Ticker": ticker,
                                "Name": name,
                                "Next Earnings Date": next_earn.isoformat(),
                            }
                        )
                except Exception:
                    continue

            if events:
                events_df = pd.DataFrame(events).sort_values("Next Earnings Date")
                st.dataframe(events_df.set_index("Ticker"), use_container_width=True)
            else:
                st.info(
                    "No upcoming earnings could be retrieved right now for the selected names. "
                    "This panel is a lightweight earnings proxy, not a full economic calendar."
                )

    with col_ev_right:
        st.markdown("**Key Macro Themes (Editable in Code)**")

        macro_rows = [
            {
                "Event / Theme": "FOMC rate decisions",
                "Focus": "Policy rates, liquidity, risk appetite",
                "Impact Bias": "Higher vol → larger SmartSafe allocation",
            },
            {
                "Event / Theme": "CPI / inflation releases",
                "Focus": "Real yields, growth vs value, long-duration assets",
                "Impact Bias": "High inflation → pressure on high-duration Waves",
            },
            {
                "Event / Theme": "Tech mega-cap earnings season",
                "Focus": "AI & Cloud, Next-Gen Compute, Small-Cap Disruptors",
                "Impact Bias": "Upside surprises → higher beta & tech Waves outperform",
            },
            {
                "Event / Theme": "Crypto market stress/liquidity events",
                "Focus": "Bitcoin Wave, Multi-Cap Crypto Growth, Crypto Income ladder",
                "Impact Bias": "Elevated crypto-vol → more in Crypto Stable Yield & SmartSafe",
            },
        ]
        macro_df = pd.DataFrame(macro_rows)
        st.dataframe(macro_df, use_container_width=True)
        st.caption(
            "Macro themes describe how the engine *tends* to react; you can edit this table in code "
            "to reflect current dates or firm-specific views."
        )

    st.markdown("---")

    # WAVES Reaction Snapshot (30D)
    st.subheader("WAVES Reaction Snapshot (30D · Mode = {})".format(mode))

    reaction_rows = []
    for wave in all_waves:
        hist_365 = compute_wave_history(wave, mode=mode, days=365)
        if hist_365.empty or len(hist_365) < 2:
            reaction_rows.append(
                {
                    "Wave": wave,
                    "30D Return": np.nan,
                    "30D Alpha": np.nan,
                    "Classification": "No data",
                }
            )
            continue

        nav_wave = hist_365["wave_nav"]
        nav_bm = hist_365["bm_nav"]

        r30_wave = compute_return_from_nav(nav_wave, window=min(30, len(nav_wave)))
        r30_bm = compute_return_from_nav(nav_bm, window=min(30, len(nav_bm)))
        a30 = r30_wave - r30_bm

        if math.isnan(a30):
            label = "No data"
        elif a30 >= 0.05:
            label = "Strong Outperformance"
        elif a30 >= 0.02:
            label = "Outperforming"
        elif a30 <= -0.03:
            label = "Lagging"
        else:
            label = "Near Benchmark"

        reaction_rows.append(
            {
                "Wave": wave,
                "30D Return": r30_wave,
                "30D Alpha": a30,
                "Classification": label,
            }
        )

    if reaction_rows:
        reaction_df = pd.DataFrame(reaction_rows)

        fmt_reaction = reaction_df.copy()
        fmt_reaction["30D Return"] = fmt_reaction["30D Return"].apply(
            lambda x: f"{x*100:0.2f}%" if pd.notna(x) else "—"
        )
        fmt_reaction["30D Alpha"] = fmt_reaction["30D Alpha"].apply(
            lambda x: f"{x*100:0.2f}%" if pd.notna(x) else "—"
        )

        st.dataframe(
            fmt_reaction.set_index("Wave"),
            use_container_width=True,
        )

        valid = reaction_df.dropna(subset=["30D Alpha"])
        if not valid.empty:
            avg_alpha = float(valid["30D Alpha"].mean())
            best_row = valid.loc[valid["30D Alpha"].idxmax()]
            worst_row = valid.loc[valid["30D Alpha"].idxmin()]

            st.markdown("#### Engine Interpretation")
            st.write(
                f"- **Average 30D alpha across Waves**: {avg_alpha*100:0.2f}% "
                "(positive means WAVES are beating their composite benchmarks on average)."
            )
            st.write(
                f"- **Best Wave (30D alpha)**: {best_row['Wave']} at {best_row['30D Alpha']*100:0.2f}%."
            )
            st.write(
                f"- **Most challenged Wave (30D alpha)**: {worst_row['Wave']} at {worst_row['30D Alpha']*100:0.2f}%."
            )
            st.caption(
                "High positive alpha suggests momentum tilts, VIX/vol scaling, SmartSafe sweeps, "
                "and mode exposure are adding value vs static ETF mixes."
            )
        else:
            st.info("Not enough data yet to summarize Wave reactions.")
    else:
        st.info("No Wave reaction data available.")


# ============================================================
# TAB 3: Factor Decomposition — institutional analytics
# ============================================================

with tab_factors:
    st.subheader("Factor Decomposition (Institution-Level Analytics)")

    st.caption(
        "Wave daily returns are regressed on key risk premia: SPY, QQQ, IWM, TLT, GLD, BTC-USD. "
        "Betas approximate sensitivity to market, growth/tech, small caps, rates, gold, and crypto."
    )

    # Use same market data as Market Intel
    factor_days = min(nav_days, 365)  # keep it around ~1Y
    factor_prices = fetch_market_assets(days=factor_days)

    needed = ["SPY", "QQQ", "IWM", "TLT", "GLD", "BTC-USD"]
    missing = [t for t in needed if t not in factor_prices.columns]

    if factor_prices.empty or missing:
        st.warning(
            "Unable to load all factor price series. "
            f"Missing: {', '.join(missing)}" if missing else ""
        )
    else:
        factor_returns = factor_prices[needed].pct_change().dropna()
        factor_returns = factor_returns.rename(
            columns={
                "SPY": "MKT_SPY",
                "QQQ": "GROWTH_QQQ",
                "IWM": "SIZE_IWM",
                "TLT": "RATES_TLT",
                "GLD": "GOLD_GLD",
                "BTC-USD": "CRYPTO_BTC",
            }
        )

        rows = []
        for wave in all_waves:
            hist = compute_wave_history(wave, mode=mode, days=factor_days)
            if hist.empty or "wave_ret" not in hist.columns:
                rows.append(
                    {
                        "Wave": wave,
                        "β_SPY": np.nan,
                        "β_QQQ": np.nan,
                        "β_IWM": np.nan,
                        "β_TLT": np.nan,
                        "β_GLD": np.nan,
                        "β_BTC": np.nan,
                    }
                )
                continue

            wret = hist["wave_ret"]
            betas = regress_factors(
                wave_ret=wret,
                factor_ret=factor_returns,
            )

            rows.append(
                {
                    "Wave": wave,
                    "β_SPY": betas.get("MKT_SPY", np.nan),
                    "β_QQQ": betas.get("GROWTH_QQQ", np.nan),
                    "β_IWM": betas.get("SIZE_IWM", np.nan),
                    "β_TLT": betas.get("RATES_TLT", np.nan),
                    "β_GLD": betas.get("GOLD_GLD", np.nan),
                    "β_BTC": betas.get("CRYPTO_BTC", np.nan),
                }
            )

        if rows:
            beta_df = pd.DataFrame(rows)
            st.markdown("### Factor Betas — All Waves")

            fmt_beta = beta_df.copy()
            for col in ["β_SPY", "β_QQQ", "β_IWM", "β_TLT", "β_GLD", "β_BTC"]:
                fmt_beta[col] = fmt_beta[col].apply(
                    lambda x: f"{x:0.2f}" if pd.notna(x) else "—"
                )

            st.dataframe(
                fmt_beta.set_index("Wave"),
                use_container_width=True,
            )

            st.markdown("### Factor Profile — Selected Wave")

            sel_row = beta_df[beta_df["Wave"] == selected_wave]
            if sel_row.empty:
                st.info("No factor data for the selected Wave.")
            else:
                r = sel_row.iloc[0]
                factors = ["β_SPY", "β_QQQ", "β_IWM", "β_TLT", "β_GLD", "β_BTC"]
                values = [float(r[c]) if pd.notna(r[c]) else 0.0 for c in factors]

                fig_beta = go.Figure(
                    data=[
                        go.Bar(
                            x=["SPY", "QQQ", "IWM", "TLT", "GLD", "BTC"],
                            y=values,
                        )
                    ]
                )
                fig_beta.update_layout(
                    title=f"Factor Betas for {selected_wave} (Mode: {mode}, ~{factor_days}D)",
                    xaxis_title="Factor",
                    yaxis_title="Beta",
                    height=400,
                    margin=dict(l=40, r=40, t=40, b=40),
                )

                st.plotly_chart(fig_beta, use_container_width=True)

                st.caption(
                    "Interpretation example: β_SPY ≈ 1.1 implies slightly higher equity beta than SPY; "
                    "β_BTC > 0 suggests positive crypto sensitivity; β_TLT < 0 implies the Wave tends to "
                    "struggle when long-duration Treasuries rally (rates falling)."
                )
        else:
            st.info("No factor data available yet.")

    # Correlation Matrix (Wave daily returns)
    st.markdown("---")
    st.subheader("Wave Correlation Matrix (Wave Daily Returns · Mode = {})".format(mode))

    cor_days = min(nav_days, 365)
    ret_mat = pd.DataFrame()

    for wave in all_waves:
        hist = compute_wave_history(wave, mode=mode, days=cor_days)
        if hist.empty or "wave_ret" not in hist.columns:
            continue
        s = hist["wave_ret"].rename(wave)
        ret_mat = pd.concat([ret_mat, s], axis=1)

    if ret_mat.empty or ret_mat.shape[1] < 2:
        st.info("Not enough overlapping data to compute a correlation matrix.")
    else:
        corr = ret_mat.corr()

        st.dataframe(
            corr.round(2),
            use_container_width=True,
        )

        fig_corr = go.Figure(
            data=go.Heatmap(
                z=corr.values,
                x=corr.columns,
                y=corr.index,
                zmin=-1,
                zmax=1,
                colorbar=dict(title="Corr"),
            )
        )
        fig_corr.update_layout(
            title=f"Correlation Heatmap (Mode: {mode}, ~{cor_days}D)",
            xaxis_title="Wave",
            yaxis_title="Wave",
            height=500,
            margin=dict(l=60, r=60, t=60, b=60),
        )

        st.plotly_chart(fig_corr, use_container_width=True)
        st.caption(
            "High positive correlations suggest overlapping risk factors; lower correlations identify "
            "diversifying Waves inside the overall stack."
        )


# ============================================================
# TAB 4: Vector OS Insights — AI-style narrative layer
# ============================================================

with tab_vector:
    st.subheader("Vector OS™ Insight Layer (Local AI Chat Preview)")

    st.caption(
        "This layer turns the raw engine stats into narrative. It uses local logic only "
        "(no external APIs), but behaves like an AI assistant sitting on top of the WAVES engine."
    )

    col_sel, col_chat = st.columns([1.2, 1.8])

    with col_sel:
        wave_for_insight = st.selectbox(
            "Wave for Insights",
            all_waves,
            index=all_waves.index(selected_wave) if selected_wave in all_waves else 0,
        )
        horizon_label = st.selectbox(
            "Horizon",
            ["30D", "60D", "365D"],
            index=0,
        )
        st.markdown("**Mode in focus:** {}".format(mode))

    with col_chat:
        user_prompt = st.text_input(
            "Question for Vector (optional, cosmetic)",
            value="How is this Wave really behaving under the hood?",
        )

    # Compute stats for the chosen Wave + horizon
    horizon_days = {"30D": 30, "60D": 60, "365D": 365}[horizon_label]
    hist_insight = compute_wave_history(wave_for_insight, mode=mode, days=365)

    if hist_insight.empty or len(hist_insight) < 2:
        st.info("Not enough history yet for Vector to generate insights on this Wave.")
    else:
        nav_wave = hist_insight["wave_nav"]
        nav_bm = hist_insight["bm_nav"]
        wave_ret = hist_insight["wave_ret"]
        bm_ret = hist_insight["bm_ret"]

        window = min(horizon_days, len(nav_wave))

        ret_wave_h = compute_return_from_nav(nav_wave, window=window)
        ret_bm_h = compute_return_from_nav(nav_bm, window=window)
        alpha_h = ret_wave_h - ret_bm_h

        vol_wave_365 = annualized_vol(wave_ret)
        vol_bm_365 = annualized_vol(bm_ret)
        mdd_wave_365 = max_drawdown(nav_wave)
        mdd_bm_365 = max_drawdown(nav_bm)
        te = tracking_error(wave_ret, bm_ret)
        ir = information_ratio(nav_wave, nav_bm, te)

        scores = compute_wave_score(
            alpha_365=compute_return_from_nav(nav_wave, window=len(nav_wave))
            - compute_return_from_nav(nav_bm, window=len(nav_bm)),
            vol_wave=vol_wave_365,
            vol_bm=vol_bm_365,
            mdd_wave=mdd_wave_365,
            mdd_bm=mdd_bm_365,
            ir=ir,
        )

        st.markdown("### Vector’s Take")

        # Headline
        alpha_str = f"{alpha_h*100:0.2f}%" if not math.isnan(alpha_h) else "—"
        ret_str = f"{ret_wave_h*100:0.2f}%" if not math.isnan(ret_wave_h) else "—"
        bm_str = f"{ret_bm_h*100:0.2f}%" if not math.isnan(ret_bm_h) else "—"

        if not math.isnan(alpha_h):
            if alpha_h > 0.05:
                headline = f"{wave_for_insight} is **crushing** its benchmark over the last {horizon_label}."
            elif alpha_h > 0.02:
                headline = f"{wave_for_insight} is **comfortably ahead** of its benchmark over the last {horizon_label}."
            elif alpha_h > -0.02:
                headline = f"{wave_for_insight} is running **close to benchmark** over the last {horizon_label}."
            else:
                headline = f"{wave_for_insight} has been **lagging** its benchmark over the last {horizon_label}."
        else:
            headline = f"{wave_for_insight} doesn’t have enough data yet for a strong view."

        st.markdown(f"**{headline}**")

        # Core bullet points
        st.markdown("#### Performance Snapshot")
        st.write(
            f"- **{horizon_label} return:** {ret_str} vs benchmark {bm_str} "
            f"→ **alpha: {alpha_str}**"
        )
        st.write(
            f"- **365D annualized vol:** "
            f"{vol_wave_365*100:0.2f}% (Wave) vs {vol_bm_365*100:0.2f}% (Benchmark)"
            if not (math.isnan(vol_wave_365) or math.isnan(vol_bm_365))
            else "- Volatility: not enough data yet."
        )
        st.write(
            f"- **Max drawdown (365D):** "
            f"{mdd_wave_365*100:0.2f}% (Wave) vs {mdd_bm_365*100:0.2f}% (Benchmark)"
            if not (math.isnan(mdd_wave_365) or math.isnan(mdd_bm_365))
            else "- Max drawdown: not enough data yet."
        )
        st.write(
            f"- **Information Ratio (365D):** {ir:0.2f}"
            if not math.isnan(ir)
            else "- Information Ratio: not enough data yet."
        )
        st.write(
            f"- **WaveScore (Preview):** {scores['WaveScore']:0.1f} / 100 "
            f"(ReturnQuality {scores['ReturnQuality']:0.1f}, "
            f"RiskControl {scores['RiskControl']:0.1f}, "
            f"AlphaEfficiency {scores['AlphaEfficiency']:0.1f})"
        )

        # Style / behavior narrative
        behaviour_lines = []

        # Vol vs benchmark
        if not (math.isnan(vol_wave_365) or math.isnan(vol_bm_365)):
            vol_ratio = vol_wave_365 / vol_bm_365 if vol_bm_365 > 0 else np.nan
            if not math.isnan(vol_ratio):
                if vol_ratio < 0.9:
                    behaviour_lines.append(
                        "Volatility is **lower than benchmark**, which is exactly what we want "
                        "when we’re trying to weaponize volatility without scaring the client."
                    )
                elif vol_ratio < 1.1:
                    behaviour_lines.append(
                        "Volatility is **in line with benchmark**, so most of the story here is pure alpha, "
                        "not just taking more risk."
                    )
                else:
                    behaviour_lines.append(
                        "Volatility is **above benchmark**, so some of the performance is coming from taking "
                        "a more aggressive stance. That’s fine as long as the payoff per unit of risk (IR) stays strong."
                    )

        # IR
        if not math.isnan(ir):
            if ir > 1.0:
                behaviour_lines.append(
                    "The Information Ratio is **strong**, which says the engine is extracting alpha efficiently "
                    "rather than just gambling on a few big moves."
                )
            elif ir > 0.3:
                behaviour_lines.append(
                    "The Information Ratio is **positive**, suggesting steady value-add, but there’s still room "
                    "to tighten risk and improve the payoff per unit of risk."
                )
            elif ir < 0:
                behaviour_lines.append(
                    "The Information Ratio is **negative**, which is a red flag for this window. "
                    "It means the benchmark is doing a better job per unit of risk taken."
                )

        # Drawdown vs benchmark
        if not (math.isnan(mdd_wave_365) or math.isnan(mdd_bm_365)):
            if abs(mdd_wave_365) < abs(mdd_bm_365):
                behaviour_lines.append(
                    "Drawdowns have been **shallower than the benchmark**, which is exactly the SmartSafe + VIX-scaling story."
                )
            elif abs(mdd_wave_365) > abs(mdd_bm_365) + 0.05:
                behaviour_lines.append(
                    "Drawdowns have been **deeper than the benchmark**, which we should watch, especially in risk-off regimes."
                )

        st.markdown("#### Style & Risk Narrative")
        if behaviour_lines:
            for line in behaviour_lines:
                st.write(f"- {line}")
        else:
            st.write("- Not enough stable data yet to characterize style and risk with confidence.")

        st.caption(
            "This insight layer is fully rule-based and computed locally using the engine’s numbers. "
            "In production, this can be upgraded to a live Vector OS™ LLM agent wired into WAVES Intelligence™."
        )