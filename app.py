# app.py — WAVES Intelligence™ Institutional Console (Vector OS Edition)
from __future__ import annotations

import math
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ------------------------------------------------------------
# Import engine safely + verify correct engine loaded
# ------------------------------------------------------------
ENGINE_OK = True
ENGINE_ERROR = ""

try:
    import waves_engine as we  # must be exactly waves_engine.py at repo root
except Exception as e:
    ENGINE_OK = False
    ENGINE_ERROR = f"Failed to import waves_engine.py: {e}"
    we = None  # type: ignore

st.set_page_config(page_title="WAVES Intelligence™ Console", layout="wide", initial_sidebar_state="expanded")

if not ENGINE_OK or we is None:
    st.error("ENGINE LOAD FAILURE")
    st.write(ENGINE_ERROR)
    st.stop()

# Verify engine has required API (this prevents your AttributeError nightmare)
required = ["get_all_waves", "get_modes", "compute_history_nav", "get_benchmark_mix_table", "get_wave_holdings"]
missing = [name for name in required if not hasattr(we, name)]
sig = getattr(we, "ENGINE_SIGNATURE", "UNKNOWN_ENGINE")
ver = getattr(we, "ENGINE_VERSION", "UNKNOWN_VERSION")

if missing:
    st.error("WRONG / PARTIAL ENGINE DETECTED")
    st.write(f"Engine signature: `{sig}`")
    st.write(f"Engine version: `{ver}`")
    st.write("Missing required functions:")
    st.write(missing)
    st.info(
        "This means Streamlit is importing an older or partial waves_engine.py, "
        "or the file crashed during import so functions were never defined."
    )
    st.stop()

try:
    import yfinance as yf
except Exception:
    yf = None


# ------------------------------------------------------------
# Helpers (cached)
# ------------------------------------------------------------
@st.cache_data(show_spinner=False)
def fetch_spy_vix(days: int = 365) -> pd.DataFrame:
    if yf is None:
        return pd.DataFrame()
    end = datetime.utcnow().date()
    start = end - timedelta(days=days + 10)
    data = yf.download(["SPY", "^VIX"], start=start.isoformat(), end=end.isoformat(),
                       interval="1d", auto_adjust=True, progress=False, group_by="column")
    if isinstance(data.columns, pd.MultiIndex):
        data = data["Adj Close"] if "Adj Close" in data.columns.get_level_values(0) else data["Close"]
    if isinstance(data.columns, pd.MultiIndex):
        data = data.droplevel(0, axis=1)
    if isinstance(data, pd.Series):
        data = data.to_frame()
    data = data.sort_index().ffill().bfill()
    return data.iloc[-days:] if len(data) > days else data


@st.cache_data(show_spinner=False)
def fetch_market_assets(days: int = 365) -> pd.DataFrame:
    if yf is None:
        return pd.DataFrame()
    end = datetime.utcnow().date()
    start = end - timedelta(days=days + 10)
    tickers = ["SPY", "QQQ", "IWM", "TLT", "GLD", "BTC-USD", "^VIX", "^TNX"]
    data = yf.download(tickers, start=start.isoformat(), end=end.isoformat(),
                       interval="1d", auto_adjust=True, progress=False, group_by="column")
    if isinstance(data.columns, pd.MultiIndex):
        data = data["Adj Close"] if "Adj Close" in data.columns.get_level_values(0) else data["Close"]
    if isinstance(data.columns, pd.MultiIndex):
        data = data.droplevel(0, axis=1)
    if isinstance(data, pd.Series):
        data = data.to_frame()
    data = data.sort_index().ffill().bfill()
    return data.iloc[-days:] if len(data) > days else data


@st.cache_data(show_spinner=False)
def compute_wave_history(wave_name: str, mode: str, days: int = 365) -> pd.DataFrame:
    try:
        return we.compute_history_nav(wave_name, mode=mode, days=days)
    except Exception:
        return pd.DataFrame(columns=["wave_nav", "bm_nav", "wave_ret", "bm_ret"])


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
    if nav is None or len(nav) < 2:
        return float("nan")
    window = min(window, len(nav))
    if window < 2:
        return float("nan")
    sub = nav.iloc[-window:]
    start, end = sub.iloc[0], sub.iloc[-1]
    return float(end / start - 1.0) if start > 0 else float("nan")


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
    if daily_wave is None or daily_bm is None or len(daily_wave) < 2 or len(daily_wave) != len(daily_bm):
        return float("nan")
    diff = (daily_wave - daily_bm).dropna()
    return float(diff.std() * np.sqrt(252)) if len(diff) >= 2 else float("nan")


def information_ratio(nav_wave: pd.Series, nav_bm: pd.Series, te: float) -> float:
    if te is None or te <= 0:
        return float("nan")
    rw = compute_return_from_nav(nav_wave, window=len(nav_wave))
    rb = compute_return_from_nav(nav_bm, window=len(nav_bm))
    return float((rw - rb) / te)


def simple_ret(series: pd.Series, window: int) -> float:
    if series is None or len(series) < 2:
        return float("nan")
    window = min(window, len(series))
    if window < 2:
        return float("nan")
    sub = series.iloc[-window:]
    return float(sub.iloc[-1] / sub.iloc[0] - 1.0)


# ------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------
all_waves = we.get_all_waves()
all_modes = we.get_modes()

with st.sidebar:
    st.title("WAVES Intelligence™")
    st.caption("Mini Bloomberg Console • Vector OS™")
    st.code(f"Engine: {sig}\nVersion: {ver}")

    mode = st.selectbox("Mode", all_modes, index=0)
    selected_wave = st.selectbox("Select Wave", all_waves, index=0)

    st.markdown("---")
    nav_days = st.slider("History window (days)", min_value=60, max_value=730, value=365, step=15)

st.title("WAVES Intelligence™ Institutional Console")
st.caption("Live Alpha Capture • SmartSafe™ • Multi-Asset • Crypto • Gold • Income Ladders")

tab_console, tab_market, tab_factors, tab_vector = st.tabs(
    ["Console", "Market Intel", "Factor Decomposition", "Vector OS Insight Layer"]
)

# ------------------------------------------------------------
# TAB 1: Console
# ------------------------------------------------------------
with tab_console:
    st.subheader("Market Regime Monitor — SPY vs VIX")
    spy_vix = fetch_spy_vix(days=nav_days)

    if spy_vix.empty or "SPY" not in spy_vix.columns or "^VIX" not in spy_vix.columns:
        st.warning("Unable to load SPY/VIX data at the moment.")
    else:
        spy = spy_vix["SPY"]
        vix = spy_vix["^VIX"]
        spy_norm = spy / spy.iloc[0] * 100.0 if len(spy) else spy

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=spy_vix.index, y=spy_norm, name="SPY (Index = 100)", mode="lines"))
        fig.add_trace(go.Scatter(x=spy_vix.index, y=vix, name="VIX Level", mode="lines", yaxis="y2"))
        fig.update_layout(
            margin=dict(l=40, r=40, t=40, b=40),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            xaxis=dict(title="Date"),
            yaxis=dict(title="SPY (Indexed)", rangemode="tozero"),
            yaxis2=dict(title="VIX", overlaying="y", side="right", rangemode="tozero"),
            height=400,
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("Multi-Window Alpha Capture (All Waves · Mode = {})".format(mode))

    alpha_rows = []
    for wave in all_waves:
        hist_365 = compute_wave_history(wave, mode=mode, days=365)
        if hist_365.empty or len(hist_365) < 2:
            alpha_rows.append({"Wave": wave, "1D Ret": np.nan, "1D Alpha": np.nan, "30D Ret": np.nan, "30D Alpha": np.nan,
                               "60D Ret": np.nan, "60D Alpha": np.nan, "365D Ret": np.nan, "365D Alpha": np.nan})
            continue

        nav_wave, nav_bm = hist_365["wave_nav"], hist_365["bm_nav"]
        ret_1d_wave = nav_wave.iloc[-1] / nav_wave.iloc[-2] - 1.0
        ret_1d_bm = nav_bm.iloc[-1] / nav_bm.iloc[-2] - 1.0
        alpha_1d = ret_1d_wave - ret_1d_bm

        ret_30_wave = compute_return_from_nav(nav_wave, 30)
        ret_30_bm = compute_return_from_nav(nav_bm, 30)
        alpha_30 = ret_30_wave - ret_30_bm

        ret_60_wave = compute_return_from_nav(nav_wave, 60)
        ret_60_bm = compute_return_from_nav(nav_bm, 60)
        alpha_60 = ret_60_wave - ret_60_bm

        ret_365_wave = compute_return_from_nav(nav_wave, len(nav_wave))
        ret_365_bm = compute_return_from_nav(nav_bm, len(nav_bm))
        alpha_365 = ret_365_wave - ret_365_bm

        alpha_rows.append({"Wave": wave, "1D Ret": ret_1d_wave, "1D Alpha": alpha_1d,
                           "30D Ret": ret_30_wave, "30D Alpha": alpha_30,
                           "60D Ret": ret_60_wave, "60D Alpha": alpha_60,
                           "365D Ret": ret_365_wave, "365D Alpha": alpha_365})

    alpha_df = pd.DataFrame(alpha_rows)
    fmt = alpha_df.copy()
    for col in [c for c in fmt.columns if c != "Wave"]:
        fmt[col] = fmt[col].apply(lambda x: f"{x*100:0.2f}%" if pd.notna(x) else "—")
    st.dataframe(fmt.set_index("Wave"), use_container_width=True)

    st.markdown("---")
    st.subheader("Benchmark ETF Mix (Composite Benchmarks)")
    bm_mix = get_benchmark_mix()
    if bm_mix.empty:
        st.info("No benchmark mix data available.")
    else:
        bm_mix = bm_mix.copy()
        bm_mix["Weight"] = bm_mix["Weight"].apply(lambda x: f"{x*100:0.2f}%" if pd.notna(x) else "—")
        st.dataframe(bm_mix, use_container_width=True)

    st.markdown("---")
    st.subheader(f"Wave Detail — {selected_wave} (Mode: {mode})")

    hist = compute_wave_history(selected_wave, mode=mode, days=nav_days)
    if hist.empty or len(hist) < 2:
        st.warning("Not enough data for this Wave.")
    else:
        nav_wave, nav_bm = hist["wave_nav"], hist["bm_nav"]
        fig_nav = go.Figure()
        fig_nav.add_trace(go.Scatter(x=hist.index, y=nav_wave, name=f"{selected_wave} NAV", mode="lines"))
        fig_nav.add_trace(go.Scatter(x=hist.index, y=nav_bm, name="Benchmark NAV", mode="lines"))
        fig_nav.update_layout(margin=dict(l=40, r=40, t=40, b=40), height=380,
                              xaxis=dict(title="Date"), yaxis=dict(title="NAV (Normalized)"))
        st.plotly_chart(fig_nav, use_container_width=True)

    st.markdown("#### Top-10 Holdings")
    holdings_df = get_wave_holdings(selected_wave)
    if holdings_df.empty:
        st.info("No holdings available for this Wave.")
    else:
        def google_link(ticker: str) -> str:
            return f"[{ticker}](https://www.google.com/finance/quote/{ticker})"
        fmt_hold = holdings_df.copy()
        fmt_hold["Weight"] = fmt_hold["Weight"].apply(lambda x: f"{x*100:0.2f}%" if pd.notna(x) else "—")
        fmt_hold["Google Finance"] = fmt_hold["Ticker"].apply(google_link)
        st.dataframe(fmt_hold[["Ticker", "Name", "Weight", "Google Finance"]], use_container_width=True)

# ------------------------------------------------------------
# TAB 2/3/4 kept minimal here (you can paste your full versions back in after it boots)
# ------------------------------------------------------------
with tab_market:
    st.info("Market Intel tab is temporarily minimized for stability. Once the engine import is verified, we can restore the full panel.")

with tab_factors:
    st.info("Factor Decomposition tab is temporarily minimized for stability. Once the engine import is verified, we can restore the full panel.")

with tab_vector:
    st.info("Vector OS Insight Layer is temporarily minimized for stability. Once the engine import is verified, we can restore the full panel.")