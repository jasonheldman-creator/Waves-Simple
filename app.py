# app.py — WAVES Intelligence™ Institutional Console
# Mini-Bloomberg style dashboard for WAVES Intelligence™
#
# Features:
#   • Tabs:
#       – Console: full WAVES dashboard
#       – Market: current market conditions, SPY/VIX, selected earnings, and Wave reactions
#   • Market Regime Monitor: SPY vs VIX on one chart
#   • Portfolio-Level Overview (All Waves)
#   • Multi-Window Alpha Capture (1D, 30D, 60D, 365D)
#   • Benchmark ETF Mix table
#   • Risk & WaveScore Ingredients (All Waves)
#   • Wave Detail view:
#       – NAV chart (Wave vs benchmark)
#       – Performance vs benchmark (30D / 365D)
#       – Mode comparison (Standard, Alpha-Minus-Beta, Private Logic)
#       – Top-10 holdings with Google Finance links
#
# Assumes waves_engine.py is in the same repo and matches the current spec.

from __future__ import annotations

import math
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

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

tab_console, tab_market = st.tabs(["Console", "Market"])


# ------------------------------------------------------------
# TAB 1: Console (existing dashboard)
# ------------------------------------------------------------

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

    if risk_rows:
        risk_df = pd.DataFrame(risk_rows)
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


# ------------------------------------------------------------
# TAB 2: Market — narrative market view
# ------------------------------------------------------------

with tab_market:
    st.subheader("Market Overview & Conditions")

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
        if spy_vix_m.empty or "SPY" not in spy_vix_m.columns or "^VIX" not in spy_vix_m.columns:
            st.info("No market summary available.")
        else:
            spy = spy_vix_m["SPY"].copy()
            vix = spy_vix_m["^VIX"].copy()

            current_spy = float(spy.iloc[-1]) if len(spy) else float("nan")
            current_vix = float(vix.iloc[-1]) if len(vix) else float("nan")

            # 30D & 60D SPY returns
            def ret_window(series: pd.Series, window: int) -> float:
                if len(series) < 2:
                    return float("nan")
                if len(series) < window:
                    window = len(series)
                if window < 2:
                    return float("nan")
                sub = series.iloc[-window:]
                return float(sub.iloc[-1] / sub.iloc[0] - 1.0)

            r30 = ret_window(spy, 30)
            r60 = ret_window(spy, 60)

            # simple realized vol (30D)
            if len(spy) > 30:
                daily_ret_spy = spy.pct_change().dropna()
                vol30 = float(daily_ret_spy.iloc[-30:].std() * np.sqrt(252))
            else:
                vol30 = float("nan")

            # Regime using engine helper (if available)
            try:
                regime = we._regime_from_return(r60)
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
                "VIX & realized vol indicate how aggressively or defensively SmartSafe & VIX scaling are behaving."
            )

    st.markdown("---")

    # Upcoming events: simple earnings for a few large caps (proxy)
    st.subheader("Upcoming Earnings (Selected Large Caps)")

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
                    # Try common index labels
                    if "Earnings Date" in cal.index:
                        val = cal.loc["Earnings Date"].iloc[0]
                        if hasattr(val, "date"):
                            next_earn = val.date()
                        else:
                            # fallback if it's a Timestamp/np.datetime64
                            try:
                                next_earn = pd.to_datetime(val).date()
                            except Exception:
                                next_earn = None
                    else:
                        # Fallback: take first cell
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

    st.markdown("---")

    # How WAVES are reacting and why — 30D view
    st.subheader("WAVES Reaction Snapshot (30D · Mode = {} )".format(mode))

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

        # Brief aggregate summary (numeric)
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
                "High positive alpha suggests momentum tilts, VIX scaling, SmartSafe sweeps, "
                "and mode exposure are adding value vs. static ETF mixes."
            )
        else:
            st.info("Not enough data yet to summarize Wave reactions.")
    else:
        st.info("No Wave reaction data available.")