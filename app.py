# app.py — WAVES Intelligence™ Institutional Console (Vector OS Edition)
from __future__ import annotations

import math
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import waves_engine as we

try:
    import yfinance as yf
except Exception:
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
    if yf is None:
        return pd.DataFrame()
    end = datetime.utcnow().date()
    start = end - timedelta(days=days + 10)

    data = yf.download(
        tickers=["SPY", "^VIX"],
        start=start.isoformat(),
        end=end.isoformat(),
        interval="1d",
        auto_adjust=True,
        progress=False,
        group_by="column",
    )
    if isinstance(data.columns, pd.MultiIndex):
        lvl0 = data.columns.get_level_values(0)
        if "Adj Close" in lvl0:
            data = data["Adj Close"]
        elif "Close" in lvl0:
            data = data["Close"]
        else:
            data = data[data.columns.levels[0][0]]
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
        lvl0 = data.columns.get_level_values(0)
        if "Adj Close" in lvl0:
            data = data["Adj Close"]
        elif "Close" in lvl0:
            data = data["Close"]
        else:
            data = data[data.columns.levels[0][0]]
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


@st.cache_data(show_spinner=False)
def get_alpha_attrib(wave_name: str, mode: str, days: int = 365) -> dict:
    try:
        return we.compute_alpha_attribution_summary(wave_name, mode=mode, days=days)
    except Exception:
        return {}


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
    start = sub.iloc[0]
    end = sub.iloc[-1]
    if start <= 0:
        return float("nan")
    return float(end / start - 1.0)


def annualized_vol(daily_ret: pd.Series) -> float:
    if daily_ret is None or len(daily_ret) < 2:
        return float("nan")
    return float(daily_ret.std() * np.sqrt(252))


def max_drawdown(nav: pd.Series) -> float:
    if nav is None or len(nav) < 2:
        return float("nan")
    running_max = nav.cummax()
    dd = nav / running_max - 1.0
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


def information_ratio(nav_wave: pd.Series, nav_bm: pd.Series, te: float) -> float:
    if nav_wave is None or nav_bm is None or len(nav_wave) < 2 or len(nav_bm) < 2:
        return float("nan")
    if te is None or te <= 0:
        return float("nan")
    ret_wave = compute_return_from_nav(nav_wave, window=len(nav_wave))
    ret_bm = compute_return_from_nav(nav_bm, window=len(nav_bm))
    excess = ret_wave - ret_bm
    return float(excess / te)


# ------------------------------------------------------------
# Sidebar (FAIL SOFT)
# ------------------------------------------------------------
with st.sidebar:
    st.title("WAVES Intelligence™")
    st.caption("Mini Bloomberg Console • Vector OS™")

    try:
        all_waves = we.get_all_waves()
        all_modes = we.get_modes()
    except Exception as e:
        st.error(f"Engine failed to load: {e}")
        all_waves = []
        all_modes = ["Standard", "Alpha-Minus-Beta", "Private Logic"]

    if not all_waves:
        st.warning("No waves found. Check wave_weights.csv or engine config.")
        mode = st.selectbox("Mode", all_modes, index=0)
        selected_wave = st.selectbox("Select Wave", ["(none)"], index=0)
    else:
        mode = st.selectbox("Mode", all_modes, index=0)
        selected_wave = st.selectbox("Select Wave", all_waves, index=0)

    st.markdown("---")
    st.markdown("**Display settings**")
    nav_days = st.slider("History window (days)", min_value=60, max_value=730, value=365, step=15)


# ------------------------------------------------------------
# Page header + tabs
# ------------------------------------------------------------
st.title("WAVES Intelligence™ Institutional Console")
st.caption("Live Alpha Capture • SmartSafe™ • Multi-Asset • Crypto • Gold • Income Ladders")

tab_console, tab_market, tab_factors, tab_vector = st.tabs(
    ["Console", "Market Intel", "Factor Decomposition", "Vector OS Insight Layer"]
)

# ============================================================
# TAB 1: Console
# ============================================================
with tab_console:
    st.subheader("Market Regime Monitor — SPY vs VIX")
    spy_vix = fetch_spy_vix(days=nav_days)

    if spy_vix.empty or "SPY" not in spy_vix.columns or "^VIX" not in spy_vix.columns:
        st.warning("Unable to load SPY/VIX data at the moment.")
    else:
        spy = spy_vix["SPY"].copy()
        vix = spy_vix["^VIX"].copy()
        spy_norm = spy / spy.iloc[0] * 100.0

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

    st.subheader("Portfolio-Level Overview (All Waves)")
    if not all_waves:
        st.info("No overview available.")
    else:
        overview_rows = []
        for wave in all_waves:
            hist_365 = compute_wave_history(wave, mode=mode, days=365)
            if hist_365.empty or len(hist_365) < 2:
                overview_rows.append({"Wave": wave, "365D Return": np.nan, "365D Alpha": np.nan})
                continue

            nav_wave = hist_365["wave_nav"]
            nav_bm = hist_365["bm_nav"]

            ret_365_wave = compute_return_from_nav(nav_wave, window=len(nav_wave))
            ret_365_bm = compute_return_from_nav(nav_bm, window=len(nav_bm))
            alpha_365 = ret_365_wave - ret_365_bm

            overview_rows.append({"Wave": wave, "365D Return": ret_365_wave, "365D Alpha": alpha_365})

        overview_df = pd.DataFrame(overview_rows)
        fmt = overview_df.copy()
        fmt["365D Return"] = fmt["365D Return"].apply(lambda x: f"{x*100:0.2f}%" if pd.notna(x) else "—")
        fmt["365D Alpha"] = fmt["365D Alpha"].apply(lambda x: f"{x*100:0.2f}%" if pd.notna(x) else "—")
        st.dataframe(fmt.set_index("Wave"), use_container_width=True)

    st.markdown("---")

    st.subheader(f"Wave Detail — {selected_wave} (Mode: {mode})")
    if not all_waves or selected_wave == "(none)":
        st.info("Select a wave.")
    else:
        col_chart, col_stats = st.columns([2.0, 1.0])

        with col_chart:
            hist = compute_wave_history(selected_wave, mode=mode, days=nav_days)
            if hist.empty or len(hist) < 2:
                st.warning("Not enough data to display NAV chart.")
            else:
                fig_nav = go.Figure()
                fig_nav.add_trace(go.Scatter(x=hist.index, y=hist["wave_nav"], name=f"{selected_wave} NAV", mode="lines"))
                fig_nav.add_trace(go.Scatter(x=hist.index, y=hist["bm_nav"], name="Benchmark NAV", mode="lines"))
                fig_nav.update_layout(
                    margin=dict(l=40, r=40, t=40, b=40),
                    xaxis=dict(title="Date"),
                    yaxis=dict(title="NAV (Normalized)"),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    height=380,
                )
                st.plotly_chart(fig_nav, use_container_width=True)

        with col_stats:
            hist = compute_wave_history(selected_wave, mode=mode, days=365)
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
                st.metric("30D Alpha", f"{alpha_30*100:0.2f}%" if math.isfinite(alpha_30) else "—")
                st.metric("365D Return", f"{ret_365_wave*100:0.2f}%" if math.isfinite(ret_365_wave) else "—")
                st.metric("365D Alpha", f"{alpha_365*100:0.2f}%" if math.isfinite(alpha_365) else "—")

        # ------------------------------------------------------------
        # Benchmark Transparency + Alpha Attribution (365D)
        # ------------------------------------------------------------
        st.markdown("---")
        with st.expander("Benchmark Transparency + Alpha Attribution (365D)", expanded=True):
            bm_mix = get_benchmark_mix()
            if bm_mix.empty:
                st.info("No benchmark mix available for this wave (defaulting to SPY in engine).")
            else:
                sub = bm_mix[bm_mix["Wave"] == selected_wave].copy()
                if sub.empty:
                    st.info("No benchmark mix rows found for this wave.")
                else:
                    sub["Weight"] = sub["Weight"].apply(lambda x: f"{x*100:0.2f}%" if pd.notna(x) else "—")
                    st.markdown("### Benchmark Transparency")
                    st.dataframe(sub[["Ticker", "Name", "Weight"]], use_container_width=True)

            st.markdown("### Alpha Attribution (Practical)")
            st.caption(
                "Attribution compares the engine’s dynamic Wave NAV vs a static fixed-weight basket of the same holdings "
                "(no SmartSafe/VIX overlay). The difference is the overlay contribution."
            )

            attrib = get_alpha_attrib(selected_wave, mode=mode, days=365)
            if not attrib:
                st.warning("Attribution not available (missing data).")
            else:
                engine_r = attrib["engine_return"]
                static_r = attrib["static_return"]
                overlay = attrib["overlay_contribution"]
                alpha = attrib["alpha_vs_benchmark"]
                bm_r = attrib["benchmark_return"]
                spy_r = attrib["spy_return"]
                bm_diff = attrib["benchmark_difficulty"]
                ir = attrib["information_ratio"]

                st.metric("Engine Return (365D)", f"{engine_r*100:0.2f}%")
                st.metric("Static Basket Return", f"{static_r*100:0.2f}%")
                st.metric("Overlay Contribution", f"{overlay*100:0.2f}%")
                st.metric("Alpha vs Benchmark", f"{alpha*100:0.2f}%")
                st.metric("Benchmark Return", f"{bm_r*100:0.2f}%")
                st.metric("SPY Return", f"{spy_r*100:0.2f}%" if math.isfinite(spy_r) else "—")
                st.metric("Benchmark Difficulty (BM–SPY)", f"{bm_diff*100:0.2f}%" if math.isfinite(bm_diff) else "—")
                st.metric("Information Ratio (IR)", f"{ir:0.2f}" if math.isfinite(ir) else "—")

                st.markdown("### Risk Context (365D)")
                st.metric("Wave Vol", f"{attrib['wave_vol']*100:0.2f}%" if math.isfinite(attrib["wave_vol"]) else "—")
                st.metric("Benchmark Vol", f"{attrib['bm_vol']*100:0.2f}%" if math.isfinite(attrib["bm_vol"]) else "—")
                st.metric("Wave MaxDD", f"{attrib['wave_maxdd']*100:0.2f}%" if math.isfinite(attrib["wave_maxdd"]) else "—")
                st.metric("Benchmark MaxDD", f"{attrib['bm_maxdd']*100:0.2f}%" if math.isfinite(attrib["bm_maxdd"]) else "—")

                st.markdown("### Return Decomposition (Engine vs Static)")
                comp_df = pd.DataFrame(
                    {
                        "Component": ["Static Basket", "Engine Overlay"],
                        "Return": [static_r, overlay],
                    }
                )
                fig_bar = go.Figure(data=[go.Bar(x=comp_df["Component"], y=comp_df["Return"])])
                fig_bar.update_layout(
                    title="365D Return Components (Static Basket + Engine Overlay)",
                    xaxis_title="Component",
                    yaxis_title="Return",
                    height=380,
                    margin=dict(l=40, r=40, t=60, b=40),
                )
                st.plotly_chart(fig_bar, use_container_width=True)

        # ------------------------------------------------------------
        # Top-10 holdings
        # ------------------------------------------------------------
        st.markdown("---")
        st.markdown("### Top-10 Holdings")
        holdings_df = get_wave_holdings(selected_wave)

        if holdings_df.empty:
            st.info("No holdings available for this Wave.")
        else:
            def google_link(ticker: str) -> str:
                base = "https://www.google.com/finance/quote"
                return f"[{ticker}]({base}/{ticker})"

            h = holdings_df.copy().sort_values("Weight", ascending=False).head(10)
            h["Weight"] = h["Weight"].apply(lambda x: f"{x*100:0.2f}%" if pd.notna(x) else "—")
            h["Google Finance"] = h["Ticker"].apply(google_link)
            st.dataframe(h[["Ticker", "Name", "Weight", "Google Finance"]], use_container_width=True)


# ============================================================
# TAB 2: Market Intel (kept light)
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
            "^VIX": "VIX",
            "^TNX": "US 10Y Yield",
        }
        rows = []
        for tkr, label in assets.items():
            if tkr not in market_df.columns:
                continue
            s = market_df[tkr]
            last = float(s.iloc[-1]) if len(s) else float("nan")
            r1d = float(s.iloc[-1] / s.iloc[-2] - 1.0) if len(s) > 1 else float("nan")
            r30 = float(s.iloc[-1] / s.iloc[-30] - 1.0) if len(s) > 30 else float("nan")
            rows.append({"Ticker": tkr, "Asset": label, "Last": last, "1D Return": r1d, "30D Return": r30})

        snap = pd.DataFrame(rows)
        snap["Last"] = snap["Last"].apply(lambda x: f"{x:0.2f}" if pd.notna(x) else "—")
        snap["1D Return"] = snap["1D Return"].apply(lambda x: f"{x*100:0.2f}%" if pd.notna(x) else "—")
        snap["30D Return"] = snap["30D Return"].apply(lambda x: f"{x*100:0.2f}%" if pd.notna(x) else "—")
        st.dataframe(snap.set_index("Ticker"), use_container_width=True)


# ============================================================
# TAB 3: Factor Decomposition (placeholder)
# ============================================================
with tab_factors:
    st.subheader("Factor Decomposition (Next)")
    st.info("We can add a clean, stable factor model next (betas + factor attribution of alpha).")


# ============================================================
# TAB 4: Vector OS Insight Layer (placeholder)
# ============================================================
with tab_vector:
    st.subheader("Vector OS Insight Layer")
    st.info("Next we’ll convert the Alpha Attribution + Risk Context into a clean Vector narrative per Wave.")