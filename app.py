# app.py — WAVES Intelligence™ Institutional Console (Vector OS Edition)
# Adds: Benchmark Transparency + Alpha Attribution (Static Basket vs Engine)

from __future__ import annotations

import math
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

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
def compute_wave_history(wave_name: str, mode: str, days: int = 365) -> pd.DataFrame:
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


@st.cache_data(show_spinner=False)
def fetch_prices_daily(tickers: List[str], days: int = 365) -> pd.DataFrame:
    """
    Generic daily price fetch used for attribution (static baskets).
    """
    if yf is None or not tickers:
        return pd.DataFrame()

    end = datetime.utcnow().date()
    start = end - timedelta(days=days + 260)

    data = yf.download(
        tickers=sorted(list(set(tickers))),
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


def compute_return_from_nav(nav: pd.Series, window: int) -> float:
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


def information_ratio(nav_wave: pd.Series, nav_bm: pd.Series, te: float) -> float:
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


def regress_factors(wave_ret: pd.Series, factor_ret: pd.DataFrame) -> dict:
    df = pd.concat([wave_ret.rename("wave"), factor_ret], axis=1).dropna()
    if df.shape[0] < 60 or df.shape[1] < 2:
        return {col: float("nan") for col in factor_ret.columns}

    y = df["wave"].values
    X = df[factor_ret.columns].values
    X_design = np.concatenate([np.ones((X.shape[0], 1)), X], axis=1)

    try:
        beta, *_ = np.linalg.lstsq(X_design, y, rcond=None)
    except Exception:
        return {col: float("nan") for col in factor_ret.columns}

    betas = beta[1:]
    return {col: float(b) for col, b in zip(factor_ret.columns, betas)}


# ------------------------------------------------------------
# Alpha Attribution (Static Basket vs Engine)
# ------------------------------------------------------------

def _weights_from_df(df: pd.DataFrame, ticker_col: str = "Ticker", weight_col: str = "Weight") -> pd.Series:
    if df is None or df.empty:
        return pd.Series(dtype=float)
    w = df[[ticker_col, weight_col]].copy()
    w[ticker_col] = w[ticker_col].astype(str)
    w[weight_col] = pd.to_numeric(w[weight_col], errors="coerce").fillna(0.0)
    w = w.groupby(ticker_col, as_index=True)[weight_col].sum()
    total = float(w.sum())
    if total <= 0:
        return pd.Series(dtype=float)
    return (w / total).sort_index()


@st.cache_data(show_spinner=False)
def compute_static_nav_from_weights(weights: pd.Series, days: int = 365) -> pd.Series:
    """
    Build a simple fixed-weight NAV (no overlays) using daily prices.
    """
    if weights is None or weights.empty:
        return pd.Series(dtype=float)

    tickers = list(weights.index)
    px = fetch_prices_daily(tickers, days=days)
    if px.empty:
        return pd.Series(dtype=float)

    weights_aligned = weights.reindex(px.columns).fillna(0.0)
    daily_ret = px.pct_change().fillna(0.0)
    port_ret = (daily_ret * weights_aligned).sum(axis=1)

    nav = (1.0 + port_ret).cumprod()
    nav.name = "static_nav"
    return nav


@st.cache_data(show_spinner=False)
def compute_alpha_attribution(wave_name: str, mode: str, days: int = 365) -> Dict[str, float]:
    """
    Practical, defensible attribution:
      - Engine wave return
      - Static basket return (same holdings, fixed weights, no overlays)
      - Overlay contribution = Engine - Static
      - Alpha vs composite benchmark
      - Alpha vs SPY (context)
      - Benchmark difficulty = (Benchmark - SPY)
      - Active intensity proxies (TE, IR)
    """
    out: Dict[str, float] = {}

    hist = compute_wave_history(wave_name, mode=mode, days=days)
    if hist.empty or len(hist) < 2:
        return out

    nav_wave = hist["wave_nav"]
    nav_bm = hist["bm_nav"]
    wave_ret = hist["wave_ret"]
    bm_ret = hist["bm_ret"]

    # Engine returns
    eng_ret = compute_return_from_nav(nav_wave, window=len(nav_wave))
    bm_ret_total = compute_return_from_nav(nav_bm, window=len(nav_bm))
    alpha_vs_bm = eng_ret - bm_ret_total

    # Static basket NAV/return
    hold_df = get_wave_holdings(wave_name)
    hold_w = _weights_from_df(hold_df, ticker_col="Ticker", weight_col="Weight")
    static_nav = compute_static_nav_from_weights(hold_w, days=days)
    static_ret = compute_return_from_nav(static_nav, window=len(static_nav)) if len(static_nav) >= 2 else float("nan")

    overlay_pp = (eng_ret - static_ret) if (pd.notna(eng_ret) and pd.notna(static_ret)) else float("nan")

    # SPY context
    spy_px = fetch_prices_daily(["SPY"], days=days)
    spy_nav = (spy_px["SPY"].pct_change().fillna(0.0) + 1.0).cumprod() if "SPY" in spy_px.columns else pd.Series(dtype=float)
    spy_ret = compute_return_from_nav(spy_nav, window=len(spy_nav)) if len(spy_nav) >= 2 else float("nan")
    alpha_vs_spy = eng_ret - spy_ret if (pd.notna(eng_ret) and pd.notna(spy_ret)) else float("nan")

    benchmark_difficulty = bm_ret_total - spy_ret if (pd.notna(bm_ret_total) and pd.notna(spy_ret)) else float("nan")

    te = tracking_error(wave_ret, bm_ret)
    ir = information_ratio(nav_wave, nav_bm, te)

    out["Engine Return"] = float(eng_ret)
    out["Static Basket Return"] = float(static_ret) if pd.notna(static_ret) else float("nan")
    out["Overlay Contribution (Engine - Static)"] = float(overlay_pp) if pd.notna(overlay_pp) else float("nan")
    out["Benchmark Return"] = float(bm_ret_total)
    out["Alpha vs Benchmark"] = float(alpha_vs_bm)
    out["SPY Return"] = float(spy_ret) if pd.notna(spy_ret) else float("nan")
    out["Alpha vs SPY"] = float(alpha_vs_spy) if pd.notna(alpha_vs_spy) else float("nan")
    out["Benchmark Difficulty (BM - SPY)"] = float(benchmark_difficulty) if pd.notna(benchmark_difficulty) else float("nan")
    out["Tracking Error (TE)"] = float(te)
    out["Information Ratio (IR)"] = float(ir)

    # Risk context
    out["Wave Vol"] = float(annualized_vol(wave_ret))
    out["Benchmark Vol"] = float(annualized_vol(bm_ret))
    out["Wave MaxDD"] = float(max_drawdown(nav_wave))
    out["Benchmark MaxDD"] = float(max_drawdown(nav_bm))

    return out


# ------------------------------------------------------------
# WaveScore proto v1.0 (unchanged)
# ------------------------------------------------------------

def _grade_from_score(score: float) -> str:
    if math.isnan(score):
        return "N/A"
    if score >= 90:
        return "A+"
    if score >= 80:
        return "A"
    if score >= 70:
        return "B"
    if score >= 60:
        return "C"
    return "D"


def compute_wavescore_for_all_waves(all_waves: list[str], mode: str, days: int = 365) -> pd.DataFrame:
    rows = []
    for wave in all_waves:
        hist = compute_wave_history(wave, mode=mode, days=days)
        if hist.empty or len(hist) < 20:
            rows.append(
                {
                    "Wave": wave,
                    "WaveScore": float("nan"),
                    "Grade": "N/A",
                    "Return Quality": float("nan"),
                    "Risk Control": float("nan"),
                    "Consistency": float("nan"),
                    "Resilience": float("nan"),
                    "Efficiency": float("nan"),
                    "Transparency": 10.0,
                }
            )
            continue

        nav_wave = hist["wave_nav"]
        nav_bm = hist["bm_nav"]
        wave_ret = hist["wave_ret"]
        bm_ret = hist["bm_ret"]

        ret_365_wave = compute_return_from_nav(nav_wave, window=len(nav_wave))
        ret_365_bm = compute_return_from_nav(nav_bm, window=len(nav_bm))
        alpha_365 = ret_365_wave - ret_365_bm

        vol_wave = annualized_vol(wave_ret)
        vol_bm = annualized_vol(bm_ret)

        te = tracking_error(wave_ret, bm_ret)
        ir = information_ratio(nav_wave, nav_bm, te)

        mdd_wave = max_drawdown(nav_wave)
        mdd_bm = max_drawdown(nav_bm)

        hit_rate = float((wave_ret >= bm_ret).mean()) if len(wave_ret) > 0 else float("nan")

        if len(nav_wave) > 1:
            trough = nav_wave.min()
            peak = nav_wave.max()
            last = nav_wave.iloc[-1]
            if peak > trough and trough > 0:
                recovery_frac = float((last - trough) / (peak - trough))
                recovery_frac = float(np.clip(recovery_frac, 0.0, 1.0))
            else:
                recovery_frac = float("nan")
        else:
            recovery_frac = float("nan")

        vol_ratio = vol_wave / vol_bm if (vol_bm and not math.isnan(vol_bm)) else float("nan")

        # Return Quality (0–25)
        rq_ir = float(np.clip(ir, 0.0, 1.5) / 1.5 * 15.0) if not math.isnan(ir) else 0.0
        rq_alpha = float(np.clip((alpha_365 + 0.05) / 0.15, 0.0, 1.0) * 10.0) if not math.isnan(alpha_365) else 0.0
        return_quality = float(np.clip(rq_ir + rq_alpha, 0.0, 25.0))

        # Risk Control (0–25)
        if math.isnan(vol_ratio):
            risk_control = 0.0
        else:
            penalty = max(0.0, abs(vol_ratio - 0.9) - 0.1)
            risk_control = float(np.clip(1.0 - penalty / 0.6, 0.0, 1.0) * 25.0)

        # Consistency (0–15)
        consistency = float(np.clip(hit_rate, 0.0, 1.0) * 15.0) if not math.isnan(hit_rate) else 0.0

        # Resilience (0–10)
        if math.isnan(recovery_frac) or math.isnan(mdd_wave) or math.isnan(mdd_bm):
            resilience = 0.0
        else:
            rec_part = np.clip(recovery_frac, 0.0, 1.0) * 6.0
            dd_edge = (mdd_bm - mdd_wave)
            dd_part = float(np.clip((dd_edge + 0.10) / 0.20, 0.0, 1.0) * 4.0)
            resilience = float(np.clip(rec_part + dd_part, 0.0, 10.0))

        # Efficiency (0–15)
        efficiency = float(np.clip((0.25 - te) / 0.20, 0.0, 1.0) * 15.0) if not math.isnan(te) else 0.0

        transparency = 10.0

        total = float(np.clip(return_quality + risk_control + consistency + resilience + efficiency + transparency, 0.0, 100.0))
        grade = _grade_from_score(total)

        rows.append(
            {
                "Wave": wave,
                "WaveScore": total,
                "Grade": grade,
                "Return Quality": return_quality,
                "Risk Control": risk_control,
                "Consistency": consistency,
                "Resilience": resilience,
                "Efficiency": efficiency,
                "Transparency": transparency,
                "IR_365D": ir,
                "Alpha_365D": alpha_365,
            }
        )

    df = pd.DataFrame(rows) if rows else pd.DataFrame()
    return df.sort_values("Wave") if not df.empty else df


# ------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------

all_waves = we.get_all_waves()
all_modes = we.get_modes()

with st.sidebar:
    st.title("WAVES Intelligence™")
    st.caption("Mini Bloomberg Console • Vector OS™")

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
        spy_norm = (spy / spy.iloc[0] * 100.0) if len(spy) > 0 else spy

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

    # Portfolio-Level Overview
    st.subheader("Portfolio-Level Overview (All Waves)")

    overview_rows = []
    for wave in all_waves:
        hist_365 = compute_wave_history(wave, mode=mode, days=365)
        if hist_365.empty or len(hist_365) < 2:
            overview_rows.append({"Wave": wave, "365D Return": np.nan, "365D Alpha": np.nan, "30D Return": np.nan, "30D Alpha": np.nan})
            continue

        nav_wave = hist_365["wave_nav"]
        nav_bm = hist_365["bm_nav"]

        ret_365_wave = compute_return_from_nav(nav_wave, window=len(nav_wave))
        ret_365_bm = compute_return_from_nav(nav_bm, window=len(nav_bm))
        alpha_365 = ret_365_wave - ret_365_bm

        ret_30_wave = compute_return_from_nav(nav_wave, window=min(30, len(nav_wave)))
        ret_30_bm = compute_return_from_nav(nav_bm, window=min(30, len(nav_bm)))
        alpha_30 = ret_30_wave - ret_30_bm

        overview_rows.append({"Wave": wave, "365D Return": ret_365_wave, "365D Alpha": alpha_365, "30D Return": ret_30_wave, "30D Alpha": alpha_30})

    if overview_rows:
        overview_df = pd.DataFrame(overview_rows)
        fmt_overview = overview_df.copy()
        for col in ["365D Return", "365D Alpha", "30D Return", "30D Alpha"]:
            fmt_overview[col] = fmt_overview[col].apply(lambda x: f"{x*100:0.2f}%" if pd.notna(x) else "—")
        st.dataframe(fmt_overview.set_index("Wave"), use_container_width=True)
    else:
        st.info("No overview data available yet.")

    st.markdown("---")

    # Multi-Window Alpha Capture
    st.subheader(f"Multi-Window Alpha Capture (All Waves · Mode = {mode})")

    alpha_rows = []
    for wave in all_waves:
        hist_365 = compute_wave_history(wave, mode=mode, days=365)
        if hist_365.empty or len(hist_365) < 2:
            alpha_rows.append({"Wave": wave, "1D Ret": np.nan, "1D Alpha": np.nan, "30D Ret": np.nan, "30D Alpha": np.nan, "60D Ret": np.nan, "60D Alpha": np.nan, "365D Ret": np.nan, "365D Alpha": np.nan})
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

        alpha_rows.append({"Wave": wave, "1D Ret": ret_1d_wave, "1D Alpha": alpha_1d, "30D Ret": ret_30_wave, "30D Alpha": alpha_30, "60D Ret": ret_60_wave, "60D Alpha": alpha_60, "365D Ret": ret_365_wave, "365D Alpha": alpha_365})

    if alpha_rows:
        alpha_df = pd.DataFrame(alpha_rows)
        fmt_alpha = alpha_df.copy()
        for col in ["1D Ret", "1D Alpha", "30D Ret", "30D Alpha", "60D Ret", "60D Alpha", "365D Ret", "365D Alpha"]:
            fmt_alpha[col] = fmt_alpha[col].apply(lambda x: f"{x*100:0.2f}%" if pd.notna(x) else "—")
        st.dataframe(fmt_alpha.set_index("Wave"), use_container_width=True)
    else:
        st.info("No alpha capture data available yet.")

    st.markdown("---")

    # Risk & WaveScore Ingredients
    st.subheader("Risk & WaveScore Ingredients (All Waves · 365D Window)")

    risk_rows = []
    for wave in all_waves:
        hist_365 = compute_wave_history(wave, mode=mode, days=365)
        if hist_365.empty or len(hist_365) < 2:
            risk_rows.append({"Wave": wave, "Wave Vol (365D)": np.nan, "Benchmark Vol (365D)": np.nan, "Max Drawdown (Wave)": np.nan, "Max Drawdown (Benchmark)": np.nan, "Tracking Error": np.nan, "Information Ratio": np.nan})
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

        risk_rows.append({"Wave": wave, "Wave Vol (365D)": vol_wave, "Benchmark Vol (365D)": vol_bm, "Max Drawdown (Wave)": mdd_wave, "Max Drawdown (Benchmark)": mdd_bm, "Tracking Error": te, "Information Ratio": ir})

    if risk_rows:
        risk_df = pd.DataFrame(risk_rows)
        fmt_risk = risk_df.copy()
        for col in ["Wave Vol (365D)", "Benchmark Vol (365D)", "Tracking Error"]:
            fmt_risk[col] = fmt_risk[col].apply(lambda x: f"{x*100:0.2f}%" if pd.notna(x) else "—")
        for col in ["Max Drawdown (Wave)", "Max Drawdown (Benchmark)"]:
            fmt_risk[col] = fmt_risk[col].apply(lambda x: f"{x*100:0.2f}%" if pd.notna(x) else "—")
        fmt_risk["Information Ratio"] = fmt_risk["Information Ratio"].apply(lambda x: f"{x:0.2f}" if pd.notna(x) else "—")
        st.dataframe(fmt_risk.set_index("Wave"), use_container_width=True)
    else:
        st.info("No risk analytics available yet.")

    st.markdown("---")

    # WaveScore Leaderboard
    st.subheader("WaveScore™ Leaderboard (Proto v1.0 · 365D Data)")

    wavescore_df = compute_wavescore_for_all_waves(all_waves, mode=mode, days=365)
    if wavescore_df.empty:
        st.info("No WaveScore data available yet.")
    else:
        fmt_ws = wavescore_df.copy()
        fmt_ws["WaveScore"] = fmt_ws["WaveScore"].apply(lambda x: f"{x:0.1f}" if pd.notna(x) else "—")
        for col in ["Return Quality", "Risk Control", "Consistency", "Resilience", "Efficiency"]:
            fmt_ws[col] = fmt_ws[col].apply(lambda x: f"{x:0.1f}" if pd.notna(x) else "—")
        fmt_ws["Alpha_365D"] = fmt_ws["Alpha_365D"].apply(lambda x: f"{x*100:0.2f}%" if pd.notna(x) else "—")
        fmt_ws["IR_365D"] = fmt_ws["IR_365D"].apply(lambda x: f"{x:0.2f}" if pd.notna(x) else "—")
        st.dataframe(fmt_ws.set_index("Wave"), use_container_width=True)

    st.markdown("---")

    # Benchmark ETF Mix Table
    st.subheader("Benchmark ETF Mix (Composite Benchmarks)")

    bm_mix = get_benchmark_mix()
    if bm_mix.empty:
        st.info("No benchmark mix data available.")
    else:
        fmt_bm = bm_mix.copy()
        fmt_bm["Weight"] = fmt_bm["Weight"].apply(lambda x: f"{x*100:0.2f}%" if pd.notna(x) else "—")
        st.dataframe(fmt_bm, use_container_width=True)

    st.markdown("---")

    # Wave Detail View
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
            fig_nav.add_trace(go.Scatter(x=hist.index, y=nav_wave, name=f"{selected_wave} NAV", mode="lines"))
            fig_nav.add_trace(go.Scatter(x=hist.index, y=nav_bm, name="Benchmark NAV", mode="lines"))
            fig_nav.update_layout(
                margin=dict(l=40, r=40, t=40, b=40),
                xaxis=dict(title="Date"),
                yaxis=dict(title="NAV (Normalized)"),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                height=380,
            )
            st.plotly_chart(fig_nav, use_container_width=True)

    with col_stats:
        hist = compute_wave_history(selected_wave, mode=mode, days=nav_days)
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
            st.metric("30D Return", f"{ret_30_wave*100:0.2f}%" if not math.isnan(ret_30_wave) else "—")
            st.metric("30D Alpha", f"{alpha_30*100:0.2f}%" if not math.isnan(alpha_30) else "—")
            st.metric("365D Return", f"{ret_365_wave*100:0.2f}%" if not math.isnan(ret_365_wave) else "—")
            st.metric("365D Alpha", f"{alpha_365*100:0.2f}%" if not math.isnan(alpha_365) else "—")

    # NEW: Benchmark Transparency + Alpha Attribution
    st.markdown("---")
    with st.expander("Benchmark Transparency + Alpha Attribution (365D)", expanded=True):
        # Benchmark transparency for selected wave
        bm_mix_df = get_benchmark_mix()
        wave_bm = bm_mix_df[bm_mix_df["Wave"] == selected_wave].copy() if not bm_mix_df.empty else pd.DataFrame()
        st.markdown("### Benchmark Transparency")
        if wave_bm.empty:
            st.warning("No composite benchmark components found for this Wave (benchmark mix table empty).")
        else:
            wave_bm = wave_bm.sort_values("Weight", ascending=False)
            fmt = wave_bm.copy()
            fmt["Weight"] = fmt["Weight"].apply(lambda x: f"{x*100:0.2f}%" if pd.notna(x) else "—")
            st.dataframe(fmt[["Ticker", "Name", "Weight"]], use_container_width=True)

        st.markdown("### Alpha Attribution (Practical)")
        st.caption(
            "Attribution compares the **engine’s dynamic Wave NAV** vs a **static fixed-weight basket** "
            "of the same holdings (no SmartSafe/vol/VIX overlays). The difference is the engine overlay contribution."
        )

        attrib = compute_alpha_attribution(selected_wave, mode=mode, days=365)
        if not attrib:
            st.warning("Not enough data to compute attribution yet for this Wave.")
        else:
            # Summary metrics
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Engine Return (365D)", f"{attrib['Engine Return']*100:0.2f}%")
            c2.metric("Static Basket Return", f"{attrib['Static Basket Return']*100:0.2f}%" if pd.notna(attrib["Static Basket Return"]) else "—")
            c3.metric("Overlay Contribution", f"{attrib['Overlay Contribution (Engine - Static)']*100:0.2f}%" if pd.notna(attrib["Overlay Contribution (Engine - Static)"]) else "—")
            c4.metric("Alpha vs Benchmark", f"{attrib['Alpha vs Benchmark']*100:0.2f}%")

            c5, c6, c7, c8 = st.columns(4)
            c5.metric("Benchmark Return", f"{attrib['Benchmark Return']*100:0.2f}%")
            c6.metric("SPY Return", f"{attrib['SPY Return']*100:0.2f}%" if pd.notna(attrib["SPY Return"]) else "—")
            c7.metric("Benchmark Difficulty (BM−SPY)", f"{attrib['Benchmark Difficulty (BM - SPY)']*100:0.2f}%" if pd.notna(attrib["Benchmark Difficulty (BM - SPY)"]) else "—")
            c8.metric("Information Ratio (IR)", f"{attrib['Information Ratio (IR)']:.2f}" if pd.notna(attrib["Information Ratio (IR)"]) else "—")

            # Risk context
            st.markdown("#### Risk Context (365D)")
            r1, r2, r3, r4 = st.columns(4)
            r1.metric("Wave Vol", f"{attrib['Wave Vol']*100:0.2f}%" if pd.notna(attrib["Wave Vol"]) else "—")
            r2.metric("Benchmark Vol", f"{attrib['Benchmark Vol']*100:0.2f}%" if pd.notna(attrib["Benchmark Vol"]) else "—")
            r3.metric("Wave MaxDD", f"{attrib['Wave MaxDD']*100:0.2f}%" if pd.notna(attrib["Wave MaxDD"]) else "—")
            r4.metric("Benchmark MaxDD", f"{attrib['Benchmark MaxDD']*100:0.2f}%" if pd.notna(attrib["Benchmark MaxDD"]) else "—")

            # Small attribution bar visual
            st.markdown("#### Return Decomposition (Engine vs Static)")
            base = attrib.get("Static Basket Return", float("nan"))
            overlay = attrib.get("Overlay Contribution (Engine - Static)", float("nan"))
            if pd.notna(base) and pd.notna(overlay):
                fig_attr = go.Figure(
                    data=[
                        go.Bar(
                            x=["Static Basket", "Engine Overlay"],
                            y=[base, overlay],
                        )
                    ]
                )
                fig_attr.update_layout(
                    title="365D Return Components (Static Basket + Engine Overlay = Engine Return)",
                    xaxis_title="Component",
                    yaxis_title="Return",
                    height=320,
                    margin=dict(l=40, r=40, t=60, b=40),
                )
                st.plotly_chart(fig_attr, use_container_width=True)

            st.caption(
                "Interpretation: If the overlay contribution is positive, the WAVES system is adding value "
                "via dynamic exposure scaling, SmartSafe sweeps, vol targeting, and regime logic — beyond simply owning the names."
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
        fmt_mode["365D Return"] = fmt_mode["365D Return"].apply(lambda x: f"{x*100:0.2f}%" if pd.notna(x) else "—")
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
        fmt_hold["Weight"] = fmt_hold["Weight"].apply(lambda x: f"{x*100:0.2f}%" if pd.notna(x) else "—")
        fmt_hold["Google Finance"] = fmt_hold["Ticker"].apply(google_link)

        st.dataframe(fmt_hold[["Ticker", "Name", "Weight", "Google Finance"]], use_container_width=True)


# ============================================================
# TAB 2: Market Intel
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
            rows.append({"Ticker": tkr, "Asset": label, "Last": last, "1D Return": r1d, "30D Return": r30})

        snap_df = pd.DataFrame(rows)
        fmt_snap = snap_df.copy()

        def fmt_ret(x: float) -> str:
            return f"{x*100:0.2f}%" if pd.notna(x) else "—"

        fmt_snap["Last"] = fmt_snap["Last"].apply(lambda x: f"{x:0.2f}" if pd.notna(x) else "—")
        fmt_snap["1D Return"] = fmt_snap["1D Return"].apply(fmt_ret)
        fmt_snap["30D Return"] = fmt_snap["30D Return"].apply(fmt_ret)

        st.dataframe(fmt_snap.set_index("Ticker"), use_container_width=True)

    st.markdown("---")
    st.subheader("WAVES Reaction Snapshot (30D · Mode = {})".format(mode))

    reaction_rows = []
    for wave in all_waves:
        hist_365 = compute_wave_history(wave, mode=mode, days=365)
        if hist_365.empty or len(hist_365) < 2:
            reaction_rows.append({"Wave": wave, "30D Return": np.nan, "30D Alpha": np.nan, "Classification": "No data"})
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

        reaction_rows.append({"Wave": wave, "30D Return": r30_wave, "30D Alpha": a30, "Classification": label})

    if reaction_rows:
        reaction_df = pd.DataFrame(reaction_rows)
        fmt_reaction = reaction_df.copy()
        fmt_reaction["30D Return"] = fmt_reaction["30D Return"].apply(lambda x: f"{x*100:0.2f}%" if pd.notna(x) else "—")
        fmt_reaction["30D Alpha"] = fmt_reaction["30D Alpha"].apply(lambda x: f"{x*100:0.2f}%" if pd.notna(x) else "—")
        st.dataframe(fmt_reaction.set_index("Wave"), use_container_width=True)
    else:
        st.info("No Wave reaction data available.")


# ============================================================
# TAB 3: Factor Decomposition
# ============================================================

with tab_factors:
    st.subheader("Factor Decomposition (Institution-Level Analytics)")

    factor_days = min(nav_days, 365)
    factor_prices = fetch_market_assets(days=factor_days)

    needed = ["SPY", "QQQ", "IWM", "TLT", "GLD", "BTC-USD"]
    missing = [t for t in needed if t not in factor_prices.columns]

    if factor_prices.empty or missing:
        st.warning("Unable to load all factor price series. " + (f"Missing: {', '.join(missing)}" if missing else ""))
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
                rows.append({"Wave": wave, "β_SPY": np.nan, "β_QQQ": np.nan, "β_IWM": np.nan, "β_TLT": np.nan, "β_GLD": np.nan, "β_BTC": np.nan})
                continue

            wret = hist["wave_ret"]
            betas = regress_factors(wave_ret=wret, factor_ret=factor_returns)

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

        beta_df = pd.DataFrame(rows)
        fmt_beta = beta_df.copy()
        for col in ["β_SPY", "β_QQQ", "β_IWM", "β_TLT", "β_GLD", "β_BTC"]:
            fmt_beta[col] = fmt_beta[col].apply(lambda x: f"{x:0.2f}" if pd.notna(x) else "—")
        st.dataframe(fmt_beta.set_index("Wave"), use_container_width=True)

    st.markdown("---")
    st.subheader(f"Correlation Matrix — Waves (Daily Returns · Mode = {mode})")

    corr_days = min(nav_days, 365)
    ret_panel = {}

    for wave in all_waves:
        hist = compute_wave_history(wave, mode=mode, days=corr_days)
        if hist.empty or "wave_ret" not in hist.columns:
            continue
        ret_panel[wave] = hist["wave_ret"]

    if not ret_panel:
        st.info("No return data available to compute correlations.")
    else:
        ret_df = pd.DataFrame(ret_panel).dropna(how="all")
        if ret_df.empty or ret_df.shape[1] < 2:
            st.info("Not enough overlapping data to compute correlations.")
        else:
            corr = ret_df.corr()
            st.dataframe(corr, use_container_width=True)

            fig_corr = go.Figure(
                data=go.Heatmap(
                    z=corr.values,
                    x=corr.columns,
                    y=corr.index,
                    zmin=-1,
                    zmax=1,
                    colorbar=dict(title="ρ"),
                )
            )
            fig_corr.update_layout(
                title="Wave Correlation Matrix (Daily Returns)",
                xaxis_title="Wave",
                yaxis_title="Wave",
                height=500,
                margin=dict(l=60, r=60, t=60, b=60),
            )
            st.plotly_chart(fig_corr, use_container_width=True)


# ============================================================
# TAB 4: Vector OS Insight Layer
# ============================================================

with tab_vector:
    st.subheader("Vector OS Insight Layer — AI Chat / Insight Panel")
    st.caption("Rules-based narrative driven purely from current engine metrics (no external API).")

    ws_df = compute_wavescore_for_all_waves(all_waves, mode=mode, days=365)
    ws_row = ws_df[ws_df["Wave"] == selected_wave]
    hist = compute_wave_history(selected_wave, mode=mode, days=365)

    if ws_row.empty or hist.empty or len(hist) < 2:
        st.info("Not enough data yet for a full Vector OS insight on this Wave.")
    else:
        row = ws_row.iloc[0]
        nav_wave = hist["wave_nav"]
        nav_bm = hist["bm_nav"]
        wave_ret = hist["wave_ret"]
        bm_ret = hist["bm_ret"]

        ret_365_wave = compute_return_from_nav(nav_wave, window=len(nav_wave))
        ret_365_bm = compute_return_from_nav(nav_bm, window=len(nav_bm))
        alpha_365 = ret_365_wave - ret_365_bm

        vol_wave = annualized_vol(wave_ret)
        vol_bm = annualized_vol(bm_ret)
        mdd_wave = max_drawdown(nav_wave)
        mdd_bm = max_drawdown(nav_bm)

        question = st.text_input("Ask Vector about this Wave or the lineup:", "")

        st.markdown(f"### Vector’s Insight — {selected_wave}")
        st.write(f"- **WaveScore (proto)**: **{row['WaveScore']:.1f}/100** (**{row['Grade']}**).")
        st.write(f"- **365D return**: {ret_365_wave*100:0.2f}% vs benchmark {ret_365_bm*100:0.2f}% (alpha: {alpha_365*100:0.2f}%).")
        st.write(f"- **Volatility (365D)**: Wave {vol_wave*100:0.2f}% vs benc