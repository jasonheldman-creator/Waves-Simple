# app.py — WAVES Intelligence™ Institutional Console (Vector OS Edition)
# Mini-Bloomberg dashboard for WAVES Intelligence™
#
# Tabs:
#   • Console: full WAVES dashboard (alpha, risk, WaveScore proto)
#   • Market Intel: global market overview, events, and Wave reactions
#   • Factor Decomposition: factor betas + correlation matrix
#   • Vector OS Insight Layer: AI-style narrative on Waves using live metrics
#
# Additions in this version:
#   ✅ Volatility Regime Attribution (365D) inside Wave Detail:
#      - Uses ^VIX to label each day into regimes
#      - Computes Engine vs Static Basket vs Benchmark within each regime
#      - Shows Overlay Contribution by regime

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
def fetch_vix_only(days: int = 365) -> pd.Series:
    """
    Fetch ^VIX close series, last N days.
    Returns Series with Date index.
    """
    if yf is None:
        return pd.Series(dtype=float)

    end = datetime.utcnow().date()
    start = end - timedelta(days=days + 10)

    data = yf.download(
        tickers=["^VIX"],
        start=start.isoformat(),
        end=end.isoformat(),
        interval="1d",
        auto_adjust=False,  # VIX doesn't need auto_adjust
        progress=False,
        group_by="column",
    )

    # Normalize possible MultiIndex
    if isinstance(data.columns, pd.MultiIndex):
        if "Close" in data.columns.get_level_values(0):
            data = data["Close"]
        else:
            data = data[data.columns.levels[0][0]]

    if isinstance(data, pd.DataFrame) and "^VIX" in data.columns:
        s = data["^VIX"].copy()
    elif isinstance(data, pd.Series):
        s = data.copy()
    else:
        # best effort
        if isinstance(data, pd.DataFrame) and data.shape[1] >= 1:
            s = data.iloc[:, 0].copy()
        else:
            return pd.Series(dtype=float)

    s = s.sort_index().ffill().bfill()
    if len(s) > days:
        s = s.iloc[-days:]
    return s


@st.cache_data(show_spinner=False)
def fetch_prices(tickers: list[str], days: int = 365) -> pd.DataFrame:
    """
    Fetch daily price history for tickers, last N days. Returns auto_adjusted prices.
    """
    if yf is None or not tickers:
        return pd.DataFrame()

    end = datetime.utcnow().date()
    start = end - timedelta(days=days + 10)

    data = yf.download(
        tickers=tickers,
        start=start.isoformat(),
        end=end.isoformat(),
        interval="1d",
        auto_adjust=True,
        progress=False,
        group_by="column",
    )

    if data is None or len(data) == 0:
        return pd.DataFrame()

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

    # Ensure all requested columns exist
    for t in tickers:
        if t not in data.columns:
            data[t] = np.nan
    data = data[tickers].ffill().bfill()

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


def information_ratio_from_rets(wave_ret: pd.Series, bm_ret: pd.Series) -> float:
    """
    IR = mean(excess) / std(excess) * sqrt(252)
    Equivalent to excess return / tracking error on return series.
    """
    if wave_ret is None or bm_ret is None:
        return float("nan")
    df = pd.concat([wave_ret.rename("w"), bm_ret.rename("b")], axis=1).dropna()
    if df.shape[0] < 20:
        return float("nan")
    excess = df["w"] - df["b"]
    te = excess.std()
    if te is None or te <= 0 or math.isnan(te):
        return float("nan")
    return float(excess.mean() / te * np.sqrt(252))


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
# Volatility Regime Attribution (Option A)
# ------------------------------------------------------------

def label_vix_regime(vix_value: float) -> str:
    if vix_value is None or math.isnan(vix_value):
        return "Unknown"
    if vix_value < 15:
        return "Low Vol (<15)"
    if vix_value < 25:
        return "Normal (15–25)"
    if vix_value < 35:
        return "Elevated (25–35)"
    return "Stress (>35)"


def basket_nav_from_prices(prices: pd.DataFrame, weights: pd.Series) -> pd.Series:
    """
    Fixed-weight basket NAV normalized to 1.0 at start:
      NAV_t = sum_i w_i * (P_i,t / P_i,0)
    """
    if prices is None or prices.empty:
        return pd.Series(dtype=float)
    w = weights.copy().astype(float)
    w = w / w.sum() if w.sum() != 0 else w

    base = prices.iloc[0].replace(0, np.nan)
    rel = prices / base
    nav = (rel * w.values).sum(axis=1)
    nav = nav.replace([np.inf, -np.inf], np.nan).ffill().bfill()
    return nav


def regime_attribution_table(
    hist: pd.DataFrame,
    holdings_df: pd.DataFrame,
    days: int = 365,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Returns:
      - per-regime attribution dataframe
      - per-day regime labels aligned to hist.index
    """
    if hist is None or hist.empty or len(hist) < 40:
        return (pd.DataFrame(), pd.Series(dtype=str))

    idx = hist.index
    wave_ret = hist["wave_ret"].copy()
    bm_ret = hist["bm_ret"].copy()

    # VIX aligned
    vix = fetch_vix_only(days=days)
    vix = vix.reindex(idx).ffill().bfill()
    regimes = vix.apply(label_vix_regime)

    # Static basket from holdings
    static_ret = None
    static_nav = None

    if holdings_df is not None and not holdings_df.empty and yf is not None:
        h = holdings_df.copy()
        h = h.dropna(subset=["Ticker", "Weight"])
        h["Weight"] = pd.to_numeric(h["Weight"], errors="coerce")
        h = h.dropna(subset=["Weight"])
        h = h[h["Weight"] > 0]

        tickers = h["Ticker"].astype(str).tolist()
        weights = h["Weight"].astype(float).values
        if tickers and np.sum(weights) > 0:
            prices = fetch_prices(tickers, days=days).reindex(idx).ffill().bfill()
            w = pd.Series(weights / np.sum(weights))
            static_nav = basket_nav_from_prices(prices, w)
            static_ret = static_nav.pct_change().fillna(0.0)

    if static_ret is None or static_ret is False:
        static_ret = pd.Series(index=idx, data=np.nan)

    # Build regime table
    rows = []
    for reg in ["Low Vol (<15)", "Normal (15–25)", "Elevated (25–35)", "Stress (>35)"]:
        mask = regimes == reg
        if mask.sum() < 10:
            rows.append(
                {
                    "Regime": reg,
                    "Days": int(mask.sum()),
                    "Engine Return": np.nan,
                    "Static Return": np.nan,
                    "Overlay (Engine-Static)": np.nan,
                    "Benchmark Return": np.nan,
                    "Alpha vs Benchmark": np.nan,
                    "Engine Vol": np.nan,
                    "Benchmark Vol": np.nan,
                    "Hit Rate (Engine>BM)": np.nan,
                    "IR (vs BM)": np.nan,
                }
            )
            continue

        wr = wave_ret[mask].dropna()
        br = bm_ret[mask].dropna()
        sr = static_ret[mask].dropna()

        # align
        df = pd.concat([wr.rename("w"), br.rename("b"), sr.rename("s")], axis=1).dropna()
        if df.shape[0] < 10:
            rows.append(
                {
                    "Regime": reg,
                    "Days": int(mask.sum()),
                    "Engine Return": np.nan,
                    "Static Return": np.nan,
                    "Overlay (Engine-Static)": np.nan,
                    "Benchmark Return": np.nan,
                    "Alpha vs Benchmark": np.nan,
                    "Engine Vol": np.nan,
                    "Benchmark Vol": np.nan,
                    "Hit Rate (Engine>BM)": np.nan,
                    "IR (vs BM)": np.nan,
                }
            )
            continue

        # Total return within regime = compounded over the days in that regime
        eng_ret = float((1.0 + df["w"]).prod() - 1.0)
        bm_ret_tot = float((1.0 + df["b"]).prod() - 1.0)
        stat_ret_tot = float((1.0 + df["s"]).prod() - 1.0)

        overlay = eng_ret - stat_ret_tot
        alpha_vs_bm = eng_ret - bm_ret_tot

        eng_vol = float(df["w"].std() * np.sqrt(252))
        bm_vol = float(df["b"].std() * np.sqrt(252))

        hit_rate = float((df["w"] > df["b"]).mean())
        ir = information_ratio_from_rets(df["w"], df["b"])

        rows.append(
            {
                "Regime": reg,
                "Days": int(df.shape[0]),
                "Engine Return": eng_ret,
                "Static Return": stat_ret_tot,
                "Overlay (Engine-Static)": overlay,
                "Benchmark Return": bm_ret_tot,
                "Alpha vs Benchmark": alpha_vs_bm,
                "Engine Vol": eng_vol,
                "Benchmark Vol": bm_vol,
                "Hit Rate (Engine>BM)": hit_rate,
                "IR (vs BM)": ir,
            }
        )

    out = pd.DataFrame(rows)
    return out, regimes


# ------------------------------------------------------------
# WaveScore proto v1.0 (console implementation)
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
        ir = information_ratio_from_rets(wave_ret.dropna(), bm_ret.dropna())

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
        if math.isnan(ir):
            rq_ir = 0.0
        else:
            rq_ir = float(np.clip(ir, 0.0, 1.5) / 1.5 * 15.0)

        if math.isnan(alpha_365):
            rq_alpha = 0.0
        else:
            rq_alpha = float(np.clip((alpha_365 + 0.05) / 0.15, 0.0, 1.0) * 10.0)

        return_quality = float(np.clip(rq_ir + rq_alpha, 0.0, 25.0))

        # Risk Control (0–25)
        if math.isnan(vol_ratio):
            risk_control = 0.0
        else:
            penalty = max(0.0, abs(vol_ratio - 0.9) - 0.1)
            risk_control = float(np.clip(1.0 - penalty / 0.6, 0.0, 1.0) * 25.0)

        # Consistency (0–15)
        if math.isnan(hit_rate):
            consistency = 0.0
        else:
            consistency = float(np.clip(hit_rate, 0.0, 1.0) * 15.0)

        # Resilience (0–10)
        if math.isnan(recovery_frac) or math.isnan(mdd_wave) or math.isnan(mdd_bm):
            resilience = 0.0
        else:
            rec_part = np.clip(recovery_frac, 0.0, 1.0) * 6.0
            dd_edge = (mdd_bm - mdd_wave)
            dd_part = float(np.clip((dd_edge + 0.10) / 0.20, 0.0, 1.0) * 4.0)
            resilience = float(np.clip(rec_part + dd_part, 0.0, 10.0))

        # Efficiency (0–15)
        if math.isnan(te):
            efficiency = 0.0
        else:
            efficiency = float(np.clip((0.25 - te) / 0.20, 0.0, 1.0) * 15.0)

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

    st.subheader(f"Multi-Window Alpha Capture (All Waves · Mode = {mode})")

    alpha_rows = []
    for wave in all_waves:
        hist_365 = compute_wave_history(wave, mode=mode, days=365)
        if hist_365.empty or len(hist_365) < 2:
            alpha_rows.append(
                {"Wave": wave, "1D Ret": np.nan, "1D Alpha": np.nan, "30D Ret": np.nan, "30D Alpha": np.nan, "60D Ret": np.nan, "60D Alpha": np.nan, "365D Ret": np.nan, "365D Alpha": np.nan}
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
            {"Wave": wave, "1D Ret": ret_1d_wave, "1D Alpha": alpha_1d, "30D Ret": ret_30_wave, "30D Alpha": alpha_30, "60D Ret": ret_60_wave, "60D Alpha": alpha_60, "365D Ret": ret_365_wave, "365D Alpha": alpha_365}
        )

    if alpha_rows:
        alpha_df = pd.DataFrame(alpha_rows)
        fmt_alpha = alpha_df.copy()
        for col in ["1D Ret", "1D Alpha", "30D Ret", "30D Alpha", "60D Ret", "60D Alpha", "365D Ret", "365D Alpha"]:
            fmt_alpha[col] = fmt_alpha[col].apply(lambda x: f"{x*100:0.2f}%" if pd.notna(x) else "—")
        st.dataframe(fmt_alpha.set_index("Wave"), use_container_width=True)
    else:
        st.info("No alpha capture data available yet.")

    st.markdown("---")

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
        ir = information_ratio_from_rets(wave_ret.dropna(), bm_ret.dropna())

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

    st.subheader("Benchmark ETF Mix (Composite Benchmarks)")

    bm_mix = get_benchmark_mix()
    if bm_mix.empty:
        st.info("No benchmark mix data available.")
    else:
        fmt_bm = bm_mix.copy()
        fmt_bm["Weight"] = fmt_bm["Weight"].apply(lambda x: f"{x*100:0.2f}%" if pd.notna(x) else "—")
        st.dataframe(fmt_bm, use_container_width=True)

    st.markdown("---")

    # 7) Wave Detail View
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

    # ✅ NEW: Volatility Regime Attribution (365D)
    st.markdown("---")
    with st.expander("Volatility Regime Attribution (365D) — Option A (VIX Regimes)", expanded=True):
        st.caption(
            "This does NOT change the strategy. It conditionally decomposes performance by volatility environment. "
            "Regimes are labeled from ^VIX: Low (<15), Normal (15–25), Elevated (25–35), Stress (>35)."
        )

        hist_365 = compute_wave_history(selected_wave, mode=mode, days=365)
        if hist_365.empty or len(hist_365) < 40:
            st.info("Not enough 365D history to compute volatility regime attribution yet.")
        elif yf is None:
            st.info("yfinance is required to fetch ^VIX and holdings prices for regime attribution.")
        else:
            table, regimes = regime_attribution_table(hist_365, holdings_df, days=365)

            if table.empty:
                st.info("Unable to compute regime attribution (missing overlap or data).")
            else:
                fmt = table.copy()

                def pct(x):
                    return f"{x*100:0.2f}%" if pd.notna(x) else "—"

                for col in ["Engine Return", "Static Return", "Overlay (Engine-Static)", "Benchmark Return", "Alpha vs Benchmark", "Engine Vol", "Benchmark Vol"]:
                    fmt[col] = fmt[col].apply(pct)

                fmt["Hit Rate (Engine>BM)"] = fmt["Hit Rate (Engine>BM)"].apply(lambda x: f"{x*100:0.1f}%" if pd.notna(x) else "—")
                fmt["IR (vs BM)"] = fmt["IR (vs BM)"].apply(lambda x: f"{x:0.2f}" if pd.notna(x) else "—")

                st.dataframe(fmt.set_index("Regime"), use_container_width=True)

                # Overlay contribution bar chart
                bar_df = table.copy()
                bar_df = bar_df.dropna(subset=["Overlay (Engine-Static)"])
                if not bar_df.empty:
                    fig_bar = go.Figure(
                        data=[
                            go.Bar(
                                x=bar_df["Regime"],
                                y=bar_df["Overlay (Engine-Static)"],
                            )
                        ]
                    )
                    fig_bar.update_layout(
                        title="Overlay Contribution by Volatility Regime (Engine − Static)",
                        xaxis_title="Regime",
                        yaxis_title="Return Contribution",
                        height=380,
                        margin=dict(l=40, r=40, t=60, b=40),
                    )
                    st.plotly_chart(fig_bar, use_container_width=True)

                # One-line interpretation
                stress = table[table["Regime"] == "Stress (>35)"]
                elev = table[table["Regime"] == "Elevated (25–35)"]
                low = table[table["Regime"] == "Low Vol (<15)"]

                def safe_val(df, col):
                    if df is None or df.empty:
                        return None
                    v = df.iloc[0].get(col, np.nan)
                    return None if pd.isna(v) else float(v)

                ov_stress = safe_val(stress, "Overlay (Engine-Static)")
                ov_elev = safe_val(elev, "Overlay (Engine-Static)")
                ov_low = safe_val(low, "Overlay (Engine-Static)")

                lines = []
                if ov_stress is not None and ov_elev is not None:
                    if ov_stress > ov_elev:
                        lines.append("Overlay contribution is most concentrated in **Stress** volatility (convexity profile).")
                    else:
                        lines.append("Overlay contribution is meaningful in **Elevated/Stress** volatility regimes.")
                if ov_low is not None and ov_low < 0:
                    lines.append("In **Low Vol** regimes, the static basket can lead (markets are easy; overlay is protective, not always additive).")

                if lines:
                    st.markdown("**Interpretation:** " + " ".join(lines))


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
    st.subheader("Market Regime Monitor — SPY vs VIX")
    spy_vix_m = fetch_spy_vix(days=nav_days)

    col_left, col_right = st.columns([2.0, 1.0])

    with col_left:
        if spy_vix_m.empty or "SPY" not in spy_vix_m.columns or "^VIX" not in spy_vix_m.columns:
            st.warning("Unable to load SPY/VIX data at the moment.")
        else:
            spy = spy_vix_m["SPY"].copy()
            vix = spy_vix_m["^VIX"].copy()
            spy_norm = spy / spy.iloc[0] * 100.0 if len(spy) else spy

            fig_m = go.Figure()
            fig_m.add_trace(go.Scatter(x=spy_vix_m.index, y=spy_norm, name="SPY (Index = 100)", mode="lines"))
            fig_m.add_trace(go.Scatter(x=spy_vix_m.index, y=vix, name="VIX Level", mode="lines", yaxis="y2"))

            fig_m.update_layout(
                margin=dict(l=40, r=40, t=40, b=40),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                xaxis=dict(title="Date"),
                yaxis=dict(title="SPY (Indexed)", rangemode="tozero"),
                yaxis2=dict(title="VIX", overlaying="y", side="right", rangemode="tozero"),
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


# ============================================================
# TAB 3: Factor Decomposition
# ============================================================

with tab_factors:
    st.subheader("Factor Decomposition (Institution-Level Analytics)")
    st.caption(
        "Wave daily returns are regressed on SPY, QQQ, IWM, TLT, GLD, BTC-USD. "
        "Betas approximate sensitivity to market, growth/tech, small caps, rates, gold, and crypto."
    )

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
        st.markdown("### Factor Betas — All Waves")

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
    st.caption("Rules-based narrative from live metrics (no external API calls).")

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

        risk_bucket = "Moderate"
        if not math.isnan(vol_wave):
            if vol_wave < 0.12:
                risk_bucket = "Low"
            elif vol_wave > 0.25:
                risk_bucket = "High"

        alpha_bucket = "Neutral vs benchmark"
        if not math.isnan(alpha_365):
            if alpha_365 > 0.08:
                alpha_bucket = "Strong outperformance"
            elif alpha_365 > 0.03:
                alpha_bucket = "Outperforming"
            elif alpha_365 < -0.03:
                alpha_bucket = "Lagging"

        question = st.text_input("Ask Vector about this Wave or the lineup:", "")

        st.markdown(f"### Vector’s Insight — {selected_wave}")
        st.write(f"- **WaveScore (proto)**: **{row['WaveScore']:.1f}/100** (**{row['Grade']}**).")
        st.write(
            f"- **365D return**: {ret_365_wave*100:0.2f}% vs benchmark {ret_365_bm*100:0.2f}% "
            f"(alpha: {alpha_365*100:0.2f}%). → **{alpha_bucket}**."
        )
        st.write(
            f"- **Volatility (365D)**: Wave {vol_wave*100:0.2f}% vs benchmark {vol_bm*100:0.2f}% "
            f"→ **{risk_bucket} risk profile**."
        )
        st.write(f"- **Max drawdown (365D)**: Wave {mdd_wave*100:0.2f}% vs benchmark {mdd_bm*100:0.2f}%.")

        if question.strip():
            st.caption(f"(Vector registered your prompt: “{question.strip()}”. This panel is rules-based.)")