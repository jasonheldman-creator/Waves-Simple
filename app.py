# app.py — WAVES Intelligence™ Institutional Console (Vector OS Edition)
# Mini-Bloomberg dashboard for WAVES Intelligence™
#
# Tabs:
#   • Console: full WAVES dashboard (alpha, risk, WaveScore proto)
#   • Market Intel: global market overview, events, and Wave reactions
#   • Factor Decomposition: factor betas + correlation matrix
#   • Vector OS Insight Layer: AI-style narrative on Waves using live metrics
#
# Console features:
#   • Market Regime Monitor: SPY vs VIX
#   • Portfolio-Level Overview (All Waves)
#   • Multi-Window Alpha Capture (1D, 30D, 60D, 365D)
#   • Risk & WaveScore Ingredients (All Waves)
#   • WaveScore Leaderboard (proto v1.0; 365D-based)
#   • Benchmark ETF Mix table
#   • Wave Detail view (NAV, stats, mode comparison, Top-10 holdings)
#
# Market Intel features:
#   • Global Market Dashboard (SPY, QQQ, IWM, TLT, GLD, BTC, VIX, 10Y)
#   • Market Regime Monitor (SPY vs VIX) + regime label
#   • Macro & Events panel (earnings snapshot + themes)
#   • WAVES Reaction Snapshot (30D)
#
# Factor Decomposition features:
#   • Factor betas vs SPY, QQQ, IWM, TLT, GLD, BTC-USD
#   • Factor bar chart for selected Wave
#   • Correlation matrix of Waves (daily returns, selected mode)
#
# Vector OS Insight Layer:
#   • Uses WaveScore proto, alpha, vol, drawdown to generate narrative
#   • Simple “ask Vector” prompt box (rules-based – no external API calls)

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
    X_design = np.concatenate([np.ones((X.shape[0], 1)), X], axis=1)

    try:
        beta, *_ = np.linalg.lstsq(X_design, y, rcond=None)
    except Exception:
        return {col: float("nan") for col in factor_ret.columns}

    betas = beta[1:]
    return {col: float(b) for col, b in zip(factor_ret.columns, betas)}


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


def compute_wavescore_for_all_waves(
    all_waves: list[str],
    mode: str,
    days: int = 365,
) -> pd.DataFrame:
    """
    Compute a proto WaveScore v1.0 using 365D daily data only.
    This is a console version aligned with the locked WaveScore spec:
      - Return Quality
      - Risk Control
      - Consistency
      - Resilience
      - Efficiency
      - Transparency & Governance (constant high score here, since engine is fully specified)
    """
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

        # Core performance & risk stats
        ret_365_wave = compute_return_from_nav(nav_wave, window=len(nav_wave))
        ret_365_bm = compute_return_from_nav(nav_bm, window=len(nav_bm))
        alpha_365 = ret_365_wave - ret_365_bm

        vol_wave = annualized_vol(wave_ret)
        vol_bm = annualized_vol(bm_ret)

        te = tracking_error(wave_ret, bm_ret)
        ir = information_ratio(nav_wave, nav_bm, te)

        mdd_wave = max_drawdown(nav_wave)
        mdd_bm = max_drawdown(nav_bm)

        # Hit-rate: proxy for monthly consistency
        hit_rate = float((wave_ret >= bm_ret).mean()) if len(wave_ret) > 0 else float("nan")

        # Recovery fraction: 0 => still at bottom, 1 => fully off the lows
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

        # --- Sub-scores (bounded) ---

        # 1) Return Quality (0–25): 15 pts from IR, 10 pts from 365D alpha
        if math.isnan(ir):
            rq_ir = 0.0
        else:
            # IR <= 0 → 0, IR >= 1.5 → full 15
            rq_ir = float(np.clip(ir, 0.0, 1.5) / 1.5 * 15.0)

        if math.isnan(alpha_365):
            rq_alpha = 0.0
        else:
            # Alpha: -5% → 0, +10% → full 10
            rq_alpha = float(np.clip((alpha_365 + 0.05) / 0.15, 0.0, 1.0) * 10.0)

        return_quality = rq_ir + rq_alpha
        return_quality = float(np.clip(return_quality, 0.0, 25.0))

        # 2) Risk Control (0–25): use volatility vs benchmark as proxy
        if math.isnan(vol_ratio):
            risk_control = 0.0
        else:
            # Ideal vol_ratio in [0.8, 1.0]. Penalize high overshoot.
            penalty = max(0.0, abs(vol_ratio - 0.9) - 0.1)  # tolerance band
            # penalty 0 → 25, penalty >= 0.6 → 0
            risk_control = float(np.clip(1.0 - penalty / 0.6, 0.0, 1.0) * 25.0)

        # 3) Consistency (0–15): hit-rate on daily alpha
        if math.isnan(hit_rate):
            consistency = 0.0
        else:
            consistency = float(np.clip(hit_rate, 0.0, 1.0) * 15.0)

        # 4) Resilience (0–10): how far off the lows we are + relative maxDD
        if math.isnan(recovery_frac) or math.isnan(mdd_wave) or math.isnan(mdd_bm):
            resilience = 0.0
        else:
            # part from recovery, part from shallower drawdown vs benchmark
            rec_part = np.clip(recovery_frac, 0.0, 1.0) * 6.0
            dd_edge = (mdd_bm - mdd_wave)  # less negative than benchmark is good
            dd_part = float(np.clip((dd_edge + 0.10) / 0.20, 0.0, 1.0) * 4.0)
            resilience = float(np.clip(rec_part + dd_part, 0.0, 10.0))

        # 5) Efficiency (0–15): lower tracking error is better
        if math.isnan(te):
            efficiency = 0.0
        else:
            # TE <= 5% → 15, TE >= 25% → 0
            efficiency = float(np.clip((0.25 - te) / 0.20, 0.0, 1.0) * 15.0)

        # 6) Transparency & Governance (0–10): console-level implementation → constant high score
        transparency = 10.0

        total = return_quality + risk_control + consistency + resilience + efficiency + transparency
        total = float(np.clip(total, 0.0, 100.0))
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

    if not rows:
        return pd.DataFrame(
            columns=[
                "Wave",
                "WaveScore",
                "Grade",
                "Return Quality",
                "Risk Control",
                "Consistency",
                "Resilience",
                "Efficiency",
                "Transparency",
                "IR_365D",
                "Alpha_365D",
            ]
        )

    df = pd.DataFrame(rows)
    return df.sort_values("Wave")


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
st.caption("Live Alpha Capture • SmartSafe™ • Multi-Asset • Crypto • Gold • Income Ladders")

tab_console, tab_market, tab_factors, tab_vector = st.tabs(
    ["Console", "Market Intel", "Factor Decomposition", "Vector OS Insight Layer"]
)


# ============================================================
# TAB 1: Console (existing dashboard + WaveScore leaderboard)
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

    # 5) WaveScore Leaderboard (proto v1.0; 365D-only)
    st.subheader("WaveScore™ Leaderboard (Proto v1.0 · 365D Data)")

    wavescore_df = compute_wavescore_for_all_waves(all_waves, mode=mode, days=365)

    if wavescore_df.empty:
        st.info("No WaveScore data available yet.")
    else:
        fmt_ws = wavescore_df.copy()

        fmt_ws["WaveScore"] = fmt_ws["WaveScore"].apply(
            lambda x: f"{x:0.1f}" if pd.notna(x) else "—"
        )
        for col in ["Return Quality", "Risk Control", "Consistency", "Resilience", "Efficiency"]:
            fmt_ws[col] = fmt_ws[col].apply(
                lambda x: f"{x:0.1f}" if pd.notna(x) else "—"
            )
        fmt_ws["Alpha_365D"] = fmt_ws["Alpha_365D"].apply(
            lambda x: f"{x*100:0.2f}%" if pd.notna(x) else "—"
        )
        fmt_ws["IR_365D"] = fmt_ws["IR_365D"].apply(
            lambda x: f"{x:0.2f}" if pd.notna(x) else "—"
        )

        st.dataframe(
            fmt_ws.set_index("Wave"),
            use_container_width=True,
        )

        st.caption(
            "Console-level WaveScore: 365D version of the locked WaveScore v1.0 spec, using daily data only. "
            "Full production WaveScore uses multi-year windows and full factor/fee/tax overlays."
        )

    st.markdown("---")

    # 6) Benchmark ETF Mix Table
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
# TAB 3: Factor Decomposition — factor betas + correlation matrix
# ============================================================

with tab_factors:
    st.subheader("Factor Decomposition (Institution-Level Analytics)")

    st.caption(
        "Wave daily returns are regressed on key risk premia: SPY, QQQ, IWM, TLT, GLD, BTC-USD. "
        "Betas approximate sensitivity to market, growth/tech, small caps, rates, gold, and crypto."
    )

    factor_days = min(nav_days, 365)
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

    st.markdown("---")

    # Correlation matrix of Waves
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
# TAB 4: Vector OS Insight Layer — AI-style narrative
# ============================================================

with tab_vector:
    st.subheader("Vector OS Insight Layer — AI Chat / Insight Panel")

    st.caption(
        "This panel gives an AI-style narrative using live metrics from the engine: "
        "WaveScore proto, alpha, volatility, drawdowns, and consistency."
    )

    # Compute WaveScore + metrics
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

        # Basic characterizations
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

        dd_edge = None
        if not (math.isnan(mdd_wave) or math.isnan(mdd_bm)):
            dd_edge = mdd_bm - mdd_wave  # positive = shallower drawdown

        question = st.text_input(
            "Ask Vector about this Wave or the lineup (rules-based, no external API):",
            "",
        )

        st.markdown("### Vector’s Insight — {}".format(selected_wave))

        st.write(
            f"- **WaveScore (proto)**: **{row['WaveScore']:.1f}/100** (**{row['Grade']}**)."
        )
        st.write(
            f"- **365D return**: {ret_365_wave*100:0.2f}% vs benchmark {ret_365_bm*100:0.2f}% "
            f"(alpha: {alpha_365*100:0.2f}%). → **{alpha_bucket}**."
        )
        st.write(
            f"- **Volatility (365D)**: Wave {vol_wave*100:0.2f}% vs benchmark {vol_bm*100:0.2f}% "
            f"→ **{risk_bucket} risk profile**."
        )
        st.write(
            f"- **Max drawdown (365D)**: Wave {mdd_wave*100:0.2f}% vs benchmark {mdd_bm*100:0.2f}%."
        )

        if dd_edge is not None:
            if dd_edge > 0.02:
                st.write(
                    "- **Drawdown behavior**: The Wave drew down **less** than its benchmark "
                    "(shallower max drawdown), indicating effective downside control, "
                    "likely helped by VIX scaling and SmartSafe sweeps."
                )
            elif dd_edge < -0.02:
                st.write(
                    "- **Drawdown behavior**: The Wave experienced **deeper** drawdowns than its benchmark. "
                    "This is the price of higher risk or more aggressive alpha targeting in this sleeve."
                )
            else:
                st.write(
                    "- **Drawdown behavior**: Max drawdown is roughly in line with the benchmark, "
                    "suggesting a balanced risk posture."
                )

        # WaveScore component commentary
        st.markdown("#### WaveScore Components (Console Proto)")
        st.write(
            f"- **Return Quality**: {row['Return Quality']:.1f}/25 — driven by information ratio "
            f"and 365D excess return."
        )
        st.write(
            f"- **Risk Control**: {row['Risk Control']:.1f}/25 — reflects how volatility compares "
            "to the composite benchmark."
        )
        st.write(
            f"- **Consistency**: {row['Consistency']:.1f}/15 — based on the hit-rate of daily alpha."
        )
        st.write(
            f"- **Resilience**: {row['Resilience']:.1f}/10 — how effectively the Wave recovered "
            "off its 12M lows vs the benchmark."
        )
        st.write(
            f"- **Efficiency**: {row['Efficiency']:.1f}/15 — lower tracking error scores higher, "
            "indicating cleaner, more intentional deviations from the benchmark."
        )

        st.markdown("#### Vector’s Take")

        narrative_lines = []

        # Alpha stance
        if alpha_365 > 0.08:
            narrative_lines.append(
                "This Wave is running in **high-conviction alpha mode**, clearly beating its composite "
                "benchmark over the last year."
            )
        elif alpha_365 > 0.03:
            narrative_lines.append(
                "This Wave is **consistently adding value**, delivering positive alpha on a 12-month basis."
            )
        elif alpha_365 < -0.03:
            narrative_lines.append(
                "This Wave has **struggled vs its benchmark** over the past 12 months, which may indicate "
                "either a tough regime for its style or an opportunity to refine parameters."
            )
        else:
            narrative_lines.append(
                "This Wave is tracking **close to its benchmark**, behaving more like a smart ETF replica "
                "than a high-octane satellite sleeve."
            )

        # Risk stance
        if not math.isnan(vol_wave) and not math.isnan(vol_bm):
            vol_ratio = vol_wave / vol_bm if vol_bm > 0 else float("nan")
            if not math.isnan(vol_ratio):
                if vol_ratio > 1.25:
                    narrative_lines.append(
                        "Risk-wise, it’s running **hotter than benchmark**, leaning into volatility "
                        "to chase more upside — appropriate for aggressive sleeves or Private Logic."
                    )
                elif vol_ratio < 0.80:
                    narrative_lines.append(
                        "Volatility is **materially lower than benchmark**, consistent with an "
                        "Alpha-Minus-Beta or SmartSafe-enhanced profile."
                    )
                else:
                    narrative_lines.append(
                        "Volatility is **close to the benchmark**, suggesting a balanced risk posture "
                        "with targeted tilts rather than extreme bets."
                    )

        # WaveScore angle
        if row["WaveScore"] >= 90:
            narrative_lines.append(
                "From a WaveScore perspective, this sits in **A+ territory** — institution-ready in terms "
                "of discipline, efficiency, and risk/return balance."
            )
        elif row["WaveScore"] >= 80:
            narrative_lines.append(
                "WaveScore puts this in the **A band**, already looking like an attractive institutional "
                "sleeve even before full production WaveScore and deeper history are layered in."
            )
        elif row["WaveScore"] >= 70:
            narrative_lines.append(
                "WaveScore in the **B range** suggests a strong core, with room to optimize one or two "
                "dimensions (consistency, resilience, or efficiency) to elevate it into flagship status."
            )
        else:
            narrative_lines.append(
                "A lower WaveScore here flags the Wave as a **work-in-progress**, best positioned right now "
                "as a satellite or R&D sleeve rather than a primary institutional core."
            )

        # Optional: lightly acknowledge the question without trying to parse it
        if question.strip():
            narrative_lines.append(
                f"(Vector registered your prompt: _“{question.strip()}”_. "
                "This console insight is rules-based and driven purely from current metrics.)"
            )

        for line in narrative_lines:
            st.write("- " + line)

        st.caption(
            "Note: This is a rules-based Vector OS narrative running **inside** the console — "
            "no external API calls. For full free-form AI chat, you still use ChatGPT directly, "
            "but this panel gives live, metrics-grounded commentary."
        )