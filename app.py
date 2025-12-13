# app.py — WAVES Intelligence™ Institutional Console (Vector OS Edition v17.1)
# Baseline engine results are unchanged.
# Adds: Diagnostics + Recommendations + What-If Lab (shadow simulation)
#
# IMPORTANT:
#   Plotly is optional. If Plotly is missing on Streamlit Cloud, charts fall back to Streamlit native charts.

from __future__ import annotations

import math
from datetime import datetime, timedelta
from typing import Dict, List, Any

import numpy as np
import pandas as pd
import streamlit as st

import waves_engine as we

# Optional Plotly
try:
    import plotly.graph_objects as go  # type: ignore
    PLOTLY_OK = True
except Exception:
    go = None
    PLOTLY_OK = False

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
def compute_wave_history_shadow(wave_name: str, mode: str, days: int, overrides: Dict[str, Any]) -> pd.DataFrame:
    try:
        df = we.simulate_history_nav(wave_name, mode=mode, days=days, overrides=overrides)
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


# ------------------------------------------------------------
# WaveScore proto v1.0 (kept simple)
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

    st.markdown("---")
    st.caption("Charts: Plotly ✅" if PLOTLY_OK else "Charts: Plotly not found → using Streamlit fallback ✅")


# ------------------------------------------------------------
# Page header + tabs
# ------------------------------------------------------------

st.title("WAVES Intelligence™ Institutional Console")
st.caption("Live Alpha Capture • SmartSafe™ • Multi-Asset • Crypto • Gold • Ladders")

tab_console, tab_market, tab_factors, tab_vector, tab_diag = st.tabs(
    ["Console", "Market Intel", "Factor Decomposition", "Vector OS Insight Layer", "Diagnostics & What-If Lab"]
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

        if PLOTLY_OK:
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
        else:
            st.line_chart(pd.DataFrame({"SPY_indexed": spy_norm, "VIX": vix}))

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

            if PLOTLY_OK:
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
            else:
                st.line_chart(pd.DataFrame({"Wave_NAV": nav_wave, "Benchmark_NAV": nav_bm}))

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


# ============================================================
# TAB 3: Factor Decomposition (simple correlation view)
# ============================================================

with tab_factors:
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
            if PLOTLY_OK:
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
# TAB 4: Vector OS Insight Layer (rules-based)
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
        st.write(f"- **Volatility (365D)**: Wave {vol_wave*100:0.2f}% vs benchmark {vol_bm*100:0.2f}%.")
        st.write(f"- **Max drawdown (365D)**: Wave {mdd_wave*100:0.2f}% vs benchmark {mdd_bm*100:0.2f}%.")

        if question.strip():
            st.caption(f"Vector registered your prompt: “{question.strip()}” (rules-based insight only).")


# ============================================================
# TAB 5: Diagnostics & What-If Lab (NEW)
# ============================================================

with tab_diag:
    st.subheader("Diagnostics & What-If Lab (Shadow Mode)")
    st.caption("Baseline results stay unchanged. What-If simulation runs separately and is clearly labeled.")

    # Baseline diagnostics snapshot from engine
    diag = we.get_latest_diagnostics(selected_wave, mode=mode, days=365)

    if not diag.get("ok", False):
        st.warning(diag.get("message", "Diagnostics unavailable."))
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Baseline 365D Return", f"{diag['return_wave']*100:0.2f}%")
        c2.metric("Baseline 365D Alpha", f"{diag['alpha']*100:0.2f}%")
        c3.metric("Tracking Error", f"{diag['tracking_error']*100:0.2f}%")
        c4.metric("IR (proxy)", f"{diag['information_ratio']:.2f}" if pd.notna(diag["information_ratio"]) else "—")

        st.markdown("### What might be going on")
        if diag.get("flags"):
            for f in diag["flags"]:
                st.write(f"- {f}")
        else:
            st.write("- No major red flags detected by the heuristic scanner.")

        st.markdown("### Recommended adjustments (suggestions)")
        if diag.get("suggestions"):
            for s in diag["suggestions"]:
                st.write(f"- {s}")
        else:
            st.write("- No specific recommendations right now.")

    st.markdown("---")
    st.markdown("## What-If Controls (Shadow Simulation)")

    defaults = we.get_parameter_defaults(selected_wave, mode)

    colA, colB, colC = st.columns(3)

    with colA:
        tilt_strength = st.slider(
            "Momentum tilt strength",
            min_value=0.0,
            max_value=1.2,
            value=float(defaults["tilt_strength"]),
            step=0.05,
        )
        vol_target = st.slider(
            "Vol target (annualized)",
            min_value=0.10,
            max_value=0.30,
            value=float(defaults["vol_target"]),
            step=0.01,
        )

    with colB:
        extra_safe_boost = st.slider(
            "Extra SmartSafe boost",
            min_value=0.00,
            max_value=0.30,
            value=float(defaults["extra_safe_boost"]),
            step=0.01,
        )
        base_exposure_mult = st.slider(
            "Base exposure multiplier",
            min_value=0.70,
            max_value=1.30,
            value=1.00,
            step=0.01,
        )

    with colC:
        exp_min = st.slider(
            "Exposure cap MIN",
            min_value=0.20,
            max_value=1.20,
            value=float(defaults["exp_min"]),
            step=0.05,
        )
        exp_max = st.slider(
            "Exposure cap MAX",
            min_value=0.50,
            max_value=2.00,
            value=float(defaults["exp_max"]),
            step=0.05,
        )

    freeze_benchmark = st.toggle("Freeze benchmark (use static benchmark only)", value=False)

    overrides = {
        "tilt_strength": float(tilt_strength),
        "vol_target": float(vol_target),
        "extra_safe_boost": float(extra_safe_boost),
        "base_exposure_mult": float(base_exposure_mult),
        "exp_min": float(exp_min),
        "exp_max": float(exp_max),
        "freeze_benchmark": bool(freeze_benchmark),
    }

    run_sim = st.button("Run What-If Simulation", type="primary")

    if run_sim:
        baseline = compute_wave_history(selected_wave, mode=mode, days=nav_days)
        shadow = compute_wave_history_shadow(selected_wave, mode=mode, days=nav_days, overrides=overrides)

        if baseline.empty or shadow.empty or len(baseline) < 2 or len(shadow) < 2:
            st.warning("Not enough data to compare baseline vs what-if.")
        else:
            nav_b = baseline["wave_nav"]
            nav_s = shadow["wave_nav"]
            bm_b = baseline["bm_nav"]  # baseline benchmark NAV

            # Returns & alpha
            ret_b = compute_return_from_nav(nav_b, window=len(nav_b))
            ret_s = compute_return_from_nav(nav_s, window=len(nav_s))

            bm_ret = compute_return_from_nav(bm_b, window=len(bm_b))
            alpha_b = ret_b - bm_ret
            alpha_s = ret_s - bm_ret  # compare to same baseline benchmark for clarity

            mdd_b = max_drawdown(nav_b)
            mdd_s = max_drawdown(nav_s)

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Baseline Return", f"{ret_b*100:0.2f}%")
            c2.metric("What-If Return", f"{ret_s*100:0.2f}%", delta=f"{(ret_s-ret_b)*100:0.2f}%")
            c3.metric("Baseline Alpha", f"{alpha_b*100:0.2f}%")
            c4.metric("What-If Alpha", f"{alpha_s*100:0.2f}%", delta=f"{(alpha_s-alpha_b)*100:0.2f}%")

            d1, d2 = st.columns(2)
            d1.metric("Baseline MaxDD", f"{mdd_b*100:0.2f}%")
            d2.metric("What-If MaxDD", f"{mdd_s*100:0.2f}%", delta=f"{(mdd_s-mdd_b)*100:0.2f}%")

            st.markdown("### Baseline vs What-If NAV")
            if PLOTLY_OK:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=baseline.index, y=nav_b, name="Baseline NAV", mode="lines"))
                fig.add_trace(go.Scatter(x=shadow.index, y=nav_s, name="What-If NAV", mode="lines"))
                fig.add_trace(go.Scatter(x=baseline.index, y=bm_b, name="Benchmark NAV (baseline)", mode="lines"))
                fig.update_layout(height=420, margin=dict(l=40, r=40, t=40, b=40))
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.line_chart(pd.DataFrame({"Baseline_NAV": nav_b, "WhatIf_NAV": nav_s, "Benchmark_NAV": bm_b}))

            # If engine provided shadow diagnostics series
            diag_df = shadow.attrs.get("diagnostics", None) if hasattr(shadow, "attrs") else None
            if isinstance(diag_df, pd.DataFrame) and not diag_df.empty:
                st.markdown("### What-If internal levers (last 60 days)")
                tail = diag_df.tail(min(60, len(diag_df))).copy()
                st.dataframe(tail, use_container_width=True)