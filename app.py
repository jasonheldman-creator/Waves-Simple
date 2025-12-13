# app.py — WAVES Intelligence™ Institutional Console (Vector OS Edition)
# Adds:
# - Conditional Attribution grid (VIX regime × SPY trend)
# - Persistent logging to /logs/diagnostics + /logs/recommendations
# - Safer auto-apply controls (preview-first, session-only, capped, revert)

from __future__ import annotations

import os
import math
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import waves_engine as we

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
# Logging helpers
# ------------------------------------------------------------

def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def log_event_csv(kind: str, row: dict) -> None:
    """
    Persistent CSV log writer. Appends to logs/<kind>/<kind>_YYYYMMDD.csv
    """
    try:
        base = os.path.join("logs", kind)
        _ensure_dir(base)
        fn = os.path.join(base, f"{kind}_{datetime.utcnow().strftime('%Y%m%d')}.csv")
        df = pd.DataFrame([row])
        if os.path.exists(fn):
            df.to_csv(fn, mode="a", header=False, index=False)
        else:
            df.to_csv(fn, index=False)
    except Exception:
        # never crash the app due to logging
        pass


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
def compute_wave_history(wave_name: str, mode: str, days: int = 365, overrides_key: str = "") -> pd.DataFrame:
    """
    Cached wrapper around engine compute_history_nav.
    overrides_key is used to bust cache when overrides change.
    """
    overrides = st.session_state.get("session_overrides", None)
    try:
        df = we.compute_history_nav(wave_name, mode=mode, days=days, overrides=overrides)
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
    overrides = st.session_state.get("session_overrides", None)
    try:
        return we.get_wave_holdings(wave_name, overrides=overrides)
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
# Regime + Conditional Attribution
# ------------------------------------------------------------

def vix_regime(v: float) -> str:
    if pd.isna(v):
        return "Unknown"
    if v < 16:
        return "Low"
    if v < 22:
        return "Medium"
    if v < 30:
        return "High"
    return "Stress"

def trend_regime(spy_prices: pd.Series, lookback: int = 60) -> pd.Series:
    """
    Trend label based on trailing lookback return sign.
    Returns Series of labels "Uptrend" / "Downtrend".
    """
    r = spy_prices.pct_change(lookback)
    return r.apply(lambda x: "Uptrend" if pd.notna(x) and x >= 0 else "Downtrend")

def conditional_attribution_grid(
    hist: pd.DataFrame,
    spy_vix: pd.DataFrame,
    lookback_trend: int = 60,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Builds a 4x2 grid:
      rows = VIX regime (Low/Medium/High/Stress)
      cols = Trend regime (Uptrend/Downtrend)
    Values:
      - mean daily alpha (bp)
      - count of days
    Returns (grid_mean_alpha_bp, grid_counts)
    """
    if hist.empty or spy_vix.empty:
        return (pd.DataFrame(), pd.DataFrame())
    if "SPY" not in spy_vix.columns:
        return (pd.DataFrame(), pd.DataFrame())

    df = pd.DataFrame(index=hist.index).copy()
    df["alpha"] = (hist["wave_ret"] - hist["bm_ret"]).astype(float)

    # align SPY/VIX
    sv = spy_vix.reindex(df.index).ffill().bfill()
    df["vix"] = sv.get("^VIX", np.nan)
    df["spy"] = sv["SPY"]

    df["VIX_Regime"] = df["vix"].apply(vix_regime)
    df["Trend_Regime"] = trend_regime(df["spy"], lookback=lookback_trend)

    # grid
    regimes = ["Low", "Medium", "High", "Stress"]
    trends = ["Uptrend", "Downtrend"]

    mean_bp = pd.DataFrame(index=regimes, columns=trends, dtype=float)
    counts = pd.DataFrame(index=regimes, columns=trends, dtype=float)

    for r in regimes:
        for t in trends:
            sub = df[(df["VIX_Regime"] == r) & (df["Trend_Regime"] == t)]["alpha"].dropna()
            counts.loc[r, t] = float(len(sub))
            if len(sub) > 0:
                mean_bp.loc[r, t] = float(sub.mean() * 10000.0)  # bp/day
            else:
                mean_bp.loc[r, t] = np.nan

    return mean_bp, counts


# ------------------------------------------------------------
# Diagnostics + Recommendations (safe, preview-first)
# ------------------------------------------------------------

def concentration_diagnostics(holdings: pd.DataFrame) -> List[dict]:
    issues = []
    if holdings is None or holdings.empty or "Weight" not in holdings.columns:
        return issues

    w = holdings["Weight"].astype(float).fillna(0.0).values
    w_sorted = np.sort(w)[::-1]
    top1 = float(w_sorted[0]) if len(w_sorted) >= 1 else 0.0
    top3 = float(w_sorted[:3].sum()) if len(w_sorted) >= 3 else float(w_sorted.sum())

    if top1 >= 0.50:
        issues.append({
            "level": "WARN",
            "title": "High single-name concentration",
            "detail": f"Top holding is {top1*100:.1f}% of the Wave."
        })
    if top3 >= 0.80:
        issues.append({
            "level": "WARN",
            "title": "High top-3 concentration",
            "detail": f"Top-3 holdings sum to {top3*100:.1f}%."
        })
    return issues

def build_recommendations(
    mean_bp: pd.DataFrame,
    counts: pd.DataFrame,
    mode: str,
) -> List[dict]:
    """
    Produce conservative, rules-based recommendations.
    Returns list of dicts with: title, rationale, suggested_overrides (session-only)
    """
    recs = []
    if mean_bp.empty or counts.empty:
        return recs

    # Example logic:
    # If Stress+Downtrend alpha is strongly negative, increase SmartSafe delta.
    stress_down = mean_bp.loc["Stress", "Downtrend"] if "Stress" in mean_bp.index else np.nan
    n_stress_down = counts.loc["Stress", "Downtrend"] if "Stress" in counts.index else 0

    if pd.notna(stress_down) and n_stress_down >= 15 and stress_down < -5:  # < -5 bp/day
        recs.append({
            "title": "Increase SmartSafe in Stress/Downtrend",
            "rationale": f"Stress+Downtrend mean daily alpha is {stress_down:.1f} bp/day over {int(n_stress_down)} days.",
            "suggested_overrides": {"smartsafe_delta": +0.10, "exposure_delta": 0.0},
            "confidence": "High",
        })

    # If Low+Uptrend alpha is strongly positive, modestly increase exposure (unless AMB).
    low_up = mean_bp.loc["Low", "Uptrend"] if "Low" in mean_bp.index else np.nan
    n_low_up = counts.loc["Low", "Uptrend"] if "Low" in counts.index else 0

    if pd.notna(low_up) and n_low_up >= 30 and low_up > 5:
        if mode != "Alpha-Minus-Beta":
            recs.append({
                "title": "Slightly increase exposure in benign regimes",
                "rationale": f"Low+Uptrend mean daily alpha is {low_up:.1f} bp/day over {int(n_low_up)} days.",
                "suggested_overrides": {"smartsafe_delta": 0.0, "exposure_delta": +0.05},
                "confidence": "Medium",
            })

    return recs


# ------------------------------------------------------------
# Sidebar init
# ------------------------------------------------------------

if "session_overrides" not in st.session_state:
    st.session_state["session_overrides"] = {"smartsafe_delta": 0.0, "exposure_delta": 0.0}
if "apply_unlocked" not in st.session_state:
    st.session_state["apply_unlocked"] = False
if "override_history" not in st.session_state:
    st.session_state["override_history"] = []

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
    st.markdown("**Session Overrides (Safe / Preview-First)**")
    st.caption("These overrides affect calculations in this browser session only.")

    col_a, col_b = st.columns(2)
    with col_a:
        ss_delta = st.number_input(
            "SmartSafe Δ",
            value=float(st.session_state["session_overrides"].get("smartsafe_delta", 0.0)),
            step=0.01,
            format="%.2f",
            help="Adds to SmartSafe risk-off fraction by VIX regime. Clamped in engine.",
        )
    with col_b:
        ex_delta = st.number_input(
            "Exposure Δ",
            value=float(st.session_state["session_overrides"].get("exposure_delta", 0.0)),
            step=0.01,
            format="%.2f",
            help="Adds to mode exposure multiplier. Clamped in engine.",
        )

    # Update session overrides
    st.session_state["session_overrides"] = {"smartsafe_delta": float(ss_delta), "exposure_delta": float(ex_delta)}
    overrides_key = f"{ss_delta:.2f}_{ex_delta:.2f}"

    st.markdown("---")
    st.markdown("**Apply Safety**")
    st.session_state["apply_unlocked"] = st.toggle(
        "Unlock Apply (session-only)",
        value=bool(st.session_state["apply_unlocked"]),
        help="Must be enabled to apply any recommendation. Still session-only.",
    )

    if st.button("Revert overrides to 0.00 / 0.00"):
        st.session_state["override_history"].append(st.session_state["session_overrides"])
        st.session_state["session_overrides"] = {"smartsafe_delta": 0.0, "exposure_delta": 0.0}
        st.rerun()


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
        hist_365 = compute_wave_history(wave, mode=mode, days=365, overrides_key=overrides_key)
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

    overview_df = pd.DataFrame(overview_rows)
    fmt_overview = overview_df.copy()
    for col in ["365D Return", "365D Alpha", "30D Return", "30D Alpha"]:
        fmt_overview[col] = fmt_overview[col].apply(lambda x: f"{x*100:0.2f}%" if pd.notna(x) else "—")
    st.dataframe(fmt_overview.set_index("Wave"), use_container_width=True)

    st.markdown("---")

    st.subheader(f"Multi-Window Alpha Capture (All Waves · Mode = {mode})")
    alpha_rows = []
    for wave in all_waves:
        hist_365 = compute_wave_history(wave, mode=mode, days=365, overrides_key=overrides_key)
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

    alpha_df = pd.DataFrame(alpha_rows)
    fmt_alpha = alpha_df.copy()
    for col in ["1D Ret", "1D Alpha", "30D Ret", "30D Alpha", "60D Ret", "60D Alpha", "365D Ret", "365D Alpha"]:
        fmt_alpha[col] = fmt_alpha[col].apply(lambda x: f"{x*100:0.2f}%" if pd.notna(x) else "—")
    st.dataframe(fmt_alpha.set_index("Wave"), use_container_width=True)

    st.markdown("---")

    st.subheader(f"Wave Detail — {selected_wave} (Mode: {mode})")
    col_chart, col_stats = st.columns([2.0, 1.0])

    hist = compute_wave_history(selected_wave, mode=mode, days=nav_days, overrides_key=overrides_key)

    with col_chart:
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
            st.metric("30D Alpha", f"{alpha_30*100:0.2f}%" if not math.isnan(alpha_30) else "—")
            st.metric("365D Return", f"{ret_365_wave*100:0.2f}%" if not math.isnan(ret_365_wave) else "—")
            st.metric("365D Alpha", f"{alpha_365*100:0.2f}%" if not math.isnan(alpha_365) else "—")

    # -------------------------
    # NEW: Conditional Attribution + Diagnostics + Safe Auto-Apply
    # -------------------------

    st.markdown("---")
    st.subheader("Conditional Attribution (365D) — VIX Regime × Trend Regime")

    spy_vix_365 = fetch_spy_vix(days=365)

    mean_bp, counts = conditional_attribution_grid(hist=compute_wave_history(selected_wave, mode=mode, days=365, overrides_key=overrides_key),
                                                   spy_vix=spy_vix_365,
                                                   lookback_trend=60)

    if mean_bp.empty:
        st.info("Not enough data to compute conditional attribution grid yet.")
    else:
        # Show tables
        col_g1, col_g2 = st.columns([1.2, 1.0])
        with col_g1:
            fmt_bp = mean_bp.copy()
            for c in fmt_bp.columns:
                fmt_bp[c] = fmt_bp[c].apply(lambda x: f"{x:,.1f}" if pd.notna(x) else "—")
            st.markdown("**Mean daily alpha (bp/day)**")
            st.dataframe(fmt_bp, use_container_width=True)

        with col_g2:
            fmt_ct = counts.copy()
            for c in fmt_ct.columns:
                fmt_ct[c] = fmt_ct[c].apply(lambda x: f"{int(x)}" if pd.notna(x) else "0")
            st.markdown("**Day counts**")
            st.dataframe(fmt_ct, use_container_width=True)

        # Heatmap for mean bp/day
        fig_h = go.Figure(
            data=go.Heatmap(
                z=mean_bp.values,
                x=mean_bp.columns.tolist(),
                y=mean_bp.index.tolist(),
                colorbar=dict(title="bp/day"),
            )
        )
        fig_h.update_layout(
            title="Conditional Alpha Heatmap (bp/day)",
            height=420,
            margin=dict(l=50, r=50, t=60, b=50),
        )
        st.plotly_chart(fig_h, use_container_width=True)

        # Persistent log (once per render)
        log_event_csv("attribution", {
            "ts_utc": datetime.utcnow().isoformat(),
            "wave": selected_wave,
            "mode": mode,
            "smartsafe_delta": st.session_state["session_overrides"]["smartsafe_delta"],
            "exposure_delta": st.session_state["session_overrides"]["exposure_delta"],
            "low_up_bp": mean_bp.loc["Low", "Uptrend"],
            "low_down_bp": mean_bp.loc["Low", "Downtrend"],
            "med_up_bp": mean_bp.loc["Medium", "Uptrend"],
            "med_down_bp": mean_bp.loc["Medium", "Downtrend"],
            "high_up_bp": mean_bp.loc["High", "Uptrend"],
            "high_down_bp": mean_bp.loc["High", "Downtrend"],
            "stress_up_bp": mean_bp.loc["Stress", "Uptrend"],
            "stress_down_bp": mean_bp.loc["Stress", "Downtrend"],
        })

    st.markdown("---")
    st.subheader("Diagnostics")

    holdings_df = get_wave_holdings(selected_wave)
    issues = concentration_diagnostics(holdings_df)

    if not issues:
        st.success("PASS — No issues detected.")
        log_event_csv("diagnostics", {
            "ts_utc": datetime.utcnow().isoformat(),
            "wave": selected_wave,
            "mode": mode,
            "status": "PASS",
            "detail": "No issues detected",
        })
    else:
        for it in issues:
            st.warning(f"{it['level']} — {it['title']}\n\n{it['detail']}")
            log_event_csv("diagnostics", {
                "ts_utc": datetime.utcnow().isoformat(),
                "wave": selected_wave,
                "mode": mode,
                "status": it["level"],
                "detail": f"{it['title']} | {it['detail']}",
            })

    st.markdown("---")
    st.subheader("Auto Recommendations (Preview-First, Session-Only Apply)")

    recs = build_recommendations(mean_bp, counts, mode=mode) if not mean_bp.empty else []

    if not recs:
        st.info("No high-confidence recommendations detected (or not enough history).")
    else:
        for i, rec in enumerate(recs, start=1):
            with st.expander(f"{i}) {rec['title']}  •  Confidence: {rec.get('confidence','—')}"):
                st.write(rec["rationale"])

                suggested = rec.get("suggested_overrides", {})
                st.code(suggested, language="json")

                # Clamp suggestion locally (still clamped again in engine)
                ss_new = float(np.clip(st.session_state["session_overrides"]["smartsafe_delta"] + float(suggested.get("smartsafe_delta", 0.0)), -0.25, 0.25))
                ex_new = float(np.clip(st.session_state["session_overrides"]["exposure_delta"] + float(suggested.get("exposure_delta", 0.0)), -0.25, 0.25))

                st.write(f"**Preview result if applied (session-only):**")
                st.write(f"- SmartSafe Δ: {st.session_state['session_overrides']['smartsafe_delta']:+.2f} → {ss_new:+.2f}")
                st.write(f"- Exposure Δ: {st.session_state['session_overrides']['exposure_delta']:+.2f} → {ex_new:+.2f}")

                c1, c2 = st.columns([1.0, 1.0])
                with c1:
                    if st.button(f"Apply (session-only) — {rec['title']}", key=f"apply_{i}"):
                        if not st.session_state["apply_unlocked"]:
                            st.error("Apply is locked. Enable 'Unlock Apply' in the sidebar first.")
                        else:
                            st.session_state["override_history"].append(st.session_state["session_overrides"])
                            st.session_state["session_overrides"] = {"smartsafe_delta": ss_new, "exposure_delta": ex_new}

                            log_event_csv("recommendations", {
                                "ts_utc": datetime.utcnow().isoformat(),
                                "wave": selected_wave,
                                "mode": mode,
                                "action": "APPLY_SESSION_ONLY",
                                "title": rec["title"],
                                "from_smartsafe_delta": float(st.session_state["override_history"][-1]["smartsafe_delta"]),
                                "to_smartsafe_delta": ss_new,
                                "from_exposure_delta": float(st.session_state["override_history"][-1]["exposure_delta"]),
                                "to_exposure_delta": ex_new,
                                "confidence": rec.get("confidence", ""),
                                "rationale": rec.get("rationale", ""),
                            })
                            st.rerun()

                with c2:
                    if st.button("Undo last apply (revert)", key=f"undo_{i}"):
                        if st.session_state["override_history"]:
                            prev = st.session_state["override_history"].pop()
                            st.session_state["session_overrides"] = prev
                            log_event_csv("recommendations", {
                                "ts_utc": datetime.utcnow().isoformat(),
                                "wave": selected_wave,
                                "mode": mode,
                                "action": "REVERT",
                                "title": "Revert to previous overrides",
                                "to_smartsafe_delta": float(prev["smartsafe_delta"]),
                                "to_exposure_delta": float(prev["exposure_delta"]),
                            })
                            st.rerun()

    st.markdown("---")
    st.subheader("Top-10 Holdings")
    if holdings_df.empty:
        st.info("No holdings available for this Wave.")
    else:
        def google_link(ticker: str) -> str:
            base = "https://www.google.com/finance/quote"
            return f"[{ticker}]({base}/{ticker})"

        fmt_hold = holdings_df.copy()
        fmt_hold["Weight"] = fmt_hold["Weight"].apply(lambda x: f"{x*100:0.2f}%" if pd.notna(x) else "—")
        fmt_hold["Google Finance"] = fmt_hold["Ticker"].astype(str).apply(google_link)
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
        fmt_snap["Last"] = fmt_snap["Last"].apply(lambda x: f"{x:0.2f}" if pd.notna(x) else "—")
        fmt_snap["1D Return"] = fmt_snap["1D Return"].apply(lambda x: f"{x*100:0.2f}%" if pd.notna(x) else "—")
        fmt_snap["30D Return"] = fmt_snap["30D Return"].apply(lambda x: f"{x*100:0.2f}%" if pd.notna(x) else "—")
        st.dataframe(fmt_snap.set_index("Ticker"), use_container_width=True)

# ============================================================
# TAB 3: Factor Decomposition (kept minimal here)
# ============================================================

with tab_factors:
    st.subheader("Factor Decomposition (Institution-Level Analytics)")
    st.caption("Use your existing factor decomposition build here. (This file keeps it minimal for stability.)")
    st.info("Factor tab intentionally kept light in this round to focus on Conditional Attribution + Safe Apply.")

# ============================================================
# TAB 4: Vector OS Insight Layer (kept minimal here)
# ============================================================

with tab_vector:
    st.subheader("Vector OS Insight Layer")
    st.caption("Use your existing narrative panel here. This round focused on Conditional Attribution + Safe Apply.")
    st.info("Vector OS tab intentionally kept light in this round to focus on the new attribution + recommendations stack.")