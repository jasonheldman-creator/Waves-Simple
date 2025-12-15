# app.py — WAVES Intelligence™ Institutional Console (Vector OS Edition)
# FULL PRODUCTION FILE (NO PATCHES) — ANALYTICS EXPANDED + GOVERNANCE ADDITIONS
#
# Keeps ALL IRB-1 features:
#   • Benchmark Truth (benchmark mix + difficulty vs SPY)
#   • Mode Separation Proof (mode shown + independent history per mode)
#   • Alpha Attribution (Engine vs Static Basket)
#   • Wave Doctor + What-If Lab (shadow simulation)
#   • Top-10 holdings with Google quote links
#   • Factor Decomposition (simple regression beta map)
#   • Vector OS Insight Layer (narrative + flags)
#
# Adds (Analytics Expansion — console-side only; engine math unchanged):
#   • Risk Lab Tab:
#       - Sharpe, Sortino, Downside Dev, Skew, Kurtosis
#       - VaR(95)/CVaR(95) (daily)
#       - Rolling 30D Alpha / Rolling Vol charts
#       - Drawdown chart + drawdown table
#       - Alpha Persistence (pct of rolling windows with positive alpha)
#   • Correlation Tab:
#       - Correlation matrix across Waves (daily returns; selected mode)
#       - “Most Correlated / Least Correlated” list for selected wave
#       - Optional correlation vs key ETFs (SPY/QQQ/IWM/TLT/GLD/BTC)
#   • WaveScore Leaderboard view (sortable) with highlights
#
# Governance Additions (NEW — console-only, engine math unchanged):
#   1) Benchmark Stability Snapshot:
#       - Benchmark Snapshot ID (hash of benchmark mix for reproducibility)
#       - Benchmark Drift alert if snapshot changes during session
#   2) Coverage & Data Integrity Panel:
#       - history length, last data date, data age (days), missing business days
#       - completeness score + flags
#
# Removed:
#   • Market Intel tab (to reduce surface area / line count risk)
#
# Notes:
#   • Does NOT modify engine math or baseline results.
#   • What-If Lab is explicitly labeled “shadow simulation”.
#   • Streamlit-safe patterns; selection events are best-effort.
#   • Plotly optional; falls back to Streamlit charts where possible.

from __future__ import annotations

import math
import hashlib
from datetime import datetime, timedelta, date
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

import waves_engine as we  # your engine module

try:
    import yfinance as yf
except Exception:
    yf = None

# Plotly optional (avoid crashes if missing)
try:
    import plotly.graph_objects as go
except Exception:
    go = None


# ============================================================
# Streamlit config
# ============================================================
st.set_page_config(
    page_title="WAVES Intelligence™ Console",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# Global UI CSS: sticky bar + scan improvements
# ============================================================
st.markdown(
    """
<style>
.block-container { padding-top: 1rem; padding-bottom: 2.0rem; }

/* Sticky summary container */
.waves-sticky {
  position: sticky;
  top: 0;
  z-index: 999;
  backdrop-filter: blur(10px);
  -webkit-backdrop-filter: blur(10px);
  padding: 10px 12px 10px 12px;
  margin: 0 0 12px 0;
  border-radius: 14px;
  border: 1px solid rgba(255,255,255,0.10);
  background: rgba(10, 15, 28, 0.66);
}

/* Summary chips */
.waves-chip {
  display: inline-block;
  padding: 6px 10px;
  margin: 6px 8px 0 0;
  border-radius: 999px;
  border: 1px solid rgba(255,255,255,0.12);
  background: rgba(255,255,255,0.04);
  font-size: 0.85rem;
  line-height: 1.0rem;
  white-space: nowrap;
}

/* Section header */
.waves-hdr { font-weight: 800; letter-spacing: 0.2px; margin-bottom: 4px; }

/* Tighter tables */
div[data-testid="stDataFrame"] { border-radius: 12px; overflow: hidden; }

/* Reduce huge whitespace for mobile */
@media (max-width: 700px) {
  .block-container { padding-left: 0.8rem; padding-right: 0.8rem; }
}
</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# Helpers: formatting
# ============================================================
def fmt_pct(x: Any, digits: int = 2) -> str:
    try:
        if x is None:
            return "—"
        x = float(x)
        if math.isnan(x):
            return "—"
        return f"{x*100:0.{digits}f}%"
    except Exception:
        return "—"


def fmt_num(x: Any, digits: int = 2) -> str:
    try:
        if x is None:
            return "—"
        x = float(x)
        if math.isnan(x):
            return "—"
        return f"{x:.{digits}f}"
    except Exception:
        return "—"


def fmt_score(x: Any) -> str:
    try:
        if x is None:
            return "—"
        x = float(x)
        if math.isnan(x):
            return "—"
        return f"{x:.1f}"
    except Exception:
        return "—"


def safe_series(s: Optional[pd.Series]) -> pd.Series:
    if s is None or len(s) == 0:
        return pd.Series(dtype=float)
    return s.copy()


def safe_df(df: Optional[pd.DataFrame]) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    return df.copy()
    # ============================================================
# Helpers: data fetching & caching
# ============================================================
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
def fetch_prices_daily(tickers: List[str], days: int = 365) -> pd.DataFrame:
    """Generic daily price fetch used for attribution + what-if shadow sim."""
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
        # ============================================================
# Core metrics
# ============================================================
def ret_from_nav(nav: pd.Series, window: int) -> float:
    nav = safe_series(nav)
    if len(nav) < 2:
        return float("nan")
    if len(nav) < window:
        window = len(nav)
    if window < 2:
        return float("nan")
    sub = nav.iloc[-window:]
    start = float(sub.iloc[0])
    end = float(sub.iloc[-1])
    if start <= 0:
        return float("nan")
    return (end / start) - 1.0


def annualized_vol(daily_ret: pd.Series) -> float:
    daily_ret = safe_series(daily_ret)
    if len(daily_ret) < 2:
        return float("nan")
    return float(daily_ret.std() * np.sqrt(252))


def max_drawdown(nav: pd.Series) -> float:
    nav = safe_series(nav)
    if len(nav) < 2:
        return float("nan")
    running_max = nav.cummax()
    dd = (nav / running_max) - 1.0
    return float(dd.min())


def tracking_error(daily_wave: pd.Series, daily_bm: pd.Series) -> float:
    daily_wave = safe_series(daily_wave)
    daily_bm = safe_series(daily_bm)
    if len(daily_wave) < 2 or len(daily_bm) < 2:
        return float("nan")
    df = pd.concat([daily_wave.rename("w"), daily_bm.rename("b")], axis=1).dropna()
    if df.shape[0] < 2:
        return float("nan")
    diff = df["w"] - df["b"]
    return float(diff.std() * np.sqrt(252))


def information_ratio(nav_wave: pd.Series, nav_bm: pd.Series, te: float) -> float:
    nav_wave = safe_series(nav_wave)
    nav_bm = safe_series(nav_bm)
    if len(nav_wave) < 2 or len(nav_bm) < 2:
        return float("nan")
    if te is None or (isinstance(te, float) and (math.isnan(te) or te <= 0)):
        return float("nan")
    ret_wave = ret_from_nav(nav_wave, window=len(nav_wave))
    ret_bm = ret_from_nav(nav_bm, window=len(nav_bm))
    excess = ret_wave - ret_bm
    return float(excess / te)


def beta_ols(y: pd.Series, x: pd.Series) -> float:
    """OLS beta of y vs x (daily returns)."""
    y = safe_series(y).astype(float)
    x = safe_series(x).astype(float)
    df = pd.concat([y.rename("y"), x.rename("x")], axis=1).dropna()
    if df.shape[0] < 20:
        return float("nan")
    vx = float(df["x"].var())
    if not math.isfinite(vx) or vx <= 0:
        return float("nan")
    cov = float(df["y"].cov(df["x"]))
    return float(cov / vx)


def sharpe_ratio(daily_ret: pd.Series, rf_annual: float = 0.0) -> float:
    """Sharpe using daily returns; rf_annual optional (default 0)."""
    r = safe_series(daily_ret).astype(float)
    if len(r) < 20:
        return float("nan")
    rf_daily = rf_annual / 252.0
    ex = r - rf_daily
    vol = float(ex.std())
    if not math.isfinite(vol) or vol <= 0:
        return float("nan")
    return float(ex.mean() / vol * np.sqrt(252))


def downside_deviation(daily_ret: pd.Series, mar_annual: float = 0.0) -> float:
    """Downside deviation vs MAR (annual) using daily returns."""
    r = safe_series(daily_ret).astype(float)
    if len(r) < 20:
        return float("nan")
    mar_daily = mar_annual / 252.0
    d = np.minimum(0.0, (r - mar_daily).values)
    dd = float(np.sqrt(np.mean(d**2)))
    return float(dd * np.sqrt(252))


def sortino_ratio(daily_ret: pd.Series, mar_annual: float = 0.0) -> float:
    r = safe_series(daily_ret).astype(float)
    if len(r) < 20:
        return float("nan")
    mar_daily = mar_annual / 252.0
    ex = float((r - mar_daily).mean()) * 252.0
    dd = downside_deviation(r, mar_annual=mar_annual)
    if not math.isfinite(dd) or dd <= 0:
        return float("nan")
    return float(ex / dd)


def var_cvar(daily_ret: pd.Series, level: float = 0.95) -> Tuple[float, float]:
    """Historical VaR/CVaR (returns; negative means loss)."""
    r = safe_series(daily_ret).astype(float)
    if len(r) < 50:
        return (float("nan"), float("nan"))
    q = float(np.quantile(r.values, 1.0 - level))
    tail = r[r <= q]
    cvar = float(tail.mean()) if len(tail) > 0 else float("nan")
    return (q, cvar)


def skew_kurt(daily_ret: pd.Series) -> Tuple[float, float]:
    r = safe_series(daily_ret).astype(float)
    if len(r) < 50:
        return (float("nan"), float("nan"))
    sk = float(pd.Series(r).skew())
    ku = float(pd.Series(r).kurtosis())
    return (sk, ku)


def regress_factors(wave_ret: pd.Series, factor_ret: pd.DataFrame) -> Dict[str, float]:
    wave_ret = safe_series(wave_ret)
    if wave_ret.empty or factor_ret is None or factor_ret.empty:
        return {col: float("nan") for col in factor_ret.columns} if factor_ret is not None else {}

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
    # ============================================================
# Rolling analytics helpers
# ============================================================
def rolling_return_from_nav(nav: pd.Series, window: int) -> pd.Series:
    nav = safe_series(nav).astype(float)
    if len(nav) < window + 1:
        return pd.Series(dtype=float)
    return (nav / nav.shift(window) - 1.0).rename(f"ret_{window}")


def rolling_alpha_from_nav(wave_nav: pd.Series, bm_nav: pd.Series, window: int) -> pd.Series:
    w = rolling_return_from_nav(wave_nav, window)
    b = rolling_return_from_nav(bm_nav, window)
    df = pd.concat([w, b], axis=1).dropna()
    if df.empty:
        return pd.Series(dtype=float)
    out = (df.iloc[:, 0] - df.iloc[:, 1]).rename(f"alpha_{window}")
    return out


def rolling_vol(daily_ret: pd.Series, window: int = 20) -> pd.Series:
    r = safe_series(daily_ret).astype(float)
    if len(r) < window + 5:
        return pd.Series(dtype=float)
    return (r.rolling(window).std() * np.sqrt(252)).rename(f"vol_{window}")


def drawdown_series(nav: pd.Series) -> pd.Series:
    nav = safe_series(nav).astype(float)
    if len(nav) < 2:
        return pd.Series(dtype=float)
    peak = nav.cummax()
    dd = (nav / peak) - 1.0
    return dd.rename("drawdown")


def alpha_persistence(alpha_series: pd.Series) -> float:
    a = safe_series(alpha_series).dropna()
    if len(a) < 30:
        return float("nan")
    return float((a > 0).mean())


# ============================================================
# Benchmark truth helpers
# ============================================================
@st.cache_data(show_spinner=False)
def compute_spy_nav(days: int = 365) -> pd.Series:
    px = fetch_prices_daily(["SPY"], days=days)
    if px.empty or "SPY" not in px.columns:
        return pd.Series(dtype=float)
    nav = (1.0 + px["SPY"].pct_change().fillna(0.0)).cumprod()
    nav.name = "spy_nav"
    return nav


def benchmark_source_label(wave_name: str, bm_mix_df: pd.DataFrame) -> str:
    try:
        static_dict = getattr(we, "BENCHMARK_WEIGHTS_STATIC", None)
        if isinstance(static_dict, dict) and wave_name in static_dict and static_dict.get(wave_name):
            return "Static Override (Engine)"
    except Exception:
        pass
    if bm_mix_df is not None and not bm_mix_df.empty:
        return "Auto-Composite (Dynamic)"
    return "Unknown"


def _normalize_bm_rows(bm_rows: pd.DataFrame) -> pd.DataFrame:
    """Normalize benchmark rows for stable hashing."""
    if bm_rows is None or bm_rows.empty:
        return pd.DataFrame(columns=["Ticker", "Weight"])
    df = bm_rows.copy()
    if "Ticker" not in df.columns or "Weight" not in df.columns:
        return pd.DataFrame(columns=["Ticker", "Weight"])
    df["Ticker"] = df["Ticker"].astype(str).str.upper().str.strip()
    df["Weight"] = pd.to_numeric(df["Weight"], errors="coerce").fillna(0.0)
    df = df.groupby("Ticker", as_index=False)["Weight"].sum()
    tot = float(df["Weight"].sum())
    if tot > 0:
        df["Weight"] = df["Weight"] / tot
    df["Weight"] = df["Weight"].round(8)
    df = df.sort_values(["Ticker"]).reset_index(drop=True)
    return df[["Ticker", "Weight"]]


def benchmark_snapshot_id(wave_name: str, bm_mix_df: pd.DataFrame) -> str:
    """Stable snapshot hash of the benchmark mix for this wave (reproducibility)."""
    try:
        if bm_mix_df is None or bm_mix_df.empty:
            return "BM-NA"
        if "Wave" in bm_mix_df.columns:
            rows = bm_mix_df[bm_mix_df["Wave"] == wave_name].copy()
        else:
            rows = bm_mix_df.copy()
        rows = _normalize_bm_rows(rows)
        if rows.empty:
            return "BM-NA"
        payload = "|".join([f"{r.Ticker}:{r.Weight:.8f}" for r in rows.itertuples(index=False)])
        h = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:10].upper()
        return f"BM-{h}"
    except Exception:
        return "BM-ERR"


def benchmark_drift_status(wave_name: str, mode: str, snapshot_id: str) -> str:
    """Tracks drift within the user session (does not write to disk)."""
    key = f"bm_snapshot::{mode}::{wave_name}"
    prior = st.session_state.get(key)
    if prior is None:
        st.session_state[key] = snapshot_id
        return "stable"
    if str(prior) == str(snapshot_id):
        return "stable"
    st.session_state[key] = snapshot_id
    return "drift"


def _business_day_range(start_dt: pd.Timestamp, end_dt: pd.Timestamp) -> pd.DatetimeIndex:
    try:
        return pd.date_range(start=start_dt.normalize(), end=end_dt.normalize(), freq="B")
    except Exception:
        return pd.DatetimeIndex([])


def coverage_report(hist: pd.DataFrame) -> Dict[str, Any]:
    """Coverage/data-integrity report from history dataframe index."""
    out: Dict[str, Any] = {
        "rows": 0,
        "first_date": None,
        "last_date": None,
        "age_days": None,
        "missing_bdays": None,
        "missing_pct": None,
        "completeness_score": None,
        "flags": [],
    }
    try:
        if hist is None or hist.empty:
            out["flags"].append("No history returned")
            return out

        idx = pd.to_datetime(hist.index)
        idx = idx.sort_values()
        out["rows"] = int(len(idx))
        out["first_date"] = idx[0].date().isoformat() if len(idx) else None
        out["last_date"] = idx[-1].date().isoformat() if len(idx) else None

        today = datetime.utcnow().date()
        last_dt = idx[-1].date()
        out["age_days"] = int((today - last_dt).days)

        # business-day gap check
        bdays = _business_day_range(idx[0], idx[-1])
        have = pd.DatetimeIndex(idx.normalize().unique())
        missing = bdays.difference(have)
        out["missing_bdays"] = int(len(missing))
        out["missing_pct"] = float(len(missing) / max(1, len(bdays)))

        # crude completeness score
        score = 100.0
        score -= min(40.0, out["missing_pct"] * 200.0)  # up to -40
        if out["age_days"] is not None and out["age_days"] > 3:
            score -= min(25.0, float(out["age_days"] - 3) * 5.0)  # up to -25 quickly
        score = float(np.clip(score, 0.0, 100.0))
        out["completeness_score"] = score

        # flags
        if out["age_days"] is not None and out["age_days"] >= 7:
            out["flags"].append("Data is stale (>=7 days old)")
        if out["missing_pct"] is not None and out["missing_pct"] >= 0.05:
            out["flags"].append("Significant missing business days (>=5%)")
        if out["rows"] < 60:
            out["flags"].append("Limited history (<60 points)")

        return out
    except Exception:
        out["flags"].append("Coverage report error")
        return out


# ============================================================
# Alpha Attribution (Engine vs Static Basket)
# ============================================================
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
    out: Dict[str, float] = {}

    hist = compute_wave_history(wave_name, mode=mode, days=days)
    if hist.empty or len(hist) < 2:
        return out

    nav_wave = hist["wave_nav"]
    nav_bm = hist["bm_nav"]
    wave_ret = hist["wave_ret"]
    bm_ret = hist["bm_ret"]

    eng_ret = ret_from_nav(nav_wave, window=len(nav_wave))
    bm_ret_total = ret_from_nav(nav_bm, window=len(nav_bm))
    alpha_vs_bm = eng_ret - bm_ret_total

    hold_df = get_wave_holdings(wave_name)
    hold_w = _weights_from_df(hold_df, ticker_col="Ticker", weight_col="Weight")
    static_nav = compute_static_nav_from_weights(hold_w, days=days)
    static_ret = ret_from_nav(static_nav, window=len(static_nav)) if len(static_nav) >= 2 else float("nan")
    overlay_pp = (eng_ret - static_ret) if (pd.notna(eng_ret) and pd.notna(static_ret)) else float("nan")

    spy_px = fetch_prices_daily(["SPY"], days=days)
    spy_nav = (spy_px["SPY"].pct_change().fillna(0.0) + 1.0).cumprod() if "SPY" in spy_px.columns else pd.Series(dtype=float)
    spy_ret = ret_from_nav(spy_nav, window=len(spy_nav)) if len(spy_nav) >= 2 else float("nan")

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

    out["Wave Vol"] = float(annualized_vol(wave_ret))
    out["Benchmark Vol"] = float(annualized_vol(bm_ret))
    out["Wave MaxDD"] = float(max_drawdown(nav_wave))
    out["Benchmark MaxDD"] = float(max_drawdown(nav_bm))

    # extra risk stats (console-side only)
    out["Beta vs Benchmark"] = float(beta_ols(wave_ret, bm_ret))
    out["Sharpe (0% rf)"] = float(sharpe_ratio(wave_ret, rf_annual=0.0))
    out["Sortino (0% MAR)"] = float(sortino_ratio(wave_ret, mar_annual=0.0))
    v, c = var_cvar(wave_ret, level=0.95)
    out["VaR 95% (daily)"] = float(v)
    out["CVaR 95% (daily)"] = float(c)

    return out
    # ============================================================
# WaveScore proto (console-side display helper)
# ============================================================
def _grade_from_score(score: float) -> str:
    if score is None or (isinstance(score, float) and math.isnan(score)):
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


def compute_wavescore_for_all_waves(all_waves: List[str], mode: str, days: int = 365) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
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
                    "IR_365D": float("nan"),
                    "Alpha_365D": float("nan"),
                }
            )
            continue

        nav_wave = hist["wave_nav"]
        nav_bm = hist["bm_nav"]
        wave_ret = hist["wave_ret"]
        bm_ret = hist["bm_ret"]

        ret_365_wave = ret_from_nav(nav_wave, window=len(nav_wave))
        ret_365_bm = ret_from_nav(nav_bm, window=len(nav_bm))
        alpha_365 = ret_365_wave - ret_365_bm

        vol_wave = annualized_vol(wave_ret)
        vol_bm = annualized_vol(bm_ret)

        te = tracking_error(wave_ret, bm_ret)
        ir = information_ratio(nav_wave, nav_bm, te)

        mdd_wave = max_drawdown(nav_wave)
        mdd_bm = max_drawdown(nav_bm)

        hit_rate = float((wave_ret >= bm_ret).mean()) if len(wave_ret) > 0 else float("nan")

        if len(nav_wave) > 1:
            trough = float(nav_wave.min())
            peak = float(nav_wave.max())
            last = float(nav_wave.iloc[-1])
            if peak > trough and trough > 0:
                recovery_frac = float((last - trough) / (peak - trough))
                recovery_frac = float(np.clip(recovery_frac, 0.0, 1.0))
            else:
                recovery_frac = float("nan")
        else:
            recovery_frac = float("nan")

        vol_ratio = vol_wave / vol_bm if (vol_bm and not math.isnan(vol_bm)) else float("nan")

        rq_ir = float(np.clip(ir, 0.0, 1.5) / 1.5 * 15.0) if not math.isnan(ir) else 0.0
        rq_alpha = float(np.clip((alpha_365 + 0.05) / 0.15, 0.0, 1.0) * 10.0) if not math.isnan(alpha_365) else 0.0
        return_quality = float(np.clip(rq_ir + rq_alpha, 0.0, 25.0))

        if math.isnan(vol_ratio):
            risk_control = 0.0
        else:
            penalty = max(0.0, abs(vol_ratio - 0.9) - 0.1)
            risk_control = float(np.clip(1.0 - penalty / 0.6, 0.0, 1.0) * 25.0)

        consistency = float(np.clip(hit_rate, 0.0, 1.0) * 15.0) if not math.isnan(hit_rate) else 0.0

        if math.isnan(recovery_frac) or math.isnan(mdd_wave) or math.isnan(mdd_bm):
            resilience = 0.0
        else:
            rec_part = float(np.clip(recovery_frac, 0.0, 1.0) * 6.0)
            dd_edge = (mdd_bm - mdd_wave)
            dd_part = float(np.clip((dd_edge + 0.10) / 0.20, 0.0, 1.0) * 4.0)
            resilience = float(np.clip(rec_part + dd_part, 0.0, 10.0))

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


# ============================================================
# Table display formatting (percent display, no math changes)
# ============================================================
def build_formatter_map(df: pd.DataFrame) -> Dict[str, Any]:
    if df is None or df.empty:
        return {}

    fmt: Dict[str, Any] = {}

    pct_keywords = [
        " Ret",
        " Return",
        " Alpha",
        "Vol",
        "Volatility",
        "MaxDD",
        "Max Drawdown",
        "Tracking Error",
        "TE",
        "Benchmark Difficulty",
        "BM Difficulty",
        "VaR",
        "CVaR",
        "Downside",
    ]

    for c in df.columns:
        cs = str(c)

        if cs == "WaveScore":
            fmt[c] = lambda v: fmt_score(v)
            continue
        if cs in ["Return Quality", "Risk Control", "Consistency", "Resilience", "Efficiency", "Transparency"]:
            fmt[c] = lambda v: fmt_num(v, 1)
            continue

        if cs.startswith("β") or cs.startswith("beta") or cs.startswith("β_") or "Beta" in cs:
            fmt[c] = lambda v: fmt_num(v, 2)
            continue

        if ("IR" in cs) and ("Ret" not in cs) and ("Return" not in cs):
            fmt[c] = lambda v: fmt_num(v, 2)
            continue

        if ("Sharpe" in cs) or ("Sortino" in cs):
            fmt[c] = lambda v: fmt_num(v, 2)
            continue

        if any(k in cs for k in pct_keywords):
            fmt[c] = lambda v: fmt_pct(v, 2)

    return fmt


# ============================================================
# Row highlighting utilities (selected wave + alpha tint)
# ============================================================
def style_selected_and_alpha(df: pd.DataFrame, selected_wave: str):
    alpha_cols = [c for c in df.columns if ("Alpha" in str(c)) or str(c).endswith("α")]
    fmt_map = build_formatter_map(df)

    def row_style(row: pd.Series):
        styles = [""] * len(row)

        if "Wave" in df.columns and str(row.get("Wave", "")) == str(selected_wave):
            styles = ["background-color: rgba(255,255,255,0.07); font-weight: 700;"] * len(row)

        for i, col in enumerate(df.columns):
            if col in alpha_cols:
                try:
                    v = float(row[col])
                except Exception:
                    continue
                if math.isnan(v):
                    continue
                if v > 0:
                    styles[i] += "background-color: rgba(0, 255, 140, 0.10);"
                elif v < 0:
                    styles[i] += "background-color: rgba(255, 80, 80, 0.10);"
        return styles

    styler = df.style.apply(row_style, axis=1)
    if fmt_map:
        styler = styler.format(fmt_map)
    return styler


def show_df(df: pd.DataFrame, selected_wave: str, key: str):
    try:
        st.dataframe(style_selected_and_alpha(df, selected_wave), use_container_width=True, key=key)
    except Exception:
        st.dataframe(df, use_container_width=True, key=key)


# ============================================================
# One-click Wave jump (best-effort selection events)
# ============================================================
def selectable_table_jump(df: pd.DataFrame, key: str) -> None:
    if df is None or df.empty or "Wave" not in df.columns:
        st.info("No waves available to jump.")
        return

    try:
        event = st.dataframe(
            df,
            use_container_width=True,
            key=key,
            on_select="rerun",
            selection_mode="single-row",
        )
        sel = getattr(event, "selection", None)
        if sel and isinstance(sel, dict):
            rows = sel.get("rows", [])
            if rows:
                idx = int(rows[0])
                wave = str(df.iloc[idx]["Wave"])
                if wave:
                    st.session_state["selected_wave"] = wave
                    st.rerun()
        return
    except Exception:
        pass

    st.dataframe(df, use_container_width=True, key=f"{key}_fallback")
    pick = st.selectbox("Jump to Wave", list(df["Wave"]), key=f"{key}_pick")
    if st.button("Jump", key=f"{key}_btn"):
        st.session_state["selected_wave"] = pick
        st.rerun()


# ============================================================
# Alpha Heatmap View (All Waves x Timeframe)
# ============================================================
@st.cache_data(show_spinner=False)
def build_alpha_matrix(all_waves: List[str], mode: str) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for wname in all_waves:
        hist = compute_wave_history(wname, mode=mode, days=365)
        if hist.empty or len(hist) < 2:
            rows.append({"Wave": wname, "1D Alpha": np.nan, "30D Alpha": np.nan, "60D Alpha": np.nan, "365D Alpha": np.nan})
            continue

        nav_w = hist["wave_nav"]
        nav_b = hist["bm_nav"]

        # 1D alpha
        if len(nav_w) >= 2 and len(nav_b) >= 2:
            r1w = float(nav_w.iloc[-1] / nav_w.iloc[-2] - 1.0)
            r1b = float(nav_b.iloc[-1] / nav_b.iloc[-2] - 1.0)
            a1 = r1w - r1b
        else:
            a1 = np.nan

        r30w = ret_from_nav(nav_w, min(30, len(nav_w)))
        r30b = ret_from_nav(nav_b, min(30, len(nav_b)))
        a30 = r30w - r30b

        r60w = ret_from_nav(nav_w, min(60, len(nav_w)))
        r60b = ret_from_nav(nav_b, min(60, len(nav_b)))
        a60 = r60w - r60b

        r365w = ret_from_nav(nav_w, len(nav_w))
        r365b = ret_from_nav(nav_b, len(nav_b))
        a365 = r365w - r365b

        rows.append({"Wave": wname, "1D Alpha": a1, "30D Alpha": a30, "60D Alpha": a60, "365D Alpha": a365})

    return pd.DataFrame(rows).sort_values("Wave")


def plot_alpha_heatmap(alpha_df: pd.DataFrame, selected_wave: str, title: str):
    if go is None or alpha_df is None or alpha_df.empty:
        st.info("Heatmap unavailable (Plotly missing or no data).")
        return

    df = alpha_df.copy()
    cols = [c for c in ["1D Alpha", "30D Alpha", "60D Alpha", "365D Alpha"] if c in df.columns]
    if not cols:
        st.info("No alpha columns to plot.")
        return

    if "Wave" in df.columns and selected_wave in set(df["Wave"]):
        top = df[df["Wave"] == selected_wave]
        rest = df[df["Wave"] != selected_wave]
        df = pd.concat([top, rest], axis=0)

    z = df[cols].values
    y = df["Wave"].tolist()
    x = cols

    v = float(np.nanmax(np.abs(z))) if np.isfinite(z).any() else 0.10
    if not math.isfinite(v) or v <= 0:
        v = 0.10
    zmin, zmax = (-float(v), float(v))

    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=x,
            y=y,
            zmin=zmin,
            zmax=zmax,
            colorbar=dict(title="Alpha"),
        )
    )
    fig.update_layout(
        title=title,
        height=min(900, 240 + 22 * max(10, len(y))),
        margin=dict(l=80, r=40, t=60, b=40),
        xaxis_title="Timeframe",
        yaxis_title="Wave",
    )
    st.plotly_chart(fig, use_container_width=True)


# ============================================================
# Wave Doctor (diagnostics + suggestions)
# ============================================================
def wave_doctor_assess(
    wave_name: str,
    mode: str,
    days: int = 365,
    alpha_warn: float = 0.08,
    te_warn: float = 0.20,
    vol_warn: float = 0.30,
    mdd_warn: float = -0.25,
) -> Dict[str, Any]:
    hist = compute_wave_history(wave_name, mode=mode, days=days)
    if hist.empty or len(hist) < 2:
        return {"ok": False, "message": "Not enough data to run Wave Doctor."}

    nav_w = hist["wave_nav"]
    nav_b = hist["bm_nav"]
    ret_w = hist["wave_ret"]
    ret_b = hist["bm_ret"]

    r365_w = ret_from_nav(nav_w, len(nav_w))
    r365_b = ret_from_nav(nav_b, len(nav_b))
    a365 = r365_w - r365_b

    r30_w = ret_from_nav(nav_w, min(30, len(nav_w)))
    r30_b = ret_from_nav(nav_b, min(30, len(nav_b)))
    a30 = r30_w - r30_b

    vol_w = annualized_vol(ret_w)
    vol_b = annualized_vol(ret_b)
    te = tracking_error(ret_w, ret_b)
    ir = information_ratio(nav_w, nav_b, te)
    mdd_w = max_drawdown(nav_w)
    mdd_b = max_drawdown(nav_b)

    spy_nav = compute_spy_nav(days=days)
    spy_ret = ret_from_nav(spy_nav, len(spy_nav)) if len(spy_nav) >= 2 else float("nan")
    bm_difficulty = (r365_b - spy_ret) if (pd.notna(r365_b) and pd.notna(spy_ret)) else float("nan")

    flags: List[str] = []
    diagnosis: List[str] = []
    recs: List[str] = []

    if pd.notna(a30) and abs(a30) >= alpha_warn:
        flags.append("Large 30D alpha (verify benchmark + coverage)")
        diagnosis.append("30D alpha is unusually large. This can happen from real signal, but also from benchmark mix or data coverage shifts.")
        recs.append("Check Benchmark Snapshot ID + Coverage panel; drift or gaps can inflate alpha. Freeze benchmark mix for demo comparisons.")

    if pd.notna(a365) and a365 < 0:
        diagnosis.append("365D alpha is negative. This might reflect true underperformance, or a tougher benchmark composition.")
        if pd.notna(bm_difficulty) and bm_difficulty > 0.03:
            diagnosis.append("Benchmark difficulty is positive vs SPY (benchmark outperformed SPY), so alpha is harder on this window.")
            recs.append("For validation, temporarily compare to SPY/QQQ-style benchmark to isolate engine effect.")
    else:
        if pd.notna(a365) and a365 > 0.06:
            diagnosis.append("365D alpha is solidly positive. Lock benchmark mix snapshot for reproducibility in demos.")

    if pd.notna(te) and te > te_warn:
        flags.append("High tracking error (active risk elevated)")
        diagnosis.append("Tracking error is high; the wave behaves very differently than its benchmark.")
        recs.append("Reduce tilt strength and/or tighten exposure caps (What-If Lab) to lower TE (shadow only).")

    if pd.notna(vol_w) and vol_w > vol_warn:
        flags.append("High volatility")
        diagnosis.append("Volatility is elevated relative to institutional tolerances.")
        recs.append("Lower vol target and/or tighten exposure caps (What-If Lab — shadow only).")

    if pd.notna(mdd_w) and mdd_w < mdd_warn:
        flags.append("Deep drawdown")
        diagnosis.append("Drawdown is deep. Consider stronger SmartSafe posture in stress regimes.")
        recs.append("Increase safe fraction or regime gating (What-If Lab — shadow only).")

    if pd.notna(vol_b) and pd.notna(vol_w) and vol_b > 0 and (vol_w / vol_b) > 1.6:
        flags.append("Volatility much higher than benchmark")
        diagnosis.append("Wave volatility is much higher than benchmark; this can inflate wins/losses.")
        recs.append("Tighten exposure caps + reduce tilt strength to stabilize (shadow only).")

    if not diagnosis:
        diagnosis.append("No major anomalies detected by Wave Doctor on the selected window.")

    # extra console-only stats
    b = beta_ols(ret_w, ret_b)
    sh = sharpe_ratio(ret_w, rf_annual=0.0)
    so = sortino_ratio(ret_w, mar_annual=0.0)
    v95, c95 = var_cvar(ret_w, level=0.95)
    sk, ku = skew_kurt(ret_w)

    return {
        "ok": True,
        "metrics": {
            "Return_365D": r365_w,
            "Alpha_365D": a365,
            "Return_30D": r30_w,
            "Alpha_30D": a30,
            "Vol_Wave": vol_w,
            "Vol_Benchmark": vol_b,
            "TE": te,
            "IR": ir,
            "MaxDD_Wave": mdd_w,
            "MaxDD_Benchmark": mdd_b,
            "Benchmark_Difficulty_BM_minus_SPY": bm_difficulty,
            "Beta_vs_BM": b,
            "Sharpe_0rf": sh,
            "Sortino_0mar": so,
            "VaR95_daily": v95,
            "CVaR95_daily": c95,
            "Skew": sk,
            "Kurtosis": ku,
        },
        "flags": flags,
        "diagnosis": diagnosis,
        "recommendations": list(dict.fromkeys(recs)),
    }


# ============================================================
# What-If Lab (shadow sim)
# ============================================================
def _regime_from_spy_60d(spy_nav: pd.Series) -> pd.Series:
    spy_nav = safe_series(spy_nav)
    if spy_nav.empty:
        return pd.Series(dtype=str)
    r60 = spy_nav / spy_nav.shift(60) - 1.0

    def label(x: float) -> str:
        if pd.isna(x):
            return "neutral"
        if x <= -0.12:
            return "panic"
        if x <= -0.04:
            return "downtrend"
        if x < 0.06:
            return "neutral"
        return "uptrend"

    return r60.apply(label)


def _vix_exposure_factor_series(vix: pd.Series, mode: str) -> pd.Series:
    vix = safe_series(vix).astype(float)
    if vix.empty:
        return pd.Series(dtype=float)

    def f(v: float) -> float:
        if pd.isna(v) or v <= 0:
            base = 1.0
        elif v < 15:
            base = 1.15
        elif v < 20:
            base = 1.05
        elif v < 25:
            base = 0.95
        elif v < 30:
            base = 0.85
        elif v < 40:
            base = 0.75
        else:
            base = 0.60
        if mode == "Alpha-Minus-Beta":
            base -= 0.05
        elif mode == "Private Logic":
            base += 0.05
        return float(np.clip(base, 0.5, 1.3))

    return vix.apply(f)


def _vix_safe_fraction_series(vix: pd.Series, mode: str) -> pd.Series:
    vix = safe_series(vix).astype(float)
    if vix.empty:
        return pd.Series(dtype=float)

    def g(v: float) -> float:
        if pd.isna(v) or v <= 0:
            base = 0.0
        elif v < 18:
            base = 0.00
        elif v < 24:
            base = 0.05
        elif v < 30:
            base = 0.15
        elif v < 40:
            base = 0.25
        else:
            base = 0.40
        if mode == "Alpha-Minus-Beta":
            base *= 1.5
        elif mode == "Private Logic":
            base *= 0.7
        return float(np.clip(base, 0.0, 0.8))

    return vix.apply(g)


@st.cache_data(show_spinner=False)
def simulate_whatif_nav(
    wave_name: str,
    mode: str,
    days: int,
    tilt_strength: float,
    vol_target: float,
    extra_safe_boost: float,
    exp_min: float,
    exp_max: float,
    freeze_benchmark: bool,
) -> pd.DataFrame:
    hold_df = get_wave_holdings(wave_name)
    weights = _weights_from_df(hold_df, "Ticker", "Weight")
    if weights.empty:
        return pd.DataFrame()

    tickers = list(weights.index)
    needed = set(tickers + ["SPY", "^VIX", "SGOV", "BIL", "SHY"])
    px = fetch_prices_daily(list(needed), days=days)
    if px.empty or "SPY" not in px.columns or "^VIX" not in px.columns:
        return pd.DataFrame()

    px = px.sort_index().ffill().bfill()
    if len(px) > days:
        px = px.iloc[-days:]

    rets = px.pct_change().fillna(0.0)
    w = weights.reindex(px.columns).fillna(0.0)

    spy_nav = (1.0 + rets["SPY"]).cumprod()
    regime = _regime_from_spy_60d(spy_nav)
    vix = px["^VIX"]

    vix_exposure = _vix_exposure_factor_series(vix, mode)
    vix_safe = _vix_safe_fraction_series(vix, mode)

    base_expo = 1.0
    try:
        base_map = getattr(we, "MODE_BASE_EXPOSURE", None)
        if isinstance(base_map, dict) and mode in base_map:
            base_expo = float(base_map[mode])
    except Exception:
        pass

    regime_exposure_map = {"panic": 0.80, "downtrend": 0.90, "neutral": 1.00, "uptrend": 1.10}
    try:
        rm = getattr(we, "REGIME_EXPOSURE", None)
        if isinstance(rm, dict):
            regime_exposure_map = {k: float(v) for k, v in rm.items()}
    except Exception:
        pass

    def regime_gate(mode_in: str, reg: str) -> float:
        try:
            rg = getattr(we, "REGIME_GATING", None)
            if isinstance(rg, dict) and mode_in in rg and reg in rg[mode_in]:
                return float(rg[mode_in][reg])
        except Exception:
            pass
        fallback = {
            "Standard": {"panic": 0.50, "downtrend": 0.30, "neutral": 0.10, "uptrend": 0.00},
            "Alpha-Minus-Beta": {"panic": 0.75, "downtrend": 0.50, "neutral": 0.25, "uptrend": 0.05},
            "Private Logic": {"panic": 0.40, "downtrend": 0.25, "neutral": 0.05, "uptrend": 0.00},
        }
        return float(fallback.get(mode_in, fallback["Standard"]).get(reg, 0.10))

    safe_ticker = "SGOV" if "SGOV" in rets.columns else ("BIL" if "BIL" in rets.columns else ("SHY" if "SHY" in rets.columns else "SPY"))
    safe_ret = rets[safe_ticker]
    mom60 = px / px.shift(60) - 1.0

    wave_ret: List[float] = []
    dates: List[pd.Timestamp] = []

    for dtt in rets.index:
        r = rets.loc[dtt]

        mom_row = mom60.loc[dtt] if dtt in mom60.index else None
        if mom_row is not None:
            mom_clipped = mom_row.reindex(px.columns).fillna(0.0).clip(lower=-0.30, upper=0.30)
            tilt = 1.0 + tilt_strength * mom_clipped
            ew = (w * tilt).clip(lower=0.0)
        else:
            ew = w.copy()

        ew_hold = ew.reindex(tickers).fillna(0.0)
        s = float(ew_hold.sum())
        if s > 0:
            rw = ew_hold / s
        else:
            rw = w.reindex(tickers).fillna(0.0)
            s2 = float(rw.sum())
            rw = (rw / s2) if s2 > 0 else rw

        port_risk_ret = float((r.reindex(tickers).fillna(0.0) * rw).sum())

        if len(wave_ret) >= 20:
            recent = np.array(wave_ret[-20:], dtype=float)
            realized = float(recent.std() * np.sqrt(252))
        else:
            realized = float(vol_target)

        vol_adj = 1.0
        if realized > 0 and math.isfinite(realized):
            vol_adj = float(np.clip(vol_target / realized, 0.7, 1.3))

        reg = str(regime.get(dtt, "neutral"))
        reg_expo = float(regime_exposure_map.get(reg, 1.0))

        vix_expo = float(vix_exposure.get(dtt, 1.0))
        vix_gate = float(vix_safe.get(dtt, 0.0))

        expo = float(np.clip(base_expo * reg_expo * vol_adj * vix_expo, exp_min, exp_max))

        sf = float(np.clip(regime_gate(mode, reg) + vix_gate + extra_safe_boost, 0.0, 0.95))
        rf = 1.0 - sf

        total = sf * float(safe_ret.get(dtt, 0.0)) + rf * expo * port_risk_ret

        if mode == "Private Logic" and len(wave_ret) >= 20:
            recent = np.array(wave_ret[-20:], dtype=float)
            daily_vol = float(recent.std())
            if daily_vol > 0 and math.isfinite(daily_vol):
                shock = 2.0 * daily_vol
                if total <= -shock:
                    total = total * 1.30
                elif total >= shock:
                    total = total * 0.70

        wave_ret.append(float(total))
        dates.append(pd.Timestamp(dtt))

    wave_ret_s = pd.Series(wave_ret, index=pd.Index(dates, name="Date"), name="whatif_ret")
    wave_nav = (1.0 + wave_ret_s).cumprod().rename("whatif_nav")

    out = pd.DataFrame({"whatif_nav": wave_nav, "whatif_ret": wave_ret_s})

    if freeze_benchmark:
        hist_eng = compute_wave_history(wave_name, mode=mode, days=days)
        if not hist_eng.empty and "bm_nav" in hist_eng.columns:
            bm_nav = hist_eng["bm_nav"].reindex(out.index).ffill().bfill()
            bm_ret = hist_eng["bm_ret"].reindex(out.index).fillna(0.0)
            out["bm_nav"] = bm_nav
            out["bm_ret"] = bm_ret
    else:
        if "SPY" in px.columns:
            spy_ret = rets["SPY"].reindex(out.index).fillna(0.0)
            spy_nav2 = (1.0 + spy_ret).cumprod()
            out["bm_nav"] = spy_nav2
            out["bm_ret"] = spy_ret

    return out


# ============================================================
# Correlation analytics (All Waves)
# ============================================================
@st.cache_data(show_spinner=False)
def build_returns_panel(all_waves: List[str], mode: str, days: int = 365) -> pd.DataFrame:
    """Wide DF: columns=Wave, values=daily returns (aligned by date)."""
    series = []
    for w in all_waves:
        h = compute_wave_history(w, mode=mode, days=days)
        if h is None or h.empty or "wave_ret" not in h.columns:
            continue
        s = h["wave_ret"].rename(w)
        series.append(s)
    if not series:
        return pd.DataFrame()
    df = pd.concat(series, axis=1).sort_index()
    return df.dropna(how="all")


@st.cache_data(show_spinner=False)
def corr_matrix_from_returns(ret_df: pd.DataFrame) -> pd.DataFrame:
    if ret_df is None or ret_df.empty:
        return pd.DataFrame()
    return ret_df.corr()


def plot_corr_heatmap(corr_df: pd.DataFrame, title: str = "Correlation Matrix"):
    if corr_df is None or corr_df.empty:
        st.info("Correlation matrix unavailable (no aligned returns).")
        return
    if go is None:
        st.dataframe(corr_df.round(2), use_container_width=True)
        return

    z = corr_df.values
    labels = corr_df.columns.tolist()
    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=labels,
            y=labels,
            zmin=-1,
            zmax=1,
            colorbar=dict(title="ρ"),
        )
    )
    fig.update_layout(
        title=title,
        height=min(900, 260 + 18 * max(10, len(labels))),
        margin=dict(l=80, r=40, t=60, b=80),
    )
    st.plotly_chart(fig, use_container_width=True)
    # ============================================================
# Sidebar: waves & modes
# ============================================================
try:
    all_waves = we.get_all_waves()
    if all_waves is None:
        all_waves = []
except Exception:
    all_waves = []

try:
    all_modes = we.get_modes()
    if all_modes is None:
        all_modes = []
except Exception:
    all_modes = []

if "selected_wave" not in st.session_state:
    st.session_state["selected_wave"] = all_waves[0] if all_waves else ""
if "mode" not in st.session_state:
    st.session_state["mode"] = all_modes[0] if all_modes else "Standard"

with st.sidebar:
    st.title("WAVES Intelligence™")
    st.caption("Mini Bloomberg Console • Vector OS™")

    if all_modes:
        st.selectbox(
            "Mode",
            all_modes,
            index=all_modes.index(st.session_state["mode"]) if st.session_state["mode"] in all_modes else 0,
            key="mode",
        )
    else:
        st.text_input("Mode", value=st.session_state.get("mode", "Standard"), key="mode")

    if all_waves:
        st.selectbox(
            "Select Wave",
            all_waves,
            index=all_waves.index(st.session_state["selected_wave"]) if st.session_state["selected_wave"] in all_waves else 0,
            key="selected_wave",
        )
    else:
        st.text_input("Select Wave", value=st.session_state.get("selected_wave", ""), key="selected_wave")

    st.markdown("---")
    st.markdown("**Display settings**")
    history_days = st.slider("History window (days)", min_value=60, max_value=730, value=365, step=15)

    st.markdown("---")
    st.markdown("**Wave Doctor settings**")
    alpha_warn = st.slider("Alpha alert threshold (abs, 30D)", 0.02, 0.20, 0.08, 0.01)
    te_warn = st.slider("Tracking error alert threshold", 0.05, 0.40, 0.20, 0.01)

    st.markdown("---")
    st.markdown("**Risk Lab settings**")
    roll_alpha_window = st.selectbox("Rolling alpha window", [20, 30, 60, 90], index=1)
    roll_vol_window = st.selectbox("Rolling vol window", [10, 20, 30, 60], index=1)

mode = st.session_state["mode"]
selected_wave = st.session_state["selected_wave"]

if not selected_wave:
    st.error("No Waves available (engine returned empty list). Verify waves_engine.get_all_waves().")
    st.stop()

# ============================================================
# Page header
# ============================================================
st.title("WAVES Intelligence™ Institutional Console")
st.caption("Live Alpha Capture • SmartSafe™ • Multi-Asset • Crypto • Gold • Income Ladders")

# ============================================================
# Pinned Summary Bar (Sticky) + Governance chips (NEW)
# ============================================================
h_bar = compute_wave_history(selected_wave, mode=mode, days=max(365, history_days))
bar_r30 = bar_a30 = bar_r365 = bar_a365 = float("nan")
bar_te = bar_ir = float("nan")
bar_src = "—"

bm_mix_for_src = get_benchmark_mix()
bar_src = benchmark_source_label(selected_wave, bm_mix_for_src)

bm_sid = benchmark_snapshot_id(selected_wave, bm_mix_for_src)
bm_drift = benchmark_drift_status(selected_wave, mode=mode, snapshot_id=bm_sid)

cov = coverage_report(h_bar)
cov_score = cov.get("completeness_score", float("nan"))
cov_rows = cov.get("rows", 0)
cov_age = cov.get("age_days", None)

if not h_bar.empty and len(h_bar) >= 2:
    nav_w = h_bar["wave_nav"]
    nav_b = h_bar["bm_nav"]
    ret_w = h_bar["wave_ret"]
    ret_b = h_bar["bm_ret"]

    r30w = ret_from_nav(nav_w, min(30, len(nav_w)))
    r30b = ret_from_nav(nav_b, min(30, len(nav_b)))
    bar_r30 = r30w
    bar_a30 = r30w - r30b

    r365w = ret_from_nav(nav_w, len(nav_w))
    r365b = ret_from_nav(nav_b, len(nav_b))
    bar_r365 = r365w
    bar_a365 = r365w - r365b

    bar_te = tracking_error(ret_w, ret_b)
    bar_ir = information_ratio(nav_w, nav_b, bar_te)

spy_vix = fetch_spy_vix(days=min(365, max(120, history_days)))
reg_now = "—"
vix_last = float("nan")
if not spy_vix.empty and "SPY" in spy_vix.columns and "^VIX" in spy_vix.columns and len(spy_vix) > 60:
    vix_last = float(spy_vix["^VIX"].iloc[-1])
    spy_nav = (1.0 + spy_vix["SPY"].pct_change().fillna(0.0)).cumprod()
    r60 = spy_nav / spy_nav.shift(60) - 1.0

    def lab(x: Any) -> str:
        try:
            if pd.isna(x):
                return "neutral"
            x = float(x)
            if x <= -0.12:
                return "panic"
            if x <= -0.04:
                return "downtrend"
            if x < 0.06:
                return "neutral"
            return "uptrend"
        except Exception:
            return "neutral"

    reg_now = str(r60.apply(lab).iloc[-1])

ws_snap = compute_wavescore_for_all_waves(all_waves, mode=mode, days=365)
ws_val = "—"
ws_grade = "—"
ws_rank = "—"
if ws_snap is not None and not ws_snap.empty:
    ws_sorted = ws_snap.sort_values("WaveScore", ascending=False).reset_index(drop=True)
    rr = ws_sorted[ws_sorted["Wave"] == selected_wave]
    if not rr.empty:
        ws_val = fmt_score(float(rr.iloc[0]["WaveScore"]))
        ws_grade = str(rr.iloc[0]["Grade"])
        ws_rank = str(int(rr.index[0]) + 1)

drift_chip = "Stable" if bm_drift == "stable" else "DRIFT"
cov_chip = f"{fmt_num(cov_score,1)}" if (cov_score is not None and not (isinstance(cov_score, float) and math.isnan(cov_score))) else "—"
age_chip = f"{cov_age}d" if isinstance(cov_age, int) else "—"

st.markdown(
    f"""
<div class="waves-sticky">
  <div class="waves-hdr">📌 Live Summary</div>
  <span class="waves-chip">Mode: <b>{mode}</b></span>
  <span class="waves-chip">Wave: <b>{selected_wave}</b></span>
  <span class="waves-chip">Benchmark: <b>{bar_src}</b></span>
  <span class="waves-chip">BM Snapshot: <b>{bm_sid}</b> · <b>{drift_chip}</b></span>
  <span class="waves-chip">Coverage: <b>{cov_chip}</b> · Rows: <b>{cov_rows}</b> · Age: <b>{age_chip}</b></span>
  <span class="waves-chip">Regime: <b>{reg_now}</b></span>
  <span class="waves-chip">VIX: <b>{fmt_num(vix_last, 1) if not math.isnan(vix_last) else "—"}</b></span>
  <span class="waves-chip">30D α: <b>{fmt_pct(bar_a30)}</b> · 30D r: <b>{fmt_pct(bar_r30)}</b></span>
  <span class="waves-chip">365D α: <b>{fmt_pct(bar_a365)}</b> · 365D r: <b>{fmt_pct(bar_r365)}</b></span>
  <span class="waves-chip">TE: <b>{fmt_pct(bar_te)}</b> · IR: <b>{fmt_num(bar_ir, 2)}</b></span>
  <span class="waves-chip">WaveScore: <b>{ws_val}</b> ({ws_grade}) · Rank: <b>{ws_rank}</b></span>
</div>
""",
    unsafe_allow_html=True,
)

# ============================================================
# Tabs (Market Intel REMOVED)
# ============================================================
tab_console, tab_factors, tab_risk, tab_corr, tab_vector = st.tabs(
    ["Console", "Factor Decomposition", "Risk Lab", "Correlation", "Vector OS Insight Layer"]
)

# ============================================================
# TAB 1: Console
# ============================================================
with tab_console:
    st.subheader("🔥 Alpha Heatmap View (All Waves × Timeframe)")
    st.caption("Fast scan. Jump table highlights selected wave. Values display as % (math unchanged).")

    alpha_df = build_alpha_matrix(all_waves, mode)
    plot_alpha_heatmap(alpha_df, selected_wave, title=f"Alpha Heatmap — Mode: {mode}")

    st.markdown("### 🧭 One-Click Jump Table")
    jump_df = alpha_df.copy()
    jump_df["RankScore"] = jump_df[["1D Alpha", "30D Alpha", "60D Alpha", "365D Alpha"]].mean(axis=1, skipna=True)
    jump_df = jump_df.sort_values("RankScore", ascending=False)

    show_df(jump_df, selected_wave, key="wave_jump_table_fmt")
    selectable_table_jump(jump_df, key="wave_jump_table_select")

    st.markdown("---")

    st.subheader("🏁 WaveScore Leaderboard (Mode Snapshot)")
    if ws_snap is None or ws_snap.empty:
        st.info("WaveScore table unavailable.")
    else:
        lb = ws_snap.sort_values("WaveScore", ascending=False).reset_index(drop=True)
        lb.insert(0, "Rank", np.arange(1, len(lb) + 1))
        show_df(lb, selected_wave, key="wavescore_leaderboard")

    st.markdown("---")

    st.subheader("Market Regime Monitor — SPY vs VIX")
    spy_vix2 = fetch_spy_vix(days=history_days)

    if spy_vix2.empty or "SPY" not in spy_vix2.columns or "^VIX" not in spy_vix2.columns:
        st.warning("Unable to load SPY/VIX data at the moment.")
    else:
        spy = spy_vix2["SPY"].copy()
        vix = spy_vix2["^VIX"].copy()
        spy_norm = (spy / spy.iloc[0] * 100.0) if len(spy) > 0 else spy

        if go is not None:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=spy_vix2.index, y=spy_norm, name="SPY (Index = 100)", mode="lines"))
            fig.add_trace(go.Scatter(x=spy_vix2.index, y=vix, name="VIX Level", mode="lines", yaxis="y2"))
            fig.update_layout(
                margin=dict(l=40, r=40, t=40, b=40),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                xaxis=dict(title="Date"),
                yaxis=dict(title="SPY (Indexed)", rangemode="tozero"),
                yaxis2=dict(title="VIX", overlaying="y", side="right", rangemode="tozero"),
                height=380,
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.line_chart(pd.DataFrame({"SPY_idx": spy_norm, "VIX": vix}))

    st.markdown("---")

    st.subheader("🧾 All Waves Overview (1D / 30D / 60D / 365D Returns + Alpha)")
    overview_rows: List[Dict[str, Any]] = []

    for wname in all_waves:
        hist = compute_wave_history(wname, mode=mode, days=365)
        if hist.empty or len(hist) < 2:
            overview_rows.append(
                {"Wave": wname, "1D Ret": np.nan, "1D Alpha": np.nan, "30D Ret": np.nan, "30D Alpha": np.nan,
                 "60D Ret": np.nan, "60D Alpha": np.nan, "365D Ret": np.nan, "365D Alpha": np.nan}
            )
            continue

        nav_w = hist["wave_nav"]
        nav_b = hist["bm_nav"]

        if len(nav_w) >= 2 and len(nav_b) >= 2:
            r1w = float(nav_w.iloc[-1] / nav_w.iloc[-2] - 1.0)
            r1b = float(nav_b.iloc[-1] / nav_b.iloc[-2] - 1.0)
            a1 = r1w - r1b
        else:
            r1w = np.nan
            a1 = np.nan

        r30w = ret_from_nav(nav_w, min(30, len(nav_w)))
        r30b = ret_from_nav(nav_b, min(30, len(nav_b)))
        a30 = r30w - r30b

        r60w = ret_from_nav(nav_w, min(60, len(nav_w)))
        r60b = ret_from_nav(nav_b, min(60, len(nav_b)))
        a60 = r60w - r60b

        r365w = ret_from_nav(nav_w, len(nav_w))
        r365b = ret_from_nav(nav_b, len(nav_b))
        a365 = r365w - r365b

        overview_rows.append(
            {
                "Wave": wname,
                "1D Ret": r1w,
                "1D Alpha": a1,
                "30D Ret": r30w,
                "30D Alpha": a30,
                "60D Ret": r60w,
                "60D Alpha": a60,
                "365D Ret": r365w,
                "365D Alpha": a365,
            }
        )

    overview_df = pd.DataFrame(overview_rows)
    show_df(overview_df, selected_wave, key="all_waves_overview")

    st.markdown("---")

    st.subheader(f"📌 Selected Wave — {selected_wave}")
    hold = get_wave_holdings(selected_wave)

    if hold is None or hold.empty:
        st.warning("No holdings returned by engine for this wave.")
    else:
        hold2 = hold.copy()
        if "Weight" in hold2.columns:
            hold2["Weight"] = pd.to_numeric(hold2["Weight"], errors="coerce")
        if "Ticker" in hold2.columns:
            hold2["Ticker"] = hold2["Ticker"].astype(str)

        top10 = hold2.sort_values("Weight", ascending=False).head(10) if "Weight" in hold2.columns else hold2.head(10)

        st.markdown("### 🧾 Top 10 Holdings (clickable)")
        for _, r in top10.iterrows():
            t = str(r.get("Ticker", ""))
            wgt = r.get("Weight", np.nan)
            nm = str(r.get("Name", t))
            if t:
                url = f"https://www.google.com/finance/quote/{t}"
                st.markdown(f"- **[{t}]({url})** — {nm} — **{fmt_pct(wgt)}**")

        try:
            wts = pd.to_numeric(hold2.get("Weight", pd.Series(dtype=float)), errors="coerce").fillna(0.0)
            wts = wts[wts > 0]
            top10_w = float(wts.sort_values(ascending=False).head(10).sum()) if len(wts) else float("nan")
            hhi = float(np.sum((wts.values) ** 2)) if len(wts) else float("nan")
        except Exception:
            top10_w, hhi = float("nan"), float("nan")

        c1, c2 = st.columns(2)
        with c1:
            st.metric("Top-10 Weight", fmt_pct(top10_w))
        with c2:
            st.metric("HHI (Concentration)", fmt_num(hhi, 4))

        st.markdown("### Full Holdings")
        show_df(hold2, selected_wave, key="holdings_full")

    st.markdown("---")

    st.subheader("✅ Benchmark Truth + Attribution (Engine vs Basket)")
    colA, colB = st.columns(2)

    with colA:
        st.markdown("#### Benchmark Mix (as used by Engine)")
        bm_mix = get_benchmark_mix()
        if bm_mix is None or bm_mix.empty:
            st.info("No benchmark mix table returned by engine.")
        else:
            show_df(bm_mix[bm_mix["Wave"] == selected_wave] if "Wave" in bm_mix.columns else bm_mix, selected_wave, key="bm_mix")

        # NEW: Governance additions
        st.markdown("#### Governance — Benchmark Stability + Coverage")
        st.write(f"- **Benchmark Snapshot ID:** {bm_sid}")
        st.write(f"- **Benchmark Drift (session):** {drift_chip}")
        st.write(f"- **Coverage Score:** {cov_chip} / 100")
        st.write(f"- **History Rows:** {cov_rows}")
        st.write(f"- **Last Data Age:** {age_chip}")
        if cov.get("flags"):
            st.warning(" | ".join(cov["flags"]))
        with st.expander("Coverage Details", expanded=False):
            st.write(cov)

    with colB:
        st.markdown("#### Attribution Snapshot (365D)")
        attrib = compute_alpha_attribution(selected_wave, mode=mode, days=365)
        if not attrib:
            st.info("Attribution unavailable.")
        else:
            arows = []
            for k, v in attrib.items():
                if any(x in k for x in ["Return", "Alpha", "Vol", "MaxDD", "TE", "Difficulty", "VaR", "CVaR"]):
                    arows.append({"Metric": k, "Value": fmt_pct(v)})
                elif any(x in k for x in ["IR", "Sharpe", "Sortino", "Beta"]):
                    arows.append({"Metric": k, "Value": fmt_num(v, 2)})
                else:
                    arows.append({"Metric": k, "Value": fmt_num(v, 4)})
            st.dataframe(pd.DataFrame(arows), use_container_width=True)

    st.markdown("---")

    st.subheader("🩺 Wave Doctor")
    wd = wave_doctor_assess(selected_wave, mode=mode, days=365, alpha_warn=alpha_warn, te_warn=te_warn)
    if not wd.get("ok", False):
        st.info(wd.get("message", "Wave Doctor unavailable."))
    else:
        m = wd["metrics"]
        mdf = pd.DataFrame(
            [
                {"Metric": "365D Return", "Value": fmt_pct(m["Return_365D"])},
                {"Metric": "365D Alpha", "Value": fmt_pct(m["Alpha_365D"])},
                {"Metric": "30D Return", "Value": fmt_pct(m["Return_30D"])},
                {"Metric": "30D Alpha", "Value": fmt_pct(m["Alpha_30D"])},
                {"Metric": "Vol (Wave)", "Value": fmt_pct(m["Vol_Wave"])},
                {"Metric": "Vol (Benchmark)", "Value": fmt_pct(m["Vol_Benchmark"])},
                {"Metric": "Tracking Error (TE)", "Value": fmt_pct(m["TE"])},
                {"Metric": "Information Ratio (IR)", "Value": fmt_num(m["IR"], 2)},
                {"Metric": "Beta vs Benchmark", "Value": fmt_num(m["Beta_vs_BM"], 2)},
                {"Metric": "Sharpe (0% rf)", "Value": fmt_num(m["Sharpe_0rf"], 2)},
                {"Metric": "Sortino (0% MAR)", "Value": fmt_num(m["Sortino_0mar"], 2)},
                {"Metric": "VaR 95% (daily)", "Value": fmt_pct(m["VaR95_daily"])},
                {"Metric": "CVaR 95% (daily)", "Value": fmt_pct(m["CVaR95_daily"])},
                {"Metric": "Skew", "Value": fmt_num(m["Skew"], 2)},
                {"Metric": "Kurtosis", "Value": fmt_num(m["Kurtosis"], 2)},
                {"Metric": "MaxDD (Wave)", "Value": fmt_pct(m["MaxDD_Wave"])},
                {"Metric": "MaxDD (Benchmark)", "Value": fmt_pct(m["MaxDD_Benchmark"])},
                {"Metric": "BM Difficulty (BM - SPY)", "Value": fmt_pct(m["Benchmark_Difficulty_BM_minus_SPY"])},
            ]
        )
        st.dataframe(mdf, use_container_width=True)

        if wd.get("flags"):
            st.warning(" | ".join(wd["flags"]))
        st.markdown("**Diagnosis**")
        for line in wd.get("diagnosis", []):
            st.write(f"- {line}")
        if wd.get("recommendations"):
            st.markdown("**Recommendations (shadow controls)**")
            for line in wd["recommendations"]:
                st.write(f"- {line}")

    st.markdown("---")

    st.subheader("🧪 What-If Lab (Shadow Simulation)")
    st.caption("This does NOT change engine math. It is a sandbox overlay simulation for diagnostics.")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        tilt_strength = st.slider("Tilt strength", 0.0, 1.0, 0.30, 0.05)
    with c2:
        vol_target = st.slider("Vol target (annual)", 0.05, 0.50, 0.20, 0.01)
    with c3:
        extra_safe = st.slider("Extra safe boost", 0.0, 0.40, 0.00, 0.01)
    with c4:
        freeze_bm = st.checkbox("Freeze benchmark (use engine BM)", value=True)

    c5, c6 = st.columns(2)
    with c5:
        exp_min = st.slider("Exposure min", 0.0, 1.5, 0.60, 0.05)
    with c6:
        exp_max = st.slider("Exposure max", 0.2, 2.0, 1.20, 0.05)

    if st.button("Run What-If Shadow Sim"):
        sim = simulate_whatif_nav(
            selected_wave,
            mode=mode,
            days=min(365, max(120, history_days)),
            tilt_strength=tilt_strength,
            vol_target=vol_target,
            extra_safe_boost=extra_safe,
            exp_min=exp_min,
            exp_max=exp_max,
            freeze_benchmark=freeze_bm,
        )
        if sim is None or sim.empty:
            st.warning("Simulation failed (insufficient prices).")
        else:
            nav = sim["whatif_nav"]
            bm_nav = sim["bm_nav"] if "bm_nav" in sim.columns else None
            ret_total = ret_from_nav(nav, len(nav))
            alpha_total = ret_total - (ret_from_nav(bm_nav, len(bm_nav)) if bm_nav is not None and len(bm_nav) > 1 else 0.0)

            st.markdown(f"**What-If Return:** {fmt_pct(ret_total)}   |   **What-If Alpha:** {fmt_pct(alpha_total)}")

            if go is not None:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=sim.index, y=sim["whatif_nav"], name="What-If NAV", mode="lines"))
                if "bm_nav" in sim.columns:
                    fig.add_trace(go.Scatter(x=sim.index, y=sim["bm_nav"], name="Benchmark NAV", mode="lines"))
                fig.update_layout(height=380, margin=dict(l=40, r=40, t=40, b=40))
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.line_chart(sim[["whatif_nav"] + (["bm_nav"] if "bm_nav" in sim.columns else [])])

# ============================================================
# TAB 2: Factor Decomposition
# ============================================================
with tab_factors:
    st.subheader("🧩 Factor Decomposition (Simple Regression Betas)")
    st.caption("Uses SPY/QQQ/IWM/TLT/GLD daily returns as factor proxies. Display only.")

    hist = compute_wave_history(selected_wave, mode=mode, days=min(365, max(120, history_days)))
    if hist is None or hist.empty or "wave_ret" not in hist.columns:
        st.warning("Wave history unavailable.")
    else:
        factors_px = fetch_prices_daily(["SPY", "QQQ", "IWM", "TLT", "GLD"], days=min(365, max(120, history_days)))
        if factors_px is None or factors_px.empty:
            st.warning("Factor price data unavailable.")
        else:
            factor_ret = factors_px.pct_change().fillna(0.0)
            wave_ret = hist["wave_ret"].reindex(factor_ret.index).fillna(0.0)
            betas = regress_factors(wave_ret, factor_ret)

            bdf = pd.DataFrame([{"Factor": k, "Beta": v} for k, v in betas.items()])
            bdf["Beta"] = bdf["Beta"].apply(lambda x: fmt_num(x, 2))
            st.dataframe(bdf, use_container_width=True)

# ============================================================
# TAB 3: Risk Lab
# ============================================================
with tab_risk:
    st.subheader("🛡️ Risk Lab (Selected Wave)")
    st.caption("Console-side analytics from the engine NAV/return series. No engine math changes.")

    hist = compute_wave_history(selected_wave, mode=mode, days=min(730, max(120, history_days)))
    if hist is None or hist.empty or "wave_ret" not in hist.columns:
        st.warning("Wave history unavailable.")
    else:
        nav_w = hist["wave_nav"]
        nav_b = hist["bm_nav"]
        r_w = hist["wave_ret"].astype(float)
        r_b = hist["bm_ret"].astype(float)

        dd_w = drawdown_series(nav_w)
        dd_b = drawdown_series(nav_b)

        ra = rolling_alpha_from_nav(nav_w, nav_b, window=int(roll_alpha_window))
        rv = rolling_vol(r_w, window=int(roll_vol_window))

        pers = alpha_persistence(ra)

        te = tracking_error(r_w, r_b)
        ir = information_ratio(nav_w, nav_b, te)
        b = beta_ols(r_w, r_b)
        sh = sharpe_ratio(r_w, rf_annual=0.0)
        so = sortino_ratio(r_w, mar_annual=0.0)
        dd = downside_deviation(r_w, mar_annual=0.0)
        v95, c95 = var_cvar(r_w, level=0.95)
        sk, ku = skew_kurt(r_w)

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Sharpe", fmt_num(sh, 2))
        c2.metric("Sortino", fmt_num(so, 2))
        c3.metric("Downside Dev", fmt_pct(dd, 2))
        c4.metric("Beta vs BM", fmt_num(b, 2))
        c5.metric("Alpha Persistence", fmt_pct(pers, 0))

        c6, c7, c8, c9 = st.columns(4)
        c6.metric("Tracking Error", fmt_pct(te, 2))
        c7.metric("Information Ratio", fmt_num(ir, 2))
        c8.metric("VaR 95% (daily)", fmt_pct(v95, 2))
        c9.metric("CVaR 95% (daily)", fmt_pct(c95, 2))

        st.markdown("---")

        st.markdown("### Rolling Alpha & Rolling Volatility")
        if go is not None and (not ra.empty or not rv.empty):
            fig = go.Figure()
            if not ra.empty:
                fig.add_trace(go.Scatter(x=ra.index, y=ra.values, name=f"Rolling α ({roll_alpha_window}D)", mode="lines"))
            if not rv.empty:
                fig.add_trace(go.Scatter(x=rv.index, y=rv.values, name=f"Rolling Vol ({roll_vol_window}D, ann.)", mode="lines", yaxis="y2"))
            fig.update_layout(
                height=420,
                margin=dict(l=40, r=40, t=40, b=40),
                yaxis=dict(title="Rolling Alpha"),
                yaxis2=dict(title="Rolling Vol (ann.)", overlaying="y", side="right"),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            if not ra.empty:
                st.line_chart(ra.rename("rolling_alpha"))
            if not rv.empty:
                st.line_chart(rv.rename("rolling_vol"))

        st.markdown("---")

        st.markdown("### Drawdown Monitor")
        if go is not None and (not dd_w.empty or not dd_b.empty):
            fig = go.Figure()
            if not dd_w.empty:
                fig.add_trace(go.Scatter(x=dd_w.index, y=dd_w.values, name="Wave DD", mode="lines"))
            if not dd_b.empty:
                fig.add_trace(go.Scatter(x=dd_b.index, y=dd_b.values, name="Benchmark DD", mode="lines"))
            fig.update_layout(
                height=380,
                margin=dict(l=40, r=40, t=40, b=40),
                yaxis=dict(title="Drawdown"),
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            dd_show = pd.DataFrame({"Wave_DD": dd_w, "BM_DD": dd_b}).dropna(how="all")
            if not dd_show.empty:
                st.line_chart(dd_show)

        st.markdown("### Risk Summary Table")
        risk_tbl = pd.DataFrame(
            [
                {"Metric": "MaxDD (Wave)", "Value": fmt_pct(float(dd_w.min()) if len(dd_w) else np.nan)},
                {"Metric": "MaxDD (BM)", "Value": fmt_pct(float(dd_b.min()) if len(dd_b) else np.nan)},
                {"Metric": "Skew", "Value": fmt_num(sk, 2)},
                {"Metric": "Kurtosis", "Value": fmt_num(ku, 2)},
            ]
        )
        st.dataframe(risk_tbl, use_container_width=True)

# ============================================================
# TAB 4: Correlation
# ============================================================
with tab_corr:
    st.subheader("🔗 Correlation (All Waves)")
    st.caption("Daily return correlations across Waves for the selected mode. Useful for multi-wave construction & diversification.")

    ret_panel = build_returns_panel(all_waves, mode=mode, days=min(365, max(120, history_days)))
    if ret_panel is None or ret_panel.empty or ret_panel.shape[1] < 3:
        st.warning("Not enough aligned wave return series to build correlations.")
    else:
        corr_df = corr_matrix_from_returns(ret_panel)
        plot_corr_heatmap(corr_df, title=f"Wave Correlation Matrix — Mode: {mode}")

        st.markdown("---")
        st.markdown(f"### Selected Wave Correlation — {selected_wave}")

        if selected_wave in corr_df.columns:
            s = corr_df[selected_wave].dropna().sort_values(ascending=False)
            top = s.drop(labels=[selected_wave], errors="ignore").head(8)
            low = s.drop(labels=[selected_wave], errors="ignore").tail(8)

            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Most Correlated**")
                st.dataframe(top.rename("ρ").reset_index().rename(columns={"index": "Wave"}), use_container_width=True)
            with c2:
                st.markdown("**Least Correlated**")
                st.dataframe(low.rename("ρ").reset_index().rename(columns={"index": "Wave"}), use_container_width=True)
        else:
            st.info("Selected wave not found in correlation table (insufficient history alignment).")

        st.markdown("---")
        st.markdown("### Correlation vs Key ETFs (Optional)")
        px = fetch_prices_daily(["SPY", "QQQ", "IWM", "TLT", "GLD", "BTC-USD"], days=min(365, max(120, history_days)))
        if px is None or px.empty or "SPY" not in px.columns:
            st.info("ETF correlation panel unavailable (price fetch issue).")
        else:
            etf_ret = px.pct_change().fillna(0.0)
            wave_h = compute_wave_history(selected_wave, mode=mode, days=min(365, max(120, history_days)))
            if wave_h is None or wave_h.empty:
                st.info("Selected wave returns unavailable.")
            else:
                wret = wave_h["wave_ret"].reindex(etf_ret.index).dropna()
                rows = []
                for c in etf_ret.columns:
                    x = etf_ret[c].reindex(wret.index)
                    rho = float(pd.concat([wret.rename("w"), x.rename("x")], axis=1).dropna().corr().iloc[0, 1]) if len(wret) > 30 else np.nan
                    rows.append({"Asset": c, "ρ (daily)": rho})
                dfc = pd.DataFrame(rows)
                dfc["ρ (daily)"] = dfc["ρ (daily)"].apply(lambda v: fmt_num(v, 2))
                st.dataframe(dfc, use_container_width=True)

# ============================================================
# TAB 5: Vector OS Insight Layer
# ============================================================
with tab_vector:
    st.subheader("🤖 Vector OS Insight Layer")
    st.caption("Narrative interpretation (non-advice).")

    wd2 = wave_doctor_assess(selected_wave, mode=mode, days=365, alpha_warn=alpha_warn, te_warn=te_warn)
    attrib2 = compute_alpha_attribution(selected_wave, mode=mode, days=365)

    st.markdown("### Vector Summary")
    st.write(f"**Wave:** {selected_wave}  |  **Mode:** {mode}  |  **Regime:** {reg_now}  |  **Benchmark:** {bar_src}")
    st.write(f"**Benchmark Snapshot:** {bm_sid}  |  **Drift:** {drift_chip}  |  **Coverage Score:** {cov_chip}")

    if wd2.get("ok", False):
        m = wd2["metrics"]
        st.markdown(
            f"""
- **365D Return:** {fmt_pct(m["Return_365D"])}
- **365D Alpha:** {fmt_pct(m["Alpha_365D"])}
- **Tracking Error:** {fmt_pct(m["TE"])}  |  **IR:** {fmt_num(m["IR"], 2)}
- **Beta vs Benchmark:** {fmt_num(m["Beta_vs_BM"], 2)}
- **Sharpe / Sortino:** {fmt_num(m["Sharpe_0rf"], 2)} / {fmt_num(m["Sortino_0mar"], 2)}
- **VaR / CVaR (95% daily):** {fmt_pct(m["VaR95_daily"])} / {fmt_pct(m["CVaR95_daily"])}
- **Max Drawdown:** {fmt_pct(m["MaxDD_Wave"])} (Wave) vs {fmt_pct(m["MaxDD_Benchmark"])} (BM)
"""
        )
        if wd2.get("flags"):
            st.warning("Flags: " + " | ".join(wd2["flags"]))

    st.markdown("### Attribution Lens")
    if attrib2:
        st.write(f"- **Engine Return:** {fmt_pct(attrib2.get('Engine Return'))}")
        st.write(f"- **Static Basket Return:** {fmt_pct(attrib2.get('Static Basket Return'))}")
        st.write(f"- **Overlay Contribution:** {fmt_pct(attrib2.get('Overlay Contribution (Engine - Static)'))}")
        st.write(f"- **Alpha vs Benchmark:** {fmt_pct(attrib2.get('Alpha vs Benchmark'))}")
        st.write(f"- **Benchmark Difficulty (BM - SPY):** {fmt_pct(attrib2.get('Benchmark Difficulty (BM - SPY)'))}")

    st.markdown("### Vector Guidance (Non-Advice)")
    st.write(
        "Vector suggests validating benchmark stability (Benchmark Snapshot ID) and coverage integrity before interpreting extreme alpha. "
        "Then use the heatmap + overview grid to identify persistent alpha across multiple windows. "
        "Use Risk Lab to sanity-check: TE, rolling alpha stability, and fat tails (CVaR/skew). "
        "Use Correlation to avoid stacking highly similar Waves when building multi-wave portfolios."
    )

    st.markdown("---")
    st.caption("Disclosure: This console is for informational / research purposes only and is not investment advice.")

# ============================================================
# Footer / Diagnostics (non-invasive)
# ============================================================
with st.expander("⚙️ Diagnostics (safe)", expanded=False):
    st.write("**Engine module:**", getattr(we, "__name__", "waves_engine"))
    st.write("**yfinance available:**", bool(yf is not None))
    st.write("**plotly available:**", bool(go is not None))
    st.write("**Waves discovered:**", len(all_waves))
    st.write("**Modes discovered:**", len(all_modes))
    st.write("**Selected:**", {"wave": selected_wave, "mode": mode})