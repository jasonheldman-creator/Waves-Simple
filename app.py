# app.py — WAVES Intelligence™ Institutional Console (Vector OS Edition)
# FULL PRODUCTION FILE (NO PATCHES) — DIAGNOSTICS++ BUILD (HARDENED)
#
# Keeps ALL IRB-1 features:
#   • Benchmark Truth (benchmark mix + difficulty vs SPY)
#   • Mode Separation Proof (mode shown + independent history per mode)
#   • Alpha Attribution (Engine vs Static Basket)
#   • Wave Doctor + What-If Lab (shadow simulation)
#   • Top-10 holdings with Google quote links
#   • Market Intel panel (SPY/QQQ/IWM/TLT/GLD/BTC/VIX/TNX)
#   • Factor Decomposition (simple regression beta map)
#   • Vector OS Insight Layer (narrative + flags)
#
# Adds Diagnostics++:
#   • Mode Separation Proof: side-by-side metrics + NAV overlay across ALL modes
#   • Rolling diagnostics: Rolling Alpha / TE / Beta / Vol + alpha persistence
#   • Beta discipline + drift flags (if engine provides beta target)
#   • Correlation matrix across waves (returns) — robust overlap handling
#   • Data quality / coverage audit + flags
#   • Alpha Captured daily (wave_ret - bm_ret, exposure-scaled if engine provides)
#   • Crash-proof fallbacks (Plotly/yfinance optional; no hard dependency)
#
# Notes:
#   • Does NOT modify engine math or baseline results.
#   • What-If Lab is explicitly labeled “shadow simulation”.
#   • This file is intentionally verbose for diagnostics + transparency.

from __future__ import annotations

import math
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

import waves_engine as we  # your engine module

# Optional deps (never crash if missing)
try:
    import yfinance as yf
except Exception:
    yf = None

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

.waves-hdr { font-weight: 800; letter-spacing: 0.2px; margin-bottom: 4px; }

div[data-testid="stDataFrame"] { border-radius: 12px; overflow: hidden; }

@media (max-width: 700px) {
  .block-container { padding-left: 0.8rem; padding-right: 0.8rem; }
}
</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# Helpers: formatting + safety
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


def google_quote_url(ticker: str) -> str:
    t = str(ticker).replace(" ", "")
    return f"https://www.google.com/finance/quote/{t}"


def safe_div(a: float, b: float) -> float:
    try:
        a = float(a)
        b = float(b)
        if not math.isfinite(a) or not math.isfinite(b) or b == 0:
            return float("nan")
        return a / b
    except Exception:
        return float("nan")


# ============================================================
# Data fetching (optional yfinance) + caching
# ============================================================
@st.cache_data(show_spinner=False)
def fetch_prices_daily(tickers: List[str], days: int = 365) -> pd.DataFrame:
    if yf is None or not tickers:
        return pd.DataFrame()

    end = datetime.utcnow().date()
    start = end - timedelta(days=int(days) + 260)

    data = yf.download(
        tickers=sorted(list(set(tickers))),
        start=start.isoformat(),
        end=end.isoformat(),
        interval="1d",
        auto_adjust=True,
        progress=False,
        group_by="column",
    )

    # Normalize yfinance shapes
    if isinstance(data, pd.Series):
        data = data.to_frame()

    if isinstance(data.columns, pd.MultiIndex):
        lvl0 = set(data.columns.get_level_values(0))
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
    if len(data) > days:
        data = data.iloc[-days:]
    return data


@st.cache_data(show_spinner=False)
def fetch_spy_vix(days: int = 365) -> pd.DataFrame:
    return fetch_prices_daily(["SPY", "^VIX"], days=days)


@st.cache_data(show_spinner=False)
def fetch_market_assets(days: int = 365) -> pd.DataFrame:
    return fetch_prices_daily(["SPY", "QQQ", "IWM", "TLT", "GLD", "BTC-USD", "^VIX", "^TNX"], days=days)


# ============================================================
# Engine wrappers (cached)
# ============================================================
@st.cache_data(show_spinner=False)
def compute_wave_history(wave_name: str, mode: str, days: int = 365) -> pd.DataFrame:
    try:
        df = we.compute_history_nav(wave_name, mode=mode, days=days)
        if df is None:
            return pd.DataFrame(columns=["wave_nav", "bm_nav", "wave_ret", "bm_ret"])
        # Ensure expected columns exist
        for c in ["wave_nav", "bm_nav", "wave_ret", "bm_ret"]:
            if c not in df.columns:
                df[c] = np.nan
        return df.copy()
    except Exception:
        return pd.DataFrame(columns=["wave_nav", "bm_nav", "wave_ret", "bm_ret"])


@st.cache_data(show_spinner=False)
def get_benchmark_mix() -> pd.DataFrame:
    try:
        df = we.get_benchmark_mix_table()
        if df is None:
            return pd.DataFrame(columns=["Wave", "Ticker", "Name", "Weight"])
        return df.copy()
    except Exception:
        return pd.DataFrame(columns=["Wave", "Ticker", "Name", "Weight"])


@st.cache_data(show_spinner=False)
def get_wave_holdings(wave_name: str) -> pd.DataFrame:
    try:
        df = we.get_wave_holdings(wave_name)
        if df is None:
            return pd.DataFrame(columns=["Ticker", "Name", "Weight"])
        return df.copy()
    except Exception:
        return pd.DataFrame(columns=["Ticker", "Name", "Weight"])


def _safe_get_all_waves() -> List[str]:
    try:
        wv = we.get_all_waves()
        if wv is None:
            return []
        return list(wv)
    except Exception:
        return []


def _safe_get_modes() -> List[str]:
    try:
        md = we.get_modes()
        if md is None:
            return ["Standard", "Alpha-Minus-Beta", "Private Logic"]
        return list(md)
    except Exception:
        return ["Standard", "Alpha-Minus-Beta", "Private Logic"]


# ============================================================
# Core metrics (robust)
# ============================================================
def ret_from_nav(nav: pd.Series, window: int) -> float:
    nav = safe_series(nav)
    if len(nav) < 2:
        return float("nan")
    window = int(min(window, len(nav)))
    if window < 2:
        return float("nan")
    sub = nav.iloc[-window:]
    start = float(sub.iloc[0])
    end = float(sub.iloc[-1])
    if not math.isfinite(start) or start <= 0 or not math.isfinite(end):
        return float("nan")
    return (end / start) - 1.0


def annualized_vol(daily_ret: pd.Series) -> float:
    r = safe_series(daily_ret)
    if len(r) < 2:
        return float("nan")
    v = float(r.std())
    if not math.isfinite(v):
        return float("nan")
    return v * float(np.sqrt(252))


def max_drawdown(nav: pd.Series) -> float:
    nav = safe_series(nav)
    if len(nav) < 2:
        return float("nan")
    running_max = nav.cummax()
    dd = (nav / running_max) - 1.0
    m = float(dd.min())
    return m if math.isfinite(m) else float("nan")


def tracking_error(wave_ret: pd.Series, bm_ret: pd.Series) -> float:
    w = safe_series(wave_ret)
    b = safe_series(bm_ret)
    df = pd.concat([w.rename("w"), b.rename("b")], axis=1).dropna()
    if df.shape[0] < 2:
        return float("nan")
    diff = df["w"] - df["b"]
    v = float(diff.std())
    if not math.isfinite(v):
        return float("nan")
    return v * float(np.sqrt(252))


def information_ratio(nav_wave: pd.Series, nav_bm: pd.Series, te: float) -> float:
    if te is None or not math.isfinite(float(te)) or float(te) <= 0:
        return float("nan")
    rw = ret_from_nav(nav_wave, window=len(nav_wave))
    rb = ret_from_nav(nav_bm, window=len(nav_bm))
    if not math.isfinite(rw) or not math.isfinite(rb):
        return float("nan")
    return float((rw - rb) / float(te))


def beta_vs_benchmark(wave_ret: pd.Series, bm_ret: pd.Series) -> float:
    w = safe_series(wave_ret)
    b = safe_series(bm_ret)
    df = pd.concat([w.rename("w"), b.rename("b")], axis=1).dropna()
    if df.shape[0] < 20:
        return float("nan")
    var_b = float(df["b"].var())
    if not math.isfinite(var_b) or var_b <= 0:
        return float("nan")
    cov = float(df["w"].cov(df["b"]))
    if not math.isfinite(cov):
        return float("nan")
    return float(cov / var_b)


def regress_factors(wave_ret: pd.Series, factor_ret: pd.DataFrame) -> Dict[str, float]:
    wave_ret = safe_series(wave_ret)
    if wave_ret.empty or factor_ret is None or factor_ret.empty:
        return {col: float("nan") for col in (factor_ret.columns if factor_ret is not None else [])}

    df = pd.concat([wave_ret.rename("wave"), factor_ret], axis=1).dropna()
    if df.shape[0] < 60 or df.shape[1] < 2:
        return {col: float("nan") for col in factor_ret.columns}

    y = df["wave"].values
    X = df[factor_ret.columns].values
    X_design = np.concatenate([np.ones((X.shape[0], 1)), X], axis=1)

    try:
        beta, *_ = np.linalg.lstsq(X_design, y, rcond=None)
        betas = beta[1:]
        return {col: float(b) for col, b in zip(factor_ret.columns, betas)}
    except Exception:
        return {col: float("nan") for col in factor_ret.columns}


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


def get_beta_target_if_available(mode: str) -> float:
    candidates = ["MODE_BETA_TARGET", "BETA_TARGET_BY_MODE", "BETA_TARGETS", "BETA_TARGET"]
    for name in candidates:
        try:
            obj = getattr(we, name, None)
            if isinstance(obj, dict) and mode in obj:
                return float(obj[mode])
        except Exception:
            pass
    return float("nan")


def get_exposure_series_if_available(wave_name: str, mode: str, days: int) -> Optional[pd.Series]:
    """
    Best-effort: if engine can provide a realized exposure series used in alpha captured scaling.
    We DO NOT require it; we attempt common method/attr names.
    """
    candidates = ["get_exposure_series", "compute_exposure_series", "exposure_series"]
    for name in candidates:
        try:
            fn = getattr(we, name, None)
            if callable(fn):
                s = fn(wave_name, mode=mode, days=days)
                if isinstance(s, pd.Series) and len(s) > 0:
                    return s.copy()
        except Exception:
            continue
    return None