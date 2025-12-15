# app.py — WAVES Intelligence™ Institutional Console (Vector OS Edition)
# FULL PRODUCTION FILE (NO PATCHES) — "MEAT RESTORE" BUILD
#
# Restores/keeps:
#   ✅ Robust history loader (engine multi-signature + mode aliases + wave_history.csv fallback)
#   ✅ Sticky summary bar
#   ✅ Alpha Heatmap
#   ✅ Risk Lab (Sharpe/Sortino/VaR/CVaR/Drawdown/Rolling)
#   ✅ Correlation matrix (raw series hidden by default to reduce surface area)
#   ✅ WaveScore leaderboard (console-side approximation)
#   ✅ Vector OS Insight Layer
#
# Adds back the "meat":
#   ✅ Market Intel panel (SPY/QQQ/IWM/TLT/GLD/BTC/VIX/TNX) when data available
#   ✅ Benchmark Truth panel (mix + drift + difficulty signals)
#   ✅ Mode Separation Proof (flags suspiciously identical mode histories)
#   ✅ Alpha Attribution (Engine vs Static Basket from holdings)
#   ✅ Wave Doctor + What-If Lab (shadow simulation; does not modify engine)
#   ✅ Factor Decomposition (multi-factor regression if yfinance available)
#
# Notes:
#   • Engine math is NOT modified.
#   • This app reads history from engine first; falls back to wave_history.csv.
#   • "Expose raw series" stays OFF by default for demos & reduced reverse-engineer surface.

from __future__ import annotations

import os
import math
import hashlib
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

# -------------------------------
# Optional libs
# -------------------------------
try:
    import yfinance as yf
except Exception:
    yf = None

try:
    import plotly.graph_objects as go
except Exception:
    go = None

# -------------------------------
# Engine import (guarded)
# -------------------------------
ENGINE_IMPORT_ERROR = None
try:
    import waves_engine as we  # your engine module
except Exception as e:
    we = None
    ENGINE_IMPORT_ERROR = e


# ============================================================
# MODE ALIASES (critical)
# ============================================================
MODE_ALIASES: Dict[str, List[str]] = {
    "Standard": ["Standard", "standard", "STANDARD", "Base", "BASE", "Normal", "NORMAL"],
    "Alpha-Minus-Beta": [
        "Alpha-Minus-Beta", "alpha-minus-beta", "ALPHA-MINUS-BETA",
        "Alpha Minus Beta", "alpha minus beta", "AMB", "amb"
    ],
    "Private Logic": [
        "Private Logic", "private logic", "PRIVATE LOGIC",
        "Private Logic™", "Private Logic Enhanced", "Private Logic Enhanced™",
        "PLE", "ple"
    ],
}

def mode_candidates(selected_mode: str) -> List[str]:
    cands = MODE_ALIASES.get(selected_mode, [selected_mode])
    seen = set()
    out: List[str] = []
    for m in cands + [selected_mode]:
        ms = str(m).strip()
        if ms and ms not in seen:
            out.append(ms)
            seen.add(ms)
    return out


# ============================================================
# Streamlit config
# ============================================================
st.set_page_config(
    page_title="WAVES Intelligence™ Console",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# Global UI CSS
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

@media (max-width: 700px) {
  .block-container { padding-left: 0.8rem; padding-right: 0.8rem; }
}
</style>
""",
    unsafe_allow_html=True,
)

# ============================================================
# Helpers: formatting / safety
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

def _loose_eq(a: str, b: str) -> bool:
    return str(a).strip().lower() == str(b).strip().lower()


# ============================================================
# Basic return/risk math
# ============================================================
def ret_from_nav(nav: pd.Series, window: int) -> float:
    nav = safe_series(nav).astype(float)
    if len(nav) < 2:
        return float("nan")
    window = max(2, min(int(window), len(nav)))
    sub = nav.iloc[-window:]
    start = float(sub.iloc[0])
    end = float(sub.iloc[-1])
    if start <= 0:
        return float("nan")
    return (end / start) - 1.0

def annualized_vol(daily_ret: pd.Series) -> float:
    daily_ret = safe_series(daily_ret).astype(float)
    if len(daily_ret) < 2:
        return float("nan")
    return float(daily_ret.std() * np.sqrt(252))

def max_drawdown(nav: pd.Series) -> float:
    nav = safe_series(nav).astype(float)
    if len(nav) < 2:
        return float("nan")
    running_max = nav.cummax()
    dd = (nav / running_max) - 1.0
    return float(dd.min())

def tracking_error(daily_wave: pd.Series, daily_bm: pd.Series) -> float:
    daily_wave = safe_series(daily_wave).astype(float)
    daily_bm = safe_series(daily_bm).astype(float)
    df = pd.concat([daily_wave.rename("w"), daily_bm.rename("b")], axis=1).dropna()
    if df.shape[0] < 2:
        return float("nan")
    diff = df["w"] - df["b"]
    return float(diff.std() * np.sqrt(252))

def information_ratio(nav_wave: pd.Series, nav_bm: pd.Series, te: float) -> float:
    nav_wave = safe_series(nav_wave).astype(float)
    nav_bm = safe_series(nav_bm).astype(float)
    if len(nav_wave) < 2 or len(nav_bm) < 2:
        return float("nan")
    if te is None or (isinstance(te, float) and (math.isnan(te) or te <= 0)):
        return float("nan")
    excess = ret_from_nav(nav_wave, len(nav_wave)) - ret_from_nav(nav_bm, len(nav_bm))
    return float(excess / te)

def beta_ols(y: pd.Series, x: pd.Series) -> float:
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
    r = safe_series(daily_ret).astype(float)
    if len(r) < 50:
        return (float("nan"), float("nan"))
    q = float(np.quantile(r.values, 1.0 - level))
    tail = r[r <= q]
    cvar = float(tail.mean()) if len(tail) else float("nan")
    return (q, cvar)

def skew_kurt(daily_ret: pd.Series) -> Tuple[float, float]:
    r = safe_series(daily_ret).astype(float)
    if len(r) < 50:
        return (float("nan"), float("nan"))
    sk = float(pd.Series(r).skew())
    ku = float(pd.Series(r).kurtosis())
    return (sk, ku)

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
    return (df.iloc[:, 0] - df.iloc[:, 1]).rename(f"alpha_{window}")

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
    return ((nav / peak) - 1.0).rename("drawdown")

def alpha_persistence(alpha_series: pd.Series) -> float:
    a = safe_series(alpha_series).dropna()
    if len(a) < 30:
        return float("nan")
    return float((a > 0).mean())


# ============================================================
# Data fetch (optional via yfinance)
# ============================================================
@st.cache_data(show_spinner=False)
def fetch_prices_daily(tickers: List[str], days: int = 365) -> pd.DataFrame:
    if yf is None or not tickers:
        return pd.DataFrame()

    end = datetime.utcnow().date()
    start = end - timedelta(days=days + 260)

    data = yf.download(
        tickers=sorted(list(set([t for t in tickers if t]))),
        start=start.isoformat(),
        end=end.isoformat(),
        interval="1d",
        auto_adjust=True,
        progress=False,
        group_by="column",
        threads=True,
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

    data.index = pd.to_datetime(data.index, errors="coerce")
    data = data[~data.index.isna()].sort_index().ffill().bfill()

    if len(data) > days:
        data = data.iloc[-days:]
    return data

@st.cache_data(show_spinner=False)
def market_intel_snapshot() -> pd.DataFrame:
    """
    Light market intel for demo: uses yfinance if available.
    """
    cols = ["Ticker", "Last", "1D", "30D"]
    tickers = ["SPY", "QQQ", "IWM", "TLT", "GLD", "BTC-USD", "^VIX", "^TNX"]
    if yf is None:
        return pd.DataFrame(columns=cols)

    px = fetch_prices_daily(tickers, days=60)
    if px.empty:
        return pd.DataFrame(columns=cols)

    rows = []
    for t in tickers:
        if t not in px.columns:
            continue
        s = px[t].dropna()
        if len(s) < 2:
            continue
        last = float(s.iloc[-1])
        r1 = float(s.iloc[-1] / s.iloc[-2] - 1.0) if len(s) >= 2 else np.nan
        r30 = float(s.iloc[-1] / s.iloc[-31] - 1.0) if len(s) >= 31 else np.nan
        rows.append({"Ticker": t, "Last": last, "1D": r1, "30D": r30})

    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=cols)

    df["Last"] = df["Last"].map(lambda x: fmt_num(x, 2))
    df["1D"] = df["1D"].map(lambda x: fmt_pct(x, 2))
    df["30D"] = df["30D"].map(lambda x: fmt_pct(x, 2))
    return df[cols]


# ============================================================
# HISTORY LOADER (engine → CSV fallback)
# ============================================================
def _standardize_history(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["wave_nav", "bm_nav", "wave_ret", "bm_ret"])

    out = df.copy()
    for dc in ["date", "Date", "timestamp", "Timestamp", "datetime", "Datetime"]:
        if dc in out.columns:
            out[dc] = pd.to_datetime(out[dc], errors="coerce")
            out = out.dropna(subset=[dc]).set_index(dc)
            break

    out.index = pd.to_datetime(out.index, errors="coerce")
    out = out[~out.index.isna()].sort_index()

    ren = {}
    for c in out.columns:
        cs = str(c).strip()
        low = cs.lower()
        if low in ["wave_nav", "nav_wave", "portfolio_nav", "wave value", "wavevalue", "nav", "portfolio_nav_usd"]:
            ren[c] = "wave_nav"
        if low in ["bm_nav", "bench_nav", "benchmark_nav", "benchmark value", "bm value", "benchmark_nav_usd"]:
            ren[c] = "bm_nav"
        if low in ["wave_ret", "ret_wave", "portfolio_ret", "wave return", "portfolio_return"]:
            ren[c] = "wave_ret"
        if low in ["bm_ret", "ret_bm", "benchmark_ret", "benchmark return", "benchmark_return"]:
            ren[c] = "bm_ret"

    out = out.rename(columns=ren)

    if "wave_ret" not in out.columns and "wave_nav" in out.columns:
        out["wave_ret"] = pd.to_numeric(out["wave_nav"], errors="coerce").pct_change()
    if "bm_ret" not in out.columns and "bm_nav" in out.columns:
        out["bm_ret"] = pd.to_numeric(out["bm_nav"], errors="coerce").pct_change()

    for col in ["wave_nav", "bm_nav", "wave_ret", "bm_ret"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    out = out[["wave_nav", "bm_nav", "wave_ret", "bm_ret"]].dropna(how="all")
    return out

@st.cache_data(show_spinner=False)
def load_wave_history_csv(path: str = "wave_history.csv") -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()

def history_from_csv(wave_name: str, mode: str, days: int) -> pd.DataFrame:
    raw = load_wave_history_csv("wave_history.csv")
    if raw is None or raw.empty:
        return pd.DataFrame(columns=["wave_nav", "bm_nav", "wave_ret", "bm_ret"])

    df = raw.copy()
    df.columns = [str(c).strip() for c in df.columns]

    wave_cols = [c for c in df.columns if c.lower() in ["wave", "wave_name", "wavename"]]
    mode_cols = [c for c in df.columns if c.lower() in ["mode", "risk_mode", "strategy_mode"]]
    date_cols = [c for c in df.columns if c.lower() in ["date", "timestamp", "datetime"]]

    wc = wave_cols[0] if wave_cols else None
    mc = mode_cols[0] if mode_cols else None
    dc = date_cols[0] if date_cols else None

    if wc:
        df[wc] = df[wc].astype(str)
        df = df[df[wc].astype(str).apply(lambda x: _loose_eq(x, wave_name))]

    if mc:
        df[mc] = df[mc].astype(str)
        cands = mode_candidates(mode)
        df = df[df[mc].astype(str).apply(lambda x: any(_loose_eq(x, m) for m in cands))]

    if dc:
        df[dc] = pd.to_datetime(df[dc], errors="coerce")
        df = df.dropna(subset=[dc]).sort_values(dc).set_index(dc)

    out = _standardize_history(df)
    if len(out) > days:
        out = out.iloc[-days:]
    return out

@st.cache_data(show_spinner=False)
def compute_wave_history(wave_name: str, mode: str, days: int = 365, force_csv: bool = False) -> pd.DataFrame:
    if force_csv:
        return history_from_csv(wave_name, mode, days)

    if we is None:
        return history_from_csv(wave_name, mode, days)

    cands = mode_candidates(mode)

    # 1) preferred function
    try:
        if hasattr(we, "compute_history_nav"):
            for m in cands:
                try:
                    df = we.compute_history_nav(wave_name, mode=m, days=days)
                    df = _standardize_history(df)
                    if not df.empty:
                        return df
                except TypeError:
                    try:
                        df = we.compute_history_nav(wave_name, m, days)
                        df = _standardize_history(df)
                        if not df.empty:
                            return df
                    except Exception:
                        pass
                except Exception:
                    pass
    except Exception:
        pass

    # 2) alternate names
    candidates = ["get_history_nav", "get_wave_history", "history_nav", "compute_nav_history", "compute_history"]
    for fn in candidates:
        if hasattr(we, fn):
            f = getattr(we, fn)
            for m in cands:
                try:
                    try:
                        df = f(wave_name, mode=m, days=days)
                    except TypeError:
                        df = f(wave_name, m, days)
                    df = _standardize_history(df)
                    if not df.empty:
                        return df
                except Exception:
                    continue

    return history_from_csv(wave_name, mode, days)

@st.cache_data(show_spinner=False)
def get_all_waves_safe() -> List[str]:
    if we is not None and hasattr(we, "get_all_waves"):
        try:
            waves = we.get_all_waves()
            if isinstance(waves, (list, tuple)):
                out = [str(x).strip() for x in waves]
                out = [w for w in out if w and w.lower() != "nan"]
                if out:
                    return sorted(out)
        except Exception:
            pass

    for p in ["wave_config.csv", "wave_weights.csv", "list.csv"]:
        if os.path.exists(p):
            try:
                df = pd.read_csv(p)
                for col in ["Wave", "wave", "wave_name", "wavename"]:
                    if col in df.columns:
                        waves = sorted(list(set(df[col].astype(str).str.strip().tolist())))
                        waves = [w for w in waves if w and w.lower() != "nan"]
                        if waves:
                            return waves
            except Exception:
                pass
    return []

@st.cache_data(show_spinner=False)
def get_benchmark_mix() -> pd.DataFrame:
    if we is None:
        return pd.DataFrame(columns=["Wave", "Ticker", "Name", "Weight"])
    if hasattr(we, "get_benchmark_mix_table"):
        try:
            df = we.get_benchmark_mix_table()
            if isinstance(df, pd.DataFrame):
                return df
        except Exception:
            pass
    return pd.DataFrame(columns=["Wave", "Ticker", "Name", "Weight"])

@st.cache_data(show_spinner=False)
def get_wave_holdings(wave_name: str) -> pd.DataFrame:
    if we is not None and hasattr(we, "get_wave_holdings"):
        try:
            df = we.get_wave_holdings(wave_name)
            if isinstance(df, pd.DataFrame):
                return df
        except Exception:
            pass

    if os.path.exists("wave_weights.csv"):
        try:
            df = pd.read_csv("wave_weights.csv")
            cols = {c.lower(): c for c in df.columns}
            if {"wave", "ticker", "weight"}.issubset(set(cols.keys())):
                wf = df[df[cols["wave"]].astype(str).apply(lambda x: _loose_eq(x, wave_name))].copy()
                wf["Ticker"] = wf[cols["ticker"]].astype(str).str.strip()
                wf["Weight"] = pd.to_numeric(w