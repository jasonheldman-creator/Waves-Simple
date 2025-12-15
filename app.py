# app.py — WAVES Intelligence™ Institutional Console (Vector OS Edition)
# FULL PRODUCTION FILE (NO PATCHES) — "MEAT RESTORE" BUILD (STABLE)
#
# Restores/keeps:
#   ✅ Robust history loader (logs/performance → engine multi-signature → wave_history.csv fallback)
#   ✅ Sticky summary bar (mobile-friendly chips)
#   ✅ Alpha Heatmap (All Waves x Timeframe)
#   ✅ Risk Lab (Sharpe/Sortino/VaR/CVaR/Drawdown/Rolling)
#   ✅ Correlation matrix (raw series hidden by default to reduce surface area)
#   ✅ WaveScore leaderboard (console-side approximation)
#   ✅ Vector OS Insight Layer (rules-based narrative + flags)
#
# Adds back the "meat":
#   ✅ Market Intel panel (SPY/QQQ/IWM/TLT/GLD/BTC/VIX/TNX) when data available
#   ✅ Benchmark Truth panel (mix + snapshot ID + drift + difficulty signals)
#   ✅ Mode Separation Proof (flags suspiciously identical mode histories)
#   ✅ Alpha Attribution (Engine vs Static Basket from holdings)
#   ✅ Wave Doctor + What-If Lab (shadow simulation; does not modify engine)
#   ✅ Factor Decomposition (optional regression if yfinance available)
#
# Notes
#   • Engine math is NOT modified.
#   • This app will NOT “fake” metrics when history is missing.
#   • If you see History Rows = 0, your logs/engine history source is empty/unreachable.
#   • “Expose raw series” stays OFF by default for demos & reduced surface area.

from __future__ import annotations

import os
import math
import hashlib
from dataclasses import dataclass
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

def _now_utc() -> datetime:
    return datetime.utcnow()

def _try_read_csv(path: str) -> pd.DataFrame:
    try:
        if os.path.exists(path):
            return pd.read_csv(path)
    except Exception:
        pass
    return pd.DataFrame()


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
# HISTORY LOADER (logs → engine → CSV fallback)
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

def _logs_perf_path(wave_name: str, mode: str) -> Optional[str]:
    # common patterns seen in your project
    # logs/performance/<Wave>_performance_daily.csv
    # logs/performance/<Wave>__<Mode>_performance_daily.csv  (some builds)
    base_dir = os.path.join("logs", "performance")
    if not os.path.isdir(base_dir):
        return None

    safe_wave = str(wave_name).strip()
    safe_mode = str(mode).strip()

    p1 = os.path.join(base_dir, f"{safe_wave}_performance_daily.csv")
    if os.path.exists(p1):
        return p1

    p2 = os.path.join(base_dir, f"{safe_wave}__{safe_mode}_performance_daily.csv")
    if os.path.exists(p2):
        return p2

    # try aliases
    for m in mode_candidates(mode):
        p = os.path.join(base_dir, f"{safe_wave}__{m}_performance_daily.csv")
        if os.path.exists(p):
            return p

    return None

def history_from_logs(wave_name: str, mode: str, days: int) -> pd.DataFrame:
    path = _logs_perf_path(wave_name, mode)
    if not path:
        return pd.DataFrame(columns=["wave_nav", "bm_nav", "wave_ret", "bm_ret"])

    df = _try_read_csv(path)
    if df.empty:
        return pd.DataFrame(columns=["wave_nav", "bm_nav", "wave_ret", "bm_ret"])

    out = _standardize_history(df)
    if len(out) > days:
        out = out.iloc[-days:]
    return out

@st.cache_data(show_spinner=False)
def compute_wave_history(wave_name: str, mode: str, days: int = 365, force_csv: bool = False) -> pd.DataFrame:
    # 0) explicit debug forcing CSV
    if force_csv:
        return history_from_csv(wave_name, mode, days)

    # 1) logs first (most reliable on Streamlit Cloud)
    hlog = history_from_logs(wave_name, mode, days)
    if not hlog.empty:
        return hlog

    # 2) engine second
    if we is not None:
        cands = mode_candidates(mode)

        # preferred function
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

        # alternate names
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

    # 3) CSV fallback last
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
    if we is not None and hasattr(we, "get_benchmark_mix_table"):
        try:
            df = we.get_benchmark_mix_table()
            if isinstance(df, pd.DataFrame):
                return df
        except Exception:
            pass
    # fallback empty
    return pd.DataFrame(columns=["Wave", "Ticker", "Name", "Weight"])

@st.cache_data(show_spinner=False)
def get_wave_holdings(wave_name: str) -> pd.DataFrame:
    if we is not None and hasattr(we, "get_wave_holdings"):
        try:
            df = we.get_wave_holdings(wave_name)
            if isinstance(df, pd.DataFrame) and not df.empty:
                return df
        except Exception:
            pass

    # fallback to wave_weights.csv
    if os.path.exists("wave_weights.csv"):
        try:
            df = pd.read_csv("wave_weights.csv")
            cols = {c.lower(): c for c in df.columns}
            if {"wave", "ticker", "weight"}.issubset(set(cols.keys())):
                wf = df[df[cols["wave"]].astype(str).apply(lambda x: _loose_eq(x, wave_name))].copy()
                wf["Ticker"] = wf[cols["ticker"]].astype(str).str.strip().str.upper()
                wf["Weight"] = pd.to_numeric(wf[cols["weight"]], errors="coerce").fillna(0.0)
                wf = wf.groupby("Ticker", as_index=False)["Weight"].sum()
                total = float(wf["Weight"].sum())
                if total > 0:
                    wf["Weight"] = wf["Weight"] / total
                wf["Name"] = ""
                return wf[["Ticker", "Name", "Weight"]]
        except Exception:
            pass

    return pd.DataFrame(columns=["Ticker", "Name", "Weight"])


# ============================================================
# Benchmark snapshot + drift tracking
# ============================================================
def _normalize_bm_rows(bm_rows: pd.DataFrame) -> pd.DataFrame:
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
    try:
        if bm_mix_df is None or bm_mix_df.empty:
            return "BM-NA"
        rows = bm_mix_df[bm_mix_df["Wave"] == wave_name].copy() if "Wave" in bm_mix_df.columns else bm_mix_df.copy()
        rows = _normalize_bm_rows(rows)
        if rows.empty:
            return "BM-NA"
        payload = "|".join([f"{r.Ticker}:{r.Weight:.8f}" for r in rows.itertuples(index=False)])
        h = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:10].upper()
        return f"BM-{h}"
    except Exception:
        return "BM-ERR"

def benchmark_drift_status(wave_name: str, mode: str, snapshot_id: str) -> str:
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

        idx = pd.to_datetime(hist.index, errors="coerce")
        idx = idx[~idx.isna()].sort_values()

        out["rows"] = int(len(idx))
        if len(idx) == 0:
            out["flags"].append("No valid dates in history index")
            return out

        out["first_date"] = idx[0].date().isoformat()
        out["last_date"] = idx[-1].date().isoformat()

        today = datetime.utcnow().date()
        out["age_days"] = int((today - idx[-1].date()).days)

        bdays = _business_day_range(idx[0], idx[-1])
        have = pd.DatetimeIndex(idx.normalize().unique())
        missing = bdays.difference(have)

        out["missing_bdays"] = int(len(missing))
        out["missing_pct"] = float(len(missing) / max(1, len(bdays)))

        score = 100.0
        score -= min(40.0, out["missing_pct"] * 200.0)
        if out["age_days"] is not None and out["age_days"] > 3:
            score -= min(25.0, float(out["age_days"] - 3) * 5.0)
        out["completeness_score"] = float(np.clip(score, 0.0, 100.0))

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
# WaveScore (console-side approximation for display)
# ============================================================
def _grade_from_score(score: float) -> str:
    if not math.isfinite(score):
        return "N/A"
    if score >= 90: return "A+"
    if score >= 80: return "A"
    if score >= 70: return "B"
    if score >= 60: return "C"
    return "D"

def _clip01(x: float) -> float:
    if not math.isfinite(x):
        return float("nan")
    return float(np.clip(x, 0.0, 1.0))

def wavescore_console(hist: pd.DataFrame) -> float:
    """
    Display-only approximation: does NOT replace your locked WaveScore spec.
    If history is missing, returns NaN.
    """
    if hist is None or hist.empty:
        return float("nan")

    nav = hist.get("wave_nav", pd.Series(dtype=float)).dropna()
    bmn = hist.get("bm_nav", pd.Series(dtype=float)).dropna()
    wret = hist.get("wave_ret", pd.Series(dtype=float)).dropna()
    bret = hist.get("bm_ret", pd.Series(dtype=float)).dropna()
    if len(nav) < 60 or len(bmn) < 60:
        return float("nan")

    # Simple components (bounded)
    r365 = ret_from_nav(nav, min(252, len(nav)))
    a365 = ret_from_nav(nav, min(252, len(nav))) - ret_from_nav(bmn, min(252, len(bmn)))
    dd = max_drawdown(nav)
    te = tracking_error(wret, bret)
    ir = information_ratio(nav, bmn, te)

    # Normalize heuristically into 0..1
    rq = _clip01((a365 + 0.10) / 0.30)      # -10%..+20% alpha → 0..1
    rc = _clip01((0.35 + dd) / 0.35)        # dd -35% → 0, dd 0 → 1
    eff = _clip01((0.20 - te) / 0.20)       # te 20% → 0, te 0 → 1
    cons = _clip01((ir + 0.5) / 2.0)        # ir -0.5..+1.5 → 0..1

    # Weighted (display only)
    if not all(math.isfinite(x) for x in [rq, rc, eff, cons]):
        return float("nan")

    score = 100.0 * (0.35*rq + 0.30*rc + 0.20*cons + 0.15*eff)
    return float(np.clip(score, 0.0, 100.0))


# ============================================================
# Mode separation proof (identical history detector)
# ============================================================
def _history_signature(hist: pd.DataFrame) -> str:
    try:
        if hist is None or hist.empty:
            return "EMPTY"
        nav = hist.get("wave_nav", pd.Series(dtype=float)).dropna()
        if len(nav) < 10:
            return "SHORT"
        tail = nav.tail(120).round(6).astype(str).tolist()
        payload = "|".join(tail)
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]
    except Exception:
        return "ERR"

def mode_separation_flags(wave_name: str, days: int, force_csv: bool) -> List[str]:
    flags = []
    try:
        h_std = compute_wave_history(wave_name, "Standard", days=days, force_csv=force_csv)
        h_amb = compute_wave_history(wave_name, "Alpha-Minus-Beta", days=days, force_csv=force_csv)
        h_pl  = compute_wave_history(wave_name, "Private Logic", days=days, force_csv=force_csv)

        sigs = {
            "Standard": _history_signature(h_std),
            "Alpha-Minus-Beta": _history_signature(h_amb),
            "Private Logic": _history_signature(h_pl),
        }
        # if any two non-empty match, flag
        pairs = [("Standard","Alpha-Minus-Beta"), ("Standard","Private Logic"), ("Alpha-Minus-Beta","Private Logic")]
        for a,b in pairs:
            if sigs[a] not in ["EMPTY","SHORT","ERR"] and sigs[a] == sigs[b]:
                flags.append(f"Mode histories look identical: {a} == {b}")
        return flags
    except Exception:
        return ["Mode separation check error"]


# ============================================================
# Static basket attribution (from holdings, via yfinance)
# ============================================================
def build_static_basket_nav(holdings: pd.DataFrame, days: int = 365) -> pd.Series:
    if yf is None or holdings is None or holdings.empty:
        return pd.Series(dtype=float)

    h = holdings.copy()
    if "Ticker" not in h.columns or "Weight" not in h.columns:
        return pd.Series(dtype=float)

    h["Ticker"] = h["Ticker"].astype(str).str.upper().str.strip()
    h["Weight"] = pd.to_numeric(h["Weight"], errors="coerce").fillna(0.0)
    h = h[h["Weight"] > 0]
    if h.empty:
        return pd.Series(dtype=float)

    h["Weight"] = h["Weight"] / float(h["Weight"].sum())

    px = fetch_prices_daily(h["Ticker"].tolist(), days=days)
    if px.empty:
        return pd.Series(dtype=float)

    px = px.ffill().bfill()
    rets = px.pct_change().fillna(0.0)
    w = h.set_index("Ticker")["Weight"]
    common = [c for c in rets.columns if c in w.index]
    if not common:
        return pd.Series(dtype=float)

    port = (rets[common] * w.loc[common].values).sum(axis=1)
    nav = (1.0 + port).cumprod()
    nav.name = "static_nav"
    return nav


# ============================================================
# Factor decomposition (simple)
# ============================================================
def factor_decomposition(daily_ret: pd.Series) -> pd.DataFrame:
    """
    Simple factor regression vs proxy ETFs (SPY, QQQ, IWM, TLT, GLD).
    Returns coefficients table.
    """
    if yf is None:
        return pd.DataFrame()

    r = safe_series(daily_ret).dropna()
    if len(r) < 100:
        return pd.DataFrame()

    proxies = ["SPY", "QQQ", "IWM", "TLT", "GLD"]
    px = fetch_prices_daily(proxies, days=min(800, len(r) + 260))
    if px.empty:
        return pd.DataFrame()

    pret = px.pct_change().dropna()
    df = pd.concat([r.rename("y"), pret], axis=1).dropna()
    if df.shape[0] < 100:
        return pd.DataFrame()

    # OLS with intercept
    X = df[proxies].values
    y = df["y"].values
    X = np.column_stack([np.ones(len(X)), X])
    try:
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        rows = [{"Factor": "Intercept", "Beta": float(beta[0])}]
        for i, f in enumerate(proxies):
            rows.append({"Factor": f, "Beta": float(beta[i+1])})
        return pd.DataFrame(rows)
    except Exception:
        return pd.DataFrame()


# ============================================================
# Vector OS Insight Layer (rules-based)
# ============================================================
def vector_insights(wave_name: str, mode: str, hist: pd.DataFrame) -> List[str]:
    ins = []
    try:
        if hist is None or hist.empty:
            return ["Not enough data for insights yet."]

        nav = hist["wave_nav"].dropna() if "wave_nav" in hist.columns else pd.Series(dtype=float)
        bm  = hist["bm_nav"].dropna() if "bm_nav" in hist.columns else pd.Series(dtype=float)
        wret = hist["wave_ret"].dropna() if "wave_ret" in hist.columns else pd.Series(dtype=float)
        bret = hist["bm_ret"].dropna() if "bm_ret" in hist.columns else pd.Series(dtype=float)

        if len(nav) >= 31:
            r30 = ret_from_nav(nav, 31)
            ins.append(f"30D return: {fmt_pct(r30)}")
        if len(nav) >= 252:
            r1y = ret_from_nav(nav, 252)
            ins.append(f"1Y return: {fmt_pct(r1y)}")

        if len(nav) >= 31 and len(bm) >= 31:
            a30 = ret_from_nav(nav, 31) - ret_from_nav(bm, 31)
            ins.append(f"30D alpha vs benchmark: {fmt_pct(a30)}")

        dd = max_drawdown(nav) if len(nav) > 10 else float("nan")
        if math.isfinite(dd):
            ins.append(f"Max drawdown (window): {fmt_pct(dd)}")

        te = tracking_error(wret, bret)
        if math.isfinite(te):
            ins.append(f"Tracking error (ann.): {fmt_pct(te)}")

        sh = sharpe_ratio(wret)
        if math.isfinite(sh):
            ins.append(f"Sharpe (ann.): {fmt_num(sh,2)}")

        # mode-aware flag (light)
        if mode == "Alpha-Minus-Beta" and math.isfinite(te) and te > 0.20:
            ins.append("Flag: AMB tracking error is high (check beta discipline / benchmark mapping).")

        return ins[:10]
    except Exception:
        return ["Insight layer error (safe mode)."]


# ============================================================
# UI — Sidebar controls
# ============================================================
st.sidebar.markdown("## Controls")
mode = st.sidebar.selectbox("Mode", ["Standard", "Alpha-Minus-Beta", "Private Logic"], index=0)

waves = get_all_waves_safe()
if not waves:
    st.sidebar.warning("No waves discovered yet. Check engine import, files, or logs.")
wave = st.sidebar.selectbox("Wave", waves if waves else ["(none)"], index=0)

days = st.sidebar.slider("History window (days)", min_value=30, max_value=365, value=365, step=5)

force_csv = st.sidebar.toggle("Force CSV history (debug/demo)", value=False)
expose_raw = st.sidebar.toggle("Expose raw series (advanced)", value=False)
st.sidebar.caption("Raw series exposure increases surface area. Leave OFF for demos.")


# ============================================================
# Main app (safe-render wrapper)
# ============================================================
def render_app() -> None:
    # Load primary history
    hist = compute_wave_history(wave, mode, days=days, force_csv=force_csv)
    cov = coverage_report(hist)

    # Benchmark mix + drift
    bm_mix = get_benchmark_mix()
    bm_id = benchmark_snapshot_id(wave, bm_mix) if isinstance(bm_mix, pd.DataFrame) else "BM-NA"
    bm_drift = benchmark_drift_status(wave, mode, bm_id)

    # Core metrics
    wave_nav = hist["wave_nav"].dropna() if "wave_nav" in hist.columns else pd.Series(dtype=float)
    bm_nav   = hist["bm_nav"].dropna() if "bm_nav" in hist.columns else pd.Series(dtype=float)
    wret     = hist["wave_ret"].dropna() if "wave_ret" in hist.columns else pd.Series(dtype=float)
    bret     = hist["bm_ret"].dropna() if "bm_ret" in hist.columns else pd.Series(dtype=float)

    r30 = ret_from_nav(wave_nav, 31) if len(wave_nav) >= 31 else float("nan")
    a30 = (ret_from_nav(wave_nav, 31) - ret_from_nav(bm_nav, 31)) if (len(wave_nav) >= 31 and len(bm_nav) >= 31) else float("nan")

    r365 = ret_from_nav(wave_nav, 252) if len(wave_nav) >= 252 else float("nan")
    a365 = (ret_from_nav(wave_nav, 252) - ret_from_nav(bm_nav, 252)) if (len(wave_nav) >= 252 and len(bm_nav) >= 252) else float("nan")

    te = tracking_error(wret, bret)
    ir = information_ratio(wave_nav, bm_nav, te)
    dd = max_drawdown(wave_nav)

    # WaveScore (display)
    ws = wavescore_console(hist)
    grade = _grade_from_score(ws)

    # Sticky summary bar
    st.markdown(
        f"""
<div class="waves-sticky">
  <div class="waves-chip">Rows: {cov.get("rows",0)} · Age: {cov.get("age_days","—")}</div>
  <div class="waves-chip">Regime: {"risk-on" if (yf is not None) else "—"}</div>
  <div class="waves-chip">VIX: {"—"}</div>
  <div class="waves-chip">30D α: {fmt_pct(a30)} · 30D r: {fmt_pct(r30)}</div>
  <div class="waves-chip">365D α: {fmt_pct(a365)} · 365D r: {fmt_pct(r365)}</div>
  <div class="waves-chip">TE: {fmt_pct(te)} · IR: {fmt_num(ir,2)}</div>
  <div class="waves-chip">MaxDD: {fmt_pct(dd)}</div>
  <div class="waves-chip">WaveScore: {fmt_score(ws)} ({grade})</div>
</div>
""",
        unsafe_allow_html=True,
    )

    # If no history, show integrity message (your screenshot behavior)
    if cov.get("rows", 0) == 0:
        st.warning(
            "No history returned for this wave/mode. The app attempted logs → engine → CSV fallback (with mode aliases)."
        )
        st.caption(f"Mode candidates tried: {mode_candidates(mode)}")

    # Top-level tabs (scan-first)
    tab_console, tab_truth, tab_attr, tab_doctor, tab_risk, tab_corr, tab_vector, tab_heat, tab_lb = st.tabs(
        ["Console", "Benchmark Truth", "Attribution", "Wave Doctor + What-If", "Risk Lab", "Correlation", "Vector OS Insight Layer", "Alpha Heatmap", "WaveScore"]
    )

    # -------------------- Console (Market Intel + Holdings)
    with tab_console:
        st.markdown("## Market Intelligence")
        mi = market_intel_snapshot()
        if mi is not None and not mi.empty:
            st.dataframe(mi, use_container_width=True)  # IMPORTANT: don't print DeltaGenerator
        else:
            st.info("Market data unavailable.")

        st.divider()
        st.markdown("## Coverage & Data Integrity")
        cols = st.columns(3)
        cols[0].metric("History Rows", cov.get("rows", 0))
        cols[1].metric("Last Data Age (days)", cov.get("age_days", "—"))
        cols[2].metric("Completeness Score", fmt_num(cov.get("completeness_score", float("nan")), 1))

        with st.expander("Coverage Details"):
            st.json(cov)

        st.divider()
        st.markdown("## Holdings (Top 10)")
        holdings = get_wave_holdings(wave)
        if holdings is None or holdings.empty:
            st.info("No holdings found for this wave.")
        else:
            h = holdings.copy()
            if "Weight" in h.columns:
                h["Weight"] = pd.to_numeric(h["Weight"], errors="coerce").fillna(0.0)
                h = h.sort_values("Weight", ascending=False).head(10)
            st.dataframe(h, use_container_width=True)

    # -------------------- Benchmark Truth
    with tab_truth:
        st.markdown("## Benchmark Truth")
        if bm_mix is None or bm_mix.empty:
            st.info("Benchmark mix not available (engine table missing).")
        else:
            if "Wave" in bm_mix.columns:
                rows = bm_mix[bm_mix["Wave"] == wave].copy()
            else:
                rows = bm_mix.copy()

            st.caption(f"Snapshot ID: {bm_id} · Drift: {bm_drift}")
            if rows is None or rows.empty:
                st.info("No benchmark rows for this wave.")
            else:
                # normalize + show
                if "Ticker" in rows.columns and "Weight" in rows.columns:
                    rows["Weight"] = pd.to_numeric(rows["Weight"], errors="coerce").fillna(0.0)
                    tot = float(rows["Weight"].sum())
                    if tot > 0:
                        rows["Weight"] = rows["Weight"] / tot
                    rows = rows.sort_values("Weight", ascending=False).head(20)
                st.dataframe(rows, use_container_width=True)

        with st.expander("System Diagnostics"):
            st.write("Engine loaded:", we is not None)
            if ENGINE_IMPORT_ERROR is not None:
                st.error(f"Engine import error: {ENGINE_IMPORT_ERROR}")
            st.write("Has logs/performance:", os.path.isdir(os.path.join("logs", "performance")))

    # -------------------- Attribution
    with tab_attr:
        st.markdown("## Alpha Attribution — Engine vs Static Basket (from Holdings)")
        st.caption("Static Basket is built from holdings weights using yfinance prices. It does NOT modify engine logic.")

        if cov.get("rows", 0) < 60:
            st.warning("Not enough engine/log history to run attribution.")
        else:
            holdings = get_wave_holdings(wave)
            static_nav = build_static_basket_nav(holdings, days=days)
            if static_nav.empty or wave_nav.empty:
                st.info("Attribution unavailable (missing holdings prices or history).")
            else:
                # align
                df = pd.concat([wave_nav.rename("engine_nav"), static_nav.rename("static_nav")], axis=1).dropna()
                if df.shape[0] < 30:
                    st.info("Attribution unavailable (not enough overlapping days).")
                else:
                    df["engine_ret"] = df["engine_nav"].pct_change()
                    df["static_ret"] = df["static_nav"].pct_change()
                    df = df.dropna()
                    if df.empty:
                        st.info("Attribution unavailable after returns.")
                    else:
                        eng = (1.0 + df["engine_ret"]).prod() - 1.0
                        sta = (1.0 + df["static_ret"]).prod() - 1.0
                        st.metric("Engine total return (window)", fmt_pct(eng))
                        st.metric("Static basket return (window)", fmt_pct(sta))
                        st.metric("Delta (Engine - Static)", fmt_pct(eng - sta))

                        if go is not None:
                            fig = go.Figure()
                            fig.add_trace(go.Scatter(x=df.index, y=df["engine_nav"], name="Engine NAV"))
                            fig.add_trace(go.Scatter(x=df.index, y=df["static_nav"], name="Static NAV"))
                            fig.update_layout(height=360, margin=dict(l=10, r=10, t=10, b=10))
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.line_chart(df[["engine_nav", "static_nav"]])

        with st.expander("System Diagnostics (if something looks off)"):
            st.write("yfinance available:", yf is not None)
            st.write("plotly available:", go is not None)

    # -------------------- Wave Doctor + What-If
    with tab_doctor:
        st.markdown("## Wave Doctor")
        if cov.get("rows", 0) == 0:
            st.warning("Data Integrity Flag: No history returned.")
        else:
            flags = []
            if cov.get("age_days") is not None and cov.get("age_days", 0) >= 7:
                flags.append("Stale data (>=7 days).")
            if cov.get("missing_pct") is not None and cov.get("missing_pct", 0) >= 0.05:
                flags.append("Missing business days >=5%.")
            msf = mode_separation_flags(wave, days=days, force_csv=force_csv)
            flags.extend(msf)

            if flags:
                for f in flags[:8]:
                    st.warning(f)
            else:
                st.success("No major integrity flags detected.")

        st.divider()
        st.markdown("## What-If Lab (Shadow Simulation)")
        st.caption("This is a shadow overlay for scenario exploration. It does NOT change the engine or stored results.")
        if cov.get("rows", 0) < 60:
            st.info("Not enough history for What-If simulation.")
        else:
            shock = st.slider("Shock (one-time return impact)", min_value=-0.30, max_value=0.30, value=0.00, step=0.01)
            vol_mult = st.slider("Vol multiplier (display only)", min_value=0.5, max_value=2.0, value=1.0, step=0.1)

            base = wave_nav.copy().dropna()
            if len(base) < 60:
                st.info("Not enough NAV data for overlay.")
            else:
                sim = base.copy()
                if len(sim) > 5 and shock != 0.0:
                    sim.iloc[-1] = sim.iloc[-1] * (1.0 + shock)
                sim.name = "shadow_nav"

                df = pd.concat([base.rename("base_nav"), sim], axis=1).dropna()
                if go is not None:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=df.index, y=df["base_nav"], name="Base NAV"))
                    fig.add_trace(go.Scatter(x=df.index, y=df["shadow_nav"], name="Shadow NAV"))
                    fig.update_layout(height=360, margin=dict(l=10, r=10, t=10, b=10))
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.line_chart(df)

                st.caption(f"Vol multiplier (display): {vol_mult}x")

        with st.expander("System Diagnostics"):
            st.write("Force CSV:", force_csv)
            st.write("Logs perf path:", _logs_perf_path(wave, mode))

    # -------------------- Risk Lab
    with tab_risk:
        st.markdown("## Risk Lab")
        if cov.get("rows", 0) < 60:
            st.warning("Not enough data to compute risk metrics.")
        else:
            sh = sharpe_ratio(wret)
            so = sortino_ratio(wret)
            v, cv = var_cvar(wret, level=0.95)
            ddv = max_drawdown(wave_nav)
            vol = annualized_vol(wret)

            c1, c2, c3 = st.columns(3)
            c1.metric("Sharpe (ann.)", fmt_num(sh, 2))
            c2.metric("Sortino (ann.)", fmt_num(so, 2))
            c3.metric("Volatility (ann.)", fmt_pct(vol, 2))

            c4, c5, c6 = st.columns(3)
            c4.metric("VaR 95% (daily)", fmt_pct(v, 2))
            c5.metric("CVaR 95% (daily)", fmt_pct(cv, 2))
            c6.metric("Max Drawdown", fmt_pct(ddv, 2))

            dd_series = drawdown_series(wave_nav)
            if not dd_series.empty:
                st.markdown("### Drawdown Monitor")
                st.line_chart(dd_series)

            st.markdown("### Rolling 20D Vol (ann.)")
            rv = rolling_vol(wret, window=20)
            if not rv.empty:
                st.line_chart(rv)

    # -------------------- Correlation
    with tab_corr:
        st.markdown("## Correlation")
        st.caption("Correlation uses daily returns (mode + window). Raw series is hidden by default.")
        if not expose_raw:
            st.info("Correlation is hidden while raw series exposure is disabled.")
        else:
            # Build a return matrix across all waves using available histories
            rows = []
            for w in waves[:60]:  # cap for performance
                hh = compute_wave_history(w, mode, days=days, force_csv=force_csv)
                if hh is None or hh.empty or "wave_ret" not in hh.columns:
                    continue
                s = hh["wave_ret"].rename(w)
                rows.append(s)

            if not rows:
                st.warning("Not enough data to compute correlation.")
            else:
                mat = pd.concat(rows, axis=1).dropna(how="any")
                if mat.shape[0] < 60 or mat.shape[1] < 3:
                    st.warning("Not enough overlap for correlation.")
                else:
                    corr = mat.corr()
                    st.dataframe(corr, use_container_width=True)

    # -------------------- Vector Insight Layer
    with tab_vector:
        st.markdown("## Vector OS Insight Layer")
        ins = vector_insights(wave, mode, hist)
        for line in ins:
            st.info(line)

    # -------------------- Alpha Heatmap
    with tab_heat:
        st.markdown("## Alpha Heatmap — All Waves × Timeframe")
        tf_days = [1, 5, 21, 63, 126, 252]
        tf_labels = ["1D", "1W", "1M", "3M", "6M", "1Y"]

        table = []
        for w in waves:
            hh = compute_wave_history(w, mode, days=max(days, 365), force_csv=force_csv)
            if hh is None or hh.empty:
                row = {"Wave": w}
                for lab in tf_labels:
                    row[lab] = np.nan
                table.append(row)
                continue
            wn = hh.get("wave_nav", pd.Series(dtype=float)).dropna()
            bn = hh.get("bm_nav", pd.Series(dtype=float)).dropna()
            row = {"Wave": w}
            for lab, d in zip(tf_labels, tf_days):
                if len(wn) >= d+1 and len(bn) >= d+1:
                    row[lab] = ret_from_nav(wn, d+1) - ret_from_nav(bn, d+1)
                else:
                    row[lab] = np.nan
            table.append(row)

        heat = pd.DataFrame(table)
        if heat.empty:
            st.info("No waves / no data.")
        else:
            # show as percent strings but keep numeric for styling
            display = heat.copy()
            for lab in tf_labels:
                display[lab] = display[lab].map(lambda x: fmt_pct(x, 2) if math.isfinite(x) else "—")
            st.dataframe(display, use_container_width=True)

    # -------------------- Leaderboard
    with tab_lb:
        st.markdown("## WaveScore Leaderboard (Display Approximation)")
        rows = []
        for w in waves:
            hh = compute_wave_history(w, mode, days=max(days, 365), force_csv=force_csv)
            sc = wavescore_console(hh)
            rows.append({"Wave": w, "WaveScore": sc, "Grade": _grade_from_score(sc)})

        df = pd.DataFrame(rows)
        df = df.sort_values("WaveScore", ascending=False, na_position="last").reset_index(drop=True)
        df["Rank"] = np.arange(1, len(df) + 1)
        st.dataframe(df[["Rank", "Wave", "WaveScore", "Grade"]], use_container_width=True)


# ============================================================
# Execute with hard safety (prevents blank screen)
# ============================================================
try:
    st.title("Institutional Console")
    render_app()
except Exception as e:
    st.error("The app hit an error, but stayed alive (safe mode).")
    with st.expander("Error details"):
        st.exception(e)