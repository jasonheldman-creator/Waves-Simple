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

@st.cache_data(show_spinner=False)
def compute_wavescore_for_all_waves(all_waves: List[str], mode: str, days: int = 365, force_csv: bool = False) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for wave in all_waves:
        hist = compute_wave_history(wave, mode=mode, days=days, force_csv=force_csv)
        if hist.empty or len(hist) < 20:
            rows.append({"Wave": wave, "WaveScore": np.nan, "Grade": "N/A", "IR": np.nan, "Alpha": np.nan})
            continue

        nav_wave = hist["wave_nav"]
        nav_bm = hist["bm_nav"]
        wave_ret = hist["wave_ret"]
        bm_ret = hist["bm_ret"]

        alpha = ret_from_nav(nav_wave, len(nav_wave)) - ret_from_nav(nav_bm, len(nav_bm))
        te = tracking_error(wave_ret, bm_ret)
        ir = information_ratio(nav_wave, nav_bm, te)
        mdd = max_drawdown(nav_wave)
        hit_rate = float((wave_ret >= bm_ret).mean()) if len(wave_ret) else np.nan

        # light scoring (display only)
        rq = float(np.clip((np.nan_to_num(ir) / 1.5), 0.0, 1.0) * 25.0)
        rc = float(np.clip(1.0 - (abs(np.nan_to_num(mdd)) / 0.35), 0.0, 1.0) * 25.0)
        co = float(np.clip(np.nan_to_num(hit_rate), 0.0, 1.0) * 15.0)
        rs = float(np.clip(1.0 - (abs(np.nan_to_num(te)) / 0.25), 0.0, 1.0) * 15.0)
        tr = 20.0  # placeholder “transparency” point bucket (display only)

        total = float(np.clip(rq + rc + co + rs + tr, 0.0, 100.0))
        rows.append({"Wave": wave, "WaveScore": total, "Grade": _grade_from_score(total), "IR": ir, "Alpha": alpha})

    df = pd.DataFrame(rows) if rows else pd.DataFrame()
    if df.empty:
        return df
    return df.sort_values("WaveScore", ascending=False, na_position="last").reset_index(drop=True)


# ============================================================
# Alpha Heatmap
# ============================================================
@st.cache_data(show_spinner=False)
def build_alpha_matrix(all_waves: List[str], mode: str, days: int = 365, force_csv: bool = False) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for wname in all_waves:
        hist = compute_wave_history(wname, mode=mode, days=days, force_csv=force_csv)
        if hist.empty or len(hist) < 2:
            rows.append({"Wave": wname, "1D Alpha": np.nan, "30D Alpha": np.nan, "60D Alpha": np.nan, "365D Alpha": np.nan})
            continue

        nav_w = hist["wave_nav"]
        nav_b = hist["bm_nav"]

        a1 = np.nan
        if len(nav_w) >= 2 and len(nav_b) >= 2:
            a1 = float(nav_w.iloc[-1] / nav_w.iloc[-2] - 1.0) - float(nav_b.iloc[-1] / nav_b.iloc[-2] - 1.0)

        a30 = ret_from_nav(nav_w, min(30, len(nav_w))) - ret_from_nav(nav_b, min(30, len(nav_b)))
        a60 = ret_from_nav(nav_w, min(60, len(nav_w))) - ret_from_nav(nav_b, min(60, len(nav_b)))
        a365 = ret_from_nav(nav_w, len(nav_w)) - ret_from_nav(nav_b, len(nav_b))

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

    fig = go.Figure(data=go.Heatmap(z=z, x=x, y=y, zmin=-v, zmax=v, colorbar=dict(title="Alpha")))
    fig.update_layout(title=title, height=min(900, 240 + 22 * max(10, len(y))), margin=dict(l=80, r=40, t=60, b=40))
    st.plotly_chart(fig, use_container_width=True)


# ============================================================
# "MEAT" UTILITIES
# ============================================================
def benchmark_truth_panel(selected_wave: str, bm_mix: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Returns (mix_for_wave, difficulty_signals)
    """
    mix = pd.DataFrame(columns=["Ticker", "Name", "Weight"])
    sig: Dict[str, Any] = {"n": 0, "top_weight": np.nan, "hhi": np.nan, "effective_n": np.nan}
    try:
        if bm_mix is None or bm_mix.empty:
            return mix, sig
        df = bm_mix.copy()
        if "Wave" in df.columns:
            df = df[df["Wave"].astype(str).apply(lambda x: _loose_eq(x, selected_wave))]
        for col in ["Ticker", "Name", "Weight"]:
            if col not in df.columns:
                df[col] = ""
        df["Ticker"] = df["Ticker"].astype(str).str.upper().str.strip()
        df["Weight"] = pd.to_numeric(df["Weight"], errors="coerce").fillna(0.0)
        df = df[df["Ticker"].astype(str).str.len() > 0].copy()
        tot = float(df["Weight"].sum())
        if tot > 0:
            df["Weight"] = df["Weight"] / tot
        df = df.sort_values("Weight", ascending=False).reset_index(drop=True)
        mix = df[["Ticker", "Name", "Weight"]].copy()

        w = mix["Weight"].values.astype(float) if not mix.empty else np.array([])
        sig["n"] = int(len(w))
        if len(w) > 0:
            sig["top_weight"] = float(np.max(w))
            sig["hhi"] = float(np.sum(w**2))
            sig["effective_n"] = float(1.0 / max(1e-9, sig["hhi"]))
        return mix, sig
    except Exception:
        return mix, sig

@st.cache_data(show_spinner=False)
def compute_static_basket_nav_from_holdings(holdings: pd.DataFrame, days: int = 365) -> pd.Series:
    """
    Builds a static basket NAV from (Ticker, Weight) using yfinance.
    """
    if yf is None or holdings is None or holdings.empty:
        return pd.Series(dtype=float)

    df = holdings.copy()
    if "Ticker" not in df.columns or "Weight" not in df.columns:
        return pd.Series(dtype=float)
    df["Ticker"] = df["Ticker"].astype(str).str.upper().str.strip()
    df["Weight"] = pd.to_numeric(df["Weight"], errors="coerce").fillna(0.0)
    df = df[df["Ticker"].astype(str).str.len() > 0]
    df = df.groupby("Ticker", as_index=False)["Weight"].sum()
    tot = float(df["Weight"].sum())
    if tot <= 0:
        return pd.Series(dtype=float)
    df["Weight"] = df["Weight"] / tot

    tickers = df["Ticker"].tolist()
    px = fetch_prices_daily(tickers, days=days)
    if px.empty:
        return pd.Series(dtype=float)

    # align
    px = px.sort_index().ffill().bfill()
    rets = px.pct_change().fillna(0.0)

    wmap = dict(zip(df["Ticker"], df["Weight"]))
    cols = [c for c in rets.columns if c in wmap]
    if not cols:
        return pd.Series(dtype=float)

    w = np.array([wmap[c] for c in cols], dtype=float)
    w = w / max(1e-12, w.sum())

    basket_ret = (rets[cols].values @ w).astype(float)
    nav = pd.Series((1.0 + basket_ret).cumprod(), index=rets.index, name="static_basket_nav")
    return nav

def mode_separation_proof(wave_name: str, days: int, force_csv: bool) -> pd.DataFrame:
    """
    Checks if different modes are suspiciously identical for the same wave.
    Returns a small table with correlations and a flag.
    """
    modes = ["Standard", "Alpha-Minus-Beta", "Private Logic"]
    navs: Dict[str, pd.Series] = {}
    rets: Dict[str, pd.Series] = {}

    for m in modes:
        h = compute_wave_history(wave_name, mode=m, days=days, force_csv=force_csv)
        if h is None or h.empty:
            continue
        navs[m] = h["wave_nav"].dropna()
        rets[m] = h["wave_ret"].dropna()

    rows = []
    for i in range(len(modes)):
        for j in range(i + 1, len(modes)):
            a, b = modes[i], modes[j]
            if a not in rets or b not in rets:
                continue
            df = pd.concat([rets[a].rename("a"), rets[b].rename("b")], axis=1).dropna()
            if len(df) < 30:
                corr = np.nan
            else:
                corr = float(df["a"].corr(df["b"]))
            flag = "OK"
            if math.isfinite(corr) and corr >= 0.995:
                flag = "SUSPICIOUSLY IDENTICAL"
            rows.append({"Mode A": a, "Mode B": b, "Return Corr": corr, "Flag": flag})

    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=["Mode A", "Mode B", "Return Corr", "Flag"])

def multi_factor_regression(wave_ret: pd.Series, days: int = 365) -> pd.DataFrame:
    """
    OLS: wave_ret ~ intercept + factor_returns
    Factors: SPY, QQQ, IWM, TLT, GLD, BTC-USD
    """
    cols = ["Factor", "Beta"]
    if yf is None:
        return pd.DataFrame(columns=cols)

    factors = ["SPY", "QQQ", "IWM", "TLT", "GLD", "BTC-USD"]
    px = fetch_prices_daily(factors, days=days)
    if px.empty:
        return pd.DataFrame(columns=cols)

    f_ret = px.pct_change().dropna()
    w = safe_series(wave_ret).dropna()
    df = pd.concat([w.rename("wave"), f_ret], axis=1).dropna()
    if df.shape[0] < 60:
        return pd.DataFrame(columns=cols)

    y = df["wave"].values.astype(float)
    X = df[factors].values.astype(float)
    X = np.column_stack([np.ones(len(X)), X])  # intercept
    try:
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        out = [{"Factor": "Intercept", "Beta": float(beta[0])}]
        for i, f in enumerate(factors, start=1):
            out.append({"Factor": f, "Beta": float(beta[i])})
        return pd.DataFrame(out)
    except Exception:
        return pd.DataFrame(columns=cols)


# ============================================================
# Main UI
# ============================================================
st.title("WAVES Intelligence™ Institutional Console")

# Engine import status
if ENGINE_IMPORT_ERROR is not None:
    st.error("Engine import failed. The app will use CSV fallbacks where possible.")
    st.code(str(ENGINE_IMPORT_ERROR))

all_waves = get_all_waves_safe()
if not all_waves:
    st.warning("No waves discovered yet. If this is unexpected, check: engine import, wave_config.csv, wave_weights.csv, or list.csv.")
    with st.expander("Diagnostics"):
        st.write("Files present:")
        st.write({p: os.path.exists(p) for p in ["wave_config.csv", "wave_weights.csv", "wave_history.csv", "list.csv", "waves_engine.py"]})
        st.stop()

modes = ["Standard", "Alpha-Minus-Beta", "Private Logic"]

with st.sidebar:
    st.header("Controls")
    mode = st.selectbox("Mode", modes, index=0)
    selected_wave = st.selectbox("Wave", all_waves, index=0)
    days = st.slider("History window (days)", min_value=90, max_value=1500, value=365, step=30)

    force_csv = st.toggle("Force CSV history (debug/demo)", value=False)
    expose_raw = st.toggle("Expose raw series (advanced)", value=False)
    st.caption("Raw series exposure increases surface area. Leave OFF for demos.")

bm_mix = get_benchmark_mix()
bm_id = benchmark_snapshot_id(selected_wave, bm_mix)
bm_drift = benchmark_drift_status(selected_wave, mode, bm_id)

hist = compute_wave_history(selected_wave, mode=mode, days=days, force_csv=force_csv)
cov = coverage_report(hist)

# regime / VIX
regime = "neutral"
vix_val = np.nan
if yf is not None:
    try:
        vix_df = fetch_prices_daily(["^VIX"], days=30)
        if not vix_df.empty and "^VIX" in vix_df.columns:
            vix_val = float(vix_df["^VIX"].iloc[-1])
            if vix_val >= 25:
                regime = "risk-off"
            elif vix_val <= 16:
                regime = "risk-on"
    except Exception:
        pass

# headline metrics
a30 = np.nan
r30 = np.nan
a365 = np.nan
r365 = np.nan
te = np.nan
ir = np.nan
mdd = np.nan

if not hist.empty and len(hist) >= 2:
    r30 = ret_from_nav(hist["wave_nav"], min(30, len(hist)))
    a30 = r30 - ret_from_nav(hist["bm_nav"], min(30, len(hist)))
    r365 = ret_from_nav(hist["wave_nav"], len(hist))
    a365 = r365 - ret_from_nav(hist["bm_nav"], len(hist))
    te = tracking_error(hist["wave_ret"], hist["bm_ret"])
    ir = information_ratio(hist["wave_nav"], hist["bm_nav"], te)
    mdd = max_drawdown(hist["wave_nav"])

# WaveScore table + rank
ws_df = compute_wavescore_for_all_waves(all_waves, mode=mode, days=min(days, 365), force_csv=force_csv)
ws_val = np.nan
rank = None
if not ws_df.empty:
    try:
        idx_match = ws_df.index[ws_df["Wave"].astype(str) == str(selected_wave)]
        if len(idx_match) > 0:
            ws_val = float(ws_df.loc[idx_match[0], "WaveScore"])
            rank = int(idx_match[0] + 1)
    except Exception:
        pass

# Sticky summary chips
chips = []
chips.append(f"BM Snapshot: {bm_id} · {'Stable' if bm_drift=='stable' else 'DRIFT'}")
chips.append(f"Coverage: {fmt_num(cov.get('completeness_score', np.nan), 1)} / 100")
chips.append(f"Rows: {cov.get('rows','—')} · Age: {cov.get('age_days','—')}")
chips.append(f"Regime: {regime}")
chips.append(f"VIX: {fmt_num(vix_val,1) if math.isfinite(vix_val) else '—'}")
chips.append(f"30D α: {fmt_pct(a30)} · 30D r: {fmt_pct(r30)}")
chips.append(f"365D α: {fmt_pct(a365)} · 365D r: {fmt_pct(r365)}")
chips.append(f"TE: {fmt_pct(te)} · IR: {fmt_num(ir,2)}")
chips.append(f"WaveScore: {fmt_score(ws_val)} ({_grade_from_score(ws_val)}) · Rank: {rank if rank else '—'}")

st.markdown('<div class="waves-sticky">', unsafe_allow_html=True)
for c in chips:
    st.markdown(f'<span class="waves-chip">{c}</span>', unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

tabs = st.tabs([
    "Console",
    "Benchmark Truth",
    "Attribution",
    "Wave Doctor + What-If",
    "Factor Decomposition",
    "Risk Lab",
    "Correlation",
    "Vector OS Insight Layer",
])

# -------------------------
# Console
# -------------------------
with tabs[0]:
    st.subheader("Market Intel")
    mi = market_intel_snapshot()
    if mi.empty:
        st.info("Market Intel unavailable (yfinance missing or data fetch failed).")
    else:
        st.dataframe(mi, use_container_width=True)

    st.subheader("Alpha Heatmap View (All Waves × Timeframe)")
    alpha_df = build_alpha_matrix(all_waves, mode=mode, days=min(days, 365), force_csv=force_csv)
    plot_alpha_heatmap(alpha_df, selected_wave, title=f"Alpha Heatmap — Mode: {mode}")

    st.subheader("Coverage & Data Integrity")
    if cov["rows"] == 0:
        st.warning("No history returned for this wave/mode. The app attempted engine → CSV fallback (with mode aliases).")
        st.caption(f"Mode candidates tried: {mode_candidates(mode)}")

    col1, col2, col3 = st.columns(3)
    col1.metric("History Rows", cov.get("rows", 0))
    col2.metric("Last Data Age (days)", cov.get("age_days", "—"))
    col3.metric("Completeness Score", fmt_num(cov.get("completeness_score", np.nan), 1))

    with st.expander("Coverage Details"):
        st.write(cov)

    st.subheader("Holdings (Top 10)")
    hold = get_wave_holdings(selected_wave)
    if hold.empty:
        st.info("Holdings unavailable (engine did not return holdings and wave_weights.csv mapping did not match).")
    else:
        hold2 = hold.copy()
        if "Weight" in hold2.columns:
            hold2["Weight"] = pd.to_numeric(hold2["Weight"], errors="coerce").fillna(0.0)
        if "Ticker" in hold2.columns:
            hold2["Google"] = hold2["Ticker"].astype(str).apply(lambda t: f"https://www.google.com/finance/quote/{t}")
        st.dataframe(hold2.head(10), use_container_width=True)

# -------------------------
# Benchmark Truth
# -------------------------
with tabs[1]:
    st.subheader("Benchmark Truth")
    mix, sig = benchmark_truth_panel(selected_wave, bm_mix)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Constituents", sig.get("n", 0))
    c2.metric("Top Weight", fmt_pct(sig.get("top_weight", np.nan), 2))
    c3.metric("HHI (concentration)", fmt_num(sig.get("hhi", np.nan), 4))
    c4.metric("Effective N", fmt_num(sig.get("effective_n", np.nan), 1))

    st.caption("These are difficulty/concentration signals for the benchmark mix, not performance claims.")

    if mix.empty:
        st.info("Benchmark mix not available (engine missing get_benchmark_mix_table or returned empty).")
    else:
        st.dataframe(mix, use_container_width=True)

    st.subheader("Benchmark Drift")
    st.write(f"Snapshot ID: **{bm_id}**")
    st.write(f"Session Drift Status: **{'Stable' if bm_drift=='stable' else 'DRIFT'}**")
    if bm_drift != "stable":
        st.warning("Benchmark mix changed during the session. For demos: freeze benchmark mix inputs to avoid confusion.")

    st.subheader("Mode Separation Proof (Same Wave across Modes)")
    proof = mode_separation_proof(selected_wave, days=min(days, 365), force_csv=force_csv)
    if proof.empty:
        st.info("Not enough history across multiple modes to test separation.")
    else:
        proof2 = proof.copy()
        if "Return Corr" in proof2.columns:
            proof2["Return Corr"] = proof2["Return Corr"].map(lambda x: fmt_num(x, 4))
        st.dataframe(proof2, use_container_width=True)
        if any(proof["Flag"].astype(str).str.contains("SUSPICIOUS", na=False)):
            st.error("Mode histories look suspiciously identical. This usually means the engine is not separating modes yet (or is reusing the same output).")

# -------------------------
# Attribution
# -------------------------
with tabs[2]:
    st.subheader("Alpha Attribution — Engine vs Static Basket (from Holdings)")
    st.caption("Static Basket is built from holdings weights using yfinance prices. It does NOT modify engine logic. If yfinance is unavailable, this will be empty.")

    if hist.empty or len(hist) < 30:
        st.info("Not enough engine history to run attribution.")
    else:
        hold = get_wave_holdings(selected_wave)
        if hold.empty:
            st.info("Holdings missing; cannot construct static basket.")
        else:
            static_nav = compute_static_basket_nav_from_holdings(hold[["Ticker", "Weight"]], days=min(days, 365))
            if static_nav.empty:
                st.info("Static basket NAV unavailable (yfinance missing or price fetch failed).")
            else:
                # align NAVs
                df = pd.concat(
                    [
                        hist["wave_nav"].rename("Engine Wave NAV"),
                        hist["bm_nav"].rename("Benchmark NAV"),
                        static_nav.rename("Static Basket NAV"),
                    ],
                    axis=1,
                ).dropna()

                if df.empty or df.shape[0] < 30:
                    st.info("Not enough overlapping days between engine history and static basket.")
                else:
                    eng_alpha = ret_from_nav(df["Engine Wave NAV"], len(df)) - ret_from_nav(df["Benchmark NAV"], len(df))
                    stat_alpha = ret_from_nav(df["Static Basket NAV"], len(df)) - ret_from_nav(df["Benchmark NAV"], len(df))
                    diff = eng_alpha - stat_alpha

                    c1, c2, c3 = st.columns(3)
                    c1.metric("Engine α (window)", fmt_pct(eng_alpha, 2))
                    c2.metric("Static Basket α (window)", fmt_pct(stat_alpha, 2))
                    c3.metric("Engine minus Static", fmt_pct(diff, 2))

                    st.write("NAV Comparison (Overlapping Window)")
                    st.line_chart(df)

                    st.caption("If Engine minus Static is consistently positive, the engine logic is adding value beyond static weights (or the static basket differs from actual engine holdings/turnover).")

# -------------------------
# Wave Doctor + What-If Lab
# -------------------------
with tabs[3]:
    st.subheader("Wave Doctor")
    notes = []
    if cov.get("flags"):
        notes.append("**Data Integrity Flags:** " + "; ".join(cov["flags"]))
    if bm_drift != "stable":
        notes.append("**Benchmark Drift:** snapshot changed during session — freeze benchmark mix for demos.")
    if math.isfinite(a30) and abs(a30) >= 0.08:
        notes.append("**Large 30D alpha:** verify benchmark mix + missing days; big alpha can be real or coverage-driven.")
    if math.isfinite(te) and te >= 0.20:
        notes.append("**High tracking error:** wave is behaving very differently than benchmark (active risk elevated).")
    if math.isfinite(mdd) and mdd <= -0.25:
        notes.append("**Deep drawdown:** consider stronger SmartSafe posture in stress regimes.")
    if not notes:
        notes.append("No major anomalies detected on this window.")
    for n in notes:
        st.markdown(f"- {n}")

    st.divider()
    st.subheader("What-If Lab (Shadow Simulation)")
    st.caption("This is a shadow overlay for scenario exploration. It does NOT change the engine or stored results.")

    if hist.empty or len(hist) < 60:
        st.info("Not enough history for What-If simulation.")
    else:
        left, right = st.columns([1, 2])

        with left:
            cash_buffer = st.slider("Cash Buffer (%)", 0, 80, 0, 5) / 100.0
            exposure_scale = st.slider("Exposure Scale (%)", 10, 150, 100, 5) / 100.0
            use_vix_gate = st.toggle("VIX Gate (reduce exposure when VIX high)", value=True)
            vix_soft = st.slider("VIX soft threshold", 14, 30, 20, 1)
            vix_hard = st.slider("VIX hard threshold", 18, 50, 30, 1)

        # build daily exposure series
        r = hist["wave_ret"].fillna(0.0).copy()
        exp = pd.Series(1.0, index=r.index)

        if use_vix_gate and yf is not None:
            try:
                vix = fetch_prices_daily(["^VIX"], days=min(days, 365))
                if not vix.empty and "^VIX" in vix.columns:
                    v = vix["^VIX"].reindex(r.index).ffill().bfill()
                    # linear ramp exposure from 1.0 at soft to 0.25 at hard, min 0.10
                    ramp = (vix_hard - vix_soft)
                    gate = []
                    for val in v.values:
                        if not math.isfinite(val):
                            gate.append(1.0)
                        elif val <= vix_soft:
                            gate.append(1.0)
                        elif val >= vix_hard:
                            gate.append(0.25)
                        else:
                            frac = float((val - vix_soft) / max(1e-9, ramp))
                            gate.append(1.0 - frac * (1.0 - 0.25))
                    exp = pd.Series(gate, index=r.index).clip(lower=0.10, upper=1.0)
            except Exception:
                pass

        exp = exp * exposure_scale
        exp = exp.clip(lower=0.0, upper=1.5)

        r_adj = r * exp * (1.0 - cash_buffer)
        nav_adj = pd.Series((1.0 + r_adj).cumprod(), index=r.index, name="Shadow NAV")
        nav_base = hist["wave_nav"].reindex(nav_adj.index).ffill().bfill().rename("Engine NAV")

        out_df = pd.concat([nav_base, nav_adj], axis=1).dropna()
        with right:
            st.write("Engine vs Shadow NAV")
            st.line_chart(out_df)

            dd_shadow = max_drawdown(out_df["Shadow NAV"])
            dd_engine = max_drawdown(out_df["Engine NAV"])
            st.write("Quick Compare")
            c1, c2, c3 = st.columns(3)
            c1.metric("Engine MaxDD", fmt_pct(dd_engine, 2))
            c2.metric("Shadow MaxDD", fmt_pct(dd_shadow, 2))
            c3.metric("Shadow Exposure (avg)", fmt_num(float(exp.mean()), 2))

# -------------------------
# Factor Decomposition
# -------------------------
with tabs[4]:
    st.subheader("Factor Decomposition (Multi-Factor Regression)")
    st.caption("OLS regression using yfinance factors: SPY, QQQ, IWM, TLT, GLD, BTC-USD (if available).")

    if hist.empty or len(hist) < 60:
        st.info("Not enough history.")
    else:
        betas = multi_factor_regression(hist["wave_ret"], days=min(days, 365))
        if betas.empty:
            st.info("Factor regression unavailable (yfinance missing or insufficient overlapping factor data).")
        else:
            betas2 = betas.copy()
            betas2["Beta"] = betas2["Beta"].map(lambda x: fmt_num(x, 4))
            st.dataframe(betas2, use_container_width=True)

# -------------------------
# Risk Lab
# -------------------------
with tabs[5]:
    st.subheader("Risk Lab")
    if hist.empty or len(hist) < 50:
        st.info("Not enough data to compute risk lab metrics.")
    else:
        r = hist["wave_ret"].dropna()

        sh = sharpe_ratio(r, 0.0)
        so = sortino_ratio(r, 0.0)
        dd = downside_deviation(r, 0.0)
        sk, ku = skew_kurt(r)
        v95, c95 = var_cvar(r, 0.95)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Sharpe (0% rf)", fmt_num(sh, 2))
        c2.metric("Sortino (0% MAR)", fmt_num(so, 2))
        c3.metric("Downside Dev", fmt_pct(dd))
        c4.metric("Max Drawdown", fmt_pct(mdd))

        c5, c6, c7, c8 = st.columns(4)
        c5.metric("Skew", fmt_num(sk, 2))
        c6.metric("Kurtosis", fmt_num(ku, 2))
        c7.metric("VaR 95% (daily)", fmt_pct(v95))
        c8.metric("CVaR 95% (daily)", fmt_pct(c95))

        st.write("Drawdown (Wave vs Benchmark)")
        dd_w = drawdown_series(hist["wave_nav"])
        dd_b = drawdown_series(hist["bm_nav"])
        dd_df = pd.concat([dd_w.rename("Wave"), dd_b.rename("Benchmark")], axis=1).dropna()
        st.line_chart(dd_df)

        st.write("Rolling 30D Alpha + Rolling Vol")
        ra = rolling_alpha_from_nav(hist["wave_nav"], hist["bm_nav"], window=30)
        rv = rolling_vol(hist["wave_ret"], window=20)
        roll_df = pd.concat([ra.rename("Rolling 30D Alpha"), rv.rename("Rolling Vol (20D)")], axis=1).dropna()
        st.line_chart(roll_df)

        st.write("Alpha Persistence (Rolling 30D windows)")
        ap = alpha_persistence(ra)
        st.metric("Alpha Persistence", fmt_pct(ap))

# -------------------------
# Correlation
# -------------------------
with tabs[6]:
    st.subheader("Correlation")
    st.caption("Correlation uses daily returns over the selected window and mode.")

    if not expose_raw:
        st.info("Correlation is hidden while raw series is disabled. Enable 'Expose raw series' if needed.")
    else:
        rets: Dict[str, pd.Series] = {}
        for w in all_waves:
            h = compute_wave_history(w, mode=mode, days=min(days, 365), force_csv=force_csv)
            if h is not None and not h.empty and "wave_ret" in h.columns:
                s = h["wave_ret"].dropna()
                if len(s) >= 30:
                    rets[w] = s

        if len(rets) < 2:
            st.info("Not enough waves with history to compute correlations.")
        else:
            ret_df = pd.DataFrame(rets).dropna(how="all")
            corr = ret_df.corr()
            st.dataframe(corr, use_container_width=True)

            if selected_wave in corr.columns:
                series = corr[selected_wave].dropna().sort_values(ascending=False)
                most = series.iloc[1:6] if len(series) > 1 else series
                least = series.tail(5)

                c1, c2 = st.columns(2)
                with c1:
                    st.write("Most Correlated")
                    st.dataframe(most.to_frame("corr"), use_container_width=True)
                with c2:
                    st.write("Least Correlated")
                    st.dataframe(least.to_frame("corr"), use_container_width=True)

# -------------------------
# Vector OS Insight Layer
# -------------------------
with tabs[7]:
    st.subheader("Vector OS Insight Layer")
    if hist.empty or len(hist) < 20:
        st.info("Not enough data for insights yet.")
    else:
        insights = []
        if cov.get("flags"):
            insights.append("**Data Integrity Flags:** " + "; ".join(cov["flags"]))
        if bm_drift != "stable":
            insights.append("**Benchmark Drift:** snapshot changed during session — freeze benchmark mix for demos.")
        if math.isfinite(a30) and abs(a30) >= 0.08:
            insights.append("**Large 30D alpha:** verify benchmark mix + missing days; big alpha can be real or coverage-driven.")
        if math.isfinite(te) and te >= 0.20:
            insights.append("**High tracking error:** wave is behaving very differently than benchmark (active risk elevated).")
        if math.isfinite(mdd) and mdd <= -0.25:
            insights.append("**Deep drawdown:** consider stronger SmartSafe posture in stress regimes.")

        if not insights:
            insights.append("No major anomalies detected on this window.")

        for it in insights:
            st.markdown(f"- {it}")

# Footer diagnostics
with st.expander("System Diagnostics (if something looks off)"):
    st.write("Engine loaded:", we is not None)
    st.write("Engine import error:", str(ENGINE_IMPORT_ERROR) if ENGINE_IMPORT_ERROR else "None")
    st.write("Mode selected:", mode)
    st.write("Mode candidates tried:", mode_candidates(mode))
    st.write("Force CSV:", force_csv)
    st.write("Expose raw series:", expose_raw)
    st.write("Files present:", {p: os.path.exists(p) for p in ["wave_config.csv", "wave_weights.csv", "wave_history.csv", "list.csv", "waves_engine.py"]})
    st.write("Selected:", {"wave": selected_wave, "days": days})
    st.write("History shape:", None if hist is None else hist.shape)
    if hist is not None and not hist.empty:
        st.write("History columns:", list(hist.columns))
        st.write("History tail:", hist.tail(3))