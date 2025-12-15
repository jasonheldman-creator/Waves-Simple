# app.py — WAVES Intelligence™ Institutional Console (Vector OS Edition)
# FULL PRODUCTION FILE (NO PATCHES) — “MEAT RESTORE + HISTORY FIX”
#
# Goal of this build:
#   ✅ Bring back the “meat” panels (Market Intel, Benchmark Truth, Attribution, Doctor/What-If, Risk Lab, Correlation, Vector Insight)
#   ✅ Fix missing history by making history retrieval robust in Streamlit Cloud:
#        logs/performance → engine → wave_history.csv fallback (with mode aliases)
#   ✅ Add System Diagnostics so we can see exactly why a wave/mode is missing.
#
# Notes:
#   • Engine math is NOT modified.
#   • Correlation remains hidden by default (Expose raw series toggle).
#   • If logs aren’t committed or generated in Cloud, Diagnostics will show that immediately.

from __future__ import annotations

import os
import re
import math
import glob
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

def _loose_eq(a: str, b: str) -> bool:
    return str(a).strip().lower() == str(b).strip().lower()

def _slug(s: str) -> str:
    s = str(s or "").strip().lower()
    s = s.replace("&", "and")
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s


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
.waves-sticky {
  position: sticky; top: 0; z-index: 999;
  backdrop-filter: blur(10px); -webkit-backdrop-filter: blur(10px);
  padding: 10px 12px; margin: 0 0 12px 0; border-radius: 14px;
  border: 1px solid rgba(255,255,255,0.10);
  background: rgba(10, 15, 28, 0.66);
}
.waves-chip {
  display: inline-block; padding: 6px 10px; margin: 6px 8px 0 0;
  border-radius: 999px; border: 1px solid rgba(255,255,255,0.12);
  background: rgba(255,255,255,0.04);
  font-size: 0.85rem; line-height: 1.0rem; white-space: nowrap;
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

def safe_series(s: Optional[pd.Series]) -> pd.Series:
    if s is None or len(s) == 0:
        return pd.Series(dtype=float)
    return s.copy()

def safe_df(df: Optional[pd.DataFrame]) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    return df.copy()

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

def var_cvar(daily_ret: pd.Series, level: float = 0.95) -> Tuple[float, float]:
    r = safe_series(daily_ret).astype(float)
    if len(r) < 50:
        return (float("nan"), float("nan"))
    q = float(np.quantile(r.values, 1.0 - level))
    tail = r[r <= q]
    cvar = float(tail.mean()) if len(tail) else float("nan")
    return (q, cvar)


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
# HISTORY: standardize + CSV fallback + LOGS reader + engine
# ============================================================
def _standardize_history(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["wave_nav", "bm_nav", "wave_ret", "bm_ret"])

    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]

    # date index
    date_col = None
    for dc in ["date", "Date", "timestamp", "Timestamp", "datetime", "Datetime"]:
        if dc in out.columns:
            date_col = dc
            break
    if date_col:
        out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
        out = out.dropna(subset=[date_col]).set_index(date_col)

    out.index = pd.to_datetime(out.index, errors="coerce")
    out = out[~out.index.isna()].sort_index()

    # rename candidates
    ren = {}
    for c in out.columns:
        low = str(c).strip().lower()
        if low in ["wave_nav", "nav_wave", "portfolio_nav", "nav", "portfolio_value", "portfolio_nav_usd"]:
            ren[c] = "wave_nav"
        if low in ["bm_nav", "bench_nav", "benchmark_nav", "benchmark_value", "bm_value", "benchmark_nav_usd"]:
            ren[c] = "bm_nav"
        if low in ["wave_ret", "ret_wave", "portfolio_ret", "wave_return", "portfolio_return", "daily_return"]:
            ren[c] = "wave_ret"
        if low in ["bm_ret", "ret_bm", "benchmark_ret", "benchmark_return", "bench_return"]:
            ren[c] = "bm_ret"
    out = out.rename(columns=ren)

    # compute returns if needed
    if "wave_ret" not in out.columns and "wave_nav" in out.columns:
        out["wave_ret"] = pd.to_numeric(out["wave_nav"], errors="coerce").pct_change()
    if "bm_ret" not in out.columns and "bm_nav" in out.columns:
        out["bm_ret"] = pd.to_numeric(out["bm_nav"], errors="coerce").pct_change()

    for col in ["wave_nav", "bm_nav", "wave_ret", "bm_ret"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    keep = [c for c in ["wave_nav", "bm_nav", "wave_ret", "bm_ret"] if c in out.columns]
    out = out[keep].dropna(how="all")
    # Ensure all four columns exist (even if NaN) for downstream UI
    for col in ["wave_nav", "bm_nav", "wave_ret", "bm_ret"]:
        if col not in out.columns:
            out[col] = np.nan
    return out[["wave_nav", "bm_nav", "wave_ret", "bm_ret"]]

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

def _list_perf_files() -> List[str]:
    paths = []
    for pat in [
        "logs/performance/*.csv",
        "logs/performance/**/*.csv",
        "logs/*.csv",
        "*.csv",
    ]:
        paths.extend(glob.glob(pat, recursive=True))
    # de-dupe
    out = []
    seen = set()
    for p in sorted(paths):
        if p not in seen:
            out.append(p)
            seen.add(p)
    return out

def history_from_logs(wave_name: str, mode: str, days: int) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Returns (history_df, debug_info)
    Attempts to find a per-wave history CSV in logs/performance.
    Supports:
      • wave name in filename (loose) OR wave column inside file
      • mode column inside file OR mode in filename (loose)
    """
    debug: Dict[str, Any] = {"attempted": [], "matched_file": None, "reason": None}
    files = _list_perf_files()
    debug["files_found"] = len(files)

    if not files:
        debug["reason"] = "No CSV files found in repo runtime."
        return (pd.DataFrame(), debug)

    ws = _slug(wave_name)
    cands = mode_candidates(mode)
    ms = [_slug(m) for m in cands]

    # scoring helper
    def score_path(p: str) -> int:
        pl = _slug(os.path.basename(p))
        s = 0
        if "performance" in pl:
            s += 3
        if ws and ws in pl:
            s += 10
        # slight credit for partial wave tokens
        for tok in ws.split("_"):
            if tok and tok in pl:
                s += 1
        # mode in filename
        for mt in ms:
            if mt and mt in pl:
                s += 2
        return s

    ranked = sorted(files, key=score_path, reverse=True)
    debug["top_candidates"] = ranked[:8]

    # Try top N, then inspect content for wave/mode columns
    for p in ranked[:25]:
        debug["attempted"].append(p)
        try:
            df = pd.read_csv(p)
        except Exception:
            continue

        if df is None or df.empty:
            continue

        cols = [str(c).strip() for c in df.columns]
        lcols = [c.lower() for c in cols]
        df.columns = cols

        # If file has wave/mode columns, filter them
        wave_col = None
        mode_col = None
        for c in cols:
            if c.lower() in ["wave", "wave_name", "wavename"]:
                wave_col = c
            if c.lower() in ["mode", "risk_mode", "strategy_mode"]:
                mode_col = c

        if wave_col:
            df[wave_col] = df[wave_col].astype(str)
            df = df[df[wave_col].astype(str).apply(lambda x: _loose_eq(x, wave_name))]

        if mode_col:
            df[mode_col] = df[mode_col].astype(str)
            df = df[df[mode_col].astype(str).apply(lambda x: any(_loose_eq(x, m) for m in cands))]

        # If we filtered everything out, keep looking
        if df is None or df.empty:
            continue

        out = _standardize_history(df)
        if out.empty:
            continue

        debug["matched_file"] = p
        if len(out) > days:
            out = out.iloc[-days:]
        return (out, debug)

    debug["reason"] = "No log file matched wave/mode with usable NAV/return columns."
    return (pd.DataFrame(), debug)

def history_from_engine(wave_name: str, mode: str, days: int) -> Tuple[pd.DataFrame, Optional[str]]:
    if we is None:
        return (pd.DataFrame(), "Engine import failed or not present.")
    cands = mode_candidates(mode)

    # preferred
    try:
        if hasattr(we, "compute_history_nav"):
            for m in cands:
                try:
                    df = we.compute_history_nav(wave_name, mode=m, days=days)
                except TypeError:
                    df = we.compute_history_nav(wave_name, m, days)
                df = _standardize_history(df)
                if not df.empty:
                    return (df, None)
    except Exception as e:
        return (pd.DataFrame(), f"Engine compute_history_nav error: {e}")

    # alternates
    for fn in ["get_history_nav", "get_wave_history", "history_nav", "compute_nav_history", "compute_history"]:
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
                        return (df, None)
                except Exception:
                    continue

    return (pd.DataFrame(), "No compatible engine history function returned data.")

@st.cache_data(show_spinner=False)
def compute_wave_history(wave_name: str, mode: str, days: int, force_csv: bool) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Unified history path:
      logs → engine → wave_history.csv
    Returns (history_df, debug_dict)
    """
    dbg: Dict[str, Any] = {"path": [], "engine_error": None, "logs_debug": None}

    if not force_csv:
        hlog, ldbg = history_from_logs(wave_name, mode, days)
        dbg["path"].append("logs")
        dbg["logs_debug"] = ldbg
        if hlog is not None and not hlog.empty:
            return (hlog, dbg)

        heng, eerr = history_from_engine(wave_name, mode, days)
        dbg["path"].append("engine")
        dbg["engine_error"] = eerr
        if heng is not None and not heng.empty:
            return (heng, dbg)

    dbg["path"].append("csv")
    hcsv = history_from_csv(wave_name, mode, days)
    return (hcsv, dbg)

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
# Coverage report + diagnostics
# ============================================================
def coverage_report(hist: pd.DataFrame) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "rows": 0,
        "first_date": None,
        "last_date": None,
        "age_days": None,
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

        if out["age_days"] >= 7:
            out["flags"].append("Data is stale (>=7 days old)")
        if out["rows"] < 60:
            out["flags"].append("Limited history (<60 points)")
        return out
    except Exception:
        out["flags"].append("Coverage report error")
        return out


# ============================================================
# Sidebar controls
# ============================================================
with st.sidebar:
    st.markdown("### Controls")

    mode = st.selectbox("Mode", ["Standard", "Alpha-Minus-Beta", "Private Logic"], index=0)

    waves = get_all_waves_safe()
    if not waves:
        waves = ["AI & Cloud MegaCap Wave", "S&P 500 Wave", "Crypto Wave"]
    wave = st.selectbox("Wave", waves, index=0)

    days = st.slider("History window (days)", min_value=60, max_value=365, value=365, step=5)

    force_csv = st.toggle("Force CSV history (debug/demo)", value=False)
    expose_raw = st.toggle("Expose raw series (advanced)", value=False)

    st.divider()
    st.caption("Tip: If Rows=0, open **System Diagnostics** in-page to see what files exist in Streamlit Cloud.")


# ============================================================
# Load history
# ============================================================
hist, hdbg = compute_wave_history(wave, mode, days, force_csv=force_csv)
hist = _standardize_history(hist)

cov = coverage_report(hist)

wave_nav = safe_series(hist["wave_nav"])
bm_nav = safe_series(hist["bm_nav"])
wave_ret = safe_series(hist["wave_ret"])
bm_ret = safe_series(hist["bm_ret"])

r30 = ret_from_nav(wave_nav, min(31, len(wave_nav))) if len(wave_nav) >= 2 else float("nan")
b30 = ret_from_nav(bm_nav, min(31, len(bm_nav))) if len(bm_nav) >= 2 else float("nan")
a30 = r30 - b30 if math.isfinite(r30) and math.isfinite(b30) else float("nan")

r365 = ret_from_nav(wave_nav, min(252, len(wave_nav))) if len(wave_nav) >= 2 else float("nan")
b365 = ret_from_nav(bm_nav, min(252, len(bm_nav))) if len(bm_nav) >= 2 else float("nan")
a365 = r365 - b365 if math.isfinite(r365) and math.isfinite(b365) else float("nan")

te = tracking_error(wave_ret, bm_ret)
ir = information_ratio(wave_nav, bm_nav, te)
mdd = max_drawdown(wave_nav)


# ============================================================
# Sticky summary
# ============================================================
regime = "risk-on"  # placeholder; you can wire this to your VIX logic later
vix_val = "—"
try:
    if yf is not None:
        vix_px = fetch_prices_daily(["^VIX"], days=10)
        if not vix_px.empty and "^VIX" in vix_px.columns:
            vix_val = fmt_num(float(vix_px["^VIX"].dropna().iloc[-1]), 1)
except Exception:
    pass

st.markdown(
    f"""
<div class="waves-sticky">
  <span class="waves-chip">Rows: {cov.get("rows",0)} · Age: {cov.get("age_days","—")}</span>
  <span class="waves-chip">Regime: {regime}</span>
  <span class="waves-chip">VIX: {vix_val}</span>
  <span class="waves-chip">30D α: {fmt_pct(a30)} · 30D r: {fmt_pct(r30)}</span>
  <span class="waves-chip">365D α: {fmt_pct(a365)} · 365D r: {fmt_pct(r365)}</span>
  <span class="waves-chip">TE: {fmt_pct(te)} · IR: {fmt_num(ir,2)}</span>
  <span class="waves-chip">MaxDD: {fmt_pct(mdd)}</span>
</div>
""",
    unsafe_allow_html=True,
)

# If no history, show banner (like your screenshot)
if hist.empty or cov.get("rows", 0) == 0:
    st.warning(
        "No history returned for this wave/mode. The app attempted logs → engine → CSV fallback (with mode aliases).",
        icon="⚠️",
    )
    st.caption(f"Mode candidates tried: {mode_candidates(mode)}")


# ============================================================
# Top-level tabs
# ============================================================
tab_console, tab_truth, tab_attr, tab_doctor = st.tabs(
    ["Console", "Benchmark Truth", "Attribution", "Wave Doctor + What-If"]
)

# -------------------------------
# Console tab (Market Intel + quick view)
# -------------------------------
with tab_console:
    st.markdown("## Market Intelligence")
    mi = market_intel_snapshot()
    if mi is None or mi.empty:
        st.info("Market data unavailable.")
    else:
        st.dataframe(mi, use_container_width=True)

    st.divider()

    st.markdown("## Wave History (NAV & Benchmark)")
    if hist is None or hist.empty:
        st.info("No history series to display.")
    else:
        show = hist.copy()
        show = show.reset_index().rename(columns={"index": "date"})
        st.dataframe(show.tail(30), use_container_width=True)

        if go is not None:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=hist.index, y=hist["wave_nav"], name="Wave NAV"))
            fig.add_trace(go.Scatter(x=hist.index, y=hist["bm_nav"], name="Benchmark NAV"))
            fig.update_layout(height=360, margin=dict(l=10, r=10, t=30, b=10))
            st.plotly_chart(fig, use_container_width=True)

    st.divider()

    st.markdown("### Holdings (Top 10)")
    h = get_wave_holdings(wave)
    if h is None or h.empty:
        st.info("Holdings unavailable.")
    else:
        hh = h.copy()
        if "Weight" in hh.columns:
            hh["Weight"] = pd.to_numeric(hh["Weight"], errors="coerce")
            hh = hh.sort_values("Weight", ascending=False).head(10)
            hh["Weight"] = hh["Weight"].map(lambda x: fmt_pct(x, 2))
        if "Ticker" in hh.columns:
            hh["Google"] = hh["Ticker"].astype(str).apply(lambda t: f"https://www.google.com/finance/quote/{t}")
        st.dataframe(hh, use_container_width=True)

    with st.expander("System Diagnostics", expanded=False):
        st.markdown("#### Environment")
        st.write(
            {
                "engine_present": we is not None,
                "engine_import_error": str(ENGINE_IMPORT_ERROR) if ENGINE_IMPORT_ERROR else None,
                "cwd": os.getcwd(),
                "files_in_root": sorted(os.listdir("."))[:80],
            }
        )
        st.markdown("#### History Debug")
        st.write(hdbg)

        st.markdown("#### Performance Files Found (first 50)")
        perf_files = _list_perf_files()
        st.write(perf_files[:50])

# -------------------------------
# Benchmark Truth tab (hooks + placeholders)
# -------------------------------
with tab_truth:
    st.markdown("## Benchmark Truth")
    st.info("Benchmark mix & difficulty can be validated at engine layer. (Hook ready.)")

    if we is None:
        st.caption("Engine not available in this environment.")
    else:
        # If you have a benchmark mix function, show it
        try:
            if hasattr(we, "get_benchmark_mix_table"):
                bm = we.get_benchmark_mix_table()
                if isinstance(bm, pd.DataFrame) and not bm.empty:
                    st.dataframe(bm, use_container_width=True)
        except Exception as e:
            st.warning(f"Benchmark mix table not available: {e}")

# -------------------------------
# Attribution tab (requires history + yfinance)
# -------------------------------
with tab_attr:
    st.markdown("## Alpha Attribution — Engine vs Static Basket (from Holdings)")
    st.caption("Static Basket is built from holdings weights using yfinance prices. It does NOT modify engine logic.")

    if hist is None or hist.empty or cov.get("rows", 0) < 30:
        st.warning("Not enough engine/log history to run attribution.")
    else:
        if yf is None:
            st.info("yfinance unavailable; attribution cannot run.")
        else:
            h = get_wave_holdings(wave)
            if h is None or h.empty or "Ticker" not in h.columns or "Weight" not in h.columns:
                st.info("Holdings unavailable for static basket attribution.")
            else:
                hh = h.copy()
                hh["Ticker"] = hh["Ticker"].astype(str).str.strip()
                hh["Weight"] = pd.to_numeric(hh["Weight"], errors="coerce").fillna(0.0)
                hh = hh.groupby("Ticker", as_index=False)["Weight"].sum()
                tot = float(hh["Weight"].sum())
                if tot > 0:
                    hh["Weight"] = hh["Weight"] / tot

                tickers = hh["Ticker"].tolist()
                px = fetch_prices_daily(tickers, days=min(days, 365))
                if px.empty:
                    st.info("Could not fetch prices for static basket.")
                else:
                    # compute weighted basket NAV
                    px = px.dropna(axis=1, how="all").ffill().bfill()
                    common = [t for t in tickers if t in px.columns]
                    if not common:
                        st.info("No overlapping tickers with price data.")
                    else:
                        wmap = {r.Ticker: float(r.Weight) for r in hh.itertuples(index=False)}
                        basket = None
                        for t in common:
                            s = px[t] / float(px[t].dropna().iloc[0])
                            basket = s * wmap.get(t, 0.0) if basket is None else basket + s * wmap.get(t, 0.0)
                        basket = basket / float(basket.dropna().iloc[0])
                        # align to engine history window
                        df = pd.DataFrame({"engine": hist["wave_nav"], "basket": basket}).dropna()
                        if len(df) < 30:
                            st.info("Not enough overlap between engine history and basket prices.")
                        else:
                            eng_r = df["engine"].pct_change().dropna()
                            bas_r = df["basket"].pct_change().dropna()
                            comp = pd.DataFrame({
                                "Engine 30D": [fmt_pct(ret_from_nav(df["engine"], min(31, len(df))))],
                                "Basket 30D": [fmt_pct(ret_from_nav(df["basket"], min(31, len(df))))],
                            })
                            st.dataframe(comp, use_container_width=True)

                            if go is not None:
                                fig = go.Figure()
                                fig.add_trace(go.Scatter(x=df.index, y=df["engine"], name="Engine NAV"))
                                fig.add_trace(go.Scatter(x=df.index, y=df["basket"], name="Static Basket NAV"))
                                fig.update_layout(height=360, margin=dict(l=10, r=10, t=30, b=10))
                                st.plotly_chart(fig, use_container_width=True)

# -------------------------------
# Doctor + What-If
# -------------------------------
with tab_doctor:
    st.markdown("## Wave Doctor")
    flags = cov.get("flags", []) if isinstance(cov, dict) else []
    if not flags:
        st.success("No integrity flags triggered.")
    else:
        st.warning("Data Integrity Flags: " + "; ".join(flags))

    st.divider()
    st.markdown("## What-If Lab (Shadow Simulation)")
    st.caption("Shadow simulation only. Engine state unchanged.")

    if hist is None or hist.empty or cov.get("rows", 0) < 60:
        st.info("Not enough history for What-If simulation.")
    else:
        shock = st.slider("One-day shock (%)", min_value=-20.0, max_value=20.0, value=0.0, step=0.5)
        nav0 = hist["wave_nav"].copy()
        sim = nav0.copy()
        sim.iloc[-1] = sim.iloc[-1] * (1.0 + shock / 100.0)

        if go is not None:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=nav0.index, y=nav0, name="Actual NAV"))
            fig.add_trace(go.Scatter(x=sim.index, y=sim, name="Shadow NAV"))
            fig.update_layout(height=320, margin=dict(l=10, r=10, t=30, b=10))
            st.plotly_chart(fig, use_container_width=True)

# ============================================================
# Secondary tabs row (Risk Lab / Correlation / Vector Insight)
# ============================================================
st.divider()
sub_whatif, sub_risk, sub_corr, sub_vector = st.tabs(
    ["+ What-If", "Risk Lab", "Correlation", "Vector OS Insight Layer"]
)

with sub_whatif:
    st.caption("Use Wave Doctor + What-If tab for shadow overlays (same engine-safe logic).")

with sub_risk:
    st.markdown("## Risk Lab")
    if hist is None or hist.empty or cov.get("rows", 0) < 50:
        st.warning("Not enough data to compute risk metrics.")
    else:
        sr = sharpe_ratio(wave_ret)
        v, cv = var_cvar(wave_ret, level=0.95)
        met = pd.DataFrame([{
            "Sharpe": fmt_num(sr, 2),
            "MaxDD": fmt_pct(mdd, 2),
            "VaR(95%)": fmt_pct(v, 2),
            "CVaR(95%)": fmt_pct(cv, 2),
            "TE": fmt_pct(te, 2),
            "IR": fmt_num(ir, 2),
        }])
        st.dataframe(met, use_container_width=True)

with sub_corr:
    st.markdown("## Correlation")
    st.caption("Correlation uses daily returns (mode + window). Raw series is hidden by default.")
    if not expose_raw:
        st.info("Correlation is hidden while raw series exposure is disabled.")
    else:
        if hist is None or hist.empty or cov.get("rows", 0) < 60:
            st.warning("Not enough history to compute correlation.")
        else:
            # Correlate Wave vs Benchmark returns
            df = pd.concat([wave_ret.rename("Wave"), bm_ret.rename("Benchmark")], axis=1).dropna()
            if df.empty:
                st.warning("No overlapping return series.")
            else:
                st.dataframe(df.corr(), use_container_width=True)

with sub_vector:
    st.markdown("## Vector OS Insight Layer")
    if hist is None or hist.empty or cov.get("rows", 0) < 60:
        st.info("Not enough data for insights yet.")
    else:
        msg = []
        if math.isfinite(a30) and a30 > 0:
            msg.append(f"30D alpha is positive ({fmt_pct(a30)}).")
        if math.isfinite(mdd) and mdd < -0.15:
            msg.append(f"Drawdown is elevated ({fmt_pct(mdd)}). Consider tightening risk.")
        if math.isfinite(te) and te > 0.15:
            msg.append(f"Tracking error is high ({fmt_pct(te)}).")
        if not msg:
            msg.append("No major flags. Monitor stability & benchmark drift.")
        st.success(" ".join(msg))