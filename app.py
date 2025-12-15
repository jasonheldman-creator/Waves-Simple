# app.py — WAVES Intelligence™ Institutional Console (Vector OS Edition)
# FULL PRODUCTION FILE (NO PATCHES) — STABLE UI + HISTORY FALLBACKS + SYNTHETIC HISTORY
#
# Fixes / Upgrades
#   ✅ Removes corrupted / duplicated blocks + stray `if` syntax break
#   ✅ Robust history loader:
#        1) engine functions (multiple signatures + mode aliases)
#        2) wave_history.csv fallback (mode alias match)
#        3) SYNTHETIC HISTORY fallback (builds NAVs from holdings + yfinance)
#   ✅ Shows clear diagnostics (optionally hidden in PRODUCTION_MODE)
#
# Keeps your structure
#   • Sticky summary bar
#   • Alpha Heatmap
#   • Risk Lab (Sharpe/Sortino/VaR/CVaR/Drawdown/Rolling)
#   • Correlation matrix (raw series exposure toggle)
#   • Vector OS Insight Layer
#
# Notes
#   • Engine math is NOT modified.
#   • Synthetic history is a *display fallback* only, used when engine/CSV history is missing.
#   • If your repo is public, source can be copied. Real protection = keep engine/data private.

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
# Security / surface toggles
# ============================================================
# If True: hides deep internals + avoids overly-helpful debug dumps.
PRODUCTION_MODE = True

# ============================================================
# MODE ALIASES (critical fix)
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
.waves-sticky {
  position: sticky;
  top: 0;
  z-index: 999;
  backdrop-filter: blur(10px);
  -webkit-backdrop-filter: blur(10px);
  padding: 10px 12px;
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
div[data-testid="stDataFrame"] { border-radius: 12px; overflow: hidden; }
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
# Data fetch
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

    if data is None or isinstance(data, (int, float)) or getattr(data, "empty", True):
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
    return data

# ============================================================
# HISTORY LOADER (engine → CSV → synthetic)
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
        low = str(c).strip().lower()
        if low in ["wave_nav", "nav_wave", "portfolio_nav", "wavevalue", "nav", "portfolio_nav_usd", "wave value"]:
            ren[c] = "wave_nav"
        if low in ["bm_nav", "bench_nav", "benchmark_nav", "benchmark_nav_usd", "benchmark value", "bm value"]:
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

def _loose_eq(a: str, b: str) -> bool:
    return str(a).strip().lower() == str(b).strip().lower()

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

# ----------------------------
# Synthetic history fallback
# ----------------------------
def _weights_dict(df: pd.DataFrame) -> Dict[str, float]:
    if df is None or df.empty:
        return {}
    if "Ticker" not in df.columns:
        return {}
    wcol = "Weight" if "Weight" in df.columns else None
    if wcol is None:
        return {}
    tmp = df.copy()
    tmp["Ticker"] = tmp["Ticker"].astype(str).str.upper().str.strip()
    tmp[wcol] = pd.to_numeric(tmp[wcol], errors="coerce").fillna(0.0)
    tmp = tmp.groupby("Ticker", as_index=False)[wcol].sum()
    s = float(tmp[wcol].sum())
    if s <= 0:
        return {}
    tmp[wcol] = tmp[wcol] / s
    return {r.Ticker: float(r.Weight) for r in tmp.itertuples(index=False)}

@st.cache_data(show_spinner=False)
def synthetic_history_from_holdings(wave_name: str, days: int = 365) -> pd.DataFrame:
    """
    Build wave NAV from holdings weights using yfinance daily prices.
    Benchmark NAV from benchmark mix weights if available; else SPY.
    """
    if yf is None:
        return pd.DataFrame(columns=["wave_nav", "bm_nav", "wave_ret", "bm_ret"])

    hold = get_wave_holdings(wave_name)
    w_w = _weights_dict(hold)
    if not w_w:
        return pd.DataFrame(columns=["wave_nav", "bm_nav", "wave_ret", "bm_ret"])

    bm_mix = get_benchmark_mix()
    bm_rows = pd.DataFrame()
    if bm_mix is not None and not bm_mix.empty and "Wave" in bm_mix.columns:
        bm_rows = bm_mix[bm_mix["Wave"] == wave_name].copy()

    if bm_rows is not None and not bm_rows.empty and {"Ticker", "Weight"}.issubset(set(bm_rows.columns)):
        b_w = _weights_dict(bm_rows.rename(columns={"Weight": "Weight", "Ticker": "Ticker"}))
    else:
        b_w = {"SPY": 1.0}

    tickers = sorted(list(set(list(w_w.keys()) + list(b_w.keys()))))
    px = fetch_prices_daily(tickers, days=days)
    if px is None or px.empty:
        return pd.DataFrame(columns=["wave_nav", "bm_nav", "wave_ret", "bm_ret"])

    # ensure all tickers present
    for t in tickers:
        if t not in px.columns:
            px[t] = np.nan
    px = px[tickers].ffill().bfill().dropna(how="all")
    if px.empty or len(px) < 5:
        return pd.DataFrame(columns=["wave_nav", "bm_nav", "wave_ret", "bm_ret"])

    rets = px.pct_change().fillna(0.0)

    wave_ret = pd.Series(0.0, index=rets.index)
    for t, w in w_w.items():
        if t in rets.columns:
            wave_ret = wave_ret + rets[t] * float(w)

    bm_ret = pd.Series(0.0, index=rets.index)
    for t, w in b_w.items():
        if t in rets.columns:
            bm_ret = bm_ret + rets[t] * float(w)

    wave_nav = (1.0 + wave_ret).cumprod()
    bm_nav = (1.0 + bm_ret).cumprod()

    out = pd.DataFrame(
        {"wave_nav": wave_nav, "bm_nav": bm_nav, "wave_ret": wave_ret, "bm_ret": bm_ret},
        index=rets.index,
    )
    out = out.dropna(how="all")
    return out

@st.cache_data(show_spinner=False)
def compute_wave_history(wave_name: str, mode: str, days: int = 365, force_csv: bool = False) -> pd.DataFrame:
    """
    Robust history fetch:
      1) engine compute_history_nav / variants (mode aliases)
      2) CSV fallback wave_history.csv
      3) SYNTHETIC fallback (holdings + yfinance)
    """
    if force_csv:
        h = history_from_csv(wave_name, mode, days)
        if h is not None and not h.empty:
            return h
        return synthetic_history_from_holdings(wave_name, days=days)

    # 0) If engine missing, try CSV then synthetic
    if we is None:
        h = history_from_csv(wave_name, mode, days)
        if h is not None and not h.empty:
            return h
        return synthetic_history_from_holdings(wave_name, days=days)

    cands = mode_candidates(mode)

    # 1) Primary expected function
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

    # 2) Alternate engine functions
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

    # 3) CSV fallback
    h = history_from_csv(wave_name, mode, days)
    if h is not None and not h.empty:
        return h

    # 4) Synthetic fallback
    return synthetic_history_from_holdings(wave_name, days=days)

# ============================================================
# WaveScore (console-side approximation)
# ============================================================
def _grade_from_score(score: float) -> str:
    if score is None or (isinstance(score, float) and math.isnan(score)):
        return "N/A"
    if score >= 90: return "A+"
    if score >= 80: return "A"
    if score >= 70: return "B"
    if score >= 60: return "C"
    return "D"

@st.cache_data(show_spinner=False)
def compute_wavescore_for_all_waves(all_waves: List[str], mode: str, days: int = 365, force_csv: bool = False) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for wave in all_waves:
        hist = compute_wave_history(wave, mode=mode, days=days, force_csv=force_csv)
        if hist.empty or len(hist) < 20:
            rows.append({"Wave": wave, "WaveScore": np.nan, "Grade": "N/A", "IR_365D": np.nan, "Alpha_365D": np.nan})
            continue

        nav_wave = hist["wave_nav"]
        nav_bm = hist["bm_nav"]
        wave_ret = hist["wave_ret"]
        bm_ret = hist["bm_ret"]

        alpha_365 = ret_from_nav(nav_wave, len(nav_wave)) - ret_from_nav(nav_bm, len(nav_bm))
        te = tracking_error(wave_ret, bm_ret)
        ir = information_ratio(nav_wave, nav_bm, te)
        mdd_wave = max_drawdown(nav_wave)
        hit_rate = float((wave_ret >= bm_ret).mean()) if len(wave_ret) else np.nan

        rq = float(np.clip((np.nan_to_num(ir) / 1.5), 0.0, 1.0) * 25.0)
        rc = float(np.clip(1.0 - (abs(np.nan_to_num(mdd_wave)) / 0.35), 0.0, 1.0) * 25.0)
        co = float(np.clip(np.nan_to_num(hit_rate), 0.0, 1.0) * 15.0)
        rs = float(np.clip(1.0 - (abs(np.nan_to_num(te)) / 0.25), 0.0, 1.0) * 15.0)
        tr = 10.0

        total = float(np.clip(rq + rc + co + rs + tr, 0.0, 100.0))
        rows.append({"Wave": wave, "WaveScore": total, "Grade": _grade_from_score(total), "IR_365D": ir, "Alpha_365D": alpha_365})

    df = pd.DataFrame(rows) if rows else pd.DataFrame()
    return df.sort_values("Wave") if not df.empty else df

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
# Main UI
# ============================================================
st.title("WAVES Intelligence™ Institutional Console")

if ENGINE_IMPORT_ERROR is not None:
    st.warning("Engine import failed. The app will use CSV/synthetic fallbacks where possible.")
    if not PRODUCTION_MODE:
        st.code(str(ENGINE_IMPORT_ERROR))

all_waves = get_all_waves_safe()
if not all_waves:
    st.error("No waves discovered. Check: engine import, wave_config.csv, wave_weights.csv, list.csv.")
    if not PRODUCTION_MODE:
        st.write("Files present:", {p: os.path.exists(p) for p in ["wave_config.csv", "wave_weights.csv", "wave_history.csv", "list.csv", "waves_engine.py"]})
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

# Sticky summary bar
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

ws_df = compute_wavescore_for_all_waves(all_waves, mode=mode, days=min(days, 365), force_csv=force_csv)
rank = None
ws_val = np.nan
if not ws_df.empty and selected_wave in set(ws_df["Wave"]):
    ws_val = float(ws_df[ws_df["Wave"] == selected_wave]["WaveScore"].iloc[0])
    ws_df_sorted = ws_df.sort_values("WaveScore", ascending=False, na_position="last").reset_index(drop=True)
    try:
        rank = int(ws_df_sorted.index[ws_df_sorted["Wave"] == selected_wave][0] + 1)
    except Exception:
        rank = None

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

tabs = st.tabs(["Console", "Factor Decomposition", "Risk Lab", "Correlation", "Vector OS Insight Layer"])

# -------------------------
# Console
# -------------------------
with tabs[0]:
    st.subheader("Alpha Heatmap View (All Waves × Timeframe)")
    alpha_df = build_alpha_matrix(all_waves, mode=mode, days=min(days, 365), force_csv=force_csv)
    plot_alpha_heatmap(alpha_df, selected_wave, title=f"Alpha Heatmap — Mode: {mode}")

    st.subheader("Coverage & Data Integrity")
    if cov["rows"] == 0:
        st.warning("No history returned for this wave/mode. Tried engine → CSV → synthetic fallback.")
        st.caption(f"Mode candidates tried: {mode_candidates(mode)}")
        if yf is None:
            st.info("yfinance is not available, so synthetic history cannot run on this deployment.")

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
# Factor Decomposition
# -------------------------
with tabs[1]:
    st.subheader("Factor Decomposition (Light)")
    st.caption("Currently shows beta vs benchmark (can be extended to multi-factor regression).")

    if hist.empty or len(hist) < 20:
        st.info("Not enough history.")
    else:
        b = beta_ols(hist["wave_ret"], hist["bm_ret"])
        st.metric("Beta vs Benchmark", fmt_num(b, 2))
        st.write("Tip: Add factor series (SPY/QQQ/IWM/TLT/GLD/BTC) for multi-factor regression.")

# -------------------------
# Risk Lab
# -------------------------
with tabs[2]:
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
with tabs[3]:
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
with tabs[4]:
    st.subheader("Vector OS Insight Layer")
    if hist.empty or len(hist) < 20:
        st.info("Not enough data for insights yet.")
    else:
        notes = []
        if cov.get("flags"):
            notes.append("**Data Integrity Flags:** " + "; ".join(cov["flags"]))
        if bm_drift != "stable":
            notes.append("**Benchmark Drift:** Snapshot changed during session — freeze benchmark mix for demos.")
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

# Footer diagnostics
if not PRODUCTION_MODE:
    with st.expander("System Diagnostics (debug)"):
        st.write("Engine loaded:", we is not None)
        st.write("Engine import error:", str(ENGINE_IMPORT_ERROR) if ENGINE_IMPORT_ERROR else "None")
        st.write("Mode selected:", mode)
        st.write("Mode candidates tried:", mode_candidates(mode))
        st.write("Force CSV:", force_csv)
        st.write("Expose raw:", expose_raw)
        st.write("Files present:", {p: os.path.exists(p) for p in ["wave_config.csv", "wave_weights.csv", "wave_history.csv", "list.csv", "waves_engine.py"]})
        st.write("Selected:", {"wave": selected_wave, "days": days})
        st.write("History shape:", None if hist is None else hist.shape)
        if hist is not None and not hist.empty:
            st.write("History columns:", list(hist.columns))
            st.write("History tail:", hist.tail(3))