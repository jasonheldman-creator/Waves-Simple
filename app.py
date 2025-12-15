# app.py — WAVES Intelligence™ Institutional Console (Vector OS Edition)
# FULL PRODUCTION FILE (NO PATCHES) — PROTECTION-FIRST HARDENING + MODE-ALIAS FIX
#
# What this version does (PROTECT FIRST):
#   ✅ Access Gate (password) via st.secrets["WAVES_CONSOLE_KEY"] or env WAVES_CONSOLE_KEY
#   ✅ "Intelligence Boundary" wrapper:
#        - UI REQUESTS snapshots (engine should OWN intelligence)
#        - UI avoids exposing raw time series by default (redaction)
#   ✅ Keeps your tabs + diagnostics, but adds safety controls to reduce reverse-engineering surface
#   ✅ Mode alias handling (Standard/AMB/Private Logic variants)
#   ✅ Engine → CSV fallback remains (wave_history.csv) for resilience
#
# Notes:
#   • Engine math not modified here.
#   • To maximize defensibility, migrate intelligence into waves_engine.py:
#       we.get_wave_snapshot(wave, mode, days)
#       we.get_wave_holdings(wave, mode)
#   • This UI can still compute explanatory stats if you allow "Expose raw series".

from __future__ import annotations

import os
import math
import time
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
    import waves_engine as we
except Exception as e:
    we = None
    ENGINE_IMPORT_ERROR = e

# ============================================================
# MODE ALIASES (critical fix)
# ============================================================
MODE_ALIASES: Dict[str, List[str]] = {
    "Standard": [
        "Standard", "standard", "STANDARD",
        "Base", "BASE", "Normal", "NORMAL"
    ],
    "Alpha-Minus-Beta": [
        "Alpha-Minus-Beta", "alpha-minus-beta", "ALPHA-MINUS-BETA",
        "Alpha Minus Beta", "alpha minus beta",
        "AMB", "amb"
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
# Access Gate (PROTECTION FIRST)
# ============================================================
def _get_console_key() -> Optional[str]:
    # Prefer Streamlit secrets, then env var
    try:
        k = st.secrets.get("WAVES_CONSOLE_KEY", None)  # type: ignore
        if k:
            return str(k).strip()
    except Exception:
        pass
    k2 = os.getenv("WAVES_CONSOLE_KEY", "").strip()
    return k2 or None

def _hash(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

def access_gate() -> bool:
    """
    Simple password gate. Not bulletproof, but blocks casual scraping and most drive-by reverse engineering.
    """
    required = _get_console_key()
    if not required:
        # If you haven't set a key yet, we don't hard-stop — but we warn loudly.
        st.warning("⚠️ WAVES_CONSOLE_KEY not set. Console is not gated. Set st.secrets['WAVES_CONSOLE_KEY'] or env var.")
        return True

    if st.session_state.get("waves_authed", False):
        return True

    st.markdown("### 🔒 Secure Access")
    st.caption("Enter access key to view the WAVES Intelligence™ console.")
    pw = st.text_input("Access key", type="password")

    if pw:
        if _hash(pw.strip()) == _hash(required):
            st.session_state["waves_authed"] = True
            st.success("Access granted.")
            st.rerun()
        else:
            st.error("Access denied.")
    return False

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
# Basic return/risk math (EXPLANATION LAYER ONLY)
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
    w = safe_series(daily_wave).astype(float)
    b = safe_series(daily_bm).astype(float)
    df = pd.concat([w.rename("w"), b.rename("b")], axis=1).dropna()
    if df.shape[0] < 2:
        return float("nan")
    diff = df["w"] - df["b"]
    return float(diff.std() * np.sqrt(252))

def information_ratio(nav_wave: pd.Series, nav_bm: pd.Series, te: float) -> float:
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

def sortino_ratio(daily_ret: pd.Series, mar_annual: float = 0.0) -> float:
    r = safe_series(daily_ret).astype(float)
    if len(r) < 20:
        return float("nan")
    mar_daily = mar_annual / 252.0
    downside = np.minimum(0.0, (r - mar_daily).values)
    dd = float(np.sqrt(np.mean(downside**2))) * np.sqrt(252)
    if not math.isfinite(dd) or dd <= 0:
        return float("nan")
    ex = float((r - mar_daily).mean()) * 252.0
    return float(ex / dd)

def var_cvar(daily_ret: pd.Series, level: float = 0.95) -> Tuple[float, float]:
    r = safe_series(daily_ret).astype(float)
    if len(r) < 50:
        return (float("nan"), float("nan"))
    q = float(np.quantile(r.values, 1.0 - level))
    tail = r[r <= q]
    cvar = float(tail.mean()) if len(tail) else float("nan")
    return (q, cvar)

def drawdown_series(nav: pd.Series) -> pd.Series:
    nav = safe_series(nav).astype(float)
    if len(nav) < 2:
        return pd.Series(dtype=float)
    peak = nav.cummax()
    return ((nav / peak) - 1.0).rename("drawdown")

# ============================================================
# Data fetch (optional)
# ============================================================
@st.cache_data(show_spinner=False)
def fetch_prices_daily(tickers: List[str], days: int = 365) -> pd.DataFrame:
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

# ============================================================
# HISTORY LOADER (engine → CSV fallback) — still needed
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
        if low in ["wave_nav", "nav_wave", "portfolio_nav", "nav", "portfolio_nav_usd"]:
            ren[c] = "wave_nav"
        if low in ["bm_nav", "bench_nav", "benchmark_nav", "benchmark_nav_usd"]:
            ren[c] = "bm_nav"
        if low in ["wave_ret", "ret_wave", "portfolio_ret", "portfolio_return"]:
            ren[c] = "wave_ret"
        if low in ["bm_ret", "ret_bm", "benchmark_ret", "benchmark_return"]:
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

def _loose_eq(a: str, b: str) -> bool:
    return str(a).strip().lower() == str(b).strip().lower()

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
        df = df[df[wc].astype(str).apply(lambda x: _loose_eq(x, wave_name))]

    if mc:
        cands = mode_candidates(mode)
        df = df[df[mc].astype(str).apply(lambda x: any(_loose_eq(x, m) for m in cands))]

    if dc:
        df[dc] = pd.to_datetime(df[dc], errors="coerce")
        df = df.dropna(subset=[dc]).sort_values(dc).set_index(dc)

    out = _standardize_history(df)
    return out.iloc[-days:] if len(out) > days else out

# ============================================================
# Intelligence Boundary Wrapper (UI -> engine snapshot)
# ============================================================
def _engine_call_snapshot(wave_name: str, mode: str, days: int) -> Optional[Dict[str, Any]]:
    if we is None:
        return None
    if hasattr(we, "get_wave_snapshot"):
        # Try aliases
        for m in mode_candidates(mode):
            try:
                snap = we.get_wave_snapshot(wave_name, m, days=days)  # type: ignore
                if isinstance(snap, dict):
                    return snap
            except TypeError:
                try:
                    snap = we.get_wave_snapshot(wave_name, m, days)  # type: ignore
                    if isinstance(snap, dict):
                        return snap
                except Exception:
                    pass
            except Exception:
                pass
    return None

@st.cache_data(show_spinner=False)
def compute_wave_history(wave_name: str, mode: str, days: int = 365, force_csv: bool = False) -> pd.DataFrame:
    if force_csv:
        return history_from_csv(wave_name, mode, days)

    snap = _engine_call_snapshot(wave_name, mode, days)
    if isinstance(snap, dict):
        h = snap.get("history", None)
        if isinstance(h, pd.DataFrame) and not h.empty:
            return _standardize_history(h).iloc[-days:]

    # fallback to legacy engine methods if present
    if we is not None:
        cands = mode_candidates(mode)

        if hasattr(we, "compute_history_nav"):
            for m in cands:
                try:
                    df = we.compute_history_nav(wave_name, mode=m, days=days)  # type: ignore
                    df = _standardize_history(df)
                    if not df.empty:
                        return df
                except TypeError:
                    try:
                        df = we.compute_history_nav(wave_name, m, days)  # type: ignore
                        df = _standardize_history(df)
                        if not df.empty:
                            return df
                    except Exception:
                        pass
                except Exception:
                    pass

        alt = ["get_history_nav", "get_wave_history", "history_nav", "compute_nav_history", "compute_history"]
        for fn in alt:
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
            waves = we.get_all_waves()  # type: ignore
            if isinstance(waves, (list, tuple)):
                out = [str(x).strip() for x in waves if str(x).strip()]
                out = [w for w in out if w.lower() != "nan"]
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
            df = we.get_benchmark_mix_table()  # type: ignore
            if isinstance(df, pd.DataFrame):
                return df
        except Exception:
            pass
    return pd.DataFrame(columns=["Wave", "Ticker", "Name", "Weight"])

@st.cache_data(show_spinner=False)
def get_wave_holdings(wave_name: str, mode: str) -> pd.DataFrame:
    # Prefer a mode-aware holdings call if you add it to engine
    if we is not None and hasattr(we, "get_wave_holdings"):
        for m in mode_candidates(mode):
            try:
                df = we.get_wave_holdings(wave_name, m)  # type: ignore
                if isinstance(df, pd.DataFrame) and not df.empty:
                    return df
            except TypeError:
                try:
                    df = we.get_wave_holdings(wave_name)  # type: ignore
                    if isinstance(df, pd.DataFrame) and not df.empty:
                        return df
                except Exception:
                    pass
            except Exception:
                pass

    # fallback to wave_weights.csv
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
# Benchmark snapshot + drift tracking (safe to keep)
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
    return df.sort_values(["Ticker"]).reset_index(drop=True)[["Ticker", "Weight"]]

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
# Alpha Heatmap (kept, but can be redacted)
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
# MAIN
# ============================================================
st.title("WAVES Intelligence™ Institutional Console")

# Gate BEFORE anything else renders
if not access_gate():
    st.stop()

# Engine import status
if ENGINE_IMPORT_ERROR is not None:
    st.error("Engine import failed. The app will use CSV fallbacks where possible.")
    st.code(str(ENGINE_IMPORT_ERROR))

all_waves = get_all_waves_safe()
if not all_waves:
    st.warning("No waves discovered yet. Check engine import + config files.")
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

    # Protection toggles
    st.subheader("Security")
    expose_raw = st.toggle("Expose raw series (advanced)", value=False)
    force_csv = st.toggle("Force CSV history (debug/demo)", value=False)
    st.caption("Leave raw series OFF for demos. This reduces reverse-engineering surface.")

# Light throttle (anti-scrape)
now = time.time()
last = st.session_state.get("last_request_ts", 0.0)
if now - last < 0.35:
    time.sleep(0.35 - (now - last))
st.session_state["last_request_ts"] = time.time()

bm_mix = get_benchmark_mix()
bm_id = benchmark_snapshot_id(selected_wave, bm_mix)
bm_drift = benchmark_drift_status(selected_wave, mode, bm_id)

hist = compute_wave_history(selected_wave, mode=mode, days=days, force_csv=force_csv)
cov = coverage_report(hist)

# Sticky summary bar metrics (aggregated)
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

chips = []
chips.append(f"BM Snapshot: {bm_id} · {'Stable' if bm_drift=='stable' else 'DRIFT'}")
chips.append(f"Coverage: {fmt_num(cov.get('completeness_score', np.nan), 1)} / 100")
chips.append(f"Rows: {cov.get('rows','—')} · Age: {cov.get('age_days','—')}")
chips.append(f"Regime: {regime}")
chips.append(f"VIX: {fmt_num(vix_val,1) if math.isfinite(vix_val) else '—'}")
chips.append(f"30D α: {fmt_pct(a30)} · 30D r: {fmt_pct(r30)}")
chips.append(f"365D α: {fmt_pct(a365)} · 365D r: {fmt_pct(r365)}")
chips.append(f"TE: {fmt_pct(te)} · IR: {fmt_num(ir,2)}")
chips.append(f"MaxDD: {fmt_pct(mdd)}")

st.markdown('<div class="waves-sticky">', unsafe_allow_html=True)
for c in chips:
    st.markdown(f'<span class="waves-chip">{c}</span>', unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

tabs = st.tabs(["Console", "Risk Lab", "Correlation", "Vector OS Insight Layer", "System Diagnostics"])

# -------------------------
# Console
# -------------------------
with tabs[0]:
    st.subheader("Alpha Heatmap View (All Waves × Timeframe)")
    alpha_df = build_alpha_matrix(all_waves, mode=mode, days=min(days, 365), force_csv=force_csv)
    plot_alpha_heatmap(alpha_df, selected_wave, title=f"Alpha Heatmap — Mode: {mode}")

    st.subheader("Coverage & Data Integrity")
    if cov.get("rows", 0) == 0:
        st.warning("No history returned for this wave/mode. Engine → CSV fallback attempted (with mode aliases).")
        st.caption(f"Mode candidates tried: {mode_candidates(mode)}")

    c1, c2, c3 = st.columns(3)
    c1.metric("History Rows", cov.get("rows", 0))
    c2.metric("Last Data Age (days)", cov.get("age_days", "—"))
    c3.metric("Completeness Score", fmt_num(cov.get("completeness_score", np.nan), 1))

    with st.expander("Coverage Details"):
        st.write(cov)

    st.subheader("Holdings (Top 10)")
    hold = get_wave_holdings(selected_wave, mode)
    if hold.empty:
        st.info("Holdings unavailable (engine did not return holdings and wave_weights.csv mapping did not match).")
    else:
        hold2 = hold.copy()
        if "Weight" in hold2.columns:
            hold2["Weight"] = pd.to_numeric(hold2["Weight"], errors="coerce").fillna(0.0)
        if "Ticker" in hold2.columns:
            hold2["Google"] = hold2["Ticker"].astype(str).apply(lambda t: f"https://www.google.com/finance/quote/{t}")
        st.dataframe(hold2.head(10), use_container_width=True)

    if expose_raw:
        with st.expander("Raw History (advanced)"):
            st.dataframe(hist.tail(200), use_container_width=True)
    else:
        st.caption("Raw series hidden (recommended for demos / defensibility).")

# -------------------------
# Risk Lab
# -------------------------
with tabs[1]:
    st.subheader("Risk Lab")
    if hist.empty or len(hist) < 50:
        st.info("Not enough data to compute risk lab metrics.")
    else:
        r = hist["wave_ret"].dropna()
        sh = sharpe_ratio(r, 0.0)
        so = sortino_ratio(r, 0.0)
        v95, c95 = var_cvar(r, 0.95)
        mdd_b = max_drawdown(hist["bm_nav"])

        a, b, c, d = st.columns(4)
        a.metric("Sharpe (0% rf)", fmt_num(sh, 2))
        b.metric("Sortino (0% MAR)", fmt_num(so, 2))
        c.metric("VaR 95% (daily)", fmt_pct(v95))
        d.metric("CVaR 95% (daily)", fmt_pct(c95))

        if expose_raw:
            st.write("Drawdown (Wave vs Benchmark)")
            dd_w = drawdown_series(hist["wave_nav"])
            dd_b = drawdown_series(hist["bm_nav"])
            dd_df = pd.concat([dd_w.rename("Wave"), dd_b.rename("Benchmark")], axis=1).dropna()
            st.line_chart(dd_df)
        else:
            st.caption("Charts hidden because raw series is disabled. Enable 'Expose raw series' to view.")

# -------------------------
# Correlation
# -------------------------
with tabs[2]:
    st.subheader("Correlation")
    st.caption("Correlation uses daily returns (mode + window). Disable raw series to reduce surface.")

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

# -------------------------
# Vector OS Insight Layer
# -------------------------
with tabs[3]:
    st.subheader("Vector OS Insight Layer")
    if hist.empty or len(hist) < 20:
        st.info("Not enough data for insights yet.")
    else:
        notes = []
        if cov.get("flags"):
            notes.append("**Data Integrity Flags:** " + "; ".join(cov["flags"]))
        if bm_drift != "stable":
            notes.append("**Benchmark Drift:** Snapshot changed — freeze benchmark mix for demos.")
        if math.isfinite(a30) and abs(a30) >= 0.08:
            notes.append("**Large 30D alpha:** verify benchmark mix + missing days; big alpha can be real or coverage-driven.")
        if math.isfinite(te) and te >= 0.20:
            notes.append("**High tracking error:** active risk elevated vs benchmark.")
        if math.isfinite(mdd) and mdd <= -0.25:
            notes.append("**Deep drawdown:** consider stronger SmartSafe posture in stress regimes.")
        if not expose_raw:
            notes.append("**Security mode:** raw series hidden (recommended).")

        for n in (notes or ["No major anomalies detected on this window."]):
            st.markdown(f"- {n}")

# -------------------------
# Diagnostics
# -------------------------
with tabs[4]:
    st.subheader("System Diagnostics")
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