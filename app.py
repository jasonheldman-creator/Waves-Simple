# app.py — WAVES Intelligence™ Institutional Console (Vector OS Edition)
# FULL PRODUCTION FILE (NO PATCHES) — v1 CLEAN DEMO MODE + DEFINITIONS INDEX + SAFE DIAGNOSTICS
#
# Goals of this build:
#   1) Prevent Streamlit indentation/syntax breakage by keeping Diagnostics inside a function,
#      and keeping all long UI blocks well-scoped.
#   2) Reduce clutter with “Advanced Analytics” toggle:
#        - OFF (default): IC Summary + Overview (All Waves) + Wave Detail + Attribution + Definitions
#        - ON: adds Risk Lab, Correlation, Mode Proof, Drawdown Monitor, Exports, Decision tabs, Diagnostics
#   3) Add David’s request: Definitions/short explanations for each summary chip + key metrics:
#        - Sidebar “Metric Definitions” expander (always available)
#        - Full “Definitions & Methodology” tab
#
# Notes:
#   • Engine math is NOT modified.
#   • Robust history loader: engine functions → wave_history.csv fallback
#   • App will not crash if decision_engine.py is missing.
#   • All UI sections are defensive: never blank-screen.
#
# Expected optional files (if present):
#   - waves_engine.py
#   - decision_engine.py (optional)
#   - wave_history.csv (optional fallback)
#   - wave_weights.csv / list.csv (optional fallback to discover waves)

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

# -------------------------------
# Decision Engine import (guarded)
# -------------------------------
DECISION_IMPORT_ERROR = None
try:
    from decision_engine import generate_decisions  # optional
except Exception as e:
    generate_decisions = None
    DECISION_IMPORT_ERROR = e

try:
    from decision_engine import build_daily_wave_activity  # optional
except Exception as e:
    build_daily_wave_activity = None
    if DECISION_IMPORT_ERROR is None:
        DECISION_IMPORT_ERROR = e

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

/* Section label */
.waves-section-label {
  font-size: 0.85rem;
  opacity: 0.75;
  margin: 0 0 6px 0;
}

/* Tighter tables */
div[data-testid="stDataFrame"] { border-radius: 12px; overflow: hidden; }

/* Reduce whitespace for mobile */
@media (max-width: 700px) {
  .block-container { padding-left: 0.8rem; padding-right: 0.8rem; }
}
</style>
""",
    unsafe_allow_html=True,
)

# ============================================================
# Metric Definitions (David “definitions index”)
# ============================================================
METRIC_DEFINITIONS: Dict[str, str] = {
    "BM Snapshot": (
        "A locked identifier for the benchmark composition used in this session. "
        "Helps detect benchmark drift (changes in the benchmark mix)."
    ),
    "Benchmark Drift": (
        "Whether the benchmark snapshot changed during the session. "
        "Stable = consistent benchmark; Drift = benchmark changed and should be frozen for IC review."
    ),
    "Rows": "Number of historical datapoints loaded for the selected Wave & Mode (typically business days).",
    "Age (days)": "Days since the most recent datapoint. Lower = fresher; higher indicates stale updates.",
    "Coverage Score": (
        "A 0–100 completeness score based on missing business days + freshness. "
        "100 means fresh and complete for the loaded window."
    ),
    "Confidence": (
        "Human-friendly label derived from Coverage Score + Age. "
        "High confidence means the history window is fresh and complete."
    ),
    "Regime": (
        "A simple regime label inferred from volatility (VIX): risk-on / neutral / risk-off. "
        "Used for narrative framing only (not a trading signal)."
    ),
    "VIX": "CBOE Volatility Index (proxy for market stress). Higher generally implies risk-off conditions.",
    "Return (1D/30D/60D/365D)": "Total return of the Wave over the window, computed from NAV.",
    "Alpha (1D/30D/60D/365D)": "Excess return vs benchmark over the window. Positive alpha = outperformance.",
    "TE (Tracking Error)": (
        "Annualized volatility of excess returns vs benchmark (active risk). "
        "Higher TE means more deviation from the benchmark."
    ),
    "IR (Information Ratio)": (
        "Risk-adjusted alpha efficiency (alpha per unit of tracking error). "
        "Higher IR = more efficient outperformance."
    ),
    "Max Drawdown": "Largest peak-to-trough decline in NAV over the history window.",
    "WaveScore": (
        "Composite 0–100 quality score (console-side approximation in this build). "
        "Summarizes return quality, drawdown control, and consistency."
    ),
    "Rank": "WaveScore rank vs the current Wave universe (1 = best).",
}

def render_metric_definitions(full: bool = False) -> None:
    st.markdown("### Metric Definitions (Index)")
    st.caption("Short, IC-grade definitions for the summary chips and key metrics.")
    keys = [
        "BM Snapshot", "Benchmark Drift",
        "Rows", "Age (days)", "Coverage Score", "Confidence",
        "Regime", "VIX",
        "Return (1D/30D/60D/365D)", "Alpha (1D/30D/60D/365D)",
        "TE (Tracking Error)", "IR (Information Ratio)",
        "Max Drawdown",
        "WaveScore", "Rank",
    ]
    for k in keys:
        st.markdown(f"**{k}** — {METRIC_DEFINITIONS.get(k, '')}")

    if full:
        st.divider()
        st.markdown("### Methodology Notes (High level)")
        st.markdown(
            "- NAV and Benchmark NAV are compared on the same dates.\n"
            "- Alpha is computed as (Wave return − Benchmark return) over the chosen window.\n"
            "- Tracking Error is annualized stdev of daily excess returns.\n"
            "- IR is total-window alpha divided by TE (if TE>0).\n"
            "- Coverage Score penalizes missing business days and staleness.\n"
            "- WaveScore here is a console-side approximation for ranking/demos (not the locked WAVESCORE™ spec).\n"
        )

# ============================================================
# Helpers: formatting
# ============================================================
def fmt_pct(x: Any, digits: int = 2) -> str:
    """Input is decimal (0.10), output is '10.00%'."""
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

def daily_returns_from_nav(nav: pd.Series) -> pd.Series:
    nav = safe_series(nav).astype(float)
    if len(nav) < 2:
        return pd.Series(dtype=float)
    return nav.pct_change().rename("ret")

def drawdown_series(nav: pd.Series) -> pd.Series:
    nav = safe_series(nav).astype(float)
    if len(nav) < 2:
        return pd.Series(dtype=float)
    peak = nav.cummax()
    return (nav / peak - 1.0).rename("drawdown")

def var_cvar(daily_ret: pd.Series, level: float = 0.95) -> Tuple[float, float]:
    r = safe_series(daily_ret).dropna().astype(float)
    if len(r) < 50:
        return (float("nan"), float("nan"))
    q = float(np.quantile(r.values, 1.0 - level))
    tail = r[r <= q]
    cvar = float(tail.mean()) if len(tail) else float("nan")
    return (q, cvar)

# ============================================================
# Optional data fetch (yfinance) — used for VIX chip
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

    if isinstance(getattr(data, "columns", None), pd.MultiIndex):
        if "Adj Close" in data.columns.get_level_values(0):
            data = data["Adj Close"]
        elif "Close" in data.columns.get_level_values(0):
            data = data["Close"]
        else:
            data = data[data.columns.levels[0][0]]

    if isinstance(getattr(data, "columns", None), pd.MultiIndex):
        data = data.droplevel(0, axis=1)

    if isinstance(data, pd.Series):
        data = data.to_frame()

    data = data.sort_index().ffill().bfill()
    if len(data) > days:
        data = data.iloc[-days:]
    return data

# ============================================================
# HISTORY LOADER (engine → CSV fallback)
# ============================================================
def _standardize_history(df: pd.DataFrame) -> pd.DataFrame:
    """
    Expected:
      index=datetime
      columns: wave_nav, bm_nav, wave_ret, bm_ret
    """
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
        if low in ["wave_nav", "nav_wave", "portfolio_nav", "nav", "wave value", "wavevalue"]:
            ren[c] = "wave_nav"
        if low in ["bm_nav", "bench_nav", "benchmark_nav", "benchmark", "bm value", "benchmark value"]:
            ren[c] = "bm_nav"
        if low in ["wave_ret", "ret_wave", "portfolio_ret", "return", "wave_return", "wave return"]:
            ren[c] = "wave_ret"
        if low in ["bm_ret", "ret_bm", "benchmark_ret", "bm_return", "benchmark_return", "benchmark return"]:
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
        df = df[df[wc] == str(wave_name)]
    if mc:
        df[mc] = df[mc].astype(str)
        df = df[df[mc].str.lower() == str(mode).lower()]
    if dc:
        df[dc] = pd.to_datetime(df[dc], errors="coerce")
        df = df.dropna(subset=[dc]).sort_values(dc).set_index(dc)

    out = _standardize_history(df)
    if len(out) > days:
        out = out.iloc[-days:]
    return out

@st.cache_data(show_spinner=False)
def compute_wave_history(wave_name: str, mode: str, days: int = 365) -> pd.DataFrame:
    if we is not None:
        if hasattr(we, "compute_history_nav"):
            try:
                df = we.compute_history_nav(wave_name, mode=mode, days=days)
                df = _standardize_history(df)
                if not df.empty:
                    return df
            except TypeError:
                try:
                    df = we.compute_history_nav(wave_name, mode, days)
                    df = _standardize_history(df)
                    if not df.empty:
                        return df
                except Exception:
                    pass
            except Exception:
                pass

        candidates = ["get_history_nav", "get_wave_history", "history_nav", "compute_nav_history", "compute_history"]
        for fn in candidates:
            if hasattr(we, fn):
                f = getattr(we, fn)
                try:
                    try:
                        df = f(wave_name, mode=mode, days=days)
                    except TypeError:
                        df = f(wave_name, mode, days)
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
                waves = [str(x) for x in waves]
                return sorted([w for w in waves if w and w.lower() != "nan"])
        except Exception:
            pass

    for p in ["wave_config.csv", "wave_weights.csv", "list.csv"]:
        if os.path.exists(p):
            try:
                df = pd.read_csv(p)
                for col in ["Wave", "wave", "wave_name"]:
                    if col in df.columns:
                        waves = sorted(list(set(df[col].astype(str).tolist())))
                        return [w for w in waves if w and w.lower() != "nan"]
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
            if isinstance(df, pd.DataFrame) and not df.empty:
                out = df.copy()
                if "Ticker" not in out.columns:
                    for alt in ["ticker", "symbol", "Symbol"]:
                        if alt in out.columns:
                            out["Ticker"] = out[alt].astype(str)
                            break
                if "Weight" not in out.columns:
                    for alt in ["weight", "w", "WeightPct"]:
                        if alt in out.columns:
                            out["Weight"] = out[alt]
                            break
                if "Name" not in out.columns:
                    out["Name"] = ""
                out["Ticker"] = out["Ticker"].astype(str).str.upper().str.strip()
                out["Weight"] = pd.to_numeric(out["Weight"], errors="coerce").fillna(0.0)
                tot = float(out["Weight"].sum())
                if tot > 0:
                    out["Weight"] = out["Weight"] / tot
                out = out[["Ticker", "Name", "Weight"]].sort_values("Weight", ascending=False).reset_index(drop=True)
                return out
        except Exception:
            pass

    if os.path.exists("wave_weights.csv"):
        try:
            df = pd.read_csv("wave_weights.csv")
            cols = {c.lower(): c for c in df.columns}
            if {"wave", "ticker", "weight"}.issubset(set(cols.keys())):
                wf = df[df[cols["wave"]].astype(str) == str(wave_name)].copy()
                if wf.empty:
                    return pd.DataFrame(columns=["Ticker", "Name", "Weight"])
                wf["Ticker"] = wf[cols["ticker"]].astype(str).str.upper().str.strip()
                wf["Weight"] = pd.to_numeric(wf[cols["weight"]], errors="coerce").fillna(0.0)
                wf = wf.groupby("Ticker", as_index=False)["Weight"].sum()
                tot = float(wf["Weight"].sum())
                if tot > 0:
                    wf["Weight"] = wf["Weight"] / tot
                wf["Name"] = ""
                wf = wf[["Ticker", "Name", "Weight"]].sort_values("Weight", ascending=False).reset_index(drop=True)
                return wf
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
    return df.sort_values("Ticker").reset_index(drop=True)[["Ticker", "Weight"]]

def benchmark_snapshot_id(wave_name: str, bm_mix_df: pd.DataFrame) -> str:
    try:
        if bm_mix_df is None or bm_mix_df.empty:
            return "BM-NA"
        rows = bm_mix_df[bm_mix_df["Wave"] == wave_name].copy() if "Wave" in bm_mix_df.columns else bm_mix_df.copy()
        if "Ticker" not in rows.columns or "Weight" not in rows.columns:
            return "BM-NA"
        rows = _normalize_bm_rows(rows[["Ticker", "Weight"]])
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
        "confidence": "Low",
        "flags": [],
    }
    try:
        if hist is None or hist.empty:
            out["flags"].append("No history returned")
            return out

        idx = pd.to_datetime(hist.index).sort_values()
        out["rows"] = int(len(idx))
        out["first_date"] = idx[0].date().isoformat() if len(idx) else None
        out["last_date"] = idx[-1].date().isoformat() if len(idx) else None

        today = datetime.utcnow().date()
        last_dt = idx[-1].date()
        out["age_days"] = int((today - last_dt).days)

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

        cs = out["completeness_score"]
        if cs is not None and cs >= 90 and (out["age_days"] is not None and out["age_days"] <= 2):
            out["confidence"] = "High"
        elif cs is not None and cs >= 75:
            out["confidence"] = "Medium"
        else:
            out["confidence"] = "Low"

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
# WaveScore (console-side approximation) + ranks
# ============================================================
@st.cache_data(show_spinner=False)
def compute_wavescore_for_all_waves(all_waves: List[str], mode: str, days: int = 365) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for wave in all_waves:
        hist = compute_wave_history(wave, mode=mode, days=days)
        if hist is None or hist.empty or len(hist) < 20:
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
    if df.empty:
        return df
    df["Rank"] = df["WaveScore"].rank(ascending=False, method="min")
    return df.sort_values("Wave")

# ============================================================
# Performance Matrix (Returns + Alpha)
# ============================================================
@st.cache_data(show_spinner=False)
def build_performance_matrix(all_waves: List[str], mode: str, days: int = 365) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for w in all_waves:
        h = compute_wave_history(w, mode=mode, days=days)
        if h is None or h.empty or len(h) < 2:
            rows.append({
                "Wave": w,
                "1D Return": np.nan, "1D Alpha": np.nan,
                "30D Return": np.nan, "30D Alpha": np.nan,
                "60D Return": np.nan, "60D Alpha": np.nan,
                "365D Return": np.nan, "365D Alpha": np.nan,
                "Rows": 0,
            })
            continue

        nav_w = h["wave_nav"]
        nav_b = h["bm_nav"]

        r1 = np.nan
        a1 = np.nan
        if len(nav_w) >= 2 and len(nav_b) >= 2:
            r1 = float(nav_w.iloc[-1] / nav_w.iloc[-2] - 1.0)
            b1 = float(nav_b.iloc[-1] / nav_b.iloc[-2] - 1.0)
            a1 = r1 - b1

        r30 = ret_from_nav(nav_w, min(30, len(nav_w)))
        b30 = ret_from_nav(nav_b, min(30, len(nav_b)))
        a30 = r30 - b30

        r60 = ret_from_nav(nav_w, min(60, len(nav_w)))
        b60 = ret_from_nav(nav_b, min(60, len(nav_b)))
        a60 = r60 - b60

        r365 = ret_from_nav(nav_w, min(365, len(nav_w)))
        b365 = ret_from_nav(nav_b, min(365, len(nav_b)))
        a365 = r365 - b365

        rows.append({
            "Wave": w,
            "1D Return": r1, "1D Alpha": a1,
            "30D Return": r30, "30D Alpha": a30,
            "60D Return": r60, "60D Alpha": a60,
            "365D Return": r365, "365D Alpha": a365,
            "Rows": int(len(h)),
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # convert to percent points for display
    cols = [c for c in df.columns if ("Return" in c) or ("Alpha" in c)]
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce") * 100.0

    return df.sort_values("Wave").reset_index(drop=True)

# ============================================================
# Simple regime + VIX
# ============================================================
@st.cache_data(show_spinner=False)
def get_vix_value() -> float:
    try:
        if yf is None:
            return float("nan")
        px = yf.download("^VIX", period="7d", interval="1d", auto_adjust=True, progress=False)
        if px is None or px.empty:
            return float("nan")
        col = "Close" if "Close" in px.columns else px.columns[-1]
        v = float(px[col].dropna().iloc[-1])
        return v
    except Exception:
        return float("nan")

def regime_from_vix(vix: float) -> str:
    if vix is None or (isinstance(vix, float) and math.isnan(vix)):
        return "unknown"
    if vix >= 25:
        return "risk-off"
    if vix <= 16:
        return "risk-on"
    return "neutral"

# ============================================================
# Rendering helpers
# ============================================================
def chip(label: str) -> str:
    safe = (label or "").replace("<", "").replace(">", "")
    return f'<span class="waves-chip">{safe}</span>'

def render_sticky(chips: List[str]) -> None:
    html = '<div class="waves-sticky">' + "".join([chip(c) for c in chips if c]) + "</div>"
    st.markdown(html, unsafe_allow_html=True)

def google_quote_link(ticker: str) -> str:
    t = (ticker or "").upper().strip()
    if not t:
        return ""
    # Works fine even if tickers include dots; Google handles most.
    return f"https://www.google.com/finance/quote/{t}:NASDAQ" if t.isalpha() else f"https://www.google.com/finance/quote/{t}"

def safe_mode_list() -> List[str]:
    return ["Standard", "Alpha-Minus-Beta", "Private Logic"]

# ============================================================
# Diagnostics renderer (inside function to prevent indentation issues)
# ============================================================
def render_diagnostics(all_waves: List[str]) -> None:
    st.subheader("Diagnostics (Safe)")
    st.caption("This section exists to prevent blank-screens. It is always safe to open/close.")

    st.markdown("**Runtime**")
    st.write("UTC:", datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"))

    st.markdown("**Engine**")
    if ENGINE_IMPORT_ERROR is not None:
        st.error("waves_engine import failed (app will use CSV fallbacks where possible).")
        st.code(str(ENGINE_IMPORT_ERROR))
    else:
        st.success("waves_engine import OK.")
        st.write("Engine module:", getattr(we, "__file__", "loaded"))

    st.markdown("**Decision Engine**")
    if DECISION_IMPORT_ERROR is not None:
        st.warning("decision_engine import issue (non-fatal). Decision tabs will fallback.")
        st.code(str(DECISION_IMPORT_ERROR))
    else:
        st.success("decision_engine import OK (or not required).")

    st.markdown("**Files present**")
    for fp in ["wave_history.csv", "wave_weights.csv", "list.csv", "requirements.txt"]:
        st.write(fp, "✅" if os.path.exists(fp) else "—")

    st.markdown("**Wave discovery**")
    st.write("Count:", len(all_waves))
    if all_waves:
        st.write(all_waves[:20])

# ============================================================
# Main UI
# ============================================================
st.title("WAVES Intelligence™ Institutional Console")

# Sidebar controls
st.sidebar.markdown("## Controls")
advanced = st.sidebar.toggle("Advanced Analytics", value=False, help="OFF = clean demo. ON = power user mode.")

mode = st.sidebar.selectbox("Mode", options=safe_mode_list(), index=0)

all_waves = get_all_waves_safe()
if not all_waves:
    st.warning("No waves discovered yet. If unexpected, check engine + CSV files.")
    # still show diagnostics + definitions so the app isn't a dead end
    with st.sidebar.expander("Metric Definitions", expanded=True):
        render_metric_definitions(full=False)
    if advanced:
        render_diagnostics(all_waves)
    st.stop()

selected_wave = st.sidebar.selectbox("Wave", options=all_waves, index=0)

with st.sidebar.expander("Metric Definitions", expanded=False):
    render_metric_definitions(full=False)

# Pull history for selected wave/mode
hist = compute_wave_history(selected_wave, mode=mode, days=365)
hist = _standardize_history(hist)

cov = coverage_report(hist)
vix_val = get_vix_value()
regime = regime_from_vix(vix_val)

bm_mix = get_benchmark_mix()
bm_id = benchmark_snapshot_id(selected_wave, bm_mix)
bm_drift = benchmark_drift_status(selected_wave, mode, bm_id)

# Compute headline metrics
nav_w = hist["wave_nav"] if "wave_nav" in hist.columns else pd.Series(dtype=float)
nav_b = hist["bm_nav"] if "bm_nav" in hist.columns else pd.Series(dtype=float)
ret_w = hist["wave_ret"] if "wave_ret" in hist.columns else daily_returns_from_nav(nav_w)
ret_b = hist["bm_ret"] if "bm_ret" in hist.columns else daily_returns_from_nav(nav_b)

r30 = ret_from_nav(nav_w, min(30, len(nav_w))) if len(nav_w) else float("nan")
b30 = ret_from_nav(nav_b, min(30, len(nav_b))) if len(nav_b) else float("nan")
a30 = r30 - b30 if (math.isfinite(r30) and math.isfinite(b30)) else float("nan")

r60 = ret_from_nav(nav_w, min(60, len(nav_w))) if len(nav_w) else float("nan")
b60 = ret_from_nav(nav_b, min(60, len(nav_b))) if len(nav_b) else float("nan")
a60 = r60 - b60 if (math.isfinite(r60) and math.isfinite(b60)) else float("nan")

r365 = ret_from_nav(nav_w, min(365, len(nav_w))) if len(nav_w) else float("nan")
b365 = ret_from_nav(nav_b, min(365, len(nav_b))) if len(nav_b) else float("nan")
a365 = r365 - b365 if (math.isfinite(r365) and math.isfinite(b365)) else float("nan")

te = tracking_error(ret_w, ret_b)
ir = information_ratio(nav_w, nav_b, te)
mdd = max_drawdown(nav_w)

# Wavescore table (for rank)
ws_table = compute_wavescore_for_all_waves(all_waves, mode=mode, days=365)
ws_row = ws_table[ws_table["Wave"] == selected_wave] if isinstance(ws_table, pd.DataFrame) and not ws_table.empty else pd.DataFrame()
ws_val = float(ws_row["WaveScore"].iloc[0]) if (not ws_row.empty and "WaveScore" in ws_row.columns and pd.notna(ws_row["WaveScore"].iloc[0])) else float("nan")
ws_grade = _grade_from_score(ws_val) if math.isfinite(ws_val) else "N/A"
ws_rank = int(ws_row["Rank"].iloc[0]) if (not ws_row.empty and "Rank" in ws_row.columns and pd.notna(ws_row["Rank"].iloc[0])) else None

# Sticky chips (this is what David saw)
chips = [
    f"BM Snapshot: {bm_id} • {bm_drift.capitalize()}",
    f"Rows: {cov.get('rows', 0)} • Age: {cov.get('age_days', '—')}",
    f"Confidence: {cov.get('confidence', 'Low')}",
    f"Regime: {regime} • VIX: {fmt_num(vix_val, 1)}",
    f"30D α: {fmt_pct(a30)} • 30D r: {fmt_pct(r30)}",
    f"60D α: {fmt_pct(a60)} • 60D r: {fmt_pct(r60)}",
    f"365D α: {fmt_pct(a365)} • 365D r: {fmt_pct(r365)}",
    f"TE: {fmt_pct(te)} • IR: {fmt_num(ir, 2)}",
    f"WaveScore: {fmt_score(ws_val)} ({ws_grade}) • Rank: {ws_rank if ws_rank else '—'}",
]
render_sticky(chips)

st.caption("Observational analytics only (not trading advice).")

# ============================================================
# Tabs: clean demo vs advanced
# ============================================================
base_tabs = ["IC Summary", "Overview (All Waves)", "Wave Detail", "Attribution", "Definitions"]
adv_tabs = ["Risk Lab", "Correlation", "Mode Proof", "Drawdown Monitor", "Exports", "Decision", "Diagnostics"]

tabs = base_tabs + (adv_tabs if advanced else [])
tab_objs = st.tabs(tabs)

# ---------------- IC SUMMARY ----------------
with tab_objs[0]:
    st.subheader(f"IC Summary — {selected_wave} ({mode})")
    st.caption("Decision-grade summary: what matters now, why, and what to check next.")

    # Action / Watch / Notes based on simple heuristics
    actions: List[str] = []
    watch: List[str] = []
    notes: List[str] = []

    if cov.get("confidence") in ["Low"]:
        actions.append("Data confidence is LOW — verify history freshness and missing business days before relying on alpha.")
    if bm_drift != "stable":
        actions.append("Benchmark drift detected — freeze benchmark snapshot for IC review / demos.")
    if math.isfinite(a30) and a30 < 0:
        watch.append("30D alpha is weak — could be timing/drawdown effects; monitor for decay vs noise.")
    if math.isfinite(a365) and a365 > 0 and math.isfinite(a30) and a30 < 0:
        watch.append("Long-term alpha positive but 30D weak — watch for short-term compression vs intact edge.")
    if math.isfinite(te) and te > 0.20:
        watch.append("Tracking error elevated — active risk higher than normal vs benchmark.")
    if math.isfinite(mdd) and mdd <= -0.25:
        watch.append("Drawdown deep — consider stronger SmartSafe posture in stress regimes.")

    if not actions:
        actions.append("No urgent actions — system appears stable on this window.")
    if not watch:
        watch.append("No major watch items detected.")
    if not notes:
        notes.append("No additional notes.")

    st.markdown("### Action")
    for a in actions:
        st.markdown(f"- {a}")

    st.markdown("### Watch")
    for w in watch:
        st.markdown(f"- {w}")

    st.markdown("### Notes")
    for n in notes:
        st.markdown(f"- {n}")

    st.divider()
    st.markdown("### Integrity & Confidence")
    st.write(f"Confidence: **{cov.get('confidence','Low')}** — Coverage Score: **{fmt_num(cov.get('completeness_score', np.nan), 1)}**")
    if cov.get("flags"):
        st.warning(" • ".join(cov["flags"]))
    else:
        st.success("No data integrity flags detected on this window.")

# ---------------- OVERVIEW (ALL WAVES) ----------------
with tab_objs[1]:
    st.subheader(f"Overview — All Waves ({mode})")
    perf = build_performance_matrix(all_waves, mode=mode, days=365)
    if perf is None or perf.empty:
        st.info("No performance matrix available (missing history).")
    else:
        st.dataframe(perf, use_container_width=True, height=650)

    st.caption("Returns/Alpha are shown in % points. Alpha = Wave − Benchmark over each window.")

# ---------------- WAVE DETAIL ----------------
with tab_objs[2]:
    st.subheader(f"Wave Detail — {selected_wave} ({mode})")

    c1, c2 = st.columns([1, 1])
    with c1:
        st.markdown("#### Holdings (Top 10)")
        h = get_wave_holdings(selected_wave)
        if h is None or h.empty:
            st.info("No holdings available from engine or wave_weights.csv.")
        else:
            hh = h.head(10).copy()
            # clickable links (markdown)
            hh["Link"] = hh["Ticker"].apply(lambda t: f"[{t}]({google_quote_link(str(t))})")
            hh["Weight"] = pd.to_numeric(hh["Weight"], errors="coerce").fillna(0.0)
            hh["Weight"] = (hh["Weight"] * 100.0).round(2)
            st.dataframe(hh[["Link", "Name", "Weight"]].rename(columns={"Weight": "Weight %"}), use_container_width=True, height=420)

    with c2:
        st.markdown("#### Performance Snapshot")
        st.write("30D Return:", fmt_pct(r30), " | 30D Alpha:", fmt_pct(a30))
        st.write("60D Return:", fmt_pct(r60), " | 60D Alpha:", fmt_pct(a60))
        st.write("365D Return:", fmt_pct(r365), " | 365D Alpha:", fmt_pct(a365))
        st.write("Max Drawdown:", fmt_pct(mdd))
        st.write("Tracking Error (TE):", fmt_pct(te))
        st.write("Information Ratio (IR):", fmt_num(ir, 2))
        st.write("WaveScore (approx):", fmt_score(ws_val), f"({ws_grade})", "| Rank:", ws_rank if ws_rank else "—")

        st.markdown("#### Benchmark Snapshot")
        st.write("BM Snapshot:", bm_id)
        st.write("Benchmark Drift:", bm_drift.capitalize())

    st.divider()
    st.markdown("#### NAV vs Benchmark (if available)")
    if hist is None or hist.empty or len(nav_w) < 2 or len(nav_b) < 2:
        st.info("NAV series not available.")
    else:
        plot_df = pd.DataFrame({"Wave NAV": nav_w.astype(float), "Benchmark NAV": nav_b.astype(float)}).dropna()
        st.line_chart(plot_df, height=320, use_container_width=True)

# ---------------- ATTRIBUTION ----------------
with tab_objs[3]:
    st.subheader("Attribution (Light)")
    st.caption("This build keeps attribution simple and safe. If your engine provides richer attribution, we can wire it here without changing engine math.")

    st.markdown("#### Alpha Decomposition (basic)")
    st.write("Alpha is computed as (Wave return − Benchmark return) over the selected window.")
    st.write("If you want: Engine vs Static Basket proxy, factor sleeve attribution, and ‘Alpha Captured’ rollups — we can add them under Advanced Analytics.")

    st.divider()
    st.markdown("#### Benchmark Mix (if provided by engine)")
    bm = get_benchmark_mix()
    if bm is None or bm.empty:
        st.info("No benchmark mix table available.")
    else:
        if "Wave" in bm.columns:
            bmw = bm[bm["Wave"].astype(str) == str(selected_wave)].copy()
        else:
            bmw = bm.copy()

        if bmw.empty:
            st.info("No benchmark rows for this wave.")
        else:
            for col in ["Weight"]:
                if col in
                                if col in bmw.columns:
                    bmw[col] = pd.to_numeric(bmw[col], errors="coerce")
            if "Weight" in bmw.columns:
                bmw["Weight %"] = (bmw["Weight"].fillna(0.0) * 100.0).round(2)
            show_cols = [c for c in ["Ticker", "Name", "Weight %"] if c in bmw.columns]
            if show_cols:
                sort_col = "Weight %" if "Weight %" in show_cols else show_cols[0]
                st.dataframe(
                    bmw[show_cols].sort_values(sort_col, ascending=False),
                    use_container_width=True,
                    height=520,
                )
            else:
                st.info("Benchmark mix table missing expected columns (Ticker/Name/Weight).")

# ---------------- DEFINITIONS ----------------
with tab_objs[4]:
    st.subheader("Definitions & Methodology")
    render_metric_definitions(full=True)
    st.divider()
    st.markdown("### Where these numbers come from (at a glance)")
    st.markdown(
        "- **Rows / Age / Coverage Score / Confidence** come from the history window you loaded for the selected wave/mode.\n"
        "- **Return** uses NAV change over the window.\n"
        "- **Alpha** = Return(Wave) − Return(Benchmark).\n"
        "- **TE** = annualized stdev of daily (Wave − Benchmark) returns.\n"
        "- **IR** = total-window alpha / TE.\n"
        "- **WaveScore** here is a *demo ranking*; your locked WAVESCORE™ spec remains separate.\n"
    )

# ---------------- ADVANCED TABS ----------------
adv_offset = len(base_tabs)
if advanced:
    # Risk Lab
    with tab_objs[adv_offset + 0]:
        st.subheader("Risk Lab (Basic)")
        if hist is None or hist.empty or len(nav_w) < 20:
            st.info("Not enough history for risk lab.")
        else:
            dr = daily_returns_from_nav(nav_w).dropna()
            dd = drawdown_series(nav_w).dropna()
            var95, cvar95 = var_cvar(dr, level=0.95)
            st.write("Max Drawdown:", fmt_pct(mdd))
            st.write("VaR 95% (daily):", fmt_pct(var95))
            st.write("CVaR 95% (daily):", fmt_pct(cvar95))
            st.line_chart(pd.DataFrame({"Drawdown": dd}), height=260, use_container_width=True)

    # Correlation
    with tab_objs[adv_offset + 1]:
        st.subheader("Correlation (Wave vs Benchmark)")
        if hist is None or hist.empty:
            st.info("No history.")
        else:
            dfc = pd.concat([ret_w.rename("Wave"), ret_b.rename("Benchmark")], axis=1).dropna()
            if dfc.shape[0] < 20:
                st.info("Not enough overlapping returns for correlation.")
            else:
                corr = float(dfc.corr().iloc[0, 1])
                st.write("Correlation:", fmt_num(corr, 3))
                st.line_chart(dfc, height=280, use_container_width=True)

    # Mode Proof
    with tab_objs[adv_offset + 2]:
        st.subheader("Mode Separation Proof (Same Wave, 3 Modes)")
        st.caption("Shows that each mode resolves independently (history + benchmark snapshot).")

        rows = []
        for m in safe_mode_list():
            h2 = compute_wave_history(selected_wave, mode=m, days=365)
            h2 = _standardize_history(h2)
            cov2 = coverage_report(h2)

            bm2 = get_benchmark_mix()
            bmid2 = benchmark_snapshot_id(selected_wave, bm2)
            drift2 = benchmark_drift_status(selected_wave, m, bmid2)

            navw2 = h2["wave_nav"] if "wave_nav" in h2.columns else pd.Series(dtype=float)
            navb2 = h2["bm_nav"] if "bm_nav" in h2.columns else pd.Series(dtype=float)

            r30_2 = ret_from_nav(navw2, min(30, len(navw2))) if len(navw2) else np.nan
            b30_2 = ret_from_nav(navb2, min(30, len(navb2))) if len(navb2) else np.nan
            a30_2 = r30_2 - b30_2 if (math.isfinite(r30_2) and math.isfinite(b30_2)) else np.nan

            rows.append({
                "Mode": m,
                "Rows": cov2.get("rows", 0),
                "Age (days)": cov2.get("age_days", None),
                "Coverage Score": cov2.get("completeness_score", None),
                "Confidence": cov2.get("confidence", "Low"),
                "BM Snapshot": bmid2,
                "Drift": drift2,
                "30D Alpha (%)": (a30_2 * 100.0) if math.isfinite(a30_2) else np.nan,
            })

        dfm = pd.DataFrame(rows)
        st.dataframe(dfm, use_container_width=True, height=420)

    # Drawdown Monitor
    with tab_objs[adv_offset + 3]:
        st.subheader("Drawdown Monitor")
        if hist is None or hist.empty or len(nav_w) < 20:
            st.info("Not enough history.")
        else:
            dd = drawdown_series(nav_w).dropna()
            cur_dd = float(dd.iloc[-1]) if len(dd) else np.nan
            st.write("Current drawdown:", fmt_pct(cur_dd))
            st.write("Max drawdown:", fmt_pct(mdd))
            st.line_chart(pd.DataFrame({"Drawdown": dd}), height=320, use_container_width=True)

    # Exports
    with tab_objs[adv_offset + 4]:
        st.subheader("Exports (Governance-ready)")
        st.caption("Quick downloads for IC / board workflows.")

        perf = build_performance_matrix(all_waves, mode=mode, days=365)
        if perf is not None and not perf.empty:
            csv = perf.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download Performance Matrix (CSV)",
                data=csv,
                file_name=f"waves_performance_matrix_{mode.replace(' ', '_')}.csv",
                mime="text/csv",
            )
        else:
            st.info("No performance matrix to export.")

        if hist is not None and not hist.empty:
            hx = hist.copy()
            hx = hx.reset_index().rename(columns={"index": "date"})
            csv2 = hx.to_csv(index=False).encode("utf-8")
            st.download_button(
                f"Download {selected_wave} History (CSV)",
                data=csv2,
                file_name=f"{selected_wave.replace(' ', '_')}_{mode.replace(' ', '_')}_history.csv",
                mime="text/csv",
            )
        else:
            st.info("No history to export for selected wave.")

    # Decision (optional)
    with tab_objs[adv_offset + 5]:
        st.subheader("Decision Intelligence (Optional)")
        st.caption("If decision_engine.py exists, we’ll show its outputs. If not, we show safe defaults.")

        ctx = {
            "wave_name": selected_wave,
            "mode": mode,
            "bm_snapshot": bm_id,
            "bm_drift": bm_drift,
            "rows": cov.get("rows"),
            "age_days": cov.get("age_days"),
            "coverage_score": cov.get("completeness_score"),
            "confidence": cov.get("confidence"),
            "regime": regime,
            "vix": vix_val,
            "r30": r30, "a30": a30,
            "r60": r60, "a60": a60,
            "r365": r365, "a365": a365,
            "te": te,
            "ir": ir,
            "mdd": mdd,
            "wavescore": ws_val,
            "rank": ws_rank,
        }

        if generate_decisions is not None:
            try:
                out = generate_decisions(ctx)
                st.markdown("### Output")
                st.write(out)
            except Exception as e:
                st.warning("Decision engine error (non-fatal).")
                st.code(str(e))
        else:
            st.info("decision_engine.py not present (or missing generate_decisions). Showing safe narrative instead.")
            st.markdown("### Action")
            for a in actions:
                st.markdown(f"- {a}")
            st.markdown("### Watch")
            for w in watch:
                st.markdown(f"- {w}")
            st.markdown("### Notes")
            for n in notes:
                st.markdown(f"- {n}")

        if build_daily_wave_activity is not None:
            st.divider()
            st.markdown("### Daily Activity (Optional)")
            try:
                activity = build_daily_wave_activity(ctx)
                st.write(activity)
            except Exception as e:
                st.warning("Daily activity builder error (non-fatal).")
                st.code(str(e))

    # Diagnostics
    with tab_objs[adv_offset + 6]:
        render_diagnostics(all_waves)

# End of file marker (intentionally left as a comment)
# EOF — app.py