# app.py — WAVES Intelligence™ Institutional Console (Vector OS Edition)
# FULL PRODUCTION FILE (NO PATCHES) — CANONICAL COHESION LOCK + IC ONE-PAGER + FIDELITY INSPECTOR + AI EXPLAIN + COMPARATOR
#
# Big upgrades added:
#   1) Executive IC One-Pager (top of IC Summary)
#   2) Cohesion Lock (Truth Table) — canonical source proof
#   3) Benchmark Fidelity Inspector (drift + composition diff + difficulty vs SPY + TE risk band)
#   4) AI Explanation Layer (deterministic rules-based narrative; no hallucinations)
#   5) Wave-to-Wave Comparator (A vs B)
#
# Canonical rule:
#   All computed metrics must come from ONE standardized history object:
#     hist_sel = _standardize_history(compute_wave_history(selected_wave, mode))
#   Every tab reuses this same dataset (no duplicate math / no crisscross).
#
# Notes:
#   • Engine math NOT modified.
#   • Robust history loader: engine functions → wave_history.csv fallback
#   • Optional libs (yfinance/plotly) are guarded; app will not crash without them.
#   • Safe conversions prevent float(None) crashes.

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
# Optional Decision Engine import (guarded)
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
.block-container { padding-top: 0.85rem; padding-bottom: 2.0rem; }

/* BIG wave header */
.waves-big-wave {
  font-size: 2.0rem;
  font-weight: 800;
  letter-spacing: 0.2px;
  line-height: 2.2rem;
  margin: 0.1rem 0 0.4rem 0;
}
.waves-subhead {
  opacity: 0.85;
  font-size: 1.0rem;
  margin: 0 0 0.6rem 0;
}

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
  padding: 7px 11px;
  margin: 6px 8px 0 0;
  border-radius: 999px;
  border: 1px solid rgba(255,255,255,0.12);
  background: rgba(255,255,255,0.04);
  font-size: 0.90rem;
  line-height: 1.05rem;
  white-space: nowrap;
}

/* Card blocks */
.waves-card {
  border: 1px solid rgba(255,255,255,0.10);
  border-radius: 14px;
  padding: 12px 14px;
  background: rgba(255,255,255,0.03);
}

/* Tile */
.waves-tile {
  border: 1px solid rgba(255,255,255,0.10);
  border-radius: 14px;
  padding: 12px 14px;
  background: rgba(255,255,255,0.03);
  min-height: 84px;
}
.waves-tile-label { opacity: 0.85; font-size: 0.90rem; margin-bottom: 0.25rem; }
.waves-tile-value { font-size: 1.55rem; font-weight: 800; line-height: 1.75rem; }
.waves-tile-sub { opacity: 0.75; font-size: 0.90rem; margin-top: 0.20rem; }

/* Reduce whitespace for mobile */
@media (max-width: 700px) {
  .block-container { padding-left: 0.8rem; padding-right: 0.8rem; }
  .waves-big-wave { font-size: 1.6rem; line-height: 1.85rem; }
  .waves-tile-value { font-size: 1.35rem; line-height: 1.55rem; }
}
</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# Helpers: formatting / safety
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


def fmt_pp(x: Any, digits: int = 2) -> str:
    """Percent points already (10.0), output '10.00%'."""
    try:
        if x is None:
            return "—"
        x = float(x)
        if math.isnan(x):
            return "—"
        return f"{x:0.{digits}f}%"
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


def safe_float(x: Any, default: float = np.nan) -> float:
    try:
        if x is None:
            return default
        v = float(x)
        if not math.isfinite(v):
            return default
        return v
    except Exception:
        return default


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
    r = safe_series(daily_ret).astype(float).dropna()
    if len(r) < 50:
        return (float("nan"), float("nan"))
    q = float(np.quantile(r.values, 1.0 - level))
    tail = r[r <= q]
    cvar = float(tail.mean()) if len(tail) else float("nan")
    return (q, cvar)


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
# Definitions / Glossary (Self-explanatory layer)
# ============================================================
GLOSSARY: Dict[str, str] = {
    "Canonical (Source of Truth)": (
        "A governance rule: the console computes ALL metrics from one standardized history object for the selected Wave+Mode "
        "(hist_sel = standardized wave_nav, bm_nav, wave_ret, bm_ret). Every tab reuses it. No duplicate math = no crisscross."
    ),
    "Return": "Portfolio return over the window (not annualized unless stated).",
    "Alpha": "Return minus Benchmark return over the same window (relative performance).",
    "Tracking Error (TE)": "Annualized volatility of (Wave daily returns − Benchmark daily returns). Higher = more active risk.",
    "Information Ratio (IR)": "Excess return divided by Tracking Error (risk-adjusted alpha).",
    "Max Drawdown (MaxDD)": "Largest peak-to-trough decline over the period (negative number).",
    "VaR 95% (daily)": "A loss threshold such that ~5% of days are worse (historical).",
    "CVaR 95% (daily)": "Average loss on the worst ~5% of days (tail risk).",
    "Sharpe": "Risk-adjusted return using total volatility (0% rf here).",
    "Sortino": "Risk-adjusted return using downside deviation only.",
    "Benchmark Snapshot / Drift": "A fingerprint of the benchmark mix. Drift means the mix changed in-session.",
    "Coverage Score": "0–100 heuristic of data completeness + freshness.",
    "Difficulty vs SPY": "Heuristic proxy based on benchmark concentration/diversification (not a promise of difficulty).",
    "Risk Reaction Score": "0–100: how 'healthy' the wave’s risk posture looks from TE/MaxDD/CVaR (heuristic).",
    "Analytics Scorecard": "Governance-native grade for analytics reliability & decision readiness (not performance).",
}


def render_definitions(keys: List[str], title: str = "Definitions"):
    with st.expander(title):
        for k in keys:
            st.markdown(f"**{k}:** {GLOSSARY.get(k, '(definition not found)')}")


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
    if we is None:
        return history_from_csv(wave_name, mode, days)

    try:
        if hasattr(we, "compute_history_nav"):
            try:
                df = we.compute_history_nav(wave_name, mode=mode, days=days)
                df = _standardize_history(df)
                if not df.empty:
                    return df
            except TypeError:
                df = we.compute_history_nav(wave_name, mode, days)
                df = _standardize_history(df)
                if not df.empty:
                    return df
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
    if we is None:
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

    if hasattr(we, "get_all_waves"):
        try:
            waves = we.get_all_waves()
            if isinstance(waves, (list, tuple)):
                waves = [str(x) for x in waves]
                return sorted([w for w in waves if w and w.lower() != "nan"])
        except Exception:
            pass

    if hasattr(we, "get_benchmark_mix_table"):
        try:
            bm = we.get_benchmark_mix_table()
            if isinstance(bm, pd.DataFrame) and "Wave" in bm.columns:
                return sorted(list(set(bm["Wave"].astype(str).tolist())))
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


# ============================================================
# Benchmark snapshot + drift tracking + difficulty proxy
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


def _bm_rows_for_wave(bm_mix_df: pd.DataFrame, wave_name: str) -> pd.DataFrame:
    if bm_mix_df is None or bm_mix_df.empty:
        return pd.DataFrame(columns=["Ticker", "Weight"])
    if "Wave" in bm_mix_df.columns:
        rows = bm_mix_df[bm_mix_df["Wave"] == wave_name].copy()
    else:
        rows = bm_mix_df.copy()
    if "Ticker" not in rows.columns or "Weight" not in rows.columns:
        return pd.DataFrame(columns=["Ticker", "Weight"])
    return _normalize_bm_rows(rows[["Ticker", "Weight"]])


def benchmark_snapshot_id(wave_name: str, bm_mix_df: pd.DataFrame) -> str:
    try:
        rows = _bm_rows_for_wave(bm_mix_df, wave_name)
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


def benchmark_diff_table(wave_name: str, mode: str, bm_rows_now: pd.DataFrame) -> pd.DataFrame:
    """
    Shows composition change vs last session_state snapshot (per wave+mode).
    Always safe; returns empty DF if no prior.
    """
    key = f"bm_rows::{mode}::{wave_name}"
    prev = st.session_state.get(key)
    now = _normalize_bm_rows(bm_rows_now)

    if prev is None:
        st.session_state[key] = now
        return pd.DataFrame()

    try:
        prev_df = prev.copy() if isinstance(prev, pd.DataFrame) else pd.DataFrame(prev)
        prev_df = _normalize_bm_rows(prev_df)
    except Exception:
        prev_df = pd.DataFrame()

    st.session_state[key] = now

    if prev_df.empty or now.empty:
        return pd.DataFrame()

    a = prev_df.rename(columns={"Weight": "PrevWeight"})
    b = now.rename(columns={"Weight": "NowWeight"})
    d = pd.merge(a, b, on="Ticker", how="outer").fillna(0.0)
    d["Delta"] = d["NowWeight"] - d["PrevWeight"]
    d = d.sort_values("Delta", ascending=False)
    # Show meaningful changes
    d = d[(d["Delta"].abs() >= 0.002) | (d["NowWeight"] >= 0.05) | (d["PrevWeight"] >= 0.05)]
    d["PrevWeight"] = (d["PrevWeight"] * 100).round(2)
    d["NowWeight"] = (d["NowWeight"] * 100).round(2)
    d["Delta"] = (d["Delta"] * 100).round(2)
    return d.reset_index(drop=True)


def benchmark_difficulty_proxy(rows: pd.DataFrame) -> Dict[str, Any]:
    out = {"hhi": np.nan, "entropy": np.nan, "top_weight": np.nan, "difficulty_vs_spy": np.nan}
    try:
        if rows is None or rows.empty:
            return out
        r = rows.copy()
        r["Weight"] = pd.to_numeric(r["Weight"], errors="coerce").fillna(0.0)
        tot = float(r["Weight"].sum())
        if tot <= 0:
            return out
        w = (r["Weight"] / tot).values
        out["top_weight"] = float(np.max(w))
        out["hhi"] = float(np.sum(w**2))
        eps = 1e-12
        out["entropy"] = float(-np.sum(w * np.log(w + eps)))
        conc_pen = (out["hhi"] - 0.06) * 180.0
        ent_bonus = (out["entropy"] - 2.6) * -12.0
        raw = conc_pen + ent_bonus
        out["difficulty_vs_spy"] = float(np.clip(raw, -25.0, 25.0))
        return out
    except Exception:
        return out


def te_risk_band(te: float) -> str:
    if te is None or (isinstance(te, float) and not math.isfinite(te)):
        return "N/A"
    if te < 0.08:
        return "Low"
    if te < 0.16:
        return "Medium"
    return "High"


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

        idx = pd.to_datetime(hist.index).sort_values()
        out["rows"] = int(len(idx))
        out["first_date"] = idx[0].date().isoformat() if len(idx) else None
        out["last_date"] = idx[-1].date().isoformat() if len(idx) else None

        today = datetime.utcnow().date()
        last_dt = idx[-1].date()
        out["age_days"] = int((today - last_dt).days)

        # Business-day missingness (between first/last)
        bdays = pd.date_range(start=idx[0].normalize(), end=idx[-1].normalize(), freq="B")
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


def confidence_from_integrity(cov: Dict[str, Any], bm_drift: str) -> Tuple[str, str]:
    """
    Returns (level, reason) where level in {"High","Medium","Low"}.
    """
    try:
        score = safe_float(cov.get("completeness_score"))
        age = safe_float(cov.get("age_days"))
        rows = int(cov.get("rows") or 0)

        drift = (str(bm_drift).lower().strip() != "stable")
        issues = []

        if drift:
            issues.append("benchmark drift")
        if math.isfinite(score) and score < 85:
            issues.append("coverage < 85")
        if math.isfinite(age) and age >= 5:
            issues.append("stale (>=5d)")
        if rows < 90:
            issues.append("limited history")

        if not issues:
            return ("High", "Fresh + complete history and stable benchmark snapshot.")
        if drift or (math.isfinite(score) and score < 75) or (math.isfinite(age) and age >= 7) or rows < 60:
            return ("Low", "Potential trust issues: " + ", ".join(issues) + ".")
        return ("Medium", "Some caution flags: " + ", ".join(issues) + ".")
    except Exception:
        return ("Medium", "Confidence heuristic unavailable (non-fatal).")


# ============================================================
# Analytics Scorecard (Governance-Native)
# ============================================================
def _score_to_grade_af(score: float) -> str:
    try:
        s = float(score)
        if not math.isfinite(s):
            return "N/A"
        if s >= 90:
            return "A"
        if s >= 80:
            return "B"
        if s >= 70:
            return "C"
        if s >= 60:
            return "D"
        return "F"
    except Exception:
        return "N/A"


def _drift_penalty(status: str) -> float:
    s = str(status).lower().strip()
    return 1.0 if s == "stable" else 0.0


def _explainability_proxy(cov: Dict[str, Any], bm_drift: str, hist_rows: int) -> float:
    """
    Conservative proxy (0-100) if you don't have a dedicated AI confidence metric.
    """
    try:
        base = 88.0
        score = safe_float(cov.get("completeness_score"))
        age = safe_float(cov.get("age_days"))
        miss = safe_float(cov.get("missing_pct"))

        if math.isfinite(score):
            base += (score - 90.0) * 0.25
        if math.isfinite(age) and age > 3:
            base -= min(20.0, (age - 3) * 3.5)
        if math.isfinite(miss) and miss > 0.02:
            base -= min(20.0, (miss - 0.02) * 400.0)
        if str(bm_drift).lower().strip() != "stable":
            base -= 15.0
        if hist_rows < 90:
            base -= 8.0
        if hist_rows < 60:
            base -= 12.0

        return float(np.clip(base, 0.0, 100.0))
    except Exception:
        return 75.0


@st.cache_data(show_spinner=False)
def compute_analytics_scorecard_all_waves(all_waves: List[str], mode: str, days: int = 365) -> pd.DataFrame:
    bm_mix = get_benchmark_mix()
    rows: List[Dict[str, Any]] = []
    for w in all_waves:
        hist = _standardize_history(compute_wave_history(w, mode=mode, days=days))
        cov = coverage_report(hist)

        bid = benchmark_snapshot_id(w, bm_mix)
        drift = benchmark_drift_status(w, mode, bid)

        te = np.nan
        ir = np.nan
        mdd = np.nan
        vol = np.nan

        if hist is not None and not hist.empty and len(hist) >= 20:
            try:
                te = tracking_error(hist["wave_ret"], hist["bm_ret"])
                ir = information_ratio(hist["wave_nav"], hist["bm_nav"], te)
                mdd = max_drawdown(hist["wave_nav"])
                vol = annualized_vol(hist["wave_ret"])
            except Exception:
                pass

        coverage_score = safe_float(cov.get("completeness_score"))
        age_days = safe_float(cov.get("age_days"))
        rows_n = int(cov.get("rows") or 0)
        miss_pct = safe_float(cov.get("missing_pct"))

        # D1 Data Integrity & Coverage
        d1 = 85.0
        if math.isfinite(coverage_score):
            d1 = 0.85 * coverage_score + 0.15 * 95.0
        if math.isfinite(age_days) and age_days > 3:
            d1 -= min(25.0, (age_days - 3) * 4.0)
        if rows_n < 60:
            d1 -= 15.0
        if math.isfinite(miss_pct) and miss_pct > 0.05:
            d1 -= min(20.0, (miss_pct - 0.05) * 400.0)
        d1 = float(np.clip(d1, 0.0, 100.0))

        # D2 Benchmark Fidelity & Drift Control
        d2 = 92.0 * _drift_penalty(drift) + 78.0 * (1.0 - _drift_penalty(drift))
        if str(bid).upper() in ["BM-NA", "BM-ERR"]:
            d2 -= 12.0
        d2 = float(np.clip(d2, 0.0, 100.0))

        # D3 Risk Discipline
        d3 = 80.0
        if math.isfinite(te):
            d3 += float(np.clip((0.10 - te) * 250.0, -25.0, 15.0))
        if math.isfinite(vol):
            d3 += float(np.clip((0.22 - vol) * 140.0, -25.0, 15.0))
        if math.isfinite(mdd):
            d3 += float(np.clip((-0.18 - mdd) * 120.0, -25.0, 15.0))
        d3 = float(np.clip(d3, 0.0, 100.0))

        # D4 Efficiency / Performance Quality proxy (IR)
        d4 = 55.0
        if math.isfinite(ir):
            d4 = float(np.clip(55.0 + ir * 30.0, 0.0, 100.0))

        # D5 Explainability & Decision Readiness
        d5 = _explainability_proxy(cov, drift, rows_n)

        total = float(np.clip((d1 + d2 + d3 + d4 + d5) / 5.0, 0.0, 100.0))
        grade = _score_to_grade_af(total)

        flags = []
        if d1 < 70:
            flags.append("DATA")
        if d2 < 70:
            flags.append("BM")
        if d3 < 70:
            flags.append("RISK")
        if d5 < 70:
            flags.append("EXPLAIN")

        rows.append(
            {
                "Wave": w,
                "AnalyticsScore": total,
                "Grade": grade,
                "D1_DataIntegrity": d1,
                "D2_BenchmarkFidelity": d2,
                "D3_RiskDiscipline": d3,
                "D4_EfficiencyQuality": d4,
                "D5_DecisionReadiness": d5,
                "CoverageScore": coverage_score,
                "Age
                Days": age_days,
                "Rows": rows_n,
                "BM_Snapshot": bid,
                "BM_Drift": drift,
                "TE": te,
                "IR": ir,
                "MaxDD": mdd,
                "Vol": vol,
                "Flags": " ".join(flags) if flags else "",
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    for c in [
        "AnalyticsScore",
        "D1_DataIntegrity",
        "D2_BenchmarkFidelity",
        "D3_RiskDiscipline",
        "D4_EfficiencyQuality",
        "D5_DecisionReadiness",
        "CoverageScore",
        "AgeDays",
        "TE",
        "IR",
        "MaxDD",
        "Vol",
    ]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").round(2)

    return df.sort_values(["AnalyticsScore", "Wave"], ascending=[False, True]).reset_index(drop=True)


# ============================================================
# Canonical metrics bundle (single source of truth)
# ============================================================
def compute_canonical_metrics(hist_sel: pd.DataFrame) -> Dict[str, Any]:
    """
    Compute all *selected-wave* metrics from the canonical hist_sel only.
    Returns decimals (0.10) for returns/alpha; TE/Vol annualized decimals; VaR/CVaR daily decimals.
    """
    out: Dict[str, Any] = {
        "end_date": None,
        "rows": 0,
        "r1": np.nan,
        "a1": np.nan,
        "r30": np.nan,
        "a30": np.nan,
        "r60": np.nan,
        "a60": np.nan,
        "r365": np.nan,
        "a365": np.nan,
        "te": np.nan,
        "ir": np.nan,
        "mdd": np.nan,
        "vol": np.nan,
        "sharpe": np.nan,
        "sortino": np.nan,
        "var95": np.nan,
        "cvar95": np.nan,
    }
    try:
        if hist_sel is None or hist_sel.empty:
            return out

        h = _standardize_history(hist_sel)
        out["rows"] = int(len(h))
        try:
            out["end_date"] = pd.to_datetime(h.index[-1]).date().isoformat()
        except Exception:
            out["end_date"] = None

        nav_w = h["wave_nav"]
        nav_b = h["bm_nav"]
        r_w = h["wave_ret"]
        r_b = h["bm_ret"]

        # 1D (if enough points)
        if len(nav_w) >= 2 and len(nav_b) >= 2:
            out["r1"] = float(nav_w.iloc[-1] / nav_w.iloc[-2] - 1.0)
            b1 = float(nav_b.iloc[-1] / nav_b.iloc[-2] - 1.0)
            out["a1"] = out["r1"] - b1

        out["r30"] = ret_from_nav(nav_w, min(30, len(nav_w)))
        out["a30"] = out["r30"] - ret_from_nav(nav_b, min(30, len(nav_b)))

        out["r60"] = ret_from_nav(nav_w, min(60, len(nav_w)))
        out["a60"] = out["r60"] - ret_from_nav(nav_b, min(60, len(nav_b)))

        out["r365"] = ret_from_nav(nav_w, min(365, len(nav_w)))
        out["a365"] = out["r365"] - ret_from_nav(nav_b, min(365, len(nav_b)))

        out["te"] = tracking_error(r_w, r_b)
        out["ir"] = information_ratio(nav_w, nav_b, out["te"])
        out["mdd"] = max_drawdown(nav_w)
        out["vol"] = annualized_vol(r_w)
        out["sharpe"] = sharpe_ratio(r_w)
        out["sortino"] = sortino_ratio(r_w)
        v, c = var_cvar(r_w, level=0.95)
        out["var95"] = v
        out["cvar95"] = c

        return out
    except Exception:
        return out


def risk_reaction_score(te: float, mdd: float, cvar95: float) -> float:
    """
    0–100 heuristic. Higher is better.
    Uses annualized TE, MaxDD (negative), and CVaR (negative).
    """
    try:
        s = 85.0
        if math.isfinite(te):
            # TE: 0.06 good, 0.20 bad
            s += float(np.clip((0.10 - te) * 240.0, -30.0, 18.0))
        if math.isfinite(mdd):
            # -0.10 good, -0.35 bad
            s += float(np.clip((-0.18 - mdd) * 120.0, -35.0, 15.0))
        if math.isfinite(cvar95):
            # -0.008 good, -0.03 bad
            s += float(np.clip((-0.016 - cvar95) * 1200.0, -30.0, 15.0))
        return float(np.clip(s, 0.0, 100.0))
    except Exception:
        return 70.0


def risk_grade_from_score(score: float) -> str:
    if score is None or (isinstance(score, float) and not math.isfinite(score)):
        return "N/A"
    if score >= 85:
        return "A"
    if score >= 75:
        return "B"
    if score >= 65:
        return "C"
    if score >= 55:
        return "D"
    return "F"


# ============================================================
# Deterministic AI Explanation Layer (rules-based; no LLM)
# ============================================================
def explain_from_metrics(
    wave_name: str,
    mode: str,
    cov: Dict[str, Any],
    bm_drift: str,
    bm_id: str,
    bm_diff: pd.DataFrame,
    m: Dict[str, Any],
    diff_proxy: Dict[str, Any],
    rr_score: float,
) -> Dict[str, Any]:
    """
    Returns dict with:
      - headline
      - what_changed
      - why_alpha
      - risk_driver
      - what_to_verify
      - next_actions
    Deterministic rules only.
    """
    headline = []
    what_changed = []
    why_alpha = []
    risk_driver = []
    verify = []
    next_actions = []

    # --- Trust / governance ---
    conf_level, conf_reason = confidence_from_integrity(cov, bm_drift)
    headline.append(f"**Confidence:** {conf_level} — {conf_reason}")

    if bm_drift != "stable":
        what_changed.append("Benchmark drift detected this session (composition snapshot changed).")
        if bm_diff is not None and not bm_diff.empty:
            top = bm_diff.head(6)
            changed = ", ".join([f"{r.Ticker} {r.Delta:+.2f}pp" for r in top.itertuples(index=False)])
            what_changed.append(f"Composition changes (top deltas): {changed}")
        verify.append("Freeze benchmark mix for demos / IC memos to eliminate drift variability.")
    else:
        what_changed.append("Benchmark snapshot is stable in-session.")

    # --- Alpha story (30D is primary) ---
    a30 = safe_float(m.get("a30"))
    r30 = safe_float(m.get("r30"))
    te = safe_float(m.get("te"))
    ir = safe_float(m.get("ir"))

    if math.isfinite(a30):
        if a30 >= 0.02:
            why_alpha.append(f"30D alpha is positive ({fmt_pct(a30)}). Likely driven by active tilts vs the benchmark mix.")
        elif a30 <= -0.02:
            why_alpha.append(f"30D alpha is negative ({fmt_pct(a30)}). This may be a regime mismatch or benchmark mix overweighting winners.")
            next_actions.append("Review factor/sector exposures vs benchmark; reduce unintended concentration.")
        else:
            why_alpha.append(f"30D alpha is near-flat ({fmt_pct(a30)}). The wave is tracking benchmark closely on this window.")

    if math.isfinite(te):
        band = te_risk_band(te)
        why_alpha.append(f"Tracking error is {fmt_pct(te)} (Active Risk: **{band}**).")
        if band == "High":
            risk_driver.append("Active risk is elevated; short-window alpha can swing widely.")
            next_actions.append("Consider SmartSafe posture or tighter exposure caps in high-vol regimes.")
        elif band == "Low":
            risk_driver.append("Active risk is low; performance will look benchmark-like.")
            next_actions.append("If alpha goal is higher, consider controlled active risk budget increases (governed).")

    if math.isfinite(ir):
        if ir >= 0.7:
            why_alpha.append(f"Information Ratio is strong ({fmt_num(ir)}), suggesting alpha is efficient relative to active risk.")
        elif ir <= 0.0:
            why_alpha.append(f"Information Ratio is weak ({fmt_num(ir)}); alpha is not compensating for active risk on this window.")
            next_actions.append("Verify missing days / stale data; if clean, adjust signal thresholds for this regime.")

    # --- Risk driver from RR score + components ---
    mdd = safe_float(m.get("mdd"))
    cvar = safe_float(m.get("cvar95"))
    rg = risk_grade_from_score(rr_score)
    risk_driver.append(f"Risk Reaction Score is **{fmt_num(rr_score,1)} / 100** (Grade **{rg}**).")

    if math.isfinite(mdd) and mdd <= -0.25:
        risk_driver.append(f"Max drawdown is deep ({fmt_pct(mdd)}).")
        next_actions.append("Strengthen downside gating; review stop/trim rules in drawdown accelerations.")
    if math.isfinite(cvar) and cvar <= -0.02:
        risk_driver.append(f"Tail risk looks heavy (CVaR95 {fmt_pct(cvar)} daily).")
        next_actions.append("Check for concentrated positions / correlated basket risk; add hedges or reduce overlap.")

    # --- What to verify (always) ---
    if cov.get("flags"):
        verify.append("Data integrity flags present: " + "; ".join(cov["flags"]))
    if cov.get("age_days") is not None and safe_float(cov.get("age_days")) >= 5:
        verify.append("History appears stale (>=5 days): confirm engine is writing latest datapoints.")
    if str(bm_id).upper() in ["BM-NA", "BM-ERR"]:
        verify.append("Benchmark mix missing/invalid: confirm get_benchmark_mix_table() is returning rows for this wave.")

    # --- Difficulty proxy callout (governance framing) ---
    dv = safe_float(diff_proxy.get("difficulty_vs_spy"))
    if math.isfinite(dv):
        if dv > 10:
            headline.append("**Benchmark Difficulty vs SPY:** Higher (more concentrated / specialized mix).")
        elif dv < -10:
            headline.append("**Benchmark Difficulty vs SPY:** Lower (more diversified / SPY-like mix).")
        else:
            headline.append("**Benchmark Difficulty vs SPY:** Moderate.")

    # Default next actions if empty
    if not next_actions:
        next_actions = [
            "Validate benchmark stability + data freshness.",
            "Review 30D alpha vs TE band and confirm it matches intent.",
            "Export the IC pack for governance trail.",
        ]

    return {
        "headline": headline,
        "what_changed": what_changed,
        "why_alpha": why_alpha,
        "risk_driver": risk_driver,
        "what_to_verify": verify if verify else ["No verification flags detected."],
        "next_actions": next_actions[:6],
    }


# ============================================================
# Wave-to-Wave comparator
# ============================================================
def comparator_metrics(all_waves: List[str], mode: str, wave_a: str, wave_b: str) -> pd.DataFrame:
    def _row(w: str) -> Dict[str, Any]:
        h = _standardize_history(compute_wave_history(w, mode=mode, days=365))
        m = compute_canonical_metrics(h)
        rr = risk_reaction_score(m["te"], m["mdd"], m["cvar95"])
        return {
            "Wave": w,
            "1D Return": m["r1"] * 100 if math.isfinite(safe_float(m["r1"])) else np.nan,
            "1D Alpha": m["a1"] * 100 if math.isfinite(safe_float(m["a1"])) else np.nan,
            "30D Return": m["r30"] * 100 if math.isfinite(safe_float(m["r30"])) else np.nan,
            "30D Alpha": m["a30"] * 100 if math.isfinite(safe_float(m["a30"])) else np.nan,
            "60D Return": m["r60"] * 100 if math.isfinite(safe_float(m["r60"])) else np.nan,
            "60D Alpha": m["a60"] * 100 if math.isfinite(safe_float(m["a60"])) else np.nan,
            "365D Return": m["r365"] * 100 if math.isfinite(safe_float(m["r365"])) else np.nan,
            "365D Alpha": m["a365"] * 100 if math.isfinite(safe_float(m["a365"])) else np.nan,
            "TE": m["te"],
            "MaxDD": m["mdd"],
            "CVaR95(d)": m["cvar95"],
            "RiskReaction": rr,
        }

    if not wave_a or not wave_b:
        return pd.DataFrame()
    df = pd.DataFrame([_row(wave_a), _row(wave_b)])
    # Format-friendly rounding
    for c in ["TE", "MaxDD", "CVaR95(d)", "RiskReaction"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in [c for c in df.columns if ("Return" in c or "Alpha" in c)]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


# ============================================================
# UI: Sidebar controls
# ============================================================
st.sidebar.markdown("## Controls")

modes = ["Standard", "Alpha-Minus-Beta", "Private Logic"]
mode = st.sidebar.selectbox("Mode", modes, index=0)

all_waves = get_all_waves_safe()
if not all_waves:
    st.sidebar.error("No waves found. Check engine import or CSVs (wave_config.csv / wave_weights.csv / list.csv).")

selected_wave = st.sidebar.selectbox("Selected Wave", all_waves if all_waves else ["(none)"], index=0 if all_waves else 0)

scan_mode = st.sidebar.toggle("Scan Mode (Fast Demo)", value=True)
show_365 = st.sidebar.toggle("Show 365D columns (Overview)", value=True)

st.sidebar.markdown("---")
st.sidebar.markdown("### Canonical Governance")
show_truth_table = st.sidebar.toggle("Show Cohesion Lock (Truth Table)", value=True)
show_defs = st.sidebar.toggle("Show Definitions drawers", value=True)

if ENGINE_IMPORT_ERROR is not None:
    st.sidebar.warning(f"waves_engine import issue: {ENGINE_IMPORT_ERROR}")
if DECISION_IMPORT_ERROR is not None:
    st.sidebar.info(f"decision_engine not loaded (optional): {DECISION_IMPORT_ERROR}")


# ============================================================
# Load canonical history once (THE SOURCE OF TRUTH)
# ============================================================
hist_sel = _standardize_history(compute_wave_history(selected_wave, mode=mode, days=365))
cov_sel = coverage_report(hist_sel)

bm_mix = get_benchmark_mix()
bm_rows_now = _bm_rows_for_wave(bm_mix, selected_wave)
bm_id = benchmark_snapshot_id(selected_wave, bm_mix)
bm_drift = benchmark_drift_status(selected_wave, mode, bm_id)
bm_diff = pd.DataFrame()
try:
    if bm_drift != "stable":
        bm_diff = benchmark_diff_table(selected_wave, mode, bm_rows_now)
except Exception:
    bm_diff = pd.DataFrame()

diff_proxy = benchmark_difficulty_proxy(bm_rows_now)

m = compute_canonical_metrics(hist_sel)
rr = risk_reaction_score(m["te"], m["mdd"], m["cvar95"])
rr_grade = risk_grade_from_score(rr)

conf_level, conf_reason = confidence_from_integrity(cov_sel, bm_drift)

# VIX chip (optional)
vix_val = np.nan
try:
    px = fetch_prices_daily(["^VIX"], days=45)
    if px is not None and not px.empty:
        vix_val = float(px.iloc[-1, 0])
except Exception:
    vix_val = np.nan

# Regime label (simple)
regime = "Neutral"
try:
    spy = fetch_prices_daily(["SPY"], days=120)
    if spy is not None and not spy.empty:
        sma60 = float(spy.iloc[-60:, 0].mean())
        last = float(spy.iloc[-1, 0])
        regime = "Risk-On" if last >= sma60 else "Risk-Off"
except Exception:
    pass

# Big Wave Title (YOUR REQUEST)
st.markdown(f'<div class="waves-big-wave">{selected_wave}</div>', unsafe_allow_html=True)
st.markdown(f'<div class="waves-subhead">Mode: <b>{mode}</b> · Benchmark Snapshot: <b>{bm_id}</b> · Drift: <b>{bm_drift}</b> · Regime: <b>{regime}</b></div>', unsafe_allow_html=True)

# Sticky chips (scan-first)
st.markdown(
    f"""
<div class="waves-sticky">
  <span class="waves-chip"><b>Confidence</b>: {conf_level}</span>
  <span class="waves-chip"><b>Coverage</b>: {fmt_num(cov_sel.get('completeness_score'),1)}</span>
  <span class="waves-chip"><b>Last Data</b>: {cov_sel.get('last_date') or '—'}</span>
  <span class="waves-chip"><b>30D Alpha</b>: {fmt_pct(m.get('a30'))}</span>
  <span class="waves-chip"><b>TE</b>: {fmt_pct(m.get('te'))}</span>
  <span class="waves-chip"><b>MaxDD</b>: {fmt_pct(m.get('mdd'))}</span>
  <span class="waves-chip"><b>Risk Reaction</b>: {fmt_num(rr,1)} ({rr_grade})</span>
  <span class="waves-chip"><b>VIX</b>: {fmt_num(vix_val,1) if math.isfinite(vix_val) else '—'}</span>
</div>
""",
    unsafe_allow_html=True,
)

if show_defs:
    render_definitions(
        [
            "Canonical (Source of Truth)",
            "Return",
            "Alpha",
            "Tracking Error (TE)",
            "Information Ratio (IR)",
            "Max Drawdown (MaxDD)",
            "VaR 95% (daily)",
            "CVaR 95% (daily)",
            "Sharpe",
            "Sortino",
            "Benchmark Snapshot / Drift",
            "Coverage Score",
            "Difficulty vs SPY",
            "Risk Reaction Score",
            "Analytics Scorecard",
        ],
        title="Definitions (what these metrics mean)",
    )

if show_truth_table:
    st.markdown("### Cohesion Lock (Truth Table — Canonical Source Proof)")
    st.caption("This is the governance proof that the same canonical dataset is used across the console (no crisscross).")
    truth = pd.DataFrame(
        [
            {"Field": "Selected Wave", "Value": selected_wave},
            {"Field": "Mode", "Value": mode},
            {"Field": "Canonical window end date", "Value": m.get("end_date")},
            {"Field": "Rows", "Value": int(m.get("rows") or 0)},
            {"Field": "BM Snapshot", "Value": bm_id},
            {"Field": "BM Drift", "Value": bm_drift},
            {"Field": "30D Return", "Value": fmt_pct(m.get("r30"))},
            {"Field": "30D Alpha", "Value": fmt_pct(m.get("a30"))},
            {"Field": "TE", "Value": fmt_pct(m.get("te"))},
            {"Field": "MaxDD", "Value": fmt_pct(m.get("mdd"))},
            {"Field": "Sharpe", "Value": fmt_num(m.get("sharpe"))},
            {"Field": "CVaR95 (daily)", "Value": fmt_pct(m.get("cvar95"))},
            {"Field": "Risk Reaction Score", "Value": f"{fmt_num(rr,1)} ({rr_grade})"},
        ]
    )
    st.dataframe(truth, use_container_width=True, hide_index=True)

# ============================================================
# Tabs (consolidated, scan-first)
# ============================================================
tabs = ["IC Summary", "Overview", "Advanced Analytics", "Governance", "Compare"]
t1, t2, t3, t4, t5 = st.tabs(tabs)

# ============================================================
# TAB 1 — IC Summary (Executive One-Pager)
# ============================================================
with t1:
    st.markdown("## Executive IC One-Pager")

    # Tiles: 2 columns card
    cL, cR = st.columns([1, 1], gap="large")

    with cL:
        st.markdown('<div class="waves-card">', unsafe_allow_html=True)
        st.markdown("### What is this wave?")
        st.write("This Wave is an AI-managed portfolio sleeve with a governed benchmark definition and audit-ready analytics outputs.")
        st.markdown("### Is the data trustworthy?")
        st.write(f"**{conf_level}** — {conf_reason}")
        st.markdown("### Is the benchmark stable?")
        if bm_drift == "stable":
            st.success(f"Stable — {bm_id}")
        else:
            st.warning(f"Drift detected — {bm_id}")
        st.markdown("</div>", unsafe_allow_html=True)

    with cR:
        # 6–8 big tiles
        r1 = m.get("r1")
        a1 = m.get("a1")
        tile_rows = [
            ("30D Alpha", fmt_pct(m.get("a30")), "vs benchmark (canonical)"),
            ("30D Return", fmt_pct(m.get("r30")), "wave return (canonical)"),
            ("TE (Active Risk)", fmt_pct(m.get("te")), f"band: {te_risk_band(safe_float(m.get('te')))}"),
            ("MaxDD", fmt_pct(m.get("mdd")), "peak-to-trough"),
            ("CVaR95 (daily)", fmt_pct(m.get("cvar95")), "tail loss proxy"),
            ("Risk Reaction", f"{fmt_num(rr,1)} / 100", f"grade: {rr_grade}"),
        ]
        st.markdown('<div class="waves-card">', unsafe_allow_html=True)
        st.markdown("### Key Tiles")
        cols = st.columns(2, gap="small")
        for i, (lab, val, sub) in enumerate(tile_rows):
            with cols[i % 2]:
                st.markdown(
                    f"""
<div class="waves-tile">
  <div class="waves-tile-label">{lab}</div>
  <div class="waves-tile-value">{val}</div>
  <div class="waves-tile-sub">{sub}</div>
</div>
""",
                    unsafe_allow_html=True,
                )
        st.markdown("</div>", unsafe_allow_html=True)

    # Deterministic explanation block
    expl = explain_from_metrics(
        selected_wave, mode, cov_sel, bm_drift, bm_id, bm_diff, m, diff_proxy, rr
    )

    st.markdown("### Key Wins / Key Risks / Next Actions")
    w1, w2, w3 = st.columns(3, gap="large")
    with w1:
        st.markdown('<div class="waves-card">', unsafe_allow_html=True)
        st.markdown("**Key Wins**")
        wins = []
        if math.isfinite(safe_float(m.get("a30"))) and safe_float(m.get("a30")) > 0:
            wins.append("Positive 30D alpha vs benchmark.")
        if conf_level == "High":
            wins.append("High confidence data integrity.")
        if bm_drift == "stable":
            wins.append("Benchmark snapshot stable in-session.")
        if not wins:
            wins = ["No major wins flagged on this window."]
        for x in wins:
            st.write("• " + x)
        st.markdown("</div>", unsafe_allow_html=True)

    with w2:
        st.markdown('<div class="waves-card">', unsafe_allow_html=True)
        st.markdown("**Key Risks**")
        risks = []
        if bm_drift != "stable":
            risks.append("Benchmark drift variability (governance issue for memos).")
        if cov_sel.get("flags"):
            risks.append("Data integrity flags present.")
        if rr_grade in ["D", "F"]:
            risks.append("Risk posture looks stressed (RR score low).")
        if not risks:
            risks = ["No major risks flagged on this window."]
        for x in risks:
            st.write("• " + x)
        st.markdown("</div>", unsafe_allow_html=True)

    with w3:
        st.markdown('<div class="waves-card">', unsafe_allow_html=True)
        st.markdown("**Next Actions**")
        for x in expl["next_actions"]:
            st.write("• " + x)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("### Deterministic AI Explanation Layer (rules-based)")
    e1, e2 = st.columns([1, 1], gap="large")
    with e1:
        st.markdown('<div class="waves-card">', unsafe_allow_html=True)
        st.markdown("**Headline**")
        for x in expl["headline"]:
            st.write("• " + x)
        st.markdown("**What changed recently**")
        for x in expl["what_changed"]:
            st.write("• " + x)
        st.markdown("</div>", unsafe_allow_html=True)
    with e2:
        st.markdown('<div class="waves-card">', unsafe_allow_html=True)
        st.markdown("**Why the alpha looks like this**")
        for x in expl["why_alpha"]:
            st.write("• " + x)
        st.markdown("**Risk driver**")
        for x in expl["risk_driver"]:
            st.write("• " + x)
        st.markdown("**What to verify**")
        for x in expl["what_to_verify"]:
            st.write("• " + x)
        st.markdown("</div>", unsafe_allow_html=True)

# ============================================================
# TAB 2 — Overview (clean + cohesive)
# ============================================================
with t2:
    st.markdown("## Overview (Canonical Performance)")

    if hist_sel is None or hist_sel.empty:
        st.warning("No history available for selected wave/mode.")
    else:
        # Small overview table (selected wave only)
        ov = pd.DataFrame(
            [
                {"Window": "1D", "Return": m.get("r1"), "Alpha": m.get("a1")},
                {"Window": "30D", "Return": m.get("r30"), "Alpha": m.get("a30")},
                {"Window": "60D", "Return": m.get("r60"), "Alpha": m.get("a60")},
                {"Window": "365D", "Return": m.get("r365"), "Alpha": m.get("a365")},
            ]
        )
        if not show_365:
            ov = ov[ov["Window"].isin(["1D", "30D", "60D"])].copy()

        ov["Return"] = ov["Return"].apply(lambda x: fmt_pct(x))
        ov["Alpha"] = ov["Alpha"].apply(lambda x: fmt_pct(x))

        st.dataframe(ov, use_container_width=True, hide_index=True)

        st.markdown("### Benchmark Fidelity Inspector")
        f1, f2, f3, f4 = st.columns(4, gap="small")
        with f1:
            st.metric("Drift", bm_drift)
        with f2:
            st.metric("Difficulty vs SPY", fmt_num(diff_proxy.get("difficulty_vs_spy"), 1))
        with f3:
            st.metric("Top Weight", fmt_pct(diff_proxy.get("top_weight"), 1))
        with f4:
            st.metric("Active Risk Band", te_risk_band(safe_float(m.get("te"))))

        if bm_drift != "stable":
            st.caption("Benchmark composition change vs last session snapshot (meaningful deltas).")
            if bm_diff is not None and not bm_diff.empty:
                st.dataframe(bm_diff, use_container_width=True, hide_index=True)
            else:
                st.info("No diff rows available (or first snapshot this session).")

# ============================================================
# TAB 3 — Advanced Analytics (with definitions + “what’s happening”)
# ============================================================
with t3:
    st.markdown("## Advanced Analytics (Canonical)")

    if hist_sel is None or hist_sel.empty or len(hist_sel) < 20:
        st.warning("Not enough history to compute advanced analytics.")
    else:
        # metrics grid
        a1, a2, a3, a4 = st.columns(4, gap="small")
        with a1:
            st.metric("TE (annualized)", fmt_pct(m.get("te")))
        with a2:
            st.metric("IR", fmt_num(m.get("ir"), 2))
        with a3:
            st.metric("Sharpe", fmt_num(m.get("sharpe"), 2))
        with a4:
            st.metric("Sortino", fmt_num(m.get("sortino"), 2))

        b1, b2, b3, b4 = st.columns(4, gap="small")
        with b1:
            st.metric("MaxDD", fmt_pct(m.get("mdd")))
        with b2:
            st.metric("Vol (ann.)", fmt_pct(m.get("vol")))
        with b3:
            st.metric("VaR95 (daily)", fmt_pct(m.get("var95")))
        with b4:
            st.metric("CVaR95 (daily)", fmt_pct(m.get("cvar95")))

        st.markdown("### What this output is telling us (deterministic)")
        expl2 = explain_from_metrics(
            selected_wave, mode, cov_sel, bm_drift, bm_id, bm_diff, m, diff_proxy, rr
        )
        st.markdown('<div class="waves-card">', unsafe_allow_html=True)
        st.markdown("**Interpretation**")
        for x in expl2["why_alpha"]:
            st.write("• " + x)
        for x in expl2["risk_driver"]:
            st.write("• " + x)
        st.markdown("**What we can do next**")
        for x in expl2["next_actions"]:
            st.write("• " + x)
        st.markdown("</div>", unsafe_allow_html=True)

        # Optional simple plots
        if go is not None:
            h = hist_sel.copy()
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=h.index, y=h["wave_nav"], name="Wave NAV"))
            fig.add_trace(go.Scatter(x=h.index, y=h["bm_nav"], name="Benchmark NAV"))
            fig.update_layout(height=360, margin=dict(l=40, r=20, t=40, b=40), title="NAV (Wave vs Benchmark)")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Plotly not installed — charts disabled (non-fatal).")

# ============================================================
# TAB 4 — Governance (Scorecards + exports)
# ============================================================
with t4:
    st.markdown("## Governance")

    st.markdown("### Analytics Quality Scorecard (All Waves)")
    if all_waves:
        score_df = compute_analytics_scorecard_all_waves(all_waves, mode=mode, days=365)
        if score_df is None or score_df.empty:
            st.info("Scorecard unavailable (no data).")
        else:
            st.dataframe(score_df, use_container_width=True, hide_index=True)

            # quick highlight for selected wave
            sel_row = score_df[score_df["Wave"] == selected_wave]
            if not sel_row.empty:
                st.markdown("### Selected Wave — Scorecard Snapshot")
                st.dataframe(sel_row, use_container_width=True, hide_index=True)
    else:
        st.warning("No waves available to compute scorecards.")

    st.markdown("### Governance Notes")
    st.write("• Scorecard grades **analytics reliability**, not returns.")
    st.write("• Cohesion Lock shows canonical source values used everywhere.")
    st.write("• Benchmark drift + diffs make governance visible (institution-grade).")

# ============================================================
# TAB 5 — Compare (Wave A vs Wave B)
# ============================================================
with t5:
    st.markdown("## Wave-to-Wave Comparator")

    if not all_waves or len(all_waves) < 2:
        st.warning("Need at least 2 waves to compare.")
    else:
        c1, c2 = st.columns(2, gap="large")
        with c1:
            wave_a = st.selectbox("Wave A", all_waves, index=max(0, all_waves.index(selected_wave)) if selected_wave in all_waves else 0)
        with c2:
            # default to next wave if possible
            default_b = 1 if all_waves[0] == wave_a and len(all_waves) > 1 else 0
            wave_b = st.selectbox("Wave B", all_waves, index=default_b)

        comp = comparator_metrics(all_waves, mode=mode, wave_a=wave_a, wave_b=wave_b)
        if comp is None or comp.empty:
            st.info("Comparator unavailable (no data).")
        else:
            st.dataframe(comp, use_container_width=True, hide_index=True)
            st.caption("Returns/Alpha are in percent points. TE/MaxDD/CVaR are decimals (format inside table).")

st.markdown("---")
st.caption("WAVES Intelligence™ — Canonical Cohesion Console (governance-first analytics).")