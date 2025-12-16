# app.py — WAVES Intelligence™ Institutional Console (Vector OS Edition)
# FULL PRODUCTION FILE (NO PATCHES) — IRB-1.1 UX REORG (NO MATH CHANGES)
#
# Goal of this version:
#   • Keep EVERY module you had (no removals), but make the UI more intuitive
#   • Consolidate overlapping tabs
#   • Add a Definitions Hub (searchable glossary + consistent “Why it matters”)
#   • Add Demo Mode (hide heavy/diagnostic sections by default)
#
# Top-level tabs (6):
#   1) IC Summary
#   2) Performance
#   3) Risk & Exposure
#   4) Benchmark Integrity
#   5) Decisions & Activity
#   6) Governance & Exports
#
# Notes:
#   • Engine math NOT modified.
#   • Robust history loader: engine functions → wave_history.csv fallback
#   • Decision Engine optional: app will not crash if decision_engine.py missing.
#   • Diagnostics are preserved, but tucked behind Demo Mode.

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
    from decision_engine import generate_decisions  # required function in your file
except Exception as e:
    generate_decisions = None
    DECISION_IMPORT_ERROR = e

try:
    from decision_engine import build_daily_wave_activity  # enhancement function
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

/* Section blocks */
.waves-card {
  border: 1px solid rgba(255,255,255,0.10);
  border-radius: 14px;
  padding: 12px 14px;
  background: rgba(255,255,255,0.03);
}

/* Small helper text */
.waves-muted {
  opacity: 0.85;
  font-size: 0.92rem;
}

/* Reduce whitespace for mobile */
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


def safe_df(df: Optional[pd.DataFrame]) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    return df.copy()


def google_quote_link(ticker: str) -> str:
    t = str(ticker).strip().upper()
    # (Google will still open common tickers fine; you can later enhance exchange mapping if desired)
    return f"https://www.google.com/finance/quote/{t}"


def safe_mode_list() -> List[str]:
    return ["Standard", "Alpha-Minus-Beta", "Private Logic"]


# ============================================================
# Basic return/risk math (unchanged)
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


def drawdown_series(nav: pd.Series) -> pd.Series:
    nav = safe_series(nav).astype(float)
    if len(nav) < 2:
        return pd.Series(dtype=float)
    peak = nav.cummax()
    return ((nav / peak) - 1.0).rename("drawdown")


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


def alpha_persistence(alpha_series: pd.Series) -> float:
    a = safe_series(alpha_series).dropna()
    if len(a) < 30:
        return float("nan")
    return float((a > 0).mean())


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
# Styling helpers: percent-point matrix + green/red heat
# ============================================================
def _heat_color(val: Any) -> str:
    try:
        if val is None:
            return ""
        v = float(val)
        if math.isnan(v):
            return ""
        if v > 0:
            return "background-color: rgba(0, 200, 120, 0.18);"
        if v < 0:
            return "background-color: rgba(255, 60, 60, 0.16);"
        return "background-color: rgba(255, 255, 255, 0.04);"
    except Exception:
        return ""


def style_perf_df(df: pd.DataFrame) -> "pd.io.formats.style.Styler":
    if df is None or df.empty:
        return df.style
    cols = [c for c in df.columns if ("Return" in c or "Alpha" in c or "MaxDD" in c or "TE" in c)]
    sty = df.style
    for c in cols:
        try:
            sty = sty.applymap(_heat_color, subset=[c])
        except Exception:
            pass
    return sty


# ============================================================
# Definitions / Glossary Hub (expanded)
# ============================================================
GLOSSARY: Dict[str, Dict[str, str]] = {
    "Return": {
        "plain": "Portfolio return over the window (not annualized unless stated).",
        "formula": "(End / Start) − 1",
        "interpret": "Higher is better, but always compare to benchmark and drawdown/vol.",
    },
    "Alpha": {
        "plain": "Return minus Benchmark return over the same window (relative performance).",
        "formula": "Return(Wave) − Return(Benchmark)",
        "interpret": "Positive alpha means outperformance versus its benchmark definition.",
    },
    "Tracking Error (TE)": {
        "plain": "How much the Wave deviates from its benchmark day-to-day (active risk).",
        "formula": "stdev(Wave daily − BM daily) × sqrt(252)",
        "interpret": "Higher TE = more active risk; alpha should justify it.",
    },
    "Information Ratio (IR)": {
        "plain": "Risk-adjusted alpha: excess return per unit of tracking error.",
        "formula": "Excess Return / TE",
        "interpret": "IR > 0.5 is strong; negative IR means underperforming for the risk taken.",
    },
    "Max Drawdown (MaxDD)": {
        "plain": "Worst peak-to-trough decline over the period.",
        "formula": "min(NAV / peakNAV − 1)",
        "interpret": "Smaller (less negative) is better; big drawdowns reduce investor comfort.",
    },
    "Benchmark Snapshot / Drift": {
        "plain": "Fingerprint of the benchmark mix. Drift means the benchmark changed in-session.",
        "formula": "hash(ticker:weight list)",
        "interpret": "If drift occurs during a demo/IC review, freeze benchmark mix for consistency.",
    },
    "Coverage Score": {
        "plain": "0–100 heuristic of data completeness + freshness.",
        "formula": "penalize missing business days + staleness",
        "interpret": "Low coverage can create misleading alpha/TE; verify history integrity first.",
    },
    "Difficulty vs SPY": {
        "plain": "Proxy for how concentrated/diversified the benchmark mix is vs SPY.",
        "formula": "HHI/entropy-based transform",
        "interpret": "Higher may be harder to beat consistently; use as a context cue, not a truth.",
    },
    "WaveScore (Console Approx.)": {
        "plain": "Console-only approximation; NOT the locked WAVESCORE™ spec.",
        "formula": "IR + drawdown + hit-rate + TE proxy blend",
        "interpret": "Use for ranking/triage; official WAVESCORE™ comes from the locked spec.",
    },
    "Decision Intelligence": {
        "plain": "OS layer: actions/watch/notes based on observable analytics (not advice).",
        "formula": "rules + thresholds on ctx",
        "interpret": "Designed to guide diligence and monitoring—never a trade command.",
    },
}


def render_definitions(keys: List[str], title: str = "Definitions"):
    with st.expander(title):
        for k in keys:
            d = GLOSSARY.get(k)
            if not d:
                st.markdown(f"**{k}:** (definition not found)")
                continue
            st.markdown(f"### {k}")
            st.markdown(f"**Plain English:** {d.get('plain','')}")
            st.markdown(f"**Formula:** {d.get('formula','')}")
            st.markdown(f"**Why it matters:** {d.get('interpret','')}")
            st.divider()


def render_definitions_hub():
    st.subheader("Definitions Hub")
    st.caption("Search a term to see plain-English meaning, a lightweight formula, and why it matters in IC diligence.")
    q = st.text_input("Search definitions (e.g., alpha, drawdown, IR, TE, drift)", value="")
    keys = list(GLOSSARY.keys())
    if q.strip():
        ql = q.strip().lower()
        keys = [k for k in keys if ql in k.lower() or ql in (GLOSSARY[k].get("plain","").lower())]
    if not keys:
        st.info("No matches found.")
        return
    for k in keys:
        d = GLOSSARY.get(k, {})
        with st.expander(k, expanded=False):
            st.markdown(f"**Plain English:** {d.get('plain','')}")
            st.markdown(f"**Formula:** {d.get('formula','')}")
            st.markdown(f"**Why it matters:** {d.get('interpret','')}")
            # ============================================================
# Confidence / Robustness meter (Trust cue)
# ============================================================
def confidence_from_integrity(cov: Dict[str, Any], bm_drift: str) -> Tuple[str, str]:
    """
    Returns (level, reason)
      level in {"High","Medium","Low"}
    """
    try:
        score = float(cov.get("completeness_score")) if cov.get("completeness_score") is not None else float("nan")
        age = float(cov.get("age_days")) if cov.get("age_days") is not None else float("nan")
        rows = int(cov.get("rows", 0)) if cov.get("rows") is not None else 0

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
                return [str(x) for x in waves]
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


def benchmark_snapshot_id(wave_name: str, bm_mix_df: pd.DataFrame) -> str:
    try:
        if bm_mix_df is None or bm_mix_df.empty:
            return "BM-NA"
        rows = bm_mix_df[bm_mix_df["Wave"] == wave_name].copy() if "Wave" in bm_mix_df.columns else bm_mix_df.copy()
        rows = _normalize_bm_rows(rows.rename(columns={"Ticker": "Ticker", "Weight": "Weight"}))
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


# ============================================================
# WaveScore (console-side approximation) — unchanged
# ============================================================
@st.cache_data(show_spinner=False)
def compute_wavescore_for_all_waves(all_waves: List[str], mode: str, days: int = 365) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for wave in all_waves:
        hist = compute_wave_history(wave, mode=mode, days=days)
        hist = _standardize_history(hist)
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
# Alpha Heatmap (All Waves × timeframe)
# ============================================================
@st.cache_data(show_spinner=False)
def build_alpha_matrix(all_waves: List[str], mode: str) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for wname in all_waves:
        hist = compute_wave_history(wname, mode=mode, days=365)
        hist = _standardize_history(hist)
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
        a365 = ret_from_nav(nav_w, min(365, len(nav_w))) - ret_from_nav(nav_b, min(365, len(nav_b)))

        rows.append({"Wave": wname, "1D Alpha": a1, "30D Alpha": a30, "60D Alpha": a60, "365D Alpha": a365})

    return pd.DataFrame(rows).sort_values("Wave")


def plot_alpha_heatmap(alpha_df: pd.DataFrame, title: str):
    if go is None or alpha_df is None or alpha_df.empty:
        st.info("Heatmap unavailable (Plotly missing or no data).")
        return

    df = alpha_df.copy()
    cols = [c for c in ["1D Alpha", "30D Alpha", "60D Alpha", "365D Alpha"] if c in df.columns]
    if not cols:
        st.info("No alpha columns to plot.")
        return

    z = df[cols].values
    y = df["Wave"].tolist()
    x = cols

    v = float(np.nanmax(np.abs(z))) if np.isfinite(z).any() else 0.10
    if not math.isfinite(v) or v <= 0:
        v = 0.10

    fig = go.Figure(data=go.Heatmap(z=z, x=x, y=y, zmin=-v, zmax=v, colorbar=dict(title="Alpha")))
    fig.update_layout(title=title, height=min(950, 260 + 22 * max(12, len(y))), margin=dict(l=80, r=40, t=60, b=40))
    st.plotly_chart(fig, use_container_width=True)


# ============================================================
# Performance Matrix (Returns + Alpha) — percent points
# ============================================================
@st.cache_data(show_spinner=False)
def build_performance_matrix(all_waves: List[str], mode: str, selected_wave: str, days: int = 365) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for w in all_waves:
        h = compute_wave_history(w, mode=mode, days=days)
        h = _standardize_history(h)
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

    for c in [c for c in df.columns if "Return" in c or "Alpha" in c]:
        df[c] = pd.to_numeric(df[c], errors="coerce") * 100.0  # percent points

    if "Wave" in df.columns and selected_wave in set(df["Wave"]):
        top = df[df["Wave"] == selected_wave]
        rest = df[df["Wave"] != selected_wave]
        df = pd.concat([top, rest], axis=0)

    return df.reset_index(drop=True)


# ============================================================
# Alerts & Flags (unchanged)
# ============================================================
def build_alerts(selected_wave: str, mode: str, hist: pd.DataFrame, cov: Dict[str, Any], bm_drift: str, te: float, a30: float, mdd: float) -> List[str]:
    notes: List[str] = []
    try:
        if cov.get("flags"):
            notes.append("Data Integrity: " + "; ".join(cov["flags"]))
        if bm_drift != "stable":
            notes.append("Benchmark Drift: snapshot changed in-session (freeze benchmark mix for demos).")
        if math.isfinite(a30) and abs(a30) >= 0.08:
            notes.append("Large 30D alpha: verify benchmark mix + missing days (big alpha can be real or coverage-driven).")
        if math.isfinite(te) and te >= 0.20:
            notes.append("High tracking error: active risk elevated vs benchmark.")
        if math.isfinite(mdd) and mdd <= -0.25:
            notes.append("Deep drawdown: consider stronger SmartSafe posture in stress regimes.")
        if hist is not None and not hist.empty and cov.get("age_days", 0) is not None and cov.get("age_days", 0) >= 5:
            notes.append("Stale history: last datapoint is >=5 days old (check engine writes).")
        if not notes:
            notes.append("No major anomalies detected on this window.")
        return notes
    except Exception:
        return ["Alert system error (non-fatal)."]


# ============================================================
# Decision Context Builder (kept; expanded fields allowed)
# ============================================================
def build_decision_ctx(
    wave: str,
    mode: str,
    bm_id: str,
    bm_drift: str,
    cov: Dict[str, Any],
    vix_val: float,
    regime: str,
    r30: float,
    a30: float,
    r60: float,
    a60: float,
    r365: float,
    a365: float,
    te: float,
    ir: float,
    mdd: float,
    wavescore: float,
    rank: Optional[int],
) -> Dict[str, Any]:
    return {
        "wave_name": wave,
        "wave": wave,
        "mode": mode,
        "bm_id": bm_id,
        "bm_drift": bm_drift,
        "completeness_score": cov.get("completeness_score"),
        "age_days": cov.get("age_days"),
        "rows": cov.get("rows"),
        "vix": vix_val,
        "regime": regime,
        "r30": r30,
        "a30": a30,
        "r60": r60,
        "a60": a60,
        "r365": r365,
        "a365": a365,
        "te": te,
        "ir": ir,
        "mdd": mdd,
        "wavescore": wavescore,
        "rank": rank,
    }


# ============================================================
# Governance Export Pack (unchanged)
# ============================================================
def make_ic_pack_markdown(
    wave: str,
    mode: str,
    bm_id: str,
    bm_drift: str,
    cov: Dict[str, Any],
    ws_val: float,
    ws_grade: str,
    rank: Optional[int],
    r30: float,
    a30: float,
    r60: float,
    a60: float,
    r365: float,
    a365: float,
    te: float,
    ir: float,
    mdd: float,
    mdd_b: float,
    difficulty: Dict[str, Any],
) -> str:
    ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    return f"""# WAVES Intelligence™ — Governance / IC Pack

**Timestamp:** {ts}  
**Wave:** {wave}  
**Mode:** {mode}  

## Benchmark Integrity
- Snapshot ID: **{bm_id}**
- Drift status: **{bm_drift.upper()}**
- Difficulty vs SPY (proxy): **{fmt_num(difficulty.get('difficulty_vs_spy'), 2)}**
- HHI (concentration): **{fmt_num(difficulty.get('hhi'), 4)}**
- Entropy (diversification): **{fmt_num(difficulty.get('entropy'), 3)}**
- Top weight: **{fmt_pct(difficulty.get('top_weight'), 2)}**

## Coverage / Data Quality
- Rows: **{cov.get('rows', '—')}**
- First date: **{cov.get('first_date', '—')}**
- Last date: **{cov.get('last_date', '—')}**
- Age (days): **{cov.get('age_days', '—')}**
- Completeness score: **{fmt_num(cov.get('completeness_score', np.nan), 1)} / 100**
- Flags: **{'; '.join(cov.get('flags', [])) if cov.get('flags') else 'None'}**

## Performance vs Benchmark
- 30D Return: **{fmt_pct(r30)}** | 30D Alpha: **{fmt_pct(a30)}**
- 60D Return: **{fmt_pct(r60)}** | 60D Alpha: **{fmt_pct(a60)}**
- 365D Return: **{fmt_pct(r365)}** | 365D Alpha: **{fmt_pct(a365)}**

## Risk / Efficiency
- Tracking Error: **{fmt_pct(te)}**
- Information Ratio: **{fmt_num(ir, 2)}**
- Max Drawdown (Wave): **{fmt_pct(mdd)}**
- Max Drawdown (Benchmark): **{fmt_pct(mdd_b)}**

## WaveScore (Console Approx.)
- WaveScore: **{fmt_score(ws_val)}**
- Grade: **{ws_grade}**
- Rank: **{rank if rank else '—'}**
"""
# ============================================================
# MAIN UI
# ============================================================
st.title("WAVES Intelligence™ Institutional Console")

# Status banners (non-fatal)
if ENGINE_IMPORT_ERROR is not None:
    st.error("Engine import failed. The app will use CSV fallbacks where possible.")
    st.code(str(ENGINE_IMPORT_ERROR))

if DECISION_IMPORT_ERROR is not None:
    st.warning("Decision Engine import issue (non-fatal). Decision panels will fallback.")
    st.code(str(DECISION_IMPORT_ERROR))

# Discover waves
all_waves = get_all_waves_safe()
if not all_waves:
    st.warning("No waves discovered yet. If unexpected, check engine import + CSV files.")
    with st.expander("Diagnostics"):
        st.write("Files present:")
        st.write({p: os.path.exists(p) for p in ["wave_config.csv", "wave_weights.csv", "wave_history.csv", "list.csv", "waves_engine.py", "decision_engine.py"]})
    st.stop()

modes = safe_mode_list()

# Sidebar controls (UX upgrade)
with st.sidebar:
    st.header("Controls")

    demo_mode = st.toggle("Demo Mode (simplified)", value=True)
    show_365_default = st.toggle("Default: show 365D columns", value=False)

    mode = st.selectbox("Mode", modes, index=0)
    selected_wave = st.selectbox("Wave", all_waves, index=0)
    days = st.slider("History window (days)", min_value=90, max_value=1500, value=365, step=30)

    st.caption("If history is empty, app falls back to wave_history.csv automatically.")

    st.divider()
    st.caption("Quick help")
    st.write("• Use **IC Summary** for the call-ready view")
    st.write("• Use **Performance** for matrix + charts + holdings")
    st.write("• Use **Risk & Exposure** for risk + correlation + beta")

# Benchmark snapshot + drift
bm_mix = get_benchmark_mix()
bm_id = benchmark_snapshot_id(selected_wave, bm_mix)
bm_drift = benchmark_drift_status(selected_wave, mode, bm_id)

# Load history
hist = compute_wave_history(selected_wave, mode=mode, days=days)
hist = _standardize_history(hist)
cov = coverage_report(hist)

# Precompute stats
mdd = np.nan
mdd_b = np.nan
r30 = np.nan
a30 = np.nan
r60 = np.nan
a60 = np.nan
r365 = np.nan
a365 = np.nan
te = np.nan
ir = np.nan

if hist is not None and (not hist.empty) and len(hist) >= 2:
    mdd = max_drawdown(hist["wave_nav"])
    mdd_b = max_drawdown(hist["bm_nav"])

    r30 = ret_from_nav(hist["wave_nav"], min(30, len(hist)))
    a30 = r30 - ret_from_nav(hist["bm_nav"], min(30, len(hist)))

    r60 = ret_from_nav(hist["wave_nav"], min(60, len(hist)))
    a60 = r60 - ret_from_nav(hist["bm_nav"], min(60, len(hist)))

    r365 = ret_from_nav(hist["wave_nav"], min(365, len(hist)))
    a365 = r365 - ret_from_nav(hist["bm_nav"], min(365, len(hist)))

    te = tracking_error(hist["wave_ret"], hist["bm_ret"])
    ir = information_ratio(hist["wave_nav"], hist["bm_nav"], te)

# Regime + VIX
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

# WaveScore
ws_df = compute_wavescore_for_all_waves(all_waves, mode=mode, days=min(days, 365))
rank = None
ws_val = np.nan
if not ws_df.empty and selected_wave in set(ws_df["Wave"]):
    ws_val = float(ws_df[ws_df["Wave"] == selected_wave]["WaveScore"].iloc[0])
    ws_df_sorted = ws_df.sort_values("WaveScore", ascending=False, na_position="last").reset_index(drop=True)
    try:
        rank = int(ws_df_sorted.index[ws_df_sorted["Wave"] == selected_wave][0] + 1)
    except Exception:
        rank = None

# Benchmark difficulty proxy
bm_rows = pd.DataFrame()
try:
    if bm_mix is not None and not bm_mix.empty and "Wave" in bm_mix.columns:
        bm_rows = bm_mix[bm_mix["Wave"] == selected_wave].copy()
        if "Ticker" in bm_rows.columns and "Weight" in bm_rows.columns:
            bm_rows = bm_rows[["Ticker", "Weight"]].copy()
            bm_rows = _normalize_bm_rows(bm_rows)
except Exception:
    bm_rows = pd.DataFrame()
difficulty = benchmark_difficulty_proxy(bm_rows)

# Confidence meter
conf_level, conf_reason = confidence_from_integrity(cov, bm_drift)

# Sticky chips
chips = []
chips.append(f"BM: {bm_id} · {'Stable' if bm_drift=='stable' else 'DRIFT'}")
chips.append(f"Coverage: {fmt_num(cov.get('completeness_score', np.nan),1)}/100")
chips.append(f"Rows: {cov.get('rows','—')} · Age: {cov.get('age_days','—')}")
chips.append(f"Confidence: {conf_level}")
chips.append(f"Regime: {regime}")
chips.append(f"VIX: {fmt_num(vix_val,1) if math.isfinite(vix_val) else '—'}")
chips.append(f"30D α: {fmt_pct(a30)} · r: {fmt_pct(r30)}")
chips.append(f"60D α: {fmt_pct(a60)} · r: {fmt_pct(r60)}")
chips.append(f"365D α: {fmt_pct(a365)} · r: {fmt_pct(r365)}")
chips.append(f"TE: {fmt_pct(te)} · IR: {fmt_num(ir,2)}")
chips.append(f"WaveScore: {fmt_score(ws_val)} ({_grade_from_score(ws_val)}) · Rank: {rank if rank else '—'}")

st.markdown('<div class="waves-sticky">', unsafe_allow_html=True)
for c in chips:
    st.markdown(f'<span class="waves-chip">{c}</span>', unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)
st.caption("Observational analytics only (not trading advice).")

# Tabs — consolidated IA
tabs = st.tabs([
    "IC Summary",
    "Performance",
    "Risk & Exposure",
    "Benchmark Integrity",
    "Decisions & Activity",
    "Governance & Exports",
])
# ============================================================
# TAB 0: IC Summary (call-ready landing)
# ============================================================
with tabs[0]:
    st.subheader(f"IC Summary — {selected_wave} ({mode})")
    st.caption("Call-ready summary: what matters now, why it matters, and what to check next.")

    ctx = build_decision_ctx(
        wave=selected_wave,
        mode=mode,
        bm_id=bm_id,
        bm_drift=bm_drift,
        cov=cov,
        vix_val=vix_val,
        regime=regime,
        r30=r30, a30=a30,
        r60=r60, a60=a60,
        r365=r365, a365=a365,
        te=te,
        ir=ir,
        mdd=mdd,
        wavescore=ws_val,
        rank=rank,
    )

    left, right = st.columns([1.25, 1.0])

    with left:
        st.markdown('<div class="waves-card">', unsafe_allow_html=True)
        st.markdown("### Decision Intelligence (Translator)")
        st.caption("If the Decision Engine is present, this becomes your ‘what to say + what to check’ output.")

        if generate_decisions is None:
            st.info("Decision Engine not available (generate_decisions missing).")
            st.write("Fallback: use Diligence Flags + Coverage + Benchmark Drift + Alpha/TE/IR.")
        else:
            try:
                d = generate_decisions(ctx)
            except Exception as e:
                d = {"actions": [f"Decision engine error: {e}"], "watch": [], "notes": []}

            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown("**Action**")
                for x in d.get("actions", []) or []:
                    st.write(f"• {x}")
            with c2:
                st.markdown("**Watch**")
                for x in d.get("watch", []) or []:
                    st.write(f"• {x}")
            with c3:
                st.markdown("**Notes**")
                for x in d.get("notes", []) or []:
                    st.write(f"• {x}")

        st.markdown("</div>", unsafe_allow_html=True)

        st.divider()

        st.markdown('<div class="waves-card">', unsafe_allow_html=True)
        st.markdown("### Integrity & Confidence")
        st.write(f"**Confidence:** {conf_level} — {conf_reason}")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Coverage", fmt_num(cov.get("completeness_score", np.nan), 1))
        c2.metric("Age (days)", cov.get("age_days", "—"))
        c3.metric("Rows", cov.get("rows", 0))
        c4.metric("Benchmark Drift", "Stable" if bm_drift == "stable" else "DRIFT")

        if cov.get("flags"):
            st.markdown("**Flags:**")
            for f in cov.get("flags", []) or []:
                st.write(f"• {f}")
        else:
            st.write("No data integrity flags detected on this window.")

        st.markdown("</div>", unsafe_allow_html=True)

        st.divider()
        st.markdown('<div class="waves-card">', unsafe_allow_html=True)
        st.markdown("### Talk Track (1-minute)")
        # Simple, deterministic talk track (no external AI calls)
        msg = []
        msg.append(f"Benchmark snapshot is **{bm_id}** ({'stable' if bm_drift=='stable' else 'DRIFT'}).")
        msg.append(f"30D alpha is **{fmt_pct(a30)}** with tracking error **{fmt_pct(te)}** (IR **{fmt_num(ir,2)}**).")
        msg.append(f"Max drawdown is **{fmt_pct(mdd)}**; regime tag is **{regime}** (VIX {fmt_num(vix_val,1)}).")
        msg.append(f"Data confidence is **{conf_level}** (coverage {fmt_num(cov.get('completeness_score',np.nan),1)}/100).")
        st.write(" ".join(msg))
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown('<div class="waves-card">', unsafe_allow_html=True)
        st.markdown("### Performance Snapshot")
        c1, c2, c3 = st.columns(3)
        c1.metric("30D Return", fmt_pct(r30))
        c2.metric("30D Alpha", fmt_pct(a30))
        c3.metric("Max Drawdown", fmt_pct(mdd))

        c4, c5, c6 = st.columns(3)
        c4.metric("Tracking Error", fmt_pct(te))
        c5.metric("IR", fmt_num(ir, 2))
        c6.metric("WaveScore", f"{fmt_score(ws_val)} ({_grade_from_score(ws_val)})")

        st.markdown("</div>", unsafe_allow_html=True)

        st.divider()

        st.markdown('<div class="waves-card">', unsafe_allow_html=True)
        st.markdown("### Orientation Chart")
        chart_mode = st.radio("Chart", ["NAV vs Benchmark", "Rolling 30D Alpha"], horizontal=True)

        if hist is None or hist.empty or len(hist) < 5:
            st.warning("Not enough history for charts for this wave/mode.")
        else:
            if chart_mode == "NAV vs Benchmark":
                nav_df = pd.concat(
                    [hist["wave_nav"].rename("Wave NAV"), hist["bm_nav"].rename("Benchmark NAV")],
                    axis=1
                ).dropna()
                if not nav_df.empty:
                    nav_df = nav_df / nav_df.iloc[0]
                    st.line_chart(nav_df)
                else:
                    st.info("NAV chart unavailable.")
            else:
                ra = rolling_alpha_from_nav(hist["wave_nav"], hist["bm_nav"], window=30).dropna()
                if len(ra):
                    st.line_chart((ra * 100.0).rename("Rolling 30D Alpha (%)"))
                else:
                    st.info("Not enough data for rolling 30D alpha.")
        st.markdown("</div>", unsafe_allow_html=True)

    st.divider()

    st.markdown("### Top Holdings (Grounding)")
    hold = get_wave_holdings(selected_wave)
    if hold is None or hold.empty:
        st.info("Holdings unavailable.")
    else:
        hold2 = hold.copy()
        hold2["Ticker"] = hold2["Ticker"].astype(str).str.upper().str.strip()
        hold2["Weight"] = pd.to_numeric(hold2["Weight"], errors="coerce").fillna(0.0)
        tot = float(hold2["Weight"].sum())
        if tot > 0:
            hold2["Weight"] = hold2["Weight"] / tot
        hold2 = hold2.sort_values("Weight", ascending=False).reset_index(drop=True)
        hold2["Google"] = hold2["Ticker"].apply(lambda t: google_quote_link(str(t)))
        hold2["Weight %"] = hold2["Weight"] * 100.0

        try:
            st.dataframe(
                hold2.head(10)[["Ticker", "Name", "Weight %", "Google"]],
                use_container_width=True,
                column_config={
                    "Weight %": st.column_config.NumberColumn("Weight %", format="%.2f%%"),
                    "Google": st.column_config.LinkColumn("Google", display_text="Open"),
                },
            )
        except Exception:
            st.dataframe(hold2.head(10), use_container_width=True)

    st.divider()
    render_definitions_hub()


# ============================================================
# TAB 1: Performance (matrix + heatmap + charts + coverage + holdings)
# ============================================================
with tabs[1]:
    st.subheader("Performance — Matrix + Heatmap + Selected Charts + Holdings")
    st.caption("This is the main scanning floor: matrix → heatmap → selected wave deep view.")

    cA, cB, cC = st.columns([1.2, 1.0, 1.2])
    with cA:
        sort_by = st.selectbox(
            "Sort matrix by",
            ["Selected First (default)", "30D Alpha", "60D Alpha", "30D Return", "WaveScore", "Rows"],
            index=0,
        )
    with cB:
        show_365 = st.toggle("Show 365D columns", value=bool(show_365_default))
    with cC:
        st.caption("Tip: Pros scan 30D/60D first; keep 365D hidden unless needed.")

    perf_df = build_performance_matrix(all_waves, mode=mode, selected_wave=selected_wave, days=min(days, 365))

    if perf_df is None or perf_df.empty:
        st.info("Performance matrix unavailable (no history).")
    else:
        df = perf_df.copy()

        if sort_by != "Selected First (default)":
            key_map = {"30D Alpha": "30D Alpha", "60D Alpha": "60D Alpha", "30D Return": "30D Return", "Rows": "Rows"}
            if sort_by in key_map and key_map[sort_by] in df.columns:
                df = df.sort_values(key_map[sort_by], ascending=False, na_position="last")
            elif sort_by == "WaveScore":
                if ws_df is not None and not ws_df.empty and "Wave" in ws_df.columns:
                    df = df.merge(ws_df[["Wave", "WaveScore"]], on="Wave", how="left")
                    df = df.sort_values("WaveScore", ascending=False, na_position="last").drop(columns=["WaveScore"], errors="ignore")

            if selected_wave in set(df["Wave"]):
                top = df[df["Wave"] == selected_wave]
                rest = df[df["Wave"] != selected_wave]
                df = pd.concat([top, rest], axis=0)

        if not show_365:
            drop_cols = [c for c in df.columns if c.startswith("365D ")]
            df = df.drop(columns=drop_cols, errors="ignore")

        st.dataframe(style_perf_df(df), use_container_width=True)
        st.caption("Values shown as **percent points**. Green = positive, Red = negative.")

        render_definitions(
            keys=["Alpha", "Return", "Tracking Error (TE)", "Information Ratio (IR)"],
            title="Definitions (Matrix)",
        )

    st.divider()
    st.subheader("Alpha Heatmap (All Waves × Timeframe)")
    alpha_df = build_alpha_matrix(all_waves, mode=mode)
    plot_alpha_heatmap(alpha_df, title=f"Alpha Heatmap — Mode: {mode}")

    st.divider()
    st.subheader("Selected Wave — Charts")
    if hist is None or hist.empty or len(hist) < 5:
        st.warning("Not enough history for charts for this wave/mode.")
    else:
        nav_df = pd.concat(
            [hist["wave_nav"].rename("Wave NAV"), hist["bm_nav"].rename("Benchmark NAV")],
            axis=1,
        ).dropna()
        if not nav_df.empty:
            nav_df = nav_df / nav_df.iloc[0]
            st.line_chart(nav_df)

        st.write("Rolling 30D Alpha")
        ra = rolling_alpha_from_nav(hist["wave_nav"], hist["bm_nav"], window=30).dropna()
        if len(ra):
            st.line_chart((ra * 100.0).rename("Rolling 30D Alpha (%)"))
        else:
            st.info("Not enough data for rolling 30D alpha.")

        st.write("Drawdown (Wave vs Benchmark)")
        dd_w = drawdown_series(hist["wave_nav"])
        dd_b = drawdown_series(hist["bm_nav"])
        dd_df = pd.concat([dd_w.rename("Wave"), dd_b.rename("Benchmark")], axis=1).dropna()
        if not dd_df.empty:
            st.line_chart(dd_df * 100.0)
        else:
            st.info("Drawdown chart unavailable.")

    st.divider()
    st.subheader("Coverage & Data Integrity")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("History Rows", cov.get("rows", 0))
    c2.metric("Last Data Age (days)", cov.get("age_days", "—"))
    c3.metric("Completeness Score", fmt_num(cov.get("completeness_score", np.nan), 1))
    c4.metric("Confidence", conf_level)
    st.caption(conf_reason)
    with st.expander("Coverage Details"):
        st.write(cov)

    st.divider()
    st.subheader("Top-10 Holdings (Clickable)")
    hold = get_wave_holdings(selected_wave)
    if hold is None or hold.empty:
        st.info("Holdings unavailable.")
    else:
        hold2 = hold.copy()
        hold2["Ticker"] = hold2["Ticker"].astype(str).str.upper().str.strip()
        hold2["Weight"] = pd.to_numeric(hold2["Weight"], errors="coerce").fillna(0.0)
        tot = float(hold2["Weight"].sum())
        if tot > 0:
            hold2["Weight"] = hold2["Weight"] / tot
        hold2 = hold2.sort_values("Weight", ascending=False).reset_index(drop=True)
        hold2["Google"] = hold2["Ticker"].apply(lambda t: google_quote_link(str(t)))
        hold2["Weight %"] = hold2["Weight"] * 100.0

        try:
            st.dataframe(
                hold2.head(10)[["Ticker", "Name", "Weight %", "Google"]],
                use_container_width=True,
                column_config={
                    "Weight %": st.column_config.NumberColumn("Weight %", format="%.2f%%"),
                    "Google": st.column_config.LinkColumn("Google", display_text="Open"),
                },
            )
        except Exception:
            st.dataframe(hold2.head(10), use_container_width=True)


# ============================================================
# TAB 2: Risk & Exposure (Risk Lab + Correlation + Beta + Drawdown)
# ============================================================
with tabs[2]:
    st.subheader("Risk & Exposure")
    st.caption("Risk diagnostics: drawdown, tail risk, correlation, and beta vs benchmark.")

    # Risk Lab
    st.markdown("### Risk Lab")
    if hist is None or hist.empty or len(hist) < 50:
        st.info("Not enough data to compute risk lab metrics.")
    else:
        r = hist["wave_ret"].dropna()
        sh = sharpe_ratio(r, 0.0)
        so = sortino_ratio(r, 0.0)
        dd = downside_deviation(r, 0.0)
        v95, c95 = var_cvar(r, 0.95)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Sharpe (0% rf)", fmt_num(sh, 2))
        c2.metric("Sortino (0% MAR)", fmt_num(so, 2))
        c3.metric("Downside Dev (ann)", fmt_pct(dd))
        c4.metric("Max Drawdown", fmt_pct(mdd))

        c5, c6, c7, c8 = st.columns(4)
        c5.metric("VaR 95% (daily)", fmt_pct(v95))
        c6.metric("CVaR 95% (daily)", fmt_pct(c95))
        c7.metric("Tracking Error (ann)", fmt_pct(te))
        c8.metric("Information Ratio", fmt_num(ir, 2))

        st.write("Drawdown (Wave vs Benchmark) — %")
        dd_w = drawdown_series(hist["wave_nav"])
        dd_b = drawdown_series(hist["bm_nav"])
        dd_df = pd.concat([dd_w.rename("Wave"), dd_b.rename("Benchmark")], axis=1).dropna()
        st.line_chart(dd_df * 100.0)

        st.write("Rolling 30D Alpha (%) + Rolling Vol (ann)")
        ra = rolling_alpha_from_nav(hist["wave_nav"], hist["bm_nav"], window=30)
        rv = rolling_vol(hist["wave_ret"], window=20)
        roll_df = pd.concat(
            [(ra * 100.0).rename("Rolling 30D Alpha (%)"), rv.rename("Rolling Vol (20D)")],
            axis=1,
        ).dropna()
        st.line_chart(roll_df)

        ap = alpha_persistence(ra)
        st.metric("Alpha Persistence (Rolling 30D windows)", fmt_pct(ap))

    render_definitions(
        keys=["Tracking Error (TE)", "Information Ratio (IR)", "Max Drawdown (MaxDD)"],
        title="Definitions (Risk Lab)",
    )

    st.divider()

    # Correlation
    st.markdown("### Correlation (Daily Returns)")
    rets: Dict[str, pd.Series] = {}
    for w in all_waves:
        h = compute_wave_history(w, mode=mode, days=min(days, 365))
        h = _standardize_history(h)
        if h is not None and not h.empty and "wave_ret" in h.columns:
            rets[w] = h["wave_ret"]

    if len(rets) < 2:
        st.info("Not enough waves with history to compute correlations.")
    else:
        ret_df = pd.DataFrame(rets).dropna(how="all")
        corr = ret_df.corr()
        st.dataframe(corr, use_container_width=True)

    st.divider()

    # Factor / beta
    st.markdown("### Factor Decomposition (Light)")
    st.caption("Beta vs benchmark from daily returns.")
    if hist is None or hist.empty or len(hist) < 20:
        st.info("Not enough history.")
    else:
        b = beta_ols(hist["wave_ret"], hist["bm_ret"])
        st.metric("Beta vs Benchmark", fmt_num(b, 2))

    render_definitions(keys=["Return"], title="Definitions (Factor)")

    if not demo_mode:
        st.divider()
        st.markdown("### Strategy Separation (Mode Proof)")
        st.caption("Same wave across modes — proves strategies are distinct.")
        modes_to_check = safe_mode_list()
        rows = []
        for m in modes_to_check:
            h = compute_wave_history(selected_wave, mode=m, days=min(days, 365))
            h = _standardize_history(h)
            if h is None or h.empty or len(h) < 10:
                rows.append({"Mode": m, "Rows": 0, "365D Return": np.nan, "365D Alpha": np.nan, "MaxDD": np.nan, "TE": np.nan})
                continue

            rw = ret_from_nav(h["wave_nav"], min(365, len(h)))
            rb = ret_from_nav(h["bm_nav"], min(365, len(h)))
            rows.append({
                "Mode": m,
                "Rows": int(len(h)),
                "365D Return": rw * 100.0,
                "365D Alpha": (rw - rb) * 100.0,
                "MaxDD": max_drawdown(h["wave_nav"]) * 100.0,
                "TE": tracking_error(h["wave_ret"], h["bm_ret"]) * 100.0,
            })
        dfm = pd.DataFrame(rows)
        st.dataframe(style_perf_df(dfm), use_container_width=True)

# ============================================================
# TAB 3: Benchmark Integrity (truth room)
# ============================================================
with tabs[3]:
    st.subheader("Benchmark Integrity")
    st.caption("Truth room: benchmark definition, drift detection, and difficulty context.")

    st.write(f"**Snapshot:** {bm_id} · **Drift:** {bm_drift.upper()}")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Difficulty vs SPY (proxy)", fmt_num(difficulty.get("difficulty_vs_spy"), 2))
    c2.metric("HHI (conc.)", fmt_num(difficulty.get("hhi"), 4))
    c3.metric("Entropy", fmt_num(difficulty.get("entropy"), 3))
    c4.metric("Top Weight", fmt_pct(difficulty.get("top_weight"), 2))

    st.divider()
    st.markdown("### Benchmark Mix (normalized)")
    if bm_rows is None or bm_rows.empty:
        st.info("Benchmark mix table unavailable (engine may not expose it).")
    else:
        show = bm_rows.copy()
        show["Weight %"] = show["Weight"] * 100.0
        st.dataframe(show[["Ticker", "Weight %"]], use_container_width=True)

    render_definitions(
        keys=["Benchmark Snapshot / Drift", "Difficulty vs SPY"],
        title="Definitions (Benchmark Integrity)",
    )

    if not demo_mode:
        st.divider()
        st.markdown("### Diagnostics: Benchmark Drift Guidance")
        st.write("• If drift occurs during IC review, freeze the benchmark mix for that session.")
        st.write("• Drift is a *signal* that definitions changed (engine/inputs), not necessarily an error.")
        # ============================================================
# TAB 12: IC Notes
# ============================================================
with tabs[12]:
    st.subheader("IC Notes")
    st.caption("Human-readable diligence notes and key flags (observational only).")

    if hist is None or hist.empty or len(hist) < 20:
        st.info("Not enough data for insights yet.")
    else:
        notes = build_alerts(selected_wave, mode, hist, cov, bm_drift, te, a30, mdd)
        for n in notes:
            st.markdown(f"- {n}")

    render_definitions(
        keys=["Decision Intelligence", "Coverage Score", "Benchmark Snapshot / Drift"],
        title="Definitions (IC Notes)",
    )

# ============================================================
# TAB 13: Daily Movement / Volatility
# ============================================================
with tabs[13]:
    st.subheader("Daily Movement / Volatility — Selected Wave")
    st.caption("Explains what changed, why it likely changed, and observable results (not advice).")

    ctx = build_decision_ctx(
        wave=selected_wave,
        mode=mode,
        bm_id=bm_id,
        bm_drift=bm_drift,
        cov=cov,
        vix_val=vix_val,
        regime=regime,
        r30=r30, a30=a30,
        r60=r60, a60=a60,
        r365=r365, a365=a365,
        te=te,
        ir=ir,
        mdd=mdd,
        wavescore=ws_val,
        rank=rank,
    )

    if build_daily_wave_activity is None:
        st.info("build_daily_wave_activity(ctx) not available. Check decision_engine.py import.")
    else:
        try:
            activity = build_daily_wave_activity(ctx)
        except Exception as e:
            activity = {
                "headline": "Daily Movement error",
                "what_changed": [],
                "why": [],
                "results": [],
                "checks": [str(e)],
            }

        if not isinstance(activity, dict):
            st.write(activity)
        else:
            if activity.get("headline"):
                st.write(f"**{activity.get('headline')}**")

            st.markdown("### What changed")
            for s in (activity.get("what_changed") or []):
                st.write(f"• {s}")

            st.markdown("### Why it changed")
            for s in (activity.get("why") or []):
                st.write(f"• {s}")

            st.markdown("### Results")
            for s in (activity.get("results") or []):
                st.write(f"• {s}")

            st.markdown("### Checks / Confidence")
            for s in (activity.get("checks") or []):
                st.write(f"• {s}")

            with st.expander("Context (ctx) used for this explanation"):
                st.json(ctx)

    render_definitions(
        keys=["Regime", "VIX", "Tracking Error (TE)", "Information Ratio (IR)"],
        title="Definitions (Daily Movement)",
    )

# ============================================================
# TAB 14: Decision Intelligence
# ============================================================
with tabs[14]:
    st.subheader("Decision Intelligence — Actions / Watch / Notes")
    st.caption("Operating-system style guidance: what to look at next (not advice).")

    ctx = build_decision_ctx(
        wave=selected_wave,
        mode=mode,
        bm_id=bm_id,
        bm_drift=bm_drift,
        cov=cov,
        vix_val=vix_val,
        regime=regime,
        r30=r30, a30=a30,
        r60=r60, a60=a60,
        r365=r365, a365=a365,
        te=te,
        ir=ir,
        mdd=mdd,
        wavescore=ws_val,
        rank=rank,
    )

    if generate_decisions is None:
        st.info("generate_decisions(ctx) not available. Check decision_engine.py import.")
    else:
        try:
            d = generate_decisions(ctx)
        except Exception as e:
            d = {"actions": [f"Decision engine error: {e}"], "watch": [], "notes": []}

        c1, c2, c3 = st.columns(3)
        with c1:
            st.write("### Actions")
            for x in (d.get("actions") or []):
                st.write(f"• {x}")
        with c2:
            st.write("### Watch")
            for x in (d.get("watch") or []):
                st.write(f"• {x}")
        with c3:
            st.write("### Notes")
            for x in (d.get("notes") or []):
                st.write(f"• {x}")

        with st.expander("Context (ctx) used for these decisions"):
            st.json(ctx)

    render_definitions(
        keys=["Decision Intelligence", "Alpha", "Coverage Score"],
        title="Definitions (Decision Intelligence)",
    )

# ============================================================
# Footer diagnostics (kept; organized)
# ============================================================
with st.expander("System Diagnostics (if something looks off)"):
    st.write("Engine loaded:", we is not None)
    st.write("Engine import error:", str(ENGINE_IMPORT_ERROR) if ENGINE_IMPORT_ERROR else "None")
    st.write(
        "Decision engine loaded:",
        (generate_decisions is not None) or (build_daily_wave_activity is not None),
    )
    st.write("Decision import error:", str(DECISION_IMPORT_ERROR) if DECISION_IMPORT_ERROR else "None")
    st.write(
        "Files present:",
        {
            p: os.path.exists(p)
            for p in [
                "wave_config.csv",
                "wave_weights.csv",
                "wave_history.csv",
                "list.csv",
                "waves_engine.py",
                "decision_engine.py",
            ]
        },
    )
    st.write("Selected:", {"wave": selected_wave, "mode": mode, "days": days})
    st.write("History shape:", None if hist is None else hist.shape)
    if hist is not None and not hist.empty:
        st.write("History columns:", list(hist.columns))
        st.write("History tail:", hist.tail(3))

# ============================================================
# End-of-file guardrails
# ============================================================
# Never leave trailing, unclosed blocks or strings below this line.
# If you add code, keep it inside functions or tab scopes.
# EOF — app.py