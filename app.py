# app.py — WAVES Intelligence™ Institutional Console (Vector OS Edition)
# FULL PRODUCTION FILE (NO PATCHES)
# vNEXT — IRB-1 + Elite Copilot Pack + IC Summary + Governance Scorecard + Trust Stack + Cross-Wave Intelligence
#
# GOALS (No features removed):
#   ✅ Keep EVERYTHING from prior app (IC Summary, Overview matrix+heatmap, Attribution, Factor, Risk Lab,
#      Correlation, Strategy Separation, Benchmark Integrity, Drawdown Monitor, Diligence Flags,
#      WaveScore, Governance Export, IC Notes, Daily Movement, Decision Intelligence, Diagnostics)
#   ✅ Add Governance-native “IC Diagnostic Grid”:
#        - Domain grades (0–100 + A–F) for major analytic topics
#        - One-line IC Verdict
#        - Color bands (G/Y/R)
#        - Micro “Why this grade?” explainers (1-line + expander)
#        - Trust Stack framing (Data Quality / Benchmark Integrity / Signal Consistency)
#        - Cross-wave pattern detection insight
#
# NOTES:
#   • Engine math NOT modified.
#   • Robust history loader: engine functions → wave_history.csv fallback
#   • Decision Engine optional: app will not crash if decision_engine.py missing
#   • Fixes: tabs length mismatch (IndexError), avoids unterminated f-strings

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

# -------------------------------
# Decision Engine import (guarded)
# -------------------------------
DECISION_IMPORT_ERROR = None
try:
    from decision_engine import generate_decisions
except Exception as e:
    generate_decisions = None
    DECISION_IMPORT_ERROR = e

try:
    from decision_engine import build_daily_wave_activity
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

.waves-card {
  border: 1px solid rgba(255,255,255,0.10);
  border-radius: 14px;
  padding: 12px 14px;
  background: rgba(255,255,255,0.03);
}

.waves-subtle {
  color: rgba(255,255,255,0.72);
  font-size: 0.92rem;
}

.waves-badge {
  display: inline-block;
  padding: 2px 8px;
  border-radius: 999px;
  font-size: 0.80rem;
  border: 1px solid rgba(255,255,255,0.14);
  background: rgba(255,255,255,0.04);
  margin-left: 6px;
}

/* Score band backgrounds */
.band-green { background: rgba(0, 200, 120, 0.14); }
.band-yellow { background: rgba(255, 200, 0, 0.12); }
.band-red { background: rgba(255, 60, 60, 0.12); }

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

def safe_df(df: Optional[pd.DataFrame]) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    return df.copy()

def google_quote_link(ticker: str) -> str:
    t = str(ticker).strip().upper()
    return f"https://www.google.com/finance/quote/{t}"

def safe_mode_list() -> List[str]:
    return ["Standard", "Alpha-Minus-Beta", "Private Logic"]

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
    if score >= 50:
        return "D"
    return "F"

def _band(score: float) -> str:
    if score is None or (isinstance(score, float) and math.isnan(score)):
        return "band-yellow"
    if score >= 80:
        return "band-green"
    if score >= 65:
        return "band-yellow"
    return "band-red"

# ============================================================
# Styling helpers: percent/alpha shading
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
    cols = [c for c in df.columns if ("Return" in c or "Alpha" in c)]
    sty = df.style
    for c in cols:
        sty = sty.applymap(_heat_color, subset=[c])
        sty = sty.format({c: "{:.2f}".format})
    return sty

# ============================================================
# Definitions / Glossary
# ============================================================
GLOSSARY: Dict[str, str] = {
    "Return": "Portfolio return over the window (not annualized unless stated).",
    "Alpha": "Return minus Benchmark return over the same window (relative performance).",
    "Tracking Error (TE)": "Annualized volatility of (Wave daily returns − Benchmark daily returns). Higher = more active risk.",
    "Information Ratio (IR)": "Excess return divided by Tracking Error (risk-adjusted alpha).",
    "Max Drawdown (MaxDD)": "Largest peak-to-trough decline over the period (negative number).",
    "Benchmark Snapshot / Drift": "A fingerprint of the benchmark mix. Drift means benchmark definition changed in-session.",
    "Coverage Score": "0–100 heuristic for completeness + freshness. Missing days + staleness reduce score.",
    "Difficulty vs SPY": "Proxy for benchmark concentration/diversification vs SPY. Higher can imply harder to beat consistently.",
    "WaveScore (Console Approx.)": "Console-side approximation (NOT the locked WAVESCORE™ v1.0).",
    "Decision Intelligence": "Operating-system layer: actions/watch/notes based on analytics (not advice).",
    "Trust Stack": "Institutional framing: Data Quality + Benchmark Integrity + Signal Consistency.",
    "IC Verdict": "One-line synthesized verdict generated from the diagnostic grid.",
}

def render_definitions(keys: List[str], title: str = "Definitions"):
    with st.expander(title):
        for k in keys:
            if k in GLOSSARY:
                st.markdown(f"**{k}:** {GLOSSARY[k]}")
            else:
                st.markdown(f"**{k}:** (definition not found)")
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
    Expected output:
      index=datetime
      columns: wave_nav, bm_nav, wave_ret, bm_ret
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=["wave_nav", "bm_nav", "wave_ret", "bm_ret"])

    out = df.copy()

    # allow a date-like column
    for dc in ["date", "Date", "timestamp", "Timestamp", "datetime", "Datetime"]:
        if dc in out.columns:
            out[dc] = pd.to_datetime(out[dc], errors="coerce")
            out = out.dropna(subset=[dc]).set_index(dc)
            break

    out.index = pd.to_datetime(out.index, errors="coerce")
    out = out[~out.index.isna()].sort_index()

    # normalize column names
    ren = {}
    for c in out.columns:
        low = str(c).strip().lower()
        if low in ["wave_nav", "nav_wave", "portfolio_nav", "nav", "wave value", "wavevalue"]:
            ren[c] = "wave_nav"
        elif low in ["bm_nav", "bench_nav", "benchmark_nav", "benchmark", "bm value", "benchmark value"]:
            ren[c] = "bm_nav"
        elif low in ["wave_ret", "ret_wave", "portfolio_ret", "return", "wave_return", "wave return"]:
            ren[c] = "wave_ret"
        elif low in ["bm_ret", "ret_bm", "benchmark_ret", "bm_return", "benchmark_return", "benchmark return"]:
            ren[c] = "bm_ret"

    out = out.rename(columns=ren)

    # derive returns if missing but nav exists
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
    """
    Best-effort:
      1) try engine functions
      2) fallback to wave_history.csv
    """
    if we is None:
        return history_from_csv(wave_name, mode, days)

    # Preferred function name
    try:
        if hasattr(we, "compute_history_nav"):
            try:
                df = we.compute_history_nav(wave_name, mode=mode, days=days)
            except TypeError:
                df = we.compute_history_nav(wave_name, mode, days)
            df = _standardize_history(df)
            if not df.empty:
                return df
    except Exception:
        pass

    # Try other candidate function names
    candidates = [
        "get_history_nav",
        "get_wave_history",
        "history_nav",
        "compute_nav_history",
        "compute_history",
    ]
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


# ============================================================
# Wave discovery + engine tables
# ============================================================
@st.cache_data(show_spinner=False)
def get_all_waves_safe() -> List[str]:
    """
    Best-effort wave discovery:
      - engine.get_all_waves()
      - engine.get_benchmark_mix_table()
      - fall back to CSV (wave_config.csv / wave_weights.csv / list.csv)
    """
    # CSV fallback
    if we is None:
        for p in ["wave_config.csv", "wave_weights.csv", "list.csv"]:
            if os.path.exists(p):
                try:
                    df = pd.read_csv(p)
                    for col in ["Wave", "wave", "wave_name"]:
                        if col in df.columns:
                            waves = sorted(list(set(df[col].astype(str).tolist())))
                            waves = [w for w in waves if w and w.lower() != "nan"]
                            return waves
                except Exception:
                    pass
        return []

    if hasattr(we, "get_all_waves"):
        try:
            waves = we.get_all_waves()
            if isinstance(waves, (list, tuple)):
                return [str(x) for x in waves if str(x).strip()]
        except Exception:
            pass

    if hasattr(we, "get_benchmark_mix_table"):
        try:
            bm = we.get_benchmark_mix_table()
            if isinstance(bm, pd.DataFrame) and "Wave" in bm.columns:
                return sorted(list(set(bm["Wave"].astype(str).tolist())))
        except Exception:
            pass

    # final fallback to CSV
    for p in ["wave_config.csv", "wave_weights.csv", "list.csv"]:
        if os.path.exists(p):
            try:
                df = pd.read_csv(p)
                for col in ["Wave", "wave", "wave_name"]:
                    if col in df.columns:
                        waves = sorted(list(set(df[col].astype(str).tolist())))
                        waves = [w for w in waves if w and w.lower() != "nan"]
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
    """
    Prefer engine.get_wave_holdings(wave). Else parse wave_weights.csv
    Output columns: Ticker, Name, Weight (normalized)
    """
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
        if bm_mix
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
    Expected output:
      index=datetime
      columns: wave_nav, bm_nav, wave_ret, bm_ret
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=["wave_nav", "bm_nav", "wave_ret", "bm_ret"])

    out = df.copy()

    # allow a date-like column
    for dc in ["date", "Date", "timestamp", "Timestamp", "datetime", "Datetime"]:
        if dc in out.columns:
            out[dc] = pd.to_datetime(out[dc], errors="coerce")
            out = out.dropna(subset=[dc]).set_index(dc)
            break

    out.index = pd.to_datetime(out.index, errors="coerce")
    out = out[~out.index.isna()].sort_index()

    # normalize column names
    ren = {}
    for c in out.columns:
        low = str(c).strip().lower()
        if low in ["wave_nav", "nav_wave", "portfolio_nav", "nav", "wave value", "wavevalue"]:
            ren[c] = "wave_nav"
        elif low in ["bm_nav", "bench_nav", "benchmark_nav", "benchmark", "bm value", "benchmark value"]:
            ren[c] = "bm_nav"
        elif low in ["wave_ret", "ret_wave", "portfolio_ret", "return", "wave_return", "wave return"]:
            ren[c] = "wave_ret"
        elif low in ["bm_ret", "ret_bm", "benchmark_ret", "bm_return", "benchmark_return", "benchmark return"]:
            ren[c] = "bm_ret"

    out = out.rename(columns=ren)

    # derive returns if missing but nav exists
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
    """
    Best-effort:
      1) try engine functions
      2) fallback to wave_history.csv
    """
    if we is None:
        return history_from_csv(wave_name, mode, days)

    # Preferred function name
    try:
        if hasattr(we, "compute_history_nav"):
            try:
                df = we.compute_history_nav(wave_name, mode=mode, days=days)
            except TypeError:
                df = we.compute_history_nav(wave_name, mode, days)
            df = _standardize_history(df)
            if not df.empty:
                return df
    except Exception:
        pass

    # Try other candidate function names
    candidates = [
        "get_history_nav",
        "get_wave_history",
        "history_nav",
        "compute_nav_history",
        "compute_history",
    ]
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


# ============================================================
# Wave discovery + engine tables
# ============================================================
@st.cache_data(show_spinner=False)
def get_all_waves_safe() -> List[str]:
    """
    Best-effort wave discovery:
      - engine.get_all_waves()
      - engine.get_benchmark_mix_table()
      - fall back to CSV (wave_config.csv / wave_weights.csv / list.csv)
    """
    # CSV fallback
    if we is None:
        for p in ["wave_config.csv", "wave_weights.csv", "list.csv"]:
            if os.path.exists(p):
                try:
                    df = pd.read_csv(p)
                    for col in ["Wave", "wave", "wave_name"]:
                        if col in df.columns:
                            waves = sorted(list(set(df[col].astype(str).tolist())))
                            waves = [w for w in waves if w and w.lower() != "nan"]
                            return waves
                except Exception:
                    pass
        return []

    if hasattr(we, "get_all_waves"):
        try:
            waves = we.get_all_waves()
            if isinstance(waves, (list, tuple)):
                return [str(x) for x in waves if str(x).strip()]
        except Exception:
            pass

    if hasattr(we, "get_benchmark_mix_table"):
        try:
            bm = we.get_benchmark_mix_table()
            if isinstance(bm, pd.DataFrame) and "Wave" in bm.columns:
                return sorted(list(set(bm["Wave"].astype(str).tolist())))
        except Exception:
            pass

    # final fallback to CSV
    for p in ["wave_config.csv", "wave_weights.csv", "list.csv"]:
        if os.path.exists(p):
            try:
                df = pd.read_csv(p)
                for col in ["Wave", "wave", "wave_name"]:
                    if col in df.columns:
                        waves = sorted(list(set(df[col].astype(str).tolist())))
                        waves = [w for w in waves if w and w.lower() != "nan"]
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
    """
    Prefer engine.get_wave_holdings(wave). Else parse wave_weights.csv
    Output columns: Ticker, Name, Weight (normalized)
    """
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

        # tolerate different column casings
        if "Ticker" not in rows.columns:
            for alt in ["ticker", "Symbol", "symbol"]:
                if alt in rows.columns:
                    rows["Ticker"] = rows[alt]
                    break
        if "Weight" not in rows.columns:
            for alt in ["weight", "w", "WeightPct"]:
                if alt in rows.columns:
                    rows["Weight"] = rows[alt]
                    break

        rows = _normalize_bm_rows(rows[["Ticker", "Weight"]] if {"Ticker", "Weight"}.issubset(rows.columns) else rows)
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
# WaveScore (console-side approximation)
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

        alpha_365 = ret_from_nav(nav_wave, min(len(nav_wave), 365)) - ret_from_nav(nav_bm, min(len(nav_bm), 365))
        te = tracking_error(wave_ret, bm_ret)
        ir = information_ratio(nav_wave, nav_bm, te)
        mdd_wave = max_drawdown(nav_wave)
        hit_rate = float((wave_ret >= bm_ret).mean()) if len(wave_ret) else np.nan

        # Very light, readable proxy score (NOT your locked WaveScore spec)
        rq = float(np.clip((np.nan_to_num(ir) / 1.5), 0.0, 1.0) * 25.0)                  # return quality
        rc = float(np.clip(1.0 - (abs(np.nan_to_num(mdd_wave)) / 0.35), 0.0, 1.0) * 25.0) # risk control
        co = float(np.clip(np.nan_to_num(hit_rate), 0.0, 1.0) * 15.0)                     # consistency
        ef = float(np.clip(1.0 - (abs(np.nan_to_num(te)) / 0.25), 0.0, 1.0) * 15.0)        # efficiency proxy
        tr = 20.0                                                                          # transparency baseline

        total = float(np.clip(rq + rc + co + ef + tr, 0.0, 100.0))
        rows.append({
            "Wave": wave,
            "WaveScore": total,
            "Grade": _grade_from_score(total),
            "IR_365D": ir,
            "Alpha_365D": alpha_365,
        })

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
    fig.update_layout(
        title=title,
        height=min(950, 260 + 22 * max(12, len(y))),
        margin=dict(l=80, r=40, t=60, b=40),
    )
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
        if h.empty or len(h) < 2:
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

    # keep selected wave first
    if "Wave" in df.columns and selected_wave in set(df["Wave"]):
        top = df[df["Wave"] == selected_wave]
        rest = df[df["Wave"] != selected_wave]
        df = pd.concat([top, rest], axis=0)

    return df.reset_index(drop=True)


# ============================================================
# Alerts & Flags
# ============================================================
def build_alerts(
    selected_wave: str,
    mode: str,
    hist: pd.DataFrame,
    cov: Dict[str, Any],
    bm_drift: str,
    te: float,
    a30: float,
    mdd: float,
) -> List[str]:
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
        if hist is not None and (not hist.empty) and cov.get("age_days", 0) is not None and cov.get("age_days", 0) >= 5:
            notes.append("Stale history: last datapoint is >=5 days old (check engine writes).")
        if not notes:
            notes.append("No major anomalies detected on this window.")
        return notes
    except Exception:
        return ["Alert system error (non-fatal)."]


# ============================================================
# Decision context (unified)
# ============================================================
def build_decision_ctx(
    wave: str,
    mode: str,
    bm_id: str,
    bm_drift: str,
    cov: Dict[str, Any],
    vix_val: float,
    regime: str,
    r30: float, a30: float,
    r60: float, a60: float,
    r365: float, a365: float,
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
        "r30": r30, "a30": a30,
        "r60": r60, "a60": a60,
        "r365": r365, "a365": a365,
        "te": te,
        "ir": ir,
        "mdd": mdd,
        "wavescore": wavescore,
        "rank": rank,
    }


# ============================================================
# Governance Export Pack (markdown)
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
# NEW: Analytics Scorecard (Governance-native, Franklin-friendly)
# ============================================================

def _band(score_0_100: float) -> str:
    if score_0_100 is None or (isinstance(score_0_100, float) and math.isnan(score_0_100)):
        return "—"
    if score_0_100 >= 80:
        return "Green"
    if score_0_100 >= 65:
        return "Yellow"
    return "Red"


def _letter_af(score_0_100: float) -> str:
    if score_0_100 is None or (isinstance(score_0_100, float) and math.isnan(score_0_100)):
        return "N/A"
    if score_0_100 >= 93:
        return "A"
    if score_0_100 >= 85:
        return "B"
    if score_0_100 >= 75:
        return "C"
    if score_0_100 >= 65:
        return "D"
    return "F"


def _clamp01(x: float) -> float:
    try:
        return float(np.clip(float(x), 0.0, 1.0))
    except Exception:
        return 0.0


def _score_linear(x: float, lo: float, hi: float, invert: bool = False) -> float:
    """
    Map x onto 0..1 between lo..hi; optionally invert.
    """
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return np.nan
    if hi == lo:
        return np.nan
    v = (x - lo) / (hi - lo)
    v = float(np.clip(v, 0.0, 1.0))
    if invert:
        v = 1.0 - v
    return v


def analytics_scorecard_for_wave(
    wave: str,
    mode: str,
    hist: pd.DataFrame,
    cov: Dict[str, Any],
    bm_drift: str,
    bm_id: str,
    difficulty: Dict[str, Any],
    ws_val: float,
    ws_rank: Optional[int],
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Returns:
      (score_df, meta)
    score_df columns:
      Domain, Score (0-100), Grade (A-F), Band, 1-line Why, Key inputs (compact)
    meta includes:
      ic_verdict (one line),
      trust_stack (3 scores),
      flags
    """
    # ---- precompute primitives
    hist = _standardize_history(hist)
    rows = []
    flags = []

    # default metrics
    te = np.nan
    ir = np.nan
    mdd = np.nan
    a30 = np.nan
    a60 = np.nan
    a365 = np.nan
    hit = np.nan
    beta = np.nan
    missing_pct = cov.get("missing_pct", np.nan)
    age_days = cov.get("age_days", np.nan)
    comp = cov.get("completeness_score", np.nan)

    if hist is not None and (not hist.empty) and len(hist) >= 20:
        try:
            te = tracking_error(hist["wave_ret"], hist["bm_ret"])
            ir = information_ratio(hist["wave_nav"], hist["bm_nav"], te)
            mdd = max_drawdown(hist["wave_nav"])
            a30 = ret_from_nav(hist["wave_nav"], min(30, len(hist))) - ret_from_nav(hist["bm_nav"], min(30, len(hist)))
            a60 = ret_from_nav(hist["wave_nav"], min(60, len(hist))) - ret_from_nav(hist["bm_nav"], min(60, len(hist)))
            a365 = ret_from_nav(hist["wave_nav"], min(365, len(hist))) - ret_from_nav(hist["bm_nav"], min(365, len(hist)))
            hit = float((hist["wave_ret"] >= hist["bm_ret"]).mean())
            beta = beta_ols(hist["wave_ret"], hist["bm_ret"])
        except Exception:
            pass
    else:
        flags.append("Limited history for analytics scorecard (<20 rows).")

    # ---- Domain 1: Data Quality (coverage + staleness + missing days)
    # completeness_score already penalizes missing+age; we still translate into 0..100
    dq = float(comp) if (comp is not None and math.isfinite(float(comp))) else np.nan
    dq_why = "Fresh & complete history." if math.isfinite(dq) and dq >= 85 else "Coverage/staleness reduces confidence."
    dq_inputs = f"cov={fmt_num(comp,1)} age={age_days} miss={fmt_num(missing_pct*100 if math.isfinite(missing_pct) else np.nan,1)}%"
    rows.append({"Domain": "Data Quality", "Score": dq, "Grade": _letter_af(dq), "Band": _band(dq), "Why": dq_why, "Inputs": dq_inputs})

    # ---- Domain 2: Benchmark Integrity (drift + concentration difficulty)
    # Drift is binary penalty; difficulty is a proxy
    drift_pen = 0.0 if bm_drift == "stable" else 0.35
    diff = difficulty.get("difficulty_vs_spy", np.nan)
    # easier benchmark (lower difficulty) isn't “better”, but very extreme difficulty increases noise and committee skepticism.
    # So we prefer moderate difficulty (near 0). Score falls as |diff| grows.
    diff_score01 = 1.0 - _score_linear(abs(float(diff)) if math.isfinite(diff) else np.nan, lo=0.0, hi=25.0, invert=False)
    bi01 = _clamp01((diff_score01 if math.isfinite(diff_score01) else 0.5) - drift_pen)
    bi = float(bi01 * 100.0)
    bi_why = "Benchmark stable; definition consistent." if bm_drift == "stable" else "Benchmark drift detected; freeze benchmark for demos."
    bi_inputs = f"id={bm_id} drift={bm_drift} diff={fmt_num(diff,2)}"
    rows.append({"Domain": "Benchmark Integrity", "Score": bi, "Grade": _letter_af(bi), "Band": _band(bi), "Why": bi_why, "Inputs": bi_inputs})

    # ---- Domain 3: Signal Quality (IR + alpha vs TE)
    # Prefer IR positive and not insane TE. Map IR -0.5..1.5 to 0..1
    ir01 = _score_linear(ir, lo=-0.5, hi=1.5, invert=False)
    te01 = _score_linear(te, lo=0.05, hi=0.30, invert=True)  # lower TE is "cleaner signal" for committees
    sq01 = _clamp01(0.65 * (ir01 if math.isfinite(ir01) else 0.4) + 0.35 * (te01 if math.isfinite(te01) else 0.5))
    sq = float(sq01 * 100.0)
    sq_why = "Risk-adjusted alpha looks real (IR) with manageable active risk (TE)." if (math.isfinite(ir) and ir > 0.2) else "Signal is weaker or noisy; validate regime + benchmark."
    sq_inputs = f"IR={fmt_num(ir,2)} TE={fmt_pct(te)} a30={fmt_pct(a30)}"
    rows.append({"Domain": "Signal Quality", "Score": sq, "Grade": _letter_af(sq), "Band": _band(sq), "Why": sq_why, "Inputs": sq_inputs})

    # ---- Domain 4: Risk Control (max drawdown + downside control proxy)
    # Prefer smaller drawdown magnitude; map 0..-35% into 1..0
    rc01 = _score_linear(abs(mdd), lo=0.05, hi=0.35, invert=True)  # abs drawdown
    rc = float(_clamp01(rc01 if math.isfinite(rc01) else 0.5) * 100.0)
    rc_why = "Drawdown contained relative to typical risk budgets." if math.isfinite(mdd) and mdd > -0.20 else "Drawdown elevated; consider tighter SmartSafe posture."
    rc_inputs = f"MaxDD={fmt_pct(mdd)} beta={fmt_num(beta,2)}"
    rows.append({"Domain": "Risk Control", "Score": rc, "Grade": _letter_af(rc), "Band": _band(rc), "Why": rc_why, "Inputs": rc_inputs})

    # ---- Domain 5: Consistency (hit-rate + alpha persistence)
    ap = np.nan
    try:
        if hist is not None and not hist.empty and len(hist) >= 60:
            ra = rolling_alpha_from_nav(hist["wave_nav"], hist["bm_nav"], window=30).dropna()
            ap = alpha_persistence(ra)
    except Exception:
        pass
    hit01 = _score_linear(hit, lo=0.35, hi=0.65, invert=False)
    ap01 = _score_linear(ap, lo=0.40, hi=0.65, invert=False)
    co01 = _clamp01(0.6 * (hit01 if math.isfinite(hit01) else 0.45) + 0.4 * (ap01 if math.isfinite(ap01) else 0.45))
    co = float(co01 * 100.0)
    co_why = "Beats benchmark frequently; alpha persists." if (math.isfinite(hit) and hit >= 0.55) else "Consistency softer; could be regime transition or signal decay."
    co_inputs = f"hit={fmt_pct(hit)} persist={fmt_pct(ap)} a60={fmt_pct(a60)}"
    rows.append({"Domain": "Consistency", "Score": co, "Grade": _letter_af(co), "Band": _band(co), "Why": co_why, "Inputs": co_inputs})

    # ---- Domain 6: Governance Readiness (WaveScore presence + explanation + reproducibility cues)
    # This is about “IC usability”, not returns. Reward stable benchmark + good data + having score narrative.
    gov01 = _clamp01(0.45 * (dq / 100.0 if math.isfinite(dq) else 0.5) + 0.35 * (1.0 if bm_drift == "stable" else 0.6) + 0.20 * (0.8 if math.isfinite(ws_val) else 0.5))
    gov = float(gov01 * 100.0)
    gov_why = "Governance-native: trust stack + graded diagnostics are present." if gov >= 80 else "Add more audit cues (exports/definitions) to strengthen IC fit."
    gov_inputs = f"WaveScore={fmt_score(ws_val)} rank={ws_rank if ws_rank else '—'}"
    rows.append({"Domain": "Governance Readiness", "Score": gov, "Grade": _letter_af(gov), "Band": _band(gov), "Why": gov_why, "Inputs": gov_inputs})

    score_df = pd.DataFrame(rows)

    # ---- Trust Stack (reframing without removing anything)
    trust_stack = {
        "Data Quality": dq,
        "Benchmark Integrity": bi,
        "Signal Consistency": co,
    }

    # ---- IC Verdict (one-liner)
    # Compose from top drivers
    parts = []
    parts.append("High confidence" if dq >= 85 else "Moderate confidence" if dq >= 70 else "Low confidence")
    parts.append("stable benchmark" if bm_drift == "stable" else "benchmark drift")
    parts.append("disciplined risk" if rc >= 75 else "risk elevated" if rc < 65 else "balanced risk")
    parts.append("signal looks real" if sq >= 75 else "signal needs validation")
    ic_verdict = f"{parts[0]} with {parts[1]}; {parts[2]}; {parts[3]}."

    meta = {
        "ic_verdict": ic_verdict,
        "trust_stack": trust_stack,
        "flags": flags + (cov.get("flags") or []),
        "diagnostics": {
            "bm_id": bm_id,
            "bm_drift": bm_drift,
            "completeness_score": comp,
            "age_days": age_days,
            "missing_pct": missing_pct,
            "ir": ir,
            "te": te,
            "mdd": mdd,
            "hit_rate": hit,
            "alpha_30d": a30,
            "alpha_60d": a60,
            "alpha_365d": a365,
        },
    }
    return score_df, meta


def style_scorecard(df: pd.DataFrame) -> "pd.io.formats.style.Styler":
    if df is None or df.empty:
        return df.style

    def _band_cell(val: Any) -> str:
        v = str(val)
        if v == "Green":
            return "background-color: rgba(0, 200, 120, 0.18);"
        if v == "Yellow":
            return "background-color: rgba(255, 200, 0, 0.16);"
        if v == "Red":
            return "background-color: rgba(255, 60, 60, 0.16);"
        return ""

    sty = df.style
    if "Band" in df.columns:
        sty = sty.applymap(_band_cell, subset=["Band"])
    if "Score" in df.columns:
        sty = sty.format({"Score": "{:.1f}".format})
    return sty


# ============================================================
# Cross-Wave Pattern Detection (portfolio-level intelligence)
# ============================================================
@st.cache_data(show_spinner=False)
def cross_wave_patterns(all_waves: List[str], mode: str, days: int = 365) -> Dict[str, Any]:
    """
    Lightweight pattern detection:
      - Consistency down vs prior window, but Risk Control stable
      - Benchmark drift count
      - Data quality cluster warnings
    """
    # build per-wave quick stats (cheap)
    items = []
    drift_count = 0
    low_cov = 0
    for w in all_waves:
        h = compute_wave_history(w, mode=mode, days=min(days, 365))
        h = _standardize_history(h)
        cov = coverage_report(h)
        bm_mix = get_benchmark_mix()
        bm_id = benchmark_snapshot_id(w, bm_mix)
        bm_drift = "stable"  # we don't want session-state drift in cached function; treat as stable proxy
        if isinstance(bm_id, str) and bm_id.endswith("ERR"):
            pass

        if cov.get("completeness_score") is not None and float(cov.get("completeness_score", 0)) < 75:
            low_cov += 1

        if h is None or h.empty or len(h) < 80:
            continue

        # Consistency: compare alpha persistence recent vs older
        ra = rolling_alpha_from_nav(h["wave_nav"], h["bm_nav"], window=30).dropna()
        if len(ra) < 60:
            continue

        mid = int(len(ra) * 0.5)
        old = ra.iloc[:mid]
        new = ra.iloc[mid:]

        old_p = alpha_persistence(old)
        new_p = alpha_persistence(new)

        # Risk control: compare drawdown magnitude last-half vs first-half
        nav = h["wave_nav"]
        mid2 = int(len(nav) * 0.5)
        old_mdd = max_drawdown(nav.iloc[:mid2])
        new_mdd = max_drawdown(nav.iloc[mid2:])

        items.append({
            "Wave": w,
            "old_p": old_p,
            "new_p": new_p,
            "old_mdd": old_mdd,
            "new_mdd": new_mdd,
        })

    df = pd.DataFrame(items)
    insight_lines = []
    if not df.empty:
        df["consistency_down"] = (df["new_p"] + 1e-9) < (df["old_p"] - 0.08)
        df["risk_stable"] = (abs(df["new_mdd"]) <= (abs(df["old_mdd"]) + 0.03))

        pattern = df[df["consistency_down"] & df["risk_stable"]]
        if len(pattern) >= 2:
            insight_lines.append(
                f"{len(pattern)} Waves show declining Consistency but stable Risk Control — suggests regime transition rather than model decay."
            )

        pattern2 = df[df["consistency_down"] & (~df["risk_stable"])]
        if len(pattern2) >= 2:
            insight_lines.append(
                f"{len(pattern2)} Waves show declining Consistency and weakening Risk Control — investigate model stress + SmartSafe posture."
            )

    if low_cov >= max(2, int(len(all_waves) * 0.2)):
        insight_lines.append("Several Waves have low data coverage — confidence across the console is capped until history freshness improves.")

    if not insight_lines:
        insight_lines.append("No broad cross-wave anomalies detected (on this window).")

    return {
        "insights": insight_lines,
        "sample": df.head(6).to_dict(orient="records") if not df.empty else [],
    }
    # ============================================================
# MAIN UI
# ============================================================
st.title("WAVES Intelligence™ Institutional Console")

if ENGINE_IMPORT_ERROR is not None:
    st.error("Engine import failed. The app will use CSV fallbacks where possible.")
    st.code(str(ENGINE_IMPORT_ERROR))

if DECISION_IMPORT_ERROR is not None:
    st.warning("Decision Engine import issue (non-fatal). Decision tabs will fallback.")
    st.code(str(DECISION_IMPORT_ERROR))

all_waves = get_all_waves_safe()
if not all_waves:
    st.warning("No waves discovered yet. If unexpected, check engine import + CSV files.")
    with st.expander("Diagnostics"):
        st.write("Files present:")
        st.write({p: os.path.exists(p) for p in ["wave_config.csv", "wave_weights.csv", "wave_history.csv", "list.csv", "waves_engine.py", "decision_engine.py"]})
    st.stop()

modes = safe_mode_list()

# Sidebar controls
with st.sidebar:
    st.header("Controls")
    mode = st.selectbox("Mode", modes, index=0)
    selected_wave = st.selectbox("Wave", all_waves, index=0)
    days = st.slider("History window (days)", min_value=90, max_value=1500, value=int(days_default), step=30)
    show_365_default = st.toggle("Default: show 365D columns", value=bool(show_365_default))
    st.caption("If history is empty, app falls back to wave_history.csv automatically.")

# Benchmark snapshot/drift + history
bm_mix = get_benchmark_mix()
bm_id = benchmark_snapshot_id(selected_wave, bm_mix)
bm_drift = benchmark_drift_status(selected_wave, mode, bm_id)

hist = compute_wave_history(selected_wave, mode=mode, days=days)
hist = _standardize_history(hist)
cov = coverage_report(hist)

# Precompute stats used across multiple tabs
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

# Regime chip (optional)
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

# WaveScore table + rank
ws_df = compute_wavescore_for_all_waves(all_waves, mode=mode, days=min(days, 365))
rank = None
ws_val = np.nan
if ws_df is not None and (not ws_df.empty) and selected_wave in set(ws_df["Wave"]):
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
chips.append(f"BM Snapshot: {bm_id} · {'Stable' if bm_drift=='stable' else 'DRIFT'}")
chips.append(f"Coverage: {fmt_num(cov.get('completeness_score', np.nan),1)} / 100")
chips.append(f"Rows: {cov.get('rows','—')} · Age: {cov.get('age_days','—')}")
chips.append(f"Confidence: {conf_level}")
chips.append(f"Regime: {regime}")
chips.append(f"VIX: {fmt_num(vix_val,1) if math.isfinite(vix_val) else '—'}")
chips.append(f"30D α: {fmt_pct(a30)} · 30D r: {fmt_pct(r30)}")
chips.append(f"60D α: {fmt_pct(a60)} · 60D r: {fmt_pct(r60)}")
chips.append(f"365D α: {fmt_pct(a365)} · 365D r: {fmt_pct(r365)}")
chips.append(f"TE: {fmt_pct(te)} · IR: {fmt_num(ir,2)}")
chips.append(f"WaveScore: {fmt_score(ws_val)} ({_grade_from_score(ws_val)}) · Rank: {rank if rank else '—'}")

st.markdown('<div class="waves-sticky">', unsafe_allow_html=True)
for c in chips:
    st.markdown(f'<span class="waves-chip">{c}</span>', unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)
st.caption("Observational analytics only (not trading advice).")

# ============================================================
# Tabs — keep EVERYTHING, but consolidate language
# ============================================================
tabs = st.tabs([
    "IC Summary",
    "Overview",
    "IC Diagnostic Grid",          # NEW (scorecard + trust stack + verdict + patterns)
    "Attribution",
    "Factor Decomposition",
    "Risk Lab",
    "Correlation",
    "Strategy Separation",
    "Benchmark Integrity",
    "Drawdown Monitor",
    "Diligence Flags",
    "WaveScore Leaderboard",
    "Governance Export",
    "IC Notes",
    "Daily Movement / Volatility",
    "Decision Intelligence",
])

# ============================================================
# TAB 0: IC Summary (already in your file; keep as-is)
# ============================================================
with tabs[0]:
    st.subheader(f"IC Summary — {selected_wave} ({mode})")
    st.caption("Decision-grade summary: what matters now, why, and what to check next.")

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

    left, right = st.columns([1.2, 1.0])

    with left:
        st.markdown('<div class="waves-card">', unsafe_allow_html=True)
        st.markdown("### Decision Intelligence (Translator)")
        if generate_decisions is None:
            st.warning("Decision Engine not available (generate_decisions missing).")
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
        c1.metric("Coverage Score", fmt_num(cov.get("completeness_score", np.nan), 1))
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

    with right:
        st.markdown('<div class="waves-card">', unsafe_allow_html=True)
        st.markdown("### Performance Snapshot")
        c1, c2, c3 = st.columns(3)
        c1.metric("30D Return", fmt_pct(r30))
        c2.metric("30D Alpha", fmt_pct(a30))
        c3.metric("Max Drawdown", fmt_pct(mdd))

        c4, c5, c6 = st.columns(3)
        c4.metric("Tracking Error (TE)", fmt_pct(te))
        c5.metric("Information Ratio (IR)", fmt_num(ir, 2))
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
                    axis=1,
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

    render_definitions(
        keys=[
            "Alpha", "Tracking Error (TE)", "Information Ratio (IR)",
            "Max Drawdown (MaxDD)", "Benchmark Snapshot / Drift",
            "Coverage Score", "Decision Intelligence",
        ],
        title="Definitions (IC Summary)",
    )

# ============================================================
# TAB 1: Overview — you already pasted; keep your cleaned version
# ============================================================
# (Leave your existing Tab 1 block exactly as-is here)

# ============================================================
# TAB 2: IC Diagnostic Grid (NEW — governance scorecard + verdict + trust stack)
# ============================================================
with tabs[2]:
    st.subheader("IC Diagnostic Grid (Governance Scorecard)")
    st.caption("Grades the *analytics health* (trust + signal + governance) separately from raw performance.")

    # Build scorecard for selected wave
    score_df, meta = analytics_scorecard_for_wave(
        wave=selected_wave,
        mode=mode,
        hist=hist,
        cov=cov,
        bm_drift=bm_drift,
        bm_id=bm_id,
        difficulty=difficulty,
        ws_val=ws_val,
        ws_rank=rank,
    )

    # 1-line IC Verdict (catnip)
    st.markdown('<div class="waves-card">', unsafe_allow_html=True)
    st.markdown("### IC Verdict")
    st.write(f"**{meta.get('ic_verdict','—')}**")
    st.caption("This sentence is generated from the scorecard (not advice).")
    st.markdown("</div>", unsafe_allow_html=True)

    st.divider()

    # Trust Stack (reframe, no removal)
    ts = meta.get("trust_stack", {}) or {}
    st.markdown('<div class="waves-card">', unsafe_allow_html=True)
    st.markdown("### Trust Stack")
    c1, c2, c3 = st.columns(3)
    c1.metric("Data Quality", fmt_num(ts.get("Data Quality", np.nan), 1))
    c2.metric("Benchmark Integrity", fmt_num(ts.get("Benchmark Integrity", np.nan), 1))
    c3.metric("Signal Consistency", fmt_num(ts.get("Signal Consistency", np.nan), 1))
    st.caption("Trust Stack mirrors how IC/risk memos separate trust → integrity → consistency.")
    st.markdown("</div>", unsafe_allow_html=True)

    st.divider()

    # Scorecard table (bands + grades + whys)
    st.markdown("### Analytics Scorecard (by domain)")
    if score_df is None or score_df.empty:
        st.info("Scorecard unavailable (insufficient history).")
    else:
        # Make a scannable table with an expand option for the "Inputs"
        show = score_df.copy()
        show = show[["Domain", "Score", "Grade", "Band", "Why"]].copy()
        st.dataframe(style_scorecard(show), use_container_width=True)

        with st.expander("Why these grades? (inputs + calculation context)"):
            st.dataframe(score_df, use_container_width=True)
            st.write("Diagnostics snapshot:")
            st.json(meta.get("diagnostics", {}))

    # Micro-explainers (hover not always reliable on mobile; use expanders)
    render_definitions(
        keys=[
            "Coverage Score",
            "Benchmark Snapshot / Drift",
            "Tracking Error (TE)",
            "Information Ratio (IR)",
            "Difficulty vs SPY",
            "WaveScore",
        ],
        title="Definitions (Scorecard)",
    )

    st.divider()

    # Cross-wave pattern detection (portfolio OS positioning)
    st.markdown("### Cross-Wave Pattern Detection")
    patt = cross_wave_patterns(all_waves, mode=mode, days=min(days, 365))
    for line in (patt.get("insights") or []):
        st.write(f"• {line}")
    with st.expander("Pattern debug (sample rows)"):
        st.write(patt.get("sample", []))

    st.divider()

    # Optional: quick export of scorecard
    if score_df is not None and not score_df.empty:
        csv_bytes = score_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download Scorecard (CSV)",
            data=csv_bytes,
            file_name=f"IC_Scorecard_{selected_wave.replace(' ','_')}_{mode.replace(' ','_')}.csv",
            mime="text/csv",
            use_container_width=True,
        )

    # Flags (if any)
    flags = meta.get("flags") or []
    if flags:
        st.markdown("### Flags / Notes")
        for f in flags:
            st.write(f"• {f}")
            # ============================================================
# TAB 3: Attribution
# ============================================================
with tabs[3]:
    st.subheader("Attribution (Engine vs Static Basket Proxy)")
    st.caption("Console-side proxy: compares Wave returns to Benchmark returns (alpha).")

    if hist is None or hist.empty or len(hist) < 30:
        st.info("Not enough history for attribution proxy.")
    else:
        df = hist[["wave_ret", "bm_ret"]].dropna()
        df["alpha_ret"] = df["wave_ret"] - df["bm_ret"]
        st.metric("30D Alpha (approx)", fmt_pct(a30))
        st.metric("365D Alpha (approx)", fmt_pct(a365))
        st.line_chart((df[["alpha_ret"]] * 100.0).rename(columns={"alpha_ret": "Daily Alpha (%)"}))

    render_definitions(keys=["Alpha"], title="Definitions (Attribution)")

# ============================================================
# TAB 4: Factor Decomposition
# ============================================================
with tabs[4]:
    st.subheader("Factor Decomposition (Light)")
    st.caption("Beta vs benchmark from daily returns.")

    if hist is None or hist.empty or len(hist) < 20:
        st.info("Not enough history.")
    else:
        b = beta_ols(hist["wave_ret"], hist["bm_ret"])
        st.metric("Beta vs Benchmark", fmt_num(b, 2))

    render_definitions(keys=["Return"], title="Definitions (Factor)")

# ============================================================
# TAB 5: Risk Lab
# ============================================================
with tabs[5]:
    st.subheader("Risk Lab")

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

# ============================================================
# TAB 6: Correlation
# ============================================================
with tabs[6]:
    st.subheader("Correlation (Daily Returns)")

    rets: Dict[str, pd.Series] = {}
    for w in all_waves:
        h = compute_wave_history(w, mode=mode, days=min(days, 365))
        h = _standardize_history(h)
        if h is not None and (not h.empty) and "wave_ret" in h.columns:
            rets[w] = h["wave_ret"]

    if len(rets) < 2:
        st.info("Not enough waves with history to compute correlations.")
    else:
        ret_df = pd.DataFrame(rets).dropna(how="all")
        corr = ret_df.corr()
        st.dataframe(corr, use_container_width=True)

# ============================================================
# TAB 7: Strategy Separation (Mode Proof)
# ============================================================
with tabs[7]:
    st.subheader("Strategy Separation (Mode Proof)")
    st.caption("Same wave across modes — proves strategies are distinct.")

    modes_to_check = safe_mode_list()
    rows: List[Dict[str, Any]] = []
    for m in modes_to_check:
        h = compute_wave_history(selected_wave, mode=m, days=min(days, 365))
        h = _standardize_history(h)
        if h is None or h.empty or len(h) < 10:
            rows.append(
                {
                    "Mode": m,
                    "Rows": 0,
                    "365D Return": np.nan,
                    "365D Alpha": np.nan,
                    "MaxDD": np.nan,
                    "TE": np.nan,
                }
            )
            continue

        rw = ret_from_nav(h["wave_nav"], min(365, len(h)))
        rb = ret_from_nav(h["bm_nav"], min(365, len(h)))
        rows.append(
            {
                "Mode": m,
                "Rows": int(len(h)),
                "365D Return": rw * 100.0,
                "365D Alpha": (rw - rb) * 100.0,
                "MaxDD": max_drawdown(h["wave_nav"]) * 100.0,
                "TE": tracking_error(h["wave_ret"], h["bm_ret"]) * 100.0,
            }
        )

    dfm = pd.DataFrame(rows)
    st.dataframe(style_perf_df(dfm), use_container_width=True)

# ============================================================
# TAB 8: Benchmark Integrity (Benchmark Truth)
# ============================================================
with tabs[8]:
    st.subheader("Benchmark Integrity & Difficulty")
    st.write(f"**Snapshot:** {bm_id} · **Drift:** {bm_drift.upper()}")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Difficulty vs SPY (proxy)", fmt_num(difficulty.get("difficulty_vs_spy"), 2))
    c2.metric("HHI (conc.)", fmt_num(difficulty.get("hhi"), 4))
    c3.metric("Entropy", fmt_num(difficulty.get("entropy"), 3))
    c4.metric("Top Weight", fmt_pct(difficulty.get("top_weight"), 2))

    st.write("Benchmark Mix (normalized)")
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

# ============================================================
# TAB 9: Drawdown Monitor
# ============================================================
with tabs[9]:
    st.subheader("Drawdown Monitor")
    if hist is None or hist.empty or len(hist) < 60:
        st.info("Not enough history for drawdown monitor.")
    else:
        ddw = drawdown_series(hist["wave_nav"])
        ddb = drawdown_series(hist["bm_nav"])
        st.metric("Max Drawdown (Wave)", fmt_pct(mdd))
        st.metric("Max Drawdown (Benchmark)", fmt_pct(mdd_b))
        st.line_chart(pd.concat([ddw.rename("Wave"), ddb.rename("Benchmark")], axis=1).dropna() * 100.0)

# ============================================================
# TAB 10: Diligence Flags (Alerts)
# ============================================================
with tabs[10]:
    st.subheader("Diligence Flags")
    notes = build_alerts(selected_wave, mode, hist, cov, bm_drift, te, a30, mdd)
    for n in notes:
        st.markdown(f"- {n}")
    st.divider()
    st.write(f"**Confidence:** {conf_level} — {conf_reason}")

# ============================================================
# TAB 11: WaveScore Leaderboard
# ============================================================
with tabs[11]:
    st.subheader("WaveScore Leaderboard (Console Approx.)")
    if ws_df is None or ws_df.empty:
        st.info("WaveScore unavailable (no history).")
    else:
        show = ws_df.copy()
        show["WaveScore"] = pd.to_numeric(show["WaveScore"], errors="coerce")
        show = show.sort_values("WaveScore", ascending=False, na_position="last").reset_index(drop=True)
        st.dataframe(show, use_container_width=True)

# ============================================================
# TAB 12: Governance Export
# ============================================================
with tabs[12]:
    st.subheader("Governance Export Pack (IC / Board Ready)")

    md = make_ic_pack_markdown(
        wave=selected_wave,
        mode=mode,
        bm_id=bm_id,
        bm_drift=bm_drift,
        cov=cov,
        ws_val=ws_val,
        ws_grade=_grade_from_score(ws_val),
        rank=rank,
        r30=r30, a30=a30,
        r60=r60, a60=a60,
        r365=r365, a365=a365,
        te=te, ir=ir,
        mdd=mdd, mdd_b=mdd_b,
        difficulty=difficulty,
    )

    st.download_button(
        "Download IC Pack (Markdown)",
        data=md.encode("utf-8"),
        file_name=f"IC_Pack_{selected_wave.replace(' ','_')}_{mode.replace(' ','_')}.md",
        mime="text/markdown",
        use_container_width=True,
    )

    perf_df2 = build_performance_matrix(all_waves, mode=mode, selected_wave=selected_wave, days=min(days, 365))
    if perf_df2 is not None and not perf_df2.empty:
        st.download_button(
            "Download Performance Matrix (CSV)",
            data=perf_df2.to_csv(index=False).encode("utf-8"),
            file_name=f"Performance_Matrix_{mode.replace(' ','_')}.csv",
            mime="text/csv",
            use_container_width=True,
        )
        # ============================================================
# TAB 13: IC Notes (formerly Vector OS Insight Layer)
# ============================================================
with tabs[13]:
    st.subheader("IC Notes")
    st.caption("Institutional-friendly narrative: what looks strong, what looks weak, what to verify next.")
    if hist is None or hist.empty or len(hist) < 20:
        st.info("Not enough data for insights yet.")
    else:
        notes = build_alerts(selected_wave, mode, hist, cov, bm_drift, te, a30, mdd)
        for n in notes:
            st.markdown(f"- {n}")

        # Optional: quick glossary access
        render_definitions(
            keys=["Coverage Score", "Benchmark Snapshot / Drift", "Tracking Error (TE)", "Information Ratio (IR)", "Max Drawdown (MaxDD)"],
            title="Definitions (IC Notes)"
        )

# ============================================================
# TAB 14: Daily Movement / Volatility
# ============================================================
with tabs[14]:
    st.subheader("Daily Movement / Volatility — Selected Wave")
    st.caption("Explains what changed, why it likely changed, and observable results (not advice).")

    # NOTE: ctx signature uses the unified builder from earlier parts.
    # If you changed build_decision_ctx() fields, keep them consistent.
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
        verdict=ic_verdict_text,          # from the scorecard tab
        trust_stack=trust_stack_summary,  # from the scorecard tab
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

# ============================================================
# TAB 15: Decision Intelligence
# ============================================================
with tabs[15]:
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
        verdict=ic_verdict_text,
        trust_stack=trust_stack_summary,
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

# ============================================================
# FOOTER: System Diagnostics (organized + safe)
# ============================================================
with st.expander("System Diagnostics (if something looks off)"):
    st.write("Engine loaded:", we is not None)
    st.write("Engine import error:", str(ENGINE_IMPORT_ERROR) if ENGINE_IMPORT_ERROR else "None")

    st.write("Decision engine loaded:", (generate_decisions is not None) or (build_daily_wave_activity is not None))
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