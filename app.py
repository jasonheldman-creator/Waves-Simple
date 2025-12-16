# app.py — WAVES Intelligence™ Institutional Console (Vector OS Edition)
# FULL PRODUCTION FILE (NO PATCHES) — PHASE 1 CLEANUP (DE-CLUTTER + ORGANIZE)
#
# What this build does:
#   ✅ Keeps ALL core behaviors: Engine guarded import, CSV fallback history, holdings, benchmark mix, WaveScore approx,
#      performance matrix, heatmap, risk lab, mode proof, benchmark integrity, exports, decision engine (optional), diagnostics.
#   ✅ Re-organizes UI into 5 top-level tabs + nested sections.
#   ✅ Adds "Advanced Analytics" toggle to hide power-user panels by default (demo-safe).
#   ✅ Shrinks sticky bar to 6 chips + "More" expander (mobile-friendly).
#
# What this build does NOT do:
#   • Does NOT modify engine math.
#   • Does NOT require decision_engine.py (optional).
#
# Optional files (if present):
#   - waves_engine.py
#   - decision_engine.py (optional)
#   - wave_history.csv (optional fallback)
#   - wave_weights.csv / list.csv (optional wave discovery)

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

/* IC cards */
.waves-card {
  border: 1px solid rgba(255,255,255,0.10);
  border-radius: 14px;
  padding: 12px 14px;
  background: rgba(255,255,255,0.03);
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
# Definitions / Glossary (Self-explanatory layer)
# ============================================================
GLOSSARY: Dict[str, str] = {
    "Return": "Portfolio return over the window (not annualized unless stated).",
    "Alpha": "Return minus Benchmark return over the same window (relative performance).",
    "Tracking Error (TE)": "Annualized volatility of (Wave daily returns − Benchmark daily returns). Higher = more active risk.",
    "Information Ratio (IR)": "Excess return divided by Tracking Error (risk-adjusted alpha).",
    "Max Drawdown (MaxDD)": "Largest peak-to-trough decline over the period (negative number).",
    "Benchmark Snapshot / Drift": "A fingerprint of the benchmark mix. Drift means the benchmark definition changed in-session.",
    "Coverage Score": "0–100 heuristic of data completeness + freshness (missing business days and staleness reduce score).",
    "Regime / VIX": "A simple market-stress cue. Higher VIX ≈ risk-off. Used for framing only (not a signal).",
    "WaveScore (Console Approx.)": "Console-side approximation (not the locked WAVESCORE™ spec) used for ranking/demos.",
    "Decision Intelligence": "Operating-system layer: actions/watch/notes based on observable analytics (not advice).",
}

def render_definitions(keys: List[str], title: str = "Definitions") -> None:
    with st.expander(title, expanded=False):
        for k in keys:
            st.markdown(f"**{k}:** {GLOSSARY.get(k, '(definition not found)')}")

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
# Core return/risk math (console-side helpers)
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

def daily_returns_from_nav(nav: pd.Series) -> pd.Series:
    nav = safe_series(nav).astype(float)
    if len(nav) < 2:
        return pd.Series(dtype=float)
    return nav.pct_change().rename("ret")

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

def var_cvar(daily_ret: pd.Series, level: float = 0.95) -> Tuple[float, float]:
    r = safe_series(daily_ret).dropna().astype(float)
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

# ============================================================
# Styling helpers (matrix heat)
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
        sty = sty.format({c: "{:.2f}%".format})
    return sty

# ============================================================
# Sticky helpers
# ============================================================
def chip(label: str) -> str:
    safe = (label or "").replace("<", "").replace(">", "")
    return f'<span class="waves-chip">{safe}</span>'

def render_sticky(primary_chips: List[str]) -> None:
    html = '<div class="waves-sticky">' + "".join([chip(c) for c in primary_chips if c]) + "</div>"
    st.markdown(html, unsafe_allow_html=True)

def google_quote_link(ticker: str) -> str:
    t = (ticker or "").upper().strip()
    if not t:
        return ""
    return f"https://www.google.com/finance/quote/{t}"

def safe_mode_list() -> List[str]:
    return ["Standard", "Alpha-Minus-Beta", "Private Logic"]
    # ============================================================
# Optional data fetch (yfinance) — VIX cue
# ============================================================
@st.cache_data(show_spinner=False)
def fetch_prices_daily(tickers: List[str], days: int = 60) -> pd.DataFrame:
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

def regime_from_vix(vix: float) -> str:
    if vix is None or (isinstance(vix, float) and math.isnan(vix)):
        return "unknown"
    if vix >= 25:
        return "risk-off"
    if vix <= 16:
        return "risk-on"
    return "neutral"

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
                waves = [w for w in waves if w and w.lower() != "nan"]
                return sorted(waves)
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
# Confidence meter (trust cue)
# ============================================================
def confidence_from_integrity(cov: Dict[str, Any], bm_drift: str) -> Tuple[str, str]:
    """
    Returns (level, reason) where level in {"High","Medium","Low"}
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
# WaveScore (console-side approximation) + ranks
# ============================================================
@st.cache_data(show_spinner=False)
def compute_wavescore_for_all_waves(all_waves: List[str], mode: str, days: int = 365) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for wave in all_waves:
        hist = compute_wave_history(wave, mode=mode, days=days)
        if hist is None:
            hist = pd.DataFrame()
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
    if df.empty:
        return df
    df["Rank"] = df["WaveScore"].rank(ascending=False, method="min")
    return df.sort_values("Wave").reset_index(drop=True)

# ============================================================
# Performance Matrix + Alpha Heatmap
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

    if selected_wave in set(df["Wave"]):
        top = df[df["Wave"] == selected_wave]
        rest = df[df["Wave"] != selected_wave]
        df = pd.concat([top, rest], axis=0)

    return df.reset_index(drop=True)

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

    df = pd.DataFrame(rows).sort_values("Wave").reset_index(drop=True)
    # keep as decimals; caller can format/plot
    return df

def plot_alpha_heatmap(alpha_df: pd.DataFrame, title: str) -> None:
    if go is None or alpha_df is None or alpha_df.empty:
        st.info("Heatmap unavailable (Plotly missing or no data).")
        return

    df = alpha_df.copy()
    cols = [c for c in ["1D Alpha", "30D Alpha", "60D Alpha", "365D Alpha"] if c in df.columns]
    if not cols:
        st.info("No alpha columns to plot.")
        return

    z = (df[cols].values * 100.0)  # percent points
    y = df["Wave"].tolist()
    x = cols

    v = float(np.nanmax(np.abs(z))) if np.isfinite(z).any() else 10.0
    if not math.isfinite(v) or v <= 0:
        v = 10.0

    fig = go.Figure(data=go.Heatmap(z=z, x=x, y=y, zmin=-v, zmax=v, colorbar=dict(title="Alpha (pp)")))
    fig.update_layout(title=title, height=min(950, 260 + 22 * max(12, len(y))), margin=dict(l=80, r=40, t=60, b=40))
    st.plotly_chart(fig, use_container_width=True)
    # ============================================================
# Alerts / flags + Decision ctx
# ============================================================
def build_alerts(cov: Dict[str, Any], bm_drift: str, te: float, a30: float, mdd: float) -> List[str]:
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
        if cov.get("age_days", 0) is not None and cov.get("age_days", 0) >= 5:
            notes.append("Stale history: last datapoint is >=5 days old (check engine writes).")
        if not notes:
            notes.append("No major anomalies detected on this window.")
        return notes
    except Exception:
        return ["Alert system error (non-fatal)."]

def build_decision_ctx(
    wave: str,
    mode: str,
    bm_snapshot: str,
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
        "bm_snapshot": bm_snapshot,
        "bm_drift": bm_drift,
        "rows": cov.get("rows"),
        "age_days": cov.get("age_days"),
        "completeness_score": cov.get("completeness_score"),
        "confidence": cov.get("confidence"),
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
# Diagnostics (kept safe + inside function)
# ============================================================
def render_diagnostics(all_waves: List[str]) -> None:
    st.subheader("Diagnostics (Safe)")
    st.caption("Use this if anything looks off. This panel should never crash the app.")
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
        st.warning("decision_engine import issue (non-fatal). Decision sections will fallback.")
        st.code(str(DECISION_IMPORT_ERROR))
    else:
        st.success("decision_engine import OK (or not required).")

    st.markdown("**Files present**")
    for fp in ["wave_history.csv", "wave_weights.csv", "list.csv", "requirements.txt", "decision_engine.py", "waves_engine.py"]:
        st.write(fp, "✅" if os.path.exists(fp) else "—")

    st.markdown("**Wave discovery**")
    st.write("Count:", len(all_waves))
    if all_waves:
        st.write(all_waves[:25])

# ============================================================
# MAIN UI
# ============================================================
st.title("WAVES Intelligence™ Institutional Console")

# Sidebar controls
st.sidebar.markdown("## Controls")
advanced = st.sidebar.toggle("Advanced Analytics", value=False, help="OFF = clean demo. ON = power panels.")

mode = st.sidebar.selectbox("Mode", options=safe_mode_list(), index=0)
days = st.sidebar.slider("History window (days)", min_value=90, max_value=1500, value=365, step=30)

# Definitions always available
with st.sidebar.expander("Metric Definitions", expanded=False):
    render_definitions(
        keys=[
            "Return", "Alpha", "Tracking Error (TE)", "Information Ratio (IR)",
            "Max Drawdown (MaxDD)", "Benchmark Snapshot / Drift",
            "Coverage Score", "Regime / VIX", "WaveScore (Console Approx.)",
            "Decision Intelligence"
        ],
        title="Open definitions"
    )

# Discover waves
all_waves = get_all_waves_safe()
if not all_waves:
    st.warning("No waves discovered yet. If unexpected, check engine + CSV files.")
    if advanced:
        render_diagnostics(all_waves)
    st.stop()

selected_wave = st.sidebar.selectbox("Wave", options=all_waves, index=0)

# Load core objects
hist = compute_wave_history(selected_wave, mode=mode, days=days)
hist = _standardize_history(hist)

bm_mix = get_benchmark_mix()
bm_id = benchmark_snapshot_id(selected_wave, bm_mix)
bm_drift = benchmark_drift_status(selected_wave, mode, bm_id)

cov = coverage_report(hist)

# VIX/regime
vix_val = float("nan")
regime = "unknown"
if yf is not None:
    try:
        vix_df = fetch_prices_daily(["^VIX"], days=30)
        if not vix_df.empty and "^VIX" in vix_df.columns:
            vix_val = float(vix_df["^VIX"].dropna().iloc[-1])
    except Exception:
        pass
regime = regime_from_vix(vix_val)

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
mdd_b = max_drawdown(nav_b)

# WaveScore table + rank
ws_table = compute_wavescore_for_all_waves(all_waves, mode=mode, days=min(365, days))
ws_row = ws_table[ws_table["Wave"] == selected_wave] if isinstance(ws_table, pd.DataFrame) and not ws_table.empty else pd.DataFrame()
ws_val = float(ws_row["WaveScore"].iloc[0]) if (not ws_row.empty and "WaveScore" in ws_row.columns and pd.notna(ws_row["WaveScore"].iloc[0])) else float("nan")
ws_grade = _grade_from_score(ws_val) if math.isfinite(ws_val) else "N/A"
ws_rank = int(ws_row["Rank"].iloc[0]) if (not ws_row.empty and "Rank" in ws_row.columns and pd.notna(ws_row["Rank"].iloc[0])) else None

# Benchmark difficulty proxy
bm_rows = pd.DataFrame()
try:
    if bm_mix is not None and not bm_mix.empty and "Wave" in bm_mix.columns:
        bm_rows = bm_mix[bm_mix["Wave"].astype(str) == str(selected_wave)].copy()
        if "Ticker" in bm_rows.columns and "Weight" in bm_rows.columns:
            bm_rows = _normalize_bm_rows(bm_rows[["Ticker", "Weight"]].copy())
except Exception:
    bm_rows = pd.DataFrame()
difficulty = benchmark_difficulty_proxy(bm_rows)

# Confidence meter
conf_level, conf_reason = confidence_from_integrity(cov, bm_drift)

# Sticky (6 chips only)
primary_chips = [
    f"BM: {bm_id} · {'Stable' if bm_drift=='stable' else 'DRIFT'}",
    f"Coverage: {fmt_num(cov.get('completeness_score', np.nan), 1)}/100",
    f"Confidence: {conf_level}",
    f"Regime: {regime} · VIX: {fmt_num(vix_val, 1)}",
    f"30D α: {fmt_pct(a30)}",
    f"WaveScore: {fmt_score(ws_val)} ({ws_grade}) · Rank: {ws_rank if ws_rank else '—'}",
]
render_sticky(primary_chips)

with st.expander("More (chips + integrity details)", expanded=False):
    st.markdown("#### Extra chips")
    st.write(
        "Rows:", cov.get("rows", 0),
        "| Age:", cov.get("age_days", "—"),
        "| 60D α:", fmt_pct(a60),
        "| 365D α:", fmt_pct(a365),
        "| TE:", fmt_pct(te),
        "| IR:", fmt_num(ir, 2),
        "| MaxDD:", fmt_pct(mdd),
    )
    st.markdown("#### Coverage flags")
    if cov.get("flags"):
        st.warning(" • ".join(cov["flags"]))
    else:
        st.success("No integrity flags on this window.")
        st.caption("Observational analytics only (not trading advice).")

# ============================================================
# 5 TOP-LEVEL TABS (Phase-1 cleanup)
# ============================================================
tabs = st.tabs([
    "IC Summary",
    "Performance",
    "Risk & Integrity",
    "Attribution & Factors",
    "Ops & Exports",
])

# ============================================================
# TAB 0: IC SUMMARY (clean landing page)
# ============================================================
with tabs[0]:
    st.subheader(f"IC Summary — {selected_wave} ({mode})")

    ctx = build_decision_ctx(
        wave=selected_wave,
        mode=mode,
        bm_snapshot=bm_id,
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
        rank=ws_rank,
    )

    left, right = st.columns([1.3, 1.0])

    with left:
        st.markdown('<div class="waves-card">', unsafe_allow_html=True)
        st.markdown("### Decision Intelligence (Translator)")
        if generate_decisions is not None:
            try:
                d = generate_decisions(ctx)
            except Exception as e:
                d = {"actions": [f"Decision engine error: {e}"], "watch": [], "notes": []}
        else:
            d = {"actions": [], "watch": [], "notes": []}
            st.info("Decision engine not present. Showing safe heuristic summary below.")

        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**Action**")
            if d.get("actions"):
                for x in d.get("actions", []):
                    st.write(f"• {x}")
            else:
                # safe heuristics
                heur = []
                if conf_level == "Low":
                    heur.append("Data confidence LOW — verify history freshness and missing business days.")
                if bm_drift != "stable":
                    heur.append("Benchmark drift detected — freeze snapshot for demos/IC.")
                st.write("• " + (" / ".join(heur) if heur else "No urgent actions detected."))
        with c2:
            st.markdown("**Watch**")
            heur = []
            if math.isfinite(a30) and a30 < 0:
                heur.append("30D alpha weak — monitor for compression vs noise.")
            if math.isfinite(te) and te > 0.20:
                heur.append("Tracking error elevated — active risk higher.")
            st.write("• " + (" / ".join(heur) if heur else "No major watch items."))
        with c3:
            st.markdown("**Notes**")
            st.write("• " + (d.get("notes")[0] if (d.get("notes") or []) else "Observational analytics only."))

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
            st.success("No data integrity flags detected on this window.")
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown('<div class="waves-card">', unsafe_allow_html=True)
        st.markdown("### Performance Snapshot")
        c1, c2, c3 = st.columns(3)
        c1.metric("30D Return", fmt_pct(r30))
        c2.metric("30D Alpha", fmt_pct(a30))
        c3.metric("Max Drawdown", fmt_pct(mdd))

        c4, c5, c6 = st.columns(3)
        c4.metric("TE (ann.)", fmt_pct(te))
        c5.metric("IR", fmt_num(ir, 2))
        c6.metric("WaveScore", f"{fmt_score(ws_val)} ({ws_grade})")
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
    st.markdown("### Top Holdings (Top 10, Clickable)")
    hold = get_wave_holdings(selected_wave)
    if hold is None or hold.empty:
        st.info("Holdings unavailable (engine not exposing or wave_weights.csv missing).")
    else:
        hold2 = hold.copy()
        hold2["Ticker"] = hold2["Ticker"].astype(str).str.upper().str.strip()
        hold2["Weight"] = pd.to_numeric(hold2["Weight"], errors="coerce").fillna(0.0)
        tot = float(hold2["Weight"].sum())
        if tot > 0:
            hold2["Weight"] = hold2["Weight"] / tot
        hold2 = hold2.sort_values("Weight", ascending=False).reset_index(drop=True)
        hold2["Weight %"] = (hold2["Weight"] * 100.0).round(2)
        hold2["Google"] = hold2["Ticker"].apply(lambda t: google_quote_link(str(t)))

        try:
            st.dataframe(
                hold2.head(10)[["Ticker", "Name", "Weight %", "Google"]],
                use_container_width=True,
                column_config={
                    "Google": st.column_config.LinkColumn("Google", display_text="Open"),
                },
            )
        except Exception:
            st.dataframe(hold2.head(10), use_container_width=True)

    render_definitions(
        keys=["Alpha", "Tracking Error (TE)", "Information Ratio (IR)", "Max Drawdown (MaxDD)", "Benchmark Snapshot / Drift", "Coverage Score", "Decision Intelligence"],
        title="Definitions (IC Summary)"
    )

# ============================================================
# TAB 1: PERFORMANCE (matrix + heatmap + charts)
# ============================================================
with tabs[1]:
    st.subheader("Performance")

    cA, cB, cC = st.columns([1.2, 1.0, 1.3])
    with cA:
        sort_by = st.selectbox("Sort by", ["Selected First (default)", "30D Alpha", "60D Alpha", "30D Return", "Rows", "WaveScore"], index=0)
    with cB:
        show_365 = st.toggle("Show 365D columns", value=False)
    with cC:
        st.caption("Scan 30D/60D first. Keep 365D hidden unless needed.")

    perf_df = build_performance_matrix(all_waves, mode=mode, selected_wave=selected_wave, days=min(days, 365))
    if perf_df is None or perf_df.empty:
        st.info("Performance matrix unavailable (no history).")
    else:
        df = perf_df.copy()
        if sort_by != "Selected First (default)":
            if sort_by in df.columns:
                df = df.sort_values(sort_by, ascending=False, na_position="last")
            elif sort_by == "WaveScore":
                if ws_table is not None and not ws_table.empty:
                    df = df.merge(ws_table[["Wave", "WaveScore"]], on="Wave", how="left")
                    df = df.sort_values("WaveScore", ascending=False, na_position="last").drop(columns=["WaveScore"], errors="ignore")
            if selected_wave in set(df["Wave"]):
                top = df[df["Wave"] == selected_wave]
                rest = df[df["Wave"] != selected_wave]
                df = pd.concat([top, rest], axis=0)

        if not show_365:
            df = df.drop(columns=[c for c in df.columns if c.startswith("365D ")], errors="ignore")

        st.dataframe(style_perf_df(df), use_container_width=True, height=650)
        st.caption("Values are **percent points**. Green=positive, Red=negative.")
        render_definitions(keys=["Return", "Alpha"], title="Definitions (Matrix)")

    st.divider()
    st.subheader("Alpha Heatmap (All Waves × Timeframe)")
    alpha_df = build_alpha_matrix(all_waves, mode=mode)
    plot_alpha_heatmap(alpha_df, title=f"Alpha Heatmap — Mode: {mode}")
    render_definitions(keys=["Alpha"], title="Definitions (Heatmap)")

    st.divider()
    st.subheader("Selected Wave Charts")
    if hist is None or hist.empty or len(hist) < 5:
        st.warning("Not enough history for charts.")
    else:
        nav_df = pd.concat([hist["wave_nav"].rename("Wave NAV"), hist["bm_nav"].rename("Benchmark NAV")], axis=1).dropna()
        if not nav_df.empty:
            st.markdown("**NAV vs Benchmark (normalized)**")
            st.line_chart((nav_df / nav_df.iloc[0]), height=300, use_container_width=True)

        st.markdown("**Rolling 30D Alpha (%)**")
        ra = rolling_alpha_from_nav(hist["wave_nav"], hist["bm_nav"], window=30).dropna()
        if len(ra):
            st.line_chart((ra * 100.0).rename("Rolling 30D Alpha (%)"), height=260, use_container_width=True)
        else:
            st.info("Not enough data for rolling alpha.")

# ============================================================
# TAB 2: RISK & INTEGRITY
# ============================================================
with tabs[2]:
    st.subheader("Risk & Integrity")

    # Always-visible integrity block
    st.markdown("### Data Integrity")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Coverage Score", fmt_num(cov.get("completeness_score", np.nan), 1))
    c2.metric("Age (days)", cov.get("age_days", "—"))
    c3.metric("Rows", cov.get("rows", 0))
    c4.metric("Confidence", conf_level)
    st.caption(conf_reason)

    if cov.get("flags"):
        st.warning(" • ".join(cov["flags"]))

    st.divider()

    # Benchmark integrity
    st.markdown("### Benchmark Integrity")
    st.write(f"**Snapshot:** {bm_id} · **Drift:** {'STABLE' if bm_drift=='stable' else 'DRIFT'}")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Difficulty vs SPY (proxy)", fmt_num(difficulty.get("difficulty_vs_spy"), 2))
    c2.metric("HHI (conc.)", fmt_num(difficulty.get("hhi"), 4))
    c3.metric("Entropy", fmt_num(difficulty.get("entropy"), 3))
    c4.metric("Top Weight", fmt_pct(difficulty.get("top_weight"), 2))

    if advanced:
        st.markdown("**Benchmark Mix (normalized)**")
        if bm_rows is None or bm_rows.empty:
            st.info("Benchmark mix table unavailable.")
        else:
            show = bm_rows.copy()
            show["Weight %"] = (show["Weight"] * 100.0).round(2)
            st.dataframe(show[["Ticker", "Weight %"]], use_container_width=True, height=360)

    render_definitions(keys=["Benchmark Snapshot / Drift"], title="Definitions (Benchmark)")

    st.divider()

    # Risk lab (basic always, extended if advanced)
    st.markdown("### Risk Lab")
    if hist is None or hist.empty or len(hist) < 50:
        st.info("Not enough history for risk lab metrics (need ~50+ points).")
    else:
        r = hist["wave_ret"].dropna()
        v95, c95 = var_cvar(r, level=0.95)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Max Drawdown", fmt_pct(mdd))
        c2.metric("TE (ann.)", fmt_pct(te))
        c3.metric("IR", fmt_num(ir, 2))
        c4.metric("VaR 95% (daily)", fmt_pct(v95))

        if advanced:
            c5, c6 = st.columns(2)
            with c5:
                st.markdown("**Drawdown (Wave vs Benchmark) — %**")
                dd_df = pd.concat([drawdown_series(nav_w).rename("Wave"), drawdown_series(nav_b).rename("Benchmark")], axis=1).dropna()
                if not dd_df.empty:
                    st.line_chart(dd_df * 100.0, height=260, use_container_width=True)
            with c6:
                st.markdown("**Daily Alpha (Wave−BM) — %**")
                df = pd.concat([ret_w.rename("Wave"), ret_b.rename("BM")], axis=1).dropna()
                if not df.empty:
                    alpha_daily = (df["Wave"] - df["BM"]) * 100.0
                    st.line_chart(alpha_daily.rename("Daily Alpha (%)"), height=260, use_container_width=True)

    # Diligence flags
    st.divider()
    st.markdown("### Diligence Flags")
    for n in build_alerts(cov, bm_drift, te, a30, mdd):
        st.markdown(f"- {n}")

# ============================================================
# TAB 3: ATTRIBUTION & FACTORS
# ============================================================
with tabs[3]:
    st.subheader("Attribution & Factors")

    st.markdown("### Attribution (Light)")
    st.caption("Console-side proxy: compares Wave returns to Benchmark returns (alpha).")
    if hist is None or hist.empty or len(hist) < 30:
        st.info("Not enough history for attribution proxy.")
    else:
        df = hist[["wave_ret", "bm_ret"]].dropna()
        df["alpha_ret"] = df["wave_ret"] - df["bm_ret"]
        st.metric("30D Alpha (approx)", fmt_pct(a30))
        st.metric("365D Alpha (approx)", fmt_pct(a365))
        st.line_chart((df[["alpha_ret"]] * 100.0).rename(columns={"alpha_ret": "Daily Alpha (%)"}), height=260, use_container_width=True)

    render_definitions(keys=["Alpha"], title="Definitions (Attribution)")

    st.divider()
    st.markdown("### Factor Decomposition (Light)")
    st.caption("Beta vs benchmark from daily returns.")
    if hist is None or hist.empty or len(hist) < 20:
        st.info("Not enough history for beta estimate.")
    else:
        b = beta_ols(hist["wave_ret"], hist["bm_ret"])
        st.metric("Beta vs Benchmark", fmt_num(b, 2))

    # Mode proof (advanced)
    if advanced:
        st.divider()
        st.markdown("### Strategy Separation (Mode Proof)")
        rows = []
        for m in safe_mode_list():
            h2 = compute_wave_history(selected_wave, mode=m, days=min(days, 365))
            h2 = _standardize_history(h2)
            if h2 is None or h2.empty or len(h2) < 10:
                rows.append({"Mode": m, "Rows": 0, "365D Return": np.nan, "365D Alpha": np.nan, "MaxDD": np.nan, "TE": np.nan})
                continue
            rw = ret_from_nav(h2["wave_nav"], min(365, len(h2)))
            rb = ret_from_nav(h2["bm_nav"], min(365, len(h2)))
            rows.append({
                "Mode": m,
                "Rows": int(len(h2)),
                "365D Return": rw * 100.0,
                "365D Alpha": (rw - rb) * 100.0,
                "MaxDD": max_drawdown(h2["wave_nav"]) * 100.0,
                "TE": tracking_error(h2["wave_ret"], h2["bm_ret"]) * 100.0,
            })
        dfm = pd.DataFrame(rows)
        st.dataframe(style_perf_df(dfm), use_container_width=True, height=340)

# ============================================================
# TAB 4: OPS & EXPORTS
# ============================================================
with tabs[4]:
    st.subheader("Ops & Exports")

    st.markdown("### WaveScore Leaderboard (Console Approx.)")
    if ws_table is None or ws_table.empty:
        st.info("WaveScore unavailable (no history).")
    else:
        show = ws_table.copy()
        show["WaveScore"] = pd.to_numeric(show["WaveScore"], errors="coerce")
        show = show.sort_values("WaveScore", ascending=False, na_position="last").reset_index(drop=True)
        st.dataframe(show, use_container_width=True, height=420)

    st.divider()
    st.markdown("### Exports")
    perf_export = build_performance_matrix(all_waves, mode=mode, selected_wave=selected_wave, days=min(days, 365))
    if perf_export is not None and not perf_export.empty:
        st.download_button(
            "Download Performance Matrix (CSV)",
            data=perf_export.to_csv(index=False).encode("utf-8"),
            file_name=f"Performance_Matrix_{mode.replace(' ','_')}.csv",
            mime="text/csv",
            use_container_width=True,
        )

    if ws_table is not None and not ws_table.empty:
        st.download_button(
            "Download WaveScore Table (CSV)",
            data=ws_table.to_csv(index=False).encode("utf-8"),
            file_name=f"WaveScore_Table_{mode.replace(' ','_')}.csv",
            mime="text/csv",
            use_container_width=True,
        )

    if hist is not None and not hist.empty:
        hx = hist.copy().reset_index().rename(columns={"index": "date"})
        st.download_button(
            f"Download {selected_wave} History (CSV)",
            data=hx.to_csv(index=False).encode("utf-8"),
            file_name=f"{selected_wave.replace(' ','_')}_{mode.replace(' ','_')}_history.csv",
            mime="text/csv",
            use_container_width=True,
        )

    # Decision engine (advanced)
    if advanced:
        st.divider()
        st.markdown("### Decision Intelligence (Advanced)")
        st.caption("If decision_engine.py exists, it will populate here. Otherwise, safe defaults show.")
        ctx = build_decision_ctx(
            wave=selected_wave,
            mode=mode,
            bm_snapshot=bm_id,
            bm_drift=bm_drift,
            cov=cov,
            vix_val=vix_val,
            regime=regime,
            r30=r30, a30=a30,
            r60=r60, a60=a60,
            r365=r365, a365=a365,
            te=te, ir=ir, mdd=mdd,
            wavescore=ws_val, rank=ws_rank,
        )

        if generate_decisions is not None:
            try:
                out = generate_decisions(ctx)
                st.write(out)
            except Exception as e:
                st.warning("Decision engine error (non-fatal).")
                st.code(str(e))
        else:
            st.info("decision_engine.py not present (or missing generate_decisions).")

        if build_daily_wave_activity is not None:
            st.divider()
            st.markdown("### Daily Activity (Optional)")
            try:
                activity = build_daily_wave_activity(ctx)
                st.write(activity)
            except Exception as e:
                st.warning("Daily activity builder error (non-fatal).")
                st.code(str(e))

        st.divider()
        render_diagnostics(all_waves)

# ============================================================
# End-of-file guardrails
# ============================================================
# Never leave trailing, unclosed blocks or strings below this line.
# If you add code, keep it inside functions or tab scopes.
# EOF