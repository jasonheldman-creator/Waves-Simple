# app.py — WAVES Intelligence™ Institutional Console (Vector OS Edition)
# FULL PRODUCTION FILE (NO PATCHES) — v1 + ELITE COPILOT PACK + INTERPRETATION LAYER (3-file)
#
# Keeps EVERYTHING from your current V1 app:
#   • Sticky Summary Bar
#   • Console (Rich View + Performance Matrix)
#   • Alpha Heatmap (raw alpha)
#   • Attribution (Engine vs Static Basket proxy)
#   • Factor Decomposition
#   • Risk Lab (Sharpe/Sortino/VaR/CVaR/Rolling/Drawdown)
#   • Correlation Matrix
#   • WaveScore Leaderboard (console-side approximation)
#   • Mode Separation Proof
#   • Benchmark Truth + Difficulty scoring + drift snapshot
#   • Drawdown Monitor
#   • Alerts & Flags Panel
#   • Governance Export Pack
#   • Vector OS Insight Layer
#
# Adds (non-destructive):
#   • Decision Intelligence (decision_engine.py)
#   • Scoring Guide tab (grade charts for key metrics) (metric_guide.py)
#   • Alpha Heat Index (AHI) tab (0–100, derived from existing alpha matrix; no engine change)
#   • Grade badges next to Difficulty/HHI/Entropy/TopWeight/WaveScore/AHI
#
# Notes:
#   • Engine math NOT modified.
#   • Robust history loader: engine functions → wave_history.csv fallback
#   • Prevents blank screen by always showing diagnostics + safe fallbacks.
#   • Guarded imports: if metric_guide/decision_engine not present, console still runs.

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
# NEW: Interpretation + Decision layer imports (guarded)
# -------------------------------
MG_IMPORT_ERROR = None
DE_IMPORT_ERROR = None
try:
    import metric_guide as mg
except Exception as e:
    mg = None
    MG_IMPORT_ERROR = e

try:
    import decision_engine as de
except Exception as e:
    de = None
    DE_IMPORT_ERROR = e
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

/* NEW: Grade badge (small) */
.waves-badge {
  display:inline-block;
  padding: 4px 10px;
  margin-left: 8px;
  border-radius: 999px;
  border: 1px solid rgba(255,255,255,0.14);
  background: rgba(255,255,255,0.06);
  font-size: 0.82rem;
  white-space: nowrap;
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
    return "D"

# ============================================================
# NEW: Interpretation Layer helpers (metric_guide.py) — SAFE
# ============================================================
def _mg_grade(metric_key: str, value: float) -> Tuple[str, str]:
    try:
        if mg is None:
            return ("N/A", "Guide not loaded")
        # Preferred API: mg.grade(metric_key, value) -> (grade,label)
        if hasattr(mg, "grade"):
            out = mg.grade(metric_key, value)
            if isinstance(out, (tuple, list)) and len(out) >= 2:
                return (str(out[0]), str(out[1]))
        # Fallback: mg.grade_<key>(value)
        fn = f"grade_{metric_key}"
        if hasattr(mg, fn):
            out = getattr(mg, fn)(value)
            if isinstance(out, (tuple, list)) and len(out) >= 2:
                return (str(out[0]), str(out[1]))
        return ("N/A", "No grade mapping")
    except Exception:
        return ("N/A", "Grade error")

def _badge_html(grade: str, label: str) -> str:
    return f'<span class="waves-badge">{grade} · {label}</span>'

def show_metric_with_badge(title: str, value_str: str, metric_key: Optional[str], metric_value: Optional[float]):
    if not metric_key or metric_value is None or not math.isfinite(float(metric_value)):
        st.markdown(f"**{title}:** {value_str}")
        return
    g, lab = _mg_grade(metric_key, float(metric_value))
    st.markdown(f"**{title}:** {value_str} {_badge_html(g, lab)}", unsafe_allow_html=True)

# ============================================================
# NEW: Decision Intelligence (decision_engine.py) — SAFE
# ============================================================
def _de_recommend(context: Dict[str, Any]) -> List[str]:
    try:
        if de is None:
            return []
        if hasattr(de, "recommend"):
            out = de.recommend(context)
            if isinstance(out, list):
                return [str(x) for x in out]
        if hasattr(de, "get_recommendations"):
            out = de.get_recommendations(context)
            if isinstance(out, list):
                return [str(x) for x in out]
        return []
    except Exception:
        return []

def show_decision_panel(context: Dict[str, Any]):
    recs = _de_recommend(context)
    if not recs:
        st.info("Decision Intelligence: No recommendations triggered on this window.")
        return
    for r in recs[:10]:
        st.markdown(f"- {r}")
        # ============================================================
# Styling helpers: % matrix + green/red heat
# ============================================================
def _heat_color(val: Any) -> str:
    try:
        if val is None:
            return ""
        v = float(val)
        if math.isnan(v):
            return ""
        # val is percent points (e.g., +5.2)
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
    """
    Console-side difficulty proxy:
      - concentration (HHI)
      - entropy (diversification)
      - top weight
      - difficulty score vs SPY baseline (heuristic)
    """
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
# Alpha Heatmap (raw alpha)
# ============================================================
@st.cache_data(show_spinner=False)
def build_alpha_matrix(all_waves: List[str], mode: str) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for wname in all_waves:
        hist = compute_wave_history(wname, mode=mode, days=365)
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
# NEW: Alpha Heat Index (AHI) builder (0–100) from alpha_df ranks
# ============================================================
def build_ahi_table(alpha_df: pd.DataFrame) -> pd.DataFrame:
    """
    Converts raw alpha columns into percentile ranks scaled 0–100.
    50 ≈ median by construction.
    """
    if alpha_df is None or alpha_df.empty:
        return pd.DataFrame()
    out = pd.DataFrame({"Wave": alpha_df["Wave"].astype(str)})
    for col in ["1D Alpha", "30D Alpha", "60D Alpha", "365D Alpha"]:
        if col in alpha_df.columns:
            s = pd.to_numeric(alpha_df[col], errors="coerce")
            out[col.replace("Alpha", "AHI")] = (s.rank(pct=True, method="average") * 100.0)
    return out


# ============================================================
# Performance Matrix (Returns + Alpha) — percent + red/green heat
# ============================================================
@st.cache_data(show_spinner=False)
def build_performance_matrix(all_waves: List[str], mode: str, selected_wave: str, days: int = 365) -> pd.DataFrame:
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

    # Convert decimals → percent points
    for c in [c for c in df.columns if "Return" in c or "Alpha" in c]:
        df[c] = pd.to_numeric(df[c], errors="coerce") * 100.0

    # Put selected wave first
    if "Wave" in df.columns and selected_wave in set(df["Wave"]):
        top = df[df["Wave"] == selected_wave]
        rest = df[df["Wave"] != selected_wave]
        df = pd.concat([top, rest], axis=0)

    return df.reset_index(drop=True)


# ============================================================
# Alerts & Flags (Elite)
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
# Governance Export Pack
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

## Benchmark Truth
- Snapshot ID: **{bm_id}**
- Drift status: **{bm_drift.upper()}**
- Difficulty vs SPY (proxy): **{fmt_num(difficulty.get('difficulty_vs_spy'), 2)}**
- HHI (concentration): **{fmt_num(difficulty.get('hhi'), 4)}**
- Entropy (diversification): **{fmt_num(difficulty.get('entropy'), 3)}**
- Top weight: **{fmt_pct(difficulty.get('top_weight'), 2)}**

## Coverage / Data Integrity
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

if ENGINE_IMPORT_ERROR is not None:
    st.error("Engine import failed. The app will use CSV fallbacks where possible.")
    st.code(str(ENGINE_IMPORT_ERROR))

all_waves = get_all_waves_safe()
if not all_waves:
    st.warning("No waves discovered yet. If unexpected, check engine import + CSV files.")
    with st.expander("Diagnostics"):
        st.write("Files present:")
        st.write({p: os.path.exists(p) for p in ["wave_config.csv", "wave_weights.csv", "wave_history.csv", "list.csv", "waves_engine.py", "metric_guide.py", "decision_engine.py"]})
        st.write("metric_guide import error:", str(MG_IMPORT_ERROR) if MG_IMPORT_ERROR else "None")
        st.write("decision_engine import error:", str(DE_IMPORT_ERROR) if DE_IMPORT_ERROR else "None")
    st.stop()

modes = ["Standard", "Alpha-Minus-Beta", "Private Logic"]

with st.sidebar:
    st.header("Controls")
    mode = st.selectbox("Mode", modes, index=0)
    selected_wave = st.selectbox("Wave", all_waves, index=0)
    days = st.slider("History window (days)", min_value=90, max_value=1500, value=365, step=30)
    st.caption("If history is empty, app falls back to wave_history.csv automatically.")

bm_mix = get_benchmark_mix()
bm_id = benchmark_snapshot_id(selected_wave, bm_mix)
bm_drift = benchmark_drift_status(selected_wave, mode, bm_id)

hist = compute_wave_history(selected_wave, mode=mode, days=days)
cov = coverage_report(hist)

# Precompute stats used across multiple tabs (avoid NameError landmines)
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

# Sticky regime chip (optional)
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

# Benchmark difficulty proxy (truth)
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

# Sticky chips
chips = []
chips.append(f"BM Snapshot: {bm_id} · {'Stable' if bm_drift=='stable' else 'DRIFT'}")
chips.append(f"Coverage: {fmt_num(cov.get('completeness_score', np.nan),1)} / 100")
chips.append(f"Rows: {cov.get('rows','—')} · Age: {cov.get('age_days','—')}")
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

# ============================================================
# Tabs (RESTORED + ELITE) + NEW tabs
# ============================================================
tabs = st.tabs([
    "Console",
    "Alpha Heat Index (AHI)",    # NEW
    "Scoring Guide",             # NEW
    "Attribution",
    "Factor Decomposition",
    "Risk Lab",
    "Correlation",
    "Mode Proof",
    "Benchmark Truth",
    "Drawdown Monitor",
    "Alerts",
    "WaveScore Leaderboard",
    "Governance Export",
    "Vector OS Insight Layer",
])

# Decision context (fed into decision_engine.py if present)
decision_context = {
    "wave": selected_wave,
    "mode": mode,
    "bm_id": bm_id,
    "bm_drift": bm_drift,
    "coverage": cov,
    "difficulty": difficulty,
    "r30": r30,
    "a30": a30,
    "r60": r60,
    "a60": a60,
    "r365": r365,
    "a365": a365,
    "te": te,
    "ir": ir,
    "mdd": mdd,
    "mdd_b": mdd_b,
    "vix": vix_val,
    "regime": regime,
    "wavescore": ws_val,
}
# -------------------------
# TAB: Console (RESTORED RICH VIEW + PERFORMANCE MATRIX)
# -------------------------
with tabs[0]:
    st.subheader("Decision Intelligence (So what / Next step)")
    show_decision_panel(decision_context)

    st.divider()
    st.subheader("All-Waves Performance Matrix (Returns + Alpha)")
    perf_df = build_performance_matrix(all_waves, mode=mode, selected_wave=selected_wave, days=min(days, 365))

    if perf_df is None or perf_df.empty:
        st.info("Performance matrix unavailable (no history).")
    else:
        st.dataframe(style_perf_df(perf_df), use_container_width=True)
        st.caption("Values shown as **percent**. Green = positive, Red = negative.")

    st.divider()

    st.subheader("Alpha Heatmap (All Waves × Timeframe)")
    alpha_df = build_alpha_matrix(all_waves, mode=mode)
    plot_alpha_heatmap(alpha_df, title=f"Alpha Heatmap — Mode: {mode}")

    st.divider()

    st.subheader("Selected Wave — NAV vs Benchmark")
    if hist is None or hist.empty or len(hist) < 5:
        st.warning("Not enough history for charts for this wave/mode.")
    else:
        nav_df = pd.concat(
            [hist["wave_nav"].rename("Wave NAV"), hist["bm_nav"].rename("Benchmark NAV")],
            axis=1
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
            st.line_chart((dd_df * 100.0))
        else:
            st.info("Drawdown chart unavailable.")

    st.divider()

    st.subheader("Coverage & Data Integrity")
    if cov.get("rows", 0) == 0:
        st.warning("No history returned for this wave/mode. Engine → CSV fallback attempted.")
    c1, c2, c3 = st.columns(3)
    c1.metric("History Rows", cov.get("rows", 0))
    c2.metric("Last Data Age (days)", cov.get("age_days", "—"))
    c3.metric("Completeness Score", fmt_num(cov.get("completeness_score", np.nan), 1))
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
        hold2["Google"] = hold2["Ticker"].apply(lambda t: f"https://www.google.com/finance/quote/{t}")
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

# -------------------------
# TAB: Alpha Heat Index (NEW)
# -------------------------
with tabs[1]:
    st.subheader("Alpha Heat Index (AHI) — 0 to 100 (NEW)")
    st.caption("AHI ranks each Wave’s raw alpha versus peers and scales it 0–100. 50 ≈ median. Informational intelligence layer (not a trading signal).")

    alpha_df = build_alpha_matrix(all_waves, mode=mode)
    ahi_df = build_ahi_table(alpha_df)

    if ahi_df is None or ahi_df.empty:
        st.info("AHI unavailable (no alpha history).")
    else:
        st.dataframe(ahi_df.round(1), use_container_width=True)

        # Show selected wave AHI badge (30D focus)
        ahi_30 = np.nan
        try:
            row = ahi_df[ahi_df["Wave"] == selected_wave]
            if not row.empty and "30D AHI" in row.columns:
                ahi_30 = float(row["30D AHI"].iloc[0])
        except Exception:
            ahi_30 = np.nan

        st.divider()
        show_metric_with_badge("Selected Wave — 30D AHI", fmt_num(ahi_30, 1), metric_key="ahi", metric_value=ahi_30)

        with st.expander("Raw alpha used (percent points)"):
            raw = alpha_df.copy()
            for col in ["1D Alpha", "30D Alpha", "60D Alpha", "365D Alpha"]:
                if col in raw.columns:
                    raw[col] = pd.to_numeric(raw[col], errors="coerce") * 100.0
            st.dataframe(raw.round(2), use_container_width=True)

# -------------------------
# TAB: Scoring Guide (NEW)
# -------------------------
with tabs[2]:
    st.subheader("Scoring Guide — Grade Charts (NEW)")
    st.caption("This tab makes the console self-explanatory. Grades come from metric_guide.py when present; otherwise it shows values without grades.")

    st.markdown("### Benchmark Truth & Difficulty")
    diff_val = float(difficulty.get("difficulty_vs_spy", np.nan))
    show_metric_with_badge("Difficulty vs SPY (proxy)", fmt_num(diff_val, 2), metric_key="difficulty_vs_spy", metric_value=diff_val)

    hhi_val = float(difficulty.get("hhi", np.nan))
    show_metric_with_badge("HHI (concentration)", fmt_num(hhi_val, 4), metric_key="hhi", metric_value=hhi_val)

    ent_val = float(difficulty.get("entropy", np.nan))
    show_metric_with_badge("Entropy (diversification)", fmt_num(ent_val, 3), metric_key="entropy", metric_value=ent_val)

    top_val = float(difficulty.get("top_weight", np.nan))
    show_metric_with_badge("Top Weight", fmt_pct(top_val, 2), metric_key="top_weight", metric_value=top_val)

    st.divider()
    st.markdown("### Wave Quality")
    show_metric_with_badge("WaveScore", fmt_score(ws_val), metric_key="wavescore", metric_value=ws_val)

    st.markdown("### Efficiency / Risk")
    show_metric_with_badge("Tracking Error (TE)", fmt_pct(te), metric_key="te", metric_value=te)
    show_metric_with_badge("Information Ratio (IR)", fmt_num(ir, 2), metric_key="ir", metric_value=ir)
    show_metric_with_badge("Max Drawdown (Wave)", fmt_pct(mdd), metric_key="maxdd", metric_value=mdd)

    st.divider()
    st.markdown("### System Status")
    st.write("metric_guide loaded:", mg is not None)
    if MG_IMPORT_ERROR is not None:
        st.code(str(MG_IMPORT_ERROR))
    st.write("decision_engine loaded:", de is not None)
    if DE_IMPORT_ERROR is not None:
        st.code(str(DE_IMPORT_ERROR))

# -------------------------
# TAB: Attribution (Engine vs Static Basket proxy)
# -------------------------
with tabs[3]:
    st.subheader("Attribution (Engine vs Static Basket Proxy)")
    st.caption("This is a **console-side** proxy: compares Wave returns to Benchmark returns (alpha).")
    if hist is None or hist.empty or len(hist) < 30:
        st.info("Not enough history for attribution proxy.")
    else:
        df = hist[["wave_ret", "bm_ret"]].dropna()
        df["alpha_ret"] = df["wave_ret"] - df["bm_ret"]
        st.metric("30D Alpha (approx)", fmt_pct(a30))
        st.metric("365D Alpha (approx)", fmt_pct(a365))
        st.line_chart((df[["alpha_ret"]] * 100.0).rename(columns={"alpha_ret": "Daily Alpha (%)"}))

# -------------------------
# TAB: Factor Decomposition
# -------------------------
with tabs[4]:
    st.subheader("Factor Decomposition (Light)")
    st.caption("Beta vs benchmark from daily returns.")
    if hist is None or hist.empty or len(hist) < 20:
        st.info("Not enough history.")
    else:
        b = beta_ols(hist["wave_ret"], hist["bm_ret"])
        st.metric("Beta vs Benchmark", fmt_num(b, 2))

# -------------------------
# TAB: Risk Lab
# -------------------------
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
        roll_df = pd.concat([(ra * 100.0).rename("Rolling 30D Alpha (%)"), rv.rename("Rolling Vol (20D)")], axis=1).dropna()
        st.line_chart(roll_df)

        ap = alpha_persistence(ra)
        st.metric("Alpha Persistence (Rolling 30D windows)", fmt_pct(ap))

# -------------------------
# TAB: Correlation
# -------------------------
with tabs[6]:
    st.subheader("Correlation (Daily Returns)")
    rets = {}
    for w in all_waves:
        h = compute_wave_history(w, mode=mode, days=min(days, 365))
        if h is not None and not h.empty and "wave_ret" in h.columns:
            rets[w] = h["wave_ret"]

    if len(rets) < 2:
        st.info("Not enough waves with history to compute correlations.")
    else:
        ret_df = pd.DataFrame(rets).dropna(how="all")
        corr = ret_df.corr()
        st.dataframe(corr, use_container_width=True)

# -------------------------
# TAB: Mode Proof (Elite)
# -------------------------
with tabs[7]:
    st.subheader("Mode Separation Proof (Side-by-Side)")
    st.caption("Same wave across modes — proves strategies are distinct.")
    modes_to_check = ["Standard", "Alpha-Minus-Beta", "Private Logic"]
    rows = []
    for m in modes_to_check:
        h = compute_wave_history(selected_wave, mode=m, days=min(days, 365))
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

# -------------------------
# TAB: Benchmark Truth (Elite)
# -------------------------
with tabs[8]:
    st.subheader("Benchmark Truth & Difficulty")
    st.write(f"**Snapshot:** {bm_id} · **Drift:** {bm_drift.upper()}")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Difficulty vs SPY (proxy)", fmt_num(difficulty.get("difficulty_vs_spy"), 2))
    c2.metric("HHI (conc.)", fmt_num(difficulty.get("hhi"), 4))
    c3.metric("Entropy", fmt_num(difficulty.get("entropy"), 3))
    c4.metric("Top Weight", fmt_pct(difficulty.get("top_weight"), 2))

    st.write("Interpretation (grades)")
    show_metric_with_badge("Difficulty vs SPY (proxy)", fmt_num(difficulty.get("difficulty_vs_spy"), 2), "difficulty_vs_spy", float(difficulty.get("difficulty_vs_spy", np.nan)))
    show_metric_with_badge("HHI", fmt_num(difficulty.get("hhi"), 4), "hhi", float(difficulty.get("hhi", np.nan)))
    show_metric_with_badge("Entropy", fmt_num(difficulty.get("entropy"), 3), "entropy", float(difficulty.get("entropy", np.nan)))
    show_metric_with_badge("Top Weight", fmt_pct(difficulty.get("top_weight"), 2), "top_weight", float(difficulty.get("top_weight", np.nan)))

    st.write("Benchmark Mix (normalized)")
    if bm_rows is None or bm_rows.empty:
        st.info("Benchmark mix table unavailable (engine may not expose it).")
    else:
        show = bm_rows.copy()
        show["Weight %"] = show["Weight"] * 100.0
        st.dataframe(show[["Ticker", "Weight %"]], use_container_width=True)

# -------------------------
# TAB: Drawdown Monitor (Elite)
# -------------------------
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

# -------------------------
# TAB: Alerts (Elite)
# -------------------------
with tabs[10]:
    st.subheader("Alerts & Flags")
    notes = build_alerts(selected_wave, mode, hist, cov, bm_drift, te, a30, mdd)
    for n in notes:
        st.markdown(f"- {n}")

# -------------------------
# TAB: WaveScore Leaderboard
# -------------------------
with tabs[11]:
    st.subheader("WaveScore Leaderboard (Console Approx.)")
    if ws_df is None or ws_df.empty:
        st.info("WaveScore unavailable (no history).")
    else:
        show = ws_df.copy()
        show["WaveScore"] = pd.to_numeric(show["WaveScore"], errors="coerce")
        show = show.sort_values("WaveScore", ascending=False, na_position="last").reset_index(drop=True)
        st.dataframe(show, use_container_width=True)

# -------------------------
# TAB: Governance Export (Elite)
# -------------------------
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

    perf_df = build_performance_matrix(all_waves, mode=mode, selected_wave=selected_wave, days=min(days, 365))
    if perf_df is not None and not perf_df.empty:
        st.download_button(
            "Download Performance Matrix (CSV)",
            data=perf_df.to_csv(index=False).encode("utf-8"),
            file_name=f"Performance_Matrix_{mode.replace(' ','_')}.csv",
            mime="text/csv",
            use_container_width=True,
        )

# -------------------------
# TAB: Vector OS Insight Layer
# -------------------------
with tabs[13]:
    st.subheader("Vector OS Insight Layer")
    if hist is None or hist.empty or len(hist) < 20:
        st.info("Not enough data for insights yet.")
    else:
        notes = build_alerts(selected_wave, mode, hist, cov, bm_drift, te, a30, mdd)
        for n in notes:
            st.markdown(f"- {n}")

# ============================================================
# Footer diagnostics (prevents silent deaths)
# ============================================================
with st.expander("System Diagnostics (if something looks off)"):
    st.write("Engine loaded:", we is not None)
    st.write("Engine import error:", str(ENGINE_IMPORT_ERROR) if ENGINE_IMPORT_ERROR else "None")
    st.write("metric_guide loaded:", mg is not None)
    st.write("metric_guide import error:", str(MG_IMPORT_ERROR) if MG_IMPORT_ERROR else "None")
    st.write("decision_engine loaded:", de is not None)
    st.write("decision_engine import error:", str(DE_IMPORT_ERROR) if DE_IMPORT_ERROR else "None")
    st.write(
        "Files present:",
        {p: os.path.exists(p) for p in ["wave_config.csv", "wave_weights.csv", "wave_history.csv", "list.csv", "waves_engine.py", "metric_guide.py", "decision_engine.py"]},
    )
    st.write("Selected:", {"wave": selected_wave, "mode": mode, "days": days})
    st.write("History shape:", None if hist is None else hist.shape)
    if hist is not None and not hist.empty:
        st.write("History columns:", list(hist.columns))
        st.write("History tail:", hist.tail(3))
        