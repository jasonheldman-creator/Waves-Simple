# app.py — WAVES Intelligence™ Institutional Console (Vector OS Edition)
# FULL PRODUCTION FILE (NO PATCHES)
# CLEAN + ORGANIZED MERGED BUILD (keeps ALL features; removes only duplicate/broken fragments)
#
# Keeps EVERYTHING from the last app:
#   • IC Summary (investor landing tab)
#   • Sticky chips (BM Snapshot, Coverage, Confidence, Regime/VIX, 30D/60D/365D, TE/IR, WaveScore/Rank)
#   • Definitions / Glossary drawers
#   • Overview (matrix + sorting + 365D toggle + alpha heatmap + charts + coverage + Top-10 links)
#   • Attribution (alpha proxy)
#   • Factor Decomposition (beta)
#   • Risk Lab (Sharpe/Sortino/VaR/CVaR/TE/IR + rolling alpha/vol + persistence)
#   • Correlation (daily return corr)
#   • Strategy Separation (mode proof)
#   • Benchmark Integrity (benchmark truth + difficulty proxy)
#   • Drawdown Monitor
#   • Diligence Flags
#   • WaveScore leaderboard (console approx)
#   • Governance Export pack (markdown + CSV)
#   • IC Notes
#   • Daily Movement / Volatility (decision_engine optional)
#   • Decision Intelligence (decision_engine optional)
#   • System Diagnostics (safe, expandable)
#
# Notes:
#   • Engine math NOT modified.
#   • Decision engine is optional. App never crashes if decision_engine.py missing.
#   • History loader is robust: engine functions → wave_history.csv fallback.
#   • This file is intentionally structured so a sophisticated reader can follow it.
#
# Required libs: streamlit, pandas, numpy
# Optional libs: yfinance, plotly

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
# Optional libs (safe)
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
# Global UI CSS (keeps sticky chips + cards + mobile spacing)
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

.waves-card {
  border: 1px solid rgba(255,255,255,0.10);
  border-radius: 14px;
  padding: 12px 14px;
  background: rgba(255,255,255,0.03);
}

@media (max-width: 700px) {
  .block-container { padding-left: 0.8rem; padding-right: 0.8rem; }
}
</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# Formatting helpers
# ============================================================
def fmt_pct(x: Any, digits: int = 2) -> str:
    """Input decimal (0.10) -> '10.00%'."""
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
# Section: Core Math / Analytics
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


def drawdown_series(nav: pd.Series) -> pd.Series:
    nav = safe_series(nav).astype(float)
    if len(nav) < 2:
        return pd.Series(dtype=float)
    peak = nav.cummax()
    return ((nav / peak) - 1.0).rename("drawdown")


def tracking_error(daily_wave: pd.Series, daily_bm: pd.Series) -> float:
    w = safe_series(daily_wave).astype(float)
    b = safe_series(daily_bm).astype(float)
    df = pd.concat([w.rename("w"), b.rename("b")], axis=1).dropna()
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
    # ============================================================
# Formatting helpers (display-safe)
# ============================================================
def fmt_pct(x: Any, digits: int = 2) -> str:
    try:
        if x is None:
            return "—"
        x = float(x)
        if math.isnan(x):
            return "—"
        return f"{x * 100:.{digits}f}%"
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
# Core math utilities (unchanged logic, just grouped)
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
# Optional data fetch (yfinance) — used for VIX/regime chip
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

    # Normalize yfinance shapes
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
# History Loader: engine → wave_history.csv fallback
# ============================================================
def _standardize_history(df: pd.DataFrame) -> pd.DataFrame:
    """
    Expected canonical output:
      index = datetime
      columns = wave_nav, bm_nav, wave_ret, bm_ret
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=["wave_nav", "bm_nav", "wave_ret", "bm_ret"])

    out = df.copy()

    # If date-like column exists, use it as index
    for dc in ["date", "Date", "timestamp", "Timestamp", "datetime", "Datetime"]:
        if dc in out.columns:
            out[dc] = pd.to_datetime(out[dc], errors="coerce")
            out = out.dropna(subset=[dc]).set_index(dc)
            break

    out.index = pd.to_datetime(out.index, errors="coerce")
    out = out[~out.index.isna()].sort_index()

    # Map common variants to canonical names
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

    # Synthesize returns if missing
    if "wave_ret" not in out.columns and "wave_nav" in out.columns:
        out["wave_ret"] = pd.to_numeric(out["wave_nav"], errors="coerce").pct_change()
    if "bm_ret" not in out.columns and "bm_nav" in out.columns:
        out["bm_ret"] = pd.to_numeric(out["bm_nav"], errors="coerce").pct_change()

    # Coerce numeric
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
    Primary: engine-provided history functions (if present).
    Fallback: wave_history.csv filtered by wave + mode.
    """
    if we is None:
        return history_from_csv(wave_name, mode, days)

    # Preferred: compute_history_nav
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

    # Alternate names
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


# ============================================================
# Discovery + Engine-access helpers (safe)
# ============================================================
@st.cache_data(show_spinner=False)
def get_all_waves_safe() -> List[str]:
    # If engine is missing, attempt to discover from CSVs
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

    # Engine pathway
    if hasattr(we, "get_all_waves"):
        try:
            waves = we.get_all_waves()
            if isinstance(waves, (list, tuple)):
                waves = [str(x) for x in waves]
                return sorted([w for w in waves if w and w.lower() != "nan"])
        except Exception:
            pass

    # Fallback: benchmark table has Wave column
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
    """
    Returns columns: Ticker, Name, Weight (normalized to 1.0)
    """
    # Engine holdings
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

                out = (
                    out[["Ticker", "Name", "Weight"]]
                    .sort_values("Weight", ascending=False)
                    .reset_index(drop=True)
                )
                return out
        except Exception:
            pass

    # CSV fallback: wave_weights.csv (Wave, Ticker, Weight)
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
# Benchmark Truth: Snapshot + Drift + Difficulty Proxy
# ============================================================
def _normalize_bm_rows(bm_rows: pd.DataFrame) -> pd.DataFrame:
    """
    Normalizes a benchmark mix table to:
      columns: Ticker, Weight
      - uppercases tickers
      - sums duplicates
      - renormalizes weights to 1.0
    """
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
    """
    Creates a stable fingerprint for a wave's benchmark mix.
    Used to detect drift (benchmark changing mid-session).
    """
    try:
        if bm_mix_df is None or bm_mix_df.empty:
            return "BM-NA"

        rows = bm_mix_df.copy()
        if "Wave" in rows.columns:
            rows = rows[rows["Wave"].astype(str) == str(wave_name)].copy()

        if rows is None or rows.empty or ("Ticker" not in rows.columns) or ("Weight" not in rows.columns):
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
    """
    Tracks benchmark snapshot drift per (wave, mode) in session_state.
    """
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
    """
    Heuristic "difficulty vs SPY" proxy based on concentration/diversification of benchmark:
      - HHI higher => more concentrated
      - entropy higher => more diversified
    Output values are bounded for safe display.
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

        # calibrated, bounded heuristic
        conc_pen = (out["hhi"] - 0.06) * 180.0
        ent_bonus = (out["entropy"] - 2.6) * -12.0
        raw = conc_pen + ent_bonus
        out["difficulty_vs_spy"] = float(np.clip(raw, -25.0, 25.0))
        return out
    except Exception:
        return out


# ============================================================
# Coverage / Data Integrity Report (Trust Cue)
# ============================================================
def _business_day_range(start_dt: pd.Timestamp, end_dt: pd.Timestamp) -> pd.DatetimeIndex:
    try:
        return pd.date_range(start=start_dt.normalize(), end=end_dt.normalize(), freq="B")
    except Exception:
        return pd.DatetimeIndex([])


def coverage_report(hist: pd.DataFrame) -> Dict[str, Any]:
    """
    Computes a simple integrity report for the loaded history window:
      - missing business days between first/last
      - age in days since last datapoint
      - completeness score (0–100)
      - flags for obvious issues
    """
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

        # flags
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
    Returns (confidence_level, reason) where confidence_level ∈ {High, Medium, Low}.
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
# WaveScore (Console Approximation) + Leaderboard Support
# ============================================================
@st.cache_data(show_spinner=False)
def compute_wavescore_for_all_waves(all_waves: List[str], mode: str, days: int = 365) -> pd.DataFrame:
    """
    Console-side ranking approximation (NOT the locked WAVESCORE™ spec).
    Uses: IR, TE, max drawdown, hit rate.
    """
    rows: List[Dict[str, Any]] = []

    for wave in all_waves:
        hist = compute_wave_history(wave, mode=mode, days=days)
        hist = _standardize_history(hist)

        if hist is None or hist.empty or len(hist) < 20:
            rows.append(
                {"Wave": wave, "WaveScore": np.nan, "Grade": "N/A", "IR_365D": np.nan, "Alpha_365D": np.nan}
            )
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

        # scaled components
        rq = float(np.clip((np.nan_to_num(ir) / 1.5), 0.0, 1.0) * 25.0)
        rc = float(np.clip(1.0 - (abs(np.nan_to_num(mdd_wave)) / 0.35), 0.0, 1.0) * 25.0)
        co = float(np.clip(np.nan_to_num(hit_rate), 0.0, 1.0) * 15.0)
        rs = float(np.clip(1.0 - (abs(np.nan_to_num(te)) / 0.25), 0.0, 1.0) * 15.0)
        tr = 10.0  # fixed governance/transparency placeholder (demo)

        total = float(np.clip(rq + rc + co + rs + tr, 0.0, 100.0))
        rows.append(
            {"Wave": wave, "WaveScore": total, "Grade": _grade_from_score(total), "IR_365D": ir, "Alpha_365D": alpha_365}
        )

    df = pd.DataFrame(rows) if rows else pd.DataFrame()
    if df.empty:
        return df

    df["WaveScore"] = pd.to_numeric(df["WaveScore"], errors="coerce")
    df["Rank"] = df["WaveScore"].rank(ascending=False, method="min")
    return df.sort_values("Wave").reset_index(drop=True)


# ============================================================
# Performance Matrix (Returns + Alpha) — percent points
# ============================================================
@st.cache_data(show_spinner=False)
def build_performance_matrix(all_waves: List[str], mode: str, selected_wave: str, days: int = 365) -> pd.DataFrame:
    """
    Builds a scan-first matrix:
      1D / 30D / 60D / 365D returns and alpha (Wave - Benchmark)
    Returned values are % points (e.g., 1.23 means 1.23%).
    """
    rows: List[Dict[str, Any]] = []

    for w in all_waves:
        h = compute_wave_history(w, mode=mode, days=days)
        h = _standardize_history(h)

        if h is None or h.empty or len(h) < 2:
            rows.append(
                {
                    "Wave": w,
                    "1D Return": np.nan, "1D Alpha": np.nan,
                    "30D Return": np.nan, "30D Alpha": np.nan,
                    "60D Return": np.nan, "60D Alpha": np.nan,
                    "365D Return": np.nan, "365D Alpha": np.nan,
                    "Rows": 0,
                }
            )
            continue

        nav_w = h["wave_nav"]
        nav_b = h["bm_nav"]

        # 1D
        r1 = np.nan
        a1 = np.nan
        if len(nav_w) >= 2 and len(nav_b) >= 2:
            r1 = float(nav_w.iloc[-1] / nav_w.iloc[-2] - 1.0)
            b1 = float(nav_b.iloc[-1] / nav_b.iloc[-2] - 1.0)
            a1 = r1 - b1

        # 30/60/365
        r30 = ret_from_nav(nav_w, min(30, len(nav_w)))
        b30 = ret_from_nav(nav_b, min(30, len(nav_b)))
        a30 = r30 - b30

        r60 = ret_from_nav(nav_w, min(60, len(nav_w)))
        b60 = ret_from_nav(nav_b, min(60, len(nav_b)))
        a60 = r60 - b60

        r365 = ret_from_nav(nav_w, min(365, len(nav_w)))
        b365 = ret_from_nav(nav_b, min(365, len(nav_b)))
        a365 = r365 - b365

        rows.append(
            {
                "Wave": w,
                "1D Return": r1, "1D Alpha": a1,
                "30D Return": r30, "30D Alpha": a30,
                "60D Return": r60, "60D Alpha": a60,
                "365D Return": r365, "365D Alpha": a365,
                "Rows": int(len(h)),
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # percent points
    for c in [c for c in df.columns if ("Return" in c) or ("Alpha" in c)]:
        df[c] = pd.to_numeric(df[c], errors="coerce") * 100.0

    # keep selected on top
    if "Wave" in df.columns and selected_wave in set(df["Wave"]):
        top = df[df["Wave"] == selected_wave]
        rest = df[df["Wave"] != selected_wave]
        df = pd.concat([top, rest], axis=0)

    return df.reset_index(drop=True)


# ============================================================
# Alpha Heatmap (All Waves × timeframe)
# ============================================================
@st.cache_data(show_spinner=False)
def build_alpha_matrix(all_waves: List[str], mode: str, days: int = 365) -> pd.DataFrame:
    """
    Alpha matrix in DECIMAL (not percent points):
      columns: 1D Alpha, 30D Alpha, 60D Alpha, 365D Alpha
    (You can multiply by 100 when displaying in UI.)
    """
    rows: List[Dict[str, Any]] = []

    for wname in all_waves:
        hist = compute_wave_history(wname, mode=mode, days=days)
        hist = _standardize_history(hist)

        if hist is None or hist.empty or len(hist) < 2:
            rows.append(
                {"Wave": wname, "1D Alpha": np.nan, "30D Alpha": np.nan, "60D Alpha": np.nan, "365D Alpha": np.nan}
            )
            continue

        nav_w = hist["wave_nav"]
        nav_b = hist["bm_nav"]

        # 1D alpha
        a1 = np.nan
        if len(nav_w) >= 2 and len(nav_b) >= 2:
            a1 = float(nav_w.iloc[-1] / nav_w.iloc[-2] - 1.0) - float(nav_b.iloc[-1] / nav_b.iloc[-2] - 1.0)

        # multi-day alpha
        a30 = ret_from_nav(nav_w, min(30, len(nav_w))) - ret_from_nav(nav_b, min(30, len(nav_b)))
        a60 = ret_from_nav(nav_w, min(60, len(nav_w))) - ret_from_nav(nav_b, min(60, len(nav_b)))
        a365 = ret_from_nav(nav_w, min(365, len(nav_w))) - ret_from_nav(nav_b, min(365, len(nav_b)))

        rows.append({"Wave": wname, "1D Alpha": a1, "30D Alpha": a30, "60D Alpha": a60, "365D Alpha": a365})

    return pd.DataFrame(rows).sort_values("Wave").reset_index(drop=True)


def plot_alpha_heatmap(alpha_df: pd.DataFrame, title: str):
    """
    Plotly heatmap for alpha matrix.
    Uses symmetric color scaling around 0 for readability.
    """
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

    fig = go.Figure(data=go.Heatmap(z=z, x=x, y=y, zmin=-v, zmax=v, colorbar=dict(title="Alpha (decimal)")))
    fig.update_layout(
        title=title,
        height=min(950, 260 + 22 * max(12, len(y))),
        margin=dict(l=80, r=40, t=60, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)
    # ============================================================
# UI / Rendering Helpers
#   - Sticky chips
#   - Quote links
#   - Table styling (heat)
#   - Definitions drawers
#   - Decision ctx builders
#   - Alerts / diligence flags
# ============================================================

def chip(label: str) -> str:
    safe = (label or "").replace("<", "").replace(">", "")
    return f'<span class="waves-chip">{safe}</span>'


def render_sticky(chips: List[str]) -> None:
    html = '<div class="waves-sticky">' + "".join([chip(c) for c in chips if c]) + "</div>"
    st.markdown(html, unsafe_allow_html=True)


def google_quote_link(ticker: str) -> str:
    """
    Uses Google Finance. Works for most tickers.
    (We do not force an exchange suffix because tickers can be crypto/ETF/dots/etc.)
    """
    t = (ticker or "").upper().strip()
    if not t:
        return ""
    return f"https://www.google.com/finance/quote/{t}"


def safe_mode_list() -> List[str]:
    return ["Standard", "Alpha-Minus-Beta", "Private Logic"]


# ============================================================
# Styling helpers: % matrix + green/red heat
# ============================================================
def _heat_color(val: Any) -> str:
    """
    Returns a background color for positive/negative cells.
    Works on percent-point tables (already multiplied by 100).
    """
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
    """
    Applies heat coloring and formats % columns.
    """
    if df is None or df.empty:
        return df.style

    cols = [c for c in df.columns if ("Return" in c or "Alpha" in c or c.endswith("%"))]
    sty = df.style
    for c in cols:
        if c in df.columns:
            sty = sty.applymap(_heat_color, subset=[c])
            # Keep format stable, even if NaN
            try:
                sty = sty.format({c: lambda x: "—" if pd.isna(x) else f"{x:.2f}%"})
            except Exception:
                pass
    return sty


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
    "Coverage Score": "A 0–100 heuristic of data completeness + freshness (missing business days and staleness reduce score).",
    "Difficulty vs SPY": "Heuristic based on benchmark concentration/diversification. Bounded proxy for ‘harder/easier’ to beat vs a broad index mix.",
    "WaveScore": "Console-side approximation (NOT the locked WAVESCORE™ spec). Uses IR, drawdown, hit-rate, TE proxy.",
    "Decision Intelligence": "Translator layer: actions/watch/notes from observable analytics (not advice).",
    "Mode Proof": "Demonstrates that Standard / Alpha-Minus-Beta / Private Logic load independently (history + snapshots).",
}


def render_definitions(keys: List[str], title: str = "Definitions") -> None:
    with st.expander(title):
        for k in keys:
            if k in GLOSSARY:
                st.markdown(f"**{k}:** {GLOSSARY[k]}")
            else:
                st.markdown(f"**{k}:** (definition not found)")


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
    """
    Non-advice diligence flags: highlights potential anomalies or checks.
    """
    notes: List[str] = []
    try:
        if cov.get("flags"):
            notes.append("Data Integrity: " + "; ".join(cov["flags"]))

        if bm_drift != "stable":
            notes.append("Benchmark Drift: snapshot changed in-session (freeze benchmark mix for demos / IC).")

        if math.isfinite(a30) and abs(a30) >= 0.08:
            notes.append("Large 30D alpha: verify benchmark mix + missing days (big alpha can be real or coverage-driven).")

        if math.isfinite(te) and te >= 0.20:
            notes.append("High tracking error: active risk elevated vs benchmark.")

        if math.isfinite(mdd) and mdd <= -0.25:
            notes.append("Deep drawdown: consider stronger SmartSafe posture in stress regimes.")

        if hist is not None and not hist.empty:
            age = cov.get("age_days", 0)
            if age is not None and isinstance(age, (int, float)) and age >= 5:
                notes.append("Stale history: last datapoint is >=5 days old (check engine writes).")

        if not notes:
            notes.append("No major anomalies detected on this window.")

        return notes
    except Exception:
        return ["Alert system error (non-fatal)."]


# ============================================================
# Decision Engine Context Builder (single source of truth)
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
    """
    Stable ctx payload for decision_engine.py (if present).
    """
    return {
        "wave_name": wave,
        "wave": wave,
        "mode": mode,
        "bm_snapshot": bm_id,
        "bm_drift": bm_drift,
        "rows": cov.get("rows"),
        "first_date": cov.get("first_date"),
        "last_date": cov.get("last_date"),
        "age_days": cov.get("age_days"),
        "missing_bdays": cov.get("missing_bdays"),
        "missing_pct": cov.get("missing_pct"),
        "completeness_score": cov.get("completeness_score"),
        "flags": cov.get("flags", []),
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
# Governance Export Pack (IC / Board Ready) — Markdown Builder
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
# MAIN APP UI
#   - Sidebar controls
#   - Discovery + history load
#   - Sticky chips (top)
#   - Tabs (IC Summary → Decision Intelligence)
# ============================================================

st.title("WAVES Intelligence™ Institutional Console")

# --------------------------------
# Early warnings (non-fatal)
# --------------------------------
if ENGINE_IMPORT_ERROR is not None:
    st.error("Engine import failed. The app will use CSV fallbacks where possible.")
    st.code(str(ENGINE_IMPORT_ERROR))

if DECISION_IMPORT_ERROR is not None:
    st.warning("Decision Engine import issue (non-fatal). Decision tabs will fallback.")
    st.code(str(DECISION_IMPORT_ERROR))

# --------------------------------
# Discover waves
# --------------------------------
all_waves = get_all_waves_safe()
all_waves = sorted([str(w) for w in all_waves if str(w).strip() and str(w).lower() != "nan"])

if not all_waves:
    st.warning("No waves discovered yet. If unexpected, check engine import + CSV files.")
    with st.expander("System Diagnostics"):
        st.write(
            "Files present:",
            {p: os.path.exists(p) for p in ["wave_config.csv", "wave_weights.csv", "wave_history.csv", "list.csv", "waves_engine.py", "decision_engine.py"]},
        )
        st.write("Engine loaded:", we is not None)
        st.write("Engine import error:", str(ENGINE_IMPORT_ERROR) if ENGINE_IMPORT_ERROR else "None")
    st.stop()

# --------------------------------
# Sidebar controls
# --------------------------------
with st.sidebar:
    st.header("Controls")

    mode = st.selectbox("Mode", safe_mode_list(), index=0)
    selected_wave = st.selectbox("Wave", all_waves, index=0)

    days = st.slider(
        "History window (days)",
        min_value=90,
        max_value=1500,
        value=365,
        step=30,
        help="History length used for charts + calculations. Falls back to wave_history.csv if engine history is missing.",
    )

    show_365_default = st.toggle("Show 365D by default", value=False)
    st.caption("Tip: keep 365D hidden for clean scans; toggle on for diligence.")

    # Lightweight definitions always available
    with st.expander("Definitions (Quick)", expanded=False):
        render_definitions(
            keys=[
                "Return", "Alpha",
                "Tracking Error (TE)", "Information Ratio (IR)",
                "Max Drawdown (MaxDD)",
                "Benchmark Snapshot / Drift",
                "Coverage Score",
                "WaveScore",
                "Decision Intelligence",
            ],
            title="Definitions",
        )

# --------------------------------
# Load benchmark mix + snapshot
# --------------------------------
bm_mix = get_benchmark_mix()
bm_id = benchmark_snapshot_id(selected_wave, bm_mix)
bm_drift = benchmark_drift_status(selected_wave, mode, bm_id)

# --------------------------------
# Load + standardize history
# --------------------------------
hist = compute_wave_history(selected_wave, mode=mode, days=days)
hist = _standardize_history(hist)

cov = coverage_report(hist)

# --------------------------------
# Precompute stats used across tabs
# --------------------------------
nav_w = hist["wave_nav"] if (hist is not None and not hist.empty and "wave_nav" in hist.columns) else pd.Series(dtype=float)
nav_b = hist["bm_nav"] if (hist is not None and not hist.empty and "bm_nav" in hist.columns) else pd.Series(dtype=float)
ret_w = hist["wave_ret"] if (hist is not None and not hist.empty and "wave_ret" in hist.columns) else pd.Series(dtype=float)
ret_b = hist["bm_ret"] if (hist is not None and not hist.empty and "bm_ret" in hist.columns) else pd.Series(dtype=float)

mdd = max_drawdown(nav_w) if len(nav_w) >= 2 else np.nan
mdd_b = max_drawdown(nav_b) if len(nav_b) >= 2 else np.nan

r30 = ret_from_nav(nav_w, min(30, len(nav_w))) if len(nav_w) >= 2 else np.nan
b30 = ret_from_nav(nav_b, min(30, len(nav_b))) if len(nav_b) >= 2 else np.nan
a30 = (r30 - b30) if (math.isfinite(r30) and math.isfinite(b30)) else np.nan

r60 = ret_from_nav(nav_w, min(60, len(nav_w))) if len(nav_w) >= 2 else np.nan
b60 = ret_from_nav(nav_b, min(60, len(nav_b))) if len(nav_b) >= 2 else np.nan
a60 = (r60 - b60) if (math.isfinite(r60) and math.isfinite(b60)) else np.nan

r365 = ret_from_nav(nav_w, min(365, len(nav_w))) if len(nav_w) >= 2 else np.nan
b365 = ret_from_nav(nav_b, min(365, len(nav_b))) if len(nav_b) >= 2 else np.nan
a365 = (r365 - b365) if (math.isfinite(r365) and math.isfinite(b365)) else np.nan

te = tracking_error(ret_w, ret_b) if (len(ret_w) >= 2 and len(ret_b) >= 2) else np.nan
ir = information_ratio(nav_w, nav_b, te) if (len(nav_w) >= 2 and len(nav_b) >= 2) else np.nan

# --------------------------------
# VIX / Regime
# --------------------------------
vix_val = get_vix_value()
regime = regime_from_vix(vix_val)

# --------------------------------
# WaveScore + leaderboard
# --------------------------------
ws_df = compute_wavescore_for_all_waves(all_waves, mode=mode, days=min(days, 365))
ws_val = np.nan
rank = None
if isinstance(ws_df, pd.DataFrame) and (not ws_df.empty) and ("Wave" in ws_df.columns) and ("WaveScore" in ws_df.columns):
    try:
        row = ws_df[ws_df["Wave"].astype(str) == str(selected_wave)]
        if not row.empty and pd.notna(row["WaveScore"].iloc[0]):
            ws_val = float(row["WaveScore"].iloc[0])
        # rank by WaveScore desc
        wss = ws_df.copy()
        wss["WaveScore"] = pd.to_numeric(wss["WaveScore"], errors="coerce")
        wss = wss.sort_values("WaveScore", ascending=False, na_position="last").reset_index(drop=True)
        idx = wss.index[wss["Wave"].astype(str) == str(selected_wave)].tolist()
        rank = int(idx[0] + 1) if idx else None
    except Exception:
        pass

ws_grade = _grade_from_score(ws_val) if math.isfinite(ws_val) else "N/A"

# --------------------------------
# Benchmark difficulty proxy (Benchmark Truth)
# --------------------------------
bm_rows = pd.DataFrame()
try:
    if bm_mix is not None and not bm_mix.empty and "Wave" in bm_mix.columns:
        tmp = bm_mix[bm_mix["Wave"].astype(str) == str(selected_wave)].copy()
        if "Ticker" in tmp.columns and "Weight" in tmp.columns:
            bm_rows = _normalize_bm_rows(tmp[["Ticker", "Weight"]].copy())
except Exception:
    bm_rows = pd.DataFrame()

difficulty = benchmark_difficulty_proxy(bm_rows)

# --------------------------------
# Confidence meter
# --------------------------------
conf_level, conf_reason = confidence_from_integrity(cov, bm_drift)

# --------------------------------
# Sticky chip row (top-of-page)
# --------------------------------
chips: List[str] = [
    f"BM Snapshot: {bm_id} · {'Stable' if bm_drift=='stable' else 'DRIFT'}",
    f"Coverage: {fmt_num(cov.get('completeness_score', np.nan), 1)} / 100",
    f"Rows: {cov.get('rows','—')} · Age: {cov.get('age_days','—')}",
    f"Confidence: {conf_level}",
    f"Regime: {regime}",
    f"VIX: {fmt_num(vix_val, 1) if math.isfinite(vix_val) else '—'}",
    f"30D α: {fmt_pct(a30)} · 30D r: {fmt_pct(r30)}",
    f"60D α: {fmt_pct(a60)} · 60D r: {fmt_pct(r60)}",
    f"365D α: {fmt_pct(a365)} · 365D r: {fmt_pct(r365)}",
    f"TE: {fmt_pct(te)} · IR: {fmt_num(ir, 2)}",
    f"WaveScore: {fmt_score(ws_val)} ({ws_grade}) · Rank: {rank if rank else '—'}",
]
render_sticky(chips)

st.caption("Observational analytics only (not trading advice).")

# ============================================================
# Tabs (unchanged feature set; organized)
# ============================================================
tabs = st.tabs([
    "IC Summary",
    "Overview",
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
# TAB 0: IC Summary (investor / diligence landing page)
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
            st.info("Decision Engine not available (generate_decisions missing). Showing safe fallbacks.")
            d = {"actions": [], "watch": [], "notes": []}
        else:
            try:
                d = generate_decisions(ctx)
                if not isinstance(d, dict):
                    d = {"actions": [str(d)], "watch": [], "notes": []}
            except Exception as e:
                d = {"actions": [f"Decision engine error: {e}"], "watch": [], "notes": []}

        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**Action**")
            for x in (d.get("actions") or []):
                st.write(f"• {x}")
        with c2:
            st.markdown("**Watch**")
            for x in (d.get("watch") or []):
                st.write(f"• {x}")
        with c3:
            st.markdown("**Notes**")
            for x in (d.get("notes") or []):
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

        r1, r2, r3 = st.columns(3)
        r1.metric("30D Return", fmt_pct(r30))
        r2.metric("30D Alpha", fmt_pct(a30))
        r3.metric("Max Drawdown", fmt_pct(mdd))

        r4, r5, r6 = st.columns(3)
        r4.metric("Tracking Error (TE)", fmt_pct(te))
        r5.metric("Information Ratio (IR)", fmt_num(ir, 2))
        r6.metric("WaveScore", f"{fmt_score(ws_val)} ({ws_grade})")

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

    render_definitions(
        keys=[
            "Alpha", "Tracking Error (TE)", "Information Ratio (IR)",
            "Max Drawdown (MaxDD)", "Benchmark Snapshot / Drift",
            "Coverage Score", "Decision Intelligence",
        ],
        title="Definitions (IC Summary)",
    )
    # ============================================================
# TAB 1: Overview — matrix + heatmap + charts + coverage + holdings
# ============================================================
with tabs[1]:
    st.subheader("Overview — All Waves (Performance Matrix + Heatmap + Selected Deep View)")

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

        # optional sort
        if sort_by != "Selected First (default)":
            key_map = {"30D Alpha": "30D Alpha", "60D Alpha": "60D Alpha", "30D Return": "30D Return", "Rows": "Rows"}
            if sort_by in key_map and key_map[sort_by] in df.columns:
                df = df.sort_values(key_map[sort_by], ascending=False, na_position="last")
            elif sort_by == "WaveScore":
                if ws_df is not None and not ws_df.empty and "Wave" in ws_df.columns:
                    df = df.merge(ws_df[["Wave", "WaveScore"]], on="Wave", how="left")
                    df = df.sort_values("WaveScore", ascending=False, na_position="last").drop(columns=["WaveScore"], errors="ignore")

            # keep selected wave on top after sort
            if selected_wave in set(df["Wave"]):
                top = df[df["Wave"] == selected_wave]
                rest = df[df["Wave"] != selected_wave]
                df = pd.concat([top, rest], axis=0)

        # hide 365D unless requested
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
    render_definitions(keys=["Alpha"], title="Definitions (Heatmap)")

    st.divider()

    st.subheader("Selected Wave — NAV vs Benchmark")
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
# TAB 2: Attribution
# ============================================================
with tabs[2]:
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
# TAB 3: Factor Decomposition
# ============================================================
with tabs[3]:
    st.subheader("Factor Decomposition (Light)")
    st.caption("Beta vs benchmark from daily returns.")

    if hist is None or hist.empty or len(hist) < 20:
        st.info("Not enough history.")
    else:
        b = beta_ols(hist["wave_ret"], hist["bm_ret"])
        st.metric("Beta vs Benchmark", fmt_num(b, 2))

    render_definitions(keys=["Return"], title="Definitions (Factor)")

# ============================================================
# TAB 4: Risk Lab
# ============================================================
with tabs[4]:
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
# TAB 5: Correlation
# ============================================================
with tabs[5]:
    st.subheader("Correlation (Daily Returns)")
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

# ============================================================
# TAB 6: Strategy Separation (Mode Proof)
# ============================================================
with tabs[6]:
    st.subheader("Strategy Separation (Mode Proof)")
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
# TAB 7: Benchmark Integrity (Benchmark Truth)
# ============================================================
with tabs[7]:
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
# TAB 8: Drawdown Monitor
# ============================================================
with tabs[8]:
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
# TAB 9: Diligence Flags (Alerts)
# ============================================================
with tabs[9]:
    st.subheader("Diligence Flags")
    notes = build_alerts(selected_wave, mode, hist, cov, bm_drift, te, a30, mdd)
    for n in notes:
        st.markdown(f"- {n}")
    st.divider()
    st.write(f"**Confidence:** {conf_level} — {conf_reason}")

# ============================================================
# TAB 10: WaveScore Leaderboard
# ============================================================
with tabs[10]:
    st.subheader("WaveScore Leaderboard (Console Approx.)")
    if ws_df is None or ws_df.empty:
        st.info("WaveScore unavailable (no history).")
    else:
        show = ws_df.copy()
        show["WaveScore"] = pd.to_numeric(show["WaveScore"], errors="coerce")
        show = show.sort_values("WaveScore", ascending=False, na_position="last").reset_index(drop=True)
        st.dataframe(show, use_container_width=True)

# ============================================================
# TAB 11: Governance Export
# ============================================================
with tabs[11]:
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
# TAB 12: IC Notes (formerly Vector OS Insight Layer)
# ============================================================
with tabs[12]:
    st.subheader("IC Notes")
    if hist is None or hist.empty or len(hist) < 20:
        st.info("Not enough data for insights yet.")
    else:
        notes = build_alerts(selected_wave, mode, hist, cov, bm_drift, te, a30, mdd)
        for n in notes:
            st.markdown(f"- {n}")

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
            activity = {"headline": "Daily Movement error", "what_changed": [], "why": [], "results": [], "checks": [str(e)]}

        if not isinstance(activity, dict):
            st.write(activity)
        else:
            if activity.get("headline"):
                st.write(f"**{activity.get('headline')}**")

            st.markdown("### What changed")
            for s in activity.get("what_changed", []) or []:
                st.write(f"• {s}")

            st.markdown("### Why it changed")
            for s in activity.get("why", []) or []:
                st.write(f"• {s}")

            st.markdown("### Results")
            for s in activity.get("results", []) or []:
                st.write(f"• {s}")

            st.markdown("### Checks / Confidence")
            for s in activity.get("checks", []) or []:
                st.write(f"• {s}")

            with st.expander("Context (ctx) used for this explanation"):
                st.json(ctx)

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

# ============================================================
# Footer diagnostics (kept; organized)
# ============================================================
with st.expander("System Diagnostics (if something looks off)"):
    st.write("Engine loaded:", we is not None)
    st.write("Engine import error:", str(ENGINE_IMPORT_ERROR) if ENGINE_IMPORT_ERROR else "None")
    st.write("Decision engine loaded:", (generate_decisions is not None) or (build_daily_wave_activity is not None))
    st.write("Decision import error:", str(DECISION_IMPORT_ERROR) if DECISION_IMPORT_ERROR else "None")
    st.write(
        "Files present:",
        {p: os.path.exists(p) for p in ["wave_config.csv", "wave_weights.csv", "wave_history.csv", "list.csv", "waves_engine.py", "decision_engine.py"]},
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