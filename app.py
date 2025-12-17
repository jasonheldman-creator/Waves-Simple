# app.py — WAVES Intelligence™ Institutional Console (Vector OS Edition)
# FULL PRODUCTION FILE (NO PATCHES) — SINGLE-FILE ONLY
#
# vNEXT — CANONICAL COHESION LOCK + IC ONE-PAGER + FIDELITY INSPECTOR + AI EXPLAIN
#         + COMPARATOR + BETA RELIABILITY + DIAGNOSTICS (ALWAYS BOOTS)
#
# Boot-safety rules:
#   • ONE canonical dataset per selected Wave+Mode: hist_sel (source-of-truth)
#   • Guarded optional imports (yfinance / plotly)
#   • Every major section wrapped so the app still boots if a panel fails
#   • Diagnostics tab always available (engine import errors, empty history, etc.)
#
# Canonical rule:
#   hist_sel = _standardize_history(compute_wave_history(selected_wave, mode, days))
#   Every metric shown uses hist_sel (no duplicate math / no crisscross).

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
# Optional libs (guarded)
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
# Streamlit config
# ============================================================
st.set_page_config(
    page_title="WAVES Intelligence™ Console",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# Feature Flags (demo-safe)
# ============================================================
ENABLE_SCORECARD = True
ENABLE_FIDELITY_INSPECTOR = True
ENABLE_AI_EXPLAIN = True
ENABLE_COMPARATOR = True
ENABLE_YFINANCE_CHIPS = True  # auto-disables if yf missing


# ============================================================
# Global UI CSS
# ============================================================
st.markdown(
    """
<style>
.block-container { padding-top: 0.85rem; padding-bottom: 2.0rem; }
.waves-big-wave {
  font-size: 2.2rem; font-weight: 850; letter-spacing: 0.2px;
  line-height: 2.45rem; margin: 0.1rem 0 0.15rem 0;
}
.waves-subhead { opacity: 0.85; font-size: 1.05rem; margin: 0 0 0.6rem 0; }

.waves-sticky {
  position: sticky; top: 0; z-index: 999;
  backdrop-filter: blur(10px); -webkit-backdrop-filter: blur(10px);
  padding: 10px 12px; margin: 0 0 12px 0;
  border-radius: 14px; border: 1px solid rgba(255,255,255,0.10);
  background: rgba(10, 15, 28, 0.66);
}
.waves-chip {
  display: inline-block; padding: 8px 12px; margin: 6px 8px 0 0;
  border-radius: 999px; border: 1px solid rgba(255,255,255,0.12);
  background: rgba(255,255,255,0.04);
  font-size: 0.92rem; line-height: 1.05rem; white-space: nowrap;
}
.waves-card {
  border: 1px solid rgba(255,255,255,0.10);
  border-radius: 14px; padding: 12px 14px;
  background: rgba(255,255,255,0.03);
}
.waves-tile {
  border: 1px solid rgba(255,255,255,0.10);
  border-radius: 14px; padding: 12px 14px;
  background: rgba(255,255,255,0.03);
  min-height: 86px;
}
.waves-tile-label { opacity: 0.85; font-size: 0.90rem; margin-bottom: 0.25rem; }
.waves-tile-value { font-size: 1.55rem; font-weight: 850; line-height: 1.75rem; }
.waves-tile-sub { opacity: 0.75; font-size: 0.92rem; margin-top: 0.20rem; }

hr { border: none; border-top: 1px solid rgba(255,255,255,0.08); margin: 0.8rem 0; }

@media (max-width: 700px) {
  .block-container { padding-left: 0.8rem; padding-right: 0.8rem; }
  .waves-big-wave { font-size: 1.75rem; line-height: 2.0rem; }
  .waves-tile-value { font-size: 1.30rem; line-height: 1.55rem; }
}
</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# Helpers: formatting / safety
# ============================================================
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


def fmt_pct(x: Any, digits: int = 2) -> str:
    v = safe_float(x)
    if not math.isfinite(v):
        return "—"
    return f"{v*100:0.{digits}f}%"


def fmt_num(x: Any, digits: int = 2) -> str:
    v = safe_float(x)
    if not math.isfinite(v):
        return "—"
    return f"{v:.{digits}f}"


def fmt_int(x: Any) -> str:
    try:
        if x is None:
            return "—"
        return str(int(x))
    except Exception:
        return "—"


# ============================================================
# Basic return/risk math
# ============================================================
def ret_from_nav(nav: pd.Series, window: int) -> float:
    nav = safe_series(nav).astype(float).dropna()
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
    r = safe_series(daily_ret).astype(float).dropna()
    if len(r) < 2:
        return float("nan")
    return float(r.std() * np.sqrt(252))


def max_drawdown(nav: pd.Series) -> float:
    nav = safe_series(nav).astype(float).dropna()
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
    nw = safe_series(nav_wave).astype(float).dropna()
    nb = safe_series(nav_bm).astype(float).dropna()
    if len(nw) < 2 or len(nb) < 2:
        return float("nan")
    if te is None or (isinstance(te, float) and (math.isnan(te) or te <= 0)):
        return float("nan")
    excess = ret_from_nav(nw, len(nw)) - ret_from_nav(nb, len(nb))
    return float(excess / te)


def sharpe_ratio(daily_ret: pd.Series, rf_annual: float = 0.0) -> float:
    r = safe_series(daily_ret).astype(float).dropna()
    if len(r) < 20:
        return float("nan")
    rf_daily = rf_annual / 252.0
    ex = r - rf_daily
    vol = float(ex.std())
    if not math.isfinite(vol) or vol <= 0:
        return float("nan")
    return float(ex.mean() / vol * np.sqrt(252))


def downside_deviation(daily_ret: pd.Series, mar_annual: float = 0.0) -> float:
    r = safe_series(daily_ret).astype(float).dropna()
    if len(r) < 20:
        return float("nan")
    mar_daily = mar_annual / 252.0
    d = np.minimum(0.0, (r - mar_daily).values)
    dd = float(np.sqrt(np.mean(d**2)))
    return float(dd * np.sqrt(252))


def sortino_ratio(daily_ret: pd.Series, mar_annual: float = 0.0) -> float:
    r = safe_series(daily_ret).astype(float).dropna()
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


def te_risk_band(te: float) -> str:
    v = safe_float(te)
    if not math.isfinite(v):
        return "N/A"
    if v < 0.08:
        return "Low"
    if v < 0.16:
        return "Medium"
    return "High"


# ============================================================
# Beta reliability (benchmark should match portfolio beta)
# ============================================================
def beta_target_for_mode(mode: str) -> float:
    m = str(mode).lower().strip()
    if "alpha-minus-beta" in m:
        return 0.85
    if "private" in m:
        return 1.00
    return 1.00  # Standard


def beta_and_r2(wave_ret: pd.Series, bm_ret: pd.Series) -> Tuple[float, float, int]:
    w = safe_series(wave_ret).astype(float)
    b = safe_series(bm_ret).astype(float)
    df = pd.concat([w.rename("w"), b.rename("b")], axis=1).dropna()
    n = int(df.shape[0])
    if n < 30:
        return (float("nan"), float("nan"), n)

    x = df["b"].values
    y = df["w"].values
    vx = float(np.var(x, ddof=1))
    if not math.isfinite(vx) or vx <= 0:
        return (float("nan"), float("nan"), n)

    cov = float(np.cov(x, y, ddof=1)[0, 1])
    beta = cov / vx

    yhat = beta * x
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - float(np.mean(y))) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return (float(beta), float(r2), n)


def beta_reliability_score(beta: float, r2: float, n: int, beta_target: float) -> float:
    b = safe_float(beta)
    r = safe_float(r2)
    if not math.isfinite(b) or not math.isfinite(r) or n < 30:
        return float("nan")

    mismatch = abs(b - beta_target)
    p_mis = float(np.clip(mismatch * 100.0, 0.0, 40.0))
    p_r2 = float(np.clip((1.0 - r) * 50.0, 0.0, 40.0))
    p_n = float(np.clip((252 - min(n, 252)) / 252.0 * 15.0, 0.0, 15.0))
    score = 100.0 - (p_mis + p_r2 + p_n)
    return float(np.clip(score, 0.0, 100.0))


def beta_band(score: float) -> str:
    s = safe_float(score)
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


# ============================================================
# Glossary / Definitions
# ============================================================
GLOSSARY: Dict[str, str] = {
    "Canonical (Source of Truth)": (
        "Governance rule: ALL metrics come from one standardized history object for the selected Wave+Mode "
        "(hist_sel = wave_nav, bm_nav, wave_ret, bm_ret). Every panel reuses it. No duplicate math = no crisscross."
    ),
    "Return": "Wave return over the window (not annualized unless stated).",
    "Alpha": "Wave return minus Benchmark return over the same window.",
    "Tracking Error (TE)": "Annualized volatility of (Wave daily returns − Benchmark daily returns).",
    "Information Ratio (IR)": "Excess return divided by Tracking Error (risk-adjusted alpha).",
    "Max Drawdown (MaxDD)": "Largest peak-to-trough decline over the period (negative).",
    "VaR 95% (daily)": "Loss threshold where ~5% of days are worse (historical).",
    "CVaR 95% (daily)": "Average loss of the worst ~5% of days (tail risk).",
    "Sharpe": "Risk-adjusted return using total volatility (0% rf here).",
    "Sortino": "Risk-adjusted return using downside deviation only.",
    "Benchmark Snapshot / Drift": "A fingerprint of benchmark composition. Drift means it changed in-session.",
    "Coverage Score": "0–100 heuristic of data completeness + freshness.",
    "Difficulty vs SPY": "Concentration/diversification proxy (not a promise).",
    "Risk Reaction Score": "0–100 heuristic of risk posture from TE/MaxDD/CVaR.",
    "Analytics Scorecard": "Governance-native reliability grade for analytics outputs (not performance).",
    "Beta (vs Benchmark)": "Regression slope of Wave daily returns vs Benchmark daily returns.",
    "Beta Reliability Score": "0–100: beta-target match + linkage quality (R²) + sample size.",
}


def render_definitions(keys: List[str], title: str = "Definitions"):
    with st.expander(title):
        for k in keys:
            st.markdown(f"**{k}:** {GLOSSARY.get(k, '(definition not found)')}")


# ============================================================
# Optional yfinance chips
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
# History loader (engine → CSV fallback)
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

    # Preferred: compute_history_nav
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

    # Fallback function names
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


# ============================================================
# Benchmark snapshot + drift + diff + difficulty proxy
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


# ============================================================
# Coverage + Confidence
# ============================================================
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
    try:
        score = safe_float(cov.get("completeness_score"))
        age = safe_float(cov.get("age_days"))
        rows = int(cov.get("rows") or 0)

        drift = (str(bm_drift).lower().strip() != "stable")
        issues: List[str] = []

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
# Governance-native Analytics Scorecard (selected wave)
# ============================================================
def _score_to_grade_af(score: float) -> str:
    s = safe_float(score)
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


def compute_analytics_score_for_selected(hist_sel: pd.DataFrame, cov: Dict[str, Any], bm_drift: str) -> Dict[str, Any]:
    try:
        rows_n = int(cov.get("rows") or 0)
        coverage_score = safe_float(cov.get("completeness_score"))
        age_days = safe_float(cov.get("age_days"))
        miss_pct = safe_float(cov.get("missing_pct"))

        te = np.nan
        ir = np.nan
        mdd = np.nan
        vol = np.nan

        if hist_sel is not None and not hist_sel.empty and len(hist_sel) >= 20:
            te = tracking_error(hist_sel["wave_ret"], hist_sel["bm_ret"])
            ir = information_ratio(hist_sel["wave_nav"], hist_sel["bm_nav"], te)
            mdd = max_drawdown(hist_sel["wave_nav"])
            vol = annualized_vol(hist_sel["wave_ret"])

        # D1 Data Integrity
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

        # D2 Benchmark fidelity (drift is a governance red flag)
        d2 = 92.0 if str(bm_drift).lower().strip() == "stable" else 78.0

        # D3 Risk discipline
        d3 = 80.0
        if math.isfinite(te):
            d3 += float(np.clip((0.10 - te) * 250.0, -25.0, 15.0))
        if math.isfinite(vol):
            d3 += float(np.clip((0.22 - vol) * 140.0, -25.0, 15.0))
        if math.isfinite(mdd):
            d3 += float(np.clip((-0.18 - mdd) * 120.0, -25.0, 15.0))
        d3 = float(np.clip(d3, 0.0, 100.0))

        # D4 Efficiency quality (IR proxy)
        d4 = 55.0
        if math.isfinite(ir):
            d4 = float(np.clip(55.0 + ir * 30.0, 0.0, 100.0))

        # D5 Decision readiness (simple proxy)
        d5 = 88.0
        if math.isfinite(coverage_score):
            d5 += (coverage_score - 90.0) * 0.25
        if math.isfinite(age_days) and age_days > 3:
            d5 -= min(20.0, (age_days - 3) * 3.5)
        if str(bm_drift).lower().strip() != "stable":
            d5 -= 15.0
        if rows_n < 90:
            d5 -= 8.0
        if rows_n < 60:
            d5 -= 12.0
        d5 = float(np.clip(d5, 0.0, 100.0))

        total = float(np.clip((d1 + d2 + d3 + d4 + d5) / 5.0, 0.0, 100.0))
        grade = _score_to_grade_af(total)

        flags: List[str] = []
        if d1 < 70:
            flags.append("DATA")
        if d2 < 70:
            flags.append("BM")
        if d3 < 70:
            flags.append("RISK")
        if d5 < 70:
            flags.append("READY")

        return {"AnalyticsScore": total, "Grade": grade, "Flags": " ".join(flags) if flags else ""}
    except Exception:
        return {"AnalyticsScore": np.nan, "Grade": "N/A", "Flags": "ERR"}


# ============================================================
# Risk Reaction Score (0-100)
# ============================================================
def risk_reaction_score(te: float, mdd: float, cvar95: float) -> float:
    try:
        te_v = safe_float(te)
        mdd_v = safe_float(mdd)
        cvar_v = safe_float(cvar95)

        score = 80.0
        if math.isfinite(te_v):
            score += float(np.clip((0.12 - te_v) * 200.0, -25.0, 15.0))
        if math.isfinite(mdd_v):
            score += float(np.clip((-0.18 - mdd_v) * 120.0, -25.0, 15.0))
        if math.isfinite(cvar_v):
            score += float(np.clip((-0.025 - cvar_v) * 900.0, -25.0, 15.0))
        return float(np.clip(score, 0.0, 100.0))
    except Exception:
        return float("nan")


# ============================================================
# Deterministic AI Explanation Layer (rules-based)
# ============================================================
def ai_explain_narrative(
    cov: Dict[str, Any],
    bm_drift: str,
    r30: float,
    a30: float,
    r60: float,
    a60: float,
    te: float,
    mdd: float,
    sharpe: float,
    cvar95: float,
    rr_score: float,
    beta_val: float,
    beta_target: float,
    beta_score: float,
) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {
        "What changed recently": [],
        "Why the alpha looks like this": [],
        "Risk driver": [],
        "What to verify": [],
    }

    age = safe_float(cov.get("age_days"))
    cov_score = safe_float(cov.get("completeness_score"))
    rows = int(cov.get("rows") or 0)

    # What changed
    if str(bm_drift).lower().strip() != "stable":
        out["What changed recently"].append("Benchmark snapshot drift detected in-session (composition changed).")
    if math.isfinite(age) and age >= 3:
        out["What changed recently"].append(f"History freshness: last datapoint is {int(age)} day(s) old.")
    if rows < 90:
        out["What changed recently"].append("Limited history window may reduce stability of risk metrics.")
    if not out["What changed recently"]:
        out["What changed recently"].append("No major governance changes detected (stable benchmark + fresh coverage).")

    # Why alpha
    if math.isfinite(a30):
        if a30 > 0.02:
            out["Why the alpha looks like this"].append("Positive 30D alpha: wave has recently outperformed its benchmark mix.")
        elif a30 < -0.02:
            out["Why the alpha looks like this"].append("Negative 30D alpha: recent underperformance versus benchmark mix.")
        else:
            out["Why the alpha looks like this"].append("30D alpha near flat: wave and benchmark moving similarly.")
    if math.isfinite(te):
        out["Why the alpha looks like this"].append(f"Active risk (TE {fmt_pct(te)}) is {te_risk_band(te)}; alpha magnitude often scales with active risk.")

    # Risk driver
    driver_bits = []
    if math.isfinite(te):
        driver_bits.append(f"TE {fmt_pct(te)}")
    if math.isfinite(mdd):
        driver_bits.append(f"MaxDD {fmt_pct(mdd)}")
    if math.isfinite(cvar95):
        driver_bits.append(f"CVaR {fmt_pct(cvar95)}")
    if driver_bits:
        out["Risk driver"].append("Risk posture driven by: " + ", ".join(driver_bits) + ".")
    if math.isfinite(rr_score):
        out["Risk driver"].append(f"Risk Reaction Score: {rr_score:.1f}/100.")
    if math.isfinite(sharpe):
        out["Risk driver"].append(f"Sharpe: {sharpe:.2f}.")
    if math.isfinite(beta_val):
        out["Risk driver"].append(
            f"Beta vs benchmark: {beta_val:.2f} (target {beta_target:.2f}) · Beta Reliability {fmt_num(beta_score,1)}/100."
        )

    # What to verify
    if math.isfinite(cov_score) and cov_score < 85:
        out["What to verify"].append("Coverage score < 85: inspect missing business days + pipeline consistency.")
    if str(bm_drift).lower().strip() != "stable":
        out["What to verify"].append("Benchmark drift: freeze benchmark mix for demos/governance.")
    if math.isfinite(a30) and abs(a30) >= 0.08:
        out["What to verify"].append("Large alpha: confirm benchmark weights + verify no gaps/rollovers in history.")
    if math.isfinite(beta_score) and beta_score < 75:
        out["What to verify"].append("Beta reliability low: benchmark may not match systematic exposure (review mix).")
    if not out["What to verify"]:
        out["What to verify"].append("No critical verification flags triggered.")

    return out


# ============================================================
# Metrics from canonical hist_sel
# ============================================================
def compute_metrics_from_hist(hist_sel: pd.DataFrame) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "r1": np.nan, "a1": np.nan,
        "r30": np.nan, "a30": np.nan,
        "r60": np.nan, "a60": np.nan,
        "r365": np.nan, "a365": np.nan,
        "te": np.nan, "ir": np.nan,
        "mdd": np.nan, "vol": np.nan,
        "sharpe": np.nan, "sortino": np.nan,
        "var95": np.nan, "cvar95": np.nan,
    }
    if hist_sel is None or hist_sel.empty:
        return out

    nav_w = hist_sel["wave_nav"].astype(float)
    nav_b = hist_sel["bm_nav"].astype(float)
    wret = hist_sel["wave_ret"].astype(float)
    bret = hist_sel["bm_ret"].astype(float)

    if len(nav_w) >= 2 and len(nav_b) >= 2:
        out["r1"] = float(nav_w.iloc[-1] / nav_w.iloc[-2] - 1.0)
        out["a1"] = float(out["r1"] - (nav_b.iloc[-1] / nav_b.iloc[-2] - 1.0))

    out["r30"] = ret_from_nav(nav_w, min(30, len(nav_w)))
    out["a30"] = out["r30"] - ret_from_nav(nav_b, min(30, len(nav_b)))

    out["r60"] = ret_from_nav(nav_w, min(60, len(nav_w)))
    out["a60"] = out["r60"] - ret_from_nav(nav_b, min(60, len(nav_b)))

    out["r365"] = ret_from_nav(nav_w, min(365, len(nav_w)))
    out["a365"] = out["r365"] - ret_from_nav(nav_b, min(365, len(nav_b)))

    out["te"] = tracking_error(wret, bret)
    out["ir"] = information_ratio(nav_w, nav_b, out["te"])
    out["mdd"] = max_drawdown(nav_w)
    out["vol"] = annualized_vol(wret)
    out["sharpe"] = sharpe_ratio(wret, rf_annual=0.0)
    out["sortino"] = sortino_ratio(wret, mar_annual=0.0)
    v, cv = var_cvar(wret, level=0.95)
    out["var95"] = v
    out["cvar95"] = cv
    return out


# ============================================================
# UI helpers
# ============================================================
def chip(label: str):
    st.markdown(f'<span class="waves-chip">{label}</span>', unsafe_allow_html=True)


def tile(label: str, value: str, sub: str = ""):
    st.markdown(
        f"""
<div class="waves-tile">
  <div class="waves-tile-label">{label}</div>
  <div class="waves-tile-value">{value}</div>
  <div class="waves-tile-sub">{sub}</div>
</div>
""",
        unsafe_allow_html=True,
    )


def safe_panel(title: str, fn):
    try:
        fn()
    except Exception as e:
        st.error(f"{title} failed (non-fatal): {e}")


# ============================================================
# Sidebar controls
# ============================================================
all_waves = get_all_waves_safe()
if not all_waves:
    st.warning("No waves found. Ensure engine is available or CSVs are present (wave_config.csv / wave_weights.csv / list.csv).")

st.sidebar.markdown("## Controls")

mode = st.sidebar.selectbox("Mode", ["Standard", "Alpha-Minus-Beta", "Private Logic"], index=0)
selected_wave = st.sidebar.selectbox("Selected Wave", options=all_waves if all_waves else ["(none)"], index=0)

scan_mode = st.sidebar.toggle("Scan Mode (fast, fewer visuals)", value=True)
days = st.sidebar.selectbox("History Window", [365, 730, 1095, 2520], index=0)

st.sidebar.markdown("---")
st.sidebar.caption("Governance: every metric shown is computed from the canonical hist_sel (selected Wave + Mode).")


# ============================================================
# Canonical source-of-truth
# ============================================================
hist_sel = _standardize_history(compute_wave_history(selected_wave, mode, days=days)) if selected_wave and selected_wave != "(none)" else pd.DataFrame()
cov = coverage_report(hist_sel)

bm_mix = get_benchmark_mix()
bm_rows_now = _bm_rows_for_wave(bm_mix, selected_wave) if selected_wave and selected_wave != "(none)" else pd.DataFrame(columns=["Ticker", "Weight"])
bm_id = benchmark_snapshot_id(selected_wave, bm_mix) if selected_wave and selected_wave != "(none)" else "BM-NA"
bm_drift = benchmark_drift_status(selected_wave, mode, bm_id) if selected_wave and selected_wave != "(none)" else "stable"
bm_diff = benchmark_diff_table(selected_wave, mode, bm_rows_now) if ENABLE_FIDELITY_INSPECTOR else pd.DataFrame()

metrics = compute_metrics_from_hist(hist_sel)
conf_level, conf_reason = confidence_from_integrity(cov, bm_drift)

sel_score = compute_analytics_score_for_selected(hist_sel, cov, bm_drift) if ENABLE_SCORECARD else {"AnalyticsScore": np.nan, "Grade": "N/A", "Flags": ""}
rr_score = risk_reaction_score(metrics["te"], metrics["mdd"], metrics["cvar95"])

difficulty = benchmark_difficulty_proxy(bm_rows_now)
te_band = te_risk_band(metrics["te"])

beta_target = beta_target_for_mode(mode)
beta_val, beta_r2, beta_n = beta_and_r2(hist_sel["wave_ret"] if not hist_sel.empty else pd.Series(dtype=float),
                                       hist_sel["bm_ret"] if not hist_sel.empty else pd.Series(dtype=float))
beta_score = beta_reliability_score(beta_val, beta_r2, beta_n, beta_target)
beta_grade = beta_band(beta_score)


# ============================================================
# BIG HEADER
# ============================================================
st.markdown(f'<div class="waves-big-wave">{selected_wave}</div>', unsafe_allow_html=True)
st.markdown(
    f'<div class="waves-subhead">Mode: <b>{mode}</b> &nbsp; • &nbsp; Canonical window: last {days} trading days (max)</div>',
    unsafe_allow_html=True,
)

# ============================================================
# Sticky chip bar
# ============================================================
st.markdown('<div class="waves-sticky">', unsafe_allow_html=True)
chip(f"Confidence: {conf_level}")
if ENABLE_SCORECARD:
    chip(f"Wave Analytics: {sel_score.get('Grade','N/A')} ({fmt_num(sel_score.get('AnalyticsScore'), 1)}) {sel_score.get('Flags','')}")
chip(f"BM: {bm_id} · {bm_drift.capitalize()}")
chip(f"Coverage: {fmt_num(cov.get('completeness_score'),1)} · AgeDays: {fmt_int(cov.get('age_days'))}")
chip(f"30D α {fmt_pct(metrics['a30'])} · r {fmt_pct(metrics['r30'])}")
chip(f"60D α {fmt_pct(metrics['a60'])} · r {fmt_pct(metrics['r60'])}")
chip(f"Risk: TE {fmt_pct(metrics['te'])} ({te_band}) · MaxDD {fmt_pct(metrics['mdd'])}")
chip(f"BetaRel: {beta_grade} ({fmt_num(beta_score,1)}) · β {fmt_num(beta_val,2)} tgt {fmt_num(beta_target,2)}")
st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# Tabs
# ============================================================
tabs = st.tabs(
    [
        "IC Summary",
        "Overview",
        "Risk + Advanced",
        "Benchmark Governance",
        "Comparator",
        "Diagnostics",
    ]
)


# ============================================================
# IC SUMMARY
# ============================================================
with tabs[0]:
    st.markdown("### Executive IC One-Pager")

    colA, colB = st.columns([1.2, 1.0], gap="large")

    with colA:
        st.markdown('<div class="waves-card">', unsafe_allow_html=True)
        st.markdown("#### What is this wave?")
        st.write(
            "A governance-native portfolio wave with a benchmark-anchored analytics stack. "
            "Designed to eliminate crisscross metrics and provide decision-ready outputs fast."
        )
        st.markdown("**Trust + Governance**")
        st.write(f"**Confidence:** {conf_level} — {conf_reason}")
        st.write(f"**Benchmark Snapshot:** {bm_id} · Drift: {bm_drift}")
        st.write(f"**Beta Reliability:** {beta_grade} ({fmt_num(beta_score,1)}/100) · β {fmt_num(beta_val,2)} vs target {fmt_num(beta_target,2)} · R² {fmt_num(beta_r2,2)} · n {beta_n}")
        st.markdown("**Performance vs Benchmark**")
        st.write(f"30D Return {fmt_pct(metrics['r30'])} | 30D Alpha {fmt_pct(metrics['a30'])}")
        st.write(f"60D Return {fmt_pct(metrics['r60'])} | 60D Alpha {fmt_pct(metrics['a60'])}")
        st.write(f"365D Return {fmt_pct(metrics['r365'])} | 365D Alpha {fmt_pct(metrics['a365'])}")
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="waves-card">', unsafe_allow_html=True)
        st.markdown("#### Key Wins / Key Risks / Next Actions")
        wins, risks, actions = [], [], []

        if conf_level == "High":
            wins.append("Fresh + complete coverage supports institutional trust.")
        if bm_drift == "stable":
            wins.append("Benchmark snapshot is stable (governance green).")
        if math.isfinite(beta_score) and beta_score >= 80:
            wins.append("Benchmark systematic exposure match (beta reliability is strong).")
        if math.isfinite(metrics["a30"]) and metrics["a30"] > 0:
            wins.append("Positive 30D alpha versus benchmark mix.")

        if conf_level != "High":
            risks.append("Data trust flags present (coverage/age/rows).")
        if bm_drift != "stable":
            risks.append("Benchmark drift detected (composition changed in-session).")
        if math.isfinite(beta_score) and beta_score < 75:
            risks.append("Beta reliability low (benchmark may not match systematic exposure).")
        if math.isfinite(metrics["mdd"]) and metrics["mdd"] <= -0.25:
            risks.append("Deep drawdown regime risk is elevated.")

        if bm_drift != "stable":
            actions.append("Freeze benchmark mix for demos/governance, then re-run.")
        if math.isfinite(beta_score) and beta_score < 75:
            actions.append("Review benchmark mix: adjust exposures to match wave beta target (or justify intentional mismatch).")
        if math.isfinite(metrics["te"]) and metrics["te"] >= 0.20:
            actions.append("Confirm exposure caps / SmartSafe posture for high active risk.")
        if conf_level != "High":
            actions.append("Inspect history pipeline for missing days or stale writes.")
        if not actions:
            actions.append("Proceed: governance is stable; use comparator to position vs other waves.")

        st.markdown("**Key Wins**")
        for w in (wins[:4] if wins else ["(none)"]):
            st.write("• " + w)

        st.markdown("**Key Risks**")
        for r in (risks[:4] if risks else ["(none)"]):
            st.write("• " + r)

        st.markdown("**Next Actions**")
        for a in (actions[:4] if actions else ["(none)"]):
            st.write("• " + a)

        st.markdown("</div>", unsafe_allow_html=True)

    with colB:
        st.markdown("#### IC Tiles")
        c1, c2 = st.columns(2, gap="medium")
        with c1:
            tile("Confidence", conf_level, conf_reason)
            tile("Benchmark", "Stable" if bm_drift == "stable" else "Drift", bm_id)
            tile("30D Alpha", fmt_pct(metrics["a30"]), f"30D Return {fmt_pct(metrics['r30'])}")
        with c2:
            tile("Analytics Grade", sel_score.get("Grade", "N/A"), f"{fmt_num(sel_score.get('AnalyticsScore'),1)}/100 {sel_score.get('Flags','')}")
            tile("Beta Reliability", beta_grade, f"{fmt_num(beta_score,1)}/100 · β {fmt_num(beta_val,2)} tgt {fmt_num(beta_target,2)}")
            tile("Active Risk (TE)", fmt_pct(metrics["te"]), f"Band: {te_band}")

        st.markdown("---")
        render_definitions(
            [
                "Canonical (Source of Truth)",
                "Return",
                "Alpha",
                "Tracking Error (TE)",
                "Max Drawdown (MaxDD)",
                "CVaR 95% (daily)",
                "Analytics Scorecard",
                "Benchmark Snapshot / Drift",
                "Beta (vs Benchmark)",
                "Beta Reliability Score",
            ],
            title="Definitions (IC)",
        )


# ============================================================
# OVERVIEW
# ============================================================
with tabs[1]:
    st.markdown("### Overview (Canonical Metrics)")
    st.caption("Everything below is computed from the same canonical hist_sel object (no duplicate math).")

    truth = {
        "CanonicalEndDate": str(pd.to_datetime(hist_sel.index).max().date()) if not hist_sel.empty else "—",
        "Rows": int(cov.get("rows") or 0),
        "BMSnapshot": bm_id,
        "BMDrift": bm_drift,
        "CoverageScore": fmt_num(cov.get("completeness_score"), 1),
        "AgeDays": fmt_int(cov.get("age_days")),
        "Beta": fmt_num(beta_val, 2),
        "BetaTarget": fmt_num(beta_target, 2),
        "BetaReliability": fmt_num(beta_score, 1),
        "30DReturn": fmt_pct(metrics["r30"]),
        "30DAlpha": fmt_pct(metrics["a30"]),
        "TE": fmt_pct(metrics["te"]),
        "MaxDD": fmt_pct(metrics["mdd"]),
        "Sharpe": fmt_num(metrics["sharpe"], 2),
        "CVaR95": fmt_pct(metrics["cvar95"]),
        "RiskReaction": fmt_num(rr_score, 1),
    }
    st.markdown("#### Cohesion Lock (Truth Table)")
    st.dataframe(pd.DataFrame([truth]), use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("#### Performance vs Benchmark")
    perf_df = pd.DataFrame(
        [
            {"Window": "1D", "Return": metrics["r1"], "Alpha": metrics["a1"]},
            {"Window": "30D", "Return": metrics["r30"], "Alpha": metrics["a30"]},
            {"Window": "60D", "Return": metrics["r60"], "Alpha": metrics["a60"]},
            {"Window": "365D", "Return": metrics["r365"], "Alpha": metrics["a365"]},
        ]
    )
    show = perf_df.copy()
    show["Return"] = show["Return"].apply(fmt_pct)
    show["Alpha"] = show["Alpha"].apply(fmt_pct)
    st.dataframe(show, use_container_width=True, hide_index=True)

    if not scan_mode and not hist_sel.empty:
        st.markdown("---")
        st.markdown("#### NAV Preview (Wave vs Benchmark)")
        nav_view = hist_sel[["wave_nav", "bm_nav"]].tail(120).copy()
        nav_view = nav_view.rename(columns={"wave_nav": "Wave NAV", "bm_nav": "Benchmark NAV"})
        st.line_chart(nav_view, height=240, use_container_width=True)


# ============================================================
# RISK + ADVANCED
# ============================================================
with tabs[2]:
    st.markdown("### Risk + Advanced Analytics (Canonical)")

    c1, c2, c3 = st.columns(3, gap="medium")
    with c1:
        st.metric("Sharpe (0% rf)", value=fmt_num(metrics["sharpe"], 2))
        st.metric("Sortino (0% MAR)", value=fmt_num(metrics["sortino"], 2))
    with c2:
        st.metric("Volatility (ann.)", value=fmt_pct(metrics["vol"], 2))
        ddv = downside_deviation(hist_sel["wave_ret"]) if not hist_sel.empty else np.nan
        st.metric("Downside Dev (ann.)", value=fmt_pct(ddv, 2))
    with c3:
        st.metric("Max Drawdown", value=fmt_pct(metrics["mdd"], 2))
        st.metric("TE (ann.)", value=fmt_pct(metrics["te"], 2))

    st.markdown("---")
    c4, c5, c6 = st.columns(3, gap="medium")
    with c4:
        st.metric("VaR 95% (daily)", value=fmt_pct(metrics["var95"], 2))
    with c5:
        st.metric("CVaR 95% (daily)", value=fmt_pct(metrics["cvar95"], 2))
    with c6:
        st.metric("Risk Reaction Score", value=f"{fmt_num(rr_score,1)}/100")

    render_definitions(
        ["Sharpe", "Sortino", "Max Drawdown (MaxDD)", "Tracking Error (TE)", "VaR 95% (daily)", "CVaR 95% (daily)", "Risk Reaction Score"],
        title="Definitions (Risk & Advanced)",
    )

    if ENABLE_AI_EXPLAIN:
        st.markdown("---")
        st.markdown("#### AI Explanation Layer (Rules-Based, Deterministic)")
        explain = ai_explain_narrative(
            cov=cov,
            bm_drift=bm_drift,
            r30=metrics["r30"],
            a30=metrics["a30"],
            r60=metrics["r60"],
            a60=metrics["a60"],
            te=metrics["te"],
            mdd=metrics["mdd"],
            sharpe=metrics["sharpe"],
            cvar95=metrics["cvar95"],
            rr_score=rr_score,
            beta_val=beta_val,
            beta_target=beta_target,
            beta_score=beta_score,
        )

        cols = st.columns(2, gap="large")
        with cols[0]:
            st.markdown("**What changed recently**")
            for s in explain["What changed recently"]:
                st.write("• " + s)
            st.markdown("**Why the alpha looks like this**")
            for s in explain["Why the alpha looks like this"]:
                st.write("• " + s)
        with cols[1]:
            st.markdown("**Risk driver**")
            for s in explain["Risk driver"]:
                st.write("• " + s)
            st.markdown("**What to verify**")
            for s in explain["What to verify"]:
                st.write("• " + s)


# ============================================================
# BENCHMARK GOVERNANCE
# ============================================================
with tabs[3]:
    st.markdown("### Benchmark Governance (Fidelity Inspector)")

    left, right = st.columns([1.0, 1.1], gap="large")

    with left:
        st.markdown('<div class="waves-card">', unsafe_allow_html=True)
        st.markdown("#### Inspector Summary")
        st.write(f"**Snapshot:** {bm_id}")
        st.write(f"**Drift Status:** {bm_drift}")
        st.write(f"**Active Risk Band (TE):** {te_band} (TE {fmt_pct(metrics['te'])})")
        st.write(f"**Beta Reliability:** {beta_grade} ({fmt_num(beta_score,1)}/100) · β {fmt_num(beta_val,2)} tgt {fmt_num(beta_target,2)}")
        st.write(f"**Difficulty vs SPY (proxy):** {fmt_num(difficulty.get('difficulty_vs_spy'), 1)} (range ~ -25 to +25)")
        st.caption("Difficulty is a concentration/diversification heuristic (not a promise).")
        st.markdown("</div>", unsafe_allow_html=True)

        if bm_rows_now is not None and not bm_rows_now.empty:
            st.markdown("#### Current Benchmark Composition")
            bm_show = bm_rows_now.copy()
            bm_show["WeightPct"] = (bm_show["Weight"] * 100).round(2)
            bm_show = bm_show.drop(columns=["Weight"]).sort_values("WeightPct", ascending=False).reset_index(drop=True)
            st.dataframe(bm_show, use_container_width=True, hide_index=True)
        else:
            st.info("No benchmark mix table available for this wave.")

    with right:
        st.markdown("#### Composition Change vs Last Session")
        if bm_diff is None or bm_diff.empty:
            st.info("No prior snapshot stored yet (or no meaningful changes).")
        else:
            st.dataframe(bm_diff, use_container_width=True, hide_index=True)

        render_definitions(
            ["Benchmark Snapshot / Drift", "Difficulty vs SPY", "Tracking Error (TE)", "Beta Reliability Score"],
            title="Definitions (Benchmark Governance)",
        )


# ============================================================
# COMPARATOR (Wave A vs Wave B)
# ============================================================
with tabs[4]:
    st.markdown("### Wave-to-Wave Comparator")

    if not ENABLE_COMPARATOR:
        st.info("Comparator disabled.")
    else:
        col1, col2 = st.columns([1.0, 1.0], gap="large")
        with col1:
            wave_a = selected_wave
            st.caption("Wave A (selected)")
            st.write(f"**{wave_a}**")
        with col2:
            wave_b = st.selectbox("Wave B", options=[w for w in all_waves if w != selected_wave] if all_waves else ["(none)"], index=0)

        def load_metrics_for(w: str) -> Dict[str, Any]:
            h = _standardize_history(compute_wave_history(w, mode, days=days))
            c = coverage_report(h)
            bid = benchmark_snapshot_id(w, bm_mix)
            d = benchmark_drift_status(w, mode, bid)
            m = compute_metrics_from_hist(h)
            bt = beta_target_for_mode(mode)
            bv, br2, bn = beta_and_r2(h["wave_ret"] if not h.empty else pd.Series(dtype=float),
                                     h["bm_ret"] if not h.empty else pd.Series(dtype=float))
            bs = beta_reliability_score(bv, br2, bn, bt)
            return {"cov": c, "bm_id": bid, "bm_drift": d, "m": m, "beta": bv, "beta_score": bs}

        if wave_b and wave_b != "(none)":
            a = load_metrics_for(wave_a)
            b = load_metrics_for(wave_b)

            comp = pd.DataFrame(
                [
                    {"Metric": "30D Return", "Wave A": fmt_pct(a["m"]["r30"]), "Wave B": fmt_pct(b["m"]["r30"])},
                    {"Metric": "30D Alpha", "Wave A": fmt_pct(a["m"]["a30"]), "Wave B": fmt_pct(b["m"]["a30"])},
                    {"Metric": "60D Return", "Wave A": fmt_pct(a["m"]["r60"]), "Wave B": fmt_pct(b["m"]["r60"])},
                    {"Metric": "60D Alpha", "Wave A": fmt_pct(a["m"]["a60"]), "Wave B": fmt_pct(b["m"]["a60"])},
                    {"Metric": "TE", "Wave A": fmt_pct(a["m"]["te"]), "Wave B": fmt_pct(b["m"]["te"])},
                    {"Metric": "MaxDD", "Wave A": fmt_pct(a["m"]["mdd"]), "Wave B": fmt_pct(b["m"]["mdd"])},
                    {"Metric": "CVaR 95%", "Wave A": fmt_pct(a["m"]["cvar95"]), "Wave B": fmt_pct(b["m"]["cvar95"])},
                    {"Metric": "Beta", "Wave A": fmt_num(a["beta"], 2), "Wave B": fmt_num(b["beta"], 2)},
                    {"Metric": "Beta Reliability", "Wave A": fmt_num(a["beta_score"], 1), "Wave B": fmt_num(b["beta_score"], 1)},
                    {"Metric": "BM Snapshot", "Wave A": a["bm_id"], "Wave B": b["bm_id"]},
                    {"Metric": "BM Drift", "Wave A": a["bm_drift"], "Wave B": b["bm_drift"]},
                ]
            )
            st.dataframe(comp, use_container_width=True, hide_index=True)
        else:
            st.info("Pick Wave B to compare.")


# ============================================================
# DIAGNOSTICS (always boots)
# ============================================================
with tabs[5]:
    st.markdown("### Diagnostics (Boot-Safe)")

    st.markdown("#### Engine status")
    if we is None:
        st.error(f"waves_engine import failed: {ENGINE_IMPORT_ERROR}")
        st.write("Fallback mode will attempt to load wave_history.csv / config CSVs.")
    else:
        st.success("waves_engine imported successfully.")

    st.markdown("---")
    st.markdown("#### Canonical history checks")
    st.write(f"Rows: {int(cov.get('rows') or 0)}")
    st.write(f"FirstDate: {cov.get('first_date')}")
    st.write(f"LastDate: {cov.get('last_date')}")
    st.write(f"AgeDays: {cov.get('age_days')}")
    st.write(f"MissingBDays: {cov.get('missing_bdays')}")
    st.write(f"MissingPct: {fmt_num(cov.get('missing_pct'), 4)}")
    st.write(f"CoverageScore: {fmt_num(cov.get('completeness_score'), 1)}")
    if cov.get("flags"):
        st.warning(" | ".join([str(x) for x in cov.get("flags")]))
    else:
        st.success("No coverage flags.")

    st.markdown("---")
    st.markdown("#### Canonical columns preview")
    if hist_sel is None or hist_sel.empty:
        st.info("hist_sel is empty.")
    else:
        st.dataframe(hist_sel.tail(20), use_container_width=True)

    st.markdown("---")
    st.markdown("#### Optional libs")
    st.write(f"yfinance: {'OK' if yf is not None else 'missing'}")
    st.write(f"plotly: {'OK' if go is not None else 'missing'}")
    