# ****App 72:***

# app.py — WAVES Intelligence™ Institutional Console (Vector OS Edition)
# FULL PRODUCTION FILE (NO PATCHES) — SINGLE-FILE ONLY
#
# vNEXT — CANONICAL COHESION LOCK + IC ONE-PAGER + FIDELITY INSPECTOR + AI EXPLAIN
#         + COMPARATOR + BETA RELIABILITY + DIAGNOSTICS (ALWAYS BOOTS)
#         + VECTOR™ TRUTH LAYER (READ-ONLY, DETERMINISTIC)
#         + VECTOR™ REFEREE (VERDICT + ASSUMPTIONS + FAILURE FLAGS)
#         + ALPHA SNAPSHOT (ALL WAVES) + ALPHA CAPTURE + RISK-ON/OFF ATTRIBUTION
#         + NEW: VECTOR GOVERNANCE LAYER (Status Bar + Final Verdict Box + Assumptions Panel
#                + Gating Warnings + Wave Purpose Statements)
#
# Architectural refinement principles (system-feedback, governance, stability):
#   • ONE canonical dataset per selected Wave+Mode: hist_sel (source-of-truth)
#   • Guarded optional imports (yfinance / plotly) for flexible onboarding
#   • Guarded vector_truth import (app boots predictably regardless of dependencies)
#   • Every major section wrapped so the app boots predictably if a panel fails
#   • Diagnostics tab always available (stable feedback loop for engine import errors, empty history, etc.)
#
# Canonical governance rule (context vs control):
#   hist_sel = _standardize_history(compute_wave_history(selected_wave, mode, days))
#   Every metric shown uses hist_sel (no duplicate math / no crisscross). This provides governance via architectural constraint.

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

# -------------------------------
# Vector Truth import (guarded)
# -------------------------------
VECTOR_TRUTH_IMPORT_ERROR = None
try:
    from vector_truth import build_vector_truth_report, format_vector_truth_markdown
except Exception as e:
    build_vector_truth_report = None
    format_vector_truth_markdown = None
    VECTOR_TRUTH_IMPORT_ERROR = e


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
ENABLE_VECTOR_TRUTH = True    # auto-disables if vector_truth import missing
ENABLE_VECTOR_REFEREE = True  # referee verdict + assumptions + failure flags
ENABLE_ALPHA_SNAPSHOT = True  # ALL WAVES snapshot table

# NEW: Vector Governance Layer (the 5 items you listed)
VECTOR_GOVERNANCE_ENABLED = True
ENABLE_VECTOR_STATUS_BAR = True
ENABLE_FINAL_VERDICT_BOX = True
ENABLE_ASSUMPTIONS_PANEL = True
ENABLE_GATING_WARNINGS = True
ENABLE_WAVE_PURPOSE_STATEMENTS = True


# ============================================================
# Global UI CSS
# ============================================================
st.markdown(
    """
<style>
.block-container { padding-top: 0.85rem; padding-bottom: 65px; }
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

.vector-status {
  border-radius: 16px;
  border: 1px solid rgba(255,255,255,0.12);
  background: rgba(255,255,255,0.035);
  padding: 10px 12px;
  margin: 10px 0 12px 0;
}
.vector-status .title { font-weight: 900; letter-spacing: 0.3px; margin-bottom: 4px; }
.vector-status .row { display: flex; flex-wrap: wrap; gap: 8px; }
.vector-pill {
  display: inline-block;
  padding: 7px 10px;
  border-radius: 999px;
  border: 1px solid rgba(255,255,255,0.12);
  background: rgba(10, 15, 28, 0.40);
  font-size: 0.90rem;
}
.vector-warn {
  border-radius: 14px;
  border: 1px solid rgba(255, 204, 0, 0.20);
  background: rgba(255, 204, 0, 0.06);
  padding: 10px 12px;
  margin: 10px 0 12px 0;
}
.vector-crit {
  border-radius: 14px;
  border: 1px solid rgba(255, 80, 80, 0.22);
  background: rgba(255, 80, 80, 0.07);
  padding: 10px 12px;
  margin: 10px 0 12px 0;
}

/* Final Verdict Badge Styles */
.verdict-badge {
  text-align: center;
  padding: 20px 16px;
  border-radius: 16px;
  margin: 12px 0;
}
.verdict-badge-green {
  border: 2px solid rgba(46, 213, 115, 0.30);
  background: rgba(46, 213, 115, 0.10);
}
.verdict-badge-amber {
  border: 2px solid rgba(255, 204, 0, 0.30);
  background: rgba(255, 204, 0, 0.10);
}
.verdict-badge-red {
  border: 2px solid rgba(255, 80, 80, 0.30);
  background: rgba(255, 80, 80, 0.10);
}
.verdict-badge-title {
  font-size: 2.5rem;
  font-weight: 900;
  letter-spacing: 1px;
  margin-bottom: 8px;
}
.verdict-badge-subtitle {
  font-size: 1.1rem;
  opacity: 0.85;
  margin-top: 4px;
}
.signal-tile-compact {
  border: 1px solid rgba(255,255,255,0.10);
  border-radius: 12px;
  padding: 10px 12px;
  background: rgba(255,255,255,0.03);
  min-height: 70px;
}
.signal-tile-label {
  opacity: 0.75;
  font-size: 0.85rem;
  margin-bottom: 0.25rem;
}
.signal-tile-value {
  font-size: 1.25rem;
  font-weight: 750;
  line-height: 1.45rem;
}
.action-checklist {
  padding: 8px 0;
}
.action-item {
  display: flex;
  align-items: flex-start;
  margin: 8px 0;
  padding: 6px 8px;
  border-radius: 8px;
  background: rgba(255,255,255,0.02);
}
.action-checkbox {
  margin-right: 10px;
  font-size: 1.1rem;
}

/* WAVES Ticker Tape™ - Fixed Bottom Scrolling Ticker */
.waves-ticker-container {
  position: fixed;
  bottom: 0;
  left: 0;
  width: 100%;
  z-index: 1000;
  background: rgba(10, 15, 28, 0.95);
  border-top: 1px solid rgba(255, 255, 255, 0.15);
  backdrop-filter: blur(12px);
  -webkit-backdrop-filter: blur(12px);
  overflow: hidden;
  height: 48px;
  display: flex;
  align-items: center;
}

.waves-ticker-label {
  position: absolute;
  left: 0;
  top: 0;
  bottom: 0;
  display: flex;
  align-items: center;
  padding: 0 16px;
  background: rgba(46, 213, 115, 0.15);
  border-right: 1px solid rgba(255, 255, 255, 0.15);
  font-weight: 900;
  font-size: 0.85rem;
  letter-spacing: 0.5px;
  white-space: nowrap;
  z-index: 10;
  min-width: 180px;
}

.waves-ticker-content {
  margin-left: 180px;
  display: flex;
  white-space: nowrap;
  animation: ticker-scroll 60s linear infinite;
  padding: 0 20px;
}

.waves-ticker-item {
  display: inline-flex;
  align-items: center;
  padding: 0 24px;
  font-size: 0.90rem;
  border-right: 1px solid rgba(255, 255, 255, 0.08);
}

.waves-ticker-item .ticker-label {
  font-weight: 700;
  opacity: 0.85;
  margin-right: 6px;
}

.waves-ticker-item .ticker-value {
  font-weight: 600;
  color: rgba(255, 255, 255, 0.95);
}

.waves-ticker-item.positive .ticker-value {
  color: rgba(46, 213, 115, 1);
}

.waves-ticker-item.negative .ticker-value {
  color: rgba(255, 80, 80, 1);
}

.waves-ticker-item.neutral .ticker-value {
  color: rgba(255, 204, 0, 1);
}

@keyframes ticker-scroll {
  0% { transform: translateX(0%); }
  100% { transform: translateX(-50%); }
}

@media (max-width: 700px) {
  .block-container { padding-left: 0.8rem; padding-right: 0.8rem; padding-bottom: 60px; }
  .waves-big-wave { font-size: 1.75rem; line-height: 2.0rem; }
  .waves-tile-value { font-size: 1.30rem; line-height: 1.55rem; }
  .verdict-badge-title { font-size: 2.0rem; }
  .verdict-badge-subtitle { font-size: 1.0rem; }
  .signal-tile-value { font-size: 1.1rem; }
  .waves-ticker-label { min-width: 140px; font-size: 0.75rem; padding: 0 12px; }
  .waves-ticker-content { margin-left: 140px; }
  .waves-ticker-item { font-size: 0.80rem; padding: 0 16px; }
}
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
    """Returns numeric score with band label for Beta Reliability.
    
    Score bands:
    - Elite: 90-100
    - Strong: 80-89
    - Good: 70-79
    - Watch: 60-69
    - Risk: below 60
    """
    s = safe_float(score)
    if not math.isfinite(s):
        return "N/A"
    # Return formatted numeric score with band
    if s >= 90:
        return f"{s:.1f} (Elite)"
    if s >= 80:
        return f"{s:.1f} (Strong)"
    if s >= 70:
        return f"{s:.1f} (Good)"
    if s >= 60:
        return f"{s:.1f} (Watch)"
    return f"{s:.1f} (Risk)"


# ============================================================
# Glossary / Definitions
# ============================================================
GLOSSARY: Dict[str, str] = {
    "Canonical (Source of Truth)": (
        "Architectural governance constraint: ALL metrics derive from one standardized history object for the selected Wave+Mode "
        "(hist_sel = wave_nav, bm_nav, wave_ret, bm_ret). Every panel reuses it predictably. Context vs control: no duplicate math = no crisscross."
    ),
    "Return": "Wave return over the window (not annualized unless stated).",
    "Alpha": "Wave return minus Benchmark return over the same window.",
    "Alpha Capture": (
        "Daily (Wave return − Benchmark return) optionally normalized by exposure (if exposure history exists). "
        "Windowed alpha capture is compounded from daily alpha-capture series."
    ),
    "Capital-Weighted Alpha": "Investor-experience alpha (Wave return − Benchmark return) over the window.",
    "Exposure-Adjusted Alpha": "Capital-weighted alpha divided by average exposure over the window (if exposure known).",
    "Risk-On vs Risk-Off Attribution": "Alpha split by benchmark regime: Risk-Off when bm_ret < 0, else Risk-On.",
    "Tracking Error (TE)": "Annualized volatility of (Wave daily returns − Benchmark daily returns).",
    "Information Ratio (IR)": "Excess return divided by Tracking Error (risk-adjusted alpha).",
    "Max Drawdown (MaxDD)": "Largest peak-to-trough decline over the period (negative).",
    "VaR 95% (daily)": "Loss threshold where ~5% of days are worse (historical).",
    "CVaR 95% (daily)": "Average loss of the worst ~5% of days (tail risk).",
    "Sharpe": "Risk-adjusted return using total volatility (0% rf here).",
    "Sortino": "Risk-adjusted return using downside deviation only.",
    "Benchmark Snapshot / Drift": "A stable fingerprint of benchmark composition. Drift indicates composition changed in-session (governance signal for context stability).",
    "Coverage Score": "0–100 heuristic of data completeness + freshness (system-feedback indicator for governance).",
    "Difficulty vs SPY": "Concentration/diversification proxy (no-predict architectural reference, not a forecast).",
    "Risk Reaction Score": "0–100 heuristic of risk posture from TE/MaxDD/CVaR.",
    "Analytics Scorecard": "Governance-native reliability grade for analytics outputs (not performance).",
    "Beta (vs Benchmark)": "Regression slope of Wave daily returns vs Benchmark daily returns.",
    "Beta Reliability Score": "0–100: beta-target match + linkage quality (R²) + sample size.",
    "Vector™ Truth Layer": (
        "Read-only governance referee: decomposes alpha sources, reconciles capital-weighted vs exposure-adjusted alpha, "
        "attributes alpha to risk-on/off regimes, and scores durability/fragility. Provides context vs control transparency (no-predict, system-feedback only)."
    ),
    "Vector™ — Truth Referee": (
        "Independent, read-only layer that provides governance verdicts on causality, validates assumptions predictably, "
        "and flags when benchmark-based explanations fail to explain observed outcomes (architectural validation, not forecasting)."
    ),
    "Alpha Classification": (
        "Structural = regime/exposure-driven or benchmark linkage degraded; "
        "Incidental = selection/tilt under stable linkage; Not Present = near-flat alpha. Governance-native context classification."
    ),
    "Assumptions Tested": "Explicit checklist of which standard investment assumptions hold vs break under the wave's regime-aware design (predictable system-feedback).",
    "Gating Warnings": "Governance warnings when data/benchmark/fit integrity fails thresholds. Read-only system-feedback; does not block the app (flexible onboarding).",
    "Wave Purpose Statement": "Plain-English definition of what the Wave is intended to do (positioning + governance context, not a performance prediction).",
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
# VIX Risk Meter™
# ============================================================
@st.cache_data(show_spinner=False, ttl=300)
def get_vix_value() -> Dict[str, Any]:
    """
    Fetch current VIX (^VIX) value from yfinance.
    Returns a dictionary with VIX value, change, and metadata.
    Uses 5-minute cache (TTL=300 seconds) to reduce API load.
    """
    result = {
        "value": None,
        "change": None,
        "prev_close": None,
        "timestamp": None,
        "available": False,
        "error": None,
    }
    
    try:
        if yf is None:
            result["error"] = "yfinance not available"
            return result
        
        # Fetch VIX ticker
        vix = yf.Ticker("^VIX")
        
        # Get current info
        info = vix.info
        if info and "regularMarketPrice" in info:
            result["value"] = float(info.get("regularMarketPrice", 0))
            result["prev_close"] = float(info.get("previousClose", result["value"]))
            result["change"] = result["value"] - result["prev_close"]
            result["timestamp"] = datetime.utcnow().isoformat()
            result["available"] = True
        else:
            # Fallback: get latest from history
            hist = vix.history(period="2d")
            if hist is not None and not hist.empty and len(hist) >= 1:
                result["value"] = float(hist["Close"].iloc[-1])
                if len(hist) >= 2:
                    result["prev_close"] = float(hist["Close"].iloc[-2])
                    result["change"] = result["value"] - result["prev_close"]
                else:
                    result["prev_close"] = result["value"]
                    result["change"] = 0.0
                result["timestamp"] = datetime.utcnow().isoformat()
                result["available"] = True
            else:
                result["error"] = "No VIX data returned from yfinance"
        
    except Exception as e:
        result["error"] = f"VIX fetch error: {str(e)[:100]}"
    
    return result


def get_vix_risk_band(vix_value: float) -> Tuple[str, str, str]:
    """
    Returns (band_name, band_color, band_description) for a given VIX value.
    
    VIX Risk Bands (standard market interpretation):
    - Low Risk: VIX < 12 (very calm market)
    - Normal: 12 <= VIX < 20 (typical market volatility)
    - Elevated: 20 <= VIX < 30 (heightened uncertainty)
    - High: 30 <= VIX < 40 (significant fear)
    - Extreme: VIX >= 40 (panic/crisis mode)
    """
    v = safe_float(vix_value)
    
    if not math.isfinite(v):
        return ("N/A", "rgba(128, 128, 128, 0.15)", "VIX data unavailable")
    
    if v < 12:
        return ("Low Risk", "rgba(46, 213, 115, 0.15)", "Very calm market conditions")
    elif v < 20:
        return ("Normal", "rgba(52, 152, 219, 0.15)", "Typical market volatility")
    elif v < 30:
        return ("Elevated", "rgba(255, 204, 0, 0.15)", "Heightened market uncertainty")
    elif v < 40:
        return ("High", "rgba(255, 140, 0, 0.15)", "Significant market fear")
    else:
        return ("Extreme", "rgba(255, 80, 80, 0.15)", "Panic/crisis mode volatility")


# ============================================================
# Volatility Utilization Index™ (VUI™)
# ============================================================
def get_vix_regime(vix_value: float) -> str:
    """
    Classify VIX value into volatility regime.
    
    Regimes:
    - Calm: VIX < 15
    - Elevated: 15 <= VIX < 25
    - Stress: 25 <= VIX < 35
    - Crisis: VIX >= 35
    """
    v = safe_float(vix_value)
    
    if not math.isfinite(v):
        return "Unknown"
    
    if v < 15:
        return "Calm"
    elif v < 25:
        return "Elevated"
    elif v < 35:
        return "Stress"
    else:
        return "Crisis"


def compute_vui_score(
    hist_sel: pd.DataFrame,
    vix_history: Optional[pd.Series] = None,
    window_days: int = 365
) -> Dict[str, Any]:
    """
    Compute Volatility Utilization Index™ (VUI™) score.
    
    VUI is a 0-100 blended score based on:
    1. Alpha captured in elevated or high VIX conditions
    2. Downside drawdown avoided during elevated or high VIX periods
    3. Recovery speed post-high VIX compared to benchmarks
    
    Returns:
        Dict with VUI score, components, and metadata
    """
    result = {
        "vui_score": float("nan"),
        "alpha_component": float("nan"),
        "protection_component": float("nan"),
        "recovery_component": float("nan"),
        "regime_breakdown": {},
        "available": False,
        "error": None,
    }
    
    try:
        # Validate inputs
        if hist_sel is None or hist_sel.empty:
            result["error"] = "No history data available"
            return result
        
        if vix_history is None or vix_history.empty:
            result["error"] = "No VIX history available"
            return result
        
        # Align VIX history with wave history
        wave_idx = pd.DatetimeIndex(hist_sel.index)
        vix_aligned = vix_history.reindex(wave_idx).ffill().bfill()
        
        if vix_aligned.isna().all():
            result["error"] = "VIX data alignment failed"
            return result
        
        # Classify each day by VIX regime
        regimes = vix_aligned.apply(get_vix_regime)
        
        # Extract returns
        wave_ret = safe_series(hist_sel["wave_ret"]).astype(float).dropna()
        bm_ret = safe_series(hist_sel["bm_ret"]).astype(float).dropna()
        
        if len(wave_ret) < 30 or len(bm_ret) < 30:
            result["error"] = "Insufficient return history"
            return result
        
        # Align all series
        df = pd.DataFrame({
            "wave_ret": wave_ret,
            "bm_ret": bm_ret,
            "regime": regimes
        }).dropna()
        
        if len(df) < 30:
            result["error"] = "Insufficient aligned data"
            return result
        
        # Component 1: Alpha captured in elevated/high VIX conditions
        # Focus on Elevated, Stress, and Crisis regimes
        elevated_mask = df["regime"].isin(["Elevated", "Stress", "Crisis"])
        
        if elevated_mask.sum() > 0:
            elevated_wave = (1 + df.loc[elevated_mask, "wave_ret"]).prod() - 1
            elevated_bm = (1 + df.loc[elevated_mask, "bm_ret"]).prod() - 1
            elevated_alpha = elevated_wave - elevated_bm
            
            # Normalize to 0-40 scale (40% weight)
            # Positive alpha in high vol = good, cap at +20% for max score
            alpha_component = float(np.clip((elevated_alpha / 0.20) * 40, 0, 40))
        else:
            alpha_component = 20.0  # Neutral if no elevated periods
        
        # Component 2: Downside protection during elevated/high VIX
        # Measure relative drawdown avoidance
        if elevated_mask.sum() > 0:
            elevated_wave_dd = df.loc[elevated_mask, "wave_ret"].clip(upper=0).sum()
            elevated_bm_dd = df.loc[elevated_mask, "bm_ret"].clip(upper=0).sum()
            
            # Protection score: how much less did we lose vs benchmark?
            protection_diff = elevated_bm_dd - elevated_wave_dd  # positive if we lost less
            
            # Normalize to 0-30 scale (30% weight)
            # If we avoided 10% more downside than benchmark = max score
            protection_component = float(np.clip((protection_diff / 0.10) * 30, 0, 30))
        else:
            protection_component = 15.0  # Neutral if no elevated periods
        
        # Component 3: Recovery speed post-high VIX
        # Find transitions from Stress/Crisis to Calm/Elevated
        df["prev_regime"] = df["regime"].shift(1)
        recovery_mask = (
            df["prev_regime"].isin(["Stress", "Crisis"]) & 
            df["regime"].isin(["Calm", "Elevated"])
        )
        
        if recovery_mask.sum() > 5:
            # Average 30-day return after high VIX episodes
            recovery_returns_wave = []
            recovery_returns_bm = []
            
            for idx in df[recovery_mask].index:
                idx_pos = df.index.get_loc(idx)
                end_pos = min(idx_pos + 30, len(df))
                
                if end_pos - idx_pos >= 10:  # At least 10 days
                    wave_recovery = (1 + df.iloc[idx_pos:end_pos]["wave_ret"]).prod() - 1
                    bm_recovery = (1 + df.iloc[idx_pos:end_pos]["bm_ret"]).prod() - 1
                    recovery_returns_wave.append(wave_recovery)
                    recovery_returns_bm.append(bm_recovery)
            
            if recovery_returns_wave and recovery_returns_bm:
                avg_wave_recovery = float(np.mean(recovery_returns_wave))
                avg_bm_recovery = float(np.mean(recovery_returns_bm))
                recovery_alpha = avg_wave_recovery - avg_bm_recovery
                
                # Normalize to 0-30 scale (30% weight)
                # +5% better recovery = max score
                recovery_component = float(np.clip((recovery_alpha / 0.05) * 30, 0, 30))
            else:
                recovery_component = 15.0  # Neutral
        else:
            recovery_component = 15.0  # Neutral if too few recovery events
        
        # Combine components (40% + 30% + 30% = 100%)
        vui_score = alpha_component + protection_component + recovery_component
        vui_score = float(np.clip(vui_score, 0, 100))
        
        # Regime breakdown for transparency
        regime_breakdown = {}
        for regime in ["Calm", "Elevated", "Stress", "Crisis"]:
            regime_mask = df["regime"] == regime
            if regime_mask.sum() > 0:
                regime_breakdown[regime] = {
                    "days": int(regime_mask.sum()),
                    "wave_ret": float((1 + df.loc[regime_mask, "wave_ret"]).prod() - 1),
                    "bm_ret": float((1 + df.loc[regime_mask, "bm_ret"]).prod() - 1),
                    "alpha": float((1 + df.loc[regime_mask, "wave_ret"]).prod() - 1 -
                                 (1 + df.loc[regime_mask, "bm_ret"]).prod() + 1),
                }
        
        result.update({
            "vui_score": vui_score,
            "alpha_component": alpha_component,
            "protection_component": protection_component,
            "recovery_component": recovery_component,
            "regime_breakdown": regime_breakdown,
            "available": True,
        })
        
    except Exception as e:
        result["error"] = f"VUI calculation error: {str(e)[:100]}"
    
    return result


def get_vui_band(vui_score: float) -> Tuple[str, str, str]:
    """
    Returns (band_name, band_color, band_description) for VUI score.
    
    VUI Bands:
    - Weaponized (>80): Outstanding volatility response optimization
    - Adaptive (60-80): Good volatility utilization
    - Neutral (40-60): Moderate volatility management
    - Weak (<40): Poor volatility utilization
    """
    v = safe_float(vui_score)
    
    if not math.isfinite(v):
        return ("N/A", "rgba(128, 128, 128, 0.15)", "VUI data unavailable")
    
    if v > 80:
        return ("Weaponized", "rgba(46, 213, 115, 0.15)", "Outstanding volatility response optimization")
    elif v >= 60:
        return ("Adaptive", "rgba(52, 152, 219, 0.15)", "Good volatility utilization")
    elif v >= 40:
        return ("Neutral", "rgba(255, 204, 0, 0.15)", "Moderate volatility management")
    else:
        return ("Weak", "rgba(255, 140, 0, 0.15)", "Poor volatility utilization")


def render_vui_meter(hist_sel: pd.DataFrame, vix_history: Optional[pd.Series] = None):
    """
    Render Volatility Utilization Index™ (VUI™) with detailed breakdown.
    Designed to match existing dark theme styling and placed under VIX Risk Meter.
    """
    if not ENABLE_YFINANCE_CHIPS or yf is None:
        return
    
    try:
        # Compute VUI
        vui_data = compute_vui_score(hist_sel, vix_history)
        
        st.markdown('<div class="waves-card">', unsafe_allow_html=True)
        st.markdown("#### Volatility Utilization Index™ (VUI™)")
        
        if not vui_data.get("available"):
            # Fallback display when VUI data is unavailable
            st.markdown('<div class="waves-tile">', unsafe_allow_html=True)
            st.markdown('<div class="waves-tile-label">VUI Score (0-100)</div>', unsafe_allow_html=True)
            st.markdown('<div class="waves-tile-value">—</div>', unsafe_allow_html=True)
            st.markdown(
                f'<div class="waves-tile-sub">Data unavailable: {vui_data.get("error", "Unknown error")}</div>',
                unsafe_allow_html=True
            )
            st.markdown("</div>", unsafe_allow_html=True)
            st.caption("VUI data temporarily unavailable. Requires VIX history and wave returns.")
        else:
            # Normal display with VUI value
            vui_score = vui_data.get("vui_score", 0)
            band_name, band_color, band_desc = get_vui_band(vui_score)
            
            # Create colored tile based on VUI band
            st.markdown(
                f"""
<div class="waves-tile" style="background: {band_color}; border-color: {band_color.replace('0.15', '0.30')};">
  <div class="waves-tile-label">Volatility Utilization Index™</div>
  <div class="waves-tile-value">{vui_score:.1f}</div>
  <div class="waves-tile-sub">
    {band_name} · {band_desc}
  </div>
</div>
""",
                unsafe_allow_html=True
            )
            
            # Component breakdown
            st.caption("**VUI Components:**")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(
                    "Alpha Capture (40%)",
                    f"{vui_data.get('alpha_component', 0):.1f}/40"
                )
            with col2:
                st.metric(
                    "Downside Protection (30%)",
                    f"{vui_data.get('protection_component', 0):.1f}/30"
                )
            with col3:
                st.metric(
                    "Recovery Speed (30%)",
                    f"{vui_data.get('recovery_component', 0):.1f}/30"
                )
            
            # Additional context
            with st.expander("VUI™ Methodology & Regime Breakdown", expanded=False):
                st.markdown("""
**What is VUI™?**
The Volatility Utilization Index™ evaluates how effectively the WAVES system utilizes market volatility to generate protected upside. Unlike raw VIX, VUI is a specialized metric that measures:

1. **Alpha Captured (40%)**: Performance advantage during elevated/high VIX conditions
2. **Downside Protection (30%)**: Drawdown avoidance vs benchmark in volatile periods
3. **Recovery Speed (30%)**: Faster recovery post-crisis vs benchmarks

**VUI Bands:**
- **Weaponized (>80):** Outstanding response optimization — system thrives in volatility
- **Adaptive (60-80):** Good volatility utilization — solid risk-adjusted performance
- **Neutral (40-60):** Moderate management — basic volatility handling
- **Weak (<40):** Poor utilization — underperforming in volatile conditions

**Volatility Regimes (VIX-based):**
- **Calm:** VIX < 15 (low volatility environment)
- **Elevated:** 15 ≤ VIX < 25 (heightened uncertainty)
- **Stress:** 25 ≤ VIX < 35 (significant market stress)
- **Crisis:** VIX ≥ 35 (extreme volatility/panic)
                """)
                
                # Show regime breakdown if available
                regime_breakdown = vui_data.get("regime_breakdown", {})
                if regime_breakdown:
                    st.markdown("**Regime Performance Breakdown:**")
                    for regime in ["Calm", "Elevated", "Stress", "Crisis"]:
                        if regime in regime_breakdown:
                            rb = regime_breakdown[regime]
                            st.write(
                                f"**{regime}:** {rb['days']} days · "
                                f"Wave Return {fmt_pct(rb['wave_ret'])} · "
                                f"Alpha {fmt_pct(rb['alpha'])}"
                            )
        
        st.markdown("</div>", unsafe_allow_html=True)
        
    except Exception as e:
        # Ultimate fallback: silent failure or minimal error display
        st.caption(f"VUI™ temporarily unavailable: {str(e)[:80]}")


# ============================================================
# WAVES Ticker Tape™ Content Generator
# ============================================================
def generate_ticker_content(
    selected_wave: str,
    mode: str,
    metrics: Dict[str, Any],
    bm_display: str,
    vix_data: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Generate HTML content for the scrolling ticker tape.
    Includes VIX, movers, wave details, and risk regime info.
    Returns HTML string with ticker items.
    """
    items = []
    
    # 1. VIX Risk Meter (if available)
    if vix_data and vix_data.get("available"):
        vix_value = vix_data.get("value", 0)
        vix_change = vix_data.get("change", 0)
        band_name, _, _ = get_vix_risk_band(vix_value)
        
        vix_class = "positive" if vix_change < 0 else "negative" if vix_change > 0 else "neutral"
        vix_sign = "+" if vix_change >= 0 else ""
        items.append(
            f'<div class="waves-ticker-item {vix_class}">'
            f'<span class="ticker-label">VIX:</span>'
            f'<span class="ticker-value">{vix_value:.2f} ({vix_sign}{vix_change:.2f}) {band_name}</span>'
            f'</div>'
        )
    
    # 2. Selected Wave Performance
    if selected_wave and selected_wave != "(none)":
        # 30D Alpha
        a30 = safe_float(metrics.get("a30", 0))
        if math.isfinite(a30):
            alpha_class = "positive" if a30 > 0 else "negative" if a30 < 0 else "neutral"
            items.append(
                f'<div class="waves-ticker-item {alpha_class}">'
                f'<span class="ticker-label">Selected Wave ({selected_wave}):</span>'
                f'<span class="ticker-value">30D α {fmt_pct(a30)}</span>'
                f'</div>'
            )
        
        # 60D Alpha
        a60 = safe_float(metrics.get("a60", 0))
        if math.isfinite(a60):
            alpha_class = "positive" if a60 > 0 else "negative" if a60 < 0 else "neutral"
            items.append(
                f'<div class="waves-ticker-item {alpha_class}">'
                f'<span class="ticker-label">60D α:</span>'
                f'<span class="ticker-value">{fmt_pct(a60)}</span>'
                f'</div>'
            )
        
        # Tracking Error
        te = safe_float(metrics.get("te", 0))
        if math.isfinite(te):
            te_class = "neutral" if te < 0.15 else "negative"
            items.append(
                f'<div class="waves-ticker-item {te_class}">'
                f'<span class="ticker-label">TE:</span>'
                f'<span class="ticker-value">{fmt_pct(te)} ({te_risk_band(te)})</span>'
                f'</div>'
            )
        
        # Max Drawdown
        mdd = safe_float(metrics.get("mdd", 0))
        if math.isfinite(mdd):
            mdd_class = "positive" if mdd > -0.10 else "negative"
            items.append(
                f'<div class="waves-ticker-item {mdd_class}">'
                f'<span class="ticker-label">MaxDD:</span>'
                f'<span class="ticker-value">{fmt_pct(mdd)}</span>'
                f'</div>'
            )
    
    # 3. Mode and Benchmark Info
    items.append(
        f'<div class="waves-ticker-item neutral">'
        f'<span class="ticker-label">Mode:</span>'
        f'<span class="ticker-value">{mode}</span>'
        f'</div>'
    )
    
    items.append(
        f'<div class="waves-ticker-item neutral">'
        f'<span class="ticker-label">Benchmark:</span>'
        f'<span class="ticker-value">{bm_display}</span>'
        f'</div>'
    )
    
    # 4. Market Movers (if yfinance available)
    if ENABLE_YFINANCE_CHIPS and yf is not None:
        try:
            # Get quick market snapshot
            tickers_snapshot = ["SPY", "QQQ", "IWM"]
            px = fetch_prices_daily(tickers_snapshot, days=5)
            
            if px is not None and not px.empty:
                for ticker in tickers_snapshot:
                    if ticker in px.columns:
                        s = px[ticker].dropna()
                        if len(s) >= 2:
                            day_ret = (s.iloc[-1] / s.iloc[-2] - 1.0)
                            ret_class = "positive" if day_ret > 0 else "negative" if day_ret < 0 else "neutral"
                            sign = "+" if day_ret >= 0 else ""
                            items.append(
                                f'<div class="waves-ticker-item {ret_class}">'
                                f'<span class="ticker-label">{ticker}:</span>'
                                f'<span class="ticker-value">{sign}{day_ret*100:.2f}%</span>'
                                f'</div>'
                            )
        except Exception:
            # Silent fallback if market movers fail
            pass
    
    # 5. Timestamp
    now = datetime.utcnow()
    items.append(
        f'<div class="waves-ticker-item neutral">'
        f'<span class="ticker-label">Updated:</span>'
        f'<span class="ticker-value">{now.strftime("%H:%M UTC")}</span>'
        f'</div>'
    )
    
    # Double the items for seamless loop
    ticker_html = "".join(items) + "".join(items)
    
    return ticker_html


def render_vix_risk_meter():
    """
    Render VIX Risk Meter™ with multi-band logic and fallback safety.
    Designed to match existing dark theme styling.
    """
    if not ENABLE_YFINANCE_CHIPS or yf is None:
        return
    
    try:
        vix_data = get_vix_value()
        
        st.markdown('<div class="waves-card">', unsafe_allow_html=True)
        st.markdown("#### VIX Risk Meter™")
        
        if not vix_data.get("available"):
            # Fallback display when VIX data is unavailable
            st.markdown('<div class="waves-tile">', unsafe_allow_html=True)
            st.markdown('<div class="waves-tile-label">VIX (Volatility Index)</div>', unsafe_allow_html=True)
            st.markdown('<div class="waves-tile-value">—</div>', unsafe_allow_html=True)
            st.markdown(
                f'<div class="waves-tile-sub">Data unavailable: {vix_data.get("error", "Unknown error")}</div>',
                unsafe_allow_html=True
            )
            st.markdown("</div>", unsafe_allow_html=True)
            st.caption("VIX data temporarily unavailable. Market intel continues with other tickers.")
        else:
            # Normal display with VIX value
            vix_value = vix_data.get("value", 0)
            vix_change = vix_data.get("change", 0)
            band_name, band_color, band_desc = get_vix_risk_band(vix_value)
            
            # Create colored tile based on risk band
            st.markdown(
                f"""
<div class="waves-tile" style="background: {band_color}; border-color: {band_color.replace('0.15', '0.30')};">
  <div class="waves-tile-label">VIX (Volatility Index)</div>
  <div class="waves-tile-value">{vix_value:.2f}</div>
  <div class="waves-tile-sub">
    {'+' if vix_change >= 0 else ''}{vix_change:.2f} · {band_name}
  </div>
</div>
""",
                unsafe_allow_html=True
            )
            
            # Risk interpretation
            st.caption(f"**{band_name}:** {band_desc}")
            
            # Additional context
            with st.expander("VIX Risk Meter™ Guide", expanded=False):
                st.markdown("""
**What is VIX?**
The CBOE Volatility Index (VIX) measures the market's expectation of 30-day volatility based on S&P 500 index options. Often called the "fear gauge."

**Risk Bands:**
- **Low Risk (VIX < 12):** Very calm, complacent markets
- **Normal (12-20):** Typical market volatility range
- **Elevated (20-30):** Heightened uncertainty, caution warranted
- **High (30-40):** Significant fear, major market stress
- **Extreme (VIX ≥ 40):** Panic mode, crisis-level volatility

**Trading Implications:**
- Rising VIX typically signals increased hedging demand and market uncertainty
- VIX > 30 often coincides with market corrections or crashes
- VIX < 12 may indicate complacency and potential for sudden volatility spikes

**Data Source:** Yahoo Finance (^VIX ticker), cached for 5 minutes.
                """)
        
        st.markdown("</div>", unsafe_allow_html=True)
        
    except Exception as e:
        # Ultimate fallback: silent failure or minimal error display
        st.caption(f"VIX Risk Meter™ temporarily unavailable: {str(e)[:80]}")


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


def benchmark_name(wave_name: str, bm_mix_df: pd.DataFrame) -> str:
    """
    Returns a human-readable benchmark name.
    For simple benchmarks (1 ticker): returns "Ticker (Name)"
    For composite benchmarks: returns detailed component list with weights
    """
    try:
        rows = _bm_rows_for_wave(bm_mix_df, wave_name)
        if rows.empty:
            return "Benchmark not available"
        
        # Get ticker names if available from bm_mix_df
        ticker_names = {}
        if "Name" in bm_mix_df.columns and "Ticker" in bm_mix_df.columns:
            name_map = bm_mix_df[["Ticker", "Name"]].drop_duplicates()
            ticker_names = dict(zip(name_map["Ticker"], name_map["Name"]))
        
        if len(rows) == 1:
            ticker = rows.iloc[0]["Ticker"]
            name = ticker_names.get(ticker, "")
            if name:
                return f"{ticker} ({name})"
            return ticker
        else:
            # Composite benchmark: show all components
            components = []
            for _, row in rows.iterrows():
                ticker = row["Ticker"]
                weight = row["Weight"]
                name = ticker_names.get(ticker, "")
                if name:
                    components.append(f"{ticker} ({name}): {weight*100:.1f}%")
                else:
                    components.append(f"{ticker}: {weight*100:.1f}%")
            return "Composite: " + " | ".join(components)
    except Exception:
        return "Benchmark error"


def benchmark_snapshot_id(wave_name: str, mode: str, bm_mix_df: pd.DataFrame) -> str:
    """
    Generates a stable, deterministic benchmark ID that includes:
    - Wave name
    - Mode (Standard/Alpha-Minus-Beta/Private Logic)
    - Benchmark tickers (sorted alphabetically)
    - Benchmark weights (rounded to 8 decimals for consistency)
    
    The ID is stable across page refreshes as long as the benchmark definition doesn't change.
    """
    try:
        rows = _bm_rows_for_wave(bm_mix_df, wave_name)
        if rows.empty:
            return "BM-NA"
        
        # Create canonical payload with wave name, mode, and sorted tickers with weights
        # Sort by ticker to ensure deterministic ordering
        sorted_rows = rows.sort_values("Ticker").reset_index(drop=True)
        ticker_weight_pairs = [f"{r.Ticker}:{r.Weight:.8f}" for r in sorted_rows.itertuples(index=False)]
        
        # Include wave name and mode in the hash for full context
        payload = f"wave={wave_name}|mode={mode}|benchmark={','.join(ticker_weight_pairs)}"
        
        # Use SHA256 for better collision resistance, take first 12 chars
        h = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12].upper()
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


def benchmark_display_description(bm_name: str, bm_drift: str, bm_rows: pd.DataFrame) -> str:
    """
    Returns a human-readable benchmark description for institutional presentation.
    Focuses on governance and stability messaging without exposing internal IDs.
    
    Args:
        bm_name: The benchmark name (e.g., "SPY (S&P 500)" or "Composite: ...")
        bm_drift: The drift status ("stable" or "drift")
        bm_rows: DataFrame with benchmark composition
    
    Returns:
        Human-readable description like "Benchmark Snapshot: Stable (Governed Composite)"
    """
    try:
        # Determine if single-ticker or composite
        is_composite = len(bm_rows) > 1 if not bm_rows.empty else False
        
        # Build status descriptor
        if bm_drift.lower() == "stable":
            status = "Stable"
        else:
            status = "Drift Detected"
        
        # Build type descriptor
        if is_composite:
            type_desc = "Governed Composite"
        else:
            type_desc = "Single Benchmark"
        
        # Combine into institutional-friendly description
        return f"Benchmark Snapshot: {status} ({type_desc})"
    except Exception:
        return "Benchmark Snapshot: Status Unknown"


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
            return ("High", "Fresh + complete history and stable benchmark snapshot (predictable governance context).")
        if drift or (math.isfinite(score) and score < 75) or (math.isfinite(age) and age >= 7) or rows < 60:
            return ("Low", "Governance issues detected: " + ", ".join(issues) + " (architectural stability requirements not met).")
        return ("Medium", "Caution flags present: " + ", ".join(issues) + " (governance attention recommended).")
    except Exception:
        return ("Medium", "Confidence heuristic unavailable (non-fatal; flexible system-feedback).")


# ============================================================
# Governance-native Analytics Scorecard (selected wave)
# ============================================================
def _score_to_grade_af(score: float) -> str:
    """Returns numeric score with band label for Analytics Score.
    
    Score bands:
    - Elite: 90-100
    - Strong: 80-89
    - Good: 70-79
    - Watch: 60-69
    - Risk: below 60
    """
    s = safe_float(score)
    if not math.isfinite(s):
        return "N/A"
    # Return formatted numeric score with band
    if s >= 90:
        return f"{s:.1f} (Elite)"
    if s >= 80:
        return f"{s:.1f} (Strong)"
    if s >= 70:
        return f"{s:.1f} (Good)"
    if s >= 60:
        return f"{s:.1f} (Watch)"
    return f"{s:.1f} (Risk)"


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

        # D2 Benchmark fidelity
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

        # D5 Decision readiness
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
# Deterministic AI Explanation Layer (rules-based, governance-native)
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

    if str(bm_drift).lower().strip() != "stable":
        out["What changed recently"].append("Benchmark snapshot drift detected in-session (composition changed; governance stability signal).")
    if math.isfinite(age) and age >= 3:
        out["What changed recently"].append(f"History freshness: last datapoint is {int(age)} day(s) old.")
    if rows < 90:
        out["What changed recently"].append("Limited history window may reduce stability of risk metrics (architectural context limitation).")
    if not out["What changed recently"]:
        out["What changed recently"].append("No major governance changes detected (stable benchmark + fresh coverage; predictable context).")

    if math.isfinite(a30):
        if a30 > 0.02:
            out["Why the alpha looks like this"].append("Positive 30D alpha: wave has recently outperformed its benchmark mix (favorable context).")
        elif a30 < -0.02:
            out["Why the alpha looks like this"].append("Negative 30D alpha: recent underperformance versus benchmark mix (context vs control signal).")
        else:
            out["Why the alpha looks like this"].append("30D alpha near flat: wave and benchmark moving similarly (stable linkage context).")
    if math.isfinite(te):
        out["Why the alpha looks like this"].append(f"Active risk (TE {fmt_pct(te)}) is {te_risk_band(te)}; alpha magnitude often scales with active risk (architectural governance relationship).")

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

    if math.isfinite(cov_score) and cov_score < 85:
        out["What to verify"].append("Coverage score < 85: inspect missing business days + pipeline consistency (governance hygiene).")
    if str(bm_drift).lower().strip() != "stable":
        out["What to verify"].append("Benchmark drift: stabilize benchmark mix for governance (architectural stability requirement).")
    if math.isfinite(a30) and abs(a30) >= 0.08:
        out["What to verify"].append("Large alpha: confirm benchmark weights + verify no gaps/rollovers in history (context validation).")
    if math.isfinite(beta_score) and beta_score < 75:
        out["What to verify"].append("Beta reliability low: benchmark may not match systematic exposure (verify architectural fit).")
    if not out["What to verify"]:
        out["What to verify"].append("No critical verification flags triggered (stable governance context).")

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
# Vector™ Alpha Enhancements (exposure + regimes + capture)
# ============================================================
def _try_get_exposure_series(wave_name: str, mode: str, hist_index: pd.DatetimeIndex) -> Optional[pd.Series]:
    """
    Optional: if engine exposes exposure history (0..1), use it.
    Otherwise return None and we fall back gracefully.
    """
    try:
        if we is None:
            return None
        candidates = ["get_exposure_series", "compute_exposure_series", "get_exposure_history", "exposure_series"]
        for fn in candidates:
            if hasattr(we, fn):
                f = getattr(we, fn)
                try:
                    try:
                        s = f(wave_name, mode=mode)
                    except TypeError:
                        s = f(wave_name, mode)
                    if isinstance(s, (pd.Series, pd.DataFrame)):
                        if isinstance(s, pd.DataFrame):
                            num_cols = [c for c in s.columns if np.issubdtype(s[c].dtype, np.number)]
                            if num_cols:
                                s = s[num_cols[0]]
                            else:
                                continue
                        s = pd.to_numeric(s, errors="coerce")
                        s.index = pd.to_datetime(s.index, errors="coerce")
                        s = s.dropna()
                        if len(s) == 0:
                            continue
                        s = s.reindex(hist_index).ffill().bfill()
                        s = s.clip(lower=0.0, upper=1.0)
                        return s
                except Exception:
                    continue
        return None
    except Exception:
        return None


def _build_regime_series_from_benchmark(hist_sel: pd.DataFrame) -> Optional[pd.Series]:
    """
    Deterministic, simple regime labeling:
      • RISK_OFF when benchmark daily return < 0
      • RISK_ON otherwise
    """
    try:
        if hist_sel is None or hist_sel.empty or "bm_ret" not in hist_sel.columns:
            return None
        b = pd.to_numeric(hist_sel["bm_ret"], errors="coerce")
        reg = np.where(b.fillna(0.0).values < 0.0, "RISK_OFF", "RISK_ON")
        return pd.Series(reg, index=hist_sel.index)
    except Exception:
        return None


def _compound_from_daily(daily: pd.Series) -> float:
    d = pd.to_numeric(daily, errors="coerce").dropna()
    if len(d) < 2:
        return float("nan")
    return float((1.0 + d).prod() - 1.0)


def _alpha_capture_series(hist: pd.DataFrame, wave_name: str, mode: str) -> pd.Series:
    """
    Alpha Capture daily series:
      daily_alpha = wave_ret - bm_ret
      if exposure exists: daily_alpha / max(0.10, exposure)
    """
    if hist is None or hist.empty:
        return pd.Series(dtype=float)
    w = pd.to_numeric(hist.get("wave_ret"), errors="coerce")
    b = pd.to_numeric(hist.get("bm_ret"), errors="coerce")
    da = (w - b).dropna()
    if len(da) == 0:
        return pd.Series(dtype=float)

    exp = _try_get_exposure_series(wave_name, mode, pd.DatetimeIndex(hist.index))
    if exp is None:
        return da
    exp = pd.to_numeric(exp, errors="coerce").reindex(da.index).ffill().bfill()
    exp = exp.clip(lower=0.10, upper=1.0)
    return (da / exp).replace([np.inf, -np.inf], np.nan).dropna()


def _risk_on_off_attrib(hist: pd.DataFrame, wave_name: str, mode: str, window: int = 60) -> Dict[str, Any]:
    """
    Returns capital-weighted alpha + exposure-adjusted alpha + risk-on/off split for a window.
    """
    out = {
        "cap_alpha": np.nan,
        "exp_adj_alpha": np.nan,
        "risk_on_alpha": np.nan,
        "risk_off_alpha": np.nan,
        "risk_on_share": np.nan,
        "risk_off_share": np.nan,
    }
    if hist is None or hist.empty:
        return out

    h = hist.tail(max(2, int(window))).copy()
    if h.empty or len(h) < 2:
        return out

    # capital-weighted alpha (window)
    try:
        cap_alpha = ret_from_nav(h["wave_nav"], len(h)) - ret_from_nav(h["bm_nav"], len(h))
    except Exception:
        cap_alpha = float("nan")
    out["cap_alpha"] = cap_alpha

    # exposure-adjusted alpha (window)
    exp = _try_get_exposure_series(wave_name, mode, pd.DatetimeIndex(h.index))
    if exp is None:
        out["exp_adj_alpha"] = cap_alpha
    else:
        exp_num = pd.to_numeric(exp, errors="coerce").dropna()
        avg_exp = float(exp_num.mean()) if len(exp_num) else 1.0
        avg_exp = max(0.10, min(1.0, avg_exp))
        out["exp_adj_alpha"] = cap_alpha / avg_exp if math.isfinite(cap_alpha) else float("nan")

    # risk-on/off attribution using alpha-capture daily series (compounded)
    ac = _alpha_capture_series(h, wave_name, mode)
    if len(ac) >= 2:
        reg = _build_regime_series_from_benchmark(h)
        if reg is not None:
            reg = reg.reindex(ac.index)
            risk_on = ac[reg == "RISK_ON"]
            risk_off = ac[reg == "RISK_OFF"]
            ro = _compound_from_daily(risk_on) if len(risk_on) >= 2 else float("nan")
            rf = _compound_from_daily(risk_off) if len(risk_off) >= 2 else float("nan")
            out["risk_on_alpha"] = ro
            out["risk_off_alpha"] = rf
            tot = 0.0
            if math.isfinite(ro):
                tot += ro
            if math.isfinite(rf):
                tot += rf
            if tot != 0.0:
                out["risk_on_share"] = ro / tot if math.isfinite(ro) else np.nan
                out["risk_off_share"] = rf / tot if math.isfinite(rf) else np.nan

    return out


# ============================================================
# Vector™ Truth Panel (existing module; boot-safe)
# ============================================================
def _vector_truth_panel(selected_wave: str, mode: str, hist_sel: pd.DataFrame, metrics: Dict[str, Any], days: int):
    if not ENABLE_VECTOR_TRUTH or build_vector_truth_report is None or format_vector_truth_markdown is None:
        st.info(f"Vector™ Truth Layer unavailable (import issue): {VECTOR_TRUTH_IMPORT_ERROR}")
        return

    if hist_sel is None or hist_sel.empty:
        st.info("Vector™ Truth Layer: no canonical history available.")
        return

    alpha_series = (
        pd.to_numeric(hist_sel["wave_ret"], errors="coerce") - pd.to_numeric(hist_sel["bm_ret"], errors="coerce")
    ).dropna()
    alpha_series = alpha_series.values.tolist() if len(alpha_series) else None

    reg_series = _build_regime_series_from_benchmark(hist_sel)
    regime_series = reg_series.values.tolist() if reg_series is not None and len(reg_series) else None

    cap_alpha = safe_float(metrics.get("a365"))
    if not math.isfinite(cap_alpha):
        try:
            cap_alpha = ret_from_nav(hist_sel["wave_nav"], len(hist_sel)) - ret_from_nav(hist_sel["bm_nav"], len(hist_sel))
        except Exception:
            cap_alpha = float("nan")

    exp_series = _try_get_exposure_series(selected_wave, mode, pd.DatetimeIndex(hist_sel.index))
    exp_adj_alpha = cap_alpha
    if exp_series is not None:
        exp_num = pd.to_numeric(exp_series, errors="coerce").dropna()
        avg_exp = float(exp_num.mean()) if len(exp_num) else 1.0
        avg_exp = max(0.10, min(1.0, avg_exp))
        if math.isfinite(cap_alpha):
            exp_adj_alpha = cap_alpha / avg_exp

    report = build_vector_truth_report(
        wave_name=str(selected_wave),
        timeframe_label=f"{min(int(days), int(len(hist_sel)))}D window",
        total_excess_return=cap_alpha,
        capital_weighted_alpha=cap_alpha,
        exposure_adjusted_alpha=exp_adj_alpha,
        alpha_series=alpha_series,
        regime_series=regime_series,
    )

    st.markdown(format_vector_truth_markdown(report))

    with st.expander("Vector™ Truth Notes (Method)"):
        st.write(
            "Capital-weighted alpha uses canonical Wave vs Benchmark performance (investor-experience). "
            "Exposure-adjusted alpha normalizes by average exposure if exposure history is available from the engine; "
            "otherwise it defaults to capital-weighted alpha (no inflation). "
            "Risk-on/off regimes are labeled deterministically from benchmark daily return sign unless a richer regime feed exists."
        )


# ============================================================
# Vector™ Referee (Deterministic, Governance-Native)
# ============================================================
def _alpha_classification(cap_alpha: float, beta_score: float, bm_drift: str, rr_score: float) -> str:
    a = safe_float(cap_alpha)
    bs = safe_float(beta_score)
    rr = safe_float(rr_score)
    drift = (str(bm_drift).lower().strip() != "stable")

    if not math.isfinite(a):
        return "Unknown"
    if abs(a) < 0.005:
        return "Not Present"
    if drift:
        return "Structural"
    if math.isfinite(bs) and bs < 75:
        return "Structural"
    if math.isfinite(rr) and rr >= 75:
        return "Structural"
    return "Incidental"


def _primary_alpha_source(mode: str, cap_alpha: float, exp_adj_alpha: float, beta_score: float) -> str:
    a = safe_float(cap_alpha)
    ea = safe_float(exp_adj_alpha)
    bs = safe_float(beta_score)

    if not math.isfinite(a):
        return "Unknown"
    if abs(a) < 0.005:
        return "No dominant alpha source detected (near-flat; predictable system-feedback)."
    if math.isfinite(ea) and math.isfinite(a) and (ea > a * 1.20):
        return "Adaptive Exposure Control (SmartSafe / VIX / risk gating) — governance-native architectural mechanism"
    if math.isfinite(bs) and bs < 75:
        return "Regime/Exposure effects (benchmark linkage degraded by design; architectural governance choice)"
    if "alpha-minus-beta" in str(mode).lower():
        return "Beta-managed alpha (alpha-minus-beta discipline; stable governance framework)"
    return "Selection/tilt vs benchmark mix (within stable linkage; predictable context)"


def _assumption_checklist(bm_drift: str, beta_score: float) -> List[Tuple[str, bool, str]]:
    drift_ok = (str(bm_drift).lower().strip() == "stable")
    bs = safe_float(beta_score)

    out: List[Tuple[str, bool, str]] = []
    out.append(("Fully-invested benchmark assumption", False, "Wave may use exposure control / cash sweeps; benchmark may be fully invested (architectural governance flexibility)."))
    out.append(("Stable beta assumption", math.isfinite(bs) and bs >= 80, "Beta reliability indicates linkage quality; low score often means mismatch (expected or needs review; context vs control)."))
    out.append(("Linear risk-return assumption", False, "Regime gating + nonlinear exposure breaks linear assumptions (by design; governance-native approach)."))
    out.append(("Regime-aware exposure control", True, "System is explicitly designed to vary exposure across regimes (predictable architectural mechanism)."))
    out.append(("Capital preservation priority", True, "Risk controls (TE/MaxDD/CVaR) are first-class governance signals (stable system-feedback)."))
    out.append(("Benchmark-anchored governance", drift_ok, "Benchmark snapshot stability is required for clean comparisons (drift = governance flag; architectural constraint)."))
    return out


def _vector_failure_flags(metrics: Dict[str, Any], cov: Dict[str, Any], bm_drift: str, beta_score: float) -> List[str]:
    flags: List[str] = []
    a365 = safe_float(metrics.get("a365"))
    a30 = safe_float(metrics.get("a30"))
    te = safe_float(metrics.get("te"))
    mdd = safe_float(metrics.get("mdd"))
    cvar95 = safe_float(metrics.get("cvar95"))
    cov_score = safe_float(cov.get("completeness_score"))
    age = safe_float(cov.get("age_days"))
    bs = safe_float(beta_score)

    if math.isfinite(a365) and a365 < -0.02:
        flags.append("Persistent negative alpha (365D) vs benchmark (governance signal for review).")
    if math.isfinite(a30) and a30 < -0.04:
        flags.append("Short-term underperformance (30D) requires context review (system-feedback).")
    if str(bm_drift).lower().strip() != "stable":
        flags.append("Benchmark drift detected — stabilize benchmark mix for governance (architectural stability constraint).")
    if math.isfinite(bs) and bs < 65:
        flags.append("Very low beta reliability — benchmark may not explain systematic exposure (verify architectural fit).")
    if math.isfinite(te) and te >= 0.22:
        flags.append("High active risk (TE) — review exposure caps and SmartSafe governance posture.")
    if math.isfinite(mdd) and mdd <= -0.25:
        flags.append("Deep drawdown observed — resilience review recommended (governance context).")
    if math.isfinite(cvar95) and cvar95 <= -0.03:
        flags.append("Tail risk elevated (CVaR) — stress review recommended (predictable system-feedback).")
    if math.isfinite(cov_score) and cov_score < 85:
        flags.append("Coverage score < 85 — history integrity check recommended (stable governance requirement).")
    if math.isfinite(age) and age >= 5:
        flags.append("Data staleness (≥5 days) — pipeline freshness check recommended (governance hygiene).")
    return flags


def _vector_referee_verdict_block(
    selected_wave: str,
    mode: str,
    hist_sel: pd.DataFrame,
    metrics: Dict[str, Any],
    cov: Dict[str, Any],
    bm_drift: str,
    beta_val: float,
    beta_r2: float,
    beta_n: int,
    beta_score: float,
    beta_grade: str,
    rr_score: float,
):
    cap_alpha = safe_float(metrics.get("a365"))
    if not math.isfinite(cap_alpha):
        cap_alpha = safe_float(metrics.get("a60"))
    if not math.isfinite(cap_alpha):
        cap_alpha = safe_float(metrics.get("a30"))

    exp_adj_alpha = cap_alpha
    try:
        if hist_sel is not None and not hist_sel.empty:
            exp_series = _try_get_exposure_series(selected_wave, mode, pd.DatetimeIndex(hist_sel.index))
            if exp_series is not None:
                exp_num = pd.to_numeric(exp_series, errors="coerce").dropna()
                avg_exp = float(exp_num.mean()) if len(exp_num) else 1.0
                avg_exp = max(0.10, min(1.0, avg_exp))
                if math.isfinite(cap_alpha):
                    exp_adj_alpha = cap_alpha / avg_exp
    except Exception:
        pass

    classification = _alpha_classification(cap_alpha, beta_score, bm_drift, rr_score)
    primary_source = _primary_alpha_source(mode, cap_alpha, exp_adj_alpha, beta_score)

    assumption_status = "Satisfied"
    if str(bm_drift).lower().strip() != "stable" or (math.isfinite(safe_float(beta_score)) and safe_float(beta_score) < 80):
        assumption_status = "Violated (Expected)"

    st.markdown("### Vector™ — Truth Referee")
    st.caption("Independent attribution + assumption-validation layer (read-only, deterministic system-feedback for governance).")

    t1, t2, t3 = st.columns(3, gap="medium")
    with t1:
        tile("Alpha Classification", classification, "Structural = regime/exposure-driven; Incidental = selection/tilt w/ stable linkage (governance-native context)")
    with t2:
        tile("Benchmark Assumption", assumption_status, f"BM drift: {bm_drift} · Beta Reliability: {beta_grade}")
    with t3:
        tile("Primary Alpha Source", "Referee inference", primary_source)

    st.markdown("#### Vector Verdict")
    verdict_lines = [
        f"**Primary Alpha Source:** {primary_source}",
        f"**Benchmark Assumption Status:** {assumption_status}",
        f"**Beta Reliability:** {beta_grade} · β {fmt_num(beta_val,2)} vs tgt {fmt_num(beta_target_for_mode(mode),2)} · R² {fmt_num(beta_r2,2)} · n {beta_n}",
        f"**Regime Dependence:** {'High' if classification == 'Structural' else 'Moderate' if classification == 'Incidental' else 'Low'}",
        f"**Alpha Classification:** {classification}",
    ]
    for line in verdict_lines:
        st.write("• " + line)

    st.markdown("#### Assumptions Tested by Vector")
    checks = _assumption_checklist(bm_drift=bm_drift, beta_score=beta_score)
    for label, passes, note in checks:
        icon = "✓" if passes else "◆"
        st.write(f"{icon} **{label}** — {note}")

    st.markdown("#### What Vector Would Flag if This Were Failing")
    ff = _vector_failure_flags(metrics, cov, bm_drift, beta_score)
    if not ff:
        st.success("No failure flags triggered by referee rules.")
    else:
        for f in ff[:6]:
            st.write("• " + f)

    st.caption("Vector provides governance feedback (no-predict architectural validation). It validates causality and flags when benchmark assumptions fail (stable system-feedback).")
    st.caption("**Role of Waves:** Reference implementations exposing platform behavior under live market regimes (context vs control via architectural constraint).")


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
# VECTOR GOVERNANCE LAYER (your 5 items)
# ============================================================
# 5️⃣ Wave Purpose Statements (edit anytime; safe + non-math)
WAVE_PURPOSE: Dict[str, str] = {
    # Put your top “hero” waves here first; anything missing falls back safely.
    "S&P 500 Wave": "Core U.S. large-cap baseline. Provides broad market participation with governance-native benchmarking. Stable architectural reference (no-predict, context vs control).",
    "US Growth Wave": "Growth-tilted U.S. equity exposure. Measures disciplined upside capture relative to a custom growth benchmark. System-feedback via predictable analytics.",
    "Small Cap Growth Wave": "Small-cap growth exposure. Targets flexible upside with governed risk controls and transparent benchmark-relative attribution (architectural stability).",
    "Small to Mid Cap Growth Wave": "SMID growth exposure. Balances upside capture and drawdown governance, with benchmark fidelity as a stability constraint.",
    "AI & Automation Wave": "Innovation exposure focused on AI/automation leadership. Governance prevents benchmark illusion; provides architectural context vs control.",
    "Quantum Computing Wave": "Frontier innovation exposure for asymmetric upside potential. Governance emphasizes benchmark fit + risk regime clarity (no-predict, stable feedback).",
    "Future Power & Energy Wave": "Energy-transition exposure capturing multi-year structural themes. Governed drawdown and tracking error controls provide predictable system-feedback.",
    "Clean Transit-Infrastructure Wave": "Infrastructure/transition exposure. Durable positioning with risk-aware regime behavior and benchmark verification (architectural governance).",
    "Crypto Income Wave": "Crypto yield/income exposure. Maximizes risk-adjusted income signals with explicit regime-aware transparency (flexible, predictable governance).",
    "SmartSafe Money Market Wave": "Capital-preservation anchor. Designed for stability and liquidity; provides governance reference during risk-off regimes (architectural stability).",
}

def wave_purpose_statement(wave_name: str) -> str:
    if not ENABLE_WAVE_PURPOSE_STATEMENTS:
        return ""
    if not wave_name or wave_name == "(none)":
        return ""
    return WAVE_PURPOSE.get(wave_name, "Purpose statement not set yet for this Wave. Add it to WAVE_PURPOSE in app.py.")


# 4️⃣ Gating Warnings (read-only; non-blocking)
def build_gating_warnings(
    cov: Dict[str, Any],
    bm_drift: str,
    beta_score: float,
    beta_n: int,
    selected_wave: str,
) -> Dict[str, List[str]]:
    warns: List[str] = []
    crits: List[str] = []

    rows = int(cov.get("rows") or 0)
    age = safe_float(cov.get("age_days"))
    cov_score = safe_float(cov.get("completeness_score"))
    miss_pct = safe_float(cov.get("missing_pct"))
    bs = safe_float(beta_score)

    if selected_wave == "(none)" or not selected_wave:
        crits.append("No wave selected — governance evaluation requires context.")
        return {"warn": warns, "crit": crits}

    if str(bm_drift).lower().strip() != "stable":
        crits.append("Benchmark drift detected — stabilize benchmark mix for governance (architectural stability requirement).")

    if math.isfinite(cov_score) and cov_score < 75:
        crits.append(f"Coverage score is low ({fmt_num(cov_score,1)}) — analytics reliability requires attention (governance signal).")
    elif math.isfinite(cov_score) and cov_score < 85:
        warns.append(f"Coverage score < 85 ({fmt_num(cov_score,1)}) — treat risk/alpha outputs with caution (system-feedback).")

    if math.isfinite(age) and age >= 7:
        crits.append(f"Data is stale ({fmt_int(age)} days old) — refresh pipeline for stable governance (predictable system-feedback).")
    elif math.isfinite(age) and age >= 5:
        warns.append(f"Data is aging ({fmt_int(age)} days old) — freshness check recommended (governance hygiene).")

    if rows < 60:
        crits.append("History window is too short (<60 points) — stability of risk metrics is weak (architectural constraint).")
    elif rows < 90:
        warns.append("Limited history (<90 points) — risk metrics may be noisy (context limitation).")

    if math.isfinite(miss_pct) and miss_pct >= 0.08:
        crits.append(f"Missing business days is high ({fmt_num(miss_pct*100,1)}%) — verify data integrity (governance requirement).")
    elif math.isfinite(miss_pct) and miss_pct >= 0.05:
        warns.append(f"Missing business days ≥5% ({fmt_num(miss_pct*100,1)}%) — verify history continuity (stable system-feedback).")

    if math.isfinite(bs) and bs < 65:
        crits.append(f"Very low beta reliability ({fmt_num(bs,1)}/100) — benchmark may not explain systematic exposure (architectural fit issue).")
    elif math.isfinite(bs) and bs < 75:
        warns.append(f"Beta reliability <75 ({fmt_num(bs,1)}/100) — benchmark fit likely weak (expected or needs review; context vs control).")

    if beta_n < 30:
        warns.append("Insufficient sample for beta reliability (<30 points; predictable statistical limitation).")

    return {"warn": warns, "crit": crits}


# 1️⃣ Vector Status Bar (always visible under title)
def render_vector_status_bar(
    conf_level: str,
    bm_drift: str,
    beta_grade: str,
    beta_score: float,
    rr_score: float,
    sel_score: Dict[str, Any],
    metrics: Dict[str, Any],
):
    if not (VECTOR_GOVERNANCE_ENABLED and ENABLE_VECTOR_STATUS_BAR):
        return

    pills = [
        f"Vector Status: {conf_level}",
        f"BM: {bm_drift.upper()}",
        f"Beta Reliability: {beta_grade}",
        f"Risk Reaction: {fmt_num(rr_score,1)}/100",
        f"30D α {fmt_pct(metrics.get('a30'))} · 60D α {fmt_pct(metrics.get('a60'))}",
    ]
    if ENABLE_SCORECARD:
        pills.insert(1, f"Analytics WaveScore: {fmt_num(sel_score.get('AnalyticsScore'),1)}/100")

    st.markdown('<div class="vector-status">', unsafe_allow_html=True)
    st.markdown('<div class="title">Vector™ Status Bar</div>', unsafe_allow_html=True)
    st.markdown('<div class="row">', unsafe_allow_html=True)
    for p in pills:
        st.markdown(f'<span class="vector-pill">{p}</span>', unsafe_allow_html=True)
    st.markdown("</div></div>", unsafe_allow_html=True)


# 2️⃣ Final Verdict Box (compact IC-grade decision box)
def build_final_verdict(
    selected_wave: str,
    mode: str,
    metrics: Dict[str, Any],
    cov: Dict[str, Any],
    bm_drift: str,
    beta_score: float,
    rr_score: float,
) -> Dict[str, Any]:
    """
    Deterministic verdict:
      - Green/Amber/Red based on gating + confidence heuristics
      - Includes classification + primary source (from Referee logic)
    """
    cap_alpha = safe_float(metrics.get("a365"))
    if not math.isfinite(cap_alpha):
        cap_alpha = safe_float(metrics.get("a60"))
    if not math.isfinite(cap_alpha):
        cap_alpha = safe_float(metrics.get("a30"))

    # exposure-adjusted alpha (best-effort)
    exp_adj_alpha = cap_alpha
    try:
        exp = _try_get_exposure_series(selected_wave, mode, pd.DatetimeIndex(hist_sel.index)) if 'hist_sel' in globals() else None
        if exp is not None:
            exp_num = pd.to_numeric(exp, errors="coerce").dropna()
            avg_exp = float(exp_num.mean()) if len(exp_num) else 1.0
            avg_exp = max(0.10, min(1.0, avg_exp))
            if math.isfinite(cap_alpha):
                exp_adj_alpha = cap_alpha / avg_exp
    except Exception:
        pass

    classification = _alpha_classification(cap_alpha, beta_score, bm_drift, rr_score)
    primary = _primary_alpha_source(mode, cap_alpha, exp_adj_alpha, beta_score)

    # verdict color logic (non-blocking)
    cov_score = safe_float(cov.get("completeness_score"))
    age = safe_float(cov.get("age_days"))
    rows = int(cov.get("rows") or 0)
    bs = safe_float(beta_score)
    drift = (str(bm_drift).lower().strip() != "stable")

    red_triggers = 0
    amber_triggers = 0

    if drift:
        red_triggers += 1
    if math.isfinite(cov_score) and cov_score < 75:
        red_triggers += 1
    if math.isfinite(age) and age >= 7:
        red_triggers += 1
    if rows < 60:
        red_triggers += 1
    if math.isfinite(bs) and bs < 65:
        red_triggers += 1

    if math.isfinite(cov_score) and 75 <= cov_score < 85:
        amber_triggers += 1
    if math.isfinite(age) and 5 <= age < 7:
        amber_triggers += 1
    if 60 <= rows < 90:
        amber_triggers += 1
    if math.isfinite(bs) and 65 <= bs < 75:
        amber_triggers += 1

    if red_triggers >= 2:
        verdict = "REVIEW REQUIRED — Stabilization recommended"
        action = "Stabilize benchmark + resolve data integrity issues; architectural refinement required before presenting as governance-ready."
    elif red_triggers == 1 or amber_triggers >= 2:
        verdict = "ADVISORY — Proceed with disclosure"
        action = "Proceed with disclosure of advisories; use Referee + Truth Layer for context vs control transparency."
    else:
        verdict = "STABLE — Governance-ready"
        action = "Proceed; use comparator + snapshot for positioning (predictable governance framework)."

    return {
        "verdict": verdict,
        "classification": classification,
        "primary_source": primary,
        "cap_alpha": cap_alpha,
        "exp_adj_alpha": exp_adj_alpha,
        "action": action,
        "red_triggers": red_triggers,
        "amber_triggers": amber_triggers,
    }


def extract_verdict_drivers(v: Dict[str, Any], metrics: Dict[str, Any], cov: Dict[str, Any], 
                           bm_drift: str, beta_score: float) -> List[str]:
    """
    Deterministically extract top 3 drivers for the verdict.
    Always returns the same drivers in the same order for the same inputs.
    """
    drivers = []
    
    # Check review-required triggers (most important)
    if v.get("red_triggers", 0) >= 2:
        if str(bm_drift).lower().strip() != "stable":
            drivers.append("Benchmark drift detected — requires freeze for governance stability")
        
        cov_score = safe_float(cov.get("completeness_score"))
        if math.isfinite(cov_score) and cov_score < 75:
            drivers.append(f"Low data coverage ({fmt_num(cov_score,1)}/100) impacts analytics reliability")
        
        age = safe_float(cov.get("age_days"))
        if math.isfinite(age) and age >= 7:
            drivers.append(f"Stale data ({fmt_int(age)} days old) requires pipeline refresh")
        
        rows = int(cov.get("rows") or 0)
        if rows < 60:
            drivers.append(f"Insufficient history ({rows} days) for stable risk metrics")
        
        if math.isfinite(beta_score) and beta_score < 65:
            drivers.append(f"Low beta reliability ({fmt_num(beta_score,1)}/100) questions benchmark fit")
    
    # Check advisory triggers
    elif v.get("amber_triggers", 0) >= 2 or v.get("red_triggers", 0) == 1:
        cov_score = safe_float(cov.get("completeness_score"))
        if math.isfinite(cov_score) and 75 <= cov_score < 85:
            drivers.append(f"Coverage score borderline ({fmt_num(cov_score,1)}/100) — proceed with caution")
        
        age = safe_float(cov.get("age_days"))
        if math.isfinite(age) and 5 <= age < 7:
            drivers.append(f"Data freshness ({fmt_int(age)} days) nearing stale threshold")
        
        rows = int(cov.get("rows") or 0)
        if 60 <= rows < 90:
            drivers.append(f"Limited history ({rows} days) may produce noisy risk metrics")
        
        if math.isfinite(beta_score) and 65 <= beta_score < 75:
            drivers.append(f"Moderate beta reliability ({fmt_num(beta_score,1)}/100) — verify benchmark fit")
    
    # STABLE - highlight strengths
    else:
        cov_score = safe_float(cov.get("completeness_score"))
        if math.isfinite(cov_score) and cov_score >= 85:
            drivers.append(f"Strong data coverage ({fmt_num(cov_score,1)}/100) supports reliable analytics")
        
        if math.isfinite(beta_score) and beta_score >= 75:
            drivers.append(f"Good beta reliability ({fmt_num(beta_score,1)}/100) confirms benchmark alignment")
        
        age = safe_float(cov.get("age_days"))
        if math.isfinite(age) and age < 5:
            drivers.append(f"Fresh data ({fmt_int(age)} days old) ensures current insights")
        
        if str(bm_drift).lower().strip() == "stable":
            drivers.append("Stable benchmark enables consistent governance comparisons")
    
    # Return top 3 (deterministic order)
    return drivers[:3] if drivers else ["No specific drivers identified"]


def extract_action_checklist(v: Dict[str, Any], metrics: Dict[str, Any], cov: Dict[str, Any],
                             bm_drift: str, beta_score: float) -> List[str]:
    """
    Extract 3-5 recommended actions as checkbox items.
    Deterministic order based on verdict severity.
    """
    actions = []
    
    verdict_str = str(v.get("verdict", "")).upper()
    
    if "REVIEW REQUIRED" in verdict_str:
        # Critical actions first
        if str(bm_drift).lower().strip() != "stable":
            actions.append("Freeze benchmark composition immediately")
        
        cov_score = safe_float(cov.get("completeness_score"))
        age = safe_float(cov.get("age_days"))
        if math.isfinite(age) and age >= 7:
            actions.append("Refresh data pipeline — do not use stale outputs")
        
        if math.isfinite(cov_score) and cov_score < 75:
            actions.append("Investigate data gaps and missing days")
        
        rows = int(cov.get("rows") or 0)
        if rows < 60:
            actions.append("Extend history window to at least 90 days")
        
        actions.append("Flag outputs as NOT governance-ready in any client reports")
        
    elif "ADVISORY" in verdict_str:
        # Caution actions
        actions.append("Disclose limitations and advisories in any client presentations")
        
        if str(bm_drift).lower().strip() != "stable":
            actions.append("Freeze benchmark for consistency checks")
        
        actions.append("Use Vector Referee and Truth Layer to validate analytics")
        actions.append("Monitor data freshness and coverage metrics")
        actions.append("Prepare to escalate to Review Required status if conditions worsen")
        
    else:  # STABLE
        # Positive governance actions
        actions.append("Proceed with confidence — governance checks passed")
        actions.append("Use Comparator to position wave relative to alternatives")
        actions.append("Review Alpha Snapshot for portfolio-level insights")
        actions.append("Monitor benchmark drift and coverage on next refresh")
        actions.append("Document governance posture for audit trail")
    
    # Return 3-5 actions
    return actions[:5] if len(actions) >= 3 else actions


def get_confidence_numeric(conf_level: str) -> int:
    """Convert confidence level to 0-100 numeric score"""
    conf_map = {
        "High": 95,
        "Medium": 75,
        "Low": 45,
        "Very Low": 20,
    }
    return conf_map.get(conf_level, 50)


def render_final_verdict_box(v: Dict[str, Any], bm_id: str, bm_name: str, bm_display: str, 
                             beta_grade: str, beta_score: float, conf_level: str, metrics: Dict[str, Any],
                             cov: Dict[str, Any], bm_drift: str):
    if not (VECTOR_GOVERNANCE_ENABLED and ENABLE_FINAL_VERDICT_BOX):
        return
    
    st.markdown("### Final Verdict (Vector™)")
    
    # === 1. Executive Verdict Badge ===
    verdict_str = str(v.get("verdict", "—"))
    verdict_color = "green"
    verdict_short = "STABLE"
    verdict_subtitle = "Governance-ready"
    
    if "REVIEW REQUIRED" in verdict_str.upper():
        verdict_color = "red"
        verdict_short = "REVIEW REQUIRED"
        verdict_subtitle = "Stabilization recommended"
    elif "ADVISORY" in verdict_str.upper():
        verdict_color = "amber"
        verdict_short = "ADVISORY"
        verdict_subtitle = "Proceed with disclosure"
    else:
        verdict_color = "green"
        verdict_short = "STABLE"
        verdict_subtitle = "Governance-ready"
    
    st.markdown(
        f"""
<div class="verdict-badge verdict-badge-{verdict_color}">
  <div class="verdict-badge-title">{verdict_short}</div>
  <div class="verdict-badge-subtitle">{verdict_subtitle}</div>
</div>
""",
        unsafe_allow_html=True,
    )
    
    # === 2. Confidence + Benchmark Display ===
    st.markdown('<div class="waves-card">', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        conf_numeric = get_confidence_numeric(conf_level)
        st.markdown(f"**Confidence:** {conf_level} ({conf_numeric}/100)")
        st.markdown(f"**Alpha Classification:** {v.get('classification', '—')}")
    with col2:
        st.markdown(f"**Benchmark:** {bm_display}")
        with st.expander("🔍 Benchmark ID (Technical)", expanded=False):
            st.caption(f"Internal ID: `{bm_id}`")
            st.caption("Used for governance drift detection and audit tracking")
    st.markdown("</div>", unsafe_allow_html=True)
    
    # === 3. Why this verdict? ===
    st.markdown("#### Why this verdict?")
    drivers = extract_verdict_drivers(v, metrics, cov, bm_drift, beta_score)
    for driver in drivers:
        st.markdown(f"• {driver}")
    
    # === 4. Action Checklist ===
    st.markdown("#### Action Checklist")
    st.markdown('<div class="action-checklist">', unsafe_allow_html=True)
    actions = extract_action_checklist(v, metrics, cov, bm_drift, beta_score)
    for action in actions:
        st.markdown(
            f'<div class="action-item"><span class="action-checkbox">☐</span> {action}</div>',
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)
    
    # === 5. Signal Tiles (Compact) ===
    st.markdown("#### Governance Signals")
    cols = st.columns(4)
    with cols[0]:
        st.markdown(
            f"""
<div class="signal-tile-compact">
  <div class="signal-tile-label">Alpha Classification</div>
  <div class="signal-tile-value">{v.get('classification', '—')}</div>
</div>
""",
            unsafe_allow_html=True,
        )
    with cols[1]:
        st.markdown(
            f"""
<div class="signal-tile-compact">
  <div class="signal-tile-label">Benchmark Fit</div>
  <div class="signal-tile-value">{beta_grade}</div>
</div>
""",
            unsafe_allow_html=True,
        )
    with cols[2]:
        st.markdown(
            f"""
<div class="signal-tile-compact">
  <div class="signal-tile-label">Beta Reliability</div>
  <div class="signal-tile-value">{fmt_num(beta_score,1)}/100</div>
</div>
""",
            unsafe_allow_html=True,
        )
    with cols[3]:
        cap_alpha = v.get("cap_alpha", 0)
        st.markdown(
            f"""
<div class="signal-tile-compact">
  <div class="signal-tile-label">Capture Alpha</div>
  <div class="signal-tile-value">{fmt_pct(cap_alpha)}</div>
</div>
""",
            unsafe_allow_html=True,
        )
    
    # === 6. Data Integrity / Assumptions (Collapsible) ===
    with st.expander("📊 Data Integrity & Assumptions", expanded=False):
        st.caption("Governance assumptions tested on every refresh. These are rule-based architectural validations, not opinions.")
        
        # Get current timestamp and environment
        now = datetime.now()
        timestamp_str = now.strftime("%Y-%m-%d %H:%M:%S UTC")
        
        # Determine environment status (LIVE/SANDBOX/HYBRID)
        # For this app, we'll check if we have real-time data or demo data
        age_days = safe_float(cov.get("age_days"))
        if math.isfinite(age_days):
            if age_days < 2:
                env_status = "LIVE"
            elif age_days < 7:
                env_status = "HYBRID"
            else:
                env_status = "SANDBOX"
        else:
            env_status = "UNKNOWN"
        
        st.markdown(f"**Timestamp:** {timestamp_str}")
        st.markdown(f"**Environment:** {env_status}")
        st.markdown("---")
        
        # Assumptions checklist
        checks = _assumption_checklist(bm_drift=bm_drift, beta_score=beta_score)
        for label, passes, note in checks:
            st.write(f"{'✓' if passes else '◆'} **{label}** — {note}")
        
        # Coverage details
        st.markdown("---")
        st.markdown("**Coverage Details:**")
        st.write(f"• Rows: {int(cov.get('rows') or 0)}")
        st.write(f"• Completeness Score: {fmt_num(cov.get('completeness_score'), 1)}/100")
        st.write(f"• Missing Days: {fmt_num(safe_float(cov.get('missing_pct')) * 100, 1)}%")
        st.write(f"• Data Age: {fmt_int(cov.get('age_days'))} days")
        st.write(f"• Benchmark Status: {bm_drift.capitalize()}")
        
        render_definitions(["Assumptions Tested", "Coverage Score", "Benchmark Snapshot / Drift"], 
                          title="Definitions")


# 3️⃣ Assumptions Panel (clean UI wrapper around checklist)
def render_assumptions_panel(bm_drift: str, beta_score: float):
    if not (VECTOR_GOVERNANCE_ENABLED and ENABLE_ASSUMPTIONS_PANEL):
        return
    with st.expander("Assumptions Panel (Vector™) — tested every refresh", expanded=False):
        st.caption("Deterministic governance checks. These are *not* opinions; they’re rule-based assumptions validation.")
        checks = _assumption_checklist(bm_drift=bm_drift, beta_score=beta_score)
        for label, passes, note in checks:
            st.write(f"{'✓' if passes else '◆'} **{label}** — {note}")
        render_definitions(["Assumptions Tested"], title="Definitions (Assumptions)")


# 4️⃣ Gating Warnings (render)
def render_gating_warnings(g: Dict[str, List[str]]):
    if not (VECTOR_GOVERNANCE_ENABLED and ENABLE_GATING_WARNINGS):
        return
    crits = g.get("crit", [])
    warns = g.get("warn", [])

    if not crits and not warns:
        return

    if crits:
        st.markdown('<div class="vector-crit">', unsafe_allow_html=True)
        st.markdown("**Architectural Flags — Governance Oversight (Vector™):**")
        for w in crits[:6]:
            st.write("• " + w)
        st.markdown("</div>", unsafe_allow_html=True)

    if warns:
        st.markdown('<div class="vector-warn">', unsafe_allow_html=True)
        st.markdown("**Advisory Signals — Context Awareness (Vector™):**")
        for w in warns[:6]:
            st.write("• " + w)
        st.markdown("</div>", unsafe_allow_html=True)

    with st.expander("What are architectural flags and advisory signals?"):
        st.write(
            "These are governance-native oversight checks providing stable system-feedback (flexible onboarding). They do not block the app. "
            "They indicate whether analytics are governance-ready, and what architectural elements merit attention "
            "(benchmark stability, data freshness, coverage completeness, history depth, benchmark alignment). Context-aware guidance via predictable signals."
        )
        render_definitions(["Gating Warnings", "Benchmark Snapshot / Drift", "Coverage Score", "Beta Reliability Score"], title="Definitions (Governance Signals)")


# ============================================================
# Sidebar controls
# ============================================================
all_waves = get_all_waves_safe()
if not all_waves:
    st.warning("No waves found. Ensure engine is available or CSVs are present (flexible onboarding: wave_config.csv / wave_weights.csv / list.csv).")

st.sidebar.markdown("## Controls")
mode = st.sidebar.selectbox("Mode", ["Standard", "Alpha-Minus-Beta", "Private Logic"], index=0)
selected_wave = st.sidebar.selectbox("Selected Wave", options=all_waves if all_waves else ["(none)"], index=0)
scan_mode = st.sidebar.toggle("Scan Mode (fast, fewer visuals)", value=True)
days = st.sidebar.selectbox("History Window", [365, 730, 1095, 2520], index=0)
st.sidebar.markdown("---")
st.sidebar.caption("Governance: every metric shown is computed from the canonical hist_sel (selected Wave + Mode).")


# ============================================================
# Canonical source-of-truth (selected wave)
# ============================================================
hist_sel = _standardize_history(compute_wave_history(selected_wave, mode, days=days)) if selected_wave and selected_wave != "(none)" else pd.DataFrame()
cov = coverage_report(hist_sel)

bm_mix = get_benchmark_mix()
bm_rows_now = _bm_rows_for_wave(bm_mix, selected_wave) if selected_wave and selected_wave != "(none)" else pd.DataFrame(columns=["Ticker", "Weight"])
bm_id = benchmark_snapshot_id(selected_wave, mode, bm_mix) if selected_wave and selected_wave != "(none)" else "BM-NA"
bm_name = benchmark_name(selected_wave, bm_mix) if selected_wave and selected_wave != "(none)" else "N/A"
bm_drift = benchmark_drift_status(selected_wave, mode, bm_id) if selected_wave and selected_wave != "(none)" else "stable"
bm_display = benchmark_display_description(bm_name, bm_drift, bm_rows_now) if selected_wave and selected_wave != "(none)" else "Benchmark Snapshot: N/A"
bm_diff = benchmark_diff_table(selected_wave, mode, bm_rows_now) if ENABLE_FIDELITY_INSPECTOR else pd.DataFrame()

metrics = compute_metrics_from_hist(hist_sel)
conf_level, conf_reason = confidence_from_integrity(cov, bm_drift)
sel_score = compute_analytics_score_for_selected(hist_sel, cov, bm_drift) if ENABLE_SCORECARD else {"AnalyticsScore": np.nan, "Grade": "N/A", "Flags": ""}
rr_score = risk_reaction_score(metrics["te"], metrics["mdd"], metrics["cvar95"])

difficulty = benchmark_difficulty_proxy(bm_rows_now)
te_band = te_risk_band(metrics["te"])

beta_target = beta_target_for_mode(mode)
beta_val, beta_r2, beta_n = beta_and_r2(
    hist_sel["wave_ret"] if not hist_sel.empty else pd.Series(dtype=float),
    hist_sel["bm_ret"] if not hist_sel.empty else pd.Series(dtype=float),
)
beta_score = beta_reliability_score(beta_val, beta_r2, beta_n, beta_target)
beta_grade = beta_band(beta_score)

# Vector governance computations (non-blocking)
gating = build_gating_warnings(cov, bm_drift, beta_score, beta_n, selected_wave) if VECTOR_GOVERNANCE_ENABLED else {"warn": [], "crit": []}
final_verdict = build_final_verdict(selected_wave, mode, metrics, cov, bm_drift, beta_score, rr_score) if VECTOR_GOVERNANCE_ENABLED else {}


# ============================================================
# BIG HEADER
# ============================================================
st.markdown(f'<div class="waves-big-wave">{selected_wave}</div>', unsafe_allow_html=True)
st.markdown(
    f'<div class="waves-subhead">Mode: <b>{mode}</b> &nbsp; • &nbsp; Canonical window: last {days} trading days (max)</div>',
    unsafe_allow_html=True,
)

# 1️⃣ Vector Status Bar (new)
render_vector_status_bar(
    conf_level=conf_level,
    bm_drift=bm_drift,
    beta_grade=beta_grade,
    beta_score=beta_score,
    rr_score=rr_score,
    sel_score=sel_score,
    metrics=metrics,
)

# 4️⃣ Gating Warnings (new)
render_gating_warnings(gating)

# ============================================================
# Sticky chip bar (existing)
# ============================================================
st.markdown('<div class="waves-sticky">', unsafe_allow_html=True)
chip(f"Confidence: {conf_level}")
if ENABLE_SCORECARD:
    chip(f"Analytics WaveScore: {fmt_num(sel_score.get('AnalyticsScore'), 1)}/100 {sel_score.get('Flags','')}")
chip(f"{bm_display}")
chip(f"Coverage: {fmt_num(cov.get('completeness_score'),1)} · AgeDays: {fmt_int(cov.get('age_days'))}")
chip(f"30D α {fmt_pct(metrics['a30'])} · r {fmt_pct(metrics['r30'])}")
chip(f"60D α {fmt_pct(metrics['a60'])} · r {fmt_pct(metrics['r60'])}")
chip(f"Risk: TE {fmt_pct(metrics['te'])} ({te_band}) · MaxDD {fmt_pct(metrics['mdd'])}")
chip(f"Beta Reliability: {beta_grade}")
st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# Tabs
# ============================================================
tab_names = [
    "IC Summary",
    "Decision Center",
    "Overview",
    "Risk + Advanced",
    "Benchmark Governance",
    "Comparator",
    "Alpha Snapshot",
    "Diagnostics",
]
tabs = st.tabs(tab_names)


# ============================================================
# IC SUMMARY
# ============================================================
with tabs[0]:
    st.markdown("### Executive IC One-Pager")
    st.caption("Executive summary with Vector™ Final Verdict. For detailed analytics, assumptions, and derivations, see Decision Center (stable architectural separation).")

    # 2️⃣ Final Verdict Box (DOMINANT ELEMENT - placed FIRST in IC)
    if VECTOR_GOVERNANCE_ENABLED and ENABLE_FINAL_VERDICT_BOX:
        render_final_verdict_box(
            final_verdict, 
            bm_id=bm_id, 
            bm_name=bm_name,
            bm_display=bm_display,
            beta_grade=beta_grade, 
            beta_score=beta_score, 
            conf_level=conf_level,
            metrics=metrics,
            cov=cov,
            bm_drift=bm_drift
        )

    colA, colB = st.columns([1.2, 1.0], gap="large")

    with colA:
        st.markdown('<div class="waves-card">', unsafe_allow_html=True)
        st.markdown("#### What is this wave?")

        # 5️⃣ Wave Purpose Statement (executive positioning)
        purpose = wave_purpose_statement(selected_wave)
        if purpose:
            st.markdown("**Wave Purpose Statement (Vector™):**")
            st.write(purpose)
            st.caption("Purpose is positioning + governance context; it does not affect analytics calculations (no-predict architectural constraint).")
        else:
            st.caption("Purpose statement disabled or not set.")

        st.write(
            "Governance-native portfolio exposure with benchmark-anchored analytics. "
            "Designed to eliminate crisscross metrics and provide decision-ready outputs (context vs control via architectural constraint)."
        )
        st.markdown("**Governance Status (Conclusions)**")
        st.write(f"**Confidence:** {conf_level}")
        st.write(f"**Benchmark Status:** {bm_drift.capitalize()}")
        st.write(f"**Beta Reliability:** {beta_grade}")
        st.markdown("**Performance vs Benchmark (Conclusions)**")
        st.write(f"30D Return {fmt_pct(metrics['r30'])} | 30D Alpha {fmt_pct(metrics['a30'])}")
        st.write(f"60D Return {fmt_pct(metrics['r60'])} | 60D Alpha {fmt_pct(metrics['a60'])}")
        st.write(f"365D Return {fmt_pct(metrics['r365'])} | 365D Alpha {fmt_pct(metrics['a365'])}")
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="waves-card">', unsafe_allow_html=True)
        st.markdown("#### Key Wins / Key Risks / Next Actions")
        wins, risks, actions = [], [], []

        if conf_level == "High":
            wins.append("Fresh + complete coverage supports governance confidence (stable system-feedback).")
        if bm_drift == "stable":
            wins.append("Benchmark snapshot is stable (governance-ready architectural context).")
        if math.isfinite(beta_score) and beta_score >= 80:
            wins.append("Benchmark systematic exposure match is strong (predictable beta linkage).")
        if math.isfinite(metrics["a30"]) and metrics["a30"] > 0:
            wins.append("Positive 30D alpha versus benchmark (favorable context within governance framework).")

        if conf_level != "High":
            risks.append("Data governance flags present (coverage/age/rows require attention).")
        if bm_drift != "stable":
            risks.append("Benchmark drift detected (composition changed in-session; affects context stability).")
        if math.isfinite(beta_score) and beta_score < 75:
            risks.append("Beta reliability requires review (benchmark may not match systematic exposure; verify architectural fit).")
        if math.isfinite(metrics["mdd"]) and metrics["mdd"] <= -0.25:
            risks.append("Deep drawdown regime observed (elevated risk context).")

        if bm_drift != "stable":
            actions.append("Stabilize benchmark mix for governance (freeze composition, then re-run for consistent context).")
        if math.isfinite(beta_score) and beta_score < 75:
            actions.append("Review benchmark mix: verify exposures match wave beta target or document intentional architectural mismatch.")
        if math.isfinite(metrics["te"]) and metrics["te"] >= 0.20:
            actions.append("Review exposure caps and SmartSafe governance posture for elevated active risk context.")
        if conf_level != "High":
            actions.append("Inspect history pipeline for missing days or stale writes (ensure stable system-feedback).")
        if not actions:
            actions.append("Proceed: governance is stable; use comparator and alpha snapshot for positioning (predictable context).")

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
        st.markdown("#### Summary Metrics")
        st.write(f"**Confidence:** {conf_level}")
        st.write(f"**Benchmark:** {bm_display}")
        st.write(f"**Coverage Score:** {fmt_num(cov.get('completeness_score'),1)}/100")
        st.write(f"**Data Age:** {fmt_int(cov.get('age_days'))} days")
        st.write(f"**Analytics WaveScore:** {fmt_num(sel_score.get('AnalyticsScore'),1)}/100")
        st.write(f"**Beta Reliability:** {beta_grade}")
        st.write(f"**30D Alpha:** {fmt_pct(metrics['a30'])}")
        st.write(f"**60D Alpha:** {fmt_pct(metrics['a60'])}")
        st.write(f"**Tracking Error:** {fmt_pct(metrics['te'])} ({te_band})")
        st.write(f"**Max Drawdown:** {fmt_pct(metrics['mdd'])}")
        
        with st.expander("🔍 Technical IDs", expanded=False):
            st.caption(f"Benchmark ID: `{bm_id}`")


# ============================================================
# DECISION CENTER
# ============================================================
with tabs[1]:
    st.markdown("### Decision Center")
    st.caption("Detailed analytics, assumptions, derivations, and decision metrics. Architectural depth with stable system-feedback (context vs control).")

    st.markdown("#### Decision Tiles")
    c1, c2 = st.columns(2, gap="medium")
    with c1:
        tile("Confidence", conf_level, conf_reason)
        tile("Benchmark", bm_display, f"Status: {bm_drift.capitalize()}")
        tile("30D Alpha", fmt_pct(metrics["a30"]), f"30D Return {fmt_pct(metrics['r30'])}")
    with c2:
        tile("Analytics WaveScore", f"{fmt_num(sel_score.get('AnalyticsScore'),1)}/100", f"{sel_score.get('Grade', 'N/A')} {sel_score.get('Flags','')}")
        tile("Beta Reliability", beta_grade, f"β {fmt_num(beta_val,2)} vs target {fmt_num(beta_target,2)}")
        tile("Active Risk (TE)", fmt_pct(metrics["te"]), f"Band: {te_band}")

    st.markdown("---")
    
    # 3️⃣ Assumptions Panel (detailed analytics - moved from IC Summary)
    if VECTOR_GOVERNANCE_ENABLED and ENABLE_ASSUMPTIONS_PANEL:
        render_assumptions_panel(bm_drift=bm_drift, beta_score=beta_score)
    
    st.markdown("---")
    st.markdown("### Detailed Analytics & Derivations")
    st.caption("All analytical assumptions, formulas, and detailed explanations (predictable governance framework, no-predict architectural refinement).")
    
    # Detailed Trust + Governance (moved from IC Summary)
    st.markdown('<div class="waves-card">', unsafe_allow_html=True)
    st.markdown("#### Trust + Governance (Detailed)")
    st.write(f"**Confidence:** {conf_level} — {conf_reason}")
    st.write(f"**Benchmark Snapshot:** {bm_display}")
    st.write(f"**Drift Status:** {bm_drift.capitalize()}")
    st.write(f"**Beta Reliability:** {beta_grade} · β {fmt_num(beta_val,2)} vs target {fmt_num(beta_target,2)} · R² {fmt_num(beta_r2,2)} · n {beta_n}")
    st.caption("Beta derivation: regression slope of Wave daily returns vs Benchmark daily returns, with R² measuring fit quality (architectural governance metric).")
    
    with st.expander("🔍 Technical IDs", expanded=False):
        st.caption(f"Benchmark ID: `{bm_id}` (used for governance drift detection)")
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Vector Referee (moved from IC Summary)
    if ENABLE_VECTOR_REFEREE:
        st.markdown('<div class="waves-card">', unsafe_allow_html=True)
        st.markdown("#### Vector™ Referee (Detailed Verdict)")
        safe_panel(
            "Vector Referee",
            lambda: _vector_referee_verdict_block(
                selected_wave=selected_wave,
                mode=mode,
                hist_sel=hist_sel,
                metrics=metrics,
                cov=cov,
                bm_drift=bm_drift,
                beta_val=beta_val,
                beta_r2=beta_r2,
                beta_n=beta_n,
                beta_score=beta_score,
                beta_grade=beta_grade,
                rr_score=rr_score,
            ),
        )
        st.markdown("</div>", unsafe_allow_html=True)

    # Vector Truth Layer (moved from IC Summary)
    st.markdown('<div class="waves-card">', unsafe_allow_html=True)
    st.markdown("#### Vector™ Truth Layer (Read-Only, Detailed Analytics)")
    safe_panel("Vector Truth", lambda: _vector_truth_panel(selected_wave, mode, hist_sel, metrics, days))
    st.markdown("</div>", unsafe_allow_html=True)

    # Alpha Enhancements with detailed derivations (moved from IC Summary)
    st.markdown('<div class="waves-card">', unsafe_allow_html=True)
    st.markdown("#### Alpha Enhancements (Detailed Derivations)")
    attrib60 = _risk_on_off_attrib(hist_sel, selected_wave, mode, window=60)
    ac_sel = _alpha_capture_series(hist_sel, selected_wave, mode)
    ac60 = _compound_from_daily(ac_sel.tail(min(60, len(ac_sel)))) if len(ac_sel) >= 2 else float("nan")
    st.write(f"**Capital-Weighted Alpha (60D):** {fmt_pct(attrib60.get('cap_alpha'))}")
    st.caption("Capital-weighted alpha: investor-experience alpha (Wave return − Benchmark return) over the window (context vs control metric).")
    st.write(f"**Exposure-Adjusted Alpha (60D):** {fmt_pct(attrib60.get('exp_adj_alpha'))}")
    st.caption("Exposure-adjusted alpha: capital-weighted alpha divided by average exposure over the window if exposure known (architectural refinement).")
    st.write(f"**Alpha Capture (60D, exposure-normalized if available):** {fmt_pct(ac60)}")
    st.caption("Alpha capture: daily (Wave return − Benchmark return) optionally normalized by exposure, compounded over window (predictable system-feedback).")
    st.write(
        f"**Risk-On Alpha (60D):** {fmt_pct(attrib60.get('risk_on_alpha'))} "
        f"({fmt_pct(attrib60.get('risk_on_share'),2)} share)"
    )
    st.write(
        f"**Risk-Off Alpha (60D):** {fmt_pct(attrib60.get('risk_off_alpha'))} "
        f"({fmt_pct(attrib60.get('risk_off_share'),2)} share)"
    )
    st.caption("Risk-On vs Risk-Off: Alpha split by benchmark regime (Risk-Off when bm_ret < 0, else Risk-On). Governance-native context classification.")
    render_definitions(
        ["Capital-Weighted Alpha", "Exposure-Adjusted Alpha", "Risk-On vs Risk-Off Attribution", "Alpha Capture"],
        title="Definitions (Alpha Enhancements)",
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("---")
    render_definitions(
        [
            "Canonical (Source of Truth)",
            "Wave Purpose Statement",
            "Gating Warnings",
            "Return",
            "Alpha",
            "Alpha Capture",
            "Tracking Error (TE)",
            "Max Drawdown (MaxDD)",
            "CVaR 95% (daily)",
            "Analytics Scorecard",
            "Benchmark Snapshot / Drift",
            "Beta (vs Benchmark)",
            "Beta Reliability Score",
            "Vector™ — Truth Referee",
            "Alpha Classification",
            "Assumptions Tested",
        ],
        title="Definitions (Decision Center)",
    )


# ============================================================
# OVERVIEW
# ============================================================
with tabs[2]:
    st.markdown("### Overview (Canonical Metrics)")
    st.caption("Everything below is computed from the same canonical hist_sel object (no duplicate math).")

    truth = {
        "CanonicalEndDate": str(pd.to_datetime(hist_sel.index).max().date()) if not hist_sel.empty else "—",
        "Rows": int(cov.get("rows") or 0),
        "BMSnapshot": bm_display,
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
    
    with st.expander("🔍 Technical Details (Internal IDs)"):
        st.caption("Internal identifiers for audit and compliance tracking:")
        st.write(f"**Benchmark Internal ID:** `{bm_id}`")
        st.write(f"**Benchmark Name:** {bm_name}")
        st.caption("These internal IDs ensure benchmark composition governance and are used for drift detection.")

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
with tabs[3]:
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
with tabs[4]:
    st.markdown("### Benchmark Governance (Fidelity Inspector)")

    left, right = st.columns([1.0, 1.1], gap="large")

    with left:
        st.markdown('<div class="waves-card">', unsafe_allow_html=True)
        st.markdown("#### Inspector Summary")
        st.write(f"**Snapshot:** {bm_display}")
        st.write(f"**Drift Status:** {bm_drift.capitalize()}")
        st.write(f"**Active Risk Band (TE):** {te_band} (TE {fmt_pct(metrics['te'])})")
        st.write(f"**Beta Reliability:** {beta_grade} · β {fmt_num(beta_val,2)} tgt {fmt_num(beta_target,2)}")
        st.write(f"**Difficulty vs SPY (proxy):** {fmt_num(difficulty.get('difficulty_vs_spy'), 1)} (range ~ -25 to +25)")
        st.caption("Difficulty is a concentration/diversification heuristic (not a promise).")
        
        with st.expander("🔍 Technical IDs", expanded=False):
            st.caption(f"Benchmark ID: `{bm_id}` (governance tracking)")
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
# COMPARATOR
# ============================================================
with tabs[5]:
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
            bm_rows = _bm_rows_for_wave(bm_mix, w)
            bid = benchmark_snapshot_id(w, mode, bm_mix)
            bname = benchmark_name(w, bm_mix)
            d = benchmark_drift_status(w, mode, bid)
            bdisplay = benchmark_display_description(bname, d, bm_rows)
            m = compute_metrics_from_hist(h)
            bt = beta_target_for_mode(mode)
            bv, br2, bn = beta_and_r2(
                h["wave_ret"] if not h.empty else pd.Series(dtype=float),
                h["bm_ret"] if not h.empty else pd.Series(dtype=float),
            )
            bs = beta_reliability_score(bv, br2, bn, bt)
            return {"cov": c, "bm_id": bid, "bm_display": bdisplay, "bm_drift": d, "m": m, "beta": bv, "beta_score": bs}

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
                    {"Metric": "Benchmark", "Wave A": a["bm_display"], "Wave B": b["bm_display"]},
                    {"Metric": "BM Drift", "Wave A": a["bm_drift"], "Wave B": b["bm_drift"]},
                ]
            )
            st.dataframe(comp, use_container_width=True, hide_index=True)
            
            # Show internal IDs in technical section
            with st.expander("🔍 Technical Details (Internal IDs)"):
                st.caption("Internal benchmark identifiers for audit and compliance tracking:")
                st.write(f"**Wave A Benchmark ID:** `{a['bm_id']}`")
                st.write(f"**Wave B Benchmark ID:** `{b['bm_id']}`")
        else:
            st.info("Pick Wave B to compare.")


# ============================================================
# ALPHA SNAPSHOT (ALL WAVES)
# ============================================================
with tabs[6]:
    st.markdown("### Alpha Snapshot (All Waves)")
    st.caption("Intraday / 30D / 60D / 365D — Return + Alpha + Alpha Capture (exposure-normalized when available).")

    if not ENABLE_ALPHA_SNAPSHOT:
        st.info("Alpha Snapshot disabled.")
    else:
        snap_days = st.selectbox("Snapshot lookback used for loading histories (>=365 recommended)", [365, 730, 1095, 2520], index=0)
        show_ro_rf = st.toggle("Include Risk-On / Risk-Off Attribution (60D)", value=False)
        limit_n = st.slider(
            "Max waves to compute (speed control)",
            min_value=5,
            max_value=max(5, len(all_waves) if all_waves else 5),
            value=min(20, len(all_waves) if all_waves else 5),
        )

        @st.cache_data(show_spinner=False)
        def _compute_snapshot_row(wave_name: str, mode: str, days: int) -> Dict[str, Any]:
            h = _standardize_history(compute_wave_history(wave_name, mode, days=days))
            m = compute_metrics_from_hist(h)

            ac = _alpha_capture_series(h, wave_name, mode)
            ac30 = _compound_from_daily(ac.tail(min(30, len(ac)))) if len(ac) >= 2 else float("nan")
            ac60 = _compound_from_daily(ac.tail(min(60, len(ac)))) if len(ac) >= 2 else float("nan")
            ac365 = _compound_from_daily(ac.tail(min(365, len(ac)))) if len(ac) >= 2 else float("nan")

            attrib60 = _risk_on_off_attrib(h, wave_name, mode, window=60)

            return {
                "Wave": wave_name,
                "1D Ret": m["r1"],
                "1D Alpha": m["a1"],
                "30D Ret": m["r30"],
                "30D Alpha": m["a30"],
                "30D AlphaCapture": ac30,
                "60D Ret": m["r60"],
                "60D Alpha": m["a60"],
                "60D AlphaCapture": ac60,
                "365D Ret": m["r365"],
                "365D Alpha": m["a365"],
                "365D AlphaCapture": ac365,
                "60D CapAlpha": attrib60.get("cap_alpha"),
                "60D ExpAdjAlpha": attrib60.get("exp_adj_alpha"),
                "60D RiskOnAlpha": attrib60.get("risk_on_alpha"),
                "60D RiskOffAlpha": attrib60.get("risk_off_alpha"),
                "CoverageRows": int(len(h)) if h is not None else 0,
            }

        if not all_waves:
            st.info("No waves available to snapshot.")
        else:
            waves_to_run = all_waves[: int(limit_n)]
            rows_out: List[Dict[str, Any]] = []
            prog = st.progress(0)

            for i, w in enumerate(waves_to_run):
                try:
                    rows_out.append(_compute_snapshot_row(w, mode, snap_days))
                except Exception:
                    rows_out.append({"Wave": w})
                prog.progress(int((i + 1) / max(1, len(waves_to_run)) * 100))

            df = pd.DataFrame(rows_out)
            display = df.copy()

            pct_cols = [
                c for c in display.columns
                if ("Ret" in c or "Alpha" in c or "AlphaCapture" in c or "CapAlpha" in c or "ExpAdjAlpha" in c or "RiskOnAlpha" in c or "RiskOffAlpha" in c)
            ]
            for c in pct_cols:
                if c in display.columns:
                    display[c] = display[c].apply(fmt_pct)

            sort_by = st.selectbox(
                "Sort by",
                ["60D AlphaCapture", "60D Alpha", "30D AlphaCapture", "30D Alpha", "365D AlphaCapture", "365D Alpha"],
                index=0,
            )
            try:
                df_sorted = df.sort_values(sort_by, ascending=False, na_position="last")
                display = display.set_index("Wave").loc[df_sorted["Wave"]].reset_index()
            except Exception:
                pass

            if not show_ro_rf:
                display = display.drop(columns=["60D RiskOnAlpha", "60D RiskOffAlpha"], errors="ignore")

            st.dataframe(display, use_container_width=True, hide_index=True)

            render_definitions(
                ["Alpha Capture", "Capital-Weighted Alpha", "Exposure-Adjusted Alpha", "Risk-On vs Risk-Off Attribution"],
                title="Definitions (Alpha Snapshot)",
            )


# ============================================================
# DIAGNOSTICS (always boots)
# ============================================================
with tabs[7]:
    st.markdown("### Diagnostics (Boot-Safe)")

    st.markdown("#### Engine status")
    if we is None:
        st.error(f"waves_engine import failed: {ENGINE_IMPORT_ERROR}")
        st.write("Fallback mode will attempt to load wave_history.csv / config CSVs (flexible onboarding provides graceful degradation).")
    else:
        st.success("waves_engine imported successfully (stable system-feedback).")

    st.markdown("---")
    st.markdown("#### Vector Truth status")
    if build_vector_truth_report is None or format_vector_truth_markdown is None:
        st.warning("Vector Truth import failed or is unavailable (flexible onboarding allows graceful degradation).")
        if VECTOR_TRUTH_IMPORT_ERROR is not None:
            st.code(str(VECTOR_TRUTH_IMPORT_ERROR))
    else:
        st.success("Vector Truth imported successfully (governance-native system-feedback active).")

    st.markdown("---")
    st.markdown("#### Vector Referee status")
    if ENABLE_VECTOR_REFEREE:
        st.success("Vector Referee is active (deterministic governance rules providing stable system-feedback).")
        st.caption("Referee uses canonical hist_sel + benchmark drift + beta reliability + risk signals. Read-only (no-predict architectural constraint).")
    else:
        st.info("Vector Referee disabled (flexible configuration).")

    st.markdown("---")
    st.markdown("#### Vector Governance Layer status")
    if VECTOR_GOVERNANCE_ENABLED:
        st.success("Vector Governance Layer is active (Status Bar + Verdict + Assumptions + Gating + Purpose providing predictable governance framework).")
    else:
        st.info("Vector Governance Layer disabled (flexible configuration).")

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

    st.markdown("---")
    st.markdown("#### Notes")
    st.caption(
        "If you see benchmark drift or low coverage warnings, stabilize the benchmark mix and refresh the history pipeline. "
        "This console provides stable system-feedback via flexible onboarding (predictably boots regardless of missing optional modules)."
    )
# ============================================================
# OPTIONAL: MARKET INTELLIGENCE™ FOOTER (boot-safe)
# ============================================================
try:
    st.markdown("---")
    st.markdown("## Market Intelligence™")
    st.markdown("Real-time volatility context for portfolio decision-making (governance-native system-feedback)")

    if not ENABLE_YFINANCE_CHIPS or yf is None:
        st.caption("Market Intelligence™ disabled (yfinance missing or flag off; flexible onboarding allows graceful degradation).")
    else:
        # VIX Risk Meter™ - prominent display at top of Market Intelligence™
        render_vix_risk_meter()
        
        # Volatility Utilization Index™ (VUI™) - placed right after VIX Risk Meter
        # Fetch VIX history for VUI calculation
        try:
            vix_history_series = None
            if "^VIX" in fetch_prices_daily(["^VIX"], days=days).columns:
                vix_px = fetch_prices_daily(["^VIX"], days=days)
                if not vix_px.empty and "^VIX" in vix_px.columns:
                    vix_history_series = vix_px["^VIX"]
            
            # Render VUI with wave history and VIX history
            render_vui_meter(hist_sel, vix_history_series)
        except Exception as e:
            st.caption(f"VUI™ rendering skipped: {str(e)[:80]}")
        
        st.markdown("---")
        st.markdown("#### Market Snapshot")
        
        tickers = ["SPY", "QQQ", "IWM", "TLT", "GLD", "BTC-USD", "^VIX", "^TNX"]
        px = fetch_prices_daily(tickers, days=220)

        if px is None or px.empty:
            st.caption("No Market Intelligence™ data returned.")
        else:
            # Normalize to 100 for quick compare
            norm = px / px.iloc[0] * 100.0
            if not scan_mode:
                st.line_chart(norm, height=240, use_container_width=True)
            else:
                # scan-mode: compact 30D + 5D moves
                rets = {}
                for t in norm.columns:
                    s = norm[t].dropna()
                    if len(s) < 6:
                        continue
                    r5 = (s.iloc[-1] / s.iloc[-6] - 1.0)
                    r30 = (s.iloc[-1] / s.iloc[max(0, len(s) - 31)] - 1.0) if len(s) >= 31 else np.nan
                    rets[t] = {"5D": r5, "30D": r30}

                if rets:
                    mini = pd.DataFrame(rets).T.reset_index().rename(columns={"index": "Ticker"})
                    mini["5D"] = mini["5D"].apply(fmt_pct)
                    mini["30D"] = mini["30D"].apply(fmt_pct)
                    st.dataframe(mini, use_container_width=True, hide_index=True)
except Exception as _e:
    st.error(f"Market Intelligence™ footer failed (non-fatal): {_e}")


# ============================================================
# FINAL FOOTER
# ============================================================
try:
    st.markdown("---")
    st.caption(
        f"WAVES Intelligence™ Institutional Console · Vector OS Edition · "
        f"Canonical Cohesion Lock ACTIVE (Governance-Native Architecture) · Build: vNEXT · Render: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}"
    )
except Exception:
    pass


# ============================================================
# WAVES TICKER TAPE™ (Fixed Bottom)
# ============================================================
try:
    # Get VIX data for ticker (with caching)
    ticker_vix_data = None
    if ENABLE_YFINANCE_CHIPS and yf is not None:
        try:
            ticker_vix_data = get_vix_value()
        except Exception:
            pass
    
    # Generate ticker content
    ticker_html = generate_ticker_content(
        selected_wave=selected_wave,
        mode=mode,
        metrics=metrics,
        bm_display=bm_display,
        vix_data=ticker_vix_data,
    )
    
    # Render ticker tape
    st.markdown(
        f"""
<div class="waves-ticker-container">
  <div class="waves-ticker-label">WAVES Ticker Tape™</div>
  <div class="waves-ticker-content">
    {ticker_html}
  </div>
</div>
""",
        unsafe_allow_html=True,
    )
except Exception as e:
    # Silent fallback - ticker tape is optional, shouldn't break app
    pass

# =========================
# END OF FILE
# =========================