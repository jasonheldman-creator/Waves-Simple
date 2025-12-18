# app.py — WAVES Intelligence™ Institutional Console (Vector OS Edition)
# FULL PRODUCTION FILE (NO PATCHES) — SINGLE-FILE ONLY
#
# vNEXT — CANONICAL COHESION LOCK + IC ONE-PAGER + FIDELITY INSPECTOR + AI EXPLAIN
#         + COMPARATOR + BETA RELIABILITY + DIAGNOSTICS (ALWAYS BOOTS)
#         + VECTOR™ TRUTH LAYER (READ-ONLY, DETERMINISTIC)
#         + ALPHA SNAPSHOT (ALL WAVES) + ALPHA CAPTURE + RISK-ON/OFF ATTRIBUTION
#         + VECTOR™ AVATAR (SIDEBAR + OPTIONAL FLOATING DOCK) — BOOT-SAFE
#
# Boot-safety rules:
#   • ONE canonical dataset per selected Wave+Mode: hist_sel (source-of-truth)
#   • Guarded optional imports (yfinance / plotly / pillow)
#   • Guarded vector_truth import (app still boots if missing)
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
import base64
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

# Pillow is optional but strongly recommended for Streamlit Cloud image loading stability.
try:
    from PIL import Image
except Exception:
    Image = None

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
ENABLE_ALPHA_SNAPSHOT = True  # ALL WAVES snapshot table
ENABLE_VECTOR_AVATAR = True   # Vector avatar (sidebar + optional floating dock)


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
# Vector Avatar (BOOT-SAFE)
# ============================================================
def _find_vector_image_path() -> Optional[str]:
    """
    Try multiple filenames so small naming mismatches don't crash the app.
    Put your preferred file at: assets/vector_robot_clean2.png
    """
    candidates = [
        "assets/vector_robot_clean2.png",
        "assets/vector_robot_clean.png",
        "assets/vector_robot.png",
        "assets/vector_full.png",
        "assets/vector_cropped.png",
        "assets/vector_avatar.png",
        "assets/vector.png",
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


def render_vector_avatar(width: int = 120) -> bool:
    """
    Loads image as bytes (PIL preferred), which is Streamlit Cloud-stable.
    Returns True if rendered.
    """
    p = _find_vector_image_path()
    if not p:
        st.caption("Vector image not found. Add it to /assets (e.g., assets/vector_robot_clean2.png).")
        return False

    try:
        if Image is not None:
            img = Image.open(p)
            st.image(img, width=width)
            return True

        # Fallback without PIL
        with open(p, "rb") as f:
            b = f.read()
        st.image(b, width=width)
        return True
    except Exception as e:
        st.caption(f"Vector image load failed: {e}")
        return False


def _vector_floating_dock():
    """
    Optional floating dock. This is guarded; if Streamlit changes behavior,
    it won't crash the app.
    """
    try:
        p = _find_vector_image_path()
        if not p:
            return
        # Build base64 <img> so it doesn't touch st.image (avoids MediaFileStorage edge cases)
        with open(p, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")

        st.markdown(
            f"""
            <style>
            .vector-dock {{
                position: fixed;
                bottom: 18px;
                right: 18px;
                z-index: 99999;
                width: 160px;
                background: rgba(10,15,28,0.72);
                border: 1px solid rgba(255,255,255,0.15);
                border-radius: 16px;
                padding: 10px 10px 8px 10px;
                backdrop-filter: blur(10px);
                -webkit-backdrop-filter: blur(10px);
                box-shadow: 0 10px 30px rgba(0,0,0,0.30);
            }}
            .vector-dock img {{
                width: 140px;
                height: auto;
                display: block;
                margin: 0 auto;
            }}
            .vector-dock .label {{
                text-align: center;
                font-weight: 800;
                letter-spacing: 0.3px;
                margin-top: 6px;
                opacity: 0.92;
                font-size: 0.95rem;
            }}
            .vector-dock .hint {{
                text-align: center;
                opacity: 0.70;
                font-size: 0.80rem;
                margin-top: 2px;
            }}
            @media (max-width: 700px) {{
                .vector-dock {{ width: 140px; right: 12px; bottom: 12px; }}
                .vector-dock img {{ width: 120px; }}
            }}
            </style>
            <div class="vector-dock">
                <img src="data:image/png;base64,{b64}" />
                <div class="label">Vector™</div>
                <div class="hint">Ask me in the sidebar</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    except Exception:
        # Non-fatal by design
        return


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
    "Benchmark Snapshot / Drift": "A fingerprint of benchmark composition. Drift means it changed in-session.",
    "Coverage Score": "0–100 heuristic of data completeness + freshness.",
    "Difficulty vs SPY": "Concentration/diversification proxy (not a promise).",
    "Risk Reaction Score": "0–100 heuristic of risk posture from TE/MaxDD/CVaR.",
    "Analytics Scorecard": "Governance-native reliability grade for analytics outputs (not performance).",
    "Beta (vs Benchmark)": "Regression slope of Wave daily returns vs Benchmark daily returns.",
    "Beta Reliability Score": "0–100: beta-target match + linkage quality (R²) + sample size.",
    "Vector™ Truth Layer": (
        "Read-only truth referee: decomposes alpha sources, reconciles capital-weighted vs exposure-adjusted alpha, "
        "attributes alpha to risk-on/off regimes, and scores durability/fragility."
    ),
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

    if str(bm_drift).lower().strip() != "stable":
        out["What changed recently"].append("Benchmark snapshot drift detected in-session (composition changed).")
    if math.isfinite(age) and age >= 3:
        out["What changed recently"].append(f"History freshness: last datapoint is {int(age)} day(s) old.")
    if rows < 90:
        out["What changed recently"].append("Limited history window may reduce stability of risk metrics.")
    if not out["What changed recently"]:
        out["What changed recently"].append("No major governance changes detected (stable benchmark + fresh coverage).")

    if math.isfinite(a30):
        if a30 > 0.02:
            out["Why the alpha looks like this"].append("Positive 30D alpha: wave has recently outperformed its benchmark mix.")
        elif a30 < -0.02:
            out["Why the alpha looks like this"].append("Negative 30D alpha: recent underperformance versus benchmark mix.")
        else:
            out["Why the alpha looks like this"].append("30D alpha near flat: wave and benchmark moving similarly.")
    if math.isfinite(te):
        out["Why the alpha looks like this"].append(f"Active risk (TE {fmt_pct(te)}) is {te_risk_band(te)}; alpha magnitude often scales with active risk.")

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
# Vector™ Truth: input builders (boot-safe, deterministic)
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
        avg_exp = float(pd.to_numeric(exp, errors="coerce").dropna().mean()) if len(pd.to_numeric(exp, errors="coerce").dropna()) else 1.0
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
            ro = _compound_from_daily(risk_on)
            rf = _compound_from_daily(risk_off)
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


def _vector_truth_panel(selected_wave: str, mode: str, hist_sel: pd.DataFrame, metrics: Dict[str, Any], days: int):
    if not ENABLE_VECTOR_TRUTH or build_vector_truth_report is None or format_vector_truth_markdown is None:
        st.info(f"Vector™ Truth Layer unavailable (import issue): {VECTOR_TRUTH_IMPORT_ERROR}")
        return

    if hist_sel is None or hist_sel.empty:
        st.info("Vector™ Truth Layer: no canonical history available.")
        return

    alpha_series = (pd.to_numeric(hist_sel["wave_ret"], errors="coerce") - pd.to_numeric(hist_sel["bm_ret"], errors="coerce")).dropna()
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
        avg_exp = float(pd.to_numeric(exp_series, errors="coerce").dropna().mean()) if len(exp_series.dropna()) else 1.0
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
# Sidebar controls + Vector™ Assistant (always visible)
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

# Vector assistant block (hard to miss)
if ENABLE_VECTOR_AVATAR:
    st.sidebar.markdown("---")
    st.sidebar.markdown("## Vector™ Assistant")
    rendered = render_vector_avatar(width=170)
    st.sidebar.caption("If you don’t see Vector, confirm your file exists at: assets/vector_robot_clean2.png")
    vector_q = st.sidebar.text_input("Ask Vector (notes)", value="", placeholder="Type a question for your demo…")
    if st.sidebar.button("Ask Vector"):
        if vector_q.strip():
            st.sidebar.success("Vector captured your prompt.")
            st.sidebar.write("**Prompt:** " + vector_q.strip())
        else:
            st.sidebar.info("Type a question first.")

    show_floating = st.sidebar.toggle("Show floating Vector dock (optional)", value=False)
else:
    show_floating = False


# ============================================================
# Canonical source-of-truth (selected wave)
# ============================================================
hist_sel = _standardize_history(compute_wave_history(selected_wave, mode, days=days)) if selected_wave and selected_wave != "(none)" else pd.DataFrame()
cov = coverage_report(hist_sel)

bm_mix = get_benchmark_mix()
bm_rows_now = _bm_rows_for_wave(bm_mix, selected_wave) if selected_wave and selected_wave != "(none)" else pd.DataFrame(columns=["Ticker", "Weight"])
bm_id = benchmark_snapshot_id(selected_wave, bm섭장이
bm_id = benchmark_snapshot_id(selected_wave, bm_mix)
bm_drift = benchmark_drift_status(selected_wave, mode, bm_id)
bm_diff = benchmark_diff_table(selected_wave, mode, bm_rows_now)
bm_diff_proxy = benchmark_difficulty_proxy(bm_rows_now)

# Optional floating Vector dock
if ENABLE_VECTOR_AVATAR and show_floating:
    _vector_floating_dock()

# Compute canonical metrics (ONLY from hist_sel)
metrics = compute_metrics_from_hist(hist_sel)

# Beta reliability
beta_target = beta_target_for_mode(mode)
beta_val, beta_r2, beta_n = beta_and_r2(
    hist_sel["wave_ret"] if (hist_sel is not None and not hist_sel.empty) else pd.Series(dtype=float),
    hist_sel["bm_ret"] if (hist_sel is not None and not hist_sel.empty) else pd.Series(dtype=float),
)
beta_score = beta_reliability_score(beta_val, beta_r2, beta_n, beta_target)

# Risk reaction score
rr_score = risk_reaction_score(metrics.get("te"), metrics.get("mdd"), metrics.get("cvar95"))

# Analytics scorecard for selected
scorecard = compute_analytics_score_for_selected(hist_sel, cov, bm_drift) if ENABLE_SCORECARD else {}

# Confidence label
conf_level, conf_reason = confidence_from_integrity(cov, bm_drift)

# ============================================================
# Header + Sticky Summary Bar
# ============================================================
st.markdown('<div class="waves-big-wave">WAVES Intelligence™ — Institutional Console</div>', unsafe_allow_html=True)
st.markdown(
    f'<div class="waves-subhead">Mode: <b>{mode}</b> · Selected: <b>{selected_wave}</b> · Benchmark Snapshot: <b>{bm_id}</b> · Confidence: <b>{conf_level}</b></div>',
    unsafe_allow_html=True,
)

# Sticky summary (always first thing to scan)
with st.container():
    st.markdown('<div class="waves-sticky">', unsafe_allow_html=True)
    c1, c2, c3, c4, c5, c6 = st.columns(6)

    with c1:
        chip(f"1D: {fmt_pct(metrics.get('r1'))} · α {fmt_pct(metrics.get('a1'))}")
    with c2:
        chip(f"30D: {fmt_pct(metrics.get('r30'))} · α {fmt_pct(metrics.get('a30'))}")
    with c3:
        chip(f"60D: {fmt_pct(metrics.get('r60'))} · α {fmt_pct(metrics.get('a60'))}")
    with c4:
        chip(f"1Y: {fmt_pct(metrics.get('r365'))} · α {fmt_pct(metrics.get('a365'))}")
    with c5:
        chip(f"TE {fmt_pct(metrics.get('te'))} · IR {fmt_num(metrics.get('ir'), 2)}")
    with c6:
        chip(f"Beta {fmt_num(beta_val,2)} · Rel {fmt_num(beta_score,1)}/100")

    st.markdown("</div>", unsafe_allow_html=True)

# ============================================================
# Tabs
# ============================================================
tabs = st.tabs([
    "IC One-Pager",
    "All Waves Snapshot",
    "Benchmark Governance",
    "Vector™ Truth",
    "Risk & Advanced",
    "Comparator",
    "AI Explain",
    "Diagnostics",
])

# ============================================================
# TAB 1: IC One-Pager
# ============================================================
with tabs[0]:
    safe_panel("IC One-Pager", lambda: _tab_ic_one_pager(selected_wave, mode, hist_sel, metrics, cov, bm_drift, bm_rows_now, bm_diff_proxy, rr_score, beta_val, beta_r2, beta_n, beta_target, beta_score, scorecard, conf_reason, scan_mode))

# ============================================================
# TAB 2: All Waves Snapshot
# ============================================================
with tabs[1]:
    safe_panel("All Waves Snapshot", lambda: _tab_all_waves_snapshot(mode, days))

# ============================================================
# TAB 3: Benchmark Governance
# ============================================================
with tabs[2]:
    safe_panel("Benchmark Governance", lambda: _tab_benchmark_governance(selected_wave, mode, bm_id, bm_drift, bm_rows_now, bm_diff, bm_diff_proxy))

# ============================================================
# TAB 4: Vector Truth
# ============================================================
with tabs[3]:
    safe_panel("Vector Truth", lambda: _tab_vector_truth(selected_wave, mode, hist_sel, metrics, days))

# ============================================================
# TAB 5: Risk & Advanced
# ============================================================
with tabs[4]:
    safe_panel("Risk & Advanced", lambda: _tab_risk_advanced(selected_wave, mode, hist_sel, metrics, rr_score, beta_val, beta_r2, beta_n, beta_target, beta_score))

# ============================================================
# TAB 6: Comparator
# ============================================================
with tabs[5]:
    safe_panel("Comparator", lambda: _tab_comparator(selected_wave, mode, hist_sel, days))

# ============================================================
# TAB 7: AI Explain
# ============================================================
with tabs[6]:
    safe_panel("AI Explain", lambda: _tab_ai_explain(cov, bm_drift, metrics, rr_score, beta_val, beta_target, beta_score))

# ============================================================
# TAB 8: Diagnostics (always boots)
# ============================================================
with tabs[7]:
    safe_panel("Diagnostics", lambda: _tab_diagnostics(hist_sel, cov, bm_id, bm_drift, ENGINE_IMPORT_ERROR, VECTOR_TRUTH_IMPORT_ERROR))


# ============================================================
# ----------------------- TAB FUNCTIONS -----------------------
# ============================================================
def _tab_ic_one_pager(
    wave_name: str,
    mode: str,
    hist_sel: pd.DataFrame,
    metrics: Dict[str, Any],
    cov: Dict[str, Any],
    bm_drift: str,
    bm_rows: pd.DataFrame,
    bm_proxy: Dict[str, Any],
    rr_score: float,
    beta_val: float,
    beta_r2: float,
    beta_n: int,
    beta_target: float,
    beta_score: float,
    scorecard: Dict[str, Any],
    conf_reason: str,
    scan_mode: bool,
):
    # Top tiles
    a, b, c, d = st.columns(4)
    with a:
        tile("30D Return", fmt_pct(metrics.get("r30")), f"30D Alpha: {fmt_pct(metrics.get('a30'))}")
    with b:
        tile("60D Return", fmt_pct(metrics.get("r60")), f"60D Alpha: {fmt_pct(metrics.get('a60'))}")
    with c:
        tile("1Y Return", fmt_pct(metrics.get("r365")), f"1Y Alpha: {fmt_pct(metrics.get('a365'))}")
    with d:
        tile("Risk Reaction", fmt_num(rr_score, 1) + "/100", f"Confidence: {conf_reason}")

    st.markdown("### Governance & Integrity")
    g1, g2, g3, g4 = st.columns(4)
    with g1:
        tile("Coverage Score", fmt_num(cov.get("completeness_score"), 1), f"Rows: {fmt_int(cov.get('rows'))}")
    with g2:
        tile("Freshness", f"{fmt_int(cov.get('age_days'))} day(s)", f"Missing BDays: {fmt_int(cov.get('missing_bdays'))}")
    with g3:
        tile("Benchmark Drift", str(bm_drift).upper(), f"Snapshot stable? {'YES' if bm_drift=='stable' else 'NO'}")
    with g4:
        tile("Analytics Score", fmt_num(scorecard.get("AnalyticsScore"), 1), f"Grade: {scorecard.get('Grade','—')} {scorecard.get('Flags','')}")

    st.markdown("### Benchmark Mix (current)")
    if bm_rows is None or bm_rows.empty:
        st.info("Benchmark mix table not available.")
    else:
        df = bm_rows.copy()
        df["Weight"] = (df["Weight"] * 100).round(2)
        st.dataframe(df, use_container_width=True, hide_index=True)

    st.markdown("### Benchmark Difficulty Proxy")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        tile("Top Weight", fmt_pct(bm_proxy.get("top_weight")), "Higher = more concentrated")
    with c2:
        tile("HHI", fmt_num(bm_proxy.get("hhi"), 3), "Concentration index")
    with c3:
        tile("Entropy", fmt_num(bm_proxy.get("entropy"), 2), "Diversification proxy")
    with c4:
        tile("Difficulty vs SPY", fmt_num(bm_proxy.get("difficulty_vs_spy"), 1), "Heuristic")

    st.markdown("### Beta Reliability")
    b1, b2, b3, b4 = st.columns(4)
    with b1:
        tile("Beta", fmt_num(beta_val, 2), f"Target: {fmt_num(beta_target,2)}")
    with b2:
        tile("R²", fmt_num(beta_r2, 2), f"N: {beta_n}")
    with b3:
        tile("Reliability", fmt_num(beta_score, 1) + "/100", f"Band: {beta_band(beta_score)}")
    with b4:
        tile("Tracking Error", fmt_pct(metrics.get("te")), f"IR: {fmt_num(metrics.get('ir'),2)}")

    if not scan_mode:
        st.markdown("### NAV Overlay (Wave vs Benchmark)")
        if go is None:
            st.info("Plotly not available; showing table instead.")
            nav_df = hist_sel[["wave_nav","bm_nav"]].dropna().tail(90) if hist_sel is not None and not hist_sel.empty else pd.DataFrame()
            st.dataframe(nav_df, use_container_width=True)
        else:
            df = hist_sel[["wave_nav","bm_nav"]].dropna() if hist_sel is not None else pd.DataFrame()
            df = df.tail(252) if len(df) > 252 else df
            fig = go.Figure()
            if not df.empty:
                fig.add_trace(go.Scatter(x=df.index, y=df["wave_nav"], name="Wave NAV"))
                fig.add_trace(go.Scatter(x=df.index, y=df["bm_nav"], name="Benchmark NAV"))
            fig.update_layout(height=360, margin=dict(l=10,r=10,t=30,b=10))
            st.plotly_chart(fig, use_container_width=True)

    render_definitions([
        "Canonical (Source of Truth)",
        "Alpha",
        "Tracking Error (TE)",
        "Information Ratio (IR)",
        "Benchmark Snapshot / Drift",
        "Coverage Score",
        "Risk Reaction Score",
        "Beta Reliability Score",
    ], title="Definitions (IC One-Pager)")


def _tab_all_waves_snapshot(mode: str, days: int):
    st.markdown("### All Waves — Return & Alpha Snapshot")
    waves = get_all_waves_safe()
    if not waves:
        st.info("No waves available.")
        return

    rows = []
    for w in waves:
        h = _standardize_history(compute_wave_history(w, mode, days=days))
        m = compute_metrics_from_hist(h)
        rows.append({
            "Wave": w,
            "1D Return": m.get("r1"),
            "1D Alpha": m.get("a1"),
            "30D Return": m.get("r30"),
            "30D Alpha": m.get("a30"),
            "60D Return": m.get("r60"),
            "60D Alpha": m.get("a60"),
            "1Y Return": m.get("r365"),
            "1Y Alpha": m.get("a365"),
        })

    df = pd.DataFrame(rows)
    for c in [c for c in df.columns if c != "Wave"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Display formatted
    out = df.copy()
    for c in [c for c in out.columns if c != "Wave"]:
        out[c] = out[c].apply(lambda x: fmt_pct(x) if pd.notna(x) else "—")

    st.dataframe(out, use_container_width=True, hide_index=True)
    render_definitions(["Return", "Alpha"], title="Definitions (All Waves Snapshot)")


def _tab_benchmark_governance(wave_name: str, mode: str, bm_id: str, bm_drift: str, bm_rows: pd.DataFrame, bm_diff: pd.DataFrame, bm_proxy: Dict[str, Any]):
    st.markdown("### Benchmark Governance")
    st.write(f"**Snapshot ID:** {bm_id}")
    st.write(f"**Drift Status:** {bm_drift.upper()}")

    st.markdown("#### Current Benchmark Mix")
    if bm_rows is None or bm_rows.empty:
        st.info("Benchmark mix not available.")
    else:
        df = bm_rows.copy()
        df["Weight"] = (df["Weight"] * 100).round(2)
        st.dataframe(df, use_container_width=True, hide_index=True)

    st.markdown("#### Drift Diff (in-session)")
    if bm_diff is None or bm_diff.empty:
        st.caption("No material drift changes detected (or first session snapshot).")
    else:
        st.dataframe(bm_diff, use_container_width=True, hide_index=True)

    st.markdown("#### Difficulty Proxy")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        tile("Top Weight", fmt_pct(bm_proxy.get("top_weight")), "")
    with c2:
        tile("HHI", fmt_num(bm_proxy.get("hhi"), 3), "")
    with c3:
        tile("Entropy", fmt_num(bm_proxy.get("entropy"), 2), "")
    with c4:
        tile("Difficulty vs SPY", fmt_num(bm_proxy.get("difficulty_vs_spy"), 1), "")

    render_definitions(["Benchmark Snapshot / Drift", "Difficulty vs SPY"], title="Definitions (Benchmark Governance)")


def _tab_vector_truth(wave_name: str, mode: str, hist_sel: pd.DataFrame, metrics: Dict[str, Any], days: int):
    st.markdown("### Vector™ Truth Layer")
    _vector_truth_panel(wave_name, mode, hist_sel, metrics, days)
    render_definitions(["Vector™ Truth Layer", "Capital-Weighted Alpha", "Exposure-Adjusted Alpha", "Risk-On vs Risk-Off Attribution"], title="Definitions (Vector Truth)")


def _tab_risk_advanced(wave_name: str, mode: str, hist_sel: pd.DataFrame, metrics: Dict[str, Any], rr_score: float, beta_val: float, beta_r2: float, beta_n: int, beta_target: float, beta_score: float):
    st.markdown("### Risk & Advanced")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        tile("Vol (ann.)", fmt_pct(metrics.get("vol")), "")
    with c2:
        tile("Max Drawdown", fmt_pct(metrics.get("mdd")), "")
    with c3:
        tile("VaR 95% (daily)", fmt_pct(metrics.get("var95")), "")
    with c4:
        tile("CVaR 95% (daily)", fmt_pct(metrics.get("cvar95")), "")

    st.markdown("### Ratios")
    r1, r2, r3, r4 = st.columns(4)
    with r1:
        tile("Sharpe", fmt_num(metrics.get("sharpe"), 2), "")
    with r2:
        tile("Sortino", fmt_num(metrics.get("sortino"), 2), "")
    with r3:
        tile("Tracking Error", fmt_pct(metrics.get("te")), "")
    with r4:
        tile("Information Ratio", fmt_num(metrics.get("ir"), 2), "")

    st.markdown("### Beta Reliability")
    b1, b2, b3, b4 = st.columns(4)
    with b1:
        tile("Beta", fmt_num(beta_val, 2), f"Target: {fmt_num(beta_target,2)}")
    with b2:
        tile("R²", fmt_num(beta_r2, 2), f"N: {beta_n}")
    with b3:
        tile("Reliability", fmt_num(beta_score, 1) + "/100", f"Band: {beta_band(beta_score)}")
    with b4:
        tile("Risk Reaction", fmt_num(rr_score, 1) + "/100", "")

    render_definitions(["Tracking Error (TE)", "Max Drawdown (MaxDD)", "VaR 95% (daily)", "CVaR 95% (daily)", "Sharpe", "Sortino", "Beta Reliability Score"], title="Definitions (Risk & Advanced)")


def _tab_comparator(wave_name: str, mode: str, hist_sel: pd.DataFrame, days: int):
    st.markdown("### Comparator")
    st.caption("Compare the selected Wave vs a user-chosen ticker (uses yfinance if available).")

    if yf is None or not ENABLE_YFINANCE_CHIPS:
        st.info("Comparator requires yfinance. Install it or disable this panel.")
        return

    comp = st.text_input("Comparator Ticker", value="SPY")
    if not comp.strip():
        st.info("Enter a ticker to compare.")
        return

    prices = fetch_prices_daily([comp.strip().upper()], days=min(days, 365))
    if prices is None or prices.empty or comp.strip().upper() not in prices.columns:
        st.warning("Could not fetch comparator price series.")
        return

    # Build comparator NAV (normalized)
    p = prices[comp.strip().upper()].dropna()
    if len(p) < 5 or hist_sel is None or hist_sel.empty:
        st.info("Not enough data to compare.")
        return

    # Align
    wnav = hist_sel["wave_nav"].dropna()
    df = pd.concat([wnav.rename("WaveNAV"), p.rename("Comp")], axis=1).dropna()
    if df.empty:
        st.info("No overlapping dates.")
        return

    df["CompNAV"] = df["Comp"] / float(df["Comp"].iloc[0]) * float(df["WaveNAV"].iloc[0])

    if go is None:
        st.dataframe(df[["WaveNAV","CompNAV"]].tail(120), use_container_width=True)
    else:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=df["WaveNAV"], name="Wave NAV"))
        fig.add_trace(go.Scatter(x=df.index, y=df["CompNAV"], name=f"{comp.strip().upper()} (scaled)"))
        fig.update_layout(height=380, margin=dict(l=10,r=10,t=30,b=10))
        st.plotly_chart(fig, use_container_width=True)


def _tab_ai_explain(cov: Dict[str, Any], bm_drift: str, metrics: Dict[str, Any], rr_score: float, beta_val: float, beta_target: float, beta_score: float):
    st.markdown("### AI Explain (Deterministic / Rules-based)")
    if not ENABLE_AI_EXPLAIN:
        st.info("AI Explain disabled.")
        return

    narrative = ai_explain_narrative(
        cov=cov,
        bm_drift=bm_drift,
        r30=safe_float(metrics.get("r30")),
        a30=safe_float(metrics.get("a30")),
        r60=safe_float(metrics.get("r60")),
        a60=safe_float(metrics.get("a60")),
        te=safe_float(metrics.get("te")),
        mdd=safe_float(metrics.get("mdd")),
        sharpe=safe_float(metrics.get("sharpe")),
        cvar95=safe_float(metrics.get("cvar95")),
        rr_score=safe_float(rr_score),
        beta_val=safe_float(beta_val),
        beta_target=safe_float(beta_target),
        beta_score=safe_float(beta_score),
    )

    for k, bullets in narrative.items():
        st.markdown(f"#### {k}")
        for b in bullets:
            st.write("• " + b)

    render_definitions(["Analytics Scorecard", "Coverage Score", "Benchmark Snapshot / Drift", "Beta Reliability Score"], title="Definitions (AI Explain)")


def _tab_diagnostics(hist_sel: pd.DataFrame, cov: Dict[str, Any], bm_id: str, bm_drift: str, engine_err: Exception, vt_err: Exception):
    st.markdown("### Diagnostics (Always Boots)")
    st.markdown("#### System")
    st.write(f"**Engine loaded:** {'YES' if we is not None else 'NO'}")
    if engine_err is not None:
        st.code(str(engine_err))

    st.write(f"**Vector Truth loaded:** {'YES' if build_vector_truth_report is not None else 'NO'}")
    if vt_err is not None:
        st.code(str(vt_err))

    st.markdown("#### Canonical History")
    st.write(f"Rows: {fmt_int(cov.get('rows'))}")
    st.write(f"First: {cov.get('first_date')}")
    st.write(f"Last: {cov.get('last_date')}")
    st.write(f"Age (days): {fmt_int(cov.get('age_days'))}")
    st.write(f"Missing bdays: {fmt_int(cov.get('missing_bdays'))}")
    st.write(f"Coverage score: {fmt_num(cov.get('completeness_score'),1)}")

    st.markdown("#### Benchmark")
    st.write(f"Snapshot ID: {bm_id}")
    st.write(f"Drift: {bm_drift}")

    st.markdown("#### Raw Canonical Tail (debug)")
    if hist_sel is None or hist_sel.empty:
        st.info("hist_sel is empty.")
    else:
        st.dataframe(hist_sel.tail(15), use_container_width=True)


# ============================================================
# End
# ============================================================