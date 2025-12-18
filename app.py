# app.py — WAVES Intelligence™ Institutional Console (Vector OS Edition)
# FULL PRODUCTION FILE (NO PATCHES) — SINGLE-FILE ONLY
#
# vNEXT — CANONICAL COHESION LOCK + IC ONE-PAGER + FIDELITY INSPECTOR + AI EXPLAIN
#         + COMPARATOR + BETA RELIABILITY + DIAGNOSTICS (ALWAYS BOOTS)
#         + VECTOR™ TRUTH LAYER (READ-ONLY, DETERMINISTIC)
#         + ALPHA SNAPSHOT (ALL WAVES) + ALPHA CAPTURE + RISK-ON/OFF ATTRIBUTION
#         + VECTOR AVATAR BUTTON (IMAGE) + ASK VECTOR PANEL  ✅ NEW
#
# Boot-safety rules:
#   • ONE canonical dataset per selected Wave+Mode: hist_sel (source-of-truth)
#   • Guarded optional imports (yfinance / plotly)
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
import base64
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
ENABLE_ALPHA_SNAPSHOT = True  # ALL WAVES snapshot table

# ✅ NEW: Vector avatar + Ask Vector
ENABLE_VECTOR_AVATAR = True


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

/* ✅ NEW: small floating Vector image (decorative) */
.vector-float {
  position: fixed;
  right: 18px;
  bottom: 18px;
  width: 92px;
  opacity: 0.92;
  z-index: 9999;
  filter: drop-shadow(0 10px 18px rgba(0,0,0,0.35));
  border-radius: 18px;
}
.vector-float img { width: 100%; border-radius: 18px; }
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
# ✅ NEW: Vector image loader + sidebar button + “Ask Vector”
# ============================================================
def _read_file_bytes(path: str) -> Optional[bytes]:
    try:
        if not path:
            return None
        if not os.path.exists(path):
            return None
        with open(path, "rb") as f:
            return f.read()
    except Exception:
        return None


def find_vector_image_bytes() -> Tuple[Optional[bytes], str]:
    """
    Tries to locate the new Vector image in common places.
    You can also override from the sidebar.
    """
    # If user already set an override path, use it first
    override = st.session_state.get("vector_img_path_override", "").strip()
    if override:
        b = _read_file_bytes(override)
        if b:
            return b, override

    # Common candidates (edit/add if your folder name differs)
    candidates = [
        # suggested folder style
        os.path.join("vector_assets", "vector_new.png"),
        os.path.join("vector_assets", "vector.png"),
        os.path.join("vector_assets", "vector_avatar.png"),
        os.path.join("vector_assets", "VECTOR.png"),
        # common root filenames
        "vector_new.png",
        "vector.png",
        "vector_avatar.png",
        "vector_robot_clean2.png",
        "vector_full.png",
        # user sometimes uses a vectors/ folder
        os.path.join("vectors", "vector_new.png"),
        os.path.join("vectors", "vector.png"),
        os.path.join("vectors", "vector_avatar.png"),
    ]

    for p in candidates:
        b = _read_file_bytes(p)
        if b:
            return b, p

    return None, ""


def bytes_to_data_uri_png(b: bytes) -> str:
    enc = base64.b64encode(b).decode("utf-8")
    return f"data:image/png;base64,{enc}"


def render_vector_floating_image(vector_bytes: Optional[bytes]):
    if not ENABLE_VECTOR_AVATAR or not vector_bytes:
        return
    try:
        uri = bytes_to_data_uri_png(vector_bytes)
        st.markdown(
            f"""
<div class="vector-float">
  <img src="{uri}" alt="Vector" />
</div>
""",
            unsafe_allow_html=True,
        )
    except Exception:
        pass


def init_vector_state():
    if "vector_open" not in st.session_state:
        st.session_state["vector_open"] = False
    if "vector_question" not in st.session_state:
        st.session_state["vector_question"] = ""


def vector_referee_answer(
    question: str,
    selected_wave: str,
    mode: str,
    metrics: Dict[str, Any],
    cov: Dict[str, Any],
    bm_id: str,
    bm_drift: str,
    beta_grade: str,
    beta_score: float,
) -> str:
    """
    Deterministic “referee” answer: concise + grounded in canonical outputs.
    (No external calls; no hallucinated data.)
    """
    q = (question or "").strip().lower()
    lines: List[str] = []

    lines.append(f"**Vector Referee (canonical)** — Wave: **{selected_wave}** · Mode: **{mode}**")
    lines.append(f"- Confidence: **{cov.get('completeness_score', '—')}** (AgeDays: **{cov.get('age_days', '—')}**) · BM: **{bm_id}** ({bm_drift})")
    lines.append(f"- Beta Reliability: **{beta_grade}** ({fmt_num(beta_score,1)}/100)")
    lines.append(f"- 30D: Return {fmt_pct(metrics.get('r30'))} · Alpha {fmt_pct(metrics.get('a30'))}")
    lines.append(f"- 60D: Return {fmt_pct(metrics.get('r60'))} · Alpha {fmt_pct(metrics.get('a60'))}")
    lines.append(f"- 365D: Return {fmt_pct(metrics.get('r365'))} · Alpha {fmt_pct(metrics.get('a365'))}")

    # Simple routing
    if any(k in q for k in ["alpha", "where", "come from", "source", "explain"]):
        lines.append("")
        lines.append("**Alpha interpretation (governance-safe):**")
        lines.append("- Capital-weighted alpha is Wave minus Benchmark over the same window (investor-experience).")
        lines.append("- If SmartSafe / VIX scaling reduces exposure, raw alpha can look “high” vs a fully-invested benchmark.")
        lines.append("- Use **Exposure-Adjusted Alpha** and **Risk-On/Risk-Off attribution** to show *how* alpha was captured, not just *that* it exists.")
    elif any(k in q for k in ["beta", "reliability", "benchmark", "match"]):
        lines.append("")
        lines.append("**Beta / benchmark fidelity:**")
        lines.append("- A low beta reliability score means the benchmark mix is not tightly explaining the wave’s returns (or the wave has exposure gating).")
        lines.append("- This can be *good* if it’s intentional (VIX scaling / cash sweeps), but it must be explicitly attributed and shown in-console.")
    elif any(k in q for k in ["trust", "confidence", "data", "coverage"]):
        lines.append("")
        lines.append("**Trust / data integrity:**")
        lines.append("- Check CoverageScore, missing business days, and benchmark drift. Freeze benchmark composition for demos.")
    else:
        lines.append("")
        lines.append("Ask me specifically about **alpha source**, **exposure-adjusted alpha**, **risk-on/off**, **beta reliability**, or **benchmark drift**, and I’ll referee it against the canonical dataset.")

    return "\n".join(lines)


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
    "Vector Referee": (
        "A deterministic, read-only referee panel that explains outputs using the canonical dataset "
        "(no external calls; no conflicting math)."
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
# VECTOR™ REFEREE — FLOATING IMAGE + PANEL (NON-DESTRUCTIVE)
# ============================================================

# Session state
if "vector_open" not in st.session_state:
    st.session_state["vector_open"] = False


# --- Vector floating image (bottom-right) ---
st.markdown(
    """
<style>
.vector-float {
    position: fixed;
    bottom: 18px;
    right: 18px;
    z-index: 9999;
    text-align: center;
}
.vector-float img {
    border-radius: 16px;
    cursor: pointer;
    box-shadow: 0 6px 18px rgba(0,0,0,0.35);
}
.vector-float button {
    margin-top: 6px;
    width: 120px;
}
</style>
""",
    unsafe_allow_html=True,
)

st.markdown('<div class="vector-float">', unsafe_allow_html=True)

st.image(
    "assets/vector_avatar.png",   # ✅ NEW IMAGE YOU SAVED
    width=120,
)

if st.button("Vector ⚖️", key="vector_toggle"):
    st.session_state["vector_open"] = not st.session_state["vector_open"]

st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# VECTOR™ REFEREE PANEL (SIDEBAR)
# ============================================================

if st.session_state.get("vector_open", False):

    with st.sidebar:
        st.markdown("## 🤖 Vector™ Referee")
        st.caption("WAVES Intelligence • Governance Layer")

        st.markdown(
            """
Vector is a **read-only referee**, not a decision engine.

Its job:
• Explain *where* alpha is coming from  
• Surface fragility vs durability  
• Highlight governance risks  
• Prevent self-deception  
"""
        )

        st.markdown("---")

        # ---- Context-aware explanation (safe reads only) ----
        if "metrics" in locals():

            st.markdown("### Current Wave Check")

            st.write(f"**Wave:** {selected_wave}")
            st.write(f"**Mode:** {mode}")

            if metrics.get("a30") is not None:
                st.write(f"**30D Alpha:** {fmt_pct(metrics['a30'])}")

            if metrics.get("a60") is not None:
                st.write(f"**60D Alpha:** {fmt_pct(metrics['a60'])}")

            if metrics.get("te") is not None:
                st.write(f"**Tracking Error:** {fmt_pct(metrics['te'])} ({te_band})")

            if metrics.get("mdd") is not None:
                st.write(f"**Max Drawdown:** {fmt_pct(metrics['mdd'])}")

            if "beta_score" in locals():
                st.write(f"**Beta Reliability:** {fmt_num(beta_score,1)}/100")

            st.markdown("---")

            # ---- Governance interpretation (deterministic) ----
            st.markdown("### Vector Interpretation")

            if bm_drift != "stable":
                st.warning("Benchmark drift detected — governance risk.")

            if conf_level == "High":
                st.success("Data integrity is strong. Outputs are IC-ready.")
            elif conf_level == "Medium":
                st.info("Some caution flags present. Review before IC use.")
            else:
                st.error("Low confidence. Do not rely on alpha conclusions yet.")

            if math.isfinite(metrics.get("a30", np.nan)) and abs(metrics["a30"]) > 0.08:
                st.warning("Large recent alpha — confirm benchmark + exposure effects.")

            if math.isfinite(beta_score) and beta_score < 75:
                st.warning("Beta reliability is weak — benchmark mismatch possible.")

        else:
            st.info("Vector awaiting canonical metrics.")

        st.markdown("---")
        st.caption(
            "Vector explanations are deterministic, reproducible, and audit-safe."
        )