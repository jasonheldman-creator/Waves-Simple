# app.py — WAVES Intelligence™ Institutional Console (Vector OS Edition)
# FULL PRODUCTION FILE (NO PATCHES) — DIAGNOSTICS++ BUILD (EXPANDED)
#
# Keeps ALL IRB-1 features:
#   • Benchmark Truth (benchmark mix + difficulty vs SPY)
#   • Mode Separation Proof (mode shown + independent history per mode)
#   • Alpha Attribution (Engine vs Static Basket)
#   • Wave Doctor + What-If Lab (shadow simulation)
#   • Top-10 holdings with Google quote links
#   • Market Intel panel (SPY/QQQ/IWM/TLT/GLD/BTC/VIX/TNX)
#   • Factor Decomposition (simple regression beta map)
#   • Vector OS Insight Layer (narrative + flags)
#
# Adds Diagnostics++ (expanded):
#   • Mode Separation Proof: side-by-side metrics + NAV overlay across ALL modes
#   • Rolling diagnostics: Rolling Alpha / TE / Beta / Vol + alpha persistence
#   • Beta discipline + drift flags (if engine provides beta target)
#   • Correlation matrix across waves (returns) + clustering-lite sort
#   • Data quality / coverage audit + flags (missing days, NaNs, date alignment)
#   • Alpha Captured daily (wave_ret - bm_ret, exposure-scaled if engine provides)
#   • Benchmark drift check: rolling correlation vs benchmark + BM difficulty vs SPY
#   • “Benchmark Truth — Proof”: shows BM mix + BM vs SPY performance (same window)
#
# Notes:
#   • Does NOT modify engine math or baseline results.
#   • What-If Lab is explicitly “shadow simulation”.
#   • Plotly/yfinance optional; app won’t crash if missing.
#   • Built to be robust on Streamlit Cloud (no optional libs required).

from __future__ import annotations

import math
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

import waves_engine as we  # your engine module

# Optional libs (do not crash if missing)
try:
    import yfinance as yf
except Exception:
    yf = None

try:
    import plotly.graph_objects as go
except Exception:
    go = None


# ============================================================
# Streamlit config
# ============================================================
st.set_page_config(
    page_title="WAVES Intelligence™ Console",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# Global UI CSS: sticky bar + scan improvements
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

/* Section header */
.waves-hdr { font-weight: 800; letter-spacing: 0.2px; margin-bottom: 4px; }

/* Tighter tables */
div[data-testid="stDataFrame"] { border-radius: 12px; overflow: hidden; }

/* Reduce huge whitespace for mobile */
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


def google_quote_url(ticker: str) -> str:
    t = str(ticker).replace(" ", "")
    return f"https://www.google.com/finance/quote/{t}"


def _safe_float(x: Any) -> float:
    try:
        v = float(x)
        if math.isnan(v):
            return float("nan")
        return v
    except Exception:
        return float("nan")


def _nanmean(vals: List[float]) -> float:
    arr = np.array([v for v in vals if v is not None and math.isfinite(float(v))], dtype=float)
    if arr.size == 0:
        return float("nan")
    return float(arr.mean())


def _clip(x: float, lo: float, hi: float) -> float:
    try:
        return float(np.clip(float(x), float(lo), float(hi)))
    except Exception:
        return float("nan")


# ============================================================
# Helpers: data fetching & caching
# ============================================================
@st.cache_data(show_spinner=False)
def fetch_spy_vix(days: int = 365) -> pd.DataFrame:
    if yf is None:
        return pd.DataFrame()
    end = datetime.utcnow().date()
    start = end - timedelta(days=days + 10)
    tickers = ["SPY", "^VIX"]

    data = yf.download(
        tickers=tickers,
        start=start.isoformat(),
        end=end.isoformat(),
        interval="1d",
        auto_adjust=True,
        progress=False,
        group_by="column",
    )

    # Normalize columns
    if isinstance(data.columns, pd.MultiIndex):
        if "Adj Close" in data.columns.get_level_values(0):
            data = data["Adj Close"]
        elif "Close" in data.columns.get_level_values(0):
            data = data["Close"]
        else:
            data = data[data.columns.levels[0][0]]

    if isinstance(data.columns, pd.MultiIndex):
        data = data.droplevel(0, axis=1)

    if isinstance(data, pd.Series):
        data = data.to_frame()

    data = data.sort_index().ffill().bfill()
    if len(data) > days:
        data = data.iloc[-days:]
    return data


@st.cache_data(show_spinner=False)
def fetch_market_assets(days: int = 365) -> pd.DataFrame:
    if yf is None:
        return pd.DataFrame()
    end = datetime.utcnow().date()
    start = end - timedelta(days=days + 10)
    tickers = ["SPY", "QQQ", "IWM", "TLT", "GLD", "BTC-USD", "^VIX", "^TNX"]

    data = yf.download(
        tickers=tickers,
        start=start.isoformat(),
        end=end.isoformat(),
        interval="1d",
        auto_adjust=True,
        progress=False,
        group_by="column",
    )

    if isinstance(data.columns, pd.MultiIndex):
        if "Adj Close" in data.columns.get_level_values(0):
            data = data["Adj Close"]
        elif "Close" in data.columns.get_level_values(0):
            data = data["Close"]
        else:
            data = data[data.columns.levels[0][0]]

    if isinstance(data.columns, pd.MultiIndex):
        data = data.droplevel(0, axis=1)

    if isinstance(data, pd.Series):
        data = data.to_frame()

    data = data.sort_index().ffill().bfill()
    if len(data) > days:
        data = data.iloc[-days:]
    return data


@st.cache_data(show_spinner=False)
def fetch_prices_daily(tickers: List[str], days: int = 365) -> pd.DataFrame:
    """Generic daily price fetch used for attribution + what-if shadow sim."""
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

    if isinstance(data.columns, pd.MultiIndex):
        if "Adj Close" in data.columns.get_level_values(0):
            data = data["Adj Close"]
        elif "Close" in data.columns.get_level_values(0):
            data = data["Close"]
        else:
            data = data[data.columns.levels[0][0]]

    if isinstance(data.columns, pd.MultiIndex):
        data = data.droplevel(0, axis=1)

    if isinstance(data, pd.Series):
        data = data.to_frame()

    data = data.sort_index().ffill().bfill()
    if len(data) > days:
        data = data.iloc[-days:]
    return data


@st.cache_data(show_spinner=False)
def compute_wave_history(wave_name: str, mode: str, days: int = 365) -> pd.DataFrame:
    """
    Engine history must return columns:
      wave_nav, bm_nav, wave_ret, bm_ret
    Optional engine columns (if present):
      exposure, beta_target, beta_real, etc.
    """
    try:
        df = we.compute_history_nav(wave_name, mode=mode, days=days)
        if df is None:
            df = pd.DataFrame()
    except Exception:
        df = pd.DataFrame()

    # Normalize minimal schema
    cols = set(df.columns) if not df.empty else set()
    need = ["wave_nav", "bm_nav", "wave_ret", "bm_ret"]
    for c in need:
        if c not in cols:
            df[c] = np.nan

    # Ensure index is datetime-ish
    try:
        if not df.empty and not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index, errors="coerce")
    except Exception:
        pass

    return df


@st.cache_data(show_spinner=False)
def get_benchmark_mix() -> pd.DataFrame:
    try:
        df = we.get_benchmark_mix_table()
        if df is None:
            df = pd.DataFrame()
        return df
    except Exception:
        return pd.DataFrame(columns=["Wave", "Ticker", "Name", "Weight"])


@st.cache_data(show_spinner=False)
def get_wave_holdings(wave_name: str) -> pd.DataFrame:
    try:
        df = we.get_wave_holdings(wave_name)
        if df is None:
            df = pd.DataFrame()
        return df
    except Exception:
        return pd.DataFrame(columns=["Ticker", "Name", "Weight"])


@st.cache_data(show_spinner=False)
def get_engine_version_info() -> Dict[str, Any]:
    """Best effort: show engine version/flags if present."""
    out: Dict[str, Any] = {}
    for attr in ["ENGINE_VERSION", "__version__", "VERSION", "BUILD_TAG", "MODE_BASE_EXPOSURE"]:
        try:
            v = getattr(we, attr, None)
            if v is not None:
                out[attr] = v
        except Exception:
            pass
    return out


# ============================================================
# Core metrics
# ============================================================
def ret_from_nav(nav: pd.Series, window: int) -> float:
    nav = safe_series(nav)
    if len(nav) < 2:
        return float("nan")
    if len(nav) < window:
        window = len(nav)
    if window < 2:
        return float("nan")
    sub = nav.iloc[-window:]
    start = float(sub.iloc[0])
    end = float(sub.iloc[-1])
    if start <= 0:
        return float("nan")
    return (end / start) - 1.0


def annualized_vol(daily_ret: pd.Series) -> float:
    daily_ret = safe_series(daily_ret)
    if len(daily_ret) < 2:
        return float("nan")
    return float(daily_ret.std() * np.sqrt(252))


def max_drawdown(nav: pd.Series) -> float:
    nav = safe_series(nav)
    if len(nav) < 2:
        return float("nan")
    running_max = nav.cummax()
    dd = (nav / running_max) - 1.0
    return float(dd.min())


def tracking_error(daily_wave: pd.Series, daily_bm: pd.Series) -> float:
    daily_wave = safe_series(daily_wave)
    daily_bm = safe_series(daily_bm)
    if len(daily_wave) < 2 or len(daily_bm) < 2:
        return float("nan")
    df = pd.concat([daily_wave.rename("w"), daily_bm.rename("b")], axis=1).dropna()
    if df.shape[0] < 2:
        return float("nan")
    diff = df["w"] - df["b"]
    return float(diff.std() * np.sqrt(252))


def information_ratio(nav_wave: pd.Series, nav_bm: pd.Series, te: float) -> float:
    nav_wave = safe_series(nav_wave)
    nav_bm = safe_series(nav_bm)
    if len(nav_wave) < 2 or len(nav_bm) < 2:
        return float("nan")
    if te is None or (isinstance(te, float) and (math.isnan(te) or te <= 0)):
        return float("nan")
    ret_wave = ret_from_nav(nav_wave, window=len(nav_wave))
    ret_bm = ret_from_nav(nav_bm, window=len(nav_bm))
    excess = ret_wave - ret_bm
    return float(excess / te)


def beta_vs_benchmark(wave_ret: pd.Series, bm_ret: pd.Series) -> float:
    """β_real = Cov(w, b) / Var(b) using aligned daily returns."""
    w = safe_series(wave_ret)
    b = safe_series(bm_ret)
    df = pd.concat([w.rename("w"), b.rename("b")], axis=1).dropna()
    if df.shape[0] < 20:
        return float("nan")
    var_b = float(df["b"].var())
    if not math.isfinite(var_b) or var_b <= 0:
        return float("nan")
    cov = float(df["w"].cov(df["b"]))
    return float(cov / var_b)


def corr_safe(a: pd.Series, b: pd.Series) -> float:
    try:
        df = pd.concat([safe_series(a).rename("a"), safe_series(b).rename("b")], axis=1).dropna()
        if df.shape[0] < 20:
            return float("nan")
        return float(df["a"].corr(df["b"]))
    except Exception:
        return float("nan")


def regress_factors(wave_ret: pd.Series, factor_ret: pd.DataFrame) -> Dict[str, float]:
    wave_ret = safe_series(wave_ret)
    if wave_ret.empty or factor_ret is None or factor_ret.empty:
        return {col: float("nan") for col in factor_ret.columns} if factor_ret is not None else {}

    df = pd.concat([wave_ret.rename("wave"), factor_ret], axis=1).dropna()
    if df.shape[0] < 60 or df.shape[1] < 2:
        return {col: float("nan") for col in factor_ret.columns}

    y = df["wave"].values
    X = df[factor_ret.columns].values
    X_design = np.concatenate([np.ones((X.shape[0], 1)), X], axis=1)

    try:
        beta, *_ = np.linalg.lstsq(X_design, y, rcond=None)
    except Exception:
        return {col: float("nan") for col in factor_ret.columns}

    betas = beta[1:]
    return {col: float(b) for col, b in zip(factor_ret.columns, betas)}


# ============================================================
# Benchmark truth helpers
# ============================================================
@st.cache_data(show_spinner=False)
def compute_spy_nav(days: int = 365) -> pd.Series:
    px = fetch_prices_daily(["SPY"], days=days)
    if px.empty or "SPY" not in px.columns:
        return pd.Series(dtype=float)
    nav = (1.0 + px["SPY"].pct_change().fillna(0.0)).cumprod()
    nav.name = "spy_nav"
    return nav


def benchmark_source_label(wave_name: str, bm_mix_df: pd.DataFrame) -> str:
    try:
        static_dict = getattr(we, "BENCHMARK_WEIGHTS_STATIC", None)
        if isinstance(static_dict, dict) and wave_name in static_dict and static_dict.get(wave_name):
            return "Static Override (Engine)"
    except Exception:
        pass
    if bm_mix_df is not None and not bm_mix_df.empty:
        return "Auto-Composite (Dynamic)"
    return "Unknown"


def get_beta_target_if_available(mode: str) -> float:
    """
    Best-effort: if engine has a beta target map, use it. Otherwise NaN.
    We DO NOT assume names; we try common attribute names.
    """
    candidates = ["MODE_BETA_TARGET", "BETA_TARGET_BY_MODE", "BETA_TARGETS", "BETA_TARGET"]
    for name in candidates:
        try:
            obj = getattr(we, name, None)
            if isinstance(obj, dict) and mode in obj:
                return float(obj[mode])
        except Exception:
            pass
    return float("nan")


def get_exposure_series_if_available(hist: pd.DataFrame) -> Optional[pd.Series]:
    """If engine emits exposure per day, use it; else None."""
    if hist is None or hist.empty:
        return None
    for c in ["exposure", "Exposure", "expo", "EXPOSURE"]:
        if c in hist.columns:
            try:
                s = pd.to_numeric(hist[c], errors="coerce")
                if s.notna().sum() > 10:
                    return s
            except Exception:
                pass
    return None


# ============================================================
# Alpha Attribution (Engine vs Static Basket)
# ============================================================
def _weights_from_df(df: pd.DataFrame, ticker_col: str = "Ticker", weight_col: str = "Weight") -> pd.Series:
    if df is None or df.empty:
        return pd.Series(dtype=float)
    w = df[[ticker_col, weight_col]].copy()
    w[ticker_col] = w[ticker_col].astype(str)
    w[weight_col] = pd.to_numeric(w[weight_col], errors="coerce").fillna(0.0)
    w = w.groupby(ticker_col, as_index=True)[weight_col].sum()
    total = float(w.sum())
    if total <= 0:
        return pd.Series(dtype=float)
    return (w / total).sort_index()


@st.cache_data(show_spinner=False)
def compute_static_nav_from_weights(weights: pd.Series, days: int = 365) -> pd.Series:
    if weights is None or weights.empty:
        return pd.Series(dtype=float)
    tickers = list(weights.index)
    px = fetch_prices_daily(tickers, days=days)
    if px.empty:
        return pd.Series(dtype=float)

    weights_aligned = weights.reindex(px.columns).fillna(0.0)
    daily_ret = px.pct_change().fillna(0.0)
    port_ret = (daily_ret * weights_aligned).sum(axis=1)

    nav = (1.0 + port_ret).cumprod()
    nav.name = "static_nav"
    return nav


@st.cache_data(show_spinner=False)
def compute_alpha_attribution(wave_name: str, mode: str, days: int = 365) -> Dict[str, float]:
    out: Dict[str, float] = {}

    hist = compute_wave_history(wave_name, mode=mode, days=days)
    if hist.empty or len(hist) < 2:
        return out

    nav_wave = hist["wave_nav"]
    nav_bm = hist["bm_nav"]
    wave_ret = hist["wave_ret"]
    bm_ret = hist["bm_ret"]

    eng_ret = ret_from_nav(nav_wave, window=len(nav_wave))
    bm_ret_total = ret_from_nav(nav_bm, window=len(nav_bm))
    alpha_vs_bm = eng_ret - bm_ret_total

    hold_df = get_wave_holdings(wave_name)
    hold_w = _weights_from_df(hold_df, ticker_col="Ticker", weight_col="Weight")
    static_nav = compute_static_nav_from_weights(hold_w, days=days)
    static_ret = ret_from_nav(static_nav, window=len(static_nav)) if len(static_nav) >= 2 else float("nan")
    overlay_pp = (eng_ret - static_ret) if (pd.notna(eng_ret) and pd.notna(static_ret)) else float("nan")

    spy_px = fetch_prices_daily(["SPY"], days=days)
    spy_nav = (spy_px["SPY"].pct_change().fillna(0.0) + 1.0).cumprod() if "SPY" in spy_px.columns else pd.Series(dtype=float)
    spy_ret = ret_from_nav(spy_nav, window=len(spy_nav)) if len(spy_nav) >= 2 else float("nan")

    alpha_vs_spy = eng_ret - spy_ret if (pd.notna(eng_ret) and pd.notna(spy_ret)) else float("nan")
    benchmark_difficulty = bm_ret_total - spy_ret if (pd.notna(bm_ret_total) and pd.notna(spy_ret)) else float("nan")

    te = tracking_error(wave_ret, bm_ret)
    ir = information_ratio(nav_wave, nav_bm, te)
    beta_real = beta_vs_benchmark(wave_ret, bm_ret)
    beta_target = get_beta_target_if_available(mode)

    out["Engine Return"] = float(eng_ret)
    out["Static Basket Return"] = float(static_ret) if pd.notna(static_ret) else float("nan")
    out["Overlay Contribution (Engine - Static)"] = float(overlay_pp) if pd.notna(overlay_pp) else float("nan")
    out["Benchmark Return"] = float(bm_ret_total)
    out["Alpha vs Benchmark"] = float(alpha_vs_bm)
    out["SPY Return"] = float(spy_ret) if pd.notna(spy_ret) else float("nan")
    out["Alpha vs SPY"] = float(alpha_vs_spy) if pd.notna(alpha_vs_spy) else float("nan")
    out["Benchmark Difficulty (BM - SPY)"] = float(benchmark_difficulty) if pd.notna(benchmark_difficulty) else float("nan")
    out["Tracking Error (TE)"] = float(te)
    out["Information Ratio (IR)"] = float(ir)
    out["β_real (Wave vs BM)"] = float(beta_real)
    out["β_target (if available)"] = float(beta_target)

    out["Wave Vol"] = float(annualized_vol(wave_ret))
    out["Benchmark Vol"] = float(annualized_vol(bm_ret))
    out["Wave MaxDD"] = float(max_drawdown(nav_wave))
    out["Benchmark MaxDD"] = float(max_drawdown(nav_bm))

    # Daily alpha captured (exposure-scaled if exposure provided)
    expo = get_exposure_series_if_available(hist)
    alpha_daily = (pd.to_numeric(wave_ret, errors="coerce") - pd.to_numeric(bm_ret, errors="coerce")).fillna(0.0)
    if expo is not None:
        try:
            expo2 = pd.to_numeric(expo, errors="coerce").fillna(1.0)
            alpha_captured = (alpha_daily * expo2).rename("alpha_captured")
        except Exception:
            alpha_captured = alpha_daily.rename("alpha_captured")
    else:
        alpha_captured = alpha_daily.rename("alpha_captured")

    try:
        out["Alpha Captured (avg daily, 60d)"] = float(alpha_captured.tail(60).mean()) if len(alpha_captured) >= 10 else float("nan")
    except Exception:
        out["Alpha Captured (avg daily, 60d)"] = float("nan")

    return out


# ============================================================
# WaveScore proto (console-side display helper)
# ============================================================
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


def compute_wavescore_for_all_waves(all_waves: List[str], mode: str, days: int = 365) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for wave in all_waves:
        hist = compute_wave_history(wave, mode=mode, days=days)
        if hist.empty or len(hist) < 20:
            rows.append(
                {
                    "Wave": wave,
                    "WaveScore": float("nan"),
                    "Grade": "N/A",
                    "Return Quality": float("nan"),
                    "Risk Control": float("nan"),
                    "Consistency": float("nan"),
                    "Resilience": float("nan"),
                    "Efficiency": float("nan"),
                    "Transparency": 10.0,
                    "IR_365D": float("nan"),
                    "Alpha_365D": float("nan"),
                }
            )
            continue

        nav_wave = hist["wave_nav"]
        nav_bm = hist["bm_nav"]
        wave_ret = hist["wave_ret"]
        bm_ret = hist["bm_ret"]

        ret_365_wave = ret_from_nav(nav_wave, window=len(nav_wave))
        ret_365_bm = ret_from_nav(nav_bm, window=len(nav_bm))
        alpha_365 = ret_365_wave - ret_365_bm

        vol_wave = annualized_vol(wave_ret)
        vol_bm = annualized_vol(bm_ret)

        te = tracking_error(wave_ret, bm_ret)
        ir = information_ratio(nav_wave, nav_bm, te)

        mdd_wave = max_drawdown(nav_wave)
        mdd_bm = max_drawdown(nav_bm)

        hit_rate = float((pd.to_numeric(wave_ret, errors="coerce") >= pd.to_numeric(bm_ret, errors="coerce")).mean()) if len(wave_ret) > 0 else float("nan")

        if len(nav_wave) > 1:
            trough = float(pd.to_numeric(nav_wave, errors="coerce").min())
            peak = float(pd.to_numeric(nav_wave, errors="coerce").max())
            last = float(pd.to_numeric(nav_wave, errors="coerce").iloc[-1])
            if peak > trough and trough > 0:
                recovery_frac = float((last - trough) / (peak - trough))
                recovery_frac = float(np.clip(recovery_frac, 0.0, 1.0))
            else:
                recovery_frac = float("nan")
        else:
            recovery_frac = float("nan")

        vol_ratio = vol_wave / vol_bm if (vol_bm and not math.isnan(vol_bm)) else float("nan")

        rq_ir = float(np.clip(ir, 0.0, 1.5) / 1.5 * 15.0) if not math.isnan(ir) else 0.0
        rq_alpha = float(np.clip((alpha_365 + 0.05) / 0.15, 0.0, 1.0) * 10.0) if not math.isnan(alpha_365) else 0.0
        return_quality = float(np.clip(rq_ir + rq_alpha, 0.0, 25.0))

        if math.isnan(vol_ratio):
            risk_control = 0.0
        else:
            penalty = max(0.0, abs(vol_ratio - 0.9) - 0.1)
            risk_control = float(np.clip(1.0 - penalty / 0.6, 0.0, 1.0) * 25.0)

        consistency = float(np.clip(hit_rate, 0.0, 1.0) * 15.0) if not math.isnan(hit_rate) else 0.0

        if math.isnan(recovery_frac) or math.isnan(mdd_wave) or math.isnan(mdd_bm):
            resilience = 0.0
        else:
            rec_part = float(np.clip(recovery_frac, 0.0, 1.0) * 6.0)
            dd_edge = (mdd_bm - mdd_wave)
            dd_part = float(np.clip((dd_edge + 0.10) / 0.20, 0.0, 1.0) * 4.0)
            resilience = float(np.clip(rec_part + dd_part, 0.0, 10.0))

        efficiency = float(np.clip((0.25 - te) / 0.20, 0.0, 1.0) * 15.0) if not math.isnan(te) else 0.0

        transparency = 10.0
        total = float(np.clip(return_quality + risk_control + consistency + resilience + efficiency + transparency, 0.0, 100.0))
        grade = _grade_from_score(total)

        rows.append(
            {
                "Wave": wave,
                "WaveScore": total,
                "Grade": grade,
                "Return Quality": return_quality,
                "Risk Control": risk_control,
                "Consistency": consistency,
                "Resilience": resilience,
                "Efficiency": efficiency,
                "Transparency": transparency,
                "IR_365D": ir,
                "Alpha_365D": alpha_365,
            }
        )

    df = pd.DataFrame(rows) if rows else pd.DataFrame()
    return df.sort_values("Wave") if not df.empty else df


# ============================================================
# Table display formatting (percent display, no math changes)
# ============================================================
def build_formatter_map(df: pd.DataFrame) -> Dict[str, Any]:
    if df is None or df.empty:
        return {}

    fmt: Dict[str, Any] = {}

    # IMPORTANT: this list must be properly closed (fixes your unterminated-string crash)
    pct_keywords = [
        " Ret",
        " Return",
        " Alpha",
        "Vol",
        "Volatility",
        "MaxDD",
        "Max Drawdown",
        "Tracking Error",
        "TE",
        "Benchmark Difficulty",
        "BM Difficulty",
        "Alpha Captured",
        "Drawdown",
    ]

    for c in df.columns:
        cs = str(c)

        if cs == "WaveScore":
            fmt[c] = (lambda v: fmt_score(v))
            continue
        if cs in ["Return Quality", "Risk Control", "Consistency", "Resilience", "Efficiency", "Transparency"]:
            fmt[c] = (lambda v: fmt_num(v, 1))
            continue

        if cs.startswith("β") or cs.lower().startswith("beta") or cs.startswith("β_"):
            fmt[c] = (lambda v: fmt_num(v, 2))
            continue

        if ("IR" in cs) and ("Ret" not in cs) and ("Return" not in cs):
            fmt[c] = (lambda v: fmt_num(v, 2))
            continue

        if any(k in cs for k in pct_keywords):
            fmt[c] = (lambda v: fmt_pct(v, 2))

    return fmt


# ============================================================
# Row highlighting utilities (selected wave + alpha tint)
# ============================================================
def style_selected_and_alpha(df: pd.DataFrame, selected_wave: str):
    alpha_cols = [c for c in df.columns if ("Alpha" in str(c)) or str(c).endswith("α") or ("Alpha Captured" in str(c))]
    fmt_map = build_formatter_map(df)

    def row_style(row: pd.Series):
        styles = [""] * len(row)

        if "Wave" in df.columns and str(row.get("Wave", "")) == str(selected_wave):
            styles = ["background-color: rgba(255,255,255,0.07); font-weight: 700;"] * len(row)

        for i, col in enumerate(df.columns):
            if col in alpha_cols:
                try:
                    v = float(row[col])
                except Exception:
                    continue
                if math.isnan(v):
                    continue
                if v > 0:
                    styles[i] += "background-color: rgba(0, 255, 140, 0.10);"
                elif v < 0:
                    styles[i] += "background-color: rgba(255, 80, 80, 0.10);"
        return styles

    styler = df.style.apply(row_style, axis=1)
    if fmt_map:
        styler = styler.format(fmt_map)
    return styler


def show_df(df: pd.DataFrame, selected_wave: str, key: str):
    try:
        st.dataframe(style_selected_and_alpha(df, selected_wave), use_container_width=True, key=key)
    except Exception:
        st.dataframe(df, use_container_width=True, key=key)


# ============================================================
# One-click Wave jump (best-effort selection events)
# ============================================================
def selectable_table_jump(df: pd.DataFrame, key: str) -> None:
    if df is None or df.empty or "Wave" not in df.columns:
        st.info("No waves available to jump.")
        return

    # Streamlit row selection is best-effort depending on version
    try:
        event = st.dataframe(
            df,
            use_container_width=True,
            key=key,
            on_select="rerun",
            selection_mode="single-row",
        )
        sel = getattr(event, "selection", None)
        if sel and isinstance(sel, dict):
            rows = sel.get("rows", [])
            if rows:
                idx = int(rows[0])
                wave = str(df.iloc[idx]["Wave"])
                if wave:
                    st.session_state["selected_wave"] = wave
                    st.rerun()
        return
    except Exception:
        pass

    st.dataframe(df, use_container_width=True, key=f"{key}_fallback")
    pick = st.selectbox("Jump to Wave", list(df["Wave"]), key=f"{key}_pick")
    if st.button("Jump", key=f"{key}_btn"):
        st.session_state["selected_wave"] = pick
        st.rerun()


# ============================================================
# Alpha Heatmap View (All Waves x Timeframe)
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

        # 1D alpha
        if len(nav_w) >= 2 and len(nav_b) >= 2:
            r1w = float(nav_w.iloc[-1] / nav_w.iloc[-2] - 1.0)
            r1b = float(nav_b.iloc[-1] / nav_b.iloc[-2] - 1.0)
            a1 = r1w - r1b
        else:
            a1 = np.nan

        r30w = ret_from_nav(nav_w, min(30, len(nav_w)))
        r30b = ret_from_nav(nav_b, min(30, len(nav_b)))
        a30 = r30w - r30b

        r60w = ret_from_nav(nav_w, min(60, len(nav_w)))
        r60b = ret_from_nav(nav_b, min(60, len(nav_b)))
        a60 = r60w - r60b

        r365w = ret_from_nav(nav_w, len(nav_w))
        r365b = ret_from_nav(nav_b, len(nav_b))
        a365 = r365w - r365b

        rows.append({"Wave": wname, "1D Alpha": a1, "30D Alpha": a30, "60D Alpha": a60, "365D Alpha": a365})

    return pd.DataFrame(rows).sort_values("Wave")


def plot_alpha_heatmap(alpha_df: pd.DataFrame, selected_wave: str, title: str):
    if go is None or alpha_df is None or alpha_df.empty:
        st.info("Heatmap unavailable (Plotly missing or no data).")
        return

    df = alpha_df.copy()
    cols = [c for c in ["1D Alpha", "30D Alpha", "60D Alpha", "365D Alpha"] if c in df.columns]
    if not cols:
        st.info("No alpha columns to plot.")
        return

    # Put selected wave on top
    if "Wave" in df.columns and selected_wave in set(df["Wave"]):
        top = df[df["Wave"] == selected_wave]
        rest = df[df["Wave"] != selected_wave]
        df = pd.concat([top, rest], axis=0)

    z = df[cols].values
    y = df["Wave"].tolist()
    x = cols

    v = float(np.nanmax(np.abs(z))) if np.isfinite(z).any() else 0.10
    if not math.isfinite(v) or v <= 0:
        v = 0.10
    zmin, zmax = (-float(v), float(v))

    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=x,
            y=y,
            zmin=zmin,
            zmax=zmax,
            colorbar=dict(title="Alpha"),
        )
    )
    fig.update_layout(
        title=title,
        height=min(900, 240 + 22 * max(10, len(y))),
        margin=dict(l=80, r=40, t=60, b=40),
        xaxis_title="Timeframe",
        yaxis_title="Wave",
    )
    st.plotly_chart(fig, use_container_width=True)


# ============================================================
# Rolling diagnostics (alpha, beta, TE, vol, persistence)
# ============================================================
def rolling_metrics(hist: pd.DataFrame, window: int = 60) -> pd.DataFrame:
    """
    Produces rolling:
      - roll_alpha (rolling mean of daily alpha)
      - roll_te (rolling TE annualized)
      - roll_beta (rolling beta vs BM)
      - roll_vol (rolling vol annualized)
      - roll_corr (rolling correlation vs BM)
      - alpha_persist (rolling % days alpha>0)
    """
    if hist is None or hist.empty:
        return pd.DataFrame()

    w = pd.to_numeric(hist["wave_ret"], errors="coerce")
    b = pd.to_numeric(hist["bm_ret"], errors="coerce")
    df = pd.DataFrame({"w": w, "b": b}).dropna()
    if df.shape[0] < max(20, window):
        return pd.DataFrame()

    alpha = (df["w"] - df["b"]).rename("alpha")
    out = pd.DataFrame(index=df.index)
    out["roll_alpha"] = alpha.rolling(window).mean()
    out["alpha_persist"] = (alpha > 0).rolling(window).mean()

    # Rolling TE: std of (w-b) annualized
    out["roll_te"] = alpha.rolling(window).std() * np.sqrt(252)

    # Rolling vol: std of wave return annualized
    out["roll_vol"] = df["w"].rolling(window).std() * np.sqrt(252)

    # Rolling beta: cov/var over window
    def _roll_beta(x: pd.DataFrame) -> float:
        try:
            if x.shape[0] < 20:
                return np.nan
            var_b = float(x["b"].var())
            if not math.isfinite(var_b) or var_b <= 0:
                return np.nan
            cov = float(x["w"].cov(x["b"]))
            return float(cov / var_b)
        except Exception:
            return np.nan

    out["roll_beta"] = df[["w", "b"]].rolling(window).apply(lambda arr: np.nan, raw=False)
    # Replace with correct rolling apply on DataFrame:
    try:
        out["roll_beta"] = df[["w", "b"]].rolling(window).apply(lambda x: _roll_beta(pd.DataFrame(x, columns=["w", "b"])), raw=False)
    except Exception:
        # Fallback approximate beta: corr*(vol_w/vol_b)
        try:
            corr_rb = df["w"].rolling(window).corr(df["b"])
            vol_w = df["w"].rolling(window).std()
            vol_b = df["b"].rolling(window).std()
            out["roll_beta"] = corr_rb * (vol_w / vol_b)
        except Exception:
            out["roll_beta"] = np.nan

    # Rolling correlation
    try:
        out["roll_corr"] = df["w"].rolling(window).corr(df["b"])
    except Exception:
        out["roll_corr"] = np.nan

    return out.dropna(how="all")


def plot_rolling_bundle(roll: pd.DataFrame, title_prefix: str):
    if roll is None or roll.empty:
        st.info("Rolling diagnostics unavailable (not enough data).")
        return

    # Plotly if available, else Streamlit line charts
    cols1 = ["roll_alpha", "alpha_persist"]
    cols2 = ["roll_te", "roll_vol"]
    cols3 = ["roll_beta", "roll_corr"]

    if go is not None:
        fig1 = go.Figure()
        if "roll_alpha" in roll.columns:
            fig1.add_trace(go.Scatter(x=roll.index, y=roll["roll_alpha"], name="Rolling α (mean daily)", mode="lines"))
        if "alpha_persist" in roll.columns:
            fig1.add_trace(go.Scatter(x=roll.index, y=roll["alpha_persist"], name="α Persistence (% days > 0)", mode="lines", yaxis="y2"))
        fig1.update_layout(
            title=f"{title_prefix} — Alpha & Persistence",
            height=320,
            margin=dict(l=40, r=40, t=50, b=30),
            yaxis=dict(title="Rolling α (daily)"),
            yaxis2=dict(title="Persistence", overlaying="y", side="right", rangemode="tozero"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig1, use_container_width=True)

        fig2 = go.Figure()
        if "roll_te" in roll.columns:
            fig2.add_trace(go.Scatter(x=roll.index, y=roll["roll_te"], name="Rolling TE (ann)", mode="lines"))
        if "roll_vol" in roll.columns:
            fig2.add_trace(go.Scatter(x=roll.index, y=roll["roll_vol"], name="Rolling Vol (ann)", mode="lines"))
        fig2.update_layout(
            title=f"{title_prefix} — Risk (TE & Vol)",
            height=320,
            margin=dict(l=40, r=40, t=50, b=30),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig2, use_container_width=True)

        fig3 = go.Figure()
        if "roll_beta" in roll.columns:
            fig3.add_trace(go.Scatter(x=roll.index, y=roll["roll_beta"], name="Rolling β vs BM", mode="lines"))
        if "roll_corr" in roll.columns:
            fig3.add_trace(go.Scatter(x=roll.index, y=roll["roll_corr"], name="Rolling Corr vs BM", mode="lines", yaxis="y2"))
        fig3.update_layout(
            title=f"{title_prefix} — Beta & Correlation",
            height=320,
            margin=dict(l=40, r=40, t=50, b=30),
            yaxis=dict(title="β"),
            yaxis2=dict(title="Corr", overlaying="y", side="right"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig3, use_container_width=True)
    else:
        # Streamlit charts
        show = roll.copy()
        st.line_chart(show[[c for c in cols1 if c in show.columns]].dropna(how="all"))
        st.line_chart(show[[c for c in cols2 if c in show.columns]].dropna(how="all"))
        st.line_chart(show[[c for c in cols3 if c in show.columns]].dropna(how="all"))


# ============================================================
# Data quality / coverage audit
# ============================================================
def coverage_audit(hist: pd.DataFrame) -> Dict[str, Any]:
    if hist is None or hist.empty:
        return {"ok": False, "message": "No history available."}

    out: Dict[str, Any] = {"ok": True}
    out["rows"] = int(hist.shape[0])
    out["start"] = str(hist.index.min()) if isinstance(hist.index, pd.DatetimeIndex) else "—"
    out["end"] = str(hist.index.max()) if isinstance(hist.index, pd.DatetimeIndex) else "—"

    # NaN counts
    for c in ["wave_nav", "bm_nav", "wave_ret", "bm_ret"]:
        try:
            out[f"nan_{c}"] = int(pd.to_numeric(hist[c], errors="coerce").isna().sum())
        except Exception:
            out[f"nan_{c}"] = None

    # Gaps in calendar days between rows
    try:
        if isinstance(hist.index, pd.DatetimeIndex) and hist.shape[0] >= 3:
            diffs = hist.index.to_series().diff().dt.days
            out["max_gap_days"] = int(diffs.max()) if diffs.notna().any() else None
            out["pct_gap_gt3d"] = float((diffs > 3).mean()) if diffs.notna().any() else float("nan")
        else:
            out["max_gap_days"] = None
            out["pct_gap_gt3d"] = float("nan")
    except Exception:
        out["max_gap_days"] = None
        out["pct_gap_gt3d"] = float("nan")

    flags: List[str] = []
    if out.get("rows", 0) < 60:
        flags.append("Limited history (<60 rows).")
    if any((out.get(f"nan_{c}", 0) or 0) > max(2, 0.05 * out.get("rows", 1)) for c in ["wave_nav", "bm_nav", "wave_ret", "bm_ret"]):
        flags.append("High NaN rate (check engine output).")
    if (out.get("max_gap_days") or 0) >= 10:
        flags.append("Large date gaps (>=10 days).")
    if math.isfinite(float(out.get("pct_gap_gt3d", float("nan")))) and float(out["pct_gap_gt3d"]) > 0.20:
        flags.append("Frequent gaps >3 days (data coverage).")
    out["flags"] = flags
    return out


# ============================================================
# Correlation matrix across waves
# ============================================================
@st.cache_data(show_spinner=False)
def build_wave_returns_matrix(all_waves: List[str], mode: str, days: int = 365) -> pd.DataFrame:
    """
    Returns a DataFrame of aligned daily wave returns: columns=Wave
    """
    series_list: List[pd.Series] = []
    for w in all_waves:
        hist = compute_wave_history(w, mode=mode, days=days)
        if hist is None or hist.empty:
            continue
        s = pd.to_numeric(hist["wave_ret"], errors="coerce").rename(w)
        if isinstance(hist.index, pd.DatetimeIndex):
            s.index = hist.index
        series_list.append(s)

    if not series_list:
        return pd.DataFrame()

    df = pd.concat(series_list, axis=1).dropna(how="all")
    return df


def corr_matrix(df_ret: pd.DataFrame) -> pd.DataFrame:
    if df_ret is None or df_ret.empty:
        return pd.DataFrame()
    try:
        return df_ret.corr()
    except Exception:
        return pd.DataFrame()


def sort_corr_by_reference(corr: pd.DataFrame, ref: str) -> pd.DataFrame:
    """
    Simple “clustering-lite”: sort by correlation to selected wave.
    """
    if corr is None or corr.empty or ref not in corr.columns:
        return corr
    try:
        order = corr[ref].sort_values(ascending=False).index.tolist()
        return corr.loc[order, order]
    except Exception:
        return corr


def plot_corr_heatmap(corr: pd.DataFrame, title: str):
    if go is None or corr is None or corr.empty:
        st.info("Correlation heatmap unavailable (Plotly missing or no data).")
        return
    z = corr.values
    fig = go.Figure(data=go.Heatmap(z=z, x=corr.columns.tolist(), y=corr.index.tolist(), zmin=-1, zmax=1))
    fig.update_layout(title=title, height=min(850, 260 + 18 * max(10, corr.shape[0])), margin=dict(l=80, r=30, t=55, b=30))
    st.plotly_chart(fig, use_container_width=True)


# ============================================================
# Wave Doctor (diagnostics + suggestions)
# ============================================================
def wave_doctor_assess(
    wave_name: str,
    mode: str,
    days: int = 365,
    alpha_warn: float = 0.08,
    te_warn: float = 0.20,
    vol_warn: float = 0.30,
    mdd_warn: float = -0.25,
    beta_drift_warn: float = 0.07,
) -> Dict[str, Any]:
    hist = compute_wave_history(wave_name, mode=mode, days=days)
    if hist.empty or len(hist) < 2:
        return {"ok": False, "message": "Not enough data to run Wave Doctor."}

    nav_w = hist["wave_nav"]
    nav_b = hist["bm_nav"]
    ret_w = hist["wave_ret"]
    ret_b = hist["bm_ret"]

    r365_w = ret_from_nav(nav_w, len(nav_w))
    r365_b = ret_from_nav(nav_b, len(nav_b))
    a365 = r365_w - r365_b

    r30_w = ret_from_nav(nav_w, min(30, len(nav_w)))
    r30_b = ret_from_nav(nav_b, min(30, len(nav_b)))
    a30 = r30_w - r30_b

    vol_w = annualized_vol(ret_w)
    vol_b = annualized_vol(ret_b)
    te = tracking_error(ret_w, ret_b)
    ir = information_ratio(nav_w, nav_b, te)
    mdd_w = max_drawdown(nav_w)
    mdd_b = max_drawdown(nav_b)

    beta_real = beta_vs_benchmark(ret_w, ret_b)
    beta_target = get_beta_target_if_available(mode)

    spy_nav = compute_spy_nav(days=days)
    spy_ret = ret_from_nav(spy_nav, len(spy_nav)) if len(spy_nav) >= 2 else float("nan")
    bm_difficulty = (r365_b - spy_ret) if (pd.notna(r365_b) and pd.notna(spy_ret)) else float("nan")

    # Alpha captured (daily avg)
    expo = get_exposure_series_if_available(hist)
    alpha_daily = (pd.to_numeric(ret_w, errors="coerce") - pd.to_numeric(ret_b, errors="coerce")).fillna(0.0)
    if expo is not None:
        alpha_cap = (alpha_daily * pd.to_numeric(expo, errors="coerce").fillna(1.0))
    else:
        alpha_cap = alpha_daily
    alpha_cap_60 = float(alpha_cap.tail(60).mean()) if len(alpha_cap) >= 10 else float("nan")

    flags: List[str] = []
    diagnosis: List[str] = []
    recs: List[str] = []

    if pd.notna(a30) and abs(a30) >= alpha_warn:
        flags.append("Large 30D alpha (verify benchmark + coverage)")
        diagnosis.append("30D alpha is unusually large. This can happen from real signal, but also from benchmark mix or data coverage shifts.")
        recs.append("Check Benchmark Truth panel for mix stability + benchmark difficulty vs SPY.")

    if pd.notna(a365) and a365 < 0:
        diagnosis.append("365D alpha is negative. This might reflect true underperformance, or a tougher benchmark composition.")
        if pd.notna(bm_difficulty) and bm_difficulty > 0.03:
            diagnosis.append("Benchmark difficulty is positive vs SPY (benchmark outperformed SPY), so alpha is harder on this window.")
            recs.append("For validation, compare to SPY/QQQ-style proxy benchmark to isolate engine effect.")
    else:
        if pd.notna(a365) and a365 > 0.06:
            diagnosis.append("365D alpha is solidly positive. Lock benchmark mix snapshot for reproducibility in demos.")

    if pd.notna(te) and te > te_warn:
        flags.append("High tracking error (active risk elevated)")
        diagnosis.append("Tracking error is high; the wave behaves very differently than its benchmark.")
        recs.append("Tighten exposure caps / reduce tilt strength (What-If Lab — shadow only).")

    if pd.notna(vol_w) and vol_w > vol_warn:
        flags.append("High volatility")
        diagnosis.append("Volatility is elevated relative to institutional tolerances.")
        recs.append("Lower vol target / tighten exposure caps (What-If Lab — shadow only).")

    if pd.notna(mdd_w) and mdd_w < mdd_warn:
        flags.append("Deep drawdown")
        diagnosis.append("Drawdown is deep. Consider stronger SmartSafe posture in stress regimes.")
        recs.append("Increase safe fraction / regime gating (What-If Lab — shadow only).")

    if pd.notna(vol_b) and pd.notna(vol_w) and vol_b > 0 and (vol_w / vol_b) > 1.6:
        flags.append("Volatility much higher than benchmark")
        diagnosis.append("Wave volatility is much higher than benchmark; this can inflate wins/losses.")
        recs.append("Tighten exposure caps + reduce tilt strength to stabilize.")

    if pd.notna(beta_target) and pd.notna(beta_real):
        drift = abs(beta_real - beta_target)
        if drift > beta_drift_warn:
            flags.append("Beta drift beyond tolerance")
            diagnosis.append(f"β_real deviates from β_target by {drift:.2f}.")
            recs.append("Verify mode scaling is distinct + check exposure series (Mode Separation Proof + Rolling Beta).")

    if pd.notna(alpha_cap_60) and abs(alpha_cap_60) > 0.0015:
        diagnosis.append("Alpha Captured (exposure-scaled if available) shows persistent daily edge (60D avg).")
    elif pd.notna(alpha_cap_60):
        diagnosis.append("Alpha Captured is small on 60D avg; alpha may be coming from infrequent spikes or benchmark effects.")

    if not diagnosis:
        diagnosis.append("No major anomalies detected by Wave Doctor on the selected window.")

    return {
        "ok": True,
        "metrics": {
            "Return_365D": r365_w,
            "Alpha_365D": a365,
            "Return_30D": r30_w,
            "Alpha_30D": a30,
            "Vol_Wave": vol_w,
            "Vol_Benchmark": vol_b,
            "TE": te,
            "IR": ir,
            "MaxDD_Wave": mdd_w,
            "MaxDD_Benchmark": mdd_b,
            "β_real": beta_real,
            "β_target": beta_target,
            "Benchmark_Difficulty_BM_minus_SPY": bm_difficulty,
            "Alpha_Captured_avg_daily_60D": alpha_cap_60,
        },
        "flags": flags,
        "diagnosis": diagnosis,
        "recommendations": list(dict.fromkeys(recs)),
    }


# ============================================================
# What-If Lab (shadow sim)
# ============================================================
def _regime_from_spy_60d(spy_nav: pd.Series) -> pd.Series:
    spy_nav = safe_series(spy_nav)
    if spy_nav.empty:
        return pd.Series(dtype=str)
    r60 = spy_nav / spy_nav.shift(60) - 1.0

    def label(x: float) -> str:
        if pd.isna(x):
            return "neutral"
        if x <= -0.12:
            return "panic"
        if x <= -0.04:
            return "downtrend"
        if x < 0.06:
            return "neutral"
        return "uptrend"

    return r60.apply(label)


def _vix_exposure_factor_series(vix: pd.Series, mode: str) -> pd.Series:
    vix = safe_series(vix).astype(float)
    if vix.empty:
        return pd.Series(dtype=float)

    def f(v: float) -> float:
        if pd.isna(v) or v <= 0:
            base = 1.0
        elif v < 15:
            base = 1.15
        elif v < 20:
            base = 1.05
        elif v < 25:
            base = 0.95
        elif v < 30:
            base = 0.85
        elif v < 40:
            base = 0.75
        else:
            base = 0.60
        if mode == "Alpha-Minus-Beta":
            base -= 0.05
        elif mode == "Private Logic":
            base += 0.05
        return float(np.clip(base, 0.5, 1.3))

    return vix.apply(f)


def _vix_safe_fraction_series(vix: pd.Series, mode: str) -> pd.Series:
    vix = safe_series(vix).astype(float)
    if vix.empty:
        return pd.Series(dtype=float)

    def g(v: float) -> float:
        if pd.isna(v) or v <= 0:
            base = 0.0
        elif v < 18:
            base = 0.00
        elif v < 24:
            base = 0.05
        elif v < 30:
            base = 0.15
        elif v < 40:
            base = 0.25
        else:
            base = 0.40
        if mode == "Alpha-Minus-Beta":
            base *= 1.5
        elif mode == "Private Logic":
            base *= 0.7
        return float(np.clip(base, 0.0, 0.8))

    return vix.apply(g)


@st.cache_data(show_spinner=False)
def simulate_whatif_nav(
    wave_name: str,
    mode: str,
    days: int,
    tilt_strength: float,
    vol_target: float,
    extra_safe_boost: float,
    exp_min: float,
    exp_max: float,
    freeze_benchmark: bool,
) -> pd.DataFrame:
    hold_df = get_wave_holdings(wave_name)
    weights = _weights_from_df(hold_df, "Ticker", "Weight")
    if weights.empty:
        return pd.DataFrame()

    tickers = list(weights.index)
    needed = set(tickers + ["SPY", "^VIX", "SGOV", "BIL", "SHY"])
    px = fetch_prices_daily(list(needed), days=days)
    if px.empty or "SPY" not in px.columns or "^VIX" not in px.columns:
        return pd.DataFrame()

    px = px.sort_index().ffill().bfill()
    if len(px) > days:
        px = px.iloc[-days:]

    rets = px.pct_change().fillna(0.0)
    w = weights.reindex(px.columns).fillna(0.0)

    spy_nav = (1.0 + rets["SPY"]).cumprod()
    regime = _regime_from_spy_60d(spy_nav)
    vix = px["^VIX"]

    vix_exposure = _vix_exposure_factor_series(vix, mode)
    vix_safe = _vix_safe_fraction_series(vix, mode)

    base_expo = 1.0
    try:
        base_map = getattr(we, "MODE_BASE_EXPOSURE", None)
        if isinstance(base_map, dict) and mode in base_map:
            base_expo = float(base_map[mode])
    except Exception:
        pass

    regime_exposure_map = {"panic": 0.80, "downtrend": 0.90, "neutral": 1.00, "uptrend": 1.10}
    try:
        rm = getattr(we, "REGIME_EXPOSURE", None)
        if isinstance(rm, dict):
            regime_exposure_map = {k: float(v) for k, v in rm.items()}
    except Exception:
        pass

    def regime_gate(mode_in: str, reg: str) -> float:
        try:
            rg = getattr(we, "REGIME_GATING", None)
            if isinstance(rg, dict) and mode_in in rg and reg in rg[mode_in]:
                return float(rg[mode_in][reg])
        except Exception:
            pass
        fallback = {
            "Standard": {"panic": 0.50, "downtrend": 0.30, "neutral": 0.10, "uptrend": 0.00},
            "Alpha-Minus-Beta": {"panic": 0.75, "downtrend": 0.50, "neutral": 0.25, "uptrend": 0.05},
            "Private Logic": {"panic": 0.40, "downtrend": 0.25, "neutral": 0.05, "uptrend": 0.00},
        }
        return float(fallback.get(mode_in, fallback["Standard"]).get(reg, 0.10))

    safe_ticker = "SGOV" if "SGOV" in rets.columns else ("BIL" if "BIL" in rets.columns else ("SHY" if "SHY" in rets.columns else "SPY"))
    safe_ret = rets[safe_ticker]
    mom60 = px / px.shift(60) - 1.0

    wave_ret: List[float] = []
    dates: List[pd.Timestamp] = []

    for dtt in rets.index:
        r = rets.loc[dtt]

        mom_row = mom60.loc[dtt] if dtt in mom60.index else None
        if mom_row is not None:
            mom_clipped = mom_row.reindex(px.columns).fillna(0.0).clip(lower=-0.30, upper=0.30)
            tilt = 1.0 + tilt_strength * mom_clipped
            ew = (w * tilt).clip(lower=0.0)
        else:
            ew = w.copy()

        ew_hold = ew.reindex(tickers).fillna(0.0)
        s = float(ew_hold.sum())
        if s > 0:
            rw = ew_hold / s
        else:
            rw = w.reindex(tickers).fillna(0.0)
            s2 = float(rw.sum())
            rw = (rw / s2) if s2 > 0 else rw

        port_risk_ret = float((r.reindex(tickers).fillna(0.0) * rw).sum())

        if len(wave_ret) >= 20:
            recent = np.array(wave_ret[-20:], dtype=float)
            realized = float(recent.std() * np.sqrt(252))
        else:
            realized = float(vol_target)

        vol_adj = 1.0
        if realized > 0 and math.isfinite(realized):
            vol_adj = float(np.clip(vol_target / realized, 0.7, 1.3))

        reg = str(regime.get(dtt, "neutral"))
        reg_expo = float(regime_exposure_map.get(reg, 1.0))

        vix_expo = float(vix_exposure.get(dtt, 1.0))
        vix_gate = float(vix_safe.get(dtt, 0.0))

        expo = float(np.clip(base_expo * reg_expo * vol_adj * vix_expo, exp_min, exp_max))

        sf = float(np.clip(regime_gate(mode, reg) + vix_gate + extra_safe_boost, 0.0, 0.95))
        rf = 1.0 - sf

        total = sf * float(safe_ret.get(dtt, 0.0)) + rf * expo * port_risk_ret

        if mode == "Private Logic" and len(wave_ret) >= 20:
            recent = np.array(wave_ret[-20:], dtype=float)
            daily_vol = float(recent.std())
            if daily_vol > 0 and math.isfinite(daily_vol):
                shock = 2.0 * daily_vol
                if total <= -shock:
                    total = total * 1.30
                elif total >= shock:
                    total = total * 0.70

        wave_ret.append(float(total))
        dates.append(pd.Timestamp(dtt))

    wave_ret_s = pd.Series(wave_ret, index=pd.Index(dates, name="Date"), name="whatif_ret")
    wave_nav = (1.0 + wave_ret_s).cumprod().rename("whatif_nav")

    out = pd.DataFrame({"whatif_nav": wave_nav, "whatif_ret": wave_ret_s})

    if freeze_benchmark:
        hist_eng = compute_wave_history(wave_name, mode=mode, days=days)
        if not hist_eng.empty and "bm_nav" in hist_eng.columns:
            bm_nav = hist_eng["bm_nav"].reindex(out.index).ffill().bfill()
            bm_ret = hist_eng["bm_ret"].reindex(out.index).fillna(0.0)
            out["bm_nav"] = bm_nav
            out["bm_ret"] = bm_ret
    else:
        if "SPY" in px.columns:
            spy_ret = rets["SPY"].reindex(out.index).fillna(0.0)
            spy_nav2 = (1.0 + spy_ret).cumprod()
            out["bm_nav"] = spy_nav2
            out["bm_ret"] = spy_ret

    return out


# ============================================================
# Sidebar
# ============================================================
def _get_all_waves() -> List[str]:
    try:
        w = we.get_all_waves()
        if w is None:
            return []
        return list(w)
    except Exception:
        return []


def _get_all_modes() -> List[str]:
    try:
        m = we.get_modes()
        if m is None:
            return []
        return list(m)
    except Exception:
        return ["Standard", "Alpha-Minus-Beta", "Private Logic"]


all_waves = _get_all_waves()
all_modes = _get_all_modes()

if "selected_wave" not in st.session_state:
    st.session_state["selected_wave"] = all_waves[0] if all_waves else ""
if "mode" not in st.session_state:
    st.session_state["mode"] = all_modes[0] if all_modes else "Standard"

with st.sidebar:
    st.title("WAVES Intelligence™")
    st.caption("Mini Bloomberg Console • Vector OS™")

    # Engine version
    ev = get_engine_version_info()
    if ev:
        with st.expander("Engine info", expanded=False):
            for k, v in ev.items():
                st.write(f"**{k}:** {v}")

    if all_modes:
        st.selectbox(
            "Mode",
            all_modes,
            index=all_modes.index(st.session_state["mode"]) if st.session_state["mode"] in all_modes else 0,
            key="mode",
        )
    else:
        st.text_input("Mode", value=st.session_state.get("mode", "Standard"), key="mode")

    if all_waves:
        st.selectbox(
            "Select Wave",
            all_waves,
            index=all_waves.index(st.session_state["selected_wave"]) if st.session_state["selected_wave"] in all_waves else 0,
            key="selected_wave",
        )
    else:
        st.text_input("Select Wave", value=st.session_state.get("selected_wave", ""), key="selected_wave")

    st.markdown("---")
    st.markdown("**Display settings**")
    history_days = st.slider("History window (days)", min_value=60, max_value=730, value=365, step=15)

    st.markdown("---")
    st.markdown("**Wave Doctor settings**")
    alpha_warn = st.slider("Alpha alert threshold (abs, 30D)", 0.02, 0.20, 0.08, 0.01)
    te_warn = st.slider("Tracking error alert threshold", 0.05, 0.40, 0.20, 0.01)
    beta_drift_warn = st.slider("Beta drift alert (|β_real-β_target|)", 0.02, 0.25, 0.07, 0.01)

    st.markdown("---")
    st.markdown("**Rolling diagnostics settings**")
    roll_win = st.slider("Rolling window (days)", 20, 180, 60, 5)

mode = st.session_state["mode"]
selected_wave = st.session_state["selected_wave"]

if not selected_wave:
    st.error("No Waves available (engine returned empty list). Verify waves_engine.get_all_waves().")
    st.stop()


# ============================================================
# Page header
# ============================================================
st.title("WAVES Intelligence™ Institutional Console")
st.caption("Live Alpha Capture • SmartSafe™ • Multi-Asset • Crypto • Gold • Income Ladders")


# ============================================================
# Pinned Summary Bar (Sticky)
# ============================================================
h_bar = compute_wave_history(selected_wave, mode=mode, days=max(365, history_days))
bar_r30 = bar_a30 = bar_r365 = bar_a365 = float("nan")
bar_te = bar_ir = float("nan")
bar_beta = float("nan")
bar_beta_t = float("nan")
bar_src = "—"

bm_mix_for_src = get_benchmark_mix()
bar_src = benchmark_source_label(selected_wave, bm_mix_for_src)

if not h_bar.empty and len(h_bar) >= 2:
    nav_w = h_bar["wave_nav"]
    nav_b = h_bar["bm_nav"]
    ret_w = h_bar["wave_ret"]
    ret_b = h_bar["bm_ret"]

    r30w = ret_from_nav(nav_w, min(30, len(nav_w)))
    r30b = ret_from_nav(nav_b, min(30, len(nav_b)))
    bar_r30 = r30w
    bar_a30 = r30w - r30b

    r365w = ret_from_nav(nav_w, len(nav_w))
    r365b = ret_from_nav(nav_b, len(nav_b))
    bar_r365 = r365w
    bar_a365 = r365w - r365b

    bar_te = tracking_error(ret_w, ret_b)
    bar_ir = information_ratio(nav_w, nav_b, bar_te)
    bar_beta = beta_vs_benchmark(ret_w, ret_b)
    bar_beta_t = get_beta_target_if_available(mode)

spy_vix = fetch_spy_vix(days=min(365, max(120, history_days)))
reg_now = "—"
vix_last = float("nan")
if not spy_vix.empty and "SPY" in spy_vix.columns and "^VIX" in spy_vix.columns and len(spy_vix) > 60:
    vix_last = float(spy_vix["^VIX"].iloc[-1])
    spy_nav = (1.0 + spy_vix["SPY"].pct_change().fillna(0.0)).cumprod()
    r60 = spy_nav / spy_nav.shift(60) - 1.0

    def lab(x: Any) -> str:
        try:
            if pd.isna(x):
                return "neutral"
            x = float(x)
            if x <= -0.12:
                return "panic"
            if x <= -0.04:
                return "downtrend"
            if x < 0.06:
                return "neutral"
            return "uptrend"
        except Exception:
            return "neutral"

    reg_now = str(r60.apply(lab).iloc[-1])

ws_snap = compute_wavescore_for_all_waves(all_waves, mode=mode, days=365)
ws_val = "—"
ws_grade = "—"
if ws_snap is not None and not ws_snap.empty:
    rr = ws_snap[ws_snap["Wave"] == selected_wave]
    if not rr.empty:
        ws_val = fmt_score(float(rr.iloc[0]["WaveScore"]))
        ws_grade = str(rr.iloc[0]["Grade"])

beta_chip = "—"
if pd.notna(bar_beta) and pd.notna(bar_beta_t):
    beta_chip = f"{fmt_num(bar_beta,2)} vs {fmt_num(bar_beta_t,2)}"
elif pd.notna(bar_beta):
    beta_chip = f"{fmt_num(bar_beta,2)}"

st.markdown(
    f"""
<div class="waves-sticky">
  <div class="waves-hdr">📌 Live Summary</div>
  <span class="waves-chip">Mode: <b>{mode}</b></span>
  <span class="waves-chip">Wave: <b>{selected_wave}</b></span>
  <span class="waves-chip">Benchmark: <b>{bar_src}</b></span>
  <span class="waves-chip">Regime: <b>{reg_now}</b></span>
  <span class="waves-chip">VIX: <b>{fmt_num(vix_last, 1) if not math.isnan(vix_last) else "—"}</b></span>
  <span class="waves-chip">30D α: <b>{fmt_pct(bar_a30)}</b> · 30D r: <b>{fmt_pct(bar_r30)}</b></span>
  <span class="waves-chip">365D α: <b>{fmt_pct(bar_a365)}</b> · 365D r: <b>{fmt_pct(bar_r365)}</b></span>
  <span class="waves-chip">TE: <b>{fmt_pct(bar_te)}</b> · IR: <b>{fmt_num(bar_ir, 2)}</b></span>
  <span class="waves-chip">β: <b>{beta_chip}</b></span>
  <span class="waves-chip">WaveScore: <b>{ws_val}</b> ({ws_grade})</span>
</div>
""",
    unsafe_allow_html=True,
)


# ============================================================
# Tabs (expanded)
# ============================================================
tab_console, tab_diag, tab_market, tab_factors, tab_vector = st.tabs(
    ["Console", "Diagnostics++", "Market Intel", "Factor Decomposition", "Vector OS Insight Layer"]
)


# ============================================================
# TAB 1: Console (scan-first)
# ============================================================
with tab_console:
    st.subheader("🔥 Alpha Heatmap View (All Waves × Timeframe)")
    st.caption("Fast scan. Jump table highlights selected wave. Values display as % (math unchanged).")

    alpha_df = build_alpha_matrix(all_waves, mode)
    plot_alpha_heatmap(alpha_df, selected_wave, title=f"Alpha Heatmap — Mode: {mode}")

    st.markdown("### 🧭 One-Click Jump Table (ranked by mean alpha)")
    jump_df = alpha_df.copy()
    jump_df["RankScore"] = jump_df[["1D Alpha", "30D Alpha", "60D Alpha", "365D Alpha"]].mean(axis=1, skipna=True)
    jump_df = jump_df.sort_values("RankScore", ascending=False)

    show_df(jump_df, selected_wave, key="wave_jump_table_fmt")
    selectable_table_jump(jump_df, key="wave_jump_table_select")

    st.markdown("---")

    st.subheader("Market Regime Monitor — SPY vs VIX")
    spy_vix2 = fetch_spy_vix(days=history_days)

    if spy_vix2.empty or "SPY" not in spy_vix2.columns or "^VIX" not in spy_vix2.columns:
        st.warning("Unable to load SPY/VIX data at the moment.")
    else:
        spy = spy_vix2["SPY"].copy()
        vix = spy_vix2["^VIX"].copy()
        spy_norm = (spy / spy.iloc[0] * 100.0) if len(spy) > 0 else spy

        if go is not None:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=spy_vix2.index, y=spy_norm, name="SPY (Index = 100)", mode="lines"))
            fig.add_trace(go.Scatter(x=spy_vix2.index, y=vix, name="VIX Level", mode="lines", yaxis="y2"))
            fig.update_layout(
                margin=dict(l=40, r=40, t=40, b=40),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                xaxis=dict(title="Date"),
                yaxis=dict(title="SPY (Indexed)", rangemode="tozero"),
                yaxis2=dict(title="VIX", overlaying="y", side="right", rangemode="tozero"),
                height=380,
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.line_chart(pd.DataFrame({"SPY_idx": spy_norm, "VIX": vix}))

    st.markdown("---")

    st.subheader("🧾 All Waves Overview (1D / 30D / 60D / 365D Returns + Alpha)")
    overview_rows: List[Dict[str, Any]] = []

    for wname in all_waves:
        hist = compute_wave_history(wname, mode=mode, days=365)
        if hist.empty or len(hist) < 2:
            overview_rows.append(
                {"Wave": wname, "1D Ret": np.nan, "1D Alpha": np.nan, "30D Ret": np.nan, "30D Alpha": np.nan,
                 "60D Ret": np.nan, "60D Alpha": np.nan, "365D Ret": np.nan, "365D Alpha": np.nan}
            )
            continue

        nav_w = hist["wave_nav"]
        nav_b = hist["bm_nav"]

        # 1D
        if len(nav_w) >= 2 and len(nav_b) >= 2:
            r1w = float(nav_w.iloc[-1] / nav_w.iloc[-2] - 1.0)
            r1b = float(nav_b.iloc[-1] / nav_b.iloc[-2] - 1.0)
            a1 = r1w - r1b
        else:
            r1w = np.nan
            a1 = np.nan

        # 30D
        r30w = ret_from_nav(nav_w, min(30, len(nav_w)))
        r30b = ret_from_nav(nav_b, min(30, len(nav_b)))
        a30 = r30w - r30b

        # 60D
        r60w = ret_from_nav(nav_w, min(60, len(nav_w)))
        r60b = ret_from_nav(nav_b, min(60, len(nav_b)))
        a60 = r60w - r60b

        # 365D (or full available)
        r365w = ret_from_nav(nav_w, len(nav_w))
        r365b = ret_from_nav(nav_b, len(nav_b))
        a365 = r365w - r365b

        overview_rows.append(
            {
                "Wave": wname,
                "1D Ret": r1w,
                "1D Alpha": a1,
                "30D Ret": r30w,
                "30D Alpha": a30,
                "60D Ret": r60w,
                "60D Alpha": a60,
                "365D Ret": r365w,
                "365D Alpha": a365,
            }
        )

    overview_df = pd.DataFrame(overview_rows)
    show_df(overview_df, selected_wave, key="all_waves_overview")

    st.markdown("---")

    st.subheader(f"📌 Selected Wave — {selected_wave}")
    hold = get_wave_holdings(selected_wave)

    if hold is None or hold.empty:
        st.warning("No holdings returned by engine for this wave.")
    else:
        hold2 = hold.copy()
        if "Weight" in hold2.columns:
            hold2["Weight"] = pd.to_numeric(hold2["Weight"], errors="coerce")
        if "Ticker" in hold2.columns:
            hold2["Ticker"] = hold2["Ticker"].astype(str)

        top10 = hold2.sort_values("Weight", ascending=False).head(10) if "Weight" in hold2.columns else hold2.head(10)

        st.markdown("### 🧾 Top 10 Holdings (clickable)")
        for _, r in top10.iterrows():
            t = str(r.get("Ticker", ""))
            wgt = r.get("Weight", np.nan)
            nm = str(r.get("Name", t))
            if t:
                url = google_quote_url(t)
                st.markdown(f"- **[{t}]({url})** — {nm} — **{fmt_pct(wgt)}**")
        st.markdown("### Full Holdings")
        show_df(hold2, selected_wave, key="holdings_full")

    st.markdown("---")

    st.subheader("✅ Benchmark Truth + Attribution (Engine vs Basket)")
    colA, colB = st.columns(2)

    with colA:
        st.markdown("#### Benchmark Mix (as used by Engine)")
        bm_mix = get_benchmark_mix()
        if bm_mix is None or bm_mix.empty:
            st.info("No benchmark mix table returned by engine.")
        else:
            if "Wave" in bm_mix.columns:
                show_df(bm_mix[bm_mix["Wave"] == selected_wave], selected_wave, key="bm_mix")
            else:
                show_df(bm_mix, selected_wave, key="bm_mix_all")

    with colB:
        st.markdown("#### Attribution Snapshot (365D)")
        attrib = compute_alpha_attribution(selected_wave, mode=mode, days=365)
        if not attrib:
            st.info("Attribution unavailable.")
        else:
            arows = []
            for k, v in attrib.items():
                if any(x in k for x in ["Return", "Alpha", "Vol", "MaxDD", "TE", "Difficulty", "Captured"]):
                    arows.append({"Metric": k, "Value": fmt_pct(v)})
                elif "IR" in k:
                    arows.append({"Metric": k, "Value": fmt_num(v, 2)})
                elif "β" in k or "beta" in k.lower():
                    arows.append({"Metric": k, "Value": fmt_num(v, 2)})
                else:
                    arows.append({"Metric": k, "Value": fmt_num(v, 6)})
            st.dataframe(pd.DataFrame(arows), use_container_width=True)

    st.markdown("---")

    st.subheader("🩺 Wave Doctor")
    wd = wave_doctor_assess(selected_wave, mode=mode, days=365, alpha_warn=alpha_warn, te_warn=te_warn, beta_drift_warn=beta_drift_warn)
    if not wd.get("ok", False):
        st.info(wd.get("message", "Wave Doctor unavailable."))
    else:
        m = wd["metrics"]
        mdf = pd.DataFrame(
            [
                {"Metric": "365D Return", "Value": fmt_pct(m["Return_365D"])},
                {"Metric": "365D Alpha", "Value": fmt_pct(m["Alpha_365D"])},
                {"Metric": "30D Return", "Value": fmt_pct(m["Return_30D"])},
                {"Metric": "30D Alpha", "Value": fmt_pct(m["Alpha_30D"])},
                {"Metric": "Alpha Captured (avg daily, 60D)", "Value": fmt_pct(m["Alpha_Captured_avg_daily_60D"])},
                {"Metric": "Vol (Wave)", "Value": fmt_pct(m["Vol_Wave"])},
                {"Metric": "Vol (Benchmark)", "Value": fmt_pct(m["Vol_Benchmark"])},
                {"Metric": "Tracking Error (TE)", "Value": fmt_pct(m["TE"])},
                {"Metric": "Information Ratio (IR)", "Value": fmt_num(m["IR"], 2)},
                {"Metric": "β_real", "Value": fmt_num(m["β_real"], 2)},
                {"Metric": "β_target", "Value": fmt_num(m["β_target"], 2)},
                {"Metric": "MaxDD (Wave)", "Value": fmt_pct(m["MaxDD_Wave"])},
                {"Metric": "MaxDD (Benchmark)", "Value": fmt_pct(m["MaxDD_Benchmark"])},
                {"Metric": "BM Difficulty (BM - SPY)", "Value": fmt_pct(m["Benchmark_Difficulty_BM_minus_SPY"])},
            ]
        )
        st.dataframe(mdf, use_container_width=True)

        if wd.get("flags"):
            st.warning(" | ".join(wd["flags"]))
        st.markdown("**Diagnosis**")
        for line in wd.get("diagnosis", []):
            st.write(f"- {line}")
        if wd.get("recommendations"):
            st.markdown("**Recommendations (shadow controls)**")
            for line in wd["recommendations"]:
                st.write(f"- {line}")

    st.markdown("---")

    st.subheader("🧪 What-If Lab (Shadow Simulation)")
    st.caption("This does NOT change engine math. It is a sandbox overlay simulation for diagnostics.")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        tilt_strength = st.slider("Tilt strength", 0.0, 1.0, 0.30, 0.05)
    with c2:
        vol_target = st.slider("Vol target (annual)", 0.05, 0.50, 0.20, 0.01)
    with c3:
        extra_safe = st.slider("Extra safe boost", 0.0, 0.40, 0.00, 0.01)
    with c4:
        freeze_bm = st.checkbox("Freeze benchmark (use engine BM)", value=True)

    c5, c6 = st.columns(2)
    with c5:
        exp_min = st.slider("Exposure min", 0.0, 1.5, 0.60, 0.05)
    with c6:
        exp_max = st.slider("Exposure max", 0.2, 2.0, 1.20, 0.05)

    if st.button("Run What-If Shadow Sim"):
        sim = simulate_whatif_nav(
            selected_wave,
            mode=mode,
            days=min(365, max(120, history_days)),
            tilt_strength=tilt_strength,
            vol_target=vol_target,
            extra_safe_boost=extra_safe,
            exp_min=exp_min,
            exp_max=exp_max,
            freeze_benchmark=freeze_bm,
        )
        if sim is None or sim.empty:
            st.warning("Simulation failed (insufficient prices).")
        else:
            nav = sim["whatif_nav"]
            bm_nav = sim["bm_nav"] if "bm_nav" in sim.columns else None
            ret_total = ret_from_nav(nav, len(nav))
            alpha_total = ret_total - (ret_from_nav(bm_nav, len(bm_nav)) if bm_nav is not None and len(bm_nav) > 1 else 0.0)

            st.markdown(f"**What-If Return:** {fmt_pct(ret_total)}   |   **What-If Alpha:** {fmt_pct(alpha_total)}")

            if go is not None:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=sim.index, y=sim["whatif_nav"], name="What-If NAV", mode="lines"))
                if "bm_nav" in sim.columns:
                    fig.add_trace(go.Scatter(x=sim.index, y=sim["bm_nav"], name="Benchmark NAV", mode="lines"))
                fig.update_layout(height=380, margin=dict(l=40, r=40, t=40, b=40))
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.line_chart(sim[["whatif_nav"] + (["bm_nav"] if "bm_nav" in sim.columns else [])])


# ============================================================
# TAB 2: Diagnostics++ (the heavy stuff)
# ============================================================
with tab_diag:
    st.subheader("🧠 Diagnostics++ (Proof, Drift, Rolling, Correlations, Coverage)")
    st.caption("These panels help verify mode separation, benchmark truth, beta discipline, rolling behavior, and data quality.")

    st.markdown("---")
    st.markdown("## 1) Mode Separation Proof — Side-by-Side (Selected Wave)")

    # Compute per-mode metrics + NAV overlay
    mode_rows: List[Dict[str, Any]] = []
    nav_bundle: Dict[str, pd.Series] = {}
    for m in all_modes:
        hist_m = compute_wave_history(selected_wave, mode=m, days=min(730, max(365, history_days)))
        if hist_m is None or hist_m.empty or len(hist_m) < 2:
            mode_rows.append(
                {"Mode": m, "365D Ret": np.nan, "365D Alpha": np.nan, "30D Alpha": np.nan, "TE": np.nan, "IR": np.nan, "β_real": np.nan}
            )
            continue

        nav_w = hist_m["wave_nav"]
        nav_b = hist_m["bm_nav"]
        r365w = ret_from_nav(nav_w, len(nav_w))
        r365b = ret_from_nav(nav_b, len(nav_b))
        a365 = r365w - r365b

        r30w = ret_from_nav(nav_w, min(30, len(nav_w)))
        r30b = ret_from_nav(nav_b, min(30, len(nav_b)))
        a30 = r30w - r30b

        te = tracking_error(hist_m["wave_ret"], hist_m["bm_ret"])
        ir = information_ratio(nav_w, nav_b, te)
        beta_r = beta_vs_benchmark(hist_m["wave_ret"], hist_m["bm_ret"])

        mode_rows.append(
            {"Mode": m, "365D Ret": r365w, "365D Alpha": a365, "30D Alpha": a30, "TE": te, "IR": ir, "β_real": beta_r, "β_target": get_beta_target_if_available(m)}
        )

        # Bundle NAV for plot (normalize to 100)
        try:
            s = pd.to_numeric(nav_w, errors="coerce").dropna()
            if s.shape[0] > 2:
                nav_bundle[m] = (s / float(s.iloc[0]) * 100.0).rename(m)
        except Exception:
            pass

    mode_df = pd.DataFrame(mode_rows)
    show_df(mode_df, selected_wave="", key="mode_sep_table")

    # NAV overlay
    if nav_bundle:
        st.markdown("### NAV Overlay Across Modes (Selected Wave)")
        nav_df = pd.concat(nav_bundle.values(), axis=1).dropna(how="all")
        if go is not None and not nav_df.empty:
            fig = go.Figure()
            for c in nav_df.columns:
                fig.add_trace(go.Scatter(x=nav_df.index, y=nav_df[c], name=c, mode="lines"))
            fig.update_layout(height=380, margin=dict(l=40, r=40, t=40, b=40), title="Selected Wave NAV (Indexed = 100) by Mode")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.line_chart(nav_df)

    st.markdown("---")
    st.markdown("## 2) Rolling Diagnostics (Selected Wave, Current Mode)")
    hist_sel = compute_wave_history(selected_wave, mode=mode, days=min(730, max(365, history_days)))
    roll = rolling_metrics(hist_sel, window=int(roll_win))
    plot_rolling_bundle(roll, title_prefix=f"{selected_wave} — {mode} (window={roll_win}d)")

    st.markdown("### Rolling Summary Stats (latest point)")
    if roll is not None and not roll.empty:
        last = roll.dropna().tail(1)
        if not last.empty:
            last_row = last.iloc[0].to_dict()
            latest_tbl = pd.DataFrame(
                [
                    {"Metric": "Rolling α (daily mean)", "Value": fmt_pct(last_row.get("roll_alpha", np.nan))},
                    {"Metric": "α Persistence (share days > 0)", "Value": fmt_pct(last_row.get("alpha_persist", np.nan))},
                    {"Metric": "Rolling TE (ann)", "Value": fmt_pct(last_row.get("roll_te", np.nan))},
                    {"Metric": "Rolling Vol (ann)", "Value": fmt_pct(last_row.get("roll_vol", np.nan))},
                    {"Metric": "Rolling β", "Value": fmt_num(last_row.get("roll_beta", np.nan), 2)},
                    {"Metric": "Rolling Corr vs BM", "Value": fmt_num(last_row.get("roll_corr", np.nan), 2)},
                ]
            )
            st.dataframe(latest_tbl, use_container_width=True)

    st.markdown("---")
    st.markdown("## 3) Beta Discipline & Drift Flags")
    beta_target = get_beta_target_if_available(mode)
    beta_real = beta_vs_benchmark(hist_sel["wave_ret"], hist_sel["bm_ret"]) if hist_sel is not None and not hist_sel.empty else float("nan")
    drift = abs(beta_real - beta_target) if (pd.notna(beta_real) and pd.notna(beta_target)) else float("nan")

    bd_tbl = pd.DataFrame(
        [
            {"Metric": "β_real (Wave vs BM)", "Value": fmt_num(beta_real, 2)},
            {"Metric": "β_target (if available)", "Value": fmt_num(beta_target, 2)},
            {"Metric": "|Drift|", "Value": fmt_num(drift, 2)},
            {"Metric": "Drift Threshold", "Value": fmt_num(beta_drift_warn, 2)},
        ]
    )
    st.dataframe(bd_tbl, use_container_width=True)

    if pd.notna(drift) and drift > beta_drift_warn:
        st.warning("Beta drift flag: |β_real − β_target| exceeds threshold. Check mode scaling + exposure series + benchmark mapping.")
    else:
        st.info("Beta discipline: no drift flag on current window (or beta target not provided).")

    st.markdown("---")
    st.markdown("## 4) Alpha Captured (Daily) — Exposure-Scaled if Engine Provides Exposure")
    if hist_sel is None or hist_sel.empty:
        st.info("No history available.")
    else:
        wret = pd.to_numeric(hist_sel["wave_ret"], errors="coerce").fillna(0.0)
        bret = pd.to_numeric(hist_sel["bm_ret"], errors="coerce").fillna(0.0)
        alpha_daily = (wret - bret).rename("alpha_daily")

        expo = get_exposure_series_if_available(hist_sel)
        if expo is not None:
            expo2 = pd.to_numeric(expo, errors="coerce").fillna(1.0)
            alpha_cap = (alpha_daily * expo2).rename("alpha_captured")
        else:
            alpha_cap = alpha_daily.rename("alpha_captured")

        ac_df = pd.DataFrame({"alpha_daily": alpha_daily, "alpha_captured": alpha_cap})
        ac_tail = ac_df.tail(min(180, max(60, int(history_days))))
        if go is not None and not ac_tail.empty:
            fig = go.Figure()
            fig.add_trace(go.Bar(x=ac_tail.index, y=ac_tail["alpha_captured"], name="Alpha Captured (daily)", opacity=0.8))
            fig.update_layout(height=300, margin=dict(l=40, r=40, t=40, b=30), title="Alpha Captured (daily)")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.line_chart(ac_tail)

        st.markdown("### Alpha Captured Quick Stats")
        ac_stats = pd.DataFrame(
            [
                {"Metric": "Mean (60D)", "Value": fmt_pct(float(alpha_cap.tail(60).mean())) if len(alpha_cap) >= 10 else "—"},
                {"Metric": "Mean (180D)", "Value": fmt_pct(float(alpha_cap.tail(180).mean())) if len(alpha_cap) >= 10 else "—"},
                {"Metric": "Hit Rate (α>0, 180D)", "Value": fmt_pct(float((alpha_daily.tail(180) > 0).mean())) if len(alpha_daily) >= 10 else "—"},
                {"Metric": "Worst Day (180D)", "Value": fmt_pct(float(alpha_cap.tail(180).min())) if len(alpha_cap) >= 10 else "—"},
                {"Metric": "Best Day (180D)", "Value": fmt_pct(float(alpha_cap.tail(180).max())) if len(alpha_cap) >= 10 else "—"},
            ]
        )
        st.dataframe(ac_stats, use_container_width=True)

    st.markdown("---")
    st.markdown("## 5) Benchmark Truth — Proof Pack (BM vs SPY + Drift)")
    attrib_now = compute_alpha_attribution(selected_wave, mode=mode, days=min(365, max(120, history_days)))
    if attrib_now:
        bt_tbl = pd.DataFrame(
            [
                {"Metric": "Benchmark Source", "Value": benchmark_source_label(selected_wave, get_benchmark_mix())},
                {"Metric": "Benchmark Return (window)", "Value": fmt_pct(attrib_now.get("Benchmark Return"))},
                {"Metric": "SPY Return (window)", "Value": fmt_pct(attrib_now.get("SPY Return"))},
                {"Metric": "BM Difficulty (BM - SPY)", "Value": fmt_pct(attrib_now.get("Benchmark Difficulty (BM - SPY)"))},
                {"Metric": "Alpha vs Benchmark", "Value": fmt_pct(attrib_now.get("Alpha vs Benchmark"))},
                {"Metric": "Alpha vs SPY", "Value": fmt_pct(attrib_now.get("Alpha vs SPY"))},
            ]
        )
        st.dataframe(bt_tbl, use_container_width=True)

    # Rolling corr vs BM (drift proxy)
    if roll is not None and not roll.empty and "roll_corr" in roll.columns:
        last_corr = float(roll["roll_corr"].dropna().iloc[-1]) if roll["roll_corr"].dropna().shape[0] else float("nan")
        st.write(f"**Latest rolling correlation (Wave vs BM):** {fmt_num(last_corr, 2)}")
        if pd.notna(last_corr) and last_corr < 0.50:
            st.warning("Low correlation vs benchmark in rolling window — check benchmark mapping or wave behavior drift.")
        elif pd.notna(last_corr):
            st.info("Correlation vs benchmark looks stable on the rolling window (drift proxy).")

    st.markdown("---")
    st.markdown("## 6) Data Quality / Coverage Audit (Selected Wave, Current Mode)")
    aud = coverage_audit(hist_sel)
    if not aud.get("ok", False):
        st.info(aud.get("message", "Audit unavailable."))
    else:
        a_tbl = pd.DataFrame(
            [
                {"Metric": "Rows", "Value": aud.get("rows")},
                {"Metric": "Start", "Value": aud.get("start")},
                {"Metric": "End", "Value": aud.get("end")},
                {"Metric": "NaNs wave_nav", "Value": aud.get("nan_wave_nav")},
                {"Metric": "NaNs bm_nav", "Value": aud.get("nan_bm_nav")},
                {"Metric": "NaNs wave_ret", "Value": aud.get("nan_wave_ret")},
                {"Metric": "NaNs bm_ret", "Value": aud.get("nan_bm_ret")},
                {"Metric": "Max gap days", "Value": aud.get("max_gap_days")},
                {"Metric": "% gaps >3d", "Value": fmt_pct(aud.get("pct_gap_gt3d"))},
            ]
        )
        st.dataframe(a_tbl, use_container_width=True)
        if aud.get("flags"):
            st.warning(" | ".join(aud["flags"]))
        else:
            st.info("No coverage flags detected on this window.")

    st.markdown("---")
    st.markdown("## 7) Correlation Matrix Across Waves (Current Mode)")
    ret_mat = build_wave_returns_matrix(all_waves, mode=mode, days=min(365, max(120, history_days)))
    if ret_mat is None or ret_mat.empty or ret_mat.shape[1] < 3:
        st.info("Not enough wave return series to compute correlations.")
    else:
        corr = corr_matrix(ret_mat.dropna(how="all"))
        corr_sorted = sort_corr_by_reference(corr, selected_wave)
        plot_corr_heatmap(corr_sorted, title=f"Wave Correlation Heatmap — Mode: {mode} (sorted by {selected_wave})")

        st.markdown("### Highest / Lowest Correlations vs Selected Wave")
        if selected_wave in corr.columns:
            s = corr[selected_wave].dropna().sort_values(ascending=False)
            top = s.head(6).reset_index()
            top.columns = ["Wave", "Corr"]
            bot = s.tail(6).reset_index()
            bot.columns = ["Wave", "Corr"]

            cA, cB = st.columns(2)
            with cA:
                st.markdown("**Most correlated**")
                top["Corr"] = top["Corr"].apply(lambda x: fmt_num(x, 2))
                st.dataframe(top, use_container_width=True)
            with cB:
                st.markdown("**Least correlated**")
                bot["Corr"] = bot["Corr"].apply(lambda x: fmt_num(x, 2))
                st.dataframe(bot, use_container_width=True)

        st.markdown("### Raw correlation matrix (table)")
        st.dataframe(corr_sorted.round(2), use_container_width=True)


# ============================================================
# TAB 3: Market Intel
# ============================================================
with tab_market:
    st.subheader("🌐 Market Intel")
    st.caption("Macro dashboard (daily).")

    mk = fetch_market_assets(days=min(365, max(120, history_days)))
    if mk is None or mk.empty:
        st.warning("Market data unavailable (yfinance missing or blocked).")
    else:
        rets = mk.pct_change().fillna(0.0)
        last = rets.tail(1).T.reset_index()
        last.columns = ["Asset", "1D Return"]
        last["1D Return"] = last["1D Return"].apply(lambda x: fmt_pct(x))
        st.dataframe(last, use_container_width=True)

        if go is not None:
            fig = go.Figure()
            for c in mk.columns:
                s = mk[c] / mk[c].iloc[0] * 100.0
                fig.add_trace(go.Scatter(x=mk.index, y=s, name=c, mode="lines"))
            fig.update_layout(height=420, margin=dict(l=40, r=40, t=40, b=40), title="Indexed Prices (Start=100)")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.line_chart(mk)


# ============================================================
# TAB 4: Factor Decomposition
# ============================================================
with tab_factors:
    st.subheader("🧩 Factor Decomposition (Simple Regression Betas)")
    st.caption("Uses SPY/QQQ/IWM/TLT/GLD daily returns as factor proxies. Display only.")

    hist = compute_wave_history(selected_wave, mode=mode, days=min(365, max(120, history_days)))
    if hist is None or hist.empty or "wave_ret" not in hist.columns:
        st.warning("Wave history unavailable.")
    else:
        factors_px = fetch_prices_daily(["SPY", "QQQ", "IWM", "TLT", "GLD"], days=min(365, max(120, history_days)))
        if factors_px is None or factors_px.empty:
            st.warning("Factor price data unavailable.")
        else:
            factor_ret = factors_px.pct_change().fillna(0.0)
            wave_ret = pd.to_numeric(hist["wave_ret"], errors="coerce").reindex(factor_ret.index).fillna(0.0)
            betas = regress_factors(wave_ret, factor_ret)

            bdf = pd.DataFrame([{"Factor": k, "Beta": v} for k, v in betas.items()])
            bdf["Beta"] = bdf["Beta"].apply(lambda x: fmt_num(x, 2))
            st.dataframe(bdf, use_container_width=True)


# ============================================================
# TAB 5: Vector OS Insight Layer
# ============================================================
with tab_vector:
    st.subheader("🤖 Vector OS Insight Layer")
    st.caption("Narrative interpretation (non-advice).")

    wd2 = wave_doctor_assess(selected_wave, mode=mode, days=365, alpha_warn=alpha_warn, te_warn=te_warn, beta_drift_warn=beta_drift_warn)
    attrib2 = compute_alpha_attribution(selected_wave, mode=mode, days=365)

    st.markdown("### Vector Summary")
    st.write(f"**Wave:** {selected_wave}  |  **Mode:** {mode}  |  **Regime:** {reg_now}  |  **Benchmark:** {bar_src}")

    if wd2.get("ok", False):
        m = wd2["metrics"]
        st.markdown(
            f"""
- **365D Return:** {fmt_pct(m["Return_365D"])}  
- **365D Alpha:** {fmt_pct(m["Alpha_365D"])}  
- **Alpha Captured (avg daily, 60D):** {fmt_pct(m["Alpha_Captured_avg_daily_60D"])}  
- **Tracking Error:** {fmt_pct(m["TE"])}  |  **IR:** {fmt_num(m["IR"], 2)}  
- **β:** {fmt_num(m["β_real"],2)} vs target {fmt_num(m["β_target"],2)}  
- **Max Drawdown:** {fmt_pct(m["MaxDD_Wave"])} (Wave) vs {fmt_pct(m["MaxDD_Benchmark"])} (BM)
"""
        )
        if wd2.get("flags"):
            st.warning("Flags: " + " | ".join(wd2["flags"]))

    st.markdown("### Attribution Lens")
    if attrib2:
        st.write(f"- **Engine Return:** {fmt_pct(attrib2.get('Engine Return'))}")
        st.write(f"- **Static Basket Return:** {fmt_pct(attrib2.get('Static Basket Return'))}")
        st.write(f"- **Overlay Contribution:** {fmt_pct(attrib2.get('Overlay Contribution (Engine - Static)'))}")
        st.write(f"- **Alpha vs Benchmark:** {fmt_pct(attrib2.get('Alpha vs Benchmark'))}")
        st.write(f"- **Benchmark Difficulty (BM - SPY):** {fmt_pct(attrib2.get('Benchmark Difficulty (BM - SPY)'))}")
        st.write(f"- **β_real:** {fmt_num(attrib2.get('β_real (Wave vs BM)'),2)} | **β_target:** {fmt_num(attrib2.get('β_target (if available)'),2)}")

    st.markdown("### Vector Guidance (Non-Advice)")
    guidance_lines = []
    if wd2.get("ok", False):
        m = wd2["metrics"]
        if pd.notna(m.get("Alpha_30D", np.nan)) and abs(float(m["Alpha_30D"])) > alpha_warn:
            guidance_lines.append("30D alpha is extreme: validate benchmark mix and recent data coverage first (Diagnostics++ → Coverage Audit + Benchmark Truth).")
        if pd.notna(m.get("β_target", np.nan)) and pd.notna(m.get("β_real", np.nan)):
            drift = abs(float(m["β_real"]) - float(m["β_target"]))
            if drift > beta_drift_warn:
                guidance_lines.append("Beta drift flagged: emphasize Mode Separation Proof and Rolling Beta in demos.")
        if pd.notna(m.get("TE", np.nan)) and float(m["TE"]) > te_warn:
            guidance_lines.append("Tracking error is elevated: highlight that this is active risk, and show Rolling TE + Vol for stability context.")
        if pd.notna(m.get("Alpha_Captured_avg_daily_60D", np.nan)):
            guidance_lines.append("Alpha Captured provides a daily edge lens: confirm persistence vs spikes using Rolling α + Persistence.")

    if not guidance_lines:
        guidance_lines.append("Vector suggests using the heatmap + overview grid to find waves with persistent alpha across multiple windows, then validating via Rolling diagnostics and Benchmark Truth proof pack.")

    for g in guidance_lines:
        st.write(f"- {g}")

    st.markdown("---")
    st.markdown("### Disclosures")
    st.write(
        "This console is informational and diagnostic only. It does not constitute investment advice. "
        "Shadow simulations (What-If Lab) do not affect the engine and are shown for analysis only."
    )