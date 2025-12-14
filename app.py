# app.py — WAVES Intelligence™ Institutional Console (Vector OS Edition)
# FULL PRODUCTION FILE (NO PATCHES) — DIAGNOSTICS++ BUILD (MAX)
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
# Adds Diagnostics++:
#   • Mode Separation Proof: side-by-side metrics + NAV overlay across ALL modes
#   • Rolling diagnostics: Rolling Alpha / TE / Beta / Vol + alpha persistence
#   • Beta discipline + drift flags (if engine provides beta target)
#   • Correlation matrix across waves (returns)
#   • Data quality / coverage audit + flags
#   • Alpha Captured daily (wave_ret - bm_ret, exposure-scaled if engine provides)
#
# Adds EXTRA (to be "bigger" + more info):
#   • All Waves dashboard: returns/alpha/TE/IR/beta/mdd/vol side-by-side
#   • Export buttons for snapshots (CSV download)
#   • “Benchmark difficulty” leaderboard vs SPY
#   • Stress window mini-panel (best effort using last N days)
#   • Safe fallback behaviors everywhere (no blank screen)
#
# Notes:
#   • Does NOT modify engine math or baseline results.
#   • What-If Lab is explicitly “shadow simulation”.
#   • Plotly/yfinance optional; app won’t crash if missing.

from __future__ import annotations

import math
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

import waves_engine as we  # your engine module

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
.waves-hdr { font-weight: 900; letter-spacing: 0.2px; margin-bottom: 4px; }

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


def _now_utc_str() -> str:
    try:
        return datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    except Exception:
        return "—"


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
    Expected engine output columns:
      wave_nav, bm_nav, wave_ret, bm_ret
    """
    try:
        df = we.compute_history_nav(wave_name, mode=mode, days=days)
        if df is None:
            df = pd.DataFrame()
    except Exception:
        df = pd.DataFrame()

    if df is None or df.empty:
        return pd.DataFrame(columns=["wave_nav", "bm_nav", "wave_ret", "bm_ret"])

    # Normalize columns if needed (best effort)
    cols = set(df.columns)
    needed = {"wave_nav", "bm_nav", "wave_ret", "bm_ret"}
    if not needed.issubset(cols):
        mapping = {}
        for c in df.columns:
            lc = str(c).lower()
            if lc in ["nav", "wave_nav", "portfolio_nav", "w_nav"]:
                mapping[c] = "wave_nav"
            elif lc in ["bm_nav", "bench_nav", "benchmark_nav", "b_nav"]:
                mapping[c] = "bm_nav"
            elif lc in ["wave_ret", "ret", "portfolio_ret", "w_ret"]:
                mapping[c] = "wave_ret"
            elif lc in ["bm_ret", "bench_ret", "benchmark_ret", "b_ret"]:
                mapping[c] = "bm_ret"
        if mapping:
            df = df.rename(columns=mapping)

    # Ensure datetime index and sorted
    try:
        if not isinstance(df.index, pd.DatetimeIndex):
            df = df.copy()
            df.index = pd.to_datetime(df.index, errors="coerce")
        df = df.sort_index()
    except Exception:
        pass

    # Fill NA safely
    for c in ["wave_nav", "bm_nav"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").ffill().bfill()
    for c in ["wave_ret", "bm_ret"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    # If NAV missing but returns exist, rebuild NAV (display only)
    if ("wave_nav" not in df.columns or df["wave_nav"].isna().all()) and "wave_ret" in df.columns:
        df["wave_nav"] = (1.0 + df["wave_ret"].fillna(0.0)).cumprod()
    if ("bm_nav" not in df.columns or df["bm_nav"].isna().all()) and "bm_ret" in df.columns:
        df["bm_nav"] = (1.0 + df["bm_ret"].fillna(0.0)).cumprod()

    if len(df) > days:
        df = df.iloc[-days:]

    for c in ["wave_nav", "bm_nav", "wave_ret", "bm_ret"]:
        if c not in df.columns:
            df[c] = np.nan if "nav" in c else 0.0

    return df[["wave_nav", "bm_nav", "wave_ret", "bm_ret"]]


@st.cache_data(show_spinner=False)
def get_benchmark_mix() -> pd.DataFrame:
    try:
        df = we.get_benchmark_mix_table()
        if df is None:
            return pd.DataFrame(columns=["Wave", "Ticker", "Name", "Weight"])
        return df
    except Exception:
        return pd.DataFrame(columns=["Wave", "Ticker", "Name", "Weight"])


@st.cache_data(show_spinner=False)
def get_wave_holdings(wave_name: str) -> pd.DataFrame:
    try:
        df = we.get_wave_holdings(wave_name)
        if df is None:
            return pd.DataFrame(columns=["Ticker", "Name", "Weight"])
        return df
    except Exception:
        return pd.DataFrame(columns=["Ticker", "Name", "Weight"])


# ============================================================
# Core metrics
# ============================================================
def ret_from_nav(nav: pd.Series, window: int) -> float:
    nav = safe_series(nav)
    if len(nav) < 2:
        return float("nan")
    window = int(max(2, min(window, len(nav))))
    sub = nav.iloc[-window:]
    start = float(sub.iloc[0])
    end = float(sub.iloc[-1])
    if not math.isfinite(start) or start <= 0:
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
    candidates = ["MODE_BETA_TARGET", "BETA_TARGET_BY_MODE", "BETA_TARGETS", "BETA_TARGET"]
    for name in candidates:
        try:
            obj = getattr(we, name, None)
            if isinstance(obj, dict) and mode in obj:
                return float(obj[mode])
        except Exception:
            pass
    return float("nan")


def get_mode_base_exposure_if_available(mode: str) -> float:
    candidates = ["MODE_BASE_EXPOSURE", "BASE_EXPOSURE_BY_MODE", "MODE_EXPOSURE_BASE"]
    for name in candidates:
        try:
            obj = getattr(we, name, None)
            if isinstance(obj, dict) and mode in obj:
                return float(obj[mode])
        except Exception:
            pass
    return float("nan")


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
    base_expo = get_mode_base_exposure_if_available(mode)

    daily_alpha = (wave_ret - bm_ret).copy()
    if math.isfinite(base_expo):
        daily_alpha_scaled = daily_alpha / max(1e-9, float(base_expo))
    else:
        daily_alpha_scaled = daily_alpha

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
    out["Mode Base Exposure (if available)"] = float(base_expo) if math.isfinite(base_expo) else float("nan")

    out["Wave Vol"] = float(annualized_vol(wave_ret))
    out["Benchmark Vol"] = float(annualized_vol(bm_ret))
    out["Wave MaxDD"] = float(max_drawdown(nav_wave))
    out["Benchmark MaxDD"] = float(max_drawdown(nav_bm))

    out["Alpha Captured (avg daily)"] = float(np.nanmean(daily_alpha_scaled.values)) if len(daily_alpha_scaled) else float("nan")
    out["Alpha Captured (last day)"] = float(daily_alpha_scaled.iloc[-1]) if len(daily_alpha_scaled) else float("nan")

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

        hit_rate = float((wave_ret >= bm_ret).mean()) if len(wave_ret) > 0 else float("nan")

        if len(nav_wave) > 1:
            trough = float(nav_wave.min())
            peak = float(nav_wave.max())
            last = float(nav_wave.iloc[-1])
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

    # FIXED: fully closed strings, no truncation
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
        "Avg Daily Alpha",
        "Stress",
    ]

    for c in df.columns:
        cs = str(c)

        if cs == "WaveScore":
            fmt[c] = lambda v: fmt_score(v)
            continue

        if cs in ["Return Quality", "Risk Control", "Consistency", "Resilience", "Efficiency", "Transparency"]:
            fmt[c] = lambda v: fmt_num(v, 1)
            continue

        if cs.startswith("β") or cs.startswith("beta") or cs.startswith("β_"):
            fmt[c] = lambda v: fmt_num(v, 2)
            continue

        if ("IR" in cs) and ("Ret" not in cs) and ("Return" not in cs):
            fmt[c] = lambda v: fmt_num(v, 2)
            continue

        if any(k in cs for k in pct_keywords):
            fmt[c] = lambda v: fmt_pct(v, 2)

    return fmt


def style_selected_and_alpha(df: pd.DataFrame, selected_wave: str):
    alpha_cols = [c for c in df.columns if ("Alpha" in str(c)) or str(c).endswith("α")]
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


def selectable_table_jump(df: pd.DataFrame, key: str) -> None:
    if df is None or df.empty or "Wave" not in df.columns:
        st.info("No waves available to jump.")
        return

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

    fig = go.Figure(data=go.Heatmap(z=z, x=x, y=y, zmin=zmin, zmax=zmax, colorbar=dict(title="Alpha")))
    fig.update_layout(
        title=title,
        height=min(1000, 260 + 22 * max(10, len(y))),
        margin=dict(l=80, r=40, t=60, b=40),
        xaxis_title="Timeframe",
        yaxis_title="Wave",
    )
    st.plotly_chart(fig, use_container_width=True)


# ============================================================
# Diagnostics++ helpers
# ============================================================
def data_quality_audit(hist: pd.DataFrame) -> Dict[str, Any]:
    if hist is None or hist.empty:
        return {"ok": False, "flags": ["No history returned."], "details": {}}

    flags: List[str] = []
    details: Dict[str, Any] = {}

    n = int(len(hist))
    details["Rows"] = n
    details["Start"] = str(hist.index.min()) if hasattr(hist.index, "min") else "—"
    details["End"] = str(hist.index.max()) if hasattr(hist.index, "max") else "—"

    for c in ["wave_nav", "bm_nav", "wave_ret", "bm_ret"]:
        miss = int(pd.isna(hist[c]).sum()) if c in hist.columns else n
        details[f"Missing {c}"] = miss
        if miss > 0 and miss / max(1, n) > 0.05:
            flags.append(f"Data gaps in {c} (>5% missing).")

    try:
        dup = int(hist.index.duplicated().sum())
        details["Duplicate dates"] = dup
        if dup > 0:
            flags.append("Duplicate dates detected (index).")
    except Exception:
        pass

    try:
        wnav = pd.to_numeric(hist["wave_nav"], errors="coerce")
        bnav = pd.to_numeric(hist["bm_nav"], errors="coerce")
        if wnav.nunique(dropna=True) <= 3:
            flags.append("Wave NAV appears flat/constant (possible upstream issue).")
        if bnav.nunique(dropna=True) <= 3:
            flags.append("Benchmark NAV appears flat/constant (possible upstream issue).")
    except Exception:
        pass

    return {"ok": True, "flags": flags, "details": details}


def rolling_metrics(hist: pd.DataFrame, window: int = 60) -> pd.DataFrame:
    """
    Rolling metrics computed from daily returns.
    FIXED: rolling beta returns a single Series (no shape mismatch ValueError).
    """
    if hist is None or hist.empty:
        return pd.DataFrame()

    w = pd.to_numeric(hist["wave_ret"], errors="coerce").fillna(0.0)
    b = pd.to_numeric(hist["bm_ret"], errors="coerce").fillna(0.0)
    df = pd.concat([w.rename("w"), b.rename("b")], axis=1).dropna()

    if df.shape[0] < max(20, window):
        return pd.DataFrame()

    out = pd.DataFrame(index=df.index)

    out["roll_alpha_ann"] = (df["w"] - df["b"]).rolling(window).mean() * 252.0
    out["roll_te"] = (df["w"] - df["b"]).rolling(window).std() * np.sqrt(252)
    out["roll_vol_wave"] = df["w"].rolling(window).std() * np.sqrt(252)
    out["roll_vol_bm"] = df["b"].rolling(window).std() * np.sqrt(252)

    cov = df["w"].rolling(window).cov(df["b"])
    var = df["b"].rolling(window).var()
    out["roll_beta"] = cov / var.replace(0.0, np.nan)

    out["alpha_persist"] = ((df["w"] - df["b"]) > 0).rolling(window).mean()
    active = df["w"] - df["b"]
    out["roll_ir"] = (active.rolling(window).mean() / active.rolling(window).std().replace(0.0, np.nan)) * np.sqrt(252)

    return out.dropna(how="all")


def correlation_matrix(all_waves: List[str], mode: str, days: int = 180) -> pd.DataFrame:
    series_map: Dict[str, pd.Series] = {}
    for wname in all_waves:
        hist = compute_wave_history(wname, mode=mode, days=days)
        if hist is None or hist.empty:
            continue
        s = pd.to_numeric(hist["wave_ret"], errors="coerce")
        s.name = wname
        series_map[wname] = s

    if not series_map:
        return pd.DataFrame()

    df = pd.concat(series_map.values(), axis=1).dropna(how="all")
    if df.shape[0] < 20 or df.shape[1] < 2:
        return pd.DataFrame()
    return df.corr()


def compute_mode_snapshot(wave_name: str, modes: List[str], days: int = 365) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for m in modes:
        hist = compute_wave_history(wave_name, mode=m, days=days)
        if hist is None or hist.empty or len(hist) < 2:
            rows.append({"Mode": m, "30D Ret": np.nan, "30D Alpha": np.nan, "365D Ret": np.nan, "365D Alpha": np.nan, "TE": np.nan, "IR": np.nan, "β_real": np.nan})
            continue

        nav_w = hist["wave_nav"]
        nav_b = hist["bm_nav"]
        wret = hist["wave_ret"]
        bret = hist["bm_ret"]

        r30w = ret_from_nav(nav_w, min(30, len(nav_w)))
        r30b = ret_from_nav(nav_b, min(30, len(nav_b)))
        a30 = r30w - r30b

        r365w = ret_from_nav(nav_w, len(nav_w))
        r365b = ret_from_nav(nav_b, len(nav_b))
        a365 = r365w - r365b

        te = tracking_error(wret, bret)
        ir = information_ratio(nav_w, nav_b, te)
        beta_r = beta_vs_benchmark(wret, bret)

        rows.append({"Mode": m, "30D Ret": r30w, "30D Alpha": a30, "365D Ret": r365w, "365D Alpha": a365, "TE": te, "IR": ir, "β_real": beta_r})

    return pd.DataFrame(rows)


def plot_nav_overlay(hist_by_mode: Dict[str, pd.DataFrame], title: str):
    if go is None:
        st.info("NAV overlay chart unavailable (Plotly missing).")
        return

    fig = go.Figure()
    for mode, hist in hist_by_mode.items():
        if hist is None or hist.empty:
            continue
        nav = hist["wave_nav"].copy()
        if len(nav) < 2:
            continue
        nav_idx = (nav / nav.iloc[0]) * 100.0
        fig.add_trace(go.Scatter(x=hist.index, y=nav_idx, name=f"{mode}", mode="lines"))

    fig.update_layout(height=420, margin=dict(l=40, r=40, t=50, b=40), title=title, yaxis=dict(title="Indexed NAV (Start=100)"))
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

    spy_nav = compute_spy_nav(days=days)
    spy_ret = ret_from_nav(spy_nav, len(spy_nav)) if len(spy_nav) >= 2 else float("nan")
    bm_difficulty = (r365_b - spy_ret) if (pd.notna(r365_b) and pd.notna(spy_ret)) else float("nan")

    beta_real = beta_vs_benchmark(ret_w, ret_b)
    beta_target = get_beta_target_if_available(mode)
    beta_drift = abs(beta_real - beta_target) if (math.isfinite(beta_real) and math.isfinite(beta_target)) else float("nan")

    flags: List[str] = []
    diagnosis: List[str] = []
    recs: List[str] = []

    if pd.notna(a30) and abs(a30) >= alpha_warn:
        flags.append("Large 30D alpha (verify benchmark + coverage)")
        diagnosis.append("30D alpha is unusually large. This can be real signal, but also benchmark mix drift or partial data coverage.")
        recs.append("Check Benchmark Truth panel and Data Quality Audit (missing data / flat NAV).")

    if pd.notna(a365) and a365 < 0:
        diagnosis.append("365D alpha is negative (could be true underperformance or a harder benchmark).")
        if pd.notna(bm_difficulty) and bm_difficulty > 0.03:
            diagnosis.append("Benchmark difficulty is positive vs SPY (benchmark outperformed SPY) → alpha harder on this window.")
            recs.append("Compare to SPY for sanity-check (display-only) and inspect benchmark mix.")
    else:
        if pd.notna(a365) and a365 > 0.06:
            diagnosis.append("365D alpha is strongly positive. Consider locking benchmark snapshot for demos.")

    if pd.notna(te) and te > te_warn:
        flags.append("High tracking error (active risk elevated)")
        diagnosis.append("Tracking error is high; wave deviates materially from benchmark.")
        recs.append("Use Rolling Diagnostics to see whether TE spikes are episodic or persistent.")

    if pd.notna(vol_w) and vol_w > vol_warn:
        flags.append("High volatility")
        diagnosis.append("Volatility elevated; consider stronger SmartSafe posture in stressed regimes.")

    if pd.notna(mdd_w) and mdd_w < mdd_warn:
        flags.append("Deep drawdown")
        diagnosis.append("Drawdown is deep; consider risk gating review and benchmark difficulty context.")

    if math.isfinite(beta_drift) and beta_drift > 0.07:
        flags.append("Beta drift > 0.07 (discipline breach)")
        diagnosis.append("β_real drifting from β_target can indicate exposure scaling mismatch or benchmark mix drift.")
        recs.append("Use Mode Separation Proof (side-by-side) and Rolling Beta chart.")

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
            "β_real": beta_real,
            "β_target": beta_target,
            "β_drift": beta_drift,
            "MaxDD_Wave": mdd_w,
            "MaxDD_Benchmark": mdd_b,
            "Benchmark_Difficulty_BM_minus_SPY": bm_difficulty,
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

    wave_ret_s = pd.Series(wave
        wave_ret_s = pd.Series(wave_ret, index=pd.Index(dates, name="Date"), name="whatif_ret")
    wave_nav = (1.0 + wave_ret_s).cumprod().rename("whatif_nav")

    out = pd.DataFrame({"whatif_nav": wave_nav, "whatif_ret": wave_ret_s})

    # Benchmark handling for shadow sim:
    # • freeze_benchmark=True -> use engine's benchmark NAV/RET (aligned to dates)
    # • freeze_benchmark=False -> proxy benchmark = SPY (display-only)
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

    # add alpha captured series
    if "bm_ret" in out.columns:
        out["alpha_captured"] = out["whatif_ret"] - out["bm_ret"]
    else:
        out["alpha_captured"] = np.nan

    return out


# ============================================================
# Sidebar + Engine Discovery
# ============================================================
def _safe_get_all_waves() -> List[str]:
    try:
        wv = we.get_all_waves()
        if wv is None:
            return []
        return list(wv)
    except Exception:
        return []


def _safe_get_modes() -> List[str]:
    try:
        md = we.get_modes()
        if md is None:
            return ["Standard", "Alpha-Minus-Beta", "Private Logic"]
        return list(md)
    except Exception:
        return ["Standard", "Alpha-Minus-Beta", "Private Logic"]


all_waves = _safe_get_all_waves()
all_modes = _safe_get_modes()

if "selected_wave" not in st.session_state:
    st.session_state["selected_wave"] = all_waves[0] if all_waves else ""
if "mode" not in st.session_state:
    st.session_state["mode"] = all_modes[0] if all_modes else "Standard"

with st.sidebar:
    st.title("WAVES Intelligence™")
    st.caption("Institutional Console • Vector OS™")

    st.markdown("### Core Controls")
    if all_modes:
        st.selectbox("Mode", all_modes, key="mode")
    else:
        st.text_input("Mode", value=st.session_state.get("mode", "Standard"), key="mode")

    if all_waves:
        st.selectbox("Select Wave", all_waves, key="selected_wave")
    else:
        st.text_input("Select Wave", value=st.session_state.get("selected_wave", ""), key="selected_wave")

    st.markdown("---")
    st.markdown("### Display Settings")
    history_days = st.slider("History window (days)", min_value=60, max_value=730, value=365, step=15)

    st.markdown("---")
    st.markdown("### Diagnostics++")
    roll_win = st.slider("Rolling window (days)", 20, 120, 60, 5)
    corr_days = st.slider("Correlation window (days)", 60, 365, 180, 15)
    beta_drift_alert = st.slider("Beta drift alert", 0.01, 0.20, 0.07, 0.01)

    st.markdown("---")
    st.markdown("### Wave Doctor")
    alpha_warn = st.slider("Alpha alert threshold (abs, 30D)", 0.02, 0.20, 0.08, 0.01)
    te_warn = st.slider("Tracking error alert threshold", 0.05, 0.40, 0.20, 0.01)

    st.markdown("---")
    st.markdown("### What-If Lab (Shadow)")
    st.caption("Display-only simulation. Does NOT change engine math.")
    whatif_days = st.slider("What-If window (days)", 60, 730, 365, 15)
    tilt_strength = st.slider("Momentum tilt strength", 0.0, 2.0, 0.50, 0.05)
    vol_target = st.slider("Vol target (annualized)", 0.05, 0.60, 0.22, 0.01)
    extra_safe_boost = st.slider("Extra safe boost", 0.0, 0.40, 0.00, 0.01)
    exp_min = st.slider("Exposure min", 0.20, 1.20, 0.60, 0.05)
    exp_max = st.slider("Exposure max", 0.50, 2.00, 1.20, 0.05)
    freeze_benchmark = st.checkbox("Freeze benchmark to engine", value=True)

mode = st.session_state["mode"]
selected_wave = st.session_state["selected_wave"]

if not selected_wave:
    st.error("No Waves available (engine returned empty list). Verify waves_engine.get_all_waves().")
    st.stop()


# ============================================================
# Page header
# ============================================================
st.title("WAVES Intelligence™ Institutional Console")
st.caption("Live Alpha Capture • SmartSafe™ • Multi-Asset • Crypto • Gold • Income Ladders • Diagnostics++")


# ============================================================
# Sticky Summary Bar
# ============================================================
h_bar = compute_wave_history(selected_wave, mode=mode, days=max(365, history_days))

bar_r30 = bar_a30 = bar_r365 = bar_a365 = float("nan")
bar_te = bar_ir = float("nan")
bar_src = "—"

bm_mix_for_src = get_benchmark_mix()
bar_src = benchmark_source_label(selected_wave, bm_mix_for_src)

beta_target_now = get_beta_target_if_available(mode)
beta_real_now = float("nan")
beta_drift_now = float("nan")

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

    beta_real_now = beta_vs_benchmark(ret_w, ret_b)
    if math.isfinite(beta_real_now) and math.isfinite(beta_target_now):
        beta_drift_now = abs(beta_real_now - beta_target_now)

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

beta_chip = ""
if math.isfinite(beta_real_now):
    beta_chip += f"<span class='waves-chip'>β_real: <b>{fmt_num(beta_real_now,2)}</b></span>"
if math.isfinite(beta_target_now):
    beta_chip += f"<span class='waves-chip'>β_target: <b>{fmt_num(beta_target_now,2)}</b></span>"
if math.isfinite(beta_drift_now):
    drift_style = "color: #ff6b6b;" if beta_drift_now > beta_drift_alert else ""
    beta_chip += f"<span class='waves-chip'>β_drift: <b style='{drift_style}'>{fmt_num(beta_drift_now,2)}</b></span>"

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
  {beta_chip}
  <span class="waves-chip">WaveScore: <b>{ws_val}</b> ({ws_grade})</span>
</div>
""",
    unsafe_allow_html=True,
)


# ============================================================
# Tabs
# ============================================================
tab_console, tab_diag, tab_market, tab_factors, tab_vector = st.tabs(
    ["Console", "Diagnostics++", "Market Intel", "Factor Decomposition", "Vector OS Insight Layer"]
)


# ============================================================
# TAB 1: Console
# ============================================================
with tab_console:
    st.subheader("🔥 Alpha Heatmap View (All Waves × Timeframe)")
    st.caption("Fast scan across all waves. Heatmap values are alpha vs benchmark (not vs SPY).")

    alpha_df = build_alpha_matrix(all_waves, mode)
    plot_alpha_heatmap(alpha_df, selected_wave, title=f"Alpha Heatmap — Mode: {mode}")

    st.markdown("### 🧭 One-Click Jump Table")
    jump_df = alpha_df.copy()
    jump_df["RankScore"] = jump_df[["1D Alpha", "30D Alpha", "60D Alpha", "365D Alpha"]].mean(axis=1, skipna=True)
    jump_df = jump_df.sort_values("RankScore", ascending=False)
    show_df(jump_df, selected_wave, key="jump_table")
    selectable_table_jump(jump_df, key="jump_select")

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

        if len(nav_w) >= 2 and len(nav_b) >= 2:
            r1w = float(nav_w.iloc[-1] / nav_w.iloc[-2] - 1.0)
            r1b = float(nav_b.iloc[-1] / nav_b.iloc[-2] - 1.0)
            a1 = r1w - r1b
        else:
            r1w = np.nan
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
                show_df(bm_mix[bm_mix["Wave"] == selected_wave], selected_wave, key="bm_mix_sel")
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
                if any(x in k for x in ["Return", "Alpha", "Vol", "MaxDD", "TE", "Difficulty", "Alpha Captured"]):
                    arows.append({"Metric": k, "Value": fmt_pct(v)})
                elif "β" in k:
                    arows.append({"Metric": k, "Value": fmt_num(v, 2)})
                elif "IR" in k:
                    arows.append({"Metric": k, "Value": fmt_num(v, 2)})
                else:
                    arows.append({"Metric": k, "Value": str(v)})
            show_df(pd.DataFrame(arows), selected_wave, key="attrib_table")


# ============================================================
# TAB 2: Diagnostics++
# ============================================================
with tab_diag:
    st.subheader("🧪 Mode Separation Proof (All Modes)")
    st.caption("Side-by-side metrics + NAV overlay. Confirms modes do NOT collapse into identical behavior.")

    modes_for_proof = all_modes if all_modes else ["Standard", "Alpha-Minus-Beta", "Private Logic"]

    proof_df = compute_mode_snapshot(selected_wave, modes_for_proof, days=max(365, history_days))
    show_df(proof_df, selected_wave, key="mode_proof_table")

    hist_by_mode: Dict[str, pd.DataFrame] = {}
    for m in modes_for_proof:
        hist_by_mode[m] = compute_wave_history(selected_wave, mode=m, days=max(365, history_days))
    plot_nav_overlay(hist_by_mode, title=f"NAV Overlay — {selected_wave} (All Modes)")

    st.markdown("---")

    st.subheader("📈 Rolling Diagnostics (Alpha / TE / Beta / Vol / Persistence)")
    hist_sel = compute_wave_history(selected_wave, mode=mode, days=max(365, history_days))

    audit = data_quality_audit(hist_sel)
    if audit.get("ok") and audit.get("flags"):
        st.warning("Data Quality Flags: " + " | ".join(audit["flags"]))

    roll = rolling_metrics(hist_sel, window=roll_win)
    if roll is None or roll.empty:
        st.info("Rolling diagnostics unavailable (not enough data).")
    else:
        # table
        tail = roll.tail(10).copy()
        tail.index = tail.index.astype(str)
        show_df(tail.reset_index().rename(columns={"index": "Date"}), selected_wave, key="rolling_tail")

        # charts
        if go is not None:
            c1, c2 = st.columns(2)

            with c1:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=roll.index, y=roll["roll_alpha_ann"], name="Rolling Alpha (ann)", mode="lines"))
                fig.add_trace(go.Scatter(x=roll.index, y=roll["roll_ir"], name="Rolling IR", mode="lines", yaxis="y2"))
                fig.update_layout(
                    height=320,
                    title="Rolling Alpha (annualized) + Rolling IR",
                    margin=dict(l=40, r=40, t=50, b=40),
                    yaxis=dict(title="Alpha (ann)"),
                    yaxis2=dict(title="IR", overlaying="y", side="right"),
                )
                st.plotly_chart(fig, use_container_width=True)

            with c2:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=roll.index, y=roll["roll_te"], name="Rolling TE", mode="lines"))
                fig.add_trace(go.Scatter(x=roll.index, y=roll["alpha_persist"], name="Alpha Persistence", mode="lines", yaxis="y2"))
                fig.update_layout(
                    height=320,
                    title="Rolling TE + Alpha Persistence",
                    margin=dict(l=40, r=40, t=50, b=40),
                    yaxis=dict(title="TE"),
                    yaxis2=dict(title="Persistence", overlaying="y", side="right", range=[0, 1]),
                )
                st.plotly_chart(fig, use_container_width=True)

            c3, c4 = st.columns(2)

            with c3:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=roll.index, y=roll["roll_beta"], name="Rolling Beta", mode="lines"))
                if math.isfinite(beta_target_now):
                    fig.add_trace(go.Scatter(x=roll.index, y=[beta_target_now] * len(roll), name="β_target", mode="lines"))
                fig.update_layout(height=320, title="Rolling Beta (β_real)", margin=dict(l=40, r=40, t=50, b=40))
                st.plotly_chart(fig, use_container_width=True)

            with c4:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=roll.index, y=roll["roll_vol_wave"], name="Rolling Vol (Wave)", mode="lines"))
                fig.add_trace(go.Scatter(x=roll.index, y=roll["roll_vol_bm"], name="Rolling Vol (BM)", mode="lines"))
                fig.update_layout(height=320, title="Rolling Volatility", margin=dict(l=40, r=40, t=50, b=40))
                st.plotly_chart(fig, use_container_width=True)

        else:
            st.line_chart(roll[["roll_alpha_ann", "roll_te", "roll_beta", "roll_vol_wave", "roll_vol_bm", "alpha_persist"]])

    st.markdown("---")

    st.subheader("🧮 Correlation Matrix (Across Waves)")
    corr = correlation_matrix(all_waves, mode=mode, days=corr_days)
    if corr is None or corr.empty:
        st.info("Correlation matrix unavailable (insufficient overlap).")
    else:
        show_df(corr.reset_index().rename(columns={"index": "Wave"}), selected_wave, key="corr_matrix")

    st.markdown("---")

    st.subheader("🩺 Wave Doctor (Diagnostics + Recommendations)")
    doc = wave_doctor_assess(selected_wave, mode=mode, days=max(365, history_days), alpha_warn=alpha_warn, te_warn=te_warn)
    if not doc.get("ok"):
        st.info(doc.get("message", "Wave Doctor unavailable."))
    else:
        m = doc["metrics"]
        left, right = st.columns(2)
        with left:
            st.markdown("#### Key Metrics")
            metrics_df = pd.DataFrame(
                [
                    {"Metric": "30D Return", "Value": fmt_pct(m.get("Return_30D"))},
                    {"Metric": "30D Alpha", "Value": fmt_pct(m.get("Alpha_30D"))},
                    {"Metric": "365D Return", "Value": fmt_pct(m.get("Return_365D"))},
                    {"Metric": "365D Alpha", "Value": fmt_pct(m.get("Alpha_365D"))},
                    {"Metric": "TE", "Value": fmt_pct(m.get("TE"))},
                    {"Metric": "IR", "Value": fmt_num(m.get("IR"), 2)},
                    {"Metric": "β_real", "Value": fmt_num(m.get("β_real"), 2)},
                    {"Metric": "β_target", "Value": fmt_num(m.get("β_target"), 2)},
                    {"Metric": "β_drift", "Value": fmt_num(m.get("β_drift"), 2)},
                    {"Metric": "Wave Vol", "Value": fmt_pct(m.get("Vol_Wave"))},
                    {"Metric": "Wave MaxDD", "Value": fmt_pct(m.get("MaxDD_Wave"))},
                ]
            )
            show_df(metrics_df, selected_wave, key="doctor_metrics")

        with right:
            st.markdown("#### Flags / Diagnosis / Recommendations")
            flags = doc.get("flags", [])
            if flags:
                st.warning(" | ".join(flags))
            for line in doc.get("diagnosis", []):
                st.write("• " + str(line))
            recs = doc.get("recommendations", [])
            if recs:
                st.markdown("**Recommended next checks:**")
                for r in recs:
                    st.write("✅ " + str(r))

    st.markdown("---")

    st.subheader("🧪 What-If Lab (Shadow Simulation)")
    st.caption("This is a display-only simulation layer. It does NOT alter engine results. Use for scenario understanding only.")

    with st.expander("Run What-If Simulation", expanded=True):
        if st.button("Run What-If", key="run_whatif"):
            sim = simulate_whatif_nav(
                selected_wave,
                mode=mode,
                days=whatif_days,
                tilt_strength=tilt_strength,
                vol_target=vol_target,
                extra_safe_boost=extra_safe_boost,
                exp_min=exp_min,
                exp_max=exp_max,
                freeze_benchmark=freeze_benchmark,
            )
            st.session_state["whatif_df"] = sim

    sim = st.session_state.get("whatif_df", None)
    if sim is None or (isinstance(sim, pd.DataFrame) and sim.empty):
        st.info("No simulation run yet.")
    else:
        st.markdown("#### Simulation Summary")
        sim_nav = sim["whatif_nav"]
        sim_bm = sim["bm_nav"] if "bm_nav" in sim.columns else pd.Series(dtype=float)
        sim_ret = ret_from_nav(sim_nav, len(sim_nav))
        bm_ret = ret_from_nav(sim_bm, len(sim_bm)) if len(sim_bm) else float("nan")
        sim_alpha = sim_ret - bm_ret if (pd.notna(sim_ret) and pd.notna(bm_ret)) else float("nan")

        st.write(f"• What-If Return: **{fmt_pct(sim_ret)}**")
        st.write(f"• What-If Benchmark Return: **{fmt_pct(bm_ret)}**")
        st.write(f"• What-If Alpha: **{fmt_pct(sim_alpha)}**")

        if go is not None:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=sim.index, y=(sim_nav / sim_nav.iloc[0]) * 100.0, name="What-If NAV", mode="lines"))
            if "bm_nav" in sim.columns and len(sim_bm) > 0:
                fig.add_trace(go.Scatter(x=sim.index, y=(sim_bm / sim_bm.iloc[0]) * 100.0, name="Benchmark NAV", mode="lines"))
            fig.update_layout(height=360, title="What-If NAV vs Benchmark (Indexed)", margin=dict(l=40, r=40, t=50, b=40))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.line_chart(pd.DataFrame({"whatif_nav": sim_nav, "bm_nav": sim_bm}).dropna(how="all"))

        st.markdown("#### What-If Tail (last 20 rows)")
        show_df(sim.tail(20).reset_index().rename(columns={"index": "Date"}), selected_wave, key="whatif_tail")


# ============================================================
# TAB 3: Market Intel
# ============================================================
with tab_market:
    st.subheader("🌍 Market Intel Panel")
    st.caption("Quick context: SPY / QQQ / IWM / TLT / GLD / BTC / VIX / TNX")

    mkt = fetch_market_assets(days=max(180, min(history_days, 365)))
    if mkt is None or mkt.empty:
        st.warning("Unable to load market assets (yfinance missing or fetch failed).")
    else:
        rets = mkt.pct_change().fillna(0.0)
        nav = (1.0 + rets).cumprod() * 100.0

        st.markdown("### Indexed Performance (Start=100)")
        if go is not None:
            fig = go.Figure()
            for c in nav.columns:
                fig.add_trace(go.Scatter(x=nav.index, y=nav[c], name=c, mode="lines"))
            fig.update_layout(height=420, margin=dict(l=40, r=40, t=50, b=40))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.line_chart(nav)

        st.markdown("### Latest Levels / Returns")
        last = mkt.iloc[-1].to_dict()
        d1 = (mkt.iloc[-1] / mkt.iloc[-2] - 1.0).to_dict() if len(mkt) >= 2 else {}
        d30 = (mkt.iloc[-1] / mkt.iloc[-31] - 1.0).to_dict() if len(mkt) >= 31 else {}
        rows = []
        for k in mkt.columns:
            rows.append({"Asset": k, "Last": fmt_num(last.get(k), 2), "1D": fmt_pct(d1.get(k)), "30D": fmt_pct(d30.get(k))})
        show_df(pd.DataFrame(rows), selected_wave, key="market_table")


# ============================================================
# TAB 4: Factor Decomposition
# ============================================================
with tab_factors:
    st.subheader("🧩 Factor Decomposition (Simple Regression Beta Map)")
    st.caption("Uses simple proxies (SPY, QQQ, IWM, TLT, GLD, BTC) as factor returns. Display-only inference.")

    hist = compute_wave_history(selected_wave, mode=mode, days=max(365, history_days))
    if hist is None or hist.empty:
        st.info("No history available.")
    else:
        fac_px = fetch_prices_daily(["SPY", "QQQ", "IWM", "TLT", "GLD", "BTC-USD"], days=max(365, history_days))
        if fac_px is None or fac_px.empty:
            st.info("Factor prices unavailable.")
        else:
            fac_ret = fac_px.pct_change().fillna(0.0)
            fac_ret = fac_ret.rename(columns={"BTC-USD": "BTC"})
            aligned = pd.concat([hist["wave_ret"].rename("wave"), fac_ret], axis=1).dropna()
            if aligned.shape[0] < 60:
                st.info("Not enough aligned data for regression (need ~60+ days).")
            else:
                betas = regress_factors(aligned["wave"], aligned.drop(columns=["wave"]))
                beta_df = pd.DataFrame([{"Factor": k, "Beta": v} for k, v in betas.items()])
                show_df(beta_df, selected_wave, key="factor_beta_table")

                if go is not None:
                    fig = go.Figure()
                    fig.add_trace(go.Bar(x=beta_df["Factor"], y=beta_df["Beta"], name="Beta"))
                    fig.update_layout(height=360, title="Factor Betas (OLS)", margin=dict(l=40, r=40, t=50, b=40))
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.bar_chart(beta_df.set_index("Factor"))


# ============================================================
# TAB 5: Vector OS Insight Layer
# ============================================================
with tab_vector:
    st.subheader("🧠 Vector OS Insight Layer")
    st.caption("Narrative flags + interpretation. No math changes — this interprets existing metrics.")

    attrib = compute_alpha_attribution(selected_wave, mode=mode, days=365)
    hist = compute_wave_history(selected_wave, mode=mode, days=max(365, history_days))
    audit = data_quality_audit(hist)

    flags: List[str] = []
    insights: List[str] = []

    # Data flags
    if audit.get("ok") and audit.get("flags"):
        flags.extend(audit["flags"])

    # Alpha/TE flags
    if attrib:
        a365 = attrib.get("Alpha vs Benchmark", float("nan"))
        te = attrib.get("Tracking Error (TE)", float("nan"))
        ir = attrib.get("Information Ratio (IR)", float("nan"))
        beta_r = attrib.get("β_real (Wave vs BM)", float("nan"))
        beta_t = attrib.get("β_target (if available)", float("nan"))

        if math.isfinite(a365) and a365 > 0.05:
            insights.append(f"365D alpha vs benchmark is positive (**{fmt_pct(a365)}**). This supports durable outperformance.")
        elif math.isfinite(a365) and a365 < 0:
            insights.append(f"365D alpha vs benchmark is negative (**{fmt_pct(a365)}**). Confirm benchmark mix and recent regime sensitivity.")

        if math.isfinite(te) and te > te_warn:
            flags.append(f"Tracking error elevated (**{fmt_pct(te)}**). Active risk is high relative to benchmark.")

        if math.isfinite(ir):
            if ir >= 0.5:
                insights.append(f"Information ratio is strong (**{fmt_num(ir,2)}**). Risk-adjusted alpha quality looks good.")
            elif ir < 0:
                flags.append(f"IR is negative (**{fmt_num(ir,2)}**). Validate active risk, benchmark fit, and exposure control.")

        if math.isfinite(beta_r) and math.isfinite(beta_t):
            drift = abs(beta_r - beta_t)
            if drift > beta_drift_alert:
                flags.append(f"Beta drift breach: β_real {fmt_num(beta_r,2)} vs β_target {fmt_num(beta_t,2)} (drift {fmt_num(drift,2)}).")

        bm_diff = attrib.get("Benchmark Difficulty (BM - SPY)", float("nan"))
        if math.isfinite(bm_diff) and bm_diff > 0.03:
            insights.append(f"Benchmark is tougher than SPY on this window (BM-SPY **{fmt_pct(bm_diff)}**). Alpha is earned on a harder bar.")

        overlay = attrib.get("Overlay Contribution (Engine - Static)", float("nan"))
        if math.isfinite(overlay):
            if overlay > 0:
                insights.append(f"Engine overlay contributed **{fmt_pct(overlay)}** beyond static basket — confirms active decision layer.")
            elif overlay < 0:
                insights.append(f"Engine overlay detracted **{fmt_pct(overlay)}** vs static basket — review overlay triggers and turnover impacts.")

        alpha_cap_last = attrib.get("Alpha Captured (last day)", float("nan"))
        if math.isfinite(alpha_cap_last):
            insights.append(f"Latest daily alpha captured: **{fmt_pct(alpha_cap_last)}** (scaled if exposure map available).")

    # WaveScore insight
    ws_snap = compute_wavescore_for_all_waves(all_waves, mode=mode, days=365)
    if ws_snap is not None and not ws_snap.empty:
        rr = ws_snap[ws_snap["Wave"] == selected_wave]
        if not rr.empty:
            score = float(rr.iloc[0]["WaveScore"])
            grade = str(rr.iloc[0]["Grade"])
            insights.append(f"WaveScore (console proxy): **{fmt_score(score)}** (**{grade}**).")

    st.markdown("### Vector Flags")
    if flags:
        st.warning(" | ".join(flags))
    else:
        st.success("No major flags triggered on current window.")

    st.markdown("### Vector Narrative")
    if insights:
        for s in insights:
            st.write("• " + str(s))
    else:
        st.write("• No narrative available (missing attribution/history).")

    st.markdown("---")
    st.subheader("🏁 WaveScore Leaderboard (All Waves)")
    if ws_snap is None or ws_snap.empty:
        st.info("WaveScore unavailable.")
    else:
        ws_show = ws_snap.sort_values("WaveScore", ascending=False).copy()
        show_df(ws_show, selected_wave, key="wavescore_leaderboard")


# ============================================================
# Footer
# ============================================================
st.markdown("---")
st.caption("WAVES Intelligence™ Institutional Console • Diagnostics++ Build • Streamlit-safe • No engine math modified.")