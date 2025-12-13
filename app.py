# app.py — WAVES Intelligence™ Institutional Console
# Institutional Reference Build (IRB-1) + Visual Scan Upgrade (Sticky Summary + Row Highlight)
# FULL PRODUCTION FILE — SINGLE FILE (no patches)
#
# Adds:
#   • Pinned Summary Bar (sticky top) for fast scanning
#   • Row highlighting in key tables (selected wave + alpha heat tint)
#
# Notes:
#   • Does NOT modify engine math or baseline results.
#   • Wave Doctor + What-If remain "shadow" and labeled.
#   • Streamlit-safe; falls back gracefully if Styler isn't supported.

from __future__ import annotations

import math
from datetime import datetime, timedelta
from typing import Dict, List

import numpy as np
import pandas as pd
import streamlit as st

import waves_engine as we  # your engine module

try:
    import yfinance as yf
except ImportError:
    yf = None

# Plotly optional (avoid crashes if missing)
try:
    import plotly.graph_objects as go
except Exception:
    go = None


# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="WAVES Intelligence™ Console",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# GLOBAL CSS (Sticky Summary + better scan)
# ============================================================
st.markdown(
    """
<style>
/* Reduce padding slightly for denser scan */
.block-container { padding-top: 1.0rem; padding-bottom: 2rem; }

/* Sticky summary container */
.waves-sticky {
  position: sticky;
  top: 0;
  z-index: 999;
  backdrop-filter: blur(8px);
  -webkit-backdrop-filter: blur(8px);
  padding: 10px 10px 8px 10px;
  margin: 0 0 12px 0;
  border-radius: 14px;
  border: 1px solid rgba(255,255,255,0.10);
  background: rgba(10, 15, 28, 0.65);
}

/* Summary chips */
.waves-chip {
  display: inline-block;
  padding: 6px 10px;
  margin: 4px 6px 0 0;
  border-radius: 999px;
  border: 1px solid rgba(255,255,255,0.12);
  background: rgba(255,255,255,0.04);
  font-size: 0.85rem;
  line-height: 1.0rem;
  white-space: nowrap;
}

/* Subtle section headers */
.waves-hdr {
  font-weight: 700;
  letter-spacing: 0.2px;
}

/* Make tables feel slightly tighter */
div[data-testid="stDataFrame"] { border-radius: 12px; overflow: hidden; }
</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# HELPERS: formatting
# ============================================================
def fmt_pct(x: float, digits: int = 2) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "—"
    return f"{x*100:0.{digits}f}%"


def fmt_num(x: float, digits: int = 2) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "—"
    return f"{x:.{digits}f}"


def fmt_score(x: float) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "—"
    return f"{x:.1f}"


def safe_series(s: pd.Series) -> pd.Series:
    if s is None or len(s) == 0:
        return pd.Series(dtype=float)
    return s.copy()


# ============================================================
# HELPERS: data fetch / caching
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
    try:
        df = we.compute_history_nav(wave_name, mode=mode, days=days)
    except Exception:
        df = pd.DataFrame(columns=["wave_nav", "bm_nav", "wave_ret", "bm_ret"])
    return df


@st.cache_data(show_spinner=False)
def get_benchmark_mix() -> pd.DataFrame:
    try:
        return we.get_benchmark_mix_table()
    except Exception:
        return pd.DataFrame(columns=["Wave", "Ticker", "Name", "Weight"])


@st.cache_data(show_spinner=False)
def get_wave_holdings(wave_name: str) -> pd.DataFrame:
    try:
        return we.get_wave_holdings(wave_name)
    except Exception:
        return pd.DataFrame(columns=["Ticker", "Name", "Weight"])


# ============================================================
# CORE METRICS
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


def simple_ret(series: pd.Series, window: int) -> float:
    series = safe_series(series)
    if len(series) < 2:
        return float("nan")
    if len(series) < window:
        window = len(series)
    if window < 2:
        return float("nan")
    sub = series.iloc[-window:]
    a = float(sub.iloc[0])
    b = float(sub.iloc[-1])
    if a <= 0:
        return float("nan")
    return float(b / a - 1.0)


def regress_factors(wave_ret: pd.Series, factor_ret: pd.DataFrame) -> dict:
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
# ATTRIBUTION (Engine vs Static Basket)
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
def compute_spy_nav(days: int = 365) -> pd.Series:
    px = fetch_prices_daily(["SPY"], days=days)
    if px.empty or "SPY" not in px.columns:
        return pd.Series(dtype=float)
    nav = (1.0 + px["SPY"].pct_change().fillna(0.0)).cumprod()
    nav.name = "spy_nav"
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

    spy_nav = compute_spy_nav(days=days)
    spy_ret = ret_from_nav(spy_nav, window=len(spy_nav)) if len(spy_nav) >= 2 else float("nan")

    alpha_vs_spy = eng_ret - spy_ret if (pd.notna(eng_ret) and pd.notna(spy_ret)) else float("nan")
    benchmark_difficulty = bm_ret_total - spy_ret if (pd.notna(bm_ret_total) and pd.notna(spy_ret)) else float("nan")

    te = tracking_error(wave_ret, bm_ret)
    ir = information_ratio(nav_wave, nav_bm, te)

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

    out["Wave Vol"] = float(annualized_vol(wave_ret))
    out["Benchmark Vol"] = float(annualized_vol(bm_ret))
    out["Wave MaxDD"] = float(max_drawdown(nav_wave))
    out["Benchmark MaxDD"] = float(max_drawdown(nav_bm))
    return out


# ============================================================
# WAVE DOCTOR (Diagnostics + Recommendations) + WHAT-IF
# ============================================================
def benchmark_source_label(selected_wave: str) -> str:
    try:
        static_dict = getattr(we, "BENCHMARK_WEIGHTS_STATIC", None)
        if isinstance(static_dict, dict) and selected_wave in static_dict and static_dict.get(selected_wave):
            return "Static Override (Engine)"
    except Exception:
        pass
    return "Auto-Composite (Dynamic)"


def wave_doctor_assess(
    wave_name: str,
    mode: str,
    days: int = 365,
    alpha_warn: float = 0.08,
    te_warn: float = 0.20,
    vol_warn: float = 0.30,
    mdd_warn: float = -0.25,
) -> Dict[str, object]:
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

    flags, diagnosis, recs = [], [], []

    if pd.notna(a30) and abs(a30) >= alpha_warn:
        flags.append("Large 30D alpha (verify benchmark + coverage)")
        diagnosis.append("30D alpha is unusually large. This can be real signal, but also benchmark mix drift or data coverage shifts.")
        recs.append("Freeze benchmark mix for demo comparisons and monitor benchmark mix drift over time.")

    if pd.notna(a365) and a365 < 0:
        diagnosis.append("365D alpha is negative. This may be true underperformance or simply a tougher benchmark mix.")
        if pd.notna(bm_difficulty) and bm_difficulty > 0.03:
            diagnosis.append("Benchmark difficulty is positive vs SPY — alpha is harder here because your benchmark beat SPY.")
            recs.append("If the intent is broad-market validation, temporarily compare against SPY/QQQ mix (diagnostics only).")
    else:
        if pd.notna(a365) and a365 > 0.06:
            diagnosis.append("365D alpha is solidly positive — this aligns with the ‘green’ institutional-style outcome.")
            recs.append("Lock a benchmark snapshot (session) for reproducibility in demos.")

    if pd.notna(te) and te > te_warn:
        flags.append("High tracking error (active risk elevated)")
        diagnosis.append("Tracking error is high; the wave behaves very differently than its benchmark.")
        recs.append("Reduce momentum tilt strength and/or tighten exposure caps to reduce TE.")

    if pd.notna(vol_w) and vol_w > vol_warn:
        flags.append("High volatility")
        diagnosis.append("Volatility is elevated. If you want more institutional smoothness, lower the vol target.")
        recs.append("Lower vol target (e.g., 20% → 16–18%) in the What-If Lab for testing.")

    if pd.notna(mdd_w) and mdd_w < mdd_warn:
        flags.append("Deep drawdown")
        diagnosis.append("Drawdown is deep versus typical institutional tolerances.")
        recs.append("Increase SmartSafe posture in stress regimes (What-If) or tighten max exposure cap.")

    if pd.notna(vol_b) and pd.notna(vol_w) and vol_b > 0 and (vol_w / vol_b) > 1.6:
        flags.append("Volatility much higher than benchmark")
        diagnosis.append("Wave volatility is much higher than benchmark — can create ‘spiky’ alpha swings.")
        recs.append("Tighten exposure cap and reduce tilt strength to smooth returns.")

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
            "Benchmark_Difficulty_BM_minus_SPY": bm_difficulty,
        },
        "flags": flags,
        "diagnosis": diagnosis,
        "recommendations": list(dict.fromkeys(recs)),
    }


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

    wave_ret, dates = [], []
    for dt in rets.index:
        r = rets.loc[dt]

        mom_row = mom60.loc[dt] if dt in mom60.index else None
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
            recent = np.array(wave_ret[-20:])
            realized = recent.std() * np.sqrt(252)
        else:
            realized = vol_target

        vol_adj = 1.0
        if realized > 0:
            vol_adj = float(np.clip(vol_target / realized, 0.7, 1.3))

        reg = str(regime.get(dt, "neutral"))
        reg_expo = float(regime_exposure_map.get(reg, 1.0))
        vix_expo = float(vix_exposure.get(dt, 1.0))
        vix_gate = float(vix_safe.get(dt, 0.0))

        expo = float(np.clip(base_expo * reg_expo * vol_adj * vix_expo, exp_min, exp_max))
        sf = float(np.clip(regime_gate(mode, reg) + vix_gate + extra_safe_boost, 0.0, 0.95))
        rf = 1.0 - sf

        total = sf * float(safe_ret.get(dt, 0.0)) + rf * expo * port_risk_ret

        if mode == "Private Logic" and len(wave_ret) >= 20:
            recent = np.array(wave_ret[-20:])
            daily_vol = recent.std()
            if daily_vol > 0:
                shock = 2.0 * daily_vol
                if total <= -shock:
                    total = total * 1.30
                elif total >= shock:
                    total = total * 0.70

        wave_ret.append(total)
        dates.append(dt)

    wave_ret = pd.Series(wave_ret, index=pd.Index(dates, name="Date"), name="whatif_ret")
    wave_nav = (1.0 + wave_ret).cumprod().rename("whatif_nav")
    out = pd.DataFrame({"whatif_nav": wave_nav, "whatif_ret": wave_ret})

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
# WAVESCORE TABLE (proto)
# ============================================================
def _grade(score: float) -> str:
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


def compute_wavescore_table(all_waves: list[str], mode_in: str, days: int = 365) -> pd.DataFrame:
    rows = []
    for w in all_waves:
        h = compute_wave_history(w, mode_in, days)
        if h.empty or len(h) < 40:
            rows.append(
                {
                    "Wave": w,
                    "WaveScore": np.nan,
                    "Grade": "N/A",
                    "Return Quality": np.nan,
                    "Risk Control": np.nan,
                    "Consistency": np.nan,
                    "Resilience": np.nan,
                    "Efficiency": np.nan,
                    "Transparency": 10.0,
                    "Alpha_365D": np.nan,
                    "IR_365D": np.nan,
                }
            )
            continue

        nav_w = h["wave_nav"]
        nav_b = h["bm_nav"]
        ret_w = h["wave_ret"]
        ret_b = h["bm_ret"]

        r_w = ret_from_nav(nav_w, len(nav_w))
        r_b = ret_from_nav(nav_b, len(nav_b))
        alpha = r_w - r_b

        te = tracking_error(ret_w, ret_b)
        ir = information_ratio(nav_w, nav_b, te)

        vol_w = annualized_vol(ret_w)
        vol_b = annualized_vol(ret_b)
        vol_ratio = (vol_w / vol_b) if (vol_b and not math.isnan(vol_b) and vol_b > 0) else np.nan

        mdd_w = max_drawdown(nav_w)
        mdd_b = max_drawdown(nav_b)

        hit = float((ret_w >= ret_b).mean()) if len(ret_w) else np.nan

        rq = 0.0
        if not math.isnan(ir):
            rq += float(np.clip(ir, 0.0, 1.5) / 1.5 * 15.0)
        if not math.isnan(alpha):
            rq += float(np.clip((alpha + 0.05) / 0.15, 0.0, 1.0) * 10.0)
        rq = float(np.clip(rq, 0.0, 25.0))

        rc = 0.0
        if not math.isnan(vol_ratio):
            penalty = max(0.0, abs(vol_ratio - 0.9) - 0.1)
            rc = float(np.clip(1.0 - penalty / 0.6, 0.0, 1.0) * 25.0)

        cons = float(np.clip(hit, 0.0, 1.0) * 15.0) if not math.isnan(hit) else 0.0

        rec = 0.0
        if len(nav_w) > 5:
            trough = float(nav_w.min())
            peak = float(nav_w.max())
            last = float(nav_w.iloc[-1])
            if peak > trough and trough > 0:
                recovery = float(np.clip((last - trough) / (peak - trough), 0.0, 1.0))
            else:
                recovery = np.nan
        else:
            recovery = np.nan

        if not math.isnan(recovery) and not math.isnan(mdd_w) and not math.isnan(mdd_b):
            rec_part = recovery * 6.0
            dd_edge = (mdd_b - mdd_w)
            dd_part = float(np.clip((dd_edge + 0.10) / 0.20, 0.0, 1.0) * 4.0)
            rec = float(np.clip(rec_part + dd_part, 0.0, 10.0))

        eff = float(np.clip((0.25 - te) / 0.20, 0.0, 1.0) * 15.0) if not math.isnan(te) else 0.0

        transparency = 10.0
        total = float(np.clip(rq + rc + cons + rec + eff + transparency, 0.0, 100.0))

        rows.append(
            {
                "Wave": w,
                "WaveScore": total,
                "Grade": _grade(total),
                "Return Quality": rq,
                "Risk Control": rc,
                "Consistency": cons,
                "Resilience": rec,
                "Efficiency": eff,
                "Transparency": transparency,
                "Alpha_365D": alpha,
                "IR_365D": ir,
            }
        )

    df = pd.DataFrame(rows)
    return df.sort_values("WaveScore", ascending=False)


# ============================================================
# ROW HIGHLIGHTING (Selected row + alpha heat tint)
# ============================================================
def _style_highlight_selected_and_alpha(df: pd.DataFrame, selected_wave: str) -> "pd.io.formats.style.Styler":
    # Columns that might contain alpha-like values
    alpha_cols = [c for c in df.columns if "Alpha" in c]

    def row_style(row: pd.Series):
        styles = [""] * len(row)

        # Selected row highlight
        if "Wave" in df.columns and str(row.get("Wave", "")) == str(selected_wave):
            styles = ["background-color: rgba(255,255,255,0.07); font-weight: 700;"] * len(row)

        # Alpha tint overlay (cell-based) if not selected
        if alpha_cols:
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
    return styler


def show_table(df: pd.DataFrame, selected_wave: str, key: str):
    """
    Uses Styler when available; otherwise falls back to normal dataframe.
    """
    try:
        st.dataframe(_style_highlight_selected_and_alpha(df, selected_wave), use_container_width=True, key=key)
    except Exception:
        st.dataframe(df, use_container_width=True, key=key)


# ============================================================
# SIDEBAR
# ============================================================
all_waves = we.get_all_waves()
all_modes = we.get_modes()

with st.sidebar:
    st.title("WAVES Intelligence™")
    st.caption("Mini Bloomberg Console • Vector OS™")

    mode = st.selectbox("Mode", all_modes, index=0)
    selected_wave = st.selectbox("Select Wave", all_waves, index=0)

    st.markdown("---")
    st.markdown("**Display settings**")
    history_days = st.slider("History window (days)", min_value=60, max_value=730, value=365, step=15)

    st.markdown("---")
    st.markdown("**Wave Doctor settings**")
    alpha_warn = st.slider("Alpha alert threshold (abs, 30D)", 0.02, 0.20, 0.08, 0.01)
    te_warn = st.slider("Tracking error alert threshold", 0.05, 0.40, 0.20, 0.01)


# ============================================================
# HEADER
# ============================================================
st.title("WAVES Intelligence™ Institutional Console")
st.caption("Live Alpha Capture • SmartSafe™ • Multi-Asset • Crypto • Gold • Income Ladders")


# ============================================================
# PINNED SUMMARY BAR (Sticky)
# ============================================================
# Compute quick metrics for selected wave
h_sel_for_bar = compute_wave_history(selected_wave, mode, max(365, history_days))

bar_r30 = bar_a30 = bar_r365 = bar_a365 = float("nan")
bar_te = bar_ir = float("nan")
bar_src = benchmark_source_label(selected_wave)

if not h_sel_for_bar.empty and len(h_sel_for_bar) >= 2:
    nav_w = h_sel_for_bar["wave_nav"]
    nav_b = h_sel_for_bar["bm_nav"]
    ret_w = h_sel_for_bar["wave_ret"]
    ret_b = h_sel_for_bar["bm_ret"]

    bar_r30 = ret_from_nav(nav_w, min(30, len(nav_w)))
    bar_rb30 = ret_from_nav(nav_b, min(30, len(nav_b)))
    bar_a30 = bar_r30 - bar_rb30

    bar_r365 = ret_from_nav(nav_w, len(nav_w))
    bar_rb365 = ret_from_nav(nav_b, len(nav_b))
    bar_a365 = bar_r365 - bar_rb365

    bar_te = tracking_error(ret_w, ret_b)
    bar_ir = information_ratio(nav_w, nav_b, bar_te)

# WaveScore for selected wave (proto)
ws_df_bar = compute_wavescore_table(all_waves, mode, 365)
bar_ws = bar_grade = "—"
if ws_df_bar is not None and not ws_df_bar.empty:
    row = ws_df_bar[ws_df_bar["Wave"] == selected_wave]
    if not row.empty:
        bar_ws = fmt_score(float(row.iloc[0]["WaveScore"]))
        bar_grade = str(row.iloc[0]["Grade"])

# VIX + Regime snapshot
spy_vix_bar = fetch_spy_vix(days=min(365, max(120, history_days)))
vix_last = float("nan")
reg_now = "—"
if not spy_vix_bar.empty and "SPY" in spy_vix_bar.columns and "^VIX" in spy_vix_bar.columns and len(spy_vix_bar) > 60:
    vix_last = float(spy_vix_bar["^VIX"].iloc[-1])
    spy_nav_bar = (spy_vix_bar["SPY"].pct_change().fillna(0.0) + 1.0).cumprod()
    reg_series = _regime_from_spy_60d(spy_nav_bar)
    if len(reg_series) > 0:
        reg_now = str(reg_series.iloc[-1])

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
  <span class="waves-chip">WaveScore: <b>{bar_ws}</b> ({bar_grade})</span>
</div>
""",
    unsafe_allow_html=True,
)


# ============================================================
# TAB LAYOUT
# ============================================================
tab_console, tab_market, tab_factors, tab_vector = st.tabs(
    ["Console", "Market Intel", "Factor Decomposition", "Vector OS Insight Layer"]
)

# ============================================================
# TAB 1: CONSOLE
# ============================================================
with tab_console:
    st.subheader("Market Regime Monitor — SPY vs VIX")
    spy_vix = fetch_spy_vix(days=history_days)

    if spy_vix.empty or "SPY" not in spy_vix.columns or "^VIX" not in spy_vix.columns:
        st.warning("Unable to load SPY/VIX data at the moment.")
    else:
        spy = spy_vix["SPY"].copy()
        vix = spy_vix["^VIX"].copy()
        spy_norm = (spy / spy.iloc[0] * 100.0) if len(spy) > 0 else spy

        if go is not None:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=spy_vix.index, y=spy_norm, name="SPY (Index = 100)", mode="lines"))
            fig.add_trace(go.Scatter(x=spy_vix.index, y=vix, name="VIX Level", mode="lines", yaxis="y2"))
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

    # 1) ALL WAVES OVERVIEW
    st.markdown("### 🧾 All Waves Overview (1D / 30D / 60D / 365D Returns + Alpha)")

    overview_rows = []
    for w in all_waves:
        h = compute_wave_history(w, mode, 365)
        if h.empty or len(h) < 2:
            overview_rows.append(
                {"Wave": w, "1D Ret": np.nan, "1D Alpha": np.nan, "30D Ret": np.nan, "30D Alpha": np.nan,
                 "60D Ret": np.nan, "60D Alpha": np.nan, "365D Ret": np.nan, "365D Alpha": np.nan}
            )
            continue

        nav_w = h["wave_nav"]
        nav_b = h["bm_nav"]

        if len(nav_w) >= 2:
            r1w = float(nav_w.iloc[-1] / nav_w.iloc[-2] - 1.0)
            r1b = float(nav_b.iloc[-1] / nav_b.iloc[-2] - 1.0)
            a1 = r1w - r1b
        else:
            r1w = r1b = a1 = np.nan

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
            {"Wave": w, "1D Ret": r1w, "1D Alpha": a1, "30D Ret": r30w, "30D Alpha": a30,
             "60D Ret": r60w, "60D Alpha": a60, "365D Ret": r365w, "365D Alpha": a365}
        )

    overview_df = pd.DataFrame(overview_rows)

    # Show formatted + highlighted
    fmt_overview = overview_df.copy()
    for col in ["1D Ret", "1D Alpha", "30D Ret", "30D Alpha", "60D Ret", "60D Alpha", "365D Ret", "365D Alpha"]:
        fmt_overview[col] = fmt_overview[col].apply(lambda x: float(x) if pd.notna(x) else np.nan)

    show_table(fmt_overview, selected_wave, key="overview_table_fmt")
    st.caption("Tip: selected Wave row is highlighted; Alpha cells tint green/red to scan fast.")
    st.markdown("---")

    # 2) RISK ANALYTICS
    st.markdown("### 🛡️ Risk Analytics (Vol, MaxDD, TE, IR) — 365D Window")

    risk_rows = []
    for w in all_waves:
        h = compute_wave_history(w, mode, 365)
        if h.empty or len(h) < 2:
            risk_rows.append(
                {"Wave": w, "Wave Vol": np.nan, "Benchmark Vol": np.nan, "Wave MaxDD": np.nan,
                 "Benchmark MaxDD": np.nan, "Tracking Error": np.nan, "Information Ratio": np.nan}
            )
            continue

        nav_w = h["wave_nav"]
        nav_b = h["bm_nav"]
        ret_w = h["wave_ret"]
        ret_b = h["bm_ret"]

        te = tracking_error(ret_w, ret_b)
        ir = information_ratio(nav_w, nav_b, te)

        risk_rows.append(
            {"Wave": w, "Wave Vol": annualized_vol(ret_w), "Benchmark Vol": annualized_vol(ret_b),
             "Wave MaxDD": max_drawdown(nav_w), "Benchmark MaxDD": max_drawdown(nav_b),
             "Tracking Error": te, "Information Ratio": ir}
        )

    risk_df = pd.DataFrame(risk_rows)
    show_table(risk_df, selected_wave, key="risk_table")
    st.markdown("---")

    # 3) WAVESCORE LEADERBOARD
    st.markdown("### 🏁 WaveScore™ Leaderboard (Proto · 365D)")
    ws_df = compute_wavescore_table(all_waves, mode, 365)
    show_table(ws_df, selected_wave, key="wavescore_table")
    st.markdown("---")

    # 4) BENCHMARK TRANSPARENCY TABLE
    st.markdown("### 🧩 Benchmark Transparency Table (Composite Components per Wave)")
    bm_mix = get_benchmark_mix()
    if bm_mix is None or bm_mix.empty:
        st.info("No benchmark mix table available from engine.")
    else:
        st.dataframe(bm_mix, use_container_width=True)

    st.markdown("---")

    # 5) WAVE DETAIL
    st.markdown(f"## 📌 Wave Detail — **{selected_wave}** (Mode: **{mode}**)")

    h_sel = compute_wave_history(selected_wave, mode, history_days)
    c_left, c_right = st.columns([2.2, 1.0])

    with c_left:
        st.markdown("#### NAV vs Benchmark")
        if h_sel.empty or len(h_sel) < 2:
            st.warning("Not enough data to chart NAV.")
        else:
            nav_w = h_sel["wave_nav"]
            nav_b = h_sel["bm_nav"]
            plot_df = pd.DataFrame({"Wave_NAV": nav_w / float(nav_w.iloc[0]), "Benchmark_NAV": nav_b / float(nav_b.iloc[0])})

            if go is not None:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df["Wave_NAV"], name="Wave (Engine)", mode="lines"))
                fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df["Benchmark_NAV"], name="Benchmark", mode="lines"))
                fig.update_layout(
                    height=420,
                    margin=dict(l=40, r=40, t=40, b=40),
                    legend=dict(orientation="h"),
                    xaxis=dict(title="Date"),
                    yaxis=dict(title="Normalized NAV (Start=1.0)"),
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.line_chart(plot_df)

    with c_right:
        st.markdown("#### Key Metrics")
        if h_sel.empty or len(h_sel) < 2:
            st.info("No metrics available.")
        else:
            nav_w = h_sel["wave_nav"]
            nav_b = h_sel["bm_nav"]
            ret_w = h_sel["wave_ret"]
            ret_b = h_sel["bm_ret"]

            r30w = ret_from_nav(nav_w, min(30, len(nav_w)))
            r30b = ret_from_nav(nav_b, min(30, len(nav_b)))
            a30 = r30w - r30b

            r365w = ret_from_nav(nav_w, len(nav_w))
            r365b = ret_from_nav(nav_b, len(nav_b))
            a365 = r365w - r365b

            te = tracking_error(ret_w, ret_b)
            ir = information_ratio(nav_w, nav_b, te)

            st.metric("30D Return", fmt_pct(r30w))
            st.metric("30D Alpha", fmt_pct(a30))
            st.metric("365D Return", fmt_pct(r365w))
            st.metric("365D Alpha", fmt_pct(a365))

            st.markdown("---")
            st.metric("Vol (ann.)", fmt_pct(annualized_vol(ret_w)))
            st.metric("Tracking Error", fmt_pct(te))
            st.metric("Information Ratio", fmt_num(ir, 2))
            st.metric("Max Drawdown", fmt_pct(max_drawdown(nav_w)))

    st.markdown("---")

    # 6) BENCHMARK TRUTH PANEL
    with st.expander("✅ Benchmark Truth Panel (composition + difficulty + diagnostics)", expanded=True):
        bm_mix_df = get_benchmark_mix()
        if bm_mix_df is None or bm_mix_df.empty:
            st.warning("Benchmark mix table not available from engine.")
        else:
            wmix = bm_mix_df[bm_mix_df["Wave"] == selected_wave].copy() if "Wave" in bm_mix_df.columns else pd.DataFrame()
            src = benchmark_source_label(selected_wave)

            colA, colB, colC, colD = st.columns(4)
            colA.metric("Benchmark Source", src)

            if h_sel.empty or len(h_sel) < 2:
                colB.metric("Benchmark Return (365D)", "—")
                colC.metric("SPY Return (365D)", "—")
                colD.metric("BM Difficulty (BM−SPY)", "—")
            else:
                bm_ret_365 = ret_from_nav(h_sel["bm_nav"], len(h_sel))
                spy_nav_365 = compute_spy_nav(365)
                spy_ret_365 = ret_from_nav(spy_nav_365, len(spy_nav_365)) if len(spy_nav_365) >= 2 else np.nan
                diff = bm_ret_365 - spy_ret_365 if (pd.notna(bm_ret_365) and pd.notna(spy_ret_365)) else np.nan

                colB.metric("Benchmark Return (365D)", fmt_pct(bm_ret_365))
                colC.metric("SPY Return (365D)", fmt_pct(spy_ret_365))
                colD.metric("BM Difficulty (BM−SPY)", fmt_pct(diff))

            st.markdown("#### Benchmark Composition (as used by engine)")
            if wmix is None or wmix.empty:
                st.warning("No benchmark components found for this Wave in the mix table.")
            else:
                wmix = wmix.sort_values("Weight", ascending=False) if "Weight" in wmix.columns else wmix
                st.dataframe(wmix, use_container_width=True)

            if not h_sel.empty and len(h_sel) >= 2:
                bm_vol = annualized_vol(h_sel["bm_ret"]) if "bm_ret" in h_sel.columns else np.nan
                bm_mdd = max_drawdown(h_sel["bm_nav"]) if "bm_nav" in h_sel.columns else np.nan
                r1, r2, r3 = st.columns(3)
                r1.metric("Benchmark Vol", fmt_pct(bm_vol))
                r2.metric("Benchmark MaxDD", fmt_pct(bm_mdd))
                r3.metric("Days in Window", str(int(len(h_sel))))

    st.markdown("---")

    # 7) MODE SEPARATION PROOF
    with st.expander("✅ Mode Separation Proof (outputs + mode levers)", expanded=False):
        rows = []
        for m in all_modes:
            hh = compute_wave_history(selected_wave, m, 365)
            if hh.empty or len(hh) < 2:
                rows.append({"Mode": m, "Base Exposure": np.nan, "Exposure Caps": "—", "365D Return": np.nan, "365D Alpha": np.nan, "TE": np.nan, "IR": np.nan})
                continue

            nav_w = hh["wave_nav"]
            nav_b = hh["bm_nav"]
            ret_w = hh["wave_ret"]
            ret_b = hh["bm_ret"]

            r365w = ret_from_nav(nav_w, len(nav_w))
            r365b = ret_from_nav(nav_b, len(nav_b))
            a365 = r365w - r365b

            te = tracking_error(ret_w, ret_b)
            ir = information_ratio(nav_w, nav_b, te)

            base_expo = np.nan
            caps = "—"
            try:
                base_map = getattr(we, "MODE_BASE_EXPOSURE", None)
                cap_map = getattr(we, "MODE_EXPOSURE_CAPS", None)
                if isinstance(base_map, dict) and m in base_map:
                    base_expo = float(base_map[m])
                if isinstance(cap_map, dict) and m in cap_map:
                    lo, hi = cap_map[m]
                    caps = f"{float(lo):.2f}–{float(hi):.2f}"
            except Exception:
                pass

            rows.append({"Mode": m, "Base Exposure": base_expo, "Exposure Caps": caps, "365D Return": r365w, "365D Alpha": a365, "TE": te, "IR": ir})

        dfm = pd.DataFrame(rows)
        st.dataframe(dfm, use_container_width=True)
        st.caption("Demo-safe proof that mode toggles are not cosmetic (shows outputs + engine levers when exposed).")

    st.markdown("---")

    # 8) ALPHA ATTRIBUTION
    with st.expander("🔍 Alpha Attribution (Engine vs Static Basket · 365D)", expanded=False):
        attrib = compute_alpha_attribution(selected_wave, mode=mode, days=365)
        if not attrib:
            st.warning("Not enough data to compute attribution for this Wave yet.")
        else:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Engine Return (365D)", fmt_pct(attrib.get("Engine Return", np.nan)))
            c2.metric("Static Basket Return", fmt_pct(attrib.get("Static Basket Return", np.nan)))
            c3.metric("Overlay Contribution", fmt_pct(attrib.get("Overlay Contribution (Engine - Static)", np.nan)))
            c4.metric("Alpha vs Benchmark", fmt_pct(attrib.get("Alpha vs Benchmark", np.nan)))

            c5, c6, c7, c8 = st.columns(4)
            c5.metric("Benchmark Return", fmt_pct(attrib.get("Benchmark Return", np.nan)))
            c6.metric("SPY Return", fmt_pct(attrib.get("SPY Return", np.nan)))
            c7.metric("Benchmark Difficulty (BM−SPY)", fmt_pct(attrib.get("Benchmark Difficulty (BM - SPY)", np.nan)))
            c8.metric("Information Ratio (IR)", fmt_num(attrib.get("Information Ratio (IR)", np.nan), 2))

    st.markdown("---")

    # 9) WAVE DOCTOR + WHAT-IF
    with st.expander("🩺 Wave Doctor™ (diagnostics + recommendations) + What-If Lab", expanded=True):
        assess = wave_doctor_assess(selected_wave, mode=mode, days=365, alpha_warn=float(alpha_warn), te_warn=float(te_warn))

        if not assess.get("ok", False):
            st.info(assess.get("message", "Wave Doctor unavailable."))
        else:
            metrics = assess["metrics"]
            flags = assess["flags"]
            diagnosis = assess["diagnosis"]
            recs = assess["recommendations"]

            a1, a2, a3, a4 = st.columns(4)
            a1.metric("365D Return", fmt_pct(metrics.get("Return_365D", np.nan)))
            a2.metric("365D Alpha", fmt_pct(metrics.get("Alpha_365D", np.nan)))
            a3.metric("Tracking Error", fmt_pct(metrics.get("TE", np.nan)))
            a4.metric("IR", fmt_num(metrics.get("IR", np.nan), 2))

            b1, b2, b3, b4 = st.columns(4)
            b1.metric("Wave Vol", fmt_pct(metrics.get("Vol_Wave", np.nan)))
            b2.metric("Wave MaxDD", fmt_pct(metrics.get("MaxDD_Wave", np.nan)))
            b3.metric("BM Difficulty (BM−SPY)", fmt_pct(metrics.get("Benchmark_Difficulty_BM_minus_SPY", np.nan)))
            b4.metric("30D Alpha", fmt_pct(metrics.get("Alpha_30D", np.nan)))

            if flags:
                st.warning("Wave Doctor Flags: " + " • ".join(flags))

            st.markdown("#### What might be going on (plain English)")
            for line in diagnosis:
                st.write(f"- {line}")

            st.markdown("#### Recommended adjustments (suggestions only — baseline unchanged)")
            if recs:
                for r in recs:
                    st.write(f"- {r}")
            else:
                st.write("- No changes recommended based on current thresholds.")

            st.markdown("---")
            st.markdown("### What-If Lab (Shadow Simulation — does NOT change official results)")

            default_tilt = 0.80
            default_vol = 0.20
            default_extra_safe = 0.00
            default_exp_min, default_exp_max = 0.70, 1.30

            try:
                cap_map = getattr(we, "MODE_EXPOSURE_CAPS", None)
                if isinstance(cap_map, dict) and mode in cap_map:
                    default_exp_min, default_exp_max = float(cap_map[mode][0]), float(cap_map[mode][1])
            except Exception:
                pass

            wcol1, wcol2, wcol3 = st.columns(3)
            with wcol1:
                tilt_strength = st.slider("Momentum tilt strength (60D)", 0.0, 1.50, float(default_tilt), 0.05)
                vol_target = st.slider("Vol target (annualized)", 0.08, 0.35, float(default_vol), 0.01)
            with wcol2:
                extra_safe = st.slider("Extra SmartSafe boost", 0.0, 0.30, float(default_extra_safe), 0.01)
                freeze_bm = st.checkbox("Freeze benchmark to engine benchmark (for comparison)", value=True)
            with wcol3:
                exp_min = st.slider("Exposure min", 0.20, 1.00, float(default_exp_min), 0.01)
                exp_max = st.slider("Exposure max", 0.80, 2.00, float(default_exp_max), 0.01)

            whatif = simulate_whatif_nav(
                wave_name=selected_wave,
                mode=mode,
                days=365,
                tilt_strength=float(tilt_strength),
                vol_target=float(vol_target),
                extra_safe_boost=float(extra_safe),
                exp_min=float(exp_min),
                exp_max=float(exp_max),
                freeze_benchmark=bool(freeze_bm),
            )

            baseline = compute_wave_history(selected_wave, mode=mode, days=365)

            if whatif.empty or baseline.empty or len(baseline) < 2:
                st.info("What-If results unavailable (missing price inputs or insufficient data).")
            else:
                b_nav = baseline["wave_nav"].reindex(whatif.index).ffill().bfill()
                b_bm = baseline["bm_nav"].reindex(whatif.index).ffill().bfill()

                w_nav = whatif["whatif_nav"]
                w_bm = whatif["bm_nav"] if "bm_nav" in whatif.columns else b_bm

                b_ret = ret_from_nav(b_nav, len(b_nav))
                b_bret = ret_from_nav(b_bm, len(b_bm))
                b_alpha = b_ret - b_bret

                w_ret = ret_from_nav(w_nav, len(w_nav))
                w_bret = ret_from_nav(w_bm, len(w_bm))
                w_alpha = w_ret - w_bret

                d1, d2, d3, d4 = st.columns(4)
                d1.metric("Baseline 365D Return", fmt_pct(b_ret))
                d2.metric("What-If 365D Return", fmt_pct(w_ret), delta=fmt_pct(w_ret - b_ret))
                d3.metric("Baseline 365D Alpha", fmt_pct(b_alpha))
                d4.metric("What-If 365D Alpha", fmt_pct(w_alpha), delta=fmt_pct(w_alpha - b_alpha))

                st.markdown("#### Baseline vs What-If NAV (normalized)")
                plot_df = pd.DataFrame(
                    {"Baseline_NAV": b_nav / float(b_nav.iloc[0]), "WhatIf_NAV": w_nav / float(w_nav.iloc[0])}
                )

                if go is not None:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df["Baseline_NAV"], name="Baseline (Engine)", mode="lines"))
                    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df["WhatIf_NAV"], name="What-If (Shadow)", mode="lines"))
                    fig.update_layout(height=360, margin=dict(l=40, r=40, t=40, b=40), legend=dict(orientation="h"))
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.line_chart(plot_df)

    st.markdown("---")

    # 10) TOP-10 HOLDINGS
    st.markdown("### 🔗 Top-10 Holdings (Google Finance links)")
    hold = get_wave_holdings(selected_wave)
    if hold is None or hold.empty:
        st.info("No holdings available for this Wave.")
    else:
        hold = hold.copy()
        if "Weight" in hold.columns:
            hold["Weight"] = pd.to_numeric(hold["Weight"], errors="coerce").fillna(0.0)
        hold = hold.sort_values("Weight", ascending=False).head(10)

        def google_finance_link(ticker: str) -> str:
            t = str(ticker).strip()
            if not t:
                return ""
            base = "https://www.google.com/finance/quote"
            return f"[{t}]({base}/{t})"

        hold["Google Finance"] = hold["Ticker"].astype(str).apply(google_finance_link)
        st.dataframe(hold, use_container_width=True)


# ============================================================
# TAB 2: MARKET INTEL
# ============================================================
with tab_market:
    st.markdown("## 🌍 Global Market Dashboard")
    market_df = fetch_market_assets(history_days)

    if market_df is None or market_df.empty:
        st.warning("Unable to load multi-asset market data right now.")
    else:
        assets = {
            "SPY": "S&P 500",
            "QQQ": "NASDAQ-100",
            "IWM": "US Small Caps",
            "TLT": "US 20+Y Treasuries",
            "GLD": "Gold",
            "BTC-USD": "Bitcoin (USD)",
            "^VIX": "VIX (Implied Vol)",
            "^TNX": "US 10Y Yield",
        }

        rows = []
        for tkr, label in assets.items():
            if tkr not in market_df.columns:
                continue
            series = market_df[tkr]
            last = float(series.iloc[-1]) if len(series) else float("nan")
            r1d = simple_ret(series, 2)
            r30 = simple_ret(series, 30)
            rows.append({"Ticker": tkr, "Asset": label, "Last": last, "1D Return": r1d, "30D Return": r30})

        snap = pd.DataFrame(rows)
        st.dataframe(snap, use_container_width=True)

    st.markdown("---")
    st.markdown(f"### ⚡ WAVES Reaction Snapshot (30D · Mode = {mode})")

    reaction_rows = []
    for w in all_waves:
        h = compute_wave_history(w, mode, 365)
        if h.empty or len(h) < 2:
            reaction_rows.append({"Wave": w, "30D Return": np.nan, "30D Alpha": np.nan, "Classification": "No data"})
            continue

        nav_w = h["wave_nav"]
        nav_b = h["bm_nav"]
        r30w = ret_from_nav(nav_w, min(30, len(nav_w)))
        r30b = ret_from_nav(nav_b, min(30, len(nav_b)))
        a30 = r30w - r30b

        if math.isnan(a30):
            label = "No data"
        elif a30 >= 0.05:
            label = "Strong Outperformance"
        elif a30 >= 0.02:
            label = "Outperforming"
        elif a30 <= -0.03:
            label = "Lagging"
        else:
            label = "Near Benchmark"

        reaction_rows.append({"Wave": w, "30D Return": r30w, "30D Alpha": a30, "Classification": label})

    reaction = pd.DataFrame(reaction_rows)
    show_table(reaction, selected_wave, key="reaction_table")


# ============================================================
# TAB 3: FACTOR DECOMPOSITION
# ============================================================
with tab_factors:
    st.markdown("## 🧬 Factor Decomposition (Institution-Style)")

    factor_days = min(history_days, 365)
    factor_prices = fetch_market_assets(factor_days)

    needed = ["SPY", "QQQ", "IWM", "TLT", "GLD", "BTC-USD"]
    missing = [t for t in needed if (factor_prices is None or factor_prices.empty or t not in factor_prices.columns)]
    if factor_prices is None or factor_prices.empty or missing:
        st.warning("Unable to load all factor series." + (f" Missing: {', '.join(missing)}" if missing else ""))
    else:
        factor_returns = factor_prices[needed].pct_change().dropna()
        factor_returns = factor_returns.rename(
            columns={
                "SPY": "MKT_SPY",
                "QQQ": "GROWTH_QQQ",
                "IWM": "SIZE_IWM",
                "TLT": "RATES_TLT",
                "GLD": "GOLD_GLD",
                "BTC-USD": "CRYPTO_BTC",
            }
        )

        rows = []
        for w in all_waves:
            h = compute_wave_history(w, mode, factor_days)
            if h.empty or "wave_ret" not in h.columns:
                rows.append({"Wave": w, "β_SPY": np.nan, "β_QQQ": np.nan, "β_IWM": np.nan, "β_TLT": np.nan, "β_GLD": np.nan, "β_BTC": np.nan})
                continue
            betas = regress_factors(h["wave_ret"], factor_returns)
            rows.append(
                {"Wave": w, "β_SPY": betas.get("MKT_SPY", np.nan), "β_QQQ": betas.get("GROWTH_QQQ", np.nan),
                 "β_IWM": betas.get("SIZE_IWM", np.nan), "β_TLT": betas.get("RATES_TLT", np.nan),
                 "β_GLD": betas.get("GOLD_GLD", np.nan), "β_BTC": betas.get("CRYPTO_BTC", np.nan)}
            )

        beta_df = pd.DataFrame(rows)
        show_table(beta_df, selected_wave, key="beta_table")

    st.markdown("---")
    st.markdown(f"### 🔁 Correlation Matrix — Waves (Daily Returns · Mode = {mode})")

    corr_days = min(history_days, 365)
    ret_panel = {}
    for w in all_waves:
        h = compute_wave_history(w, mode, corr_days)
        if h.empty or "wave_ret" not in h.columns:
            continue
        ret_panel[w] = h["wave_ret"]

    if not ret_panel:
        st.info("No return data available to compute correlations.")
    else:
        ret_df = pd.DataFrame(ret_panel).dropna(how="all")
        if ret_df.empty or ret_df.shape[1] < 2:
            st.info("Not enough overlapping data to compute correlations.")
        else:
            corr = ret_df.corr()
            st.dataframe(corr, use_container_width=True)

            if go is not None:
                fig_corr = go.Figure(
                    data=go.Heatmap(
                        z=corr.values,
                        x=corr.columns,
                        y=corr.index,
                        zmin=-1,
                        zmax=1,
                        colorbar=dict(title="ρ"),
                    )
                )
                fig_corr.update_layout(
                    title="Wave Correlation Matrix (Daily Returns)",
                    xaxis_title="Wave",
                    yaxis_title="Wave",
                    height=520,
                    margin=dict(l=60, r=60, t=60, b=60),
                )
                st.plotly_chart(fig_corr, use_container_width=True)


# ============================================================
# TAB 4: VECTOR OS INSIGHT LAYER (RULES-BASED)
# ============================================================
with tab_vector:
    st.markdown("## 🤖 Vector OS Insight Layer — Rules-Based Narrative")
    st.caption("Narrative derived from current computed metrics (no external LLM calls).")

    ws_df_local = ws_df if ws_df is not None else pd.DataFrame()
    if ws_df_local is None or ws_df_local.empty:
        try:
            ws_df_local = compute_wavescore_table(all_waves, mode, 365)
        except Exception:
            ws_df_local = pd.DataFrame()

    h = compute_wave_history(selected_wave, mode, 365)

    if ws_df_local.empty or h.empty or len(h) < 2:
        st.info("Not enough data yet for a full Vector OS insight on this Wave.")
    else:
        row = ws_df_local[ws_df_local["Wave"] == selected_wave]
        row = row.iloc[0] if not row.empty else None

        nav_w = h["wave_nav"]
        nav_b = h["bm_nav"]
        ret_w = h["wave_ret"]
        ret_b = h["bm_ret"]

        r365w = ret_from_nav(nav_w, len(nav_w))
        r365b = ret_from_nav(nav_b, len(nav_b))
        a365 = r365w - r365b

        vol_w = annualized_vol(ret_w)
        vol_b = annualized_vol(ret_b)
        mdd_w = max_drawdown(nav_w)
        mdd_b = max_drawdown(nav_b)

        question = st.text_input("Ask Vector about this Wave or the lineup:", "")

        st.markdown(f"### Vector’s Insight — {selected_wave}")
        if row is not None:
            st.write(f"- **WaveScore (proto)**: **{float(row['WaveScore']):.1f}/100** (**{row['Grade']}**).")
            st.write(f"- **365D alpha**: {fmt_pct(float(row.get('Alpha_365D', np.nan)))} · **IR**: {fmt_num(float(row.get('IR_365D', np.nan)), 2)}")
        st.write(f"- **365D return**: {fmt_pct(r365w)} vs benchmark {fmt_pct(r365b)} (alpha: {fmt_pct(a365)}).")
        st.write(f"- **Volatility (365D)**: Wave {fmt_pct(vol_w)} vs benchmark {fmt_pct(vol_b)}.")
        st.write(f"- **Max drawdown (365D)**: Wave {fmt_pct(mdd_w)} vs benchmark {fmt_pct(mdd_b)}.")

        st.markdown("#### Vector Read")
        if pd.notna(a365) and a365 >= 0.05:
            st.success("Vector: Strong risk-adjusted outperformance signal on the 365D window.")
        elif pd.notna(a365) and a365 >= 0.0:
            st.info("Vector: Mild outperformance; watch TE/vol and benchmark difficulty.")
        else:
            st.warning("Vector: Underperformance vs benchmark; check benchmark mix, TE, and SmartSafe drag.")

        if pd.notna(vol_w) and pd.notna(vol_b) and vol_b > 0 and (vol_w / vol_b) > 1.5:
            st.write("- Volatility is elevated vs benchmark: returns may look ‘spiky’ and alpha may swing.")
        if pd.notna(mdd_w) and mdd_w < -0.25:
            st.write("- Drawdown is deep: consider stronger SmartSafe posture in stress regimes (What-If Lab).")

        if question.strip():
            st.caption(f"Vector registered your prompt: “{question.strip()}” (rules-based insight only).")


# ============================================================
# FOOTER
# ============================================================
st.markdown("---")
st.caption("WAVES Intelligence™ • Institutional Console (IRB-1) • Demo-safe analytics • Baseline engine results unchanged")