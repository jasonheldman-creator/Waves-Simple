# app.py — WAVES Intelligence™ Institutional Console (Vector OS Edition)
# FULL FILE (no patches)
#
# Adds (NEW, non-destructive):
#   1) Benchmark Truth Panel (composition + difficulty + basic diagnostics)
#   2) Mode Separation Proof Panel (shows outputs + key mode levers from engine globals)
#   3) Wave Doctor™ (diagnostics + recommendations) + What-If Lab (shadow sim; baseline unchanged)
#
# Notes:
#   • Does NOT modify your engine math or baseline results.
#   • What-If Lab is explicitly labeled “shadow simulation”.
#   • Keeps mobile-friendly expanders + Streamlit-safe patterns.

from __future__ import annotations

import math
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

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


# ------------------------------------------------------------
# Streamlit config
# ------------------------------------------------------------
st.set_page_config(
    page_title="WAVES Intelligence™ Console",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ------------------------------------------------------------
# Helpers: formatting
# ------------------------------------------------------------
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


# ------------------------------------------------------------
# Helpers: data fetching & caching
# ------------------------------------------------------------
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


# ------------------------------------------------------------
# Core metrics
# ------------------------------------------------------------
def compute_return_from_nav(nav: pd.Series, window: int) -> float:
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
    ret_wave = compute_return_from_nav(nav_wave, window=len(nav_wave))
    ret_bm = compute_return_from_nav(nav_bm, window=len(nav_bm))
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


# ------------------------------------------------------------
# Alpha Attribution (Static Basket vs Engine)
# ------------------------------------------------------------
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
    """Fixed-weight NAV (no overlays) using daily prices."""
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
    """
    Practical, defensible attribution:
      - Engine wave return
      - Static basket return (same holdings, fixed weights, no overlays)
      - Overlay contribution = Engine - Static
      - Alpha vs composite benchmark
      - Alpha vs SPY (context)
      - Benchmark difficulty = (Benchmark - SPY)
      - Active intensity proxies (TE, IR)
    """
    out: Dict[str, float] = {}

    hist = compute_wave_history(wave_name, mode=mode, days=days)
    if hist.empty or len(hist) < 2:
        return out

    nav_wave = hist["wave_nav"]
    nav_bm = hist["bm_nav"]
    wave_ret = hist["wave_ret"]
    bm_ret = hist["bm_ret"]

    eng_ret = compute_return_from_nav(nav_wave, window=len(nav_wave))
    bm_ret_total = compute_return_from_nav(nav_bm, window=len(nav_bm))
    alpha_vs_bm = eng_ret - bm_ret_total

    hold_df = get_wave_holdings(wave_name)
    hold_w = _weights_from_df(hold_df, ticker_col="Ticker", weight_col="Weight")
    static_nav = compute_static_nav_from_weights(hold_w, days=days)
    static_ret = compute_return_from_nav(static_nav, window=len(static_nav)) if len(static_nav) >= 2 else float("nan")
    overlay_pp = (eng_ret - static_ret) if (pd.notna(eng_ret) and pd.notna(static_ret)) else float("nan")

    spy_px = fetch_prices_daily(["SPY"], days=days)
    spy_nav = (spy_px["SPY"].pct_change().fillna(0.0) + 1.0).cumprod() if "SPY" in spy_px.columns else pd.Series(dtype=float)
    spy_ret = compute_return_from_nav(spy_nav, window=len(spy_nav)) if len(spy_nav) >= 2 else float("nan")

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


# ------------------------------------------------------------
# WaveScore proto v1.0 (kept; console-side)
# ------------------------------------------------------------
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


def compute_wavescore_for_all_waves(all_waves: list[str], mode: str, days: int = 365) -> pd.DataFrame:
    rows = []
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
                }
            )
            continue

        nav_wave = hist["wave_nav"]
        nav_bm = hist["bm_nav"]
        wave_ret = hist["wave_ret"]
        bm_ret = hist["bm_ret"]

        ret_365_wave = compute_return_from_nav(nav_wave, window=len(nav_wave))
        ret_365_bm = compute_return_from_nav(nav_bm, window=len(nav_bm))
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
            rec_part = np.clip(recovery_frac, 0.0, 1.0) * 6.0
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


# ------------------------------------------------------------
# NEW: Benchmark Truth utilities
# ------------------------------------------------------------
@st.cache_data(show_spinner=False)
def compute_spy_nav(days: int = 365) -> pd.Series:
    px = fetch_prices_daily(["SPY"], days=days)
    if px.empty or "SPY" not in px.columns:
        return pd.Series(dtype=float)
    nav = (1.0 + px["SPY"].pct_change().fillna(0.0)).cumprod()
    nav.name = "spy_nav"
    return nav


def benchmark_source_label(wave_name: str, bm_mix_df: pd.DataFrame) -> str:
    """Heuristic label: if engine exposes static dict and wave exists, mark Static Override else Auto-Composite."""
    try:
        static_dict = getattr(we, "BENCHMARK_WEIGHTS_STATIC", None)
        if isinstance(static_dict, dict) and wave_name in static_dict and static_dict.get(wave_name):
            return "Static Override (Engine)"
    except Exception:
        pass
    # If not static override, assume auto composite when mix exists
    if bm_mix_df is not None and not bm_mix_df.empty:
        return "Auto-Composite (Dynamic)"
    return "Unknown"


# ------------------------------------------------------------
# NEW: Wave Doctor (diagnostics + suggestions)
# ------------------------------------------------------------
def wave_doctor_assess(
    wave_name: str,
    mode: str,
    days: int = 365,
    alpha_warn: float = 0.08,
    te_warn: float = 0.20,
    vol_warn: float = 0.30,
    mdd_warn: float = -0.25,
) -> Dict[str, object]:
    """
    Produces:
      • summary metrics
      • issue flags
      • plain-English diagnosis
      • non-destructive recommended adjustments
    """
    hist = compute_wave_history(wave_name, mode=mode, days=days)
    if hist.empty or len(hist) < 2:
        return {"ok": False, "message": "Not enough data to run Wave Doctor."}

    nav_w = hist["wave_nav"]
    nav_b = hist["bm_nav"]
    ret_w = hist["wave_ret"]
    ret_b = hist["bm_ret"]

    r365_w = compute_return_from_nav(nav_w, len(nav_w))
    r365_b = compute_return_from_nav(nav_b, len(nav_b))
    a365 = r365_w - r365_b

    r30_w = compute_return_from_nav(nav_w, min(30, len(nav_w)))
    r30_b = compute_return_from_nav(nav_b, min(30, len(nav_b)))
    a30 = r30_w - r30_b

    vol_w = annualized_vol(ret_w)
    vol_b = annualized_vol(ret_b)
    te = tracking_error(ret_w, ret_b)
    ir = information_ratio(nav_w, nav_b, te)
    mdd_w = max_drawdown(nav_w)
    mdd_b = max_drawdown(nav_b)

    # benchmark difficulty vs SPY
    spy_nav = compute_spy_nav(days=days)
    spy_ret = compute_return_from_nav(spy_nav, len(spy_nav)) if len(spy_nav) >= 2 else float("nan")
    bm_difficulty = (r365_b - spy_ret) if (pd.notna(r365_b) and pd.notna(spy_ret)) else float("nan")

    flags = []
    diagnosis = []
    recs = []

    # Alpha flags
    if pd.notna(a30) and abs(a30) >= alpha_warn:
        flags.append("Large 30D alpha (verify benchmark + coverage)")
        diagnosis.append("30D alpha is unusually large. This can happen from real signal, but also from benchmark mix or data coverage shifts.")
        recs.append("Consider freezing benchmark for demo comparisons and check benchmark mix drift.")

    if pd.notna(a365) and a365 < 0:
        diagnosis.append("365D alpha is negative. This might reflect true underperformance, or a tougher benchmark composition.")
        if pd.notna(bm_difficulty) and bm_difficulty > 0.03:
            diagnosis.append("Benchmark difficulty is positive vs SPY, meaning your benchmark outperformed SPY over the window—alpha is harder here.")
            recs.append("If your intent is broad-market comparison, temporarily switch/freeze benchmark to SPY/QQQ style mix for validation.")
    else:
        if pd.notna(a365) and a365 > 0.06:
            diagnosis.append("365D alpha is solidly positive. This aligns with the ‘green’ outcome you were looking for.")
            recs.append("Lock benchmark mix (session snapshot) for reproducibility in demos.")

    # Risk flags
    if pd.notna(te) and te > te_warn:
        flags.append("High tracking error (active risk elevated)")
        diagnosis.append("Tracking error is high; the wave behaves very differently than its benchmark.")
        recs.append("Wave Doctor suggestion: reduce momentum tilt strength and/or tighten exposure caps to reduce TE.")

    if pd.notna(vol_w) and vol_w > vol_warn:
        flags.append("High volatility")
        diagnosis.append("Volatility is elevated. If this wave is intended to be ‘disciplined’, you may want to lower the vol target.")
        recs.append("Wave Doctor suggestion: lower vol target (e.g., 20% → 16–18%).")

    if pd.notna(mdd_w) and mdd_w < mdd_warn:
        flags.append("Deep drawdown")
        diagnosis.append("Drawdown is deep versus typical institutional tolerances.")
        recs.append("Wave Doctor suggestion: increase SmartSafe boost in downtrend/panic or tighten max exposure cap.")

    # Benchmark sanity
    if pd.notna(vol_b) and pd.notna(vol_w) and vol_b > 0 and vol_w / vol_b > 1.6:
        flags.append("Volatility much higher than benchmark")
        diagnosis.append("Wave volatility is much higher than benchmark; this can inflate both wins and losses.")
        recs.append("Wave Doctor suggestion: tighten exposure cap and reduce tilt strength.")

    # Minimal fallback if no messages
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
        "recommendations": list(dict.fromkeys(recs)),  # de-dupe preserve order
    }


# ------------------------------------------------------------
# NEW: What-If Lab (shadow sim)
# ------------------------------------------------------------
def _regime_from_spy_60d(spy_nav: pd.Series) -> pd.Series:
    spy_nav = safe_series(spy_nav)
    if spy_nav.empty:
        return pd.Series(dtype=str)
    r60 = spy_nav / spy_nav.shift(60) - 1.0
    # Mirror the engine thresholds (best-effort)
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
    """
    Shadow sim intended for diagnostics only.
    Uses:
      • Wave holdings (fixed universe)
      • SPY regime + VIX gating
      • momentum tilt (60D), vol targeting (20D)
      • mode base exposure if available from engine
    """
    # Inputs
    hold_df = get_wave_holdings(wave_name)
    weights = _weights_from_df(hold_df, "Ticker", "Weight")
    if weights.empty:
        return pd.DataFrame()

    # Tickers for shadow sim
    tickers = list(weights.index)
    needed = set(tickers + ["SPY", "^VIX", "SGOV", "BIL", "SHY"])
    px = fetch_prices_daily(list(needed), days=days)
    if px.empty or "SPY" not in px.columns or "^VIX" not in px.columns:
        return pd.DataFrame()

    px = px.sort_index().ffill().bfill()
    if len(px) > days:
        px = px.iloc[-days:]

    rets = px.pct_change().fillna(0.0)

    # Align holdings
    w = weights.reindex(px.columns).fillna(0.0)

    # Regime and VIX levers
    spy_nav = (1.0 + rets["SPY"]).cumprod()
    regime = _regime_from_spy_60d(spy_nav)
    vix = px["^VIX"]

    vix_exposure = _vix_exposure_factor_series(vix, mode)
    vix_safe = _vix_safe_fraction_series(vix, mode)

    # Try to borrow engine base exposure if present
    base_expo = 1.0
    try:
        base_map = getattr(we, "MODE_BASE_EXPOSURE", None)
        if isinstance(base_map, dict) and mode in base_map:
            base_expo = float(base_map[mode])
    except Exception:
        pass

    # Regime exposure map (best-effort; fall back to engine if available)
    regime_exposure_map = {"panic": 0.80, "downtrend": 0.90, "neutral": 1.00, "uptrend": 1.10}
    try:
        rm = getattr(we, "REGIME_EXPOSURE", None)
        if isinstance(rm, dict):
            regime_exposure_map = {k: float(v) for k, v in rm.items()}
    except Exception:
        pass

    # Regime gating map (best-effort)
    def regime_gate(mode_in: str, reg: str) -> float:
        try:
            rg = getattr(we, "REGIME_GATING", None)
            if isinstance(rg, dict) and mode_in in rg and reg in rg[mode_in]:
                return float(rg[mode_in][reg])
        except Exception:
            pass
        # fallback rough
        fallback = {
            "Standard": {"panic": 0.50, "downtrend": 0.30, "neutral": 0.10, "uptrend": 0.00},
            "Alpha-Minus-Beta": {"panic": 0.75, "downtrend": 0.50, "neutral": 0.25, "uptrend": 0.05},
            "Private Logic": {"panic": 0.40, "downtrend": 0.25, "neutral": 0.05, "uptrend": 0.00},
        }
        return float(fallback.get(mode_in, fallback["Standard"]).get(reg, 0.10))

    # Safe ticker selection
    safe_ticker = "SGOV" if "SGOV" in rets.columns else ("BIL" if "BIL" in rets.columns else ("SHY" if "SHY" in rets.columns else "SPY"))
    safe_ret = rets[safe_ticker]

    # momentum (60D) and realized vol (20D)
    mom60 = px / px.shift(60) - 1.0

    wave_ret = []
    dates = []

    # For vol targeting, we need trailing realized vol of wave returns
    for dt in rets.index:
        r = rets.loc[dt]

        # Momentum tilt on holdings only
        mom_row = mom60.loc[dt] if dt in mom60.index else None
        if mom_row is not None:
            mom_clipped = mom_row.reindex(px.columns).fillna(0.0).clip(lower=-0.30, upper=0.30)
            tilt = 1.0 + tilt_strength * mom_clipped
            ew = (w * tilt).clip(lower=0.0)
        else:
            ew = w.copy()

        # Normalize risk weights on holdings universe
        # Ensure we only allocate to holdings tickers
        ew_hold = ew.reindex(tickers).fillna(0.0)
        s = float(ew_hold.sum())
        if s > 0:
            rw = ew_hold / s
        else:
            rw = w.reindex(tickers).fillna(0.0)
            s2 = float(rw.sum())
            rw = (rw / s2) if s2 > 0 else rw

        port_risk_ret = float((r.reindex(tickers).fillna(0.0) * rw).sum())

        # realized vol (20D) on shadow wave returns
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

        # Private Logic shock dampener (best-effort)
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

    # Benchmark (for what-if) — either freeze to engine benchmark or SPY
    if freeze_benchmark:
        # Use engine benchmark NAV for alignment (baseline benchmark)
        hist_eng = compute_wave_history(wave_name, mode=mode, days=days)
        if not hist_eng.empty and "bm_nav" in hist_eng.columns:
            bm_nav = hist_eng["bm_nav"].reindex(out.index).ffill().bfill()
            bm_ret = hist_eng["bm_ret"].reindex(out.index).fillna(0.0)
            out["bm_nav"] = bm_nav
            out["bm_ret"] = bm_ret
    else:
        # SPY as a neutral comparator
        if "SPY" in px.columns:
            spy_ret = rets["SPY"].reindex(out.index).fillna(0.0)
            spy_nav = (1.0 + spy_ret).cumprod()
            out["bm_nav"] = spy_nav
            out["bm_ret"] = spy_ret

    return out


# ------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------
all_waves = we.get_all_waves()
all_modes = we.get_modes()

with st.sidebar:
    st.title("WAVES Intelligence™")
    st.caption("Mini Bloomberg Console • Vector OS™")

    mode = st.selectbox("Mode", all_modes, index=0)
    selected_wave = st.selectbox("Select Wave", all_waves, index=0)

    st.markdown("---")
    st.markdown("**Display settings**")
    nav_days = st.slider("History window (days)", min_value=60, max_value=730, value=365, step=15)

    st.markdown("---")
    st.markdown("**Wave Doctor settings**")
    alpha_warn = st.slider("Alpha alert threshold (abs, 30D)", 0.02, 0.20, 0.08, 0.01)
    te_warn = st.slider("Tracking error alert threshold", 0.05, 0.40, 0.20, 0.01)


# ------------------------------------------------------------
# Page header + tabs
# ------------------------------------------------------------
st.title("WAVES Intelligence™ Institutional Console")
st.caption("Live Alpha Capture • SmartSafe™ • Multi-Asset • Crypto • Gold • Income Ladders")

tab_console, tab_market, tab_factors, tab_vector = st.tabs(
    ["Console", "Market Intel", "Factor Decomposition", "Vector OS Insight Layer"]
)


# ============================================================
# TAB 1: Console
# ============================================================
with tab_console:
    st.subheader("Market Regime Monitor — SPY vs VIX")
    spy_vix = fetch_spy_vix(days=nav_days)

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

    # ------------------------------------------------------------
    # Portfolio-Level Overview (All Waves)
    # ------------------------------------------------------------
    st.subheader("All Waves Overview (1D / 30D / 60D / 365D Returns + Alpha)")

    overview_rows = []
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
        if len(nav_w) >= 2:
            r1w = float(nav_w.iloc[-1] / nav_w.iloc[-2] - 1.0)
            r1b = float(nav_b.iloc[-1] / nav_b.iloc[-2] - 1.0)
            a1 = r1w - r1b
        else:
            r1w = r1b = a1 = np.nan

        # windows
        r30w = compute_return_from_nav(nav_w, min(30, len(nav_w)))
        r30b = compute_return_from_nav(nav_b, min(30, len(nav_b)))
        a30 = r30w - r30b

        r60w = compute_return_from_nav(nav_w, min(60, len(nav_w)))
        r60b = compute_return_from_nav(nav_b, min(60, len(nav_b)))
        a60 = r60w - r60b

        r365w = compute_return_from_nav(nav_w, len(nav_w))
        r365b = compute_return_from_nav(nav_b, len(nav_b))
        a365 = r365w - r365b

        overview_rows.append(
            {"Wave": wname, "1D Ret": r1w, "1D Alpha": a1,
             "30D Ret": r30w, "30D Alpha": a30,
             "60D Ret": r60w, "60D Alpha": a60,
             "365D Ret": r365w, "365D Alpha": a365}
        )

    overview_df = pd.DataFrame(overview_rows)
    fmt_overview = overview_df.copy()
    for col in ["1D Ret", "1D Alpha", "30D Ret", "30D Alpha", "60D Ret", "60D Alpha", "365D Ret", "365D Alpha"]:
        fmt_overview[col] = fmt_overview[col].apply(lambda x: fmt_pct(x))
    st.dataframe(fmt_overview.set_index("Wave"), use_container_width=True)

    st.markdown("---")

    # ------------------------------------------------------------
    # Risk analytics + WaveScore (All Waves)
    # ------------------------------------------------------------
    st.subheader("Risk Analytics (Vol, MaxDD, TE, IR) — 365D Window")

    risk_rows = []
    for wname in all_waves:
        hist = compute_wave_history(wname, mode=mode, days=365)
        if hist.empty or len(hist) < 2:
            risk_rows.append(
                {"Wave": wname, "Wave Vol": np.nan, "Benchmark Vol": np.nan, "Wave MaxDD": np.nan,
                 "Benchmark MaxDD": np.nan, "Tracking Error": np.nan, "Information Ratio": np.nan}
            )
            continue
        nav_w = hist["wave_nav"]
        nav_b = hist["bm_nav"]
        ret_w = hist["wave_ret"]
        ret_b = hist["bm_ret"]

        risk_rows.append(
            {"Wave": wname,
             "Wave Vol": annualized_vol(ret_w),
             "Benchmark Vol": annualized_vol(ret_b),
             "Wave MaxDD": max_drawdown(nav_w),
             "Benchmark MaxDD": max_drawdown(nav_b),
             "Tracking Error": tracking_error(ret_w, ret_b),
             "Information Ratio": information_ratio(nav_w, nav_b, tracking_error(ret_w, ret_b))}
        )

    risk_df = pd.DataFrame(risk_rows)
    fmt_risk = risk_df.copy()
    for col in ["Wave Vol", "Benchmark Vol", "Tracking Error", "Wave MaxDD", "Benchmark MaxDD"]:
        fmt_risk[col] = fmt_risk[col].apply(lambda x: fmt_pct(x))
    fmt_risk["Information Ratio"] = fmt_risk["Information Ratio"].apply(lambda x: fmt_num(x, 2))
    st.dataframe(fmt_risk.set_index("Wave"), use_container_width=True)

    st.markdown("---")

    st.subheader("WaveScore™ Leaderboard (Proto v1.0 · 365D Data)")
    wavescore_df = compute_wavescore_for_all_waves(all_waves, mode=mode, days=365)
    if wavescore_df.empty:
        st.info("No WaveScore data available yet.")
    else:
        fmt_ws = wavescore_df.copy()
        fmt_ws["WaveScore"] = fmt_ws["WaveScore"].apply(fmt_score)
        for col in ["Return Quality", "Risk Control", "Consistency", "Resilience", "Efficiency"]:
            fmt_ws[col] = fmt_ws[col].apply(lambda x: fmt_num(x, 1))
        fmt_ws["Alpha_365D"] = fmt_ws["Alpha_365D"].apply(fmt_pct)
        fmt_ws["IR_365D"] = fmt_ws["IR_365D"].apply(lambda x: fmt_num(x, 2))
        st.dataframe(fmt_ws.set_index("Wave"), use_container_width=True)

    st.markdown("---")

    # ------------------------------------------------------------
    # Benchmark transparency table (All Waves)
    # ------------------------------------------------------------
    st.subheader("Benchmark Transparency Table (Composite Benchmark Components per Wave)")
    bm_mix = get_benchmark_mix()
    if bm_mix.empty:
        st.info("No benchmark mix data available.")
    else:
        fmt_bm = bm_mix.copy()
        fmt_bm["Weight"] = fmt_bm["Weight"].apply(lambda x: fmt_pct(x))
        st.dataframe(fmt_bm, use_container_width=True)

    st.markdown("---")

    # ------------------------------------------------------------
    # Wave Detail View
    # ------------------------------------------------------------
    st.subheader(f"Wave Detail — {selected_wave} (Mode: {mode})")
    col_chart, col_stats = st.columns([2.0, 1.0])

    hist_sel = compute_wave_history(selected_wave, mode=mode, days=nav_days)

    with col_chart:
        if hist_sel.empty or len(hist_sel) < 2:
            st.warning("Not enough data to display NAV chart.")
        else:
            nav_w = hist_sel["wave_nav"]
            nav_b = hist_sel["bm_nav"]

            if go is not None:
                fig_nav = go.Figure()
                fig_nav.add_trace(go.Scatter(x=hist_sel.index, y=nav_w, name=f"{selected_wave} NAV", mode="lines"))
                fig_nav.add_trace(go.Scatter(x=hist_sel.index, y=nav_b, name="Benchmark NAV", mode="lines"))
                fig_nav.update_layout(
                    margin=dict(l=40, r=40, t=40, b=40),
                    xaxis=dict(title="Date"),
                    yaxis=dict(title="NAV (Normalized)"),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    height=360,
                )
                st.plotly_chart(fig_nav, use_container_width=True)
            else:
                st.line_chart(pd.DataFrame({"wave_nav": nav_w, "bm_nav": nav_b}))

    with col_stats:
        if hist_sel.empty or len(hist_sel) < 2:
            st.info("No stats available.")
        else:
            nav_w = hist_sel["wave_nav"]
            nav_b = hist_sel["bm_nav"]
            ret_w = hist_sel["wave_ret"]
            ret_b = hist_sel["bm_ret"]

            r30w = compute_return_from_nav(nav_w, min(30, len(nav_w)))
            r30b = compute_return_from_nav(nav_b, min(30, len(nav_b)))
            a30 = r30w - r30b

            r365w = compute_return_from_nav(nav_w, len(nav_w))
            r365b = compute_return_from_nav(nav_b, len(nav_b))
            a365 = r365w - r365b

            st.markdown("**Performance vs Benchmark**")
            st.metric("30D Return", fmt_pct(r30w))
            st.metric("30D Alpha", fmt_pct(a30))
            st.metric("365D Return", fmt_pct(r365w))
            st.metric("365D Alpha", fmt_pct(a365))

            st.markdown("---")
            st.markdown("**Risk (365D)**")
            st.metric("Wave Vol", fmt_pct(annualized_vol(ret_w)))
            st.metric("Tracking Error", fmt_pct(tracking_error(ret_w, ret_b)))
            st.metric("IR", fmt_num(information_ratio(nav_w, nav_b, tracking_error(ret_w, ret_b)), 2))
            st.metric("MaxDD", fmt_pct(max_drawdown(nav_w)))

    # ------------------------------------------------------------
    # NEW: Benchmark Truth Panel
    # ------------------------------------------------------------
    with st.expander("✅ Benchmark Truth Panel (composition + difficulty + diagnostics)", expanded=True):
        bm_mix_df = get_benchmark_mix()
        wave_bm = bm_mix_df[bm_mix_df["Wave"] == selected_wave].copy() if not bm_mix_df.empty else pd.DataFrame()

        src_label = benchmark_source_label(selected_wave, wave_bm)

        cA, cB, cC, cD = st.columns(4)
        cA.metric("Benchmark Source", src_label)
        if hist_sel.empty or len(hist_sel) < 2:
            cB.metric("Benchmark Return (365D)", "—")
            cC.metric("SPY Return (365D)", "—")
            cD.metric("BM Difficulty (BM−SPY)", "—")
        else:
            bm_nav = hist_sel["bm_nav"]
            bm_ret_total = compute_return_from_nav(bm_nav, len(bm_nav))
            spy_nav = compute_spy_nav(days=365)
            spy_ret = compute_return_from_nav(spy_nav, len(spy_nav)) if len(spy_nav) >= 2 else float("nan")
            diff = bm_ret_total - spy_ret if (pd.notna(bm_ret_total) and pd.notna(spy_ret)) else float("nan")
            cB.metric("Benchmark Return (365D)", fmt_pct(bm_ret_total))
            cC.metric("SPY Return (365D)", fmt_pct(spy_ret))
            cD.metric("BM Difficulty (BM−SPY)", fmt_pct(diff))

        st.markdown("### Benchmark Composition (as used in engine benchmark mix)")
        if wave_bm.empty:
            st.warning("No benchmark components found in mix table for this Wave.")
        else:
            wave_bm = wave_bm.sort_values("Weight", ascending=False)
            fmt = wave_bm.copy()
            fmt["Weight"] = fmt["Weight"].apply(lambda x: fmt_pct(x))
            st.dataframe(fmt[["Ticker", "Name", "Weight"]], use_container_width=True)

        if not hist_sel.empty and len(hist_sel) >= 2:
            bm_ret = hist_sel["bm_ret"]
            bm_vol = annualized_vol(bm_ret)
            bm_mdd = max_drawdown(hist_sel["bm_nav"])
            st.markdown("### Benchmark Risk Diagnostics (365D window)")
            r1, r2, r3 = st.columns(3)
            r1.metric("Benchmark Vol", fmt_pct(bm_vol))
            r2.metric("Benchmark MaxDD", fmt_pct(bm_mdd))
            r3.metric("Days in Window", str(int(len(hist_sel))))

    # ------------------------------------------------------------
    # NEW: Mode Separation Proof Panel
    # ------------------------------------------------------------
    with st.expander("✅ Mode Separation Proof (outputs + mode levers)", expanded=False):
        rows = []
        for m in all_modes:
            h = compute_wave_history(selected_wave, mode=m, days=365)
            if h.empty or len(h) < 2:
                rows.append(
                    {"Mode": m, "Base Exposure": np.nan, "Exposure Caps": "—",
                     "365D Return": np.nan, "365D Alpha": np.nan, "TE": np.nan, "IR": np.nan}
                )
                continue

            nav_w = h["wave_nav"]
            nav_b = h["bm_nav"]
            ret_w = h["wave_ret"]
            ret_b = h["bm_ret"]

            r365w = compute_return_from_nav(nav_w, len(nav_w))
            r365b = compute_return_from_nav(nav_b, len(nav_b))
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
                    caps = f"{lo:.2f}–{hi:.2f}"
            except Exception:
                pass

            rows.append(
                {"Mode": m, "Base Exposure": base_expo, "Exposure Caps": caps,
                 "365D Return": r365w, "365D Alpha": a365, "TE": te, "IR": ir}
            )

        dfm = pd.DataFrame(rows)
        fmtm = dfm.copy()
        fmtm["Base Exposure"] = fmtm["Base Exposure"].apply(lambda x: fmt_num(x, 2))
        for col in ["365D Return", "365D Alpha"]:
            fmtm[col] = fmtm[col].apply(fmt_pct)
        fmtm["TE"] = fmtm["TE"].apply(fmt_pct)
        fmtm["IR"] = fmtm["IR"].apply(lambda x: fmt_num(x, 2))
        st.dataframe(fmtm.set_index("Mode"), use_container_width=True)

        st.caption(
            "This panel is the demo-safe proof that mode toggles are not cosmetic. "
            "It shows both the engine’s lever settings (when available) and the realized output differences."
        )

    # ------------------------------------------------------------
    # Alpha Attribution (kept)
    # ------------------------------------------------------------
    with st.expander("Benchmark Transparency + Alpha Attribution (365D)", expanded=False):
        st.caption(
            "Attribution compares the engine’s dynamic Wave NAV vs a static fixed-weight basket of the same holdings "
            "(no SmartSafe/vol/VIX overlays). The difference is the engine overlay contribution."
        )
        attrib = compute_alpha_attribution(selected_wave, mode=mode, days=365)
        if not attrib:
            st.warning("Not enough data to compute attribution yet for this Wave.")
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

            st.markdown("#### Risk Context (365D)")
            r1, r2, r3, r4 = st.columns(4)
            r1.metric("Wave Vol", fmt_pct(attrib.get("Wave Vol", np.nan)))
            r2.metric("Benchmark Vol", fmt_pct(attrib.get("Benchmark Vol", np.nan)))
            r3.metric("Wave MaxDD", fmt_pct(attrib.get("Wave MaxDD", np.nan)))
            r4.metric("Benchmark MaxDD", fmt_pct(attrib.get("Benchmark MaxDD", np.nan)))

            if go is not None:
                base = attrib.get("Static Basket Return", float("nan"))
                overlay = attrib.get("Overlay Contribution (Engine - Static)", float("nan"))
                if pd.notna(base) and pd.notna(overlay):
                    fig_attr = go.Figure(data=[go.Bar(x=["Static Basket", "Engine Overlay"], y=[base, overlay])])
                    fig_attr.update_layout(
                        title="365D Return Components (Static Basket + Engine Overlay = Engine Return)",
                        xaxis_title="Component",
                        yaxis_title="Return",
                        height=320,
                        margin=dict(l=40, r=40, t=60, b=40),
                    )
                    st.plotly_chart(fig_attr, use_container_width=True)

    # ------------------------------------------------------------
    # NEW: Wave Doctor™ + What-If Lab
    # ------------------------------------------------------------
    with st.expander("✅ Wave Doctor™ (diagnostics + recommendations) + What-If Lab", expanded=True):
        assess = wave_doctor_assess(
            selected_wave, mode=mode, days=365,
            alpha_warn=float(alpha_warn),
            te_warn=float(te_warn),
        )

        if not assess.get("ok", False):
            st.info(assess.get("message", "Wave Doctor unavailable."))
        else:
            metrics = assess["metrics"]
            flags = assess["flags"]
            diagnosis = assess["diagnosis"]
            recs = assess["recommendations"]

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("365D Return", fmt_pct(metrics["Return_365D"]))
            c2.metric("365D Alpha", fmt_pct(metrics["Alpha_365D"]))
            c3.metric("TE", fmt_pct(metrics["TE"]))
            c4.metric("IR", fmt_num(metrics["IR"], 2))

            c5, c6, c7, c8 = st.columns(4)
            c5.metric("Wave Vol", fmt_pct(metrics["Vol_Wave"]))
            c6.metric("Wave MaxDD", fmt_pct(metrics["MaxDD_Wave"]))
            c7.metric("BM Difficulty (BM−SPY)", fmt_pct(metrics["Benchmark_Difficulty_BM_minus_SPY"]))
            c8.metric("30D Alpha", fmt_pct(metrics["Alpha_30D"]))

            if flags:
                st.warning("Wave Doctor Flags: " + " • ".join(flags))

            st.markdown("### What might be going on (plain English)")
            for line in diagnosis:
                st.write(f"- {line}")

            st.markdown("### Recommended adjustments (suggestions only — baseline unchanged)")
            if recs:
                for r in recs:
                    st.write(f"- {r}")
            else:
                st.write("- No changes recommended based on current thresholds.")

            st.markdown("---")
            st.markdown("## What-If Lab (Shadow Simulation — does NOT change official results)")
            st.caption(
                "Use sliders to test hypothetical parameter changes. "
                "This computes a **shadow NAV** for insight only and is not the engine’s official audited NAV."
            )

            # Defaults informed by engine, best-effort
            default_tilt = 0.80
            default_vol = 0.20
            default_extra_safe = 0.00
            default_exp_min, default_exp_max = 0.70, 1.30
            try:
                if selected_wave == "Demas Fund Wave":
                    default_tilt = 0.45
                    default_vol = 0.15
                    default_extra_safe = 0.03
            except Exception:
                pass

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

            # Run shadow sim
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

            # Baseline series (engine official)
            baseline = compute_wave_history(selected_wave, mode=mode, days=365)

            if whatif.empty or baseline.empty or len(baseline) < 2:
                st.info("What-If results unavailable (missing price inputs or insufficient data).")
            else:
                # Align indices
                b_nav = baseline["wave_nav"].reindex(whatif.index).ffill().bfill()
                b_bm = baseline["bm_nav"].reindex(whatif.index).ffill().bfill()

                w_nav = whatif["whatif_nav"]
                w_bm = whatif["bm_nav"] if "bm_nav" in whatif.columns else b_bm

                b_ret = compute_return_from_nav(b_nav, len(b_nav))
                b_bret = compute_return_from_nav(b_bm, len(b_bm))
                b_alpha = b_ret - b_bret

                w_ret = compute_return_from_nav(w_nav, len(w_nav))
                w_bret = compute_return_from_nav(w_bm, len(w_bm))
                w_alpha = w_ret - w_bret

                d1, d2, d3, d4 = st.columns(4)
                d1.metric("Baseline 365D Return", fmt_pct(b_ret))
                d2.metric("What-If 365D Return", fmt_pct(w_ret), delta=fmt_pct(w_ret - b_ret))
                d3.metric("Baseline 365D Alpha", fmt_pct(b_alpha))
                d4.metric("What-If 365D Alpha", fmt_pct(w_alpha), delta=fmt_pct(w_alpha - b_alpha))

                # Drawdown / vol comparison
                b_mdd = max_drawdown(b_nav)
                w_mdd = max_drawdown(w_nav)
                b_vol = annualized_vol(b_nav.pct_change().fillna(0.0))
                w_vol = annualized_vol(w_nav.pct_change().fillna(0.0))

                e1, e2, e3, e4 = st.columns(4)
                e1.metric("Baseline Vol", fmt_pct(b_vol))
                e2.metric("What-If Vol", fmt_pct(w_vol), delta=fmt_pct(w_vol - b_vol))
                e3.metric("Baseline MaxDD", fmt_pct(b_mdd))
                e4.metric("What-If MaxDD", fmt_pct(w_mdd), delta=fmt_pct(w_mdd - b_mdd))

                # Plot NAVs
                st.markdown("### Baseline vs What-If NAV (normalized)")
                plot_df = pd.DataFrame(
                    {
                        "Baseline_NAV": b_nav / float(b_nav.iloc[0]) if len(b_nav) else b_nav,
                        "WhatIf_NAV": w_nav / float(w_nav.iloc[0]) if len(w_nav) else w_nav,
                    }
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

    # ------------------------------------------------------------
    # Top-10 holdings with Google Finance links
    # ------------------------------------------------------------
    st.subheader("Top-10 Holdings (with Google Finance links)")
    holdings_df = get_wave_holdings(selected_wave)
    if holdings_df.empty:
        st.info("No holdings available for this Wave.")
    else:
        def google_link(ticker: str) -> str:
            base = "https://www.google.com/finance/quote"
            return f"[{ticker}]({base}/{ticker})"

        fmt_hold = holdings_df.copy()
        fmt_hold["Weight"] = fmt_hold["Weight"].apply(lambda x: fmt_pct(x))
        fmt_hold["Google Finance"] = fmt_hold["Ticker"].apply(google_link)
        st.dataframe(fmt_hold[["Ticker", "Name", "Weight", "Google Finance"]], use_container_width=True)

# ============================================================
# TAB 2: Market Intel
# ============================================================
with tab_market:
    st.subheader("Global Market Dashboard")
    market_df = fetch_market_assets(days=nav_days)

    if market_df.empty:
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

        snap_df = pd.DataFrame(rows)
        fmt_snap = snap_df.copy()
        fmt_snap["Last"] = fmt_snap["Last"].apply(lambda x: fmt_num(x, 2))
        fmt_snap["1D Return"] = fmt_snap["1D Return"].apply(fmt_pct)
        fmt_snap["30D Return"] = fmt_snap["30D Return"].apply(fmt_pct)
        st.dataframe(fmt_snap.set_index("Ticker"), use_container_width=True)

    st.markdown("---")
    st.subheader(f"WAVES Reaction Snapshot (30D · Mode = {mode})")

    reaction_rows = []
    for wname in all_waves:
        hist = compute_wave_history(wname, mode=mode, days=365)
        if hist.empty or len(hist) < 2:
            reaction_rows.append({"Wave": wname, "30D Return": np.nan, "30D Alpha": np.nan, "Classification": "No data"})
            continue

        nav_w = hist["wave_nav"]
        nav_b = hist["bm_nav"]

        r30w = compute_return_from_nav(nav_w, min(30, len(nav_w)))
        r30b = compute_return_from_nav(nav_b, min(30, len(nav_b)))
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

        reaction_rows.append({"Wave": wname, "30D Return": r30w, "30D Alpha": a30, "Classification": label})

    reaction_df = pd.DataFrame(reaction_rows)
    fmt_reaction = reaction_df.copy()
    fmt_reaction["30D Return"] = fmt_reaction["30D Return"].apply(fmt_pct)
    fmt_reaction["30D Alpha"] = fmt_reaction["30D Alpha"].apply(fmt_pct)
    st.dataframe(fmt_reaction.set_index("Wave"), use_container_width=True)

# ============================================================
# TAB 3: Factor Decomposition
# ============================================================
with tab_factors:
    st.subheader("Factor Decomposition (Institution-Level Analytics)")

    factor_days = min(nav_days, 365)
    factor_prices = fetch_market_assets(days=factor_days)

    needed = ["SPY", "QQQ", "IWM", "TLT", "GLD", "BTC-USD"]
    missing = [t for t in needed if t not in factor_prices.columns] if not factor_prices.empty else needed

    if factor_prices.empty or missing:
        st.warning("Unable to load all factor price series. " + (f"Missing: {', '.join(missing)}" if missing else ""))
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
        for wname in all_waves:
            hist = compute_wave_history(wname, mode=mode, days=factor_days)
            if hist.empty or "wave_ret" not in hist.columns:
                rows.append({"Wave": wname, "β_SPY": np.nan, "β_QQQ": np.nan, "β_IWM": np.nan, "β_TLT": np.nan, "β_GLD": np.nan, "β_BTC": np.nan})
                continue

            wret = hist["wave_ret"]
            betas = regress_factors(wave_ret=wret, factor_ret=factor_returns)

            rows.append(
                {
                    "Wave": wname,
                    "β_SPY": betas.get("MKT_SPY", np.nan),
                    "β_QQQ": betas.get("GROWTH_QQQ", np.nan),
                    "β_IWM": betas.get("SIZE_IWM", np.nan),
                    "β_TLT": betas.get("RATES_TLT", np.nan),
                    "β_GLD": betas.get("GOLD_GLD", np.nan),
                    "β_BTC": betas.get("CRYPTO_BTC", np.nan),
                }
            )

        beta_df = pd.DataFrame(rows)
        fmt_beta = beta_df.copy()
        for col in ["β_SPY", "β_QQQ", "β_IWM", "β_TLT", "β_GLD", "β_BTC"]:
            fmt_beta[col] = fmt_beta[col].apply(lambda x: fmt_num(x, 2))
        st.dataframe(fmt_beta.set_index("Wave"), use_container_width=True)

    st.markdown("---")
    st.subheader(f"Correlation Matrix — Waves (Daily Returns · Mode = {mode})")

    corr_days = min(nav_days, 365)
    ret_panel = {}
    for wname in all_waves:
        hist = compute_wave_history(wname, mode=mode, days=corr_days)
        if hist.empty or "wave_ret" not in hist.columns:
            continue
        ret_panel[wname] = hist["wave_ret"]

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
# TAB 4: Vector OS Insight Layer (rules-based narrative)
# ============================================================
with tab_vector:
    st.subheader("Vector OS Insight Layer — Rules-Based Narrative Panel")
    st.caption("Narrative derived from current computed metrics (no external LLM calls).")

    ws_df = compute_wavescore_for_all_waves(all_waves, mode=mode, days=365)
    ws_row = ws_df[ws_df["Wave"] == selected_wave] if not ws_df.empty else pd.DataFrame()
    hist = compute_wave_history(selected_wave, mode=mode, days=365)

    if ws_row.empty or hist.empty or len(hist) < 2:
        st.info("Not enough data yet for a full Vector OS insight on this Wave.")
    else:
        row = ws_row.iloc[0]
        nav_w = hist["wave_nav"]
        nav_b = hist["bm_nav"]
        ret_w = hist["wave_ret"]
        ret_b = hist["bm_ret"]

        r365w = compute_return_from_nav(nav_w, len(nav_w))
        r365b = compute_return_from_nav(nav_b, len(nav_b))
        a365 = r365w - r365b

        vol_w = annualized_vol(ret_w)
        vol_b = annualized_vol(ret_b)
        mdd_w = max_drawdown(nav_w)
        mdd_b = max_drawdown(nav_b)

        question = st.text_input("Ask Vector about this Wave or the lineup:", "")

        st.markdown(f"### Vector’s Insight — {selected_wave}")
        st.write(f"- **WaveScore (proto)**: **{row['WaveScore']:.1f}/100** (**{row['Grade']}**).")
        st.write(f"- **365D return**: {fmt_pct(r365w)} vs benchmark {fmt_pct(r365b)} (alpha: {fmt_pct(a365)}).")
        st.write(f"- **Volatility (365D)**: Wave {fmt_pct(vol_w)} vs benchmark {fmt_pct(vol_b)}.")
        st.write(f"- **Max drawdown (365D)**: Wave {fmt_pct(mdd_w)} vs benchmark {fmt_pct(mdd_b)}.")

        # Simple narrative rules
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
                    # Optional: quick lineup snapshot (Top/Bottom alpha) for context
        st.markdown("---")
        st.markdown("#### Lineup Snapshot (Mode-adjusted · 365D)")

        snap_rows = []
        for wname in all_waves:
            hh = compute_wave_history(wname, mode=mode, days=365)
            if hh.empty or len(hh) < 2:
                continue
            navw = hh["wave_nav"]
            navb = hh["bm_nav"]
            rw = compute_return_from_nav(navw, len(navw))
            rb = compute_return_from_nav(navb, len(navb))
            snap_rows.append({"Wave": wname, "365D Return": rw, "365D Alpha": (rw - rb)})

        if snap_rows:
            snap_df = pd.DataFrame(snap_rows).dropna()
            if not snap_df.empty:
                top_alpha = snap_df.sort_values("365D Alpha", ascending=False).head(5).copy()
                bot_alpha = snap_df.sort_values("365D Alpha", ascending=True).head(5).copy()

                cL, cR = st.columns(2)
                with cL:
                    st.markdown("**Top 5 by 365D Alpha**")
                    tdf = top_alpha.copy()
                    tdf["365D Return"] = tdf["365D Return"].apply(fmt_pct)
                    tdf["365D Alpha"] = tdf["365D Alpha"].apply(fmt_pct)
                    st.dataframe(tdf.set_index("Wave"), use_container_width=True)
                with cR:
                    st.markdown("**Bottom 5 by 365D Alpha**")
                    bdf = bot_alpha.copy()
                    bdf["365D Return"] = bdf["365D Return"].apply(fmt_pct)
                    bdf["365D Alpha"] = bdf["365D Alpha"].apply(fmt_pct)
                    st.dataframe(bdf.set_index("Wave"), use_container_width=True)
        else:
            st.info("Lineup snapshot unavailable (no sufficient history yet).")

# ============================================================
# Footer
# ============================================================
st.markdown("---")
st.caption(
    "WAVES Intelligence™ Institutional Console • Vector OS Edition • "
    "Benchmark Truth + Mode Separation + Wave Doctor + What-If Lab are non-destructive overlays."
)