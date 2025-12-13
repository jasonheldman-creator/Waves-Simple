# app.py — WAVES Intelligence™ Institutional Console
# Institutional Reference Build (IRB-1) — FINAL VISUAL PASS → v1.0 LOCK READY
# FULL FILE (no patches)
#
# Adds FINAL scan-first UX (non-destructive; engine math unchanged):
#   ✅ "Pinned" Summary Bar (top-of-page, always first)
#   ✅ Scan Mode toggle (fast iPhone demo mode)
#   ✅ Alpha Heatmap View (All Waves x Timeframe)
#   ✅ One-click Wave Jump (reliable: Quick Jump buttons + selector synced to session_state)
#   ✅ Row highlighting (mobile-safe: ◀ SELECTED marker + optional sort)
#
# Notes
#   • Does NOT change your engine math or baseline results.
#   • What-If Lab remains explicitly “shadow simulation”.
#   • Works even if optional libs (plotly/yfinance) are missing.

from __future__ import annotations

import math
from datetime import datetime, timedelta
from typing import Dict, List, Optional

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
    page_title="WAVES Intelligence™ Console (IRB-1)",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# Formatting helpers
# ============================================================
def fmt_pct(x: float, digits: int = 2) -> str:
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "—"
    return f"{x*100:0.{digits}f}%"

def fmt_num(x: float, digits: int = 2) -> str:
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "—"
    return f"{x:.{digits}f}"

def fmt_score(x: float) -> str:
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "—"
    return f"{x:.1f}"

def safe_series(s: pd.Series) -> pd.Series:
    if s is None or len(s) == 0:
        return pd.Series(dtype=float)
    return s.copy()

# ============================================================
# Engine-safe wrappers
# ============================================================
@st.cache_data(show_spinner=False)
def get_all_waves() -> List[str]:
    try:
        waves = we.get_all_waves()
        return list(waves) if waves else []
    except Exception:
        return []

@st.cache_data(show_spinner=False)
def get_modes() -> List[str]:
    try:
        modes = we.get_modes()
        return list(modes) if modes else ["Standard", "Alpha-Minus-Beta", "Private Logic"]
    except Exception:
        return ["Standard", "Alpha-Minus-Beta", "Private Logic"]

@st.cache_data(show_spinner=False)
def compute_wave_history(wave_name: str, mode: str, days: int = 365) -> pd.DataFrame:
    try:
        df = we.compute_history_nav(wave_name, mode=mode, days=days)
    except Exception:
        df = pd.DataFrame(columns=["wave_nav", "bm_nav", "wave_ret", "bm_ret"])
    if df is None or df.empty:
        return pd.DataFrame(columns=["wave_nav", "bm_nav", "wave_ret", "bm_ret"])
    # ensure expected cols exist
    for c in ["wave_nav", "bm_nav", "wave_ret", "bm_ret"]:
        if c not in df.columns:
            df[c] = np.nan
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
# Data fetch (yfinance) — cached
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

    # normalize columns
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
def fetch_spy_vix(days: int = 365) -> pd.DataFrame:
    return fetch_prices_daily(["SPY", "^VIX"], days=days)

@st.cache_data(show_spinner=False)
def fetch_market_assets(days: int = 365) -> pd.DataFrame:
    tickers = ["SPY", "QQQ", "IWM", "TLT", "GLD", "BTC-USD", "^VIX", "^TNX"]
    return fetch_prices_daily(tickers, days=days)

# ============================================================
# Core metrics
# ============================================================
def ret_from_nav(nav: pd.Series, window: int) -> float:
    nav = safe_series(nav)
    if len(nav) < 2:
        return float("nan")
    window = min(window, len(nav))
    if window < 2:
        return float("nan")
    sub = nav.iloc[-window:]
    a = float(sub.iloc[0])
    b = float(sub.iloc[-1])
    if a <= 0:
        return float("nan")
    return (b / a) - 1.0

def annualized_vol(daily_ret: pd.Series) -> float:
    daily_ret = safe_series(daily_ret)
    if len(daily_ret) < 2:
        return float("nan")
    return float(daily_ret.std() * np.sqrt(252))

def max_drawdown(nav: pd.Series) -> float:
    nav = safe_series(nav)
    if len(nav) < 2:
        return float("nan")
    peak = nav.cummax()
    dd = (nav / peak) - 1.0
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
    ew = ret_from_nav(nav_wave, len(nav_wave))
    eb = ret_from_nav(nav_bm, len(nav_bm))
    return float((ew - eb) / te)

def simple_ret(series: pd.Series, window: int) -> float:
    series = safe_series(series)
    if len(series) < 2:
        return float("nan")
    window = min(window, len(series))
    if window < 2:
        return float("nan")
    sub = series.iloc[-window:]
    a = float(sub.iloc[0])
    b = float(sub.iloc[-1])
    if a <= 0:
        return float("nan")
    return float((b / a) - 1.0)

# ============================================================
# Benchmark source label (best-effort)
# ============================================================
def benchmark_source_label(wave_name: str) -> str:
    try:
        static_dict = getattr(we, "BENCHMARK_WEIGHTS_STATIC", None)
        if isinstance(static_dict, dict) and wave_name in static_dict and static_dict.get(wave_name):
            return "Static Override (Engine)"
    except Exception:
        pass
    bm_mix = get_benchmark_mix()
    if bm_mix is not None and not bm_mix.empty:
        return "Auto-Composite (Dynamic)"
    return "Unknown"

# ============================================================
# WaveScore proto (console-side)
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

@st.cache_data(show_spinner=False)
def compute_wavescore_table(mode_in: str, days: int = 365) -> pd.DataFrame:
    waves = get_all_waves()
    rows = []
    for w in waves:
        h = compute_wave_history(w, mode_in, days)
        if h.empty or len(h) < 40:
            rows.append(
                {
                    "Wave": w,
                    "WaveScore": np.nan,
                    "Grade": "N/A",
                    "Alpha_365D": np.nan,
                    "IR_365D": np.nan,
                    "TE_365D": np.nan,
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

        # bounded components (conservative)
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

        # resilience proxy
        rec = 0.0
        recovery = np.nan
        if len(nav_w) > 5:
            trough = float(nav_w.min())
            peak = float(nav_w.max())
            last = float(nav_w.iloc[-1])
            if peak > trough and trough > 0:
                recovery = float(np.clip((last - trough) / (peak - trough), 0.0, 1.0))
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
                "Alpha_365D": alpha,
                "IR_365D": ir,
                "TE_365D": te,
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.sort_values("WaveScore", ascending=False)


# ============================================================
# Alpha Attribution (Engine vs Static Basket) — optional/defensible
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
    px = fetch_prices_daily(list(weights.index), days=days)
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

    eng_ret = ret_from_nav(nav_wave, len(nav_wave))
    bm_ret_total = ret_from_nav(nav_bm, len(nav_bm))
    alpha_vs_bm = eng_ret - bm_ret_total

    hold_df = get_wave_holdings(wave_name)
    hold_w = _weights_from_df(hold_df, "Ticker", "Weight")
    static_nav = compute_static_nav_from_weights(hold_w, days=days)
    static_ret = ret_from_nav(static_nav, len(static_nav)) if len(static_nav) >= 2 else float("nan")
    overlay = (eng_ret - static_ret) if (pd.notna(eng_ret) and pd.notna(static_ret)) else float("nan")

    spy_px = fetch_prices_daily(["SPY"], days=days)
    if spy_px is None or spy_px.empty or "SPY" not in spy_px.columns:
        spy_ret = float("nan")
    else:
        spy_nav = (1.0 + spy_px["SPY"].pct_change().fillna(0.0)).cumprod()
        spy_ret = ret_from_nav(spy_nav, len(spy_nav))

    alpha_vs_spy = eng_ret - spy_ret if (pd.notna(eng_ret) and pd.notna(spy_ret)) else float("nan")
    bm_difficulty = bm_ret_total - spy_ret if (pd.notna(bm_ret_total) and pd.notna(spy_ret)) else float("nan")

    te = tracking_error(wave_ret, bm_ret)
    ir = information_ratio(nav_wave, nav_bm, te)

    out["Engine Return"] = float(eng_ret)
    out["Static Basket Return"] = float(static_ret) if pd.notna(static_ret) else float("nan")
    out["Overlay Contribution (Engine - Static)"] = float(overlay) if pd.notna(overlay) else float("nan")
    out["Benchmark Return"] = float(bm_ret_total)
    out["Alpha vs Benchmark"] = float(alpha_vs_bm)
    out["SPY Return"] = float(spy_ret) if pd.notna(spy_ret) else float("nan")
    out["Alpha vs SPY"] = float(alpha_vs_spy) if pd.notna(alpha_vs_spy) else float("nan")
    out["Benchmark Difficulty (BM - SPY)"] = float(bm_difficulty) if pd.notna(bm_difficulty) else float("nan")
    out["Tracking Error (TE)"] = float(te)
    out["Information Ratio (IR)"] = float(ir)
    out["Wave Vol"] = float(annualized_vol(wave_ret))
    out["Benchmark Vol"] = float(annualized_vol(bm_ret))
    out["Wave MaxDD"] = float(max_drawdown(nav_wave))
    out["Benchmark MaxDD"] = float(max_drawdown(nav_bm))
    return out


# ============================================================
# Wave Doctor™ (diagnostics + suggestions)
# ============================================================
@st.cache_data(show_spinner=False)
def compute_spy_nav(days: int = 365) -> pd.Series:
    px = fetch_prices_daily(["SPY"], days=days)
    if px.empty or "SPY" not in px.columns:
        return pd.Series(dtype=float)
    nav = (1.0 + px["SPY"].pct_change().fillna(0.0)).cumprod()
    nav.name = "spy_nav"
    return nav

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

    flags = []
    diagnosis = []
    recs = []

    if pd.notna(a30) and abs(a30) >= alpha_warn:
        flags.append("Large 30D alpha (verify benchmark + coverage)")
        diagnosis.append("30D alpha is unusually large. This can be real signal, but also benchmark mix drift or data coverage changes.")
        recs.append("Freeze benchmark for demo comparisons and watch benchmark composition drift over time.")

    if pd.notna(a365) and a365 < 0:
        diagnosis.append("365D alpha is negative. Could be true underperformance or a tougher benchmark.")
        if pd.notna(bm_difficulty) and bm_difficulty > 0.03:
            diagnosis.append("Benchmark difficulty is positive vs SPY (harder comparator).")
            recs.append("If intent is broad-market validation, temporarily compare to SPY/QQQ mix as a check (without changing baseline).")
    else:
        if pd.notna(a365) and a365 > 0.06:
            diagnosis.append("365D alpha is solidly positive — aligns with the ‘green’ outcome you wanted.")
            recs.append("Lock a benchmark snapshot for reproducibility in demos.")

    if pd.notna(te) and te > te_warn:
        flags.append("High tracking error (active risk elevated)")
        diagnosis.append("Tracking error is high; behavior differs materially from benchmark.")
        recs.append("Reduce tilt strength and/or tighten exposure caps to reduce TE (use What-If Lab).")

    if pd.notna(vol_w) and vol_w > vol_warn:
        flags.append("High volatility")
        diagnosis.append("Volatility is elevated. If the objective is ‘disciplined’, consider lower vol target.")
        recs.append("Lower vol target (e.g., 20% → 16–18%) in What-If Lab.")

    if pd.notna(mdd_w) and mdd_w < mdd_warn:
        flags.append("Deep drawdown")
        diagnosis.append("Drawdown is deep vs typical institutional tolerance.")
        recs.append("Increase SmartSafe posture in stress regimes (What-If Lab).")

    if pd.notna(vol_b) and pd.notna(vol_w) and vol_b > 0 and (vol_w / vol_b) > 1.6:
        flags.append("Volatility much higher than benchmark")
        diagnosis.append("Wave volatility is much higher than benchmark; wins/losses may be spiky.")
        recs.append("Tighten exposure cap + reduce tilt strength (What-If Lab).")

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


# ============================================================
# What-If Lab (shadow simulation) — kept intentionally simple
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

def _vix_exposure_factor(v: float, mode: str) -> float:
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

def _vix_safe_fraction(v: float, mode: str) -> float:
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

    # baseline weights on holdings only
    w = weights.reindex(tickers).fillna(0.0)
    w = w / float(w.sum()) if float(w.sum()) > 0 else w

    spy_nav = (1.0 + rets["SPY"]).cumprod()
    regime = _regime_from_spy_60d(spy_nav)

    safe_ticker = "SGOV" if "SGOV" in rets.columns else ("BIL" if "BIL" in rets.columns else ("SHY" if "SHY" in rets.columns else "SPY"))
    safe_ret = rets[safe_ticker]

    mom60 = px / px.shift(60) - 1.0

    wave_ret = []
    dates = []

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

    for dt in rets.index:
        r = rets.loc[dt]
        # momentum tilt on holdings
        tilt = pd.Series(1.0, index=tickers, dtype=float)
        if dt in mom60.index:
            m = mom60.loc[dt].reindex(tickers).fillna(0.0).clip(-0.30, 0.30)
            tilt = (1.0 + tilt_strength * m).clip(0.0, 10.0)

        ew = (w * tilt).clip(lower=0.0)
        s = float(ew.sum())
        rw = (ew / s) if s > 0 else w

        port_risk = float((r.reindex(tickers).fillna(0.0) * rw).sum())

        # realized vol target proxy
        if len(wave_ret) >= 20:
            recent = np.array(wave_ret[-20:])
            realized = float(recent.std() * np.sqrt(252))
        else:
            realized = float(vol_target)

        vol_adj = float(np.clip(vol_target / realized, 0.7, 1.3)) if realized > 0 else 1.0

        reg = str(regime.get(dt, "neutral"))
        reg_expo = float(regime_exposure_map.get(reg, 1.0))

        vix_val = float(px.loc[dt, "^VIX"]) if "^VIX" in px.columns else float("nan")
        vix_expo = _vix_exposure_factor(vix_val, mode)
        vix_safe = _vix_safe_fraction(vix_val, mode)

        expo = float(np.clip(base_expo * reg_expo * vol_adj * vix_expo, exp_min, exp_max))

        safe_frac = float(np.clip(vix_safe + extra_safe_boost, 0.0, 0.95))
        risk_frac = 1.0 - safe_frac

        total = safe_frac * float(safe_ret.get(dt, 0.0)) + risk_frac * expo * port_risk

        wave_ret.append(total)
        dates.append(dt)

    wave_ret = pd.Series(wave_ret, index=pd.Index(dates, name="Date"), name="whatif_ret")
    wave_nav = (1.0 + wave_ret).cumprod().rename("whatif_nav")
    out = pd.DataFrame({"whatif_nav": wave_nav, "whatif_ret": wave_ret})

    if freeze_benchmark:
        hist_eng = compute_wave_history(wave_name, mode=mode, days=days)
        if not hist_eng.empty and "bm_nav" in hist_eng.columns:
            out["bm_nav"] = hist_eng["bm_nav"].reindex(out.index).ffill().bfill()
            out["bm_ret"] = hist_eng["bm_ret"].reindex(out.index).fillna(0.0)
    else:
        if "SPY" in px.columns:
            spy_ret = rets["SPY"].reindex(out.index).fillna(0.0)
            out["bm_ret"] = spy_ret
            out["bm_nav"] = (1.0 + spy_ret).cumprod()

    return out


# ============================================================
# Session State (selected_wave sync)
# ============================================================
all_waves = get_all_waves()
all_modes = get_modes()

if "selected_wave" not in st.session_state:
    st.session_state["selected_wave"] = all_waves[0] if all_waves else ""
if "mode" not in st.session_state:
    st.session_state["mode"] = all_modes[0] if all_modes else "Standard"

def set_selected_wave(w: str):
    if w and w in all_waves:
        st.session_state["selected_wave"] = w

def set_mode(m: str):
    if m and m in all_modes:
        st.session_state["mode"] = m


# ============================================================
# Sidebar (controls)
# ============================================================
with st.sidebar:
    st.title("WAVES Intelligence™")
    st.caption("Institutional Console • IRB-1 • v1.0 Candidate")

    mode_in = st.selectbox("Mode", all_modes, index=max(0, all_modes.index(st.session_state["mode"])) if st.session_state["mode"] in all_modes else 0)
    set_mode(mode_in)

    # main wave selector (always available)
    wave_in = st.selectbox("Select Wave", all_waves, index=max(0, all_waves.index(st.session_state["selected_wave"])) if st.session_state["selected_wave"] in all_waves else 0)
    set_selected_wave(wave_in)

    st.markdown("---")
    st.markdown("**Display**")
    history_days = st.slider("History window (days)", min_value=60, max_value=730, value=365, step=15)

    scan_mode = st.toggle("⚡ Scan Mode (fast view)", value=True)

    st.markdown("---")
    st.markdown("**Wave Doctor thresholds**")
    alpha_warn = st.slider("Alpha alert threshold (abs, 30D)", 0.02, 0.20, 0.08, 0.01)
    te_warn = st.slider("Tracking error alert threshold", 0.05, 0.40, 0.20, 0.01)


mode = st.session_state["mode"]
selected_wave = st.session_state["selected_wave"]


# ============================================================
# Header
# ============================================================
st.title("WAVES Intelligence™ Institutional Console")
st.caption("Live Alpha Capture • SmartSafe™ overlays • Benchmark Transparency • Wave Doctor™ • IRB-1")

# ============================================================
# "Pinned" Summary Bar (top-of-page)
# ============================================================
with st.container():
    # regime snapshot
    spy_vix = fetch_spy_vix(days=min(history_days, 365))
    regime_label = "—"
    vix_last = float("nan")
    spy_60d = float("nan")

    if spy_vix is not None and not spy_vix.empty and "SPY" in spy_vix.columns and "^VIX" in spy_vix.columns:
        spy = spy_vix["SPY"].dropna()
        vix = spy_vix["^VIX"].dropna()
        if len(vix) > 0:
            vix_last = float(vix.iloc[-1])
        if len(spy) > 60:
            spy_60d = float(spy.iloc[-1] / spy.iloc[-61] - 1.0)
        # simple regime heuristic
        if not math.isnan(spy_60d):
            if spy_60d <= -0.12:
                regime_label = "PANIC"
            elif spy_60d <= -0.04:
                regime_label = "DOWNTREND"
            elif spy_60d < 0.06:
                regime_label = "NEUTRAL"
            else:
                regime_label = "UPTREND"

    # selected wave quick metrics
    h_top = compute_wave_history(selected_wave, mode, days=min(history_days, 365))
    r30 = a30 = r365 = a365 = te = ir = float("nan")
    if h_top is not None and not h_top.empty and len(h_top) >= 2:
        nav_w = h_top["wave_nav"]
        nav_b = h_top["bm_nav"]
        ret_w = h_top["wave_ret"]
        ret_b = h_top["bm_ret"]
        r30 = ret_from_nav(nav_w, min(30, len(nav_w)))
        a30 = r30 - ret_from_nav(nav_b, min(30, len(nav_b)))
        r365 = ret_from_nav(nav_w, len(nav_w))
        a365 = r365 - ret_from_nav(nav_b, len(nav_b))
        te = tracking_error(ret_w, ret_b)
        ir = information_ratio(nav_w, nav_b, te)

    ws_df = compute_wavescore_table(mode, days=365)
    ws_score = float("nan")
    ws_grade = "—"
    if ws_df is not None and not ws_df.empty:
        row = ws_df[ws_df["Wave"] == selected_wave]
        if not row.empty:
            ws_score = float(row.iloc[0]["WaveScore"])
            ws_grade = str(row.iloc[0]["Grade"])

    bm_src = benchmark_source_label(selected_wave)

    st.markdown("### 📌 Pinned Summary Bar")
    c1, c2, c3, c4, c5, c6, c7, c8 = st.columns([1.35, 1.0, 0.9, 1.15, 0.9, 0.9, 0.9, 1.1])
    c1.metric("Wave", selected_wave if selected_wave else "—")
    c2.metric("Mode", mode)
    c3.metric("Regime", regime_label)
    c4.metric("Benchmark", bm_src)
    c5.metric("30D α", fmt_pct(a30))
    c6.metric("365D α", fmt_pct(a365))
    c7.metric("TE", fmt_pct(te))
    c8.metric("WaveScore", f"{fmt_score(ws_score)}  ({ws_grade})")

st.markdown("---")

# ============================================================
# Tabs
# ============================================================
tab_console, tab_market, tab_factors, tab_vector = st.tabs(
    ["Console", "Market Intel", "Factor Decomposition", "Vector OS Insight Layer"]
)

# ============================================================
# TAB 1: CONSOLE
# ============================================================
with tab_console:
    # ------------------------------------------------------------
    # A) Alpha Heatmap View + One-Click Jump (scan-first)
    # ------------------------------------------------------------
    st.markdown("## 🔥 Alpha Heatmap Scanner (All Waves × Timeframe)")
    st.caption("Fast scan view. Uses engine NAV vs engine benchmark NAV. Baseline results unchanged.")

    # build heatmap frame
    heat_rows = []
    for w in all_waves:
        h = compute_wave_history(w, mode, days=365)
        if h is None or h.empty or len(h) < 2:
            heat_rows.append({"Wave": w, "◀": "", "α_1D": np.nan, "α_30D": np.nan, "α_60D": np.nan, "α_365D": np.nan})
            continue

        nav_w = h["wave_nav"]
        nav_b = h["bm_nav"]

        # 1D alpha
        if len(nav_w) >= 2:
            r1w = float(nav_w.iloc[-1] / nav_w.iloc[-2] - 1.0)
            r1b = float(nav_b.iloc[-1] / nav_b.iloc[-2] - 1.0)
            a1 = r1w - r1b
        else:
            a1 = np.nan

        a30 = ret_from_nav(nav_w, min(30, len(nav_w))) - ret_from_nav(nav_b, min(30, len(nav_b)))
        a60 = ret_from_nav(nav_w, min(60, len(nav_w))) - ret_from_nav(nav_b, min(60, len(nav_b)))
        a365 = ret_from_nav(nav_w, len(nav_w)) - ret_from_nav(nav_b, len(nav_b))

        heat_rows.append(
            {
                "Wave": w,
                "◀": "◀ SELECTED" if w == selected_wave else "",
                "α_1D": a1,
                "α_30D": a30,
                "α_60D": a60,
                "α_365D": a365,
            }
        )

    heat_df = pd.DataFrame(heat_rows)

    # optional sort for scan mode
    sort_key = "α_30D" if scan_mode else "α_365D"
    if sort_key in heat_df.columns:
        heat_df = heat_df.sort_values(sort_key, ascending=False, na_position="last").reset_index(drop=True)

    # "heatmap" styling (works best in desktop; still readable on mobile)
    show_heat = heat_df.copy()
    for c in ["α_1D", "α_30D", "α_60D", "α_365D"]:
        show_heat[c] = show_heat[c].apply(fmt_pct)

    left, right = st.columns([2.2, 1.0], vertical_alignment="top")

    with left:
        st.dataframe(
            show_heat.set_index("Wave"),
            use_container_width=True,
            height=420 if scan_mode else 520,
        )
        st.caption("Row highlighting is mobile-safe: see the ◀ SELECTED marker.")

    with right:
        st.markdown("### 🎯 One-click Wave Jump")
        st.caption("This is the reliable Streamlit method (row-tap selection isn’t consistent across devices).")

        # quick jump buttons: top 12 by 30D alpha (current sort)
        top_n = 12 if len(heat_df) >= 12 else len(heat_df)
        top_list = heat_df.head(top_n)["Wave"].tolist()

        # buttons in grid
        cols = st.columns(2)
        for i, w in enumerate(top_list):
            with cols[i % 2]:
                if st.button(f"➡️ {w}", use_container_width=True, key=f"jump_{w}"):
                    set_selected_wave(w)
                    st.rerun()

        st.markdown("---")
        st.markdown("#### Search / Select")
        w_pick = st.selectbox("Jump to any Wave", all_waves, index=all_waves.index(selected_wave) if selected_wave in all_waves else 0, key="jump_selectbox")
        if w_pick != selected_wave:
            set_selected_wave(w_pick)
            st.rerun()

    st.markdown("---")

    # ------------------------------------------------------------
    # B) Scan Mode: keep only essentials
    # ------------------------------------------------------------
    if scan_mode:
        st.markdown("## ⚡ Scan Mode Essentials")
        assess = wave_doctor_assess(selected_wave, mode=mode, days=365, alpha_warn=float(alpha_warn), te_warn=float(te_warn))
        if assess.get("ok", False):
            flags = assess.get("flags", [])
            if flags:
                st.warning("Wave Doctor Flags: " + " • ".join(flags))
            else:
                st.success("Wave Doctor: No major anomalies flagged at current thresholds.")
        else:
            st.info(assess.get("message", "Wave Doctor unavailable."))

        st.caption("Turn off Scan Mode in the sidebar to open the full institutional panels below.")
        st.markdown("---")

    # ------------------------------------------------------------
    # C) Full Panels (only if Scan Mode OFF)
    # ------------------------------------------------------------
    if not scan_mode:
        # Market regime monitor
        st.markdown("## 📈 Market Regime Monitor — SPY vs VIX")
        if spy_vix is None or spy_vix.empty or "SPY" not in spy_vix.columns or "^VIX" not in spy_vix.columns:
            st.warning("Unable to load SPY/VIX data right now.")
        else:
            spy = spy_vix["SPY"].copy()
            vix = spy_vix["^VIX"].copy()
            spy_norm = (spy / spy.iloc[0] * 100.0) if len(spy) else spy

            if go is not None:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=spy_vix.index, y=spy_norm, name="SPY (Index = 100)", mode="lines"))
                fig.add_trace(go.Scatter(x=spy_vix.index, y=vix, name="VIX", mode="lines", yaxis="y2"))
                fig.update_layout(
                    margin=dict(l=40, r=40, t=40, b=40),
                    legend=dict(orientation="h"),
                    xaxis=dict(title="Date"),
                    yaxis=dict(title="SPY (Indexed)", rangemode="tozero"),
                    yaxis2=dict(title="VIX", overlaying="y", side="right", rangemode="tozero"),
                    height=380,
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.line_chart(pd.DataFrame({"SPY_idx": spy_norm, "VIX": vix}))

        st.markdown("---")

        # All Waves Overview table
        st.markdown("## 🧾 All Waves Overview (Returns + Alpha)")
        overview_rows = []
        for w in all_waves:
            h = compute_wave_history(w, mode, 365)
            if h is None or h.empty or len(h) < 2:
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
                r1w = a1 = np.nan

            r30w = ret_from_nav(nav_w, min(30, len(nav_w)))
            r30b = ret_from_nav(nav_b, min(30, len(nav_b)))
            a30 = r30w - r30b

            r60w = ret_from_nav(nav_w, min(60, len(nav_w)))
            r60b = ret_from_nav(nav_b, min(60, len(nav_b)))
            a60 = r60w - r60b

            r365w = ret_from_nav(nav_w, len(nav_w))
            r365b = ret_from_nav(nav_b, len(nav_b)))
            a365 = r365w - r365b

            overview_rows.append(
                {"Wave": w, "1D Ret": r1w, "1D Alpha": a1,
                 "30D Ret": r30w, "30D Alpha": a30,
                 "60D Ret": r60w, "60D Alpha": a60,
                 "365D Ret": r365w, "365D Alpha": a365}
            )

        overview_df = pd.DataFrame(overview_rows)
        fmt_overview = overview_df.copy()
        for col in ["1D Ret", "1D Alpha", "30D Ret", "30D Alpha", "60D Ret", "60D Alpha", "365D Ret", "365D Alpha"]:
            fmt_overview[col] = fmt_overview[col].apply(fmt_pct)
        st.dataframe(fmt_overview.set_index("Wave"), use_container_width=True)
        st.markdown("---")

        # Risk analytics
        st.markdown("## 🛡️ Risk Analytics (Vol, MaxDD, TE, IR) — 365D")
        risk_rows = []
        for w in all_waves:
            h = compute_wave_history(w, mode, 365)
            if h is None or h.empty or len(h) < 2:
                risk_rows.append({"Wave": w, "Wave Vol": np.nan, "Benchmark Vol": np.nan, "Wave MaxDD": np.nan,
                                  "Benchmark MaxDD": np.nan, "Tracking Error": np.nan, "Information Ratio": np.nan})
                continue
            nav_w = h["wave_nav"]
            nav_b = h["bm_nav"]
            ret_w = h["wave_ret"]
            ret_b = h["bm_ret"]
            te = tracking_error(ret_w, ret_b)
            ir = information_ratio(nav_w, nav_b, te)
            risk_rows.append(
                {"Wave": w,
                 "Wave Vol": annualized_vol(ret_w),
                 "Benchmark Vol": annualized_vol(ret_b),
                 "Wave MaxDD": max_drawdown(nav_w),
                 "Benchmark MaxDD": max_drawdown(nav_b),
                 "Tracking Error": te,
                 "Information Ratio": ir}
            )
        risk_df = pd.DataFrame(risk_rows)
        fmt_risk = risk_df.copy()
        for col in ["Wave Vol", "Benchmark Vol", "Tracking Error", "Wave MaxDD", "Benchmark MaxDD"]:
            fmt_risk[col] = fmt_risk[col].apply(fmt_pct)
        fmt_risk["Information Ratio"] = fmt_risk["Information Ratio"].apply(lambda x: fmt_num(x, 2))
        st.dataframe(fmt_risk.set_index("Wave"), use_container_width=True)
        st.markdown("---")

        # WaveScore leaderboard
        st.markdown("## 🏁 WaveScore™ Leaderboard (Proto · 365D)")
        ws_df_full = compute_wavescore_table(mode, 365)
        if ws_df_full is None or ws_df_full.empty:
            st.info("No WaveScore data available.")
        else:
            show = ws_df_full.copy()
            show["WaveScore"] = show["WaveScore"].apply(fmt_score)
            show["Alpha_365D"] = show["Alpha_365D"].apply(fmt_pct)
            show["IR_365D"] = show["IR_365D"].apply(lambda x: fmt_num(x, 2))
            show["TE_365D"] = show["TE_365D"].apply(fmt_pct)
            st.dataframe(show.set_index("Wave"), use_container_width=True)
        st.markdown("---")

        # Benchmark transparency table
        st.markdown("## 🧩 Benchmark Transparency Table (Composite Components per Wave)")
        bm_mix = get_benchmark_mix()
        if bm_mix is None or bm_mix.empty:
            st.info("No benchmark mix table available from engine.")
        else:
            bm_show = bm_mix.copy()
            if "Weight" in bm_show.columns:
                bm_show["Weight"] = bm_show["Weight"].apply(fmt_pct)
            st.dataframe(bm_show, use_container_width=True)

        st.markdown("---")

        # Wave Detail
        st.markdown(f"## 📌 Wave Detail — **{selected_wave}** (Mode: **{mode}**)")
        h_sel = compute_wave_history(selected_wave, mode, history_days)

        left2, right2 = st.columns([2.2, 1.0])
        with left2:
            st.markdown("#### NAV vs Benchmark")
            if h_sel is None or h_sel.empty or len(h_sel) < 2:
                st.warning("Not enough data to chart NAV.")
            else:
                nav_w = h_sel["wave_nav"]
                nav_b = h_sel["bm_nav"]
                plot_df = pd.DataFrame(
                    {"Wave_NAV": nav_w / float(nav_w.iloc[0]), "Benchmark_NAV": nav_b / float(nav_b.iloc[0])}
                )
                if go is not None:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df["Wave_NAV"], name="Wave (Engine)", mode="lines"))
                    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df["Benchmark_NAV"], name="Benchmark", mode="lines"))
                    fig.update_layout(height=420, margin=dict(l=40, r=40, t=40, b=40), legend=dict(orientation="h"))
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.line_chart(plot_df)

        with right2:
            st.markdown("#### Key Metrics")
            if h_sel is None or h_sel.empty or len(h_sel) < 2:
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

        # Benchmark Truth Panel
        with st.expander("✅ Benchmark Truth Panel (composition + difficulty + diagnostics)", expanded=True):
            bm_mix_df = get_benchmark_mix()
            if bm_mix_df is None or bm_mix_df.empty:
                st.warning("Benchmark mix table not available from engine.")
            else:
                wmix = bm_mix_df[bm_mix_df["Wave"] == selected_wave].copy() if "Wave" in bm_mix_df.columns else pd.DataFrame()
                src = benchmark_source_label(selected_wave)

                colA, colB, colC, colD = st.columns(4)
                colA.metric("Benchmark Source", src)

                if h_sel is None or h_sel.empty or len(h_sel) < 2:
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
                    show = wmix.copy()
                    if "Weight" in show.columns:
                        show["Weight"] = show["Weight"].apply(fmt_pct)
                    cols = [c for c in ["Ticker", "Name", "Weight"] if c in show.columns]
                    st.dataframe(show[cols], use_container_width=True)

        # Mode separation proof
        with st.expander("✅ Mode Separation Proof (outputs + mode levers)", expanded=False):
            rows = []
            for m in all_modes:
                hh = compute_wave_history(selected_wave, m, 365)
                if hh is None or hh.empty or len(hh) < 2:
                    rows.append({"Mode": m, "Base Exposure": np.nan, "Exposure Caps": "—",
                                 "365D Return": np.nan, "365D Alpha": np.nan, "TE": np.nan, "IR": np.nan})
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

                rows.append({"Mode": m, "Base Exposure": base_expo, "Exposure Caps": caps,
                             "365D Return": r365w, "365D Alpha": a365, "TE": te, "IR": ir})

            dfm = pd.DataFrame(rows)
            fmtm = dfm.copy()
            fmtm["Base Exposure"] = fmtm["Base Exposure"].apply(lambda x: fmt_num(x, 2))
            for col in ["365D Return", "365D Alpha"]:
                fmtm[col] = fmtm[col].apply(fmt_pct)
            fmtm["TE"] = fmtm["TE"].apply(fmt_pct)
            fmtm["IR"] = fmtm["IR"].apply(lambda x: fmt_num(x, 2))
            st.dataframe(fmtm.set_index("Mode"), use_container_width=True)
            st.caption("Demo-safe proof that mode toggles are not cosmetic (shows realized outputs + lever values when exposed).")

        st.markdown("---")

        # Alpha Attribution
        with st.expander("🔍 Alpha Attribution (Engine vs Static Basket · 365D)", expanded=False):
            st.caption("Compares engine dynamic NAV vs static fixed-weight basket of the same holdings (no overlays).")
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
                c7.metric("BM Difficulty (BM−SPY)", fmt_pct(attrib.get("Benchmark Difficulty (BM - SPY)", np.nan)))
                c8.metric("IR", fmt_num(attrib.get("Information Ratio (IR)", np.nan), 2))

        # Wave Doctor + What-If Lab
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
                st.caption("Use sliders to test hypothetical parameter changes (shadow NAV only).")

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

        # Top-10 holdings
        st.markdown("## 🔗 Top-10 Holdings (Google Finance links)")
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
                return f"[{t}](https://www.google.com/finance/quote/{t})"

            hold["Google Finance"] = hold["Ticker"].astype(str).apply(google_finance_link)
            show = hold.copy()
            show["Weight"] = show["Weight"].apply(fmt_pct)
            cols = [c for c in ["Ticker", "Name", "Weight", "Google Finance"] if c in show.columns]
            st.dataframe(show[cols], use_container_width=True)


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
            if tkr not
            # app.py — WAVES Intelligence™ Institutional Console
# Institutional Reference Build (IRB-1) — FINAL VISUAL PASS → v1.0 LOCK READY
# FULL FILE (no patches)
#
# Adds FINAL scan-first UX (non-destructive; engine math unchanged):
#   ✅ "Pinned" Summary Bar (top-of-page, always first)
#   ✅ Scan Mode toggle (fast iPhone demo mode)
#   ✅ Alpha Heatmap View (All Waves x Timeframe) + row highlight marker
#   ✅ One-click Wave Jump (Quick Jump buttons + selector synced to session_state)
#   ✅ Optional table styling (desktop) + mobile-safe row highlight via ◀ SELECTED
#
# Notes
#   • Does NOT change your engine math or baseline results.
#   • What-If Lab remains explicitly “shadow simulation”.
#   • Works even if optional libs (plotly/yfinance) are missing.

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
    page_title="WAVES Intelligence™ Console (IRB-1)",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# Formatting helpers
# ============================================================
def fmt_pct(x: float, digits: int = 2) -> str:
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "—"
    return f"{x*100:0.{digits}f}%"

def fmt_num(x: float, digits: int = 2) -> str:
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "—"
    return f"{x:.{digits}f}"

def fmt_score(x: float) -> str:
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "—"
    return f"{x:.1f}"

def safe_series(s: pd.Series) -> pd.Series:
    if s is None or len(s) == 0:
        return pd.Series(dtype=float)
    return s.copy()


# ============================================================
# Engine-safe wrappers
# ============================================================
@st.cache_data(show_spinner=False)
def get_all_waves() -> List[str]:
    try:
        waves = we.get_all_waves()
        return list(waves) if waves else []
    except Exception:
        return []

@st.cache_data(show_spinner=False)
def get_modes() -> List[str]:
    try:
        modes = we.get_modes()
        return list(modes) if modes else ["Standard", "Alpha-Minus-Beta", "Private Logic"]
    except Exception:
        return ["Standard", "Alpha-Minus-Beta", "Private Logic"]

@st.cache_data(show_spinner=False)
def compute_wave_history(wave_name: str, mode: str, days: int = 365) -> pd.DataFrame:
    try:
        df = we.compute_history_nav(wave_name, mode=mode, days=days)
    except Exception:
        df = pd.DataFrame(columns=["wave_nav", "bm_nav", "wave_ret", "bm_ret"])
    if df is None or df.empty:
        return pd.DataFrame(columns=["wave_nav", "bm_nav", "wave_ret", "bm_ret"])
    for c in ["wave_nav", "bm_nav", "wave_ret", "bm_ret"]:
        if c not in df.columns:
            df[c] = np.nan
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
# Data fetch (yfinance) — cached
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

    if data is None or len(data) == 0:
        return pd.DataFrame()

    # normalize columns
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
def fetch_spy_vix(days: int = 365) -> pd.DataFrame:
    return fetch_prices_daily(["SPY", "^VIX"], days=days)

@st.cache_data(show_spinner=False)
def fetch_market_assets(days: int = 365) -> pd.DataFrame:
    tickers = ["SPY", "QQQ", "IWM", "TLT", "GLD", "BTC-USD", "^VIX", "^TNX"]
    return fetch_prices_daily(tickers, days=days)


# ============================================================
# Core metrics
# ============================================================
def ret_from_nav(nav: pd.Series, window: int) -> float:
    nav = safe_series(nav)
    if len(nav) < 2:
        return float("nan")
    window = min(window, len(nav))
    if window < 2:
        return float("nan")
    sub = nav.iloc[-window:]
    a = float(sub.iloc[0])
    b = float(sub.iloc[-1])
    if a <= 0:
        return float("nan")
    return (b / a) - 1.0

def annualized_vol(daily_ret: pd.Series) -> float:
    daily_ret = safe_series(daily_ret)
    if len(daily_ret) < 2:
        return float("nan")
    return float(daily_ret.std() * np.sqrt(252))

def max_drawdown(nav: pd.Series) -> float:
    nav = safe_series(nav)
    if len(nav) < 2:
        return float("nan")
    peak = nav.cummax()
    dd = (nav / peak) - 1.0
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
    ew = ret_from_nav(nav_wave, len(nav_wave))
    eb = ret_from_nav(nav_bm, len(nav_bm))
    return float((ew - eb) / te)

def simple_ret(series: pd.Series, window: int) -> float:
    series = safe_series(series)
    if len(series) < 2:
        return float("nan")
    window = min(window, len(series))
    if window < 2:
        return float("nan")
    sub = series.iloc[-window:]
    a = float(sub.iloc[0])
    b = float(sub.iloc[-1])
    if a <= 0:
        return float("nan")
    return float((b / a) - 1.0)


# ============================================================
# Benchmark source label (best-effort)
# ============================================================
def benchmark_source_label(wave_name: str) -> str:
    try:
        static_dict = getattr(we, "BENCHMARK_WEIGHTS_STATIC", None)
        if isinstance(static_dict, dict) and wave_name in static_dict and static_dict.get(wave_name):
            return "Static Override (Engine)"
    except Exception:
        pass
    bm_mix = get_benchmark_mix()
    if bm_mix is not None and not bm_mix.empty:
        return "Auto-Composite (Dynamic)"
    return "Unknown"


# ============================================================
# WaveScore proto (console-side)
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

@st.cache_data(show_spinner=False)
def compute_wavescore_table(mode_in: str, days: int = 365) -> pd.DataFrame:
    waves = get_all_waves()
    rows = []
    for w in waves:
        h = compute_wave_history(w, mode_in, days)
        if h.empty or len(h) < 40:
            rows.append({"Wave": w, "WaveScore": np.nan, "Grade": "N/A", "Alpha_365D": np.nan, "IR_365D": np.nan, "TE_365D": np.nan})
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
        recovery = np.nan
        if len(nav_w) > 5:
            trough = float(nav_w.min())
            peak = float(nav_w.max())
            last = float(nav_w.iloc[-1])
            if peak > trough and trough > 0:
                recovery = float(np.clip((last - trough) / (peak - trough), 0.0, 1.0))
        if not math.isnan(recovery) and not math.isnan(mdd_w) and not math.isnan(mdd_b):
            rec_part = recovery * 6.0
            dd_edge = (mdd_b - mdd_w)
            dd_part = float(np.clip((dd_edge + 0.10) / 0.20, 0.0, 1.0) * 4.0)
            rec = float(np.clip(rec_part + dd_part, 0.0, 10.0))

        eff = float(np.clip((0.25 - te) / 0.20, 0.0, 1.0) * 15.0) if not math.isnan(te) else 0.0
        transparency = 10.0
        total = float(np.clip(rq + rc + cons + rec + eff + transparency, 0.0, 100.0))

        rows.append({"Wave": w, "WaveScore": total, "Grade": _grade(total), "Alpha_365D": alpha, "IR_365D": ir, "TE_365D": te})

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.sort_values("WaveScore", ascending=False)


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
    px = fetch_prices_daily(list(weights.index), days=days)
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

    eng_ret = ret_from_nav(nav_wave, len(nav_wave))
    bm_ret_total = ret_from_nav(nav_bm, len(nav_bm))
    alpha_vs_bm = eng_ret - bm_ret_total

    hold_df = get_wave_holdings(wave_name)
    hold_w = _weights_from_df(hold_df, "Ticker", "Weight")
    static_nav = compute_static_nav_from_weights(hold_w, days=days)
    static_ret = ret_from_nav(static_nav, len(static_nav)) if len(static_nav) >= 2 else float("nan")
    overlay = (eng_ret - static_ret) if (pd.notna(eng_ret) and pd.notna(static_ret)) else float("nan")

    spy_px = fetch_prices_daily(["SPY"], days=days)
    if spy_px is None or spy_px.empty or "SPY" not in spy_px.columns:
        spy_ret = float("nan")
    else:
        spy_nav = (1.0 + spy_px["SPY"].pct_change().fillna(0.0)).cumprod()
        spy_ret = ret_from_nav(spy_nav, len(spy_nav))

    bm_difficulty = bm_ret_total - spy_ret if (pd.notna(bm_ret_total) and pd.notna(spy_ret)) else float("nan")

    te = tracking_error(wave_ret, bm_ret)
    ir = information_ratio(nav_wave, nav_bm, te)

    out["Engine Return"] = float(eng_ret)
    out["Static Basket Return"] = float(static_ret) if pd.notna(static_ret) else float("nan")
    out["Overlay Contribution (Engine - Static)"] = float(overlay) if pd.notna(overlay) else float("nan")
    out["Benchmark Return"] = float(bm_ret_total)
    out["Alpha vs Benchmark"] = float(alpha_vs_bm)
    out["SPY Return"] = float(spy_ret) if pd.notna(spy_ret) else float("nan")
    out["Benchmark Difficulty (BM - SPY)"] = float(bm_difficulty) if pd.notna(bm_difficulty) else float("nan")
    out["Tracking Error (TE)"] = float(te)
    out["Information Ratio (IR)"] = float(ir)
    out["Wave Vol"] = float(annualized_vol(wave_ret))
    out["Benchmark Vol"] = float(annualized_vol(bm_ret))
    out["Wave MaxDD"] = float(max_drawdown(nav_wave))
    out["Benchmark MaxDD"] = float(max_drawdown(nav_bm))
    return out


# ============================================================
# Wave Doctor™
# ============================================================
@st.cache_data(show_spinner=False)
def compute_spy_nav(days: int = 365) -> pd.Series:
    px = fetch_prices_daily(["SPY"], days=days)
    if px.empty or "SPY" not in px.columns:
        return pd.Series(dtype=float)
    nav = (1.0 + px["SPY"].pct_change().fillna(0.0)).cumprod()
    nav.name = "spy_nav"
    return nav

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

    spy_nav = compute_spy_nav(days=days)
    spy_ret = ret_from_nav(spy_nav, len(spy_nav)) if len(spy_nav) >= 2 else float("nan")
    bm_difficulty = (r365_b - spy_ret) if (pd.notna(r365_b) and pd.notna(spy_ret)) else float("nan")

    flags = []
    diagnosis = []
    recs = []

    if pd.notna(a30) and abs(a30) >= alpha_warn:
        flags.append("Large 30D alpha (verify benchmark + coverage)")
        diagnosis.append("30D alpha is unusually large. This can be real signal, but also benchmark mix drift or data coverage changes.")
        recs.append("Freeze benchmark for demo comparisons and watch benchmark composition drift over time.")

    if pd.notna(te) and te > te_warn:
        flags.append("High tracking error (active risk elevated)")
        diagnosis.append("Tracking error is high; behavior differs materially from benchmark.")
        recs.append("Reduce tilt strength and/or tighten exposure caps to reduce TE (use What-If Lab).")

    if pd.notna(vol_w) and vol_w > vol_warn:
        flags.append("High volatility")
        diagnosis.append("Volatility is elevated. If the objective is ‘disciplined’, consider lower vol target.")
        recs.append("Lower vol target in What-If Lab.")

    if pd.notna(mdd_w) and mdd_w < mdd_warn:
        flags.append("Deep drawdown")
        diagnosis.append("Drawdown is deep vs typical institutional tolerance.")
        recs.append("Increase SmartSafe posture in stress regimes (What-If Lab).")

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
            "Benchmark_Difficulty_BM_minus_SPY": bm_difficulty,
        },
        "flags": flags,
        "diagnosis": diagnosis,
        "recommendations": list(dict.fromkeys(recs)),
    }


# ============================================================
# What-If Lab (shadow simulation) — intentionally simple
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

def _vix_exposure_factor(v: float, mode: str) -> float:
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

def _vix_safe_fraction(v: float, mode: str) -> float:
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

    w = weights.reindex(tickers).fillna(0.0)
    w = w / float(w.sum()) if float(w.sum()) > 0 else w

    spy_nav = (1.0 + rets["SPY"]).cumprod()
    regime = _regime_from_spy_60d(spy_nav)

    safe_ticker = "SGOV" if "SGOV" in rets.columns else ("BIL" if "BIL" in rets.columns else ("SHY" if "SHY" in rets.columns else "SPY"))
    safe_ret = rets[safe_ticker]

    mom60 = px / px.shift(60) - 1.0

    wave_ret = []
    dates = []

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

    for dt in rets.index:
        r = rets.loc[dt]
        tilt = pd.Series(1.0, index=tickers, dtype=float)
        if dt in mom60.index:
            m = mom60.loc[dt].reindex(tickers).fillna(0.0).clip(-0.30, 0.30)
            tilt = (1.0 + tilt_strength * m).clip(0.0, 10.0)

        ew = (w * tilt).clip(lower=0.0)
        s = float(ew.sum())
        rw = (ew / s) if s > 0 else w

        port_risk = float((r.reindex(tickers).fillna(0.0) * rw).sum())

        if len(wave_ret) >= 20:
            recent = np.array(wave_ret[-20:])
            realized = float(recent.std() * np.sqrt(252))
        else:
            realized = float(vol_target)

        vol_adj = float(np.clip(vol_target / realized, 0.7, 1.3)) if realized > 0 else 1.0

        reg = str(regime.get(dt, "neutral"))
        reg_expo = float(regime_exposure_map.get(reg, 1.0))

        vix_val = float(px.loc[dt, "^VIX"]) if "^VIX" in px.columns else float("nan")
        vix_expo = _vix_exposure_factor(vix_val, mode)
        vix_safe = _vix_safe_fraction(vix_val, mode)

        expo = float(np.clip(base_expo * reg_expo * vol_adj * vix_expo, exp_min, exp_max))

        safe_frac = float(np.clip(vix_safe + extra_safe_boost, 0.0, 0.95))
        risk_frac = 1.0 - safe_frac

        total = safe_frac * float(safe_ret.get(dt, 0.0)) + risk_frac * expo * port_risk

        wave_ret.append(total)
        dates.append(dt)

    wave_ret = pd.Series(wave_ret, index=pd.Index(dates, name="Date"), name="whatif_ret")
    wave_nav = (1.0 + wave_ret).cumprod().rename("whatif_nav")
    out = pd.DataFrame({"whatif_nav": wave_nav, "whatif_ret": wave_ret})

    if freeze_benchmark:
        hist_eng = compute_wave_history(wave_name, mode=mode, days=days)
        if not hist_eng.empty and "bm_nav" in hist_eng.columns:
            out["bm_nav"] = hist_eng["bm_nav"].reindex(out.index).ffill().bfill()
            out["bm_ret"] = hist_eng["bm_ret"].reindex(out.index).fillna(0.0)
    else:
        if "SPY" in px.columns:
            spy_ret = rets["SPY"].reindex(out.index).fillna(0.0)
            out["bm_ret"] = spy_ret
            out["bm_nav"] = (1.0 + spy_ret).cumprod()

    return out


# ============================================================
# Factor regression (simple OLS) + Correlation matrix
# ============================================================
def regress_factors(y: pd.Series, X: pd.DataFrame) -> Dict[str, float]:
    """Returns betas for columns in X. y and X are daily returns."""
    y = safe_series(y).dropna()
    if X is None or X.empty:
        return {}
    df = pd.concat([y.rename("y"), X], axis=1).dropna()
    if df.shape[0] < 30:
        return {}
    Y = df["y"].values.astype(float)
    Xmat = df.drop(columns=["y"]).values.astype(float)
    # add intercept
    ones = np.ones((Xmat.shape[0], 1), dtype=float)
    Xaug = np.concatenate([ones, Xmat], axis=1)
    try:
        beta = np.linalg.lstsq(Xaug, Y, rcond=None)[0]
    except Exception:
        return {}
    # beta[0] is intercept
    cols = list(df.drop(columns=["y"]).columns)
    out = {c: float(beta[i + 1]) for i, c in enumerate(cols)}
    return out


# ============================================================
# Session State (selected_wave sync)
# ============================================================
all_waves = get_all_waves()
all_modes = get_modes()

if "selected_wave" not in st.session_state:
    st.session_state["selected_wave"] = all_waves[0] if all_waves else ""
if "mode" not in st.session_state:
    st.session_state["mode"] = all_modes[0] if all_modes else "Standard"

def set_selected_wave(w: str):
    if w and w in all_waves:
        st.session_state["selected_wave"] = w

def set_mode(m: str):
    if m and m in all_modes:
        st.session_state["mode"] = m


# ============================================================
# Sidebar (controls)
# ============================================================
with st.sidebar:
    st.title("WAVES Intelligence™")
    st.caption("Institutional Console • IRB-1 • v1.0 Candidate")

    mode_in = st.selectbox(
        "Mode",
        all_modes,
        index=max(0, all_modes.index(st.session_state["mode"])) if st.session_state["mode"] in all_modes else 0,
    )
    set_mode(mode_in)

    wave_in = st.selectbox(
        "Select Wave",
        all_waves,
        index=max(0, all_waves.index(st.session_state["selected_wave"])) if st.session_state["selected_wave"] in all_waves else 0,
    )
    set_selected_wave(wave_in)

    st.markdown("---")
    st.markdown("**Display**")
    history_days = st.slider("History window (days)", min_value=60, max_value=730, value=365, step=15)
    scan_mode = st.toggle("⚡ Scan Mode (fast view)", value=True)

    st.markdown("---")
    st.markdown("**Wave Doctor thresholds**")
    alpha_warn = st.slider("Alpha alert threshold (abs, 30D)", 0.02, 0.20, 0.08, 0.01)
    te_warn = st.slider("Tracking error alert threshold", 0.05, 0.40, 0.20, 0.01)

mode = st.session_state["mode"]
selected_wave = st.session_state["selected_wave"]


# ============================================================
# Header
# ============================================================
st.title("WAVES Intelligence™ Institutional Console")
st.caption("Live Alpha Capture • SmartSafe™ overlays • Benchmark Transparency • Wave Doctor™ • IRB-1")


# ============================================================
# "Pinned" Summary Bar (top-of-page)
# ============================================================
with st.container():
    spy_vix = fetch_spy_vix(days=min(history_days, 365))
    regime_label = "—"
    vix_last = float("nan")
    spy_60d = float("nan")

    if spy_vix is not None and not spy_vix.empty and "SPY" in spy_vix.columns and "^VIX" in spy_vix.columns:
        spy = spy_vix["SPY"].dropna()
        vix = spy_vix["^VIX"].dropna()
        if len(vix) > 0:
            vix_last = float(vix.iloc[-1])
        if len(spy) > 60:
            spy_60d = float(spy.iloc[-1] / spy.iloc[-61] - 1.0)
        if not math.isnan(spy_60d):
            if spy_60d <= -0.12:
                regime_label = "PANIC"
            elif spy_60d <= -0.04:
                regime_label = "DOWNTREND"
            elif spy_60d < 0.06:
                regime_label = "NEUTRAL"
            else:
                regime_label = "UPTREND"

    h_top = compute_wave_history(selected_wave, mode, days=min(history_days, 365))
    a30 = a365 = te = ir = float("nan")
    if h_top is not None and not h_top.empty and len(h_top) >= 2:
        nav_w = h_top["wave_nav"]
        nav_b = h_top["bm_nav"]
        ret_w = h_top["wave_ret"]
        ret_b = h_top["bm_ret"]
        r30w = ret_from_nav(nav_w, min(30, len(nav_w)))
        r30b = ret_from_nav(nav_b, min(30, len(nav_b)))
        a30 = r30w - r30b
        r365w = ret_from_nav(nav_w, len(nav_w))
        r365b = ret_from_nav(nav_b, len(nav_b))
        a365 = r365w - r365b
        te = tracking_error(ret_w, ret_b)
        ir = information_ratio(nav_w, nav_b, te)

    ws_df = compute_wavescore_table(mode, days=365)
    ws_score = float("nan")
    ws_grade = "—"
    if ws_df is not None and not ws_df.empty:
        row = ws_df[ws_df["Wave"] == selected_wave]
        if not row.empty:
            ws_score = float(row.iloc[0]["WaveScore"])
            ws_grade = str(row.iloc[0]["Grade"])

    bm_src = benchmark_source_label(selected_wave)

    st.markdown("### 📌 Pinned Summary Bar")
    c1, c2, c3, c4, c5, c6, c7, c8 = st.columns([1.35, 1.0, 0.9, 1.15, 0.9, 0.9, 0.9, 1.1])
    c1.metric("Wave", selected_wave if selected_wave else "—")
    c2.metric("Mode", mode)
    c3.metric("Regime", regime_label)
    c4.metric("Benchmark", bm_src)
    c5.metric("30D α", fmt_pct(a30))
    c6.metric("365D α", fmt_pct(a365))
    c7.metric("TE", fmt_pct(te))
    c8.metric("WaveScore", f"{fmt_score(ws_score)}  ({ws_grade})")

st.markdown("---")


# ============================================================
# Tabs
# ============================================================
tab_console, tab_market, tab_factors, tab_vector = st.tabs(
    ["Console", "Market Intel", "Factor Decomposition", "Vector OS Insight Layer"]
)


# ============================================================
# TAB 1: CONSOLE
# ============================================================
with tab_console:
    # ------------------------------------------------------------
    # A) Alpha Heatmap Scanner + One-Click Jump
    # ------------------------------------------------------------
    st.markdown("## 🔥 Alpha Heatmap Scanner (All Waves × Timeframe)")
    st.caption("Fast scan view. Uses engine NAV vs engine benchmark NAV. Baseline results unchanged.")

    heat_rows = []
    for w in all_waves:
        h = compute_wave_history(w, mode, days=365)
        if h is None or h.empty or len(h) < 2:
            heat_rows.append({"Wave": w, "◀": "", "α_1D": np.nan, "α_30D": np.nan, "α_60D": np.nan, "α_365D": np.nan})
            continue

        nav_w = h["wave_nav"]
        nav_b = h["bm_nav"]

        if len(nav_w) >= 2 and len(nav_b) >= 2:
            r1w = float(nav_w.iloc[-1] / nav_w.iloc[-2] - 1.0)
            r1b = float(nav_b.iloc[-1] / nav_b.iloc[-2] - 1.0)
            a1 = r1w - r1b
        else:
            a1 = np.nan

        a30x = ret_from_nav(nav_w, min(30, len(nav_w))) - ret_from_nav(nav_b, min(30, len(nav_b)))
        a60x = ret_from_nav(nav_w, min(60, len(nav_w))) - ret_from_nav(nav_b, min(60, len(nav_b)))
        a365x = ret_from_nav(nav_w, len(nav_w)) - ret_from_nav(nav_b, len(nav_b))

        heat_rows.append(
            {
                "Wave": w,
                "◀": "◀ SELECTED" if w == selected_wave else "",
                "α_1D": a1,
                "α_30D": a30x,
                "α_60D": a60x,
                "α_365D": a365x,
            }
        )

    heat_df = pd.DataFrame(heat_rows)

    # default scan sort: 30D alpha
    sort_key = "α_30D"
    if sort_key in heat_df.columns:
        heat_df = heat_df.sort_values(sort_key, ascending=False, na_position="last").reset_index(drop=True)

    show_heat = heat_df.copy()
    for c in ["α_1D", "α_30D", "α_60D", "α_365D"]:
        show_heat[c] = show_heat[c].apply(fmt_pct)

    left, right = st.columns([2.2, 1.0], vertical_alignment="top")

    with left:
        st.dataframe(
            show_heat.set_index("Wave"),
            use_container_width=True,
            height=420 if scan_mode else 520,
        )
        st.caption("Row highlight is mobile-safe via the ◀ SELECTED marker (Streamlit row-tap selection isn’t reliable on iPhone).")

    with right:
        st.markdown("### 🎯 One-click Wave Jump")
        st.caption("Quick jump buttons + selector (reliable across devices).")

        top_n = min(12, len(heat_df))
        top_list = heat_df.head(top_n)["Wave"].tolist()

        cols = st.columns(2)
        for i, w in enumerate(top_list):
            with cols[i % 2]:
                if st.button(f"➡️ {w}", use_container_width=True, key=f"jump_{w}"):
                    set_selected_wave(w)
                    st.rerun()

        st.markdown("---")
        st.markdown("#### Search / Select")
        w_pick = st.selectbox(
            "Jump to any Wave",
            all_waves,
            index=all_waves.index(selected_wave) if selected_wave in all_waves else 0,
            key="jump_selectbox",
        )
        if w_pick != selected_wave:
            set_selected_wave(w_pick)
            st.rerun()

    st.markdown("---")

    # ------------------------------------------------------------
    # B) Scan Mode Essentials
    # ------------------------------------------------------------
    if scan_mode:
        st.markdown("## ⚡ Scan Mode Essentials")
        assess = wave_doctor_assess(selected_wave, mode=mode, days=365, alpha_warn=float(alpha_warn), te_warn=float(te_warn))
        if assess.get("ok", False):
            flags = assess.get("flags", [])
            if flags:
                st.warning("Wave Doctor Flags: " + " • ".join(flags))
            else:
                st.success("Wave Doctor: No major anomalies flagged at current thresholds.")
        else:
            st.info(assess.get("message", "Wave Doctor unavailable."))

        st.caption("Turn OFF Scan Mode in the sidebar to open the full institutional panels below.")
        st.markdown("---")

    # ------------------------------------------------------------
    # C) Full Institutional Panels (Scan Mode OFF)
    # ------------------------------------------------------------
    if not scan_mode:
        # Market regime monitor
        st.markdown("## 📈 Market Regime Monitor — SPY vs VIX")
        if spy_vix is None or spy_vix.empty or "SPY" not in spy_vix.columns or "^VIX" not in spy_vix.columns:
            st.warning("Unable to load SPY/VIX data right now.")
        else:
            spy = spy_vix["SPY"].
            copy()
            vix = spy_vix["^VIX"].copy()
            spy_norm = (spy / spy.iloc[0] * 100.0) if len(spy) else spy

            if go is not None:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=spy_vix.index, y=spy_norm, name="SPY (Index = 100)", mode="lines"))
                fig.add_trace(go.Scatter(x=spy_vix.index, y=vix, name="VIX", mode="lines", yaxis="y2"))
                fig.update_layout(
                    height=380,
                    margin=dict(l=40, r=40, t=40, b=40),
                    legend=dict(orientation="h"),
                    xaxis=dict(title="Date"),
                    yaxis=dict(title="SPY (Indexed)"),
                    yaxis2=dict(title="VIX", overlaying="y", side="right"),
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.line_chart(pd.DataFrame({"SPY_idx": spy_norm, "VIX": vix}))

        st.markdown("---")

        # ------------------------------------------------------------
        # All Waves Overview
        # ------------------------------------------------------------
        st.markdown("## 🧾 All Waves Overview (Returns + Alpha)")

        overview_rows = []
        for w in all_waves:
            h = compute_wave_history(w, mode, 365)
            if h.empty or len(h) < 2:
                overview_rows.append(
                    {"Wave": w, "30D Ret": np.nan, "30D Alpha": np.nan, "365D Ret": np.nan, "365D Alpha": np.nan}
                )
                continue

            nav_w = h["wave_nav"]
            nav_b = h["bm_nav"]

            r30w = ret_from_nav(nav_w, min(30, len(nav_w)))
            r30b = ret_from_nav(nav_b, min(30, len(nav_b)))
            r365w = ret_from_nav(nav_w, len(nav_w))
            r365b = ret_from_nav(nav_b, len(nav_b))

            overview_rows.append(
                {
                    "Wave": w,
                    "30D Ret": r30w,
                    "30D Alpha": r30w - r30b,
                    "365D Ret": r365w,
                    "365D Alpha": r365w - r365b,
                }
            )

        ov = pd.DataFrame(overview_rows)
        ov_show = ov.copy()
        for c in ["30D Ret", "30D Alpha", "365D Ret", "365D Alpha"]:
            ov_show[c] = ov_show[c].apply(fmt_pct)

        st.dataframe(ov_show.set_index("Wave"), use_container_width=True)
        st.markdown("---")

        # ------------------------------------------------------------
        # WaveScore Leaderboard
        # ------------------------------------------------------------
        st.markdown("## 🏁 WaveScore™ Leaderboard (Proto · 365D)")
        ws_full = compute_wavescore_table(mode, 365)
        if ws_full.empty:
            st.info("WaveScore data not available.")
        else:
            ws_show = ws_full.copy()
            ws_show["WaveScore"] = ws_show["WaveScore"].apply(fmt_score)
            ws_show["Alpha_365D"] = ws_show["Alpha_365D"].apply(fmt_pct)
            ws_show["IR_365D"] = ws_show["IR_365D"].apply(lambda x: fmt_num(x, 2))
            ws_show["TE_365D"] = ws_show["TE_365D"].apply(fmt_pct)
            st.dataframe(ws_show.set_index("Wave"), use_container_width=True)

        st.markdown("---")

        # ------------------------------------------------------------
        # Top-10 Holdings
        # ------------------------------------------------------------
        st.markdown("## 🔗 Top-10 Holdings (Google Finance Links)")
        hold = get_wave_holdings(selected_wave)

        if hold.empty:
            st.info("No holdings available for this Wave.")
        else:
            hold = hold.copy()
            hold["Weight"] = pd.to_numeric(hold["Weight"], errors="coerce").fillna(0.0)
            hold = hold.sort_values("Weight", ascending=False).head(10)

            def gfin(t: str) -> str:
                return f"[{t}](https://www.google.com/finance/quote/{t})"

            hold["Google Finance"] = hold["Ticker"].astype(str).apply(gfin)
            hold["Weight"] = hold["Weight"].apply(fmt_pct)

            st.dataframe(
                hold[["Ticker", "Name", "Weight", "Google Finance"]],
                use_container_width=True,
            )

# ============================================================
# TAB 2: MARKET INTEL
# ============================================================
with tab_market:
    st.markdown("## 🌍 Global Market Snapshot")

    market_df = fetch_market_assets(history_days)
    if market_df.empty:
        st.warning("Market data unavailable.")
    else:
        rows = []
        for tkr in market_df.columns:
            s = market_df[tkr].dropna()
            if len(s) < 2:
                continue
            rows.append(
                {
                    "Asset": tkr,
                    "Last": s.iloc[-1],
                    "1D Return": simple_ret(s, 2),
                    "30D Return": simple_ret(s, 30),
                }
            )

        snap = pd.DataFrame(rows)
        snap["Last"] = snap["Last"].apply(lambda x: fmt_num(x, 2))
        snap["1D Return"] = snap["1D Return"].apply(fmt_pct)
        snap["30D Return"] = snap["30D Return"].apply(fmt_pct)

        st.dataframe(snap.set_index("Asset"), use_container_width=True)

# ============================================================
# TAB 3: FACTOR DECOMPOSITION
# ============================================================
with tab_factors:
    st.markdown("## 🧬 Factor Decomposition + Correlation Matrix")

    factor_prices = fetch_market_assets(min(history_days, 365))
    needed = ["SPY", "QQQ", "IWM", "TLT", "GLD", "BTC-USD"]

    if factor_prices.empty or any(t not in factor_prices.columns for t in needed):
        st.warning("Insufficient data for factor analysis.")
    else:
        factor_returns = factor_prices[needed].pct_change().dropna()
        factor_returns = factor_returns.rename(
            columns={
                "SPY": "MKT",
                "QQQ": "GROWTH",
                "IWM": "SIZE",
                "TLT": "RATES",
                "GLD": "GOLD",
                "BTC-USD": "CRYPTO",
            }
        )

        rows = []
        for w in all_waves:
            h = compute_wave_history(w, mode, min(history_days, 365))
            if h.empty:
                continue
            betas = regress_factors(h["wave_ret"], factor_returns)
            row = {"Wave": w}
            row.update(betas)
            rows.append(row)

        beta_df = pd.DataFrame(rows).set_index("Wave")
        beta_df = beta_df.applymap(lambda x: fmt_num(x, 2))
        st.dataframe(beta_df, use_container_width=True)

        st.markdown("---")
        st.markdown("### 🔁 Correlation Matrix (Waves)")

        ret_panel = {}
        for w in all_waves:
            h = compute_wave_history(w, mode, min(history_days, 365))
            if not h.empty:
                ret_panel[w] = h["wave_ret"]

        if ret_panel:
            corr = pd.DataFrame(ret_panel).corr()
            st.dataframe(corr, use_container_width=True)

# ============================================================
# TAB 4: VECTOR OS INSIGHT LAYER
# ============================================================
with tab_vector:
    st.markdown("## 🤖 Vector OS — Rules-Based Insight")

    ws_df_local = compute_wavescore_table(mode, 365)
    h = compute_wave_history(selected_wave, mode, 365)

    if ws_df_local.empty or h.empty:
        st.info("Not enough data for Vector insight.")
    else:
        row = ws_df_local[ws_df_local["Wave"] == selected_wave].iloc[0]
        nav_w = h["wave_nav"]
        nav_b = h["bm_nav"]

        r365w = ret_from_nav(nav_w, len(nav_w))
        r365b = ret_from_nav(nav_b, len(nav_b))
        a365 = r365w - r365b

        st.markdown(f"### Vector Insight — **{selected_wave}**")
        st.write(f"- **WaveScore**: {fmt_score(row['WaveScore'])} ({row['Grade']})")
        st.write(f"- **365D Alpha**: {fmt_pct(a365)}")
        st.write(f"- **IR**: {fmt_num(row['IR_365D'], 2)}")

        if a365 > 0.05:
            st.success("Vector: Strong institutional-grade outperformance signal.")
        elif a365 > 0:
            st.info("Vector: Mild outperformance — monitor tracking error.")
        else:
            st.warning("Vector: Underperformance vs benchmark — investigate benchmark difficulty.")

# ============================================================
# Footer
# ============================================================
st.markdown("---")
st.caption("WAVES Intelligence™ • Institutional Console v1.0 • Engine math unchanged • Demo-safe")