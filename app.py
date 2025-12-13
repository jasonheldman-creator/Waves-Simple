# app.py — WAVES Intelligence™ Institutional Console
# Institutional Reference Build (IRB-1) — FULL FILE (no patches)
#
# Goals:
#  1) Make console easier to scan (mobile + institutional)
#  2) Add Benchmark Truth + Mode Separation Proof
#  3) Add Wave Doctor™ + What-If Lab (SHADOW ONLY, baseline unchanged)
#  4) Keep engine math/results untouched (read-only)
#
# Requires: streamlit, pandas, numpy (plotly optional), yfinance optional
# Engine: waves_engine.py must expose:
#   - get_all_waves()
#   - get_modes()
#   - compute_history_nav(wave_name, mode, days) -> DataFrame with cols: wave_nav, bm_nav, wave_ret, bm_ret
# Optional engine extras (if present, UI will use them):
#   - get_benchmark_mix_table()
#   - get_wave_holdings(wave_name)
#   - MODE_BASE_EXPOSURE (dict)
#   - MODE_EXPOSURE_CAPS (dict: mode -> (min,max))
#   - BENCHMARK_WEIGHTS_STATIC (dict)

from __future__ import annotations

import math
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import streamlit as st

import waves_engine as we

# Optional deps (do not hard-fail)
try:
    import yfinance as yf
except Exception:
    yf = None

try:
    import plotly.graph_objects as go
except Exception:
    go = None


# ============================================================
# Page Config + lightweight styling
# ============================================================
st.set_page_config(
    page_title="WAVES Intelligence™ Console",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      .block-container { padding-top: 0.9rem; padding-bottom: 1.2rem; }
      .stMetric { border-radius: 10px; }
      .smallcap { font-size: 0.84rem; opacity: 0.85; }
      .chip { display:inline-block; padding:2px 8px; border-radius:999px; border:1px solid rgba(255,255,255,0.18); font-size:0.80rem; }
      .chip-green { background: rgba(0,255,140,0.10); }
      .chip-red { background: rgba(255,90,90,0.10); }
      .chip-blue { background: rgba(80,170,255,0.10); }
      .chip-gray { background: rgba(160,160,160,0.10); }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# Formatting helpers
# ============================================================
def fmt_pct(x: float, digits: int = 2) -> str:
    try:
        if x is None or (isinstance(x, float) and math.isnan(x)):
            return "—"
        return f"{x*100:0.{digits}f}%"
    except Exception:
        return "—"


def fmt_num(x: float, digits: int = 2) -> str:
    try:
        if x is None or (isinstance(x, float) and math.isnan(x)):
            return "—"
        return f"{x:.{digits}f}"
    except Exception:
        return "—"


def fmt_score(x: float) -> str:
    try:
        if x is None or (isinstance(x, float) and math.isnan(x)):
            return "—"
        return f"{x:.1f}"
    except Exception:
        return "—"


def safe_series(s: pd.Series) -> pd.Series:
    if s is None:
        return pd.Series(dtype=float)
    try:
        return s.dropna().copy()
    except Exception:
        return pd.Series(dtype=float)


# ============================================================
# Data fetch (optional) + caching
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

    # normalize shape
    if isinstance(data.columns, pd.MultiIndex):
        # prefer Adj Close / Close
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
# Engine adapters (safe)
# ============================================================
@st.cache_data(show_spinner=False)
def compute_wave_history(wave_name: str, mode: str, days: int = 365) -> pd.DataFrame:
    try:
        df = we.compute_history_nav(wave_name, mode=mode, days=days)
        if df is None:
            return pd.DataFrame()
        # ensure expected columns exist
        needed = ["wave_nav", "bm_nav", "wave_ret", "bm_ret"]
        for c in needed:
            if c not in df.columns:
                df[c] = np.nan
        return df.dropna(how="all")
    except Exception:
        return pd.DataFrame(columns=["wave_nav", "bm_nav", "wave_ret", "bm_ret"])


@st.cache_data(show_spinner=False)
def get_benchmark_mix() -> pd.DataFrame:
    try:
        df = we.get_benchmark_mix_table()
        return df if df is not None else pd.DataFrame()
    except Exception:
        return pd.DataFrame(columns=["Wave", "Ticker", "Name", "Weight"])


@st.cache_data(show_spinner=False)
def get_wave_holdings(wave_name: str) -> pd.DataFrame:
    try:
        df = we.get_wave_holdings(wave_name)
        return df if df is not None else pd.DataFrame()
    except Exception:
        return pd.DataFrame(columns=["Ticker", "Name", "Weight"])


# ============================================================
# Core metrics
# ============================================================
def ret_from_nav(nav: pd.Series, window: int) -> float:
    nav = safe_series(nav)
    if len(nav) < 2:
        return float("nan")
    if window < 2:
        return float("nan")
    if len(nav) < window:
        window = len(nav)
    sub = nav.iloc[-window:]
    a = float(sub.iloc[0])
    b = float(sub.iloc[-1])
    if a <= 0:
        return float("nan")
    return float(b / a - 1.0)


def annualized_vol(daily_ret: pd.Series) -> float:
    r = safe_series(daily_ret)
    if len(r) < 2:
        return float("nan")
    return float(r.std() * np.sqrt(252))


def max_drawdown(nav: pd.Series) -> float:
    nav = safe_series(nav)
    if len(nav) < 2:
        return float("nan")
    running_max = nav.cummax()
    dd = (nav / running_max) - 1.0
    return float(dd.min())


def tracking_error(daily_wave: pd.Series, daily_bm: pd.Series) -> float:
    w = safe_series(daily_wave)
    b = safe_series(daily_bm)
    if len(w) < 2 or len(b) < 2:
        return float("nan")
    df = pd.concat([w.rename("w"), b.rename("b")], axis=1).dropna()
    if df.shape[0] < 2:
        return float("nan")
    diff = df["w"] - df["b"]
    return float(diff.std() * np.sqrt(252))


def information_ratio(nav_wave: pd.Series, nav_bm: pd.Series, te: float) -> float:
    if te is None or (isinstance(te, float) and (math.isnan(te) or te <= 0)):
        return float("nan")
    rw = ret_from_nav(nav_wave, len(nav_wave))
    rb = ret_from_nav(nav_bm, len(nav_bm))
    if math.isnan(rw) or math.isnan(rb):
        return float("nan")
    return float((rw - rb) / te)


def simple_ret(series: pd.Series, window: int) -> float:
    s = safe_series(series)
    if len(s) < 2:
        return float("nan")
    if len(s) < window:
        window = len(s)
    if window < 2:
        return float("nan")
    sub = s.iloc[-window:]
    a = float(sub.iloc[0])
    b = float(sub.iloc[-1])
    if a <= 0:
        return float("nan")
    return float(b / a - 1.0)


def regress_factors(wave_ret: pd.Series, factor_ret: pd.DataFrame) -> dict:
    wave_ret = safe_series(wave_ret)
    if wave_ret.empty or factor_ret is None or factor_ret.empty:
        return {col: float("nan") for col in (factor_ret.columns if factor_ret is not None else [])}

    df = pd.concat([wave_ret.rename("wave"), factor_ret], axis=1).dropna()
    if df.shape[0] < 60 or df.shape[1] < 2:
        return {col: float("nan") for col in factor_ret.columns}

    y = df["wave"].values
    X = df[factor_ret.columns].values
    X_design = np.concatenate([np.ones((X.shape[0], 1)), X], axis=1)

    try:
        beta, *_ = np.linalg.lstsq(X_design, y, rcond=None)
        betas = beta[1:]
        return {col: float(b) for col, b in zip(factor_ret.columns, betas)}
    except Exception:
        return {col: float("nan") for col in factor_ret.columns}


# ============================================================
# Benchmark context helpers
# ============================================================
@st.cache_data(show_spinner=False)
def compute_spy_nav(days: int = 365) -> pd.Series:
    px = fetch_prices_daily(["SPY"], days=days)
    if px.empty or "SPY" not in px.columns:
        return pd.Series(dtype=float)
    nav = (1.0 + px["SPY"].pct_change().fillna(0.0)).cumprod()
    nav.name = "spy_nav"
    return nav


def benchmark_source_label(wave_name: str) -> str:
    try:
        static_dict = getattr(we, "BENCHMARK_WEIGHTS_STATIC", None)
        if isinstance(static_dict, dict) and wave_name in static_dict and static_dict.get(wave_name):
            return "Static Override (Engine)"
    except Exception:
        pass
    # if mix exists, likely auto composite
    bm_mix = get_benchmark_mix()
    if bm_mix is not None and not bm_mix.empty:
        return "Auto-Composite (Dynamic)"
    return "Unknown"


# ============================================================
# Attribution: Engine vs Static Basket (read-only)
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

    eng_ret = ret_from_nav(nav_wave, len(nav_wave))
    bm_ret_total = ret_from_nav(nav_bm, len(nav_bm))
    alpha_vs_bm = eng_ret - bm_ret_total

    hold_df = get_wave_holdings(wave_name)
    hold_w = _weights_from_df(hold_df, "Ticker", "Weight")
    static_nav = compute_static_nav_from_weights(hold_w, days=days)
    static_ret = ret_from_nav(static_nav, len(static_nav)) if len(static_nav) >= 2 else float("nan")
    overlay_pp = (eng_ret - static_ret) if (pd.notna(eng_ret) and pd.notna(static_ret)) else float("nan")

    spy_nav = compute_spy_nav(days=days)
    spy_ret = ret_from_nav(spy_nav, len(spy_nav)) if len(spy_nav) >= 2 else float("nan")

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
# WaveScore proto (console-side, read-only)
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
def compute_wavescore_table(all_waves: List[str], mode_in: str, days: int = 365) -> pd.DataFrame:
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

        # Return Quality (25)
        rq = 0.0
        if not math.isnan(ir):
            rq += float(np.clip(ir, 0.0, 1.5) / 1.5 * 15.0)
        if not math.isnan(alpha):
            rq += float(np.clip((alpha + 0.05) / 0.15, 0.0, 1.0) * 10.0)
        rq = float(np.clip(rq, 0.0, 25.0))

        # Risk Control (25)
        rc = 0.0
        if not math.isnan(vol_ratio):
            penalty = max(0.0, abs(vol_ratio - 0.9) - 0.1)
            rc = float(np.clip(1.0 - penalty / 0.6, 0.0, 1.0) * 25.0)

        # Consistency (15)
        cons = float(np.clip(hit, 0.0, 1.0) * 15.0) if not math.isnan(hit) else 0.0

        # Resilience (10) – recovery + drawdown edge proxy
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

        # Efficiency (15)
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
    if df.empty:
        return df
    return df.sort_values("WaveScore", ascending=False)


# ============================================================
# Wave Doctor™ (diagnostics + suggestions) — read-only
# ============================================================
def wave_doctor_assess(
    wave_name: str,
    mode: str,
    days: int = 365,
    alpha_warn: float = 0.08,
    te_warn: float = 0.20,
    vol_warn: float = 0.30,
    mdd_warn: float = -0.25,
) -> Dict[str, object]:
    h = compute_wave_history(wave_name, mode, days)
    if h.empty or len(h) < 2:
        return {"ok": False, "message": "Not enough data to run Wave Doctor."}

    nav_w = h["wave_nav"]
    nav_b = h["bm_nav"]
    ret_w = h["wave_ret"]
    ret_b = h["bm_ret"]

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

    # alpha anomalies
    if pd.notna(a30) and abs(a30) >= alpha_warn:
        flags.append("Large 30D alpha (verify benchmark + coverage)")
        diagnosis.append("30D alpha is unusually large. This can be real, but also can reflect benchmark mix drift or partial data coverage.")
        recs.append("For demos: freeze benchmark mix snapshot + confirm benchmark component weights for this Wave.")

    if pd.notna(a365) and a365 < 0:
        diagnosis.append("365D alpha is negative. This can be true underperformance or a tougher benchmark composition.")
        if pd.notna(bm_difficulty) and bm_difficulty > 0.03:
            diagnosis.append("Benchmark difficulty is positive vs SPY (your benchmark beat SPY over the window). Alpha is harder here.")
            recs.append("If you intend broad-market comparability: temporarily compare vs SPY/QQQ mix for validation (demo-only).")
    else:
        if pd.notna(a365) and a365 > 0.06:
            diagnosis.append("365D alpha is solidly positive. This matches the ‘good’ regime you were looking for.")
            recs.append("Lock benchmark mix (session snapshot) so results remain reproducible.")

    # risk flags
    if pd.notna(te) and te > te_warn:
        flags.append("High tracking error (active risk elevated)")
        diagnosis.append("Tracking error is high; behavior is materially different than benchmark.")
        recs.append("If you want smoother behavior: tighten exposure caps or reduce tilt strength (What-If Lab).")

    if pd.notna(vol_w) and vol_w > vol_warn:
        flags.append("High volatility")
        diagnosis.append("Volatility is elevated; returns may look ‘spiky’.")
        recs.append("Lower vol target (e.g., 20% → 16–18%) in the What-If Lab to test smoothing.")

    if pd.notna(mdd_w) and mdd_w < mdd_warn:
        flags.append("Deep drawdown")
        diagnosis.append("Drawdown is deep relative to typical institutional comfort bands.")
        recs.append("Increase SmartSafe posture in stress regimes (What-If Lab extra safe boost).")

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
# What-If Lab (shadow simulation) — baseline unchanged
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

    # safe asset
    safe_ticker = "SGOV" if "SGOV" in rets.columns else ("BIL" if "BIL" in rets.columns else ("SHY" if "SHY" in rets.columns else "SPY"))
    safe_ret = rets[safe_ticker]

    # mode base exposure (best effort)
    base_expo = 1.0
    try:
        base_map = getattr(we, "MODE_BASE_EXPOSURE", None)
        if isinstance(base_map, dict) and mode in base_map:
            base_expo = float(base_map[mode])
    except Exception:
        pass

    # regime map (best effort)
    regime_exposure_map = {"panic": 0.80, "downtrend": 0.90, "neutral": 1.00, "uptrend": 1.10}

    # momentum and realized vol
    mom60 = px / px.shift(60) - 1.0

    spy_nav = (1.0 + rets["SPY"]).cumprod()
    regime = _regime_from_spy_60d(spy_nav)
    vix = px["^VIX"]

    w = weights.reindex(px.columns).fillna(0.0)

    wave_ret = []
    dates = []

    for dt in rets.index:
        r = rets.loc[dt]

        # momentum tilt on holdings
        mom_row = mom60.loc[dt] if dt in mom60.index else None
        if mom_row is not None:
            mom_clipped = mom_row.reindex(px.columns).fillna(0.0).clip(lower=-0.30, upper=0.30)
            tilt = 1.0 + tilt_strength * mom_clipped
            ew = (w * tilt).clip(lower=0.0)
        else:
            ew = w.copy()

        ew_hold = ew.reindex(tickers).fillna(0.0)
        s = float(ew_hold.sum())
        rw = (ew_hold / s) if s > 0 else weights.copy()

        port_risk_ret = float((r.reindex(tickers).fillna(0.0) * rw).sum())

        # realized vol targeting on shadow returns
        if len(wave_ret) >= 20:
            recent = np.array(wave_ret[-20:])
            realized = recent.std() * np.sqrt(252)
        else:
            realized = vol_target

        vol_adj = float(np.clip((vol_target / realized) if realized > 0 else 1.0, 0.7, 1.3))

        reg = str(regime.get(dt, "neutral"))
        reg_expo = float(regime_exposure_map.get(reg, 1.0))

        vix_expo = _vix_exposure_factor(float(vix.get(dt, np.nan)), mode)
        vix_safe = _vix_safe_fraction(float(vix.get(dt, np.nan)), mode)

        expo = float(np.clip(base_expo * reg_expo * vol_adj * vix_expo, exp_min, exp_max))

        # safe fraction
        sf = float(np.clip(vix_safe + extra_safe_boost, 0.0, 0.95))
        rf = 1.0 - sf

        total = sf * float(safe_ret.get(dt, 0.0)) + rf * expo * port_risk_ret
        wave_ret.append(total)
        dates.append(dt)

    wave_ret = pd.Series(wave_ret, index=pd.Index(dates, name="Date"), name="whatif_ret")
    wave_nav = (1.0 + wave_ret).cumprod().rename("whatif_nav")
    out = pd.DataFrame({"whatif_nav": wave_nav, "whatif_ret": wave_ret})

    # benchmark alignment for what-if
    if freeze_benchmark:
        hist_eng = compute_wave_history(wave_name, mode=mode, days=days)
        if not hist_eng.empty and "bm_nav" in hist_eng.columns:
            out["bm_nav"] = hist_eng["bm_nav"].reindex(out.index).ffill().bfill()
            out["bm_ret"] = hist_eng["bm_ret"].reindex(out.index).fillna(0.0)
    else:
        spy = rets["SPY"].reindex(out.index).fillna(0.0)
        out["bm_ret"] = spy
        out["bm_nav"] = (1.0 + spy).cumprod()

    return out


# ============================================================
# Sidebar controls (scan-fast)
# ============================================================
all_waves = we.get_all_waves()
all_modes = we.get_modes()

with st.sidebar:
    st.title("WAVES Intelligence™")
    st.caption("Institutional Console • Vector OS™")

    mode = st.selectbox("Mode", all_modes, index=0)
    selected_wave = st.selectbox("Select Wave", all_waves, index=0)

    st.markdown("---")
    st.markdown("**Display**")
    history_days = st.slider("History window (days)", 60, 730, 365, 15)
    compact_tables = st.toggle("Compact tables (faster scan)", value=True)
    expand_key_panels = st.toggle("Expand key panels by default", value=True)

    st.markdown("---")
    st.markdown("**Wave Doctor thresholds**")
    alpha_warn = st.slider("Alpha alert threshold (abs, 30D)", 0.02, 0.20, 0.08, 0.01)
    te_warn = st.slider("Tracking error alert threshold", 0.05, 0.40, 0.20, 0.01)


# ============================================================
# Header
# ============================================================
st.title("WAVES Intelligence™ Institutional Console")
st.caption("Live Alpha Capture • SmartSafe™ • Mode Separation • Benchmark Truth • Wave Doctor™")


# ============================================================
# Tabs
# ============================================================
tab_console, tab_market, tab_factors, tab_vector = st.tabs(
    ["Console", "Market Intel", "Factor Decomposition", "Vector OS Insight Layer"]
)


# ============================================================
# TAB 1: Console
# ============================================================
with tab_console:
    # --- quick market regime monitor
    st.markdown("### 📈 Market Regime Monitor — SPY vs VIX")
    spy_vix = fetch_spy_vix(days=history_days)

    if spy_vix.empty or "SPY" not in spy_vix.columns or "^VIX" not in spy_vix.columns:
        st.warning("Unable to load SPY/VIX right now (yfinance unavailable or rate-limited).")
    else:
        spy = spy_vix["SPY"].copy()
        vix = spy_vix["^VIX"].copy()
        spy_idx = (spy / spy.iloc[0] * 100.0) if len(spy) else spy

        if go is not None:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=spy_vix.index, y=spy_idx, name="SPY (Index=100)", mode="lines"))
            fig.add_trace(go.Scatter(x=spy_vix.index, y=vix, name="VIX", mode="lines", yaxis="y2"))
            fig.update_layout(
                height=330,
                margin=dict(l=40, r=40, t=40, b=40),
                legend=dict(orientation="h"),
                xaxis=dict(title="Date"),
                yaxis=dict(title="SPY (Indexed)"),
                yaxis2=dict(title="VIX", overlaying="y", side="right"),
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.line_chart(pd.DataFrame({"SPY_idx": spy_idx, "VIX": vix}))

    st.markdown("---")

    # ============================================================
    # All Waves Overview (scan-first)
    # ============================================================
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

        # 1D
        r1w = float(nav_w.iloc[-1] / nav_w.iloc[-2] - 1.0) if len(nav_w) >= 2 else np.nan
        r1b = float(nav_b.iloc[-1] / nav_b.iloc[-2] - 1.0) if len(nav_b) >= 2 else np.nan
        a1 = r1w - r1b

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
    # Scan helpers: quick "signal" label based on 30D alpha
    def _signal(a: float) -> str:
        if a is None or (isinstance(a, float) and math.isnan(a)):
            return "—"
        if a >= 0.05:
            return "🟢 Strong"
        if a >= 0.02:
            return "🟦 Outperform"
        if a <= -0.03:
            return "🔴 Lagging"
        return "⚪ Near BM"

    overview_df["30D Signal"] = overview_df["30D Alpha"].apply(_signal)

    # Display formatting
    show_overview = overview_df.copy()
    for col in ["1D Ret", "1D Alpha", "30D Ret", "30D Alpha", "60D Ret", "60D Alpha", "365D Ret", "365D Alpha"]:
        show_overview[col] = show_overview[col].apply(fmt_pct)

    cols_order = ["Wave", "30D Signal", "1D Ret", "1D Alpha", "30D Ret", "30D Alpha", "60D Ret", "60D Alpha", "365D Ret", "365D Alpha"]
    show_overview = show_overview[cols_order]

    st.dataframe(show_overview.set_index("Wave"), use_container_width=True, height=520 if not compact_tables else 420)

    st.markdown("---")

    # ============================================================
    # Risk Analytics
    # ============================================================
    st.markdown("### 🛡️ Risk Analytics (Vol, MaxDD, TE, IR) — 365D Window")
    risk_rows = []
    for w in all_waves:
        h = compute_wave_history(w, mode, 365)
        if h.empty or len(h) < 2:
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
            {
                "Wave": w,
                "Wave Vol": annualized_vol(ret_w),
                "Benchmark Vol": annualized_vol(ret_b),
                "Wave MaxDD": max_drawdown(nav_w),
                "Benchmark MaxDD": max_drawdown(nav_b),
                "Tracking Error": te,
                "Information Ratio": ir,
            }
        )

    risk_df = pd.DataFrame(risk_rows)
    show_risk = risk_df.copy()
    for col in ["Wave Vol", "Benchmark Vol", "Tracking Error", "Wave MaxDD", "Benchmark MaxDD"]:
        show_risk[col] = show_risk[col].apply(fmt_pct)
    show_risk["Information Ratio"] = show_risk["Information Ratio"].apply(lambda x: fmt_num(x, 2))

    st.dataframe(show_risk.set_index("Wave"), use_container_width=True, height=480 if not compact_tables else 380)

    st.markdown("---")

    # ============================================================
    # WaveScore Leaderboard
    # ============================================================
    st.markdown("### 🏁 WaveScore™ Leaderboard (Proto · 365D)")
    ws_df = compute_wavescore_table(all_waves, mode, 365)
    if ws_df is None or ws_df.empty:
        st.info("WaveScore table not available yet.")
    else:
        show_ws = ws_df.copy()
        show_ws["WaveScore"] = show_ws["WaveScore"].apply(fmt_score)
        for c in ["Return Quality", "Risk Control", "Consistency", "Resilience", "Efficiency"]:
            show_ws[c] = show_ws[c].apply(lambda x: fmt_num(x, 1))
        show_ws["Alpha_365D"] = show_ws["Alpha_365D"].apply(fmt_pct)
        show_ws["IR_365D"] = show_ws["IR_365D"].apply(lambda x: fmt_num(x, 2))

        st.dataframe(show_ws.set_index("Wave"), use_container_width=True, height=520 if not compact_tables else 420)

    st.markdown("---")

    # ============================================================
    # Benchmark transparency (all waves)
    # ============================================================
    st.markdown("### 🧩 Benchmark Transparency Table (Composite Components per Wave)")
    bm_mix = get_benchmark_mix()
    if bm_mix is None or bm_mix.empty:
        st.info("No benchmark mix table available from engine.")
    else:
        bm_show = bm_mix.copy()
        if "Weight" in bm_show.columns:
            bm_show["Weight"] = bm_show["Weight"].apply(lambda x: fmt_pct(float(x) if pd.notna(x) else np.nan))
        st.dataframe(bm_show, use_container_width=True, height=420 if not compact_tables else 340)

    st.markdown("---")

    # ============================================================
    # Wave Detail
    # ============================================================
    st.markdown(f"## 📌 Wave Detail — **{selected_wave}**  (Mode: **{mode}**)")
    h_sel = compute_wave_history(selected_wave, mode, history_days)

    # KPI header row (scan-fast)
    if not h_sel.empty and len(h_sel) >= 2:
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

        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("30D Return", fmt_pct(r30w))
        k2.metric("30D Alpha", fmt_pct(a30))
        k3.metric("365D Alpha", fmt_pct(a365))
        k4.metric("TE", fmt_pct(te))
        k5.metric("IR", fmt_num(ir, 2))

    c_left, c_right = st.columns([2.2, 1.0])

    with c_left:
        st.markdown("#### NAV vs Benchmark")
        if h_sel.empty or len(h_sel) < 2:
            st.warning("Not enough data to chart NAV.")
        else:
            nav_w = h_sel["wave_nav"]
            nav_b = h_sel["bm_nav"]
            plot_df = pd.DataFrame(
                {
                    "Wave_NAV": nav_w / float(nav_w.iloc[0]),
                    "Benchmark_NAV": nav_b / float(nav_b.iloc[0]),
                }
            )
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
        st.markdown("#### Risk (365D)")
        if h_sel.empty or len(h_sel) < 2:
            st.info("No metrics available.")
        else:
            ret_w = h_sel["wave_ret"]
            ret_b = h_sel["bm_ret"]
            nav_w = h_sel["wave_nav"]

            te = tracking_error(ret_w, ret_b)
            st.metric("Vol (ann.)", fmt_pct(annualized_vol(ret_w)))
            st.metric("Tracking Error", fmt_pct(te))
            st.metric("Information Ratio", fmt_num(information_ratio(h_sel["wave_nav"], h_sel["bm_nav"], te), 2))
            st.metric("Max Drawdown", fmt_pct(max_drawdown(nav_w)))

    st.markdown("---")

    # ============================================================
    # Benchmark Truth Panel
    # ============================================================
    with st.expander("✅ Benchmark Truth Panel (composition + difficulty + diagnostics)", expanded=expand_key_panels):
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
        bm_mix_df = get_benchmark_mix()
        if bm_mix_df is None or bm_mix_df.empty:
            st.warning("Benchmark mix table not available from engine.")
        else:
            wmix = bm_mix_df[bm_mix_df["Wave"] == selected_wave].copy() if "Wave" in bm_mix_df.columns else pd.DataFrame()
            if wmix.empty:
                st.warning("No benchmark components found for this Wave in the mix table.")
            else:
                if "Weight" in wmix.columns:
                    wmix = wmix.sort_values("Weight", ascending=False)
                    wmix["Weight"] = wmix["Weight"].apply(lambda x: fmt_pct(float(x) if pd.notna(x) else np.nan))
                cols = [c for c in ["Ticker", "Name", "Weight"] if c in wmix.columns]
                st.dataframe(wmix[cols], use_container_width=True)

        if not h_sel.empty and len(h_sel) >= 2:
            bm_vol = annualized_vol(h_sel["bm_ret"]) if "bm_ret" in h_sel.columns else np.nan
            bm_mdd = max_drawdown(h_sel["bm_nav"]) if "bm_nav" in h_sel.columns else np.nan
            r1, r2, r3 = st.columns(3)
            r1.metric("Benchmark Vol", fmt_pct(bm_vol))
            r2.metric("Benchmark MaxDD", fmt_pct(bm_mdd))
            r3.metric("Days in Window", str(int(len(h_sel))))

    st.markdown("---")

    # ============================================================
    # Mode Separation Proof
    # ============================================================
    with st.expander("✅ Mode Separation Proof (outputs + engine levers)", expanded=False):
        rows = []
        for m in all_modes:
            hh = compute_wave_history(selected_wave, m, 365)
            if hh.empty or len(hh) < 2:
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
        showm = dfm.copy()
        showm["Base Exposure"] = showm["Base Exposure"].apply(lambda x: fmt_num(x, 2))
        for col in ["365D Return", "365D Alpha"]:
            showm[col] = showm[col].apply(fmt_pct)
        showm["TE"] = showm["TE"].apply(fmt_pct)
        showm["IR"] = showm["IR"].apply(lambda x: fmt_num(x, 2))

        st.dataframe(showm.set_index("Mode"), use_container_width=True)
        st.caption("Demo-safe proof that mode toggles are real: shows outputs + lever settings (when exposed).")

    st.markdown("---")

    # ============================================================
    # Alpha Attribution
    # ============================================================
    with st.expander("🔍 Alpha Attribution (Engine vs Static Basket · 365D)", expanded=False):
        st.caption(
            "Compares engine NAV vs a static fixed-weight basket of the same holdings (no overlays). "
            "Baseline engine results remain unchanged."
        )
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

    st.markdown("---")

    # ============================================================
    # Wave Doctor + What-If Lab
    # ============================================================
    with st.expander("🩺 Wave Doctor™ (diagnostics + recommendations) + What-If Lab", expanded=expand_key_panels):
        assess = wave_doctor_assess(
            selected_wave,
            mode=mode,
            days=365,
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
            st.caption(
                "Use sliders to test hypothetical parameter changes. "
                "This computes a **shadow NAV** for insight only."
            )

            # Defaults
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
                freeze_bm = st.checkbox("Freeze benchmark to engine benchmark (comparison)", value=True)
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

                b_mdd = max_drawdown(b_nav)
                w_mdd = max_drawdown(w_nav)
                b_vol = annualized_vol(b_nav.pct_change().fillna(0.0))
                w_vol = annualized_vol(w_nav.pct_change().fillna(0.0))

                e1, e2, e3, e4 = st.columns(4)
                e1.metric("Baseline Vol", fmt_pct(b_vol))
                e2.metric("What-If Vol", fmt_pct(w_vol), delta=fmt_pct(w_vol - b_vol))
                e3.metric("Baseline MaxDD", fmt_pct(b_mdd))
                e4.metric("What-If MaxDD", fmt_pct(w_mdd), delta=fmt_pct(w_mdd - b_mdd))

                st.markdown("#### Baseline vs What-If NAV (normalized)")
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

    # ============================================================
    # Top 10 holdings
    # ============================================================
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

        if "Ticker" in hold.columns:
            hold["Google Finance"] = hold["Ticker"].astype(str).apply(google_finance_link)

        show = hold.copy()
        if "Weight" in show.columns:
            show["Weight"] = show["Weight"].apply(lambda x: fmt_pct(float(x) if pd.notna(x) else np.nan))

        cols = [c for c in ["Ticker", "Name", "Weight", "Google Finance"] if c in show.columns]
        st.dataframe(show[cols], use_container_width=True)


# ============================================================
# TAB 2: Market Intel
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
        snap_show = snap.copy()
        snap_show["Last"] = snap_show["Last"].apply(lambda x: fmt_num(x, 2))
        snap_show["1D Return"] = snap_show["1D Return"].apply(fmt_pct)
        snap_show["30D Return"] = snap_show["30D Return"].apply(fmt_pct)
        st.dataframe(snap_show.set_index("Ticker"), use_container_width=True)

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
    reaction_show = reaction.copy()
    reaction_show["30D Return"] = reaction_show["30D Return"].apply(fmt_pct)
    reaction_show["30D Alpha"] = reaction_show["30D Alpha"].apply(fmt_pct)
    st.dataframe(reaction_show.set_index("Wave"), use_container_width=True)


# ============================================================
# TAB 3: Factor Decomposition
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
                {
                    "Wave": w,
                    "β_SPY": betas.get("MKT_SPY", np.nan),
                    "β_QQQ": betas.get("GROWTH_QQQ", np.nan),
                    "β_IWM": betas.get("SIZE_IWM", np.nan),
                    "β_TLT": betas.get("RATES_TLT", np.nan),
                    "β_GLD": betas.get("GOLD_GLD", np.nan),
                    "β_BTC": betas.get("CRYPTO_BTC", np.nan),
                }
            )

        beta_df = pd.DataFrame(rows)
        beta_show = beta_df.copy()
        for col in ["β_SPY", "β_QQQ", "β_IWM", "β_TLT", "β_GLD", "β_BTC"]:
            beta_show[col] = beta_show[col].apply(lambda x: fmt_num(x, 2))
        st.dataframe(beta_show.set_index("Wave"), use_container_width=True)

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
# TAB 4: Vector OS Insight Layer
# ============================================================
with tab_vector:
    st.markdown("## 🤖 Vector OS Insight Layer — Rules-Based Narrative")
    st.caption("Narrative derived from current computed metrics (no external LLM calls).")

    # reuse ws_df from Console tab (already computed)
    try:
        ws_df_local = ws_df if ws_df is not None else pd.DataFrame()
    except Exception:
        ws_df_local = pd.DataFrame()

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
# Footer
# ============================================================
st.markdown("---")
st.caption(
    "WAVES Intelligence™ • Institutional Console (IRB-1) • Demo-safe analytics • Baseline engine results unchanged "
    f"• {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}"
)