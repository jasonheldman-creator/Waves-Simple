# app.py — WAVES Intelligence™ Institutional Console (Vector OS Edition)
# FULL PRODUCTION FILE (NO PATCHES)
#
# UI/UX Upgrade (NO ENGINE MATH CHANGES)
# --------------------------------------
# ✅ Wave-first layout: selecting a Wave immediately shows that Wave’s Snapshot first
# ✅ Selected Wave + Mode are obvious (premium header + sticky badge)
# ✅ Technical sections moved into expanders / Diagnostics tab (no longer bury the wave info)
# ✅ Top header redesigned to look sharp & sophisticated (less drab)
#
# Keeps IRB-1 feature set:
#   • Benchmark Truth
#   • Mode Separation Proof (engine-side)
#   • Attribution (Engine vs Static)
#   • Wave Doctor + What-If Lab (shadow simulation)
#   • Top-10 holdings + Google quote links
#   • Market Intel
#   • Factor Decomposition
#   • Vector OS Insight Layer
#
# Display change only:
#   • Returns/Alpha/Risk display as percentages (still stored as decimals; no math changes)

from __future__ import annotations

import math
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

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
    page_title="WAVES Intelligence™ Institutional Console",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# Premium UI CSS
# ============================================================
st.markdown(
    """
<style>
/* page padding */
.block-container { padding-top: 0.6rem; padding-bottom: 2.0rem; }

/* Premium top header */
.waves-hero {
  border-radius: 18px;
  padding: 18px 18px 14px 18px;
  margin: 0 0 12px 0;
  border: 1px solid rgba(255,255,255,0.10);
  background:
    radial-gradient(1400px 500px at 15% 0%, rgba(0,255,170,0.10), transparent 55%),
    radial-gradient(900px 420px at 80% 15%, rgba(0,140,255,0.12), transparent 55%),
    linear-gradient(135deg, rgba(8,10,26,0.92), rgba(10,18,40,0.82));
  box-shadow: 0 18px 50px rgba(0,0,0,0.35);
}

.waves-hero-title {
  font-size: 1.35rem;
  font-weight: 900;
  letter-spacing: 0.2px;
  margin: 0;
}

.waves-hero-sub {
  margin: 4px 0 0 0;
  opacity: 0.85;
  font-size: 0.95rem;
}

/* Big selected badge */
.waves-badge {
  display: inline-block;
  padding: 8px 12px;
  border-radius: 999px;
  border: 1px solid rgba(255,255,255,0.14);
  background: rgba(255,255,255,0.06);
  font-weight: 800;
  letter-spacing: 0.2px;
}

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

.waves-hdr { font-weight: 900; letter-spacing: 0.2px; margin-bottom: 4px; }

/* Tighter tables */
div[data-testid="stDataFrame"] { border-radius: 12px; overflow: hidden; }

/* Reduce whitespace for mobile */
@media (max-width: 700px) {
  .block-container { padding-left: 0.8rem; padding-right: 0.8rem; }
  .waves-hero-title { font-size: 1.15rem; }
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

# ============================================================
# Attribution: Engine vs Static Basket
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
# WaveScore proto v1.0 (console-side summary only; no spec changes)
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
            rows.append({"Wave": wave, "WaveScore": float("nan"), "Grade": "N/A",
                        "Return Quality": float("nan"), "Risk Control": float("nan"), "Consistency": float("nan"),
                        "Resilience": float("nan"), "Efficiency": float("nan"), "Transparency": 10.0,
                        "IR_365D": float("nan"), "Alpha_365D": float("nan")})
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

        mdd_wave = max_drawdown(nav_wave)
        mdd_bm = max_drawdown(nav_bm)
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

        rows.append({"Wave": wave, "WaveScore": total, "Grade": grade,
                     "Return Quality": return_quality, "Risk Control": risk_control, "Consistency": consistency,
                     "Resilience": resilience, "Efficiency": efficiency, "Transparency": transparency,
                     "IR_365D": ir, "Alpha_365D": alpha_365})

    df = pd.DataFrame(rows) if rows else pd.DataFrame()
    return df.sort_values("Wave") if not df.empty else df

# ============================================================
# Display formatting (percent display, no math changes)
# ============================================================
def build_formatter_map(df: pd.DataFrame) -> Dict[str, Any]:
    if df is None or df.empty:
        return {}
    fmt: Dict[str, Any] = {}
    pct_keywords = [" Ret", " Return", " Alpha", "Vol", "Volatility", "MaxDD", "Max Drawdown",
                    "Tracking Error", "TE", "Benchmark Difficulty", "BM Difficulty"]

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

def show_df(df: pd.DataFrame, key: str):
    if df is None:
        return
    try:
        fmt_map = build_formatter_map(df)
        if fmt_map:
            st.dataframe(df.style.format(fmt_map), use_container_width=True, key=key)
        else:
            st.dataframe(df, use_container_width=True, key=key)
    except Exception:
        st.dataframe(df, use_container_width=True, key=key)

# ============================================================
# Alpha matrix (heatmap source)
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
    fig = go.Figure(data=go.Heatmap(z=z, x=x, y=y, zmin=-v, zmax=v, colorbar=dict(title="Alpha")))
    fig.update_layout(title=title, height=min(900, 240 + 22 * max(10, len(y))),
                      margin=dict(l=80, r=40, t=60, b=40),
                      xaxis_title="Timeframe", yaxis_title="Wave")
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

    flags: List[str] = []
    diagnosis: List[str] = []
    recs: List[str] = []

    if pd.notna(a30) and abs(a30) >= alpha_warn:
        flags.append("Large 30D alpha (verify benchmark + coverage)")
        diagnosis.append("30D alpha is unusually large. This can happen from true signal, but also from benchmark mix or data coverage shifts.")
        recs.append("Freeze benchmark mix for demo comparisons and check benchmark drift in the Benchmark Truth Panel.")

    if pd.notna(a365) and a365 < 0:
        diagnosis.append("365D alpha is negative. This might reflect true underperformance, or a tougher benchmark composition.")
        if pd.notna(bm_difficulty) and bm_difficulty > 0.03:
            diagnosis.append("Benchmark difficulty is positive vs SPY (benchmark outperformed SPY), so alpha is harder on this window.")
            recs.append("For validation, temporarily compare to SPY/QQQ-style benchmark to isolate engine effect.")
    else:
        if pd.notna(a365) and a365 > 0.06:
            diagnosis.append("365D alpha is solidly positive.")
            recs.append("Lock benchmark mix snapshot for reproducibility in demos.")

    if pd.notna(te) and te > te_warn:
        flags.append("High tracking error (active risk elevated)")
        diagnosis.append("Tracking error is high; the wave behaves very differently than its benchmark.")
        recs.append("Reduce tilt strength and/or tighten exposure caps (What-If Lab) to lower TE.")

    if pd.notna(vol_w) and vol_w > vol_warn:
        flags.append("High volatility")
        diagnosis.append("Volatility is elevated.")
        recs.append("Lower vol target in What-If Lab (shadow only).")

    if pd.notna(mdd_w) and mdd_w < mdd_warn:
        flags.append("Deep drawdown")
        diagnosis.append("Drawdown is deep versus typical institutional tolerances.")
        recs.append("Increase SmartSafe posture in stress regimes (What-If Lab).")

    if pd.notna(vol_b) and pd.notna(vol_w) and vol_b > 0 and (vol_w / vol_b) > 1.6:
        flags.append("Volatility much higher than benchmark")
        diagnosis.append("Wave volatility is much higher than its benchmark; this can inflate wins and losses.")
        recs.append("Tighten exposure caps + reduce tilt strength to stabilize.")

    if not diagnosis:
        diagnosis.append("No major anomalies detected on the selected window.")

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
# Sidebar
# ============================================================
try:
    all_waves = we.get_all_waves()
    if all_waves is None:
        all_waves = []
except Exception:
    all_waves = []

try:
    all_modes = we.get_modes()
    if all_modes is None:
        all_modes = []
except Exception:
    all_modes = []

if "selected_wave" not in st.session_state:
    st.session_state["selected_wave"] = all_waves[0] if all_waves else ""
if "mode" not in st.session_state:
    st.session_state["mode"] = all_modes[0] if all_modes else "Standard"

with st.sidebar:
    st.title("WAVES Intelligence™")
    st.caption("Institutional Console • Vector OS™")

    if all_modes:
        st.selectbox("Mode", all_modes,
                     index=all_modes.index(st.session_state["mode"]) if st.session_state["mode"] in all_modes else 0,
                     key="mode")
    else:
        st.text_input("Mode", value=st.session_state.get("mode", "Standard"), key="mode")

    if all_waves:
        st.selectbox("Select Wave", all_waves,
                     index=all_waves.index(st.session_state["selected_wave"]) if st.session_state["selected_wave"] in all_waves else 0,
                     key="selected_wave")
    else:
        st.text_input("Select Wave", value=st.session_state.get("selected_wave", ""), key="selected_wave")

    st.markdown("---")
    st.markdown("**Display settings**")
    history_days = st.slider("History window (days)", min_value=60, max_value=730, value=365, step=15)

    st.markdown("---")
    st.markdown("**Wave Doctor settings**")
    alpha_warn = st.slider("Alpha alert threshold (abs, 30D)", 0.02, 0.20, 0.08, 0.01)
    te_warn = st.slider("Tracking error alert threshold", 0.05, 0.40, 0.20, 0.01)

mode = st.session_state["mode"]
selected_wave = st.session_state["selected_wave"]

if not selected_wave:
    st.error("No Waves available (engine returned empty list). Verify waves_engine.get_all_waves().")
    st.stop()

# ============================================================
# Premium top header (clear wave + mode)
# ============================================================
st.markdown(
    f"""
<div class="waves-hero">
  <div class="waves-hero-title">WAVES Intelligence™ Institutional Console</div>
  <div class="waves-hero-sub">
    <span class="waves-badge">Selected: {selected_wave} · {mode}</span>
    &nbsp;&nbsp;•&nbsp;&nbsp;Live Alpha Capture · SmartSafe™ · Multi-Asset · Crypto · Gold · Income Ladders
  </div>
</div>
""",
    unsafe_allow_html=True,
)

# ============================================================
# Sticky Summary Bar (quick scan)
# ============================================================
h_bar = compute_wave_history(selected_wave, mode=mode, days=max(365, history_days))
bar_r30 = bar_a30 = bar_r365 = bar_a365 = float("nan")
bar_te = bar_ir = float("nan")

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
  <span class="waves-chip">WaveScore: <b>{ws_val}</b> ({ws_grade})</span>
</div>
""",
    unsafe_allow_html=True,
)

# ============================================================
# Tabs (Wave-first)
# ============================================================
tab_wave, tab_console, tab_diagnostics, tab_market, tab_factors, tab_vector = st.tabs(
    ["Wave Snapshot", "Console Scan", "Diagnostics", "Market Intel", "Factor Decomposition", "Vector OS"]
)

# ============================================================
# TAB: Wave Snapshot (SHOW THIS FIRST, CLEAN)
# ============================================================
with tab_wave:
    st.subheader(f"📍 {selected_wave} — Snapshot")
    st.caption("This is the wave-first view. Technical proof/attribution is in Diagnostics.")

    hist = compute_wave_history(selected_wave, mode=mode, days=max(365, history_days))
    if hist.empty or len(hist) < 2:
        st.warning("Not enough history data returned for this wave.")
    else:
        nav_w = hist["wave_nav"]
        nav_b = hist["bm_nav"]
        ret_w = hist["wave_ret"]
        ret_b = hist["bm_ret"]

        # returns + alpha
        r1w = float(nav_w.iloc[-1] / nav_w.iloc[-2] - 1.0) if len(nav_w) >= 2 else np.nan
        r1b = float(nav_b.iloc[-1] / nav_b.iloc[-2] - 1.0) if len(nav_b) >= 2 else np.nan
        a1 = r1w - r1b if pd.notna(r1w) and pd.notna(r1b) else np.nan

        r30w = ret_from_nav(nav_w, min(30, len(nav_w)))
        r30b = ret_from_nav(nav_b, min(30, len(nav_b)))
        a30 = r30w - r30b

        r60w = ret_from_nav(nav_w, min(60, len(nav_w)))
        r60b = ret_from_nav(nav_b, min(60, len(nav_b)))
        a60 = r60w - r60b

        r365w = ret_from_nav(nav_w, len(nav_w))
        r365b = ret_from_nav(nav_b, len(nav_b))
        a365 = r365w - r365b

        te = tracking_error(ret_w, ret_b)
        ir = information_ratio(nav_w, nav_b, te)
        vol_w = annualized_vol(ret_w)
        mdd_w = max_drawdown(nav_w)

        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("1D Return", fmt_pct(r1w), fmt_pct(a1) + " α")
        c2.metric("30D Return", fmt_pct(r30w), fmt_pct(a30) + " α")
        c3.metric("60D Return", fmt_pct(r60w), fmt_pct(a60) + " α")
        c4.metric("365D Return", fmt_pct(r365w), fmt_pct(a365) + " α")
        c5.metric("Tracking Error", fmt_pct(te), "Active risk")
        c6.metric("IR", fmt_num(ir, 2), "Risk-adjusted")

        st.markdown("---")

        # NAV chart
        st.markdown("### Performance (NAV)")
        nav_df = pd.DataFrame({"Wave NAV": nav_w, "Benchmark NAV": nav_b}).dropna()
        if go is not None and not nav_df.empty:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=nav_df.index, y=nav_df["Wave NAV"], name="Wave", mode="lines"))
            fig.add_trace(go.Scatter(x=nav_df.index, y=nav_df["Benchmark NAV"], name="Benchmark", mode="lines"))
            fig.update_layout(height=360, margin=dict(l=40, r=40, t=40, b=40))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.line_chart(nav_df)

        # holdings
        st.markdown("### Top Holdings (with Google links)")
        hold = get_wave_holdings(selected_wave)
        if hold is None or hold.empty:
            st.info("Holdings not available for this wave.")
        else:
            hold = hold.copy()
            if "Weight" in hold.columns:
                hold["Weight"] = pd.to_numeric(hold["Weight"], errors="coerce").fillna(0.0)
            if "Ticker" in hold.columns:
                hold["Google"] = hold["Ticker"].astype(str).apply(lambda t: google_quote_url(t))
            # top 10
            if "Weight" in hold.columns:
                hold = hold.sort_values("Weight", ascending=False)
            top10 = hold.head(10)

            # show as table + links
            if "Google" in top10.columns:
                disp = top10[["Ticker", "Name", "Weight", "Google"]].copy() if "Name" in top10.columns else top10[["Ticker", "Weight", "Google"]].copy()
            else:
                disp = top10

            # show pretty weights
            if "Weight" in disp.columns:
                disp["Weight"] = disp["Weight"].astype(float)

            show_df(disp, key="top10_table")

            if "Ticker" in top10.columns:
                links = []
                for _, r in top10.iterrows():
                    t = str(r.get("Ticker", ""))
                    if not t:
                        continue
                    links.append(f"- [{t}]({google_quote_url(t)})")
                if links:
                    st.markdown("**Quick links**")
                    st.markdown("\n".join(links))

        # quick risk strip
        st.markdown("---")
        st.markdown("### Risk Snapshot")
        rc1, rc2, rc3 = st.columns(3)
        rc1.metric("Ann. Vol", fmt_pct(vol_w), "Wave")
        rc2.metric("Max Drawdown", fmt_pct(mdd_w), "Wave")
        rc3.metric("Benchmark Source", bar_src, "Truth label")

        with st.expander("Benchmark Truth (show me the mix)", expanded=False):
            bm_mix = get_benchmark_mix()
            if bm_mix is None or bm_mix.empty:
                st.info("Benchmark mix table not available.")
            else:
                # filter to this wave if possible
                if "Wave" in bm_mix.columns:
                    bm_sub = bm_mix[bm_mix["Wave"].astype(str) == str(selected_wave)].copy()
                else:
                    bm_sub = bm_mix.copy()
                show_df(bm_sub, key="bm_mix_table")

# ============================================================
# TAB: Console Scan (heatmap + all waves overview)
# ============================================================
with tab_console:
    st.subheader("🔥 Alpha Heatmap (All Waves × Timeframe)")
    st.caption("Fast scan. This view is for cross-wave scanning — your selected wave details are in Wave Snapshot.")

    alpha_df = build_alpha_matrix(all_waves, mode)
    plot_alpha_heatmap(alpha_df, selected_wave, title=f"Alpha Heatmap — Mode: {mode}")

    st.markdown("---")
    st.subheader("🧾 All Waves Overview (1D / 30D / 60D / 365D Returns + Alpha)")

    overview_rows: List[Dict[str, Any]] = []
    for wname in all_waves:
        hist = compute_wave_history(wname, mode=mode, days=365)
        if hist.empty or len(hist) < 2:
            overview_rows.append({"Wave": wname,
                                 "1D Ret": np.nan, "1D Alpha": np.nan,
                                 "30D Ret": np.nan, "30D Alpha": np.nan,
                                 "60D Ret": np.nan, "60D Alpha": np.nan,
                                 "365D Ret": np.nan, "365D Alpha": np.nan})
            continue

        nav_w = hist["wave_nav"]
        nav_b = hist["bm_nav"]

        if len(nav_w) >= 2 and len(nav_b) >= 2:
            r1w = float(nav_w.iloc[-1] / nav_w.iloc[-2] - 1.0)
            r1b = float(nav_b.iloc[-1] / nav_b.iloc[-2] - 1.0)
            a1 = r1w - r1b
        else:
            r1w, a1 = np.nan, np.nan

        r30w = ret_from_nav(nav_w, min(30, len(nav_w)))
        r30b = ret_from_nav(nav_b, min(30, len(nav_b)))
        a30 = r30w - r30b

        r60w = ret_from_nav(nav_w, min(60, len(nav_w)))
        r60b = ret_from_nav(nav_b, min(60, len(nav_b)))
        a60 = r60w - r60b

        r365w = ret_from_nav(nav_w, len(nav_w))
        r365b = ret_from_nav(nav_b, len(nav_b))
        a365 = r365w - r365b

        overview_rows.append({"Wave": wname,
                              "1D Ret": r1w, "1D Alpha": a1,
                              "30D Ret": r30w, "30D Alpha": a30,
                              "60D Ret": r60w, "60D Alpha": a60,
                              "365D Ret": r365w, "365D Alpha": a365})

    overview_df = pd.DataFrame(overview_rows).sort_values("365D Alpha", ascending=False)
    show_df(overview_df, key="overview_table")

# ============================================================
# TAB: Diagnostics (technical proof moved here)
# ============================================================
with tab_diagnostics:
    st.subheader("🧪 Diagnostics (Technical Proof & Tools)")
    st.caption("All the technical panels live here so the Wave Snapshot stays clean and readable.")

    # Attribution
    with st.expander("Alpha Attribution — Engine vs Static Basket", expanded=True):
        attr = compute_alpha_attribution(selected_wave, mode=mode, days=max(365, history_days))
        if not attr:
            st.info("Attribution not available.")
        else:
            adf = pd.DataFrame([{"Metric": k, "Value": v} for k, v in attr.items()])
            # format values: percent where appropriate
            def _pretty_metric_row(row):
                m = str(row["Metric"])
                v = row["Value"]
                if any(x in m for x in ["Return", "Alpha", "Contribution", "Vol", "MaxDD", "TE", "Difficulty"]):
                    return fmt_pct(v)
                if "IR" in m:
                    return fmt_num(v, 2)
                return fmt_num(v, 4)

            adf["Display"] = adf.apply(_pretty_metric_row, axis=1)
            st.dataframe(adf[["Metric", "Display"]], use_container_width=True)

    # Wave Doctor
    with st.expander("Wave Doctor", expanded=False):
        doc = wave_doctor_assess(selected_wave, mode=mode, days=max(365, history_days), alpha_warn=alpha_warn, te_warn=te_warn)
        if not doc.get("ok"):
            st.warning(doc.get("message", "Wave Doctor unavailable."))
        else:
            m = doc["metrics"]
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("365D Return", fmt_pct(m["Return_365D"]), fmt_pct(m["Alpha_365D"]) + " α")
            c2.metric("30D Return", fmt_pct(m["Return_30D"]), fmt_pct(m["Alpha_30D"]) + " α")
            c3.metric("TE", fmt_pct(m["TE"]), "Active risk")
            c4.metric("IR", fmt_num(m["IR"], 2), "Risk-adjusted")

            if doc.get("flags"):
                st.error("Flags:\n" + "\n".join([f"- {x}" for x in doc["flags"]]))
            st.markdown("**Diagnosis**")
            st.markdown("\n".join([f"- {x}" for x in doc.get("diagnosis", [])]))
            if doc.get("recommendations"):
                st.markdown("**Recommendations**")
                st.markdown("\n".join([f"- {x}" for x in doc["recommendations"]]))

# ============================================================
# TAB: Market Intel
# ============================================================
with tab_market:
    st.subheader("📈 Market Intel")
    mkt = fetch_market_assets(days=max(365, history_days))
    if mkt is None or mkt.empty:
        st.info("Market intel data unavailable.")
    else:
        st.caption("Daily prices (auto-adjusted).")
        st.line_chart(mkt)

# ============================================================
# TAB: Factor Decomposition
# ============================================================
with tab_factors:
    st.subheader("🧩 Factor Decomposition (Light)")
    st.caption("Simple multi-factor regression proxy (for directional read).")

    hist = compute_wave_history(selected_wave, mode=mode, days=max(365, history_days))
    if hist.empty or len(hist) < 60:
        st.info("Need more history to estimate factor betas.")
    else:
        # Simple factor set using available ETFs (proxy)
        factors_px = fetch_prices_daily(["SPY", "QQQ", "IWM", "TLT", "GLD"], days=max(365, history_days))
        if factors_px.empty:
            st.info("Factor proxy prices unavailable.")
        else:
            f_ret = factors_px.pct_change().dropna()
            wave_ret = hist["wave_ret"].reindex(f_ret.index).dropna()

            f_ret = f_ret.reindex(wave_ret.index).dropna()
            wave_ret = wave_ret.reindex(f_ret.index).dropna()

            betas = regress_factors(wave_ret, f_ret.rename(columns={
                "SPY": "Market (SPY)",
                "QQQ": "Growth (QQQ)",
                "IWM": "Small (IWM)",
                "TLT": "Rates (TLT)",
                "GLD": "Gold (GLD)",
            }))
            bdf = pd.DataFrame([{"Factor": k, "Beta": v} for k, v in betas.items()])
            st.dataframe(bdf, use_container_width=True)

# ============================================================
# TAB: Vector OS Insight Layer
# ============================================================
with tab_vector:
    st.subheader("🧠 Vector OS — Insight Layer")
    st.caption("This is a placeholder panel for the Vector narrative + decision layer.")
    st.markdown(
        f"""
**Selected Wave:** `{selected_wave}`  
**Mode:** `{mode}`  

**What Vector sees right now (quick read):**
- Regime: **{reg_now}**
- 30D alpha: **{fmt_pct(bar_a30)}**
- 365D alpha: **{fmt_pct(bar_a365)}**
- TE: **{fmt_pct(bar_te)}**
- IR: **{fmt_num(bar_ir, 2)}**

Next: We can add “Vector Notes” (editable) and auto-generated talking points for demos.
"""
    )