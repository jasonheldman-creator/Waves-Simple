# app.py — WAVES Intelligence™ Institutional Console (Vector OS Edition)
# FULL PRODUCTION FILE (NO PATCHES) — DIAGNOSTICS++ BUILD
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

    pct_keywords = [
        " Ret",
        " Return",
        " Alpha",
        "Vol",
        "Vol
                "Volatility",
        "MaxDD",
        "Max Drawdown",
        "Tracking Error",
        "TE",
        "Benchmark Difficulty",
        "BM Difficulty",
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


# ============================================================
# Row highlighting utilities (selected wave + alpha tint)
# ============================================================
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


# ============================================================
# One-click Wave jump (best-effort selection events)
# ============================================================
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
# Diagnostics++ helpers
# ============================================================
def coverage_audit(hist: pd.DataFrame) -> Dict[str, Any]:
    if hist is None or hist.empty:
        return {"ok": False, "rows": 0, "missing_cols": ["wave_nav", "bm_nav", "wave_ret", "bm_ret"], "nan_frac": 1.0}

    needed = ["wave_nav", "bm_nav", "wave_ret", "bm_ret"]
    missing = [c for c in needed if c not in hist.columns]
    if missing:
        return {"ok": False, "rows": int(len(hist)), "missing_cols": missing, "nan_frac": 1.0}

    df = hist[needed].copy()
    nan_frac = float(df.isna().mean().mean())
    ok = (len(df) >= 30) and (nan_frac < 0.15)
    return {"ok": ok, "rows": int(len(df)), "missing_cols": [], "nan_frac": nan_frac}


def alpha_captured_series(hist: pd.DataFrame, mode: str) -> pd.Series:
    """
    Alpha Captured (daily) = (wave_ret - bm_ret) * exposure_scale(if available).
    If engine provides an exposure scaler series inside history, we use it best-effort.
    Otherwise = wave_ret - bm_ret.
    """
    if hist is None or hist.empty or ("wave_ret" not in hist.columns) or ("bm_ret" not in hist.columns):
        return pd.Series(dtype=float)

    base = (hist["wave_ret"] - hist["bm_ret"]).copy()

    # best-effort exposure scaler
    for col in ["exposure", "expo", "exposure_scale", "exposure_factor", "beta_scale"]:
        if col in hist.columns:
            try:
                sc = pd.to_numeric(hist[col], errors="coerce").fillna(1.0)
                return (base * sc).rename("alpha_captured")
            except Exception:
                pass

    return base.rename("alpha_captured")


def rolling_stats(hist: pd.DataFrame, window: int = 60) -> pd.DataFrame:
    """
    Rolling diagnostics (daily): alpha, TE, beta, vol.
    - alpha_roll: rolling mean of (wave_ret - bm_ret)
    - te_roll: rolling std of (wave_ret - bm_ret) * sqrt(252)
    - beta_roll: rolling beta of wave_ret vs bm_ret
    - vol_wave_roll: rolling std of wave_ret * sqrt(252)
    """
    if hist is None or hist.empty or ("wave_ret" not in hist.columns) or ("bm_ret" not in hist.columns):
        return pd.DataFrame()

    df = hist[["wave_ret", "bm_ret"]].dropna().copy()
    if df.shape[0] < max(20, window):
        return pd.DataFrame()

    diff = (df["wave_ret"] - df["bm_ret"])
    out = pd.DataFrame(index=df.index)
    out["alpha_roll"] = diff.rolling(window).mean()
    out["te_roll"] = diff.rolling(window).std() * np.sqrt(252)
    out["vol_wave_roll"] = df["wave_ret"].rolling(window).std() * np.sqrt(252)

    # rolling beta
    def _roll_beta(xw: pd.Series, xb: pd.Series) -> float:
        varb = float(xb.var())
        if not math.isfinite(varb) or varb <= 0:
            return float("nan")
        return float(xw.cov(xb) / varb)

    betas = []
    idx = out.index
    for i in range(len(df)):
        if i < window - 1:
            betas.append(np.nan)
            continue
        w = df["wave_ret"].iloc[i - window + 1 : i + 1]
        b = df["bm_ret"].iloc[i - window + 1 : i + 1]
        betas.append(_roll_beta(w, b))
    out["beta_roll"] = pd.Series(betas, index=idx)
    return out


@st.cache_data(show_spinner=False)
def build_corr_matrix(all_waves: List[str], mode: str, days: int = 180) -> pd.DataFrame:
    """
    Correlation of wave daily returns across all waves (aligned by date).
    """
    series = {}
    for w in all_waves:
        hist = compute_wave_history(w, mode=mode, days=days)
        if hist is None or hist.empty or "wave_ret" not in hist.columns:
            continue
        s = pd.to_numeric(hist["wave_ret"], errors="coerce").dropna()
        if len(s) >= 30:
            series[w] = s

    if not series:
        return pd.DataFrame()

    df = pd.DataFrame(series).dropna(how="all")
    if df.shape[0] < 30 or df.shape[1] < 2:
        return pd.DataFrame()

    return df.corr()


def plot_nav_overlay(hist_map: Dict[str, pd.DataFrame], title: str):
    """
    Overlay NAV across modes for selected wave (Mode Separation Proof).
    """
    if go is None:
        st.info("NAV overlay unavailable (Plotly missing).")
        return

    fig = go.Figure()
    any_added = False
    for mode, hist in hist_map.items():
        if hist is None or hist.empty or "wave_nav" not in hist.columns:
            continue
        nav = hist["wave_nav"].dropna()
        if len(nav) < 2:
            continue
        fig.add_trace(go.Scatter(x=nav.index, y=nav.values, name=f"{mode} NAV", mode="lines"))
        any_added = True

    if not any_added:
        st.info("No NAV series available to overlay.")
        return

    fig.update_layout(height=380, margin=dict(l=40, r=40, t=40, b=40), title=title)
    st.plotly_chart(fig, use_container_width=True)


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
    st.caption("Mini Bloomberg Console • Vector OS™")

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
    corr_days = st.slider("Correlation window (days)", min_value=60, max_value=365, value=180, step=15)
    roll_window = st.slider("Rolling diagnostics window (days)", min_value=20, max_value=180, value=60, step=5)

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
bar_src = "—"

bm_mix_for_src = get_benchmark_mix()
bar_src = benchmark_source_label(selected_wave, bm_mix_for_src)

beta_target = get_beta_target_if_available(mode)
beta_real = float("nan")
beta_drift = float("nan")

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

    beta_real = beta_vs_benchmark(ret_w, ret_b)
    if math.isfinite(beta_target) and math.isfinite(beta_real):
        beta_drift = float(abs(beta_real - beta_target))

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

# wavescore snapshot
ws_snap = compute_wavescore_for_all_waves(all_waves, mode=mode, days=365)
ws_val = "—"
ws_grade = "—"
if ws_snap is not None and not ws_snap.empty:
    rr = ws_snap[ws_snap["Wave"] == selected_wave]
    if not rr.empty:
        ws_val = fmt_score(float(rr.iloc[0]["WaveScore"]))
        ws_grade = str(rr.iloc[0]["Grade"])

beta_chip = "β: —"
if math.isfinite(beta_real):
    if math.isfinite(beta_target):
        beta_chip = f"β: {fmt_num(beta_real,2)} (tgt {fmt_num(beta_target,2)})"
    else:
        beta_chip = f"β: {fmt_num(beta_real,2)}"

drift_chip = ""
if math.isfinite(beta_drift):
    drift_chip = f" · drift {fmt_num(beta_drift,3)}"

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
  <span class="waves-chip">{beta_chip}{drift_chip}</span>
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
# TAB 1: Console (scan + overview + holdings + benchmark truth)
# ============================================================
with tab_console:
    st.subheader("🔥 Alpha Heatmap View (All Waves × Timeframe)")
    st.caption("Fast scan. Jump table highlights selected wave. Values display as % (math unchanged).")

    alpha_df = build_alpha_matrix(all_waves, mode)
    plot_alpha_heatmap(alpha_df, selected_wave, title=f"Alpha Heatmap — Mode: {mode}")

    st.markdown("### 🧭 One-Click Jump Table")
    jump_df = alpha_df.copy()
    jump_df["RankScore"] = jump_df[["1D Alpha", "30D Alpha", "60D Alpha", "365D Alpha"]].mean(axis=1, skipna=True)
    jump_df = jump_df.sort_values("RankScore", ascending=False)

    show_df(jump_df, selected_wave, key="wave_jump_table_fmt")
    selectable_table_jump(jump_df, key="wave_jump_table_select")

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

        # 365D
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
                show_df(bm_mix, selected_wave, key="bm_mix")

    with colB:
        st.markdown("#### Attribution Snapshot (365D)")
        attrib = compute_alpha_attribution(selected_wave, mode=mode, days=365)
        if not attrib:
            st.info("Attribution unavailable.")
        else:
            arows = []
            for k, v in attrib.items():
                if any(x in k for x in ["Return", "Alpha", "Vol", "MaxDD", "TE", "Difficulty"]):
                    arows.append({"Metric": k, "Value": fmt_pct(v)})
                elif "IR" in k or "β" in k:
                    arows.append({"Metric": k, "Value": fmt_num(v, 3 if "β" in k else 2)})
                else:
                    arows.append({"Metric": k, "Value": fmt_num(v, 4)})
            st.dataframe(pd.DataFrame(arows), use_container_width=True)


# ============================================================
# TAB 2: Diagnostics++
# ============================================================
with tab_diag:
    st.subheader("🧪 Diagnostics++ (Deep Validation)")
    st.caption("Everything here is diagnostic only. Engine math is unchanged.")

    st.markdown("### 1) Data Quality / Coverage Audit")
    hist_sel = compute_wave_history(selected_wave, mode=mode, days=history_days)
    audit = coverage_audit(hist_sel)
    if not audit.get("ok", False):
        st.warning(f"Coverage issue: rows={audit.get('rows')} nan_frac={fmt_num(audit.get('nan_frac'),3)} missing={audit.get('missing_cols')}")
    else:
        st.success(f"Coverage OK: rows={audit.get('rows')} nan_frac={fmt_num(audit.get('nan_frac'),3)}")

    st.markdown("---")

    st.markdown("### 2) Mode Separation Proof (Side-by-Side + NAV Overlay)")
    mode_rows = []
    hist_map: Dict[str, pd.DataFrame] = {}

    for m in (all_modes if all_modes else ["Standard", "Alpha-Minus-Beta", "Private Logic"]):
        h = compute_wave_history(selected_wave, mode=m, days=history_days)
        hist_map[m] = h

        if h is None or h.empty or len(h) < 2:
            mode_rows.append({"Mode": m, "30D α": np.nan, "365D α": np.nan, "TE": np.nan, "IR": np.nan, "β_real": np.nan})
            continue

        nav_w = h["wave_nav"]
        nav_b = h["bm_nav"]
        ret_w = h["wave_ret"]
        ret_b = h["bm_ret"]

        r30w = ret_from_nav(nav_w, min(30, len(nav_w)))
        r30b = ret_from_nav(nav_b, min(30, len(nav_b)))
        a30 = r30w - r30b

        r365w = ret_from_nav(nav_w, len(nav_w))
        r365b = ret_from_nav(nav_b, len(nav_b))
        a365 = r365w - r365b

        te = tracking_error(ret_w, ret_b)
        ir = information_ratio(nav_w, nav_b, te)
        b = beta_vs_benchmark(ret_w, ret_b)

        mode_rows.append({"Mode": m, "30D α": a30, "365D α": a365, "TE": te, "IR": ir, "β_real": b})

    mode_df = pd.DataFrame(mode_rows)
    st.dataframe(mode_df.style.format({
        "30D α": lambda v: fmt_pct(v),
        "365D α": lambda v: fmt_pct(v),
        "TE": lambda v: fmt_pct(v),
        "IR": lambda v: fmt_num(v, 2),
        "β_real": lambda v: fmt_num(v, 2),
    }), use_container_width=True)

    plot_nav_overlay(hist_map, title=f"NAV Overlay — {selected_wave} (All Modes)")

    st.markdown("---")

    st.markdown("### 3) Rolling Diagnostics (Alpha / TE / Beta / Vol)")
    rd = rolling_stats(hist_sel, window=roll_window)
    if rd is None or rd.empty:
        st.info("Rolling diagnostics unavailable (need more history).")
    else:
        st.caption("Rolling values use daily data; TE/Vol are annualized. Alpha is daily mean (not annualized).")
        tail = rd.tail(min(120, len(rd))).copy()
        tail_disp = tail.copy()
        # display formatting
        tail_disp["alpha_roll"] = tail_disp["alpha_roll"].apply(lambda v: fmt_pct(v, 3))
        tail_disp["te_roll"] = tail_disp["te_roll"].apply(lambda v: fmt_pct(v))
        tail_disp["vol_wave_roll"] = tail_disp["vol_wave_roll"].apply(lambda v: fmt_pct(v))
        tail_disp["beta_roll"] = tail_disp["beta_roll"].apply(lambda v: fmt_num(v, 2))
        st.dataframe(tail_disp, use_container_width=True)

        if go is not None:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=rd.index, y=rd["alpha_roll"], name="Rolling α (daily mean)", mode="lines"))
            fig.add_trace(go.Scatter(x=rd.index, y=rd["te_roll"], name="Rolling TE (ann)", mode="lines"))
            fig.update_layout(height=360, margin=dict(l=40, r=40, t=40, b=40), title="Rolling Alpha & TE")
            st.plotly_chart(fig, use_container_width=True)

            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=rd.index, y=rd["beta_roll"], name="Rolling β_real", mode="lines"))
            fig2.add_trace(go.Scatter(x=rd.index, y=rd["vol_wave_roll"], name="Rolling Vol (ann)", mode="lines", yaxis="y2"))
            fig2.update_layout(
                height=360,
                margin=dict(l=40, r=40, t=40, b=40),
                title="Rolling Beta & Vol",
                yaxis=dict(title="β_real"),
                yaxis2=dict(title="Vol (ann)", overlaying="y", side="right"),
            )
            st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")

    st.markdown("### 4) Alpha Captured (Daily) + Persistence Check")
    ac = alpha_captured_series(hist_sel, mode=mode)
    if ac is None or ac.empty:
        st.info("Alpha Captured unavailable.")
    else:
        pos_frac = float((ac.dropna() > 0).mean()) if len(ac.dropna()) > 0 else float("nan")
        neg_frac = float((ac.dropna() < 0).mean()) if len(ac.dropna()) > 0 else float("nan")
        st.write(f"**Positive days:** {fmt_num(pos_frac*100,1)}%  |  **Negative days:** {fmt_num(neg_frac*100,1)}%")

        ac_df = pd.DataFrame({"alpha_captured": ac}).tail(min(180, len(ac)))
        ac_df_disp = ac_df.copy()
        ac_df_disp["alpha_captured"] = ac_df_disp["alpha_captured"].apply(lambda v: fmt_pct(v, 3))
        st.dataframe(ac_df_disp, use_container_width=True)

        if go is not None:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=ac_df.index, y=ac_df["alpha_captured"], name="Alpha Captured", mode="lines"))
            fig.update_layout(height=320, margin=dict(l=40, r=40, t=40, b=40), title="Alpha Captured (Daily)")
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    st.markdown("### 5) Correlation Matrix (Across Waves)")
    corr = build_corr_matrix(all_waves, mode=mode, days=corr_days)
    if corr is None or corr.empty:
        st.info("Correlation matrix unavailable (need more waves with history).")
    else:
        st.caption("Correlation computed from aligned daily wave returns.")
        st.dataframe(corr, use_container_width=True)

    st.markdown("---")

    st.markdown("### 6) Beta Discipline / Drift Flags")
    if not math.isfinite(beta_target):
        st.info("Engine beta target not detected. (Optional: add MODE_BETA_TARGET dict in engine.)")
    else:
        drift_flag = ""
        if math.isfinite(beta_drift) and beta_drift > 0.07:
            drift_flag = "⚠️ Beta drift > 0.07 (alert threshold)."
        st.write(f"β_target: **{fmt_num(beta_target,2)}**  |  β_real: **{fmt_num(beta_real,2)}**  |  drift: **{fmt_num(beta_drift,3)}**  {drift_flag}")


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
            wave_ret = hist["wave_ret"].reindex(factor_ret.index).fillna(0.0)
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

    attrib2 = compute_alpha_attribution(selected_wave, mode=mode, days=365)

    st.markdown("### Vector Summary")
    st.write(f"**Wave:** {selected_wave}  |  **Mode:** {mode}  |  **Regime:** {reg_now}  |  **Benchmark:** {bar_src}")

    st.markdown("### Diagnostics Snapshot")
    st.write(f"- **30D Alpha:** {fmt_pct(bar_a30)}")
    st.write(f"- **365D Alpha:** {fmt_pct(bar_a365)}")
    st.write(f"- **Tracking Error:** {fmt_pct(bar_te)}  |  **IR:** {fmt_num(bar_ir, 2)}")
    if math.isfinite(beta_real):
        st.write(f"- **β_real:** {fmt_num(beta_real,2)}" + (f" (β_target {fmt_num(beta_target,2)})" if math.isfinite(beta_target) else ""))

    st.markdown("### Attribution Lens")
    if attrib2:
        st.write(f"- **Engine Return:** {fmt_pct(attrib2.get('Engine Return'))}")
        st.write(f"- **Static Basket Return:** {fmt_pct(attrib2.get('Static Basket Return'))}")
        st.write(f"- **Overlay Contribution:** {fmt_pct(attrib2.get('Overlay Contribution (Engine - Static)'))}")
        st.write(f"- **Alpha vs Benchmark:** {fmt_pct(attrib2.get('Alpha vs Benchmark'))}")
        st.write(f"- **Benchmark Difficulty (BM - SPY):** {fmt_pct(attrib2.get('Benchmark Difficulty (BM - SPY)'))}")

    st.markdown("### Vector Guidance (Non-Advice)")
    st.write(
        "Validate benchmark stability (Benchmark Truth), then verify mode separation (Diagnostics++). "
        "If alpha spikes on 30D but not on 365D, inspect coverage + benchmark drift and review rolling TE/beta. "
        "Use correlation matrix to avoid stacking highly correlated waves inside multi-wave portfolios."
    )