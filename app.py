# app.py — WAVES Intelligence™ Institutional Console (Vector OS Edition)
# FULL PRODUCTION FILE (NO PATCHES) — GOVERNANCE FRAMING + PHASE 2 PLACEHOLDERS
#
# Keeps ALL IRB-1 features:
#   • Benchmark Truth (benchmark mix + difficulty vs SPY)
#   • Mode Separation Proof (mode shown + independent history per mode)
#   • Alpha Attribution (Engine vs Static Basket)
#   • Wave Doctor + What-If Lab (shadow simulation)
#   • Top-10 holdings with Google quote links
#   • Factor Decomposition (simple regression beta map)
#   • Vector OS Insight Layer (narrative + flags)
#
# Keeps expanded analytics:
#   • Risk Lab (Sharpe, Sortino, Downside Dev, VaR/CVaR, Rolling Alpha/Vol, Drawdowns, Persistence)
#   • Correlation tab (matrix, most/least correlated, optional ETFs)
#   • WaveScore leaderboard
#
# NEW (safe, non-destructive — UI/labeling only; engine math unchanged):
#   • Explicit "Institutional Portfolio Intelligence & Governance Layer" framing
#   • Governance mission statement + scope box (what this is / is not)
#   • Alpha Difficulty explanation (Benchmark Difficulty vs SPY) — no new math
#   • Phase 2 placeholders (planned features, NOT active)
#
# IMPORTANT:
#   • No engine math changes. No trading/execution logic. No new dependencies required.

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

# Plotly optional (avoid crashes if missing)
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
# NEW: Top-of-page Governance Framing (UI only)
# ============================================================
st.markdown(
    """
### WAVES Intelligence™
**Institutional Portfolio Intelligence & Governance Layer**

*Transparent attribution • Dynamic benchmark truth • Risk diagnostics • Committee-ready oversight*
---
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


def safe_df(df: Optional[pd.DataFrame]) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    return df.copy()


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


def beta_ols(y: pd.Series, x: pd.Series) -> float:
    """OLS beta of y vs x (daily returns)."""
    y = safe_series(y).astype(float)
    x = safe_series(x).astype(float)
    df = pd.concat([y.rename("y"), x.rename("x")], axis=1).dropna()
    if df.shape[0] < 20:
        return float("nan")
    vx = float(df["x"].var())
    if not math.isfinite(vx) or vx <= 0:
        return float("nan")
    cov = float(df["y"].cov(df["x"]))
    return float(cov / vx)


def sharpe_ratio(daily_ret: pd.Series, rf_annual: float = 0.0) -> float:
    """Sharpe using daily returns; rf_annual optional (default 0)."""
    r = safe_series(daily_ret).astype(float)
    if len(r) < 20:
        return float("nan")
    rf_daily = rf_annual / 252.0
    ex = r - rf_daily
    vol = float(ex.std())
    if not math.isfinite(vol) or vol <= 0:
        return float("nan")
    return float(ex.mean() / vol * np.sqrt(252))


def downside_deviation(daily_ret: pd.Series, mar_annual: float = 0.0) -> float:
    """Downside deviation vs MAR (annual) using daily returns."""
    r = safe_series(daily_ret).astype(float)
    if len(r) < 20:
        return float("nan")
    mar_daily = mar_annual / 252.0
    d = np.minimum(0.0, (r - mar_daily).values)
    dd = float(np.sqrt(np.mean(d**2)))
    return float(dd * np.sqrt(252))


def sortino_ratio(daily_ret: pd.Series, mar_annual: float = 0.0) -> float:
    r = safe_series(daily_ret).astype(float)
    if len(r) < 20:
        return float("nan")
    mar_daily = mar_annual / 252.0
    ex = float((r - mar_daily).mean()) * 252.0
    dd = downside_deviation(r, mar_annual=mar_annual)
    if not math.isfinite(dd) or dd <= 0:
        return float("nan")
    return float(ex / dd)


def var_cvar(daily_ret: pd.Series, level: float = 0.95) -> Tuple[float, float]:
    """Historical VaR/CVaR (returns; negative means loss)."""
    r = safe_series(daily_ret).astype(float)
    if len(r) < 50:
        return (float("nan"), float("nan"))
    q = float(np.quantile(r.values, 1.0 - level))
    tail = r[r <= q]
    cvar = float(tail.mean()) if len(tail) > 0 else float("nan")
    return (q, cvar)


def skew_kurt(daily_ret: pd.Series) -> Tuple[float, float]:
    r = safe_series(daily_ret).astype(float)
    if len(r) < 50:
        return (float("nan"), float("nan"))
    sk = float(pd.Series(r).skew())
    ku = float(pd.Series(r).kurtosis())
    return (sk, ku)
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
# Rolling analytics helpers
# ============================================================
def rolling_return_from_nav(nav: pd.Series, window: int) -> pd.Series:
    nav = safe_series(nav).astype(float)
    if len(nav) < window + 1:
        return pd.Series(dtype=float)
    return (nav / nav.shift(window) - 1.0).rename(f"ret_{window}")


def rolling_alpha_from_nav(wave_nav: pd.Series, bm_nav: pd.Series, window: int) -> pd.Series:
    w = rolling_return_from_nav(wave_nav, window)
    b = rolling_return_from_nav(bm_nav, window)
    df = pd.concat([w, b], axis=1).dropna()
    if df.empty:
        return pd.Series(dtype=float)
    out = (df.iloc[:, 0] - df.iloc[:, 1]).rename(f"alpha_{window}")
    return out


def rolling_vol(daily_ret: pd.Series, window: int = 20) -> pd.Series:
    r = safe_series(daily_ret).astype(float)
    if len(r) < window + 5:
        return pd.Series(dtype=float)
    return (r.rolling(window).std() * np.sqrt(252)).rename(f"vol_{window}")


def drawdown_series(nav: pd.Series) -> pd.Series:
    nav = safe_series(nav).astype(float)
    if len(nav) < 2:
        return pd.Series(dtype=float)
    peak = nav.cummax()
    dd = (nav / peak) - 1.0
    return dd.rename("drawdown")


def alpha_persistence(alpha_series: pd.Series) -> float:
    a = safe_series(alpha_series).dropna()
    if len(a) < 30:
        return float("nan")
    return float((a > 0).mean())


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

    out["Engine Return"] = float(eng_ret)
    out["Static Basket Return"] = float(static_ret) if pd.notna(static_ret) else float("nan")
    out["Overlay Contribution (Engine - Static)"] = float(overlay_pp) if pd.notna(overlay_pp) else float("nan")
    out["Benchmark Return"] = float(bm_ret_total)
    out["Alpha vs Benchmark"] = float(alpha_vs_bm)
    out["SPY Return"] = float(spy_ret) if pd.notna(spy_ret) else float("nan")
    out["Alpha vs SPY"] = float(alpha_vs_spy) if pd.notna(alpha_vs_spy) else float("nan")
    out["Benchmark Difficulty (BM - SPY)"] = float(benchmark_difficulty)
    out["Tracking Error (TE)"] = float(te)
    out["Information Ratio (IR)"] = float(ir)

    out["Wave Vol"] = float(annualized_vol(wave_ret))
    out["Benchmark Vol"] = float(annualized_vol(bm_ret))
    out["Wave MaxDD"] = float(max_drawdown(nav_wave))
    out["Benchmark MaxDD"] = float(max_drawdown(nav_bm))

    # extra risk stats (console-side only)
    out["Beta vs Benchmark"] = float(beta_ols(wave_ret, bm_ret))
    out["Sharpe (0% rf)"] = float(sharpe_ratio(wave_ret, rf_annual=0.0))
    out["Sortino (0% MAR)"] = float(sortino_ratio(wave_ret, mar_annual=0.0))
    v, c = var_cvar(wave_ret, level=0.95)
    out["VaR 95% (daily)"] = float(v)
    out["CVaR 95% (daily)"] = float(c)

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
        "Volatility",
        "MaxDD",
        "Max Drawdown",
        "Tracking Error",
        "TE",
        "Benchmark Difficulty",
        "BM Difficulty",
        "VaR",
        "CVaR",
        "Downside",
    ]

    for c in df.columns:
        cs = str(c)

        if cs == "WaveScore":
            fmt[c] = lambda v: fmt_score(v)
            continue
        if cs in ["Return Quality", "Risk Control", "Consistency", "Resilience", "Efficiency", "Transparency"]:
            fmt[c] = lambda v: fmt_num(v, 1)
            continue

        if cs.startswith("β") or cs.startswith("beta") or cs.startswith("β_") or "Beta" in cs:
            fmt[c] = lambda v: fmt_num(v, 2)
            continue

        if ("IR" in cs) and ("Ret" not in cs) and ("Return" not in cs):
            fmt[c] = lambda v: fmt_num(v, 2)
            continue

        if ("Sharpe" in cs) or ("Sortino" in cs):
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
    st.markdown("Display settings")
    history_days = st.slider("History window (days)", min_value=60, max_value=730, value=365, step=15)

    st.markdown("---")
    st.markdown("Wave Doctor settings")
    alpha_warn = st.slider("Alpha alert threshold (abs, 30D)", 0.02, 0.20, 0.08, 0.01)
    te_warn = st.slider("Tracking error alert threshold", 0.05, 0.40, 0.20, 0.01)

    st.markdown("---")
    st.markdown("Risk Lab settings")
    roll_alpha_window = st.selectbox("Rolling alpha window", [20, 30, 60, 90], index=1)
    roll_vol_window = st.selectbox("Rolling vol window", [10, 20, 30, 60], index=1)

mode = st.session_state["mode"]
selected_wave = st.session_state["selected_wave"]

if not selected_wave:
    st.error("No Waves available (engine returned empty list). Verify waves_engine.get_all_waves().")
    st.stop()


# ============================================================
# Governance Mission Statement (NEW - UI only)
# ============================================================
st.info(
    "This system is designed to make adaptive portfolios transparent, explainable, and governable at a committee level. "
    "All results are evaluated against dynamically constructed benchmarks with explicit difficulty context."
)

with st.expander("System Scope & Governance Posture", expanded=False):
    st.markdown(
        """
What this system is:
- Portfolio intelligence and governance layer
- Attribution, benchmark truth, and risk diagnostics
- Designed for institutional oversight and review

What this system is not:
- An autonomous trading system
- A black-box AI decision maker
- A custody, execution, or fund product
        """
    )

with st.expander("Phase 2 – Institutional Governance Extensions (Planned)", expanded=False):
    st.markdown(
        """
The following capabilities are planned extensions of the governance layer (not active in the current build):

- Scenario and stress testing across macro regimes
- Liquidity and capacity diagnostics for scalable mandates
- Audit-ready governance reports (exportable PDF/CSV)
- Decision logs and override tracking

These are intentionally not active in the current system.
        """
    )

# ============================================================
# Page header + Sticky summary (unchanged math)
# ============================================================
st.title("WAVES Intelligence™ Institutional Console")
st.caption("Live Alpha Capture • SmartSafe™ • Multi-Asset • Governance-Grade Transparency")

h_bar = compute_wave_history(selected_wave, mode=mode, days=max(365, history_days))
bar_r30 = bar_a30 = bar_r365 = bar_a365 = float("nan")
bar_te = bar_ir = float("nan")
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
ws_rank = "—"
if ws_snap is not None and not ws_snap.empty:
    ws_sorted = ws_snap.sort_values("WaveScore", ascending=False).reset_index(drop=True)
    rr = ws_sorted[ws_sorted["Wave"] == selected_wave]
    if not rr.empty:
        ws_val = fmt_score(float(rr.iloc[0]["WaveScore"]))
        ws_grade = str(rr.iloc[0]["Grade"])
        ws_rank = str(int(rr.index[0]) + 1)

st.markdown(
    f"""
<div class="waves-sticky">
  <div class="waves-hdr">Live Governance Summary</div>
  <span class="waves-chip">Role: <b>Intelligence & Governance Layer</b></span>
  <span class="waves-chip">Mode: <b>{mode}</b></span>
  <span class="waves-chip">Wave: <b>{selected_wave}</b></span>
  <span class="waves-chip">Benchmark: <b>{bar_src}</b></span>
  <span class="waves-chip">Regime: <b>{reg_now}</b></span>
  <span class="waves-chip">VIX: <b>{fmt_num(vix_last, 1) if not math.isnan(vix_last) else "—"}</b></span>
  <span class="waves-chip">30D alpha: <b>{fmt_pct(bar_a30)}</b> • 30D ret: <b>{fmt_pct(bar_r30)}</b></span>
  <span class="waves-chip">365D alpha: <b>{fmt_pct(bar_a365)}</b> • 365D ret: <b>{fmt_pct(bar_r365)}</b></span>
  <span class="waves-chip">TE: <b>{fmt_pct(bar_te)}</b> • IR: <b>{fmt_num(bar_ir, 2)}</b></span>
  <span class="waves-chip">WaveScore: <b>{ws_val}</b> ({ws_grade}) • Rank: <b>{ws_rank}</b></span>
</div>
""",
    unsafe_allow_html=True,
)

# ============================================================
# Tabs (Market Intel removed)
# ============================================================
tab_console, tab_factors, tab_risk, tab_corr, tab_vector = st.tabs(
    ["Console", "Factor Decomposition", "Risk Lab", "Correlation", "Vector OS Insight Layer"]
)

# ============================================================
# TAB 1: Console
# ============================================================
with tab_console:
    st.subheader("Alpha Heatmap View (All Waves x Timeframe)")
    st.caption("Fast scan. Values display as % (math unchanged).")

    alpha_df = build_alpha_matrix(all_waves, mode)
    plot_alpha_heatmap(alpha_df, selected_wave, title=f"Alpha Heatmap — Mode: {mode}")

    st.markdown("### One-Click Jump Table")
    jump_df = alpha_df.copy()
    jump_df["RankScore"] = jump_df[["1D Alpha", "30D Alpha", "60D Alpha", "365D Alpha"]].mean(axis=1, skipna=True)
    jump_df = jump_df.sort_values("RankScore", ascending=False)

    show_df(jump_df, selected_wave, key="wave_jump_table_fmt")
    selectable_table_jump(jump_df, key="wave_jump_table_select")

    st.markdown("---")

    st.subheader("WaveScore Leaderboard (Mode Snapshot)")
    if ws_snap is None or ws_snap.empty:
        st.info("WaveScore table unavailable.")
    else:
        lb = ws_snap.sort_values("WaveScore", ascending=False).reset_index(drop=True)
        lb.insert(0, "Rank", np.arange(1, len(lb) + 1))
        show_df(lb, selected_wave, key="wavescore_leaderboard")

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
    st.subheader("All Waves Overview (1D / 30D / 60D / 365D Returns + Alpha)")
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

    st.caption(
        "Alpha Difficulty Note: Benchmark Difficulty (BM - SPY) measures how hard alpha was to generate. "
        "Positive values indicate a stronger-than-market benchmark, making alpha harder to earn."
    )

    st.markdown("---")
    st.subheader(f"Selected Wave — {selected_wave}")
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

        st.markdown("### Top 10 Holdings (clickable)")
        for _, r in top10.iterrows():
            t = str(r.get("Ticker", ""))
            wgt = r.get("Weight", np.nan)
            nm = str(r.get("Name", t))
            if t:
                url = f"https://www.google.com/finance/quote/{t}"
                st.markdown(f"- **[{t}]({url})** — {nm} — **{fmt_pct(wgt)}**")

        st.markdown("### Full Holdings")
        show_df(hold2, selected_wave, key="holdings_full")

    st.markdown("---")
    st.subheader("Benchmark Truth + Attribution (Engine vs Basket)")
    colA, colB = st.columns(2)

    with colA:
        st.markdown("#### Benchmark Mix (as used by Engine)")
        bm_mix = get_benchmark_mix()
        if bm_mix is None or bm_mix.empty:
            st.info("No benchmark mix table returned by engine.")
        else:
            show_df(bm_mix[bm_mix["Wave"] == selected_wave] if "Wave" in bm_mix.columns else bm_mix, selected_wave, key="bm_mix")

    with colB:
        st.markdown("#### Attribution Snapshot (365D)")
        st.caption(
            "Benchmark Difficulty (BM - SPY) provides context for alpha capture. "
            "If the benchmark outperformed SPY, alpha is harder to achieve."
        )

        attrib = compute_alpha_attribution(selected_wave, mode=mode, days=365)
        if not attrib:
            st.info("Attribution unavailable.")
        else:
            arows = []
            for k, v in attrib.items():
                if any(x in k for x in ["Return", "Alpha", "Vol", "MaxDD", "TE", "Difficulty", "VaR", "CVaR"]):
                    arows.append({"Metric": k, "Value": fmt_pct(v)})
                elif any(x in k for x in ["IR", "Sharpe", "Sortino", "Beta"]):
                    arows.append({"Metric": k, "Value": fmt_num(v, 2)})
                else:
                    arows.append({"Metric": k, "Value": fmt_num(v, 4)})
            st.dataframe(pd.DataFrame(arows), use_container_width=True)

    # NOTE:
    # The remainder of your original file continues here unchanged:
    # - Wave Doctor
    # - What-If Lab
    # - Factor Decomposition tab
    # - Risk Lab tab
    # - Correlation tab
    # - Vector OS Insight Layer
    # - Diagnostics expander
    #
    # Paste the rest of your existing working code below this point WITHOUT CHANGES.