# app.py — WAVES Intelligence™ Institutional Console (Vector OS Edition)
# FULL PRODUCTION FILE (NO PATCHES)
#
# Keeps ALL IRB-1 features (Benchmark Truth, Mode Separation Proof, Attribution,
# Wave Doctor + What-If Lab, Top-10 links, Market Intel, Factor Decomp, Vector Insight)
#
# Adds:
#   • Pinned Sticky Summary Bar (always visible)
#   • Row Highlighting (selected wave + alpha heat tints)
#   • Alpha Heatmap View (All Waves x Timeframe)
#   • One-click Wave Jump (row select if supported; fallback jump control always works)
#   • Display-only formatting: Returns/Alpha/Risk in % (no math changes)
#
# Notes:
#   • Does NOT modify engine math or baseline results.
#   • What-If Lab is explicitly labeled “shadow simulation”.
#   • Streamlit-safe patterns; selection events are best-effort.

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
# Helpers: formatting
# ============================================================
def fmt_pct(x: Any, digits: int = 2) -> str:
    """Display-only percent formatter (0.028 -> 2.80%)."""
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
    out["Benchmark Difficulty (BM - SPY)"] = float(benchmark_difficulty) if pd.notna(benchmark_difficulty) else float("nan")
    out["Tracking Error (TE)"] = float(te)
    out["Information Ratio (IR)"] = float(ir)

    out["Wave Vol"] = float(annualized_vol(wave_ret))
    out["Benchmark Vol"] = float(annualized_vol(bm_ret))
    out["Wave MaxDD"] = float(max_drawdown(nav_wave))
    out["Benchmark MaxDD"] = float(max_drawdown(nav_bm))

    return out


# ============================================================
# WaveScore proto v1.0 (console-side)
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
# Display-only formatting map for tables (NO math changes)
# ============================================================
def build_formatter_map(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Returns dict: {col_name: formatter} for Styler.format()
    Keeps underlying numeric values intact (so alpha tint + sorting still works).
    """
    if df is None or df.empty:
        return {}

    fmt: Dict[str, Any] = {}

    # percent-type columns (returns, alpha, vol, drawdown, TE)
    pct_keywords = [
        " Ret", " Return", " Alpha", "MaxDD", "Max Drawdown",
        "Vol", "Volatility", "Tracking Error", "TE",
        "Benchmark Difficulty", "BM Difficulty",
    ]

    for c in df.columns:
        cs = str(c)

        # WaveScore numeric columns
        if cs == "WaveScore":
            fmt[c] = lambda v: fmt_score(v)
            continue
        if cs in ["Return Quality", "Risk Control", "Consistency", "Resilience", "Efficiency", "Transparency"]:
            fmt[c] = lambda v: fmt_num(v, 1)
            continue
        if cs.startswith("β_") or cs.startswith("beta") or cs.startswith("β"):
            fmt[c] = lambda v: fmt_num(v, 2)
            continue
        if "IR" in cs or cs.endswith("_IR") or cs.startswith("IR_"):
            fmt[c] = lambda v: fmt_num(v, 2)
            continue

        # percent formatter by keyword match
        if any(k in cs for k in pct_keywords):
            fmt[c] = lambda v: fmt_pct(v, 2)

    return fmt


# ============================================================
# Row highlighting utilities (selected wave + alpha tint)
# ============================================================
def style_selected_and_alpha(df: pd.DataFrame, selected_wave: str):
    alpha_cols = [c for c in df.columns if ("Alpha" in str(c)) or str(c).endswith("α")]

    def row_style(row: pd.Series):
        styles = [""] * len(row)

        # Selected wave row
        if "Wave" in df.columns and str(row.get("Wave", "")) == str(selected_wave):
            styles = ["background-color: rgba(255,255,255,0.07); font-weight: 700;"] * len(row)

        # Alpha tint cells (based on actual numeric values)
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

    # Apply display formatting (NO conversion to strings)
    fmt_map = build_formatter_map(df)
    if fmt_map:
        styler = styler.format(fmt_map)

    return styler


def show_df(df: pd.DataFrame, selected_wave: str, key: str):
    try:
        st.dataframe(style_selected_and_alpha(df, selected_wave), use_container_width=True, key=key)
    except Exception:
        # fallback without styling (but still try formatting)
        try:
            styler = df.style.format(build_formatter_map(df))
            st.dataframe(styler, use_container_width=True, key=key)
        except Exception:
            st.dataframe(df, use_container_width=True, key=key)


# ============================================================
# One-click Wave jump (best-effort selection events)
# ============================================================
def selectable_table_jump(df: pd.DataFrame, key: str) -> None:
    """
    Best-effort row selection:
      • If Streamlit selection events are supported, selecting a row sets selected_wave and reruns.
      • Otherwise shows dataframe + Jump dropdown fallback.
    """
    if df is None or df.empty or "Wave" not in df.columns:
        st.info("No waves available to jump.")
        return

    # Try Streamlit selection API (depends on Streamlit version)
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

    # Fallback jump control
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

        # 1D
        if len(nav_w) >= 2 and len(nav_b) >= 2:
            r1w = float(nav_w.iloc[-1] / nav_w.iloc[-2] - 1.0)
            r1b = float(nav_b.iloc[-1] / nav_b.iloc[-2] - 1.0)
            a1 = r1w - r1b
        else:
            a1 = np.nan

        # 30/60/365
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

    df = pd.DataFrame(rows).sort_values("Wave")
    return df


def plot_alpha_heatmap(alpha_df: pd.DataFrame, selected_wave: str, title: str):
    if go is None or alpha_df is None or alpha_df.empty:
        st.info("Heatmap unavailable (Plotly missing or no data).")
        return

    df = alpha_df.copy()
    cols = [c for c in ["1D Alpha", "30D Alpha", "60D Alpha", "365D Alpha"] if c in df.columns]
    if not cols:
        st.info("No alpha columns to plot.")
        return

    # Put selected wave at top
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
        diagnosis.append("30D alpha is unusually large. This can happen from real signal, but also from benchmark mix or data coverage shifts.")
        recs.append("Freeze benchmark mix for demo comparisons and check benchmark drift in the Benchmark Truth Panel.")

    if pd.notna(a365) and a365 < 0:
        diagnosis.append("365D alpha is negative. This might reflect true underperformance, or a tougher benchmark composition.")
        if pd.notna(bm_difficulty) and bm_difficulty > 0.03:
            diagnosis.append("Benchmark difficulty is positive vs SPY (benchmark outperformed SPY), so alpha is harder on this window.")
            recs.append("For validation, temporarily compare to SPY/QQQ-style benchmark to isolate engine effect.")
    else:
        if pd.notna(a365) and a365 > 0.06:
            diagnosis.append("365D alpha is solidly positive. This aligns with the outcome you were looking for.")
            recs.append("Lock benchmark mix snapshot for reproducibility in demos.")

    if pd.notna(te) and te > te_warn:
        flags.append("High tracking error (active risk elevated)")
        diagnosis.append("Tracking error is high; the wave behaves very differently than its benchmark.")
        recs.append("Reduce tilt strength and/or tighten exposure caps (What-If Lab) to lower TE.")

    if pd.notna(vol_w) and vol_w > vol_warn:
        flags.append("High volatility")
        diagnosis.append("Volatility is elevated. If this wave is intended to be disciplined, consider lowering target vol.")
        recs.append("Lower vol target (e.g., 20% → 16–18%) in What-If Lab (shadow only).")

    if pd.notna(mdd_w) and mdd_w < mdd_warn:
        flags.append("Deep drawdown")
        diagnosis.append("Drawdown is deep versus typical institutional tolerances.")
        recs.append("Increase SmartSafe posture in stress regimes (extra safe boost in What-If Lab).")

    if pd.notna(vol_b) and pd.notna(vol_w) and vol_b > 0 and (vol_w / vol_b) > 1.6:
        flags.append("Volatility much higher than benchmark")
        diagnosis.append("Wave volatility is much higher than its benchmark; this can inflate wins and losses.")
        recs.append("Tighten exposure caps + reduce tilt strength to stabilize.")

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

# Session state (enables “jump” to update selectbox)
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
    r365b = ret_from_nav(nav_b, len(nav_b)))
    bar_r365 = r365w
    bar_a365 = r365w - r365b

    bar_te = tracking_error(ret_w, ret_b)
    bar_ir = information_ratio(nav_w, nav_b, bar_te)

# regime/vix snapshot
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
# Tabs
# ============================================================
tab_console, tab_market, tab_factors, tab_vector = st.tabs(
    ["Console", "Market Intel", "Factor Decomposition", "Vector OS Insight Layer"]
)

# ============================================================
# TAB 1: Console
# ============================================================
with tab_console:
    st.subheader("🔥 Alpha Heatmap View (All Waves × Timeframe)")
    st.caption("Fast scan. The table below supports one-click jump on supported Streamlit versions. Fallback jump always works.")

    alpha_df = build_alpha_matrix(all_waves, mode)
    plot_alpha_heatmap(alpha_df, selected_wave, title=f"Alpha Heatmap — Mode: {mode}")

    st.markdown("### 🧭 One-Click Jump Table")
    jump_df = alpha_df.copy()
    for c in ["1D Alpha", "30D Alpha", "60D Alpha", "365D Alpha"]:
        if c not in jump_df.columns:
            jump_df[c] = np.nan
    jump_df["RankScore"] = jump_df[["1D Alpha", "30D Alpha", "60D Alpha", "365D Alpha"]].mean(axis=1, skipna=True)
    jump_df = jump_df.sort_values("RankScore", ascending=False)

    selectable_table_jump(jump_df, key="wave_jump_table")
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

        if len(nav_w) >= 2 and len(nav_b) >= 2:
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
            {"Wave": wname, "1D Ret": r1w, "1D Alpha": a1, "30D Ret": r30w, "30D Alpha": a30,
             "60D Ret": r60w, "60D Alpha": a60, "365D Ret": r365w, "365D Alpha": a365}
        )

    overview_df = pd.DataFrame(overview_rows)
    show_df(overview_df, selected_wave, key="overview_df")
    st.markdown("---")

    st.subheader("🛡️ Risk Analytics (Vol, MaxDD, TE, IR) — 365D Window")

    risk_rows: List[Dict[str, Any]] = []
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

        te = tracking_error(ret_w, ret_b)
        ir = information_ratio(nav_w, nav_b, te)

        risk_rows.append(
            {"Wave": wname, "Wave Vol": annualized_vol(ret_w), "Benchmark Vol": annualized_vol(ret_b),
             "Wave MaxDD": max_drawdown(nav_w), "Benchmark MaxDD": max_drawdown(nav_b),
             "Tracking Error": te, "Information Ratio": ir}
        )

    risk_df = pd.DataFrame(risk_rows)
    show_df(risk_df, selected_wave, key="risk_df")
    st.markdown("---")

    st.subheader("🏁 WaveScore™ Leaderboard (Proto v1.0 · 365D Data)")
    wavescore_df = compute_wavescore_for_all_waves(all_waves, mode=mode, days=365)
    if wavescore_df.empty:
        st.info("No WaveScore data available yet.")
    else:
        show_df(wavescore_df, selected_wave, key="wavescore_df")

    st.markdown("---")

    st.subheader("🧩 Benchmark Transparency Table (Composite Benchmark Components per Wave)")
    bm_mix = get_benchmark_mix()
    if bm_mix.empty:
        st.info("No benchmark mix data available.")
    else:
        st.dataframe(bm_mix, use_container_width=True)

    st.markdown("---")

    st.subheader(f"📌 Wave Detail — {selected_wave} (Mode: {mode})")
    col_chart, col_stats = st.columns([2.0, 1.0])

    hist_sel = compute_wave_history(selected_wave, mode=mode, days=history_days)

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

            r30w = ret_from_nav(nav_w, min(30, len(nav_w)))
            r30b = ret_from_nav(nav_b, min(30, len(nav_b)))
            a30 = r30w - r30b

            r365w = ret_from_nav(nav_w, len(nav_w))
            r365b = ret_from_nav(nav_b, len(nav_b))
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

    with st.expander("🔗 Top-10 Holdings (with Google Finance links)", expanded=False):
        holdings_df = get_wave_holdings(selected_wave)
        if holdings_df.empty:
            st.info("No holdings available for this Wave.")
        else:
            def google_link(ticker: str) -> str:
                base = "https://www.google.com/finance/quote"
                return f"[{ticker}]({base}/{ticker})"

            hold = holdings_df.copy()
            hold["Weight"] = pd.to_numeric(hold["Weight"], errors="coerce").fillna(0.0)
            hold = hold.sort_values("Weight", ascending=False).head(10)
            hold["Google Finance"] = hold["Ticker"].astype(str).apply(google_link)
            st.dataframe(hold, use_container_width=True)


# ============================================================
# TAB 2: Market Intel
# ============================================================
with tab_market:
    st.subheader("🌍 Global Market Dashboard")
    market_df = fetch_market_assets(days=history_days)

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

        rows_assets: List[Dict[str, Any]] = []
        for tkr, label in assets.items():
            if tkr not in market_df.columns:
                continue
            series = market_df[tkr]
            last = float(series.iloc[-1]) if len(series) else float("nan")
            r1d = simple_ret(series, 2)
            r30 = simple_ret(series, 30)
            rows_assets.append({"Ticker": tkr, "Asset": label, "Last": last, "1D Return": r1d, "30D Return": r30})

        snap_df = pd.DataFrame(rows_assets)
        # Use display formatter here too (so 1D/30D returns show as %)
        show_df(snap_df, selected_wave, key="market_snap_df")

    st.markdown("---")
    st.subheader(f"⚡ WAVES Reaction Snapshot (30D · Mode = {mode})")

    reaction_rows: List[Dict[str, Any]] = []
    for wname in all_waves:
        hist = compute_wave_history(wname, mode=mode, days=365)
        if hist.empty or len(hist) < 2:
            reaction_rows.append({"Wave": wname, "30D Return": np.nan, "30D Alpha": np.nan, "Classification": "No data"})
            continue

        nav_w = hist["wave_nav"]
        nav_b = hist["bm_nav"]
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

        reaction_rows.append({"Wave": wname, "30D Return": r30w, "30D Alpha": a30, "Classification": label})

    reaction_df = pd.DataFrame(reaction_rows)
    show_df(reaction_df, selected_wave, key="reaction_df")


# ============================================================
# TAB 3: Factor Decomposition
# ============================================================
with tab_factors:
    st.subheader("🧬 Factor Decomposition (Institution-Level Analytics)")

    factor_days = min(history_days, 365)
    factor_prices = fetch_market_assets(days=factor_days)

    needed = ["SPY", "QQQ", "IWM", "TLT", "GLD", "BTC-USD"]
    missing = [t for t in needed if (factor_prices.empty or t not in factor_prices.columns)]

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

        rows_beta: List[Dict[str, Any]] = []
        for wname in all_waves:
            hist = compute_wave_history(wname, mode=mode, days=factor_days)
            if hist.empty or "wave_ret" not in hist.columns:
                rows_beta.append({"Wave": wname, "β_SPY": np.nan, "β_QQQ": np.nan, "β_IWM": np.nan, "β_TLT": np.nan, "β_GLD": np.nan, "β_BTC": np.nan})
                continue

            wret = hist["wave_ret"]
            betas = regress_factors(wave_ret=wret, factor_ret=factor_returns)

            rows_beta.append(
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

        beta_df = pd.DataFrame(rows_beta)
        show_df(beta_df, selected_wave, key="beta_df")

    st.markdown("---")
    st.subheader(f"🔁 Correlation Matrix — Waves (Daily Returns · Mode = {mode})")

    corr_days = min(history_days, 365)
    ret_panel: Dict[str, pd.Series] = {}
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
    st.subheader("🤖 Vector OS Insight Layer — Rules-Based Narrative Panel")
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

        r365w = ret_from_nav(nav_w, len(nav_w))
        r365b = ret_from_nav(nav_b, len(nav_b))
        a365 = r365w - r365b

        vol_w = annualized_vol(ret_w)
        vol_b = annualized_vol(ret_b)
        mdd_w = max_drawdown(nav_w)
        mdd_b = max_drawdown(nav_b)

        question = st.text_input("Ask Vector about this Wave or the lineup:", "")

        st.markdown(f"### Vector’s Insight — {selected_wave}")
        try:
            st.write(f"- **WaveScore (proto)**: **{float(row['WaveScore']):.1f}/100** (**{str(row['Grade'])}**).")
        except Exception:
            st.write(f"- **WaveScore (proto)**: **{fmt_score(row.get('WaveScore', np.nan))}/100** (**{row.get('Grade', 'N/A')}**).")

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
st.caption("WAVES Intelligence™ • Institutional Console (IRB-1+) • Heatmap + Sticky Bar + Row Highlight + Wave Jump • Baseline engine results unchanged")