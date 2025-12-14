# app.py — WAVES Intelligence™ Institutional Console (Vector OS Edition)
# FULL PRODUCTION FILE (NO PATCHES) — MEGA SINGLE-FILE BUILD (DIAGNOSTICS++++)
#
# IMPORTANT:
#  • Intentionally DOES NOT use `from __future__ import annotations` to avoid Streamlit/iPhone hidden-char failures.
#  • This file is designed to be pasted in 5 parts (PART 1→PART 5) into ONE app.py.
#
# What this build includes:
#  ✅ Sticky summary bar (mode/wave/alpha/TE/IR/beta)
#  ✅ All Waves overview (1D/30D/60D/365D returns + alpha)
#  ✅ Alpha heatmap (Plotly optional)
#  ✅ Top-10 holdings w/ Google links + full holdings table
#  ✅ Benchmark Truth: benchmark mix + difficulty vs SPY + attribution (Engine vs Static basket)
#  ✅ Mode Separation Proof: side-by-side metrics + NAV overlay across modes
#  ✅ Rolling diagnostics: rolling alpha / TE / beta / vol + alpha persistence + drift flags
#  ✅ Correlation matrix across waves (returns)
#  ✅ Data quality / coverage audit
#  ✅ Factor decomposition regression betas (SPY/QQQ/IWM/TLT/GLD)
#  ✅ What-If Lab (shadow simulation — DOES NOT change engine math)
#  ✅ Market Intel panel
#  ✅ WaveScore proto leaderboard display (console-side; no engine math changes)
#
# Engine assumptions (best-effort; all guarded):
#  - waves_engine.compute_history_nav(wave_name, mode=..., days=...) -> df with wave_nav,bm_nav,wave_ret,bm_ret
#  - waves_engine.get_wave_holdings(wave_name) -> df with Ticker,Name,Weight
#  - waves_engine.get_benchmark_mix_table() -> df with Wave,Ticker,Name,Weight
#  - waves_engine.get_all_waves() optional
#  - waves_engine.get_modes() optional
#  - Optional constants for targets: MODE_BETA_TARGET, MODE_BASE_EXPOSURE, REGIME_GATING, REGIME_EXPOSURE

import math
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

import waves_engine as we

# Optional libs: yfinance + plotly (never crash if missing)
try:
    import yfinance as yf
except Exception:
    yf = None

try:
    import plotly.graph_objects as go
except Exception:
    go = None
# ============================================================
# SAFETY STUBS (prevents NameError during Streamlit parse)
# ============================================================
def selectable_table_jump(*args, **kwargs):
    return None
# ============================================================
# Streamlit config
# ============================================================
st.set_page_config(
    page_title="WAVES Intelligence™ Console",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# Global UI CSS
# ============================================================
st.markdown(
    """
<style>
.block-container { padding-top: 0.8rem; padding-bottom: 2.0rem; }
.waves-sticky {
  position: sticky; top: 0; z-index: 999;
  backdrop-filter: blur(10px); -webkit-backdrop-filter: blur(10px);
  padding: 10px 12px; margin: 0 0 12px 0;
  border-radius: 14px; border: 1px solid rgba(255,255,255,0.10);
  background: rgba(10, 15, 28, 0.66);
}
.waves-chip {
  display: inline-block; padding: 6px 10px; margin: 6px 8px 0 0;
  border-radius: 999px; border: 1px solid rgba(255,255,255,0.12);
  background: rgba(255,255,255,0.04);
  font-size: 0.85rem; line-height: 1.0rem; white-space: nowrap;
}
.waves-hdr { font-weight: 900; letter-spacing: 0.2px; margin-bottom: 4px; }
div[data-testid="stDataFrame"] { border-radius: 12px; overflow: hidden; }
@media (max-width: 700px) { .block-container { padding-left: 0.8rem; padding-right: 0.8rem; } }
</style>
""",
    unsafe_allow_html=True,
)

# ============================================================
# Formatting helpers
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
# Data fetchers (safe)
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
    return fetch_prices_daily(["SPY","QQQ","IWM","TLT","GLD","BTC-USD","^VIX","^TNX"], days=days)

# ============================================================
# Engine wrappers (safe)
# ============================================================
@st.cache_data(show_spinner=False)
def get_all_waves() -> List[str]:
    # Try engine function first
    try:
        w = we.get_all_waves()
        if isinstance(w, list) and len(w) > 0:
            return [str(x) for x in w]
    except Exception:
        pass
    # Try benchmark mix
    try:
        bm = we.get_benchmark_mix_table()
        if isinstance(bm, pd.DataFrame) and (not bm.empty) and ("Wave" in bm.columns):
            return sorted([str(x) for x in bm["Wave"].dropna().unique().tolist()])
    except Exception:
        pass
    # fallback list (never crash)
    return [
        "S&P 500 Wave",
        "AI Wave",
        "Quantum Computing Wave",
        "Crypto Wave",
        "Income Wave",
        "Small Cap Growth Wave",
        "Small to Mid Cap Growth Wave",
        "Future Power & Energy Wave",
        "Crypto Income Wave",
        "Clean Transit-Infrastructure Wave",
        "Cloud & Software Wave",
        "Cybersecurity Wave",
        "Healthcare Innovation Wave",
        "Consumer Disruption Wave",
        "Defense-Tech Wave",
    ]

@st.cache_data(show_spinner=False)
def get_modes() -> List[str]:
    try:
        m = we.get_modes()
        if isinstance(m, list) and len(m) > 0:
            return [str(x) for x in m]
    except Exception:
        pass
    return ["Standard", "Alpha-Minus-Beta", "Private Logic"]

@st.cache_data(show_spinner=False)
def compute_wave_history(wave_name: str, mode: str, days: int = 365) -> pd.DataFrame:
    try:
        df = we.compute_history_nav(wave_name, mode=mode, days=days)
        if df is None:
            return pd.DataFrame(columns=["wave_nav","bm_nav","wave_ret","bm_ret"])
        return df.copy()
    except Exception:
        return pd.DataFrame(columns=["wave_nav","bm_nav","wave_ret","bm_ret"])

@st.cache_data(show_spinner=False)
def get_wave_holdings(wave_name: str) -> pd.DataFrame:
    try:
        df = we.get_wave_holdings(wave_name)
        if df is None:
            return pd.DataFrame(columns=["Ticker","Name","Weight"])
        return df.copy()
    except Exception:
        return pd.DataFrame(columns=["Ticker","Name","Weight"])

@st.cache_data(show_spinner=False)
def get_benchmark_mix() -> pd.DataFrame:
    try:
        df = we.get_benchmark_mix_table()
        if df is None:
            return pd.DataFrame(columns=["Wave","Ticker","Name","Weight"])
        return df.copy()
    except Exception:
        return pd.DataFrame(columns=["Wave","Ticker","Name","Weight"])

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
    s = float(sub.iloc[0]); e = float(sub.iloc[-1])
    if s <= 0:
        return float("nan")
    return (e / s) - 1.0

def annualized_vol(daily_ret: pd.Series) -> float:
    daily_ret = safe_series(daily_ret)
    if len(daily_ret) < 2:
        return float("nan")
    return float(daily_ret.std() * np.sqrt(252))

def max_drawdown(nav: pd.Series) -> float:
    nav = safe_series(nav)
    if len(nav) < 2:
        return float("nan")
    rm = nav.cummax()
    dd = (nav / rm) - 1.0
    return float(dd.min())

def tracking_error(w: pd.Series, b: pd.Series) -> float:
    w = safe_series(w); b = safe_series(b)
    if len(w) < 2 or len(b) < 2:
        return float("nan")
    df = pd.concat([w.rename("w"), b.rename("b")], axis=1).dropna()
    if df.shape[0] < 2:
        return float("nan")
    d = df["w"] - df["b"]
    return float(d.std() * np.sqrt(252))

def information_ratio(nav_w: pd.Series, nav_b: pd.Series, te: float) -> float:
    if te is None or (isinstance(te, float) and (math.isnan(te) or te <= 0)):
        return float("nan")
    rw = ret_from_nav(nav_w, len(nav_w))
    rb = ret_from_nav(nav_b, len(nav_b))
    if not (math.isfinite(rw) and math.isfinite(rb)):
        return float("nan")
    return float((rw - rb) / te)

def beta_vs_benchmark(w: pd.Series, b: pd.Series) -> float:
    w = safe_series(w); b = safe_series(b)
    df = pd.concat([w.rename("w"), b.rename("b")], axis=1).dropna()
    if df.shape[0] < 20:
        return float("nan")
    vb = float(df["b"].var())
    if not math.isfinite(vb) or vb <= 0:
        return float("nan")
    cov = float(df["w"].cov(df["b"]))
    return float(cov / vb)

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

# ============================================================
# Styling / table helper
# ============================================================
def build_formatter_map(df: pd.DataFrame) -> Dict[str, Any]:
    if df is None or df.empty:
        return {}
    fmt = {}
    pct_keywords = [" Ret", " Return", " Alpha", "Vol", "MaxDD", "Tracking Error", "TE", "Difficulty", "Captured"]
    for c in df.columns:
        cs = str(c)
        if cs == "WaveScore":
            fmt[c] = lambda v: fmt_score(v)
        elif cs in ["Return Quality","Risk Control","Consistency","Resilience","Efficiency","Transparency"]:
            fmt[c] = lambda v: fmt_num(v, 1)
        elif ("IR" in cs) and ("Return" not in cs) and ("Ret" not in cs):
            fmt[c] = lambda v: fmt_num(v, 2)
        elif cs.startswith("β") or cs.lower().startswith("beta"):
            fmt[c] = lambda v: fmt_num(v, 2)
        elif any(k in cs for k in pct_keywords):
            fmt[c] = lambda v: fmt_pct(v, 2)
    return fmt

def style_selected_and_alpha(df: pd.DataFrame, selected_wave: str):
    alpha_cols = [c for c in df.columns if ("Alpha" in str(c)) or ("Captured" in str(c))]
    fmt_map = build_formatter_map(df)

    def row_style(row: pd.Series):
        styles = [""] * len(row)
        if "Wave" in df.columns and str(row.get("Wave","")) == str(selected_wave):
            styles = ["background-color: rgba(255,255,255,0.07); font-weight: 700;"] * len(row)
        for i, col in enumerate(df.columns):
            if col in alpha_cols:
                try:
                    v = float(row[col])
                except Exception:
                    continue
                if not math.isfinite(v):
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
# Alpha matrix + heatmap
# ============================================================
@st.cache_data(show_spinner=False)
def build_alpha_matrix(all_waves: List[str], mode: str) -> pd.DataFrame:
    rows = []
    for wname in all_waves:
        hist = compute_wave_history(wname, mode=mode, days=365)
        if hist.empty or len(hist) < 2:
            rows.append({"Wave": wname, "1D Alpha": np.nan, "30D Alpha": np.nan, "60D Alpha": np.nan, "365D Alpha": np.nan})
            continue
        nav_w = hist["wave_nav"]; nav_b = hist["bm_nav"]
        # 1D
        if len(nav_w) >= 2 and len(nav_b) >= 2:
            r1w = float(nav_w.iloc[-1] / nav_w.iloc[-2] - 1.0)
            r1b = float(nav_b.iloc[-1] / nav_b.iloc[-2] - 1.0)
            a1 = r1w - r1b
        else:
            a1 = np.nan
        # 30/60/365
        w30 = min(30, len(nav_w)); w60 = min(60, len(nav_w))
        r30w = ret_from_nav(nav_w, w30); r30b = ret_from_nav(nav_b, w30); a30 = r30w - r30b
        r60w = ret_from_nav(nav_w, w60); r60b = ret_from_nav(nav_b, w60); a60 = r60w - r60b
        r365w = ret_from_nav(nav_w, len(nav_w)); r365b = ret_from_nav(nav_b, len(nav_b)); a365 = r365w - r365b
        rows.append({"Wave": wname, "1D Alpha": a1, "30D Alpha": a30, "60D Alpha": a60, "365D Alpha": a365})
    return pd.DataFrame(rows).sort_values("Wave")

def plot_alpha_heatmap(alpha_df: pd.DataFrame, selected_wave: str, title: str):
    if go is None or alpha_df is None or alpha_df.empty:
        st.info("Heatmap unavailable (Plotly missing or no data).")
        return
    df = alpha_df.copy()
    cols = [c for c in ["1D Alpha","30D Alpha","60D Alpha","365D Alpha"] if c in df.columns]
    if "Wave" in df.columns and selected_wave in set(df["Wave"]):
        top = df[df["Wave"] == selected_wave]
        rest = df[df["Wave"] != selected_wave]
        df = pd.concat([top, rest], axis=0)
    z = df[cols].values
    y = df["Wave"].tolist()
    v = float(np.nanmax(np.abs(z))) if np.isfinite(z).any() else 0.10
    if not math.isfinite(v) or v <= 0:
        v = 0.10
    fig = go.Figure(data=go.Heatmap(z=z, x=cols, y=y, zmin=-v, zmax=v, colorbar=dict(title="Alpha")))
    fig.update_layout(title=title, height=min(900, 240 + 22 * max(10, len(y))), margin=dict(l=80, r=40, t=60, b=40))
    st.plotly_chart(fig, use_container_width=True)

# ============================================================
# END PART 1 / 5
# Next: PART 2 continues with attribution + wavescore + diagnostics panels
# ============================================================
# ============================================================
# PART 2 / 5 — Attribution + WaveScore + Diagnostics++ core
# ============================================================

# ============================================================
# Weights helpers (for static basket attribution + what-if)
# ============================================================
def _weights_from_df(df: pd.DataFrame, ticker_col: str = "Ticker", weight_col: str = "Weight") -> pd.Series:
    if df is None or df.empty:
        return pd.Series(dtype=float)
    w = df[[ticker_col, weight_col]].copy()
    w[ticker_col] = w[ticker_col].astype(str)
    w[weight_col] = pd.to_numeric(w[weight_col], errors="coerce").fillna(0.0)
    w = w.groupby(ticker_col, as_index=True)[weight_col].sum()
    total = float(w.sum())
    if total <= 0 or not math.isfinite(total):
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
    w = weights.reindex(px.columns).fillna(0.0)
    daily_ret = px.pct_change().fillna(0.0)
    port_ret = (daily_ret * w).sum(axis=1)
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

# ============================================================
# Alpha Attribution (Engine vs Static Basket) + Beta
# ============================================================
@st.cache_data(show_spinner=False)
def compute_alpha_attribution(wave_name: str, mode: str, days: int = 365) -> Dict[str, float]:
    out: Dict[str, float] = {}
    hist = compute_wave_history(wave_name, mode=mode, days=days)
    if hist.empty or len(hist) < 2:
        return out

    nav_wave = hist.get("wave_nav", pd.Series(dtype=float))
    nav_bm = hist.get("bm_nav", pd.Series(dtype=float))
    wave_ret = hist.get("wave_ret", pd.Series(dtype=float))
    bm_ret = hist.get("bm_ret", pd.Series(dtype=float))

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
# WaveScore (console-side proto display)
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

@st.cache_data(show_spinner=False)
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

        # simple recovery proxy
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

        # Return Quality (0-25)
        rq_ir = float(np.clip(ir, 0.0, 1.5) / 1.5 * 15.0) if not math.isnan(ir) else 0.0
        rq_alpha = float(np.clip((alpha_365 + 0.05) / 0.15, 0.0, 1.0) * 10.0) if not math.isnan(alpha_365) else 0.0
        return_quality = float(np.clip(rq_ir + rq_alpha, 0.0, 25.0))

        # Risk Control (0-25) — prefer vol ratio near ~0.9–1.0
        if math.isnan(vol_ratio):
            risk_control = 0.0
        else:
            penalty = max(0.0, abs(vol_ratio - 0.95) - 0.10)
            risk_control = float(np.clip(1.0 - penalty / 0.70, 0.0, 1.0) * 25.0)

        # Consistency (0-15)
        consistency = float(np.clip(hit_rate, 0.0, 1.0) * 15.0) if not math.isnan(hit_rate) else 0.0

        # Resilience (0-10)
        if math.isnan(recovery_frac) or math.isnan(mdd_wave) or math.isnan(mdd_bm):
            resilience = 0.0
        else:
            rec_part = float(np.clip(recovery_frac, 0.0, 1.0) * 6.0)
            dd_edge = (mdd_bm - mdd_wave)
            dd_part = float(np.clip((dd_edge + 0.10) / 0.20, 0.0, 1.0) * 4.0)
            resilience = float(np.clip(rec_part + dd_part, 0.0, 10.0))

        # Efficiency (0-15) — TE based proxy
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
# Alpha Captured (daily) — exposure-scaled if engine provides
# ============================================================
def _get_exposure_series_if_available(hist: pd.DataFrame) -> Optional[pd.Series]:
    # Best-effort: look for common columns
    for col in ["exposure", "Exposure", "expo", "EXPOSURE", "beta_scale", "exposure_scale"]:
        if col in hist.columns:
            try:
                s = pd.to_numeric(hist[col], errors="coerce").astype(float)
                return s
            except Exception:
                pass
    return None

def compute_alpha_captured_series(hist: pd.DataFrame) -> pd.Series:
    """
    Alpha Captured daily = wave_ret - bm_ret, optionally divided by exposure (or scaled) if exposure is provided.
    We will NOT assume; if exposure exists and is valid, we compute:
        alpha_captured = (wave_ret - bm_ret) / max(exposure, 0.01)
    Otherwise:
        alpha_captured = wave_ret - bm_ret
    """
    if hist is None or hist.empty or ("wave_ret" not in hist.columns) or ("bm_ret" not in hist.columns):
        return pd.Series(dtype=float)
    w = pd.to_numeric(hist["wave_ret"], errors="coerce").astype(float)
    b = pd.to_numeric(hist["bm_ret"], errors="coerce").astype(float)
    base = (w - b)
    expo = _get_exposure_series_if_available(hist)
    if expo is None:
        base.name = "alpha_captured"
        return base
    expo = pd.to_numeric(expo, errors="coerce").astype(float).replace([np.inf, -np.inf], np.nan)
    denom = expo.clip(lower=0.01).fillna(1.0)
    out = base / denom
    out.name = "alpha_captured"
    return out

# ============================================================
# Rolling diagnostics (alpha/TE/beta/vol + persistence)
# ============================================================
def rolling_metrics(hist: pd.DataFrame, window: int = 60) -> pd.DataFrame:
    """
    Returns a df with:
      roll_alpha (mean daily alpha * 252 approx),
      roll_te (std of active returns * sqrt(252)),
      roll_beta (rolling beta),
      roll_vol (rolling wave vol),
      roll_bm_vol (rolling bm vol),
      alpha_persistence (pct days alpha > 0 over window),
      drift_beta (beta - beta_target if available)
    """
    if hist is None or hist.empty:
        return pd.DataFrame()
    if ("wave_ret" not in hist.columns) or ("bm_ret" not in hist.columns):
        return pd.DataFrame()

    df = pd.DataFrame(index=hist.index.copy())
    w = pd.to_numeric(hist["wave_ret"], errors="coerce").astype(float)
    b = pd.to_numeric(hist["bm_ret"], errors="coerce").astype(float)
    active = (w - b)

    # Rolling alpha annualized approximation
    df["roll_alpha"] = active.rolling(window).mean() * 252.0

    # Rolling TE annualized
    df["roll_te"] = active.rolling(window).std() * np.sqrt(252)

    # Rolling vol wave/bm
    df["roll_vol"] = w.rolling(window).std() * np.sqrt(252)
    df["roll_bm_vol"] = b.rolling(window).std() * np.sqrt(252)

    # Rolling beta
    # Using rolling cov/var
    cov = w.rolling(window).cov(b)
    varb = b.rolling(window).var()
    df["roll_beta"] = cov / varb.replace(0.0, np.nan)

    # Alpha persistence
    df["alpha_persistence"] = active.rolling(window).apply(lambda x: float(np.mean(np.array(x) > 0.0)), raw=False)

    return df

def beta_drift_flags(beta_real: float, beta_target: float, thresh: float = 0.07) -> Tuple[str, bool]:
    if (beta_real is None) or (beta_target is None) or (not math.isfinite(beta_real)) or (not math.isfinite(beta_target)):
        return ("—", False)
    drift = beta_real - beta_target
    flag = abs(drift) >= thresh
    s = f"{fmt_num(drift,2)}"
    return (s, flag)

# ============================================================
# Correlation matrix across waves (daily returns)
# ============================================================
@st.cache_data(show_spinner=False)
def compute_wave_return_panel(all_waves: List[str], mode: str, days: int = 365) -> pd.DataFrame:
    """
    Build a wide panel of wave daily returns aligned by date:
        columns = wave names
        index = dates
    """
    series_list = []
    for wname in all_waves:
        hist = compute_wave_history(wname, mode=mode, days=days)
        if hist is None or hist.empty or "wave_ret" not in hist.columns:
            continue
        s = pd.to_numeric(hist["wave_ret"], errors="coerce").astype(float)
        s.name = wname
        series_list.append(s)
    if not series_list:
        return pd.DataFrame()
    df = pd.concat(series_list, axis=1).sort_index()
    return df

def correlation_matrix(df_ret: pd.DataFrame) -> pd.DataFrame:
    if df_ret is None or df_ret.empty:
        return pd.DataFrame()
    return df_ret.corr()

# ============================================================
# Data quality / coverage audit
# ============================================================
def audit_history_df(hist: pd.DataFrame) -> Dict[str, Any]:
    """
    Quick audit:
      - rows
      - start/end dates
      - missing % in key cols
      - nav monotonic issues
      - duplicate dates
      - bm coverage
    """
    if hist is None or hist.empty:
        return {"ok": False, "msg": "Empty history returned."}
    out: Dict[str, Any] = {"ok": True}
    out["rows"] = int(hist.shape[0])
    out["cols"] = int(hist.shape[1])
    try:
        out["start"] = str(hist.index.min().date()) if hasattr(hist.index.min(), "date") else str(hist.index.min())
        out["end"] = str(hist.index.max().date()) if hasattr(hist.index.max(), "date") else str(hist.index.max())
    except Exception:
        out["start"] = "—"
        out["end"] = "—"

    # Duplicate index?
    try:
        out["dupe_dates"] = int(hist.index.duplicated().sum())
    except Exception:
        out["dupe_dates"] = 0

    key_cols = ["wave_nav","bm_nav","wave_ret","bm_ret"]
    for c in key_cols:
        if c not in hist.columns:
            out[f"missing_{c}"] = 1.0
        else:
            s = pd.to_numeric(hist[c], errors="coerce")
            out[f"missing_{c}"] = float(s.isna().mean())

    # NAV sanity: should be positive
    if "wave_nav" in hist.columns:
        wnav = pd.to_numeric(hist["wave_nav"], errors="coerce")
        out["wave_nav_min"] = float(np.nanmin(wnav.values)) if len(wnav) else float("nan")
        out["wave_nav_nonpos"] = int(np.sum((wnav <= 0).fillna(False)))
    else:
        out["wave_nav_min"] = float("nan")
        out["wave_nav_nonpos"] = 0

    if "bm_nav" in hist.columns:
        bnav = pd.to_numeric(hist["bm_nav"], errors="coerce")
        out["bm_nav_min"] = float(np.nanmin(bnav.values)) if len(bnav) else float("nan")
        out["bm_nav_nonpos"] = int(np.sum((bnav <= 0).fillna(False)))
    else:
        out["bm_nav_min"] = float("nan")
        out["bm_nav_nonpos"] = 0

    # Flag summary
    flags = []
    if out.get("dupe_dates", 0) > 0:
        flags.append("Duplicate dates found")
    for c in key_cols:
        if out.get(f"missing_{c}", 0.0) > 0.05:
            flags.append(f"Missing data >5% in {c}")
    if out.get("wave_nav_nonpos", 0) > 0:
        flags.append("Non-positive wave_nav values")
    if out.get("bm_nav_nonpos", 0) > 0:
        flags.append("Non-positive bm_nav values")
    out["flags"] = flags
    return out

# ============================================================
# END PART 2 / 5
# Next: PART 3 adds Wave Doctor + Mode Separation Proof + rolling charts + persistence panel
# ============================================================
# ============================================================
# PART 3 / 5 — Wave Doctor++ + Mode Separation Proof + Rolling Panels
# ============================================================

# ============================================================
# Wave Doctor (Diagnostics++)
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

    nav_w = hist.get("wave_nav", pd.Series(dtype=float))
    nav_b = hist.get("bm_nav", pd.Series(dtype=float))
    ret_w = hist.get("wave_ret", pd.Series(dtype=float))
    ret_b = hist.get("bm_ret", pd.Series(dtype=float))

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
    drift_str, drift_flag = beta_drift_flags(beta_real, beta_target, thresh=beta_drift_warn)

    spy_nav = compute_spy_nav(days=days)
    spy_ret = ret_from_nav(spy_nav, len(spy_nav)) if len(spy_nav) >= 2 else float("nan")
    bm_difficulty = (r365_b - spy_ret) if (pd.notna(r365_b) and pd.notna(spy_ret)) else float("nan")

    alpha_captured = compute_alpha_captured_series(hist)
    alpha_capture_30d = float(alpha_captured.tail(min(30, len(alpha_captured))).sum()) if len(alpha_captured) else float("nan")

    flags: List[str] = []
    diagnosis: List[str] = []
    recs: List[str] = []

    # Data quality audit
    audit = audit_history_df(hist)
    if audit.get("flags"):
        flags.extend([f"DATA: {x}" for x in audit["flags"]])
        diagnosis.append("History audit detected data-quality flags. Validate coverage and engine logging inputs.")

    # Alpha swings
    if pd.notna(a30) and abs(a30) >= alpha_warn:
        flags.append("Large 30D alpha (verify benchmark + coverage)")
        diagnosis.append("30D alpha is unusually large. Could be real signal OR benchmark mix / coverage shift.")
        recs.append("Check Benchmark Truth panel; freeze benchmark mix snapshot for demo reproducibility.")

    # 365D alpha context
    if pd.notna(a365) and a365 < 0:
        diagnosis.append("365D alpha is negative (could be true underperformance or tougher benchmark).")
        if pd.notna(bm_difficulty) and bm_difficulty > 0.03:
            diagnosis.append("Benchmark difficulty is positive vs SPY (BM outperformed SPY), making alpha harder.")
            recs.append("Cross-check vs SPY/QQQ in Benchmark Truth to isolate engine effect.")
    elif pd.notna(a365) and a365 > 0.06:
        diagnosis.append("365D alpha is solidly positive. Snapshot benchmark mix for reproducible demos.")

    # Risk flags
    if pd.notna(te) and te > te_warn:
        flags.append("High tracking error (active risk elevated)")
        diagnosis.append("Tracking error is high; Wave deviates strongly from its benchmark.")
        recs.append("Use What-If Lab to tighten exposure caps (shadow only) and observe TE reduction.")

    if pd.notna(vol_w) and vol_w > vol_warn:
        flags.append("High volatility")
        diagnosis.append("Volatility is elevated relative to typical institutional tolerances.")
        recs.append("Use What-If Lab to lower vol target (shadow only).")

    if pd.notna(mdd_w) and mdd_w < mdd_warn:
        flags.append("Deep drawdown")
        diagnosis.append("Drawdown is deep. Consider stronger SmartSafe posture in stress regimes.")
        recs.append("Use What-If Lab to increase safe fraction (shadow only).")

    # Beta drift
    if drift_flag:
        flags.append(f"Beta drift vs target (Δβ={drift_str})")
        diagnosis.append("β_real deviates from target for this mode. This can explain similarity across modes.")
        recs.append("Verify mode separation in the Mode Proof panel; ensure engine applies mode scaling distinctly.")

    # Alpha captured lens
    if pd.notna(alpha_capture_30d) and abs(alpha_capture_30d) > alpha_warn:
        diagnosis.append("Alpha Captured (sum of daily active returns) is large on 30D window; check persistence & drift.")
        recs.append("Review Rolling Diagnostics: alpha persistence and rolling beta/TE.")

    if not diagnosis:
        diagnosis.append("No major anomalies detected by Wave Doctor on the selected window.")

    return {
        "ok": True,
        "metrics": {
            "Return_365D": r365_w,
            "Alpha_365D": a365,
            "Return_30D": r30_w,
            "Alpha_30D": a30,
            "AlphaCaptured_30D": alpha_capture_30d,
            "Vol_Wave": vol_w,
            "Vol_Benchmark": vol_b,
            "TE": te,
            "IR": ir,
            "Beta_Real": beta_real,
            "Beta_Target": beta_target,
            "Beta_Drift": (beta_real - beta_target) if (math.isfinite(beta_real) and math.isfinite(beta_target)) else float("nan"),
            "MaxDD_Wave": mdd_w,
            "MaxDD_Benchmark": mdd_b,
            "Benchmark_Difficulty_BM_minus_SPY": bm_difficulty,
        },
        "audit": audit,
        "flags": list(dict.fromkeys(flags)),
        "diagnosis": diagnosis,
        "recommendations": list(dict.fromkeys(recs)),
    }

# ============================================================
# Mode Separation Proof (ALL modes side-by-side)
# ============================================================
def summarize_hist(hist: pd.DataFrame) -> Dict[str, float]:
    if hist is None or hist.empty or len(hist) < 2:
        return {
            "1D Ret": float("nan"),
            "30D Ret": float("nan"),
            "60D Ret": float("nan"),
            "365D Ret": float("nan"),
            "1D Alpha": float("nan"),
            "30D Alpha": float("nan"),
            "60D Alpha": float("nan"),
            "365D Alpha": float("nan"),
            "TE": float("nan"),
            "IR": float("nan"),
            "Vol": float("nan"),
            "MaxDD": float("nan"),
            "Beta": float("nan"),
            "AlphaCaptured_30D": float("nan"),
        }

    nav_w = hist["wave_nav"]
    nav_b = hist["bm_nav"]
    wret = hist["wave_ret"]
    bret = hist["bm_ret"]

    # 1D
    if len(nav_w) >= 2 and len(nav_b) >= 2:
        r1w = float(nav_w.iloc[-1] / nav_w.iloc[-2] - 1.0)
        r1b = float(nav_b.iloc[-1] / nav_b.iloc[-2] - 1.0)
        a1 = r1w - r1b
    else:
        r1w = float("nan")
        a1 = float("nan")

    r30w = ret_from_nav(nav_w, min(30, len(nav_w)))
    r30b = ret_from_nav(nav_b, min(30, len(nav_b)))
    a30 = r30w - r30b

    r60w = ret_from_nav(nav_w, min(60, len(nav_w)))
    r60b = ret_from_nav(nav_b, min(60, len(nav_b)))
    a60 = r60w - r60b

    r365w = ret_from_nav(nav_w, len(nav_w))
    r365b = ret_from_nav(nav_b, len(nav_b))
    a365 = r365w - r365b

    te = tracking_error(wret, bret)
    ir = information_ratio(nav_w, nav_b, te)
    vol = annualized_vol(wret)
    mdd = max_drawdown(nav_w)
    beta = beta_vs_benchmark(wret, bret)

    ac = compute_alpha_captured_series(hist)
    ac30 = float(ac.tail(min(30, len(ac))).sum()) if len(ac) else float("nan")

    return {
        "1D Ret": r1w,
        "30D Ret": r30w,
        "60D Ret": r60w,
        "365D Ret": r365w,
        "1D Alpha": a1,
        "30D Alpha": a30,
        "60D Alpha": a60,
        "365D Alpha": a365,
        "TE": te,
        "IR": ir,
        "Vol": vol,
        "MaxDD": mdd,
        "Beta": beta,
        "AlphaCaptured_30D": ac30,
    }

@st.cache_data(show_spinner=False)
def mode_proof_table(wave_name: str, modes: List[str], days: int = 365) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for m in modes:
        hist = compute_wave_history(wave_name, mode=m, days=days)
        summ = summarize_hist(hist)
        beta_target = get_beta_target_if_available(m)
        drift = (summ["Beta"] - beta_target) if (math.isfinite(summ["Beta"]) and math.isfinite(beta_target)) else float("nan")
        row = {"Mode": m, **summ, "β_target": beta_target, "β_drift": drift}
        rows.append(row)
    df = pd.DataFrame(rows) if rows else pd.DataFrame()
    return df

def plot_mode_nav_overlay(wave_name: str, modes: List[str], days: int = 365, title: str = "Mode NAV Overlay"):
    if go is None:
        st.info("Plotly not available; NAV overlay uses Streamlit line chart fallback.")
        # fallback: build dataframe
        data = {}
        for m in modes:
            hist = compute_wave_history(wave_name, mode=m, days=days)
            if hist is None or hist.empty or "wave_nav" not in hist.columns:
                continue
            data[m] = pd.to_numeric(hist["wave_nav"], errors="coerce")
        if not data:
            st.warning("No mode NAV data available.")
            return
        df = pd.DataFrame(data).dropna(how="all")
        st.line_chart(df)
        return

    fig = go.Figure()
    added = False
    for m in modes:
        hist = compute_wave_history(wave_name, mode=m, days=days)
        if hist is None or hist.empty:
            continue
        if "wave_nav" not in hist.columns:
            continue
        s = pd.to_numeric(hist["wave_nav"], errors="coerce").astype(float)
        if s.dropna().empty:
            continue
        fig.add_trace(go.Scatter(x=s.index, y=s, name=f"{m} — Wave NAV", mode="lines"))
        added = True

    if not added:
        st.warning("No NAV series returned for modes.")
        return

    fig.update_layout(
        title=title,
        height=420,
        margin=dict(l=40, r=40, t=50, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis_title="Date",
        yaxis_title="NAV",
    )
    st.plotly_chart(fig, use_container_width=True)

# ============================================================
# Rolling Diagnostics Panel helpers
# ============================================================
def plot_rolling_panels(hist: pd.DataFrame, mode: str, window: int = 60, title_prefix: str = ""):
    if hist is None or hist.empty:
        st.warning("No history for rolling diagnostics.")
        return

    roll = rolling_metrics(hist, window=window)
    if roll is None or roll.empty:
        st.warning("Rolling metrics unavailable.")
        return

    # add alpha captured daily for line chart context
    ac = compute_alpha_captured_series(hist)
    ac_name = "alpha_captured"
    df_plot = roll.copy()
    if len(ac) == len(hist):
        df_plot[ac_name] = ac

    # Plotly optional
    if go is None:
        st.write("Plotly missing — showing rolling metrics tables only.")
        st.dataframe(df_plot.tail(240), use_container_width=True)
        return

    # 1) Rolling alpha + persistence
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=df_plot.index, y=df_plot["roll_alpha"], name=f"Rolling Alpha (ann, {window}d)", mode="lines"))
    fig1.add_trace(go.Scatter(x=df_plot.index, y=df_plot["alpha_persistence"], name=f"Alpha Persistence ({window}d)", mode="lines", yaxis="y2"))
    fig1.update_layout(
        title=f"{title_prefix} Rolling Alpha + Persistence — Mode: {mode}",
        height=380,
        margin=dict(l=40, r=40, t=55, b=40),
        xaxis=dict(title="Date"),
        yaxis=dict(title="Rolling Alpha (annualized)"),
        yaxis2=dict(title="Persistence (0–1)", overlaying="y", side="right", range=[0, 1]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig1, use_container_width=True)

    # 2) Rolling TE + Beta
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=df_plot.index, y=df_plot["roll_te"], name=f"Rolling TE (ann, {window}d)", mode="lines"))
    fig2.add_trace(go.Scatter(x=df_plot.index, y=df_plot["roll_beta"], name=f"Rolling Beta ({window}d)", mode="lines", yaxis="y2"))
    fig2.update_layout(
        title=f"{title_prefix} Rolling TE + Beta — Mode: {mode}",
        height=380,
        margin=dict(l=40, r=40, t=55, b=40),
        xaxis=dict(title="Date"),
        yaxis=dict(title="TE (annualized)"),
        yaxis2=dict(title="Beta", overlaying="y", side="right"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig2, use_container_width=True)

    # 3) Rolling Vol (wave vs bm) + Alpha Captured (daily)
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=df_plot.index, y=df_plot["roll_vol"], name=f"Wave Vol (ann, {window}d)", mode="lines"))
    fig3.add_trace(go.Scatter(x=df_plot.index, y=df_plot["roll_bm_vol"], name=f"BM Vol (ann, {window}d)", mode="lines"))
    if ac_name in df_plot.columns:
        fig3.add_trace(go.Scatter(x=df_plot.index, y=df_plot[ac_name], name="Alpha Captured (daily)", mode="lines", yaxis="y2"))
    fig3.update_layout(
        title=f"{title_prefix} Rolling Vol + Alpha Captured — Mode: {mode}",
        height=380,
        margin=dict(l=40, r=40, t=55, b=40),
        xaxis=dict(title="Date"),
        yaxis=dict(title="Vol (annualized)"),
        yaxis2=dict(title="Alpha Captured (daily)", overlaying="y", side="right"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig3, use_container_width=True)

    # Table for deep scan
    st.markdown("#### Rolling Metrics Table (tail)")
    st.dataframe(df_plot.tail(180), use_container_width=True)

# ============================================================
# Diagnostics++: cross-wave scan panels (alpha captured, TE, beta)
# ============================================================
@st.cache_data(show_spinner=False)
def crosswave_diagnostics_table(all_waves: List[str], mode: str, days: int = 365) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for wname in all_waves:
        hist = compute_wave_history(wname, mode=mode, days=days)
        if hist is None or hist.empty or len(hist) < 2:
            rows.append(
                {
                    "Wave": wname,
                    "30D Alpha": np.nan,
                    "365D Alpha": np.nan,
                    "30D AlphaCaptured(sum)": np.nan,
                    "TE": np.nan,
                    "IR": np.nan,
                    "Beta": np.nan,
                    "Vol": np.nan,
                    "MaxDD": np.nan,
                    "Data Flags": "NO DATA",
                }
            )
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
        beta = beta_vs_benchmark(wret, bret)
        vol = annualized_vol(wret)
        mdd = max_drawdown(nav_w)

        ac = compute_alpha_captured_series(hist)
        ac30 = float(ac.tail(min(30, len(ac))).sum()) if len(ac) else float("nan")

        audit = audit_history_df(hist)
        flags = " | ".join(audit.get("flags", [])) if audit.get("flags") else ""

        rows.append(
            {
                "Wave": wname,
                "30D Alpha": a30,
                "365D Alpha": a365,
                "30D AlphaCaptured(sum)": ac30,
                "TE": te,
                "IR": ir,
                "Beta": beta,
                "Vol": vol,
                "MaxDD": mdd,
                "Data Flags": flags,
            }
        )

    df = pd.DataFrame(rows) if rows else pd.DataFrame()
    if df.empty:
        return df
    # Rank score: favor positive alpha and lower TE
    df["RankScore"] = (df["365D Alpha"].fillna(0) * 0.6 + df["30D Alpha"].fillna(0) * 0.4) - (df["TE"].fillna(0) * 0.15)
    return df.sort_values("RankScore", ascending=False)

# ============================================================
# END PART 3 / 5
# Next: PART 4 adds What-If Lab + Heatmap + Correlations + UI Tabs + Summary Bar + Market/Factors/Vector
# ============================================================
# ============================================================
# PART 4 / 5 — What-If Lab + Heatmaps + Correlations + UI Wiring (major tabs)
# ============================================================

# ============================================================
# Alpha Heatmap View (All Waves x Timeframe)
# ============================================================
@st.cache_data(show_spinner=False)
def build_alpha_matrix(all_waves: List[str], mode: str, days: int = 365) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for wname in all_waves:
        hist = compute_wave_history(wname, mode=mode, days=days)
        if hist is None or hist.empty or len(hist) < 2:
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

    df = pd.DataFrame(rows) if rows else pd.DataFrame()
    return df.sort_values("Wave") if not df.empty else df


def plot_alpha_heatmap(alpha_df: pd.DataFrame, selected_wave: str, title: str):
    if go is None or alpha_df is None or alpha_df.empty:
        st.info("Heatmap unavailable (Plotly missing or no data).")
        return

    df = alpha_df.copy()
    cols = [c for c in ["1D Alpha", "30D Alpha", "60D Alpha", "365D Alpha"] if c in df.columns]
    if not cols:
        st.info("No alpha columns to plot.")
        return

    # Move selected wave to top
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

    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=x,
            y=y,
            zmin=-v,
            zmax=v,
            colorbar=dict(title="Alpha"),
        )
    )
    fig.update_layout(
        title=title,
        height=min(980, 260 + 22 * max(10, len(y))),
        margin=dict(l=90, r=40, t=70, b=40),
        xaxis_title="Timeframe",
        yaxis_title="Wave",
    )
    st.plotly_chart(fig, use_container_width=True)


# ============================================================
# Correlation Matrix across waves (returns)
# ============================================================
@st.cache_data(show_spinner=False)
def build_wave_returns_matrix(all_waves: List[str], mode: str, days: int = 365) -> pd.DataFrame:
    """Aligned daily returns per wave (engine wave_ret)."""
    series: Dict[str, pd.Series] = {}
    for wname in all_waves:
        hist = compute_wave_history(wname, mode=mode, days=days)
        if hist is None or hist.empty or "wave_ret" not in hist.columns:
            continue
        s = pd.to_numeric(hist["wave_ret"], errors="coerce").rename(wname)
        if s.dropna().shape[0] < 30:
            continue
        series[wname] = s

    if not series:
        return pd.DataFrame()

    df = pd.concat(series.values(), axis=1).dropna(how="all")
    # require at least 60 aligned rows for correlation to be meaningful
    if df.dropna().shape[0] < 60:
        return pd.DataFrame()
    return df


def plot_corr_heatmap(corr: pd.DataFrame, title: str = "Wave Correlation Matrix"):
    if corr is None or corr.empty:
        st.info("Correlation matrix unavailable (not enough aligned history).")
        return

    if go is None:
        st.dataframe(corr, use_container_width=True)
        return

    fig = go.Figure(
        data=go.Heatmap(
            z=corr.values,
            x=corr.columns.tolist(),
            y=corr.index.tolist(),
            zmin=-1,
            zmax=1,
            colorbar=dict(title="ρ"),
        )
    )
    fig.update_layout(
        title=title,
        height=min(980, 240 + 18 * max(12, len(corr.index))),
        margin=dict(l=110, r=40, t=70, b=60),
    )
    st.plotly_chart(fig, use_container_width=True)


# ============================================================
# What-If Lab (shadow sim) — COMPLETE & FIXED (no unclosed parens)
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
    """
    Shadow simulation ONLY.
    - Builds a static basket from holdings.
    - Applies momentum tilt + vol targeting + regime/VIX gating.
    - Can freeze benchmark to engine bm_nav (demo reproducibility).
    """
    hold_df = get_wave_holdings(wave_name)
    weights = _weights_from_df(hold_df, "Ticker", "Weight")
    if weights is None or weights.empty:
        return pd.DataFrame()

    tickers = list(weights.index)
    needed = set(tickers + ["SPY", "^VIX", "SGOV", "BIL", "SHY"])
    px = fetch_prices_daily(list(needed), days=days)
    if px is None or px.empty:
        return pd.DataFrame()
    if "SPY" not in px.columns or "^VIX" not in px.columns:
        return pd.DataFrame()

    px = px.sort_index().ffill().bfill()
    if len(px) > days:
        px = px.iloc[-days:]

    rets = px.pct_change().fillna(0.0)
    w_all = weights.reindex(px.columns).fillna(0.0)

    spy_nav = (1.0 + rets["SPY"]).cumprod()
    regime = _regime_from_spy_60d(spy_nav)
    vix = px["^VIX"]

    vix_exposure = _vix_exposure_factor_series(vix, mode)
    vix_safe = _vix_safe_fraction_series(vix, mode)

    # base exposure from engine if available
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
        # engine override if exists
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

    # safe asset proxy
    safe_ticker = "SGOV" if "SGOV" in rets.columns else ("BIL" if "BIL" in rets.columns else ("SHY" if "SHY" in rets.columns else "SPY"))
    safe_ret = rets[safe_ticker]

    # 60D momentum
    mom60 = px / px.shift(60) - 1.0

    wave_ret_list: List[float] = []
    date_list: List[pd.Timestamp] = []

    for dtt in rets.index:
        r = rets.loc[dtt]

        # momentum tilt
        if dtt in mom60.index:
            mom_row = mom60.loc[dtt]
            mom_clipped = mom_row.reindex(px.columns).fillna(0.0).clip(lower=-0.30, upper=0.30)
            tilt = 1.0 + tilt_strength * mom_clipped
            ew = (w_all * tilt).clip(lower=0.0)
        else:
            ew = w_all.copy()

        ew_hold = ew.reindex(tickers).fillna(0.0)
        s = float(ew_hold.sum())
        if s > 0:
            rw = ew_hold / s
        else:
            rw = weights.reindex(tickers).fillna(0.0)
            s2 = float(rw.sum())
            rw = (rw / s2) if s2 > 0 else rw

        port_risk_ret = float((r.reindex(tickers).fillna(0.0) * rw).sum())

        # realized vol for vol targeting
        if len(wave_ret_list) >= 20:
            recent = np.array(wave_ret_list[-20:], dtype=float)
            realized = float(np.nanstd(recent) * np.sqrt(252))
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

        # optional PL "shock symmetry" tweak (shadow only)
        if mode == "Private Logic" and len(wave_ret_list) >= 20:
            recent = np.array(wave_ret_list[-20:], dtype=float)
            daily_vol = float(np.nanstd(recent))
            if daily_vol > 0 and math.isfinite(daily_vol):
                shock = 2.0 * daily_vol
                if total <= -shock:
                    total = total * 1.30
                elif total >= shock:
                    total = total * 0.70

        wave_ret_list.append(float(total))
        date_list.append(pd.Timestamp(dtt))

    wave_ret_s = pd.Series(wave_ret_list, index=pd.Index(date_list, name="Date"), name="whatif_ret")
    wave_nav = (1.0 + wave_ret_s).cumprod().rename("whatif_nav")

    out = pd.DataFrame({"whatif_nav": wave_nav, "whatif_ret": wave_ret_s})

    if freeze_benchmark:
        hist_eng = compute_wave_history(wave_name, mode=mode, days=days)
        if hist_eng is not None and not hist_eng.empty and "bm_nav" in hist_eng.columns:
            out["bm_nav"] = hist_eng["bm_nav"].reindex(out.index).ffill().bfill()
            out["bm_ret"] = hist_eng["bm_ret"].reindex(out.index).fillna(0.0)
    else:
        spy_ret = rets["SPY"].reindex(out.index).fillna(0.0)
        out["bm_ret"] = spy_ret
        out["bm_nav"] = (1.0 + spy_ret).cumprod()

    return out


# ============================================================
# UI Wiring — Sidebar + Page header + Tabs (Diagnostics++)
# ============================================================

# IMPORTANT: Ensure "__future__ import" remains ONLY at very top of file.
# If you see "from __future__ import annotations" anywhere below the first lines, DELETE IT.

# Pull waves & modes from engine
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
    st.caption("Institutional Console • Diagnostics++ • Vector OS™")

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
    st.markdown("**Wave Doctor thresholds**")
    alpha_warn = st.slider("Alpha alert threshold (abs, 30D)", 0.02, 0.25, 0.08, 0.01)
    te_warn = st.slider("Tracking error alert threshold", 0.05, 0.50, 0.20, 0.01)
    beta_drift_warn = st.slider("Beta drift threshold (abs Δβ)", 0.02, 0.20, 0.07, 0.01)

    st.markdown("---")
    st.markdown("**Rolling diagnostics**")
    roll_window = st.slider("Rolling window (days)", 20, 200, 60, 5)

mode = st.session_state["mode"]
selected_wave = st.session_state["selected_wave"]

if not selected_wave:
    st.error("No Waves available (engine returned empty list). Verify waves_engine.get_all_waves().")
    st.stop()


# ============================================================
# Page header
# ============================================================
st.title("WAVES Intelligence™ Institutional Console — Diagnostics++")
st.caption("Wave-first scan • Mode Proof • Rolling diagnostics • Correlations • Attribution • Vector OS Insight")


# ============================================================
# Pinned Summary Bar (Sticky) — Diagnostics++
# ============================================================
h_bar = compute_wave_history(selected_wave, mode=mode, days=max(365, history_days))
bar_r30 = bar_a30 = bar_r365 = bar_a365 = float("nan")
bar_te = bar_ir = float("nan")
bar_beta = bar_beta_t = bar_beta_d = float("nan")
bar_src = "—"

bm_mix_for_src = get_benchmark_mix()
bar_src = benchmark_source_label(selected_wave, bm_mix_for_src)

if h_bar is not None and not h_bar.empty and len(h_bar) >= 2:
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
    bar_beta_d = (bar_beta - bar_beta_t) if (math.isfinite(bar_beta) and math.isfinite(bar_beta_t)) else float("nan")

# Regime + VIX
spy_vix = fetch_spy_vix(days=min(365, max(120, history_days)))
reg_now = "—"
vix_last = float("nan")
if spy_vix is not None and not spy_vix.empty and "SPY" in spy_vix.columns and "^VIX" in spy_vix.columns and len(spy_vix) > 60:
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

# WaveScore snapshot
ws_snap = compute_wavescore_for_all_waves(all_waves, mode=mode, days=365)
ws_val = "—"
ws_grade = "—"
if ws_snap is not None and not ws_snap.empty:
    rr = ws_snap[ws_snap["Wave"] == selected_wave]
    if not rr.empty:
        ws_val = fmt_score(float(rr.iloc[0]["WaveScore"]))
        ws_grade = str(rr.iloc[0]["Grade"])

beta_drift_str, beta_drift_flag = beta_drift_flags(bar_beta, bar_beta_t, thresh=beta_drift_warn)
beta_chip = f"{fmt_num(bar_beta,2)} vs {fmt_num(bar_beta_t,2)} (Δ{beta_drift_str})" if (math.isfinite(bar_beta) or math.isfinite(bar_beta_t)) else "—"

st.markdown(
    f"""
<div class="waves-sticky">
  <div class="waves-hdr">📌 Live Summary (Diagnostics++)</div>
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

if beta_drift_flag:
    st.warning(f"Beta drift flag: Δβ={beta_drift_str} exceeds threshold ({fmt_num(beta_drift_warn,2)}). Review Mode Proof + Rolling Beta.")


# ============================================================
# Tabs — expanded
# ============================================================
tab_console, tab_modeproof, tab_rolling, tab_corr, tab_market, tab_factors, tab_vector = st.tabs(
    [
        "Console Scan",
        "Mode Proof",
        "Rolling Diagnostics",
        "Correlations",
        "Market Intel",
        "Factor Decomp",
        "Vector OS Insight",
    ]
)


# ============================================================
# TAB: Console Scan — Heatmap + Jump + Overview + Holdings + Benchmark Truth + Attribution + Doctor + What-If
# ============================================================
with tab_console:
    st.subheader("🔥 Alpha Heatmap View (All Waves × Timeframe)")
    st.caption("Fast scan. Selected wave pinned to top in heatmap; tables highlight alpha cells. Display-only.")

    alpha_df = build_alpha_matrix(all_waves, mode, days=min(365, max(120, history_days)))
    plot_alpha_heatmap(alpha_df, selected_wave, title=f"Alpha Heatmap — Mode: {mode}")

    st.markdown("### 🧭 Jump Table (Ranked by average alpha)")
    jump_df = alpha_df.copy() if alpha_df is not None else pd.DataFrame()
    if jump_df is None or jump_df.empty:
        st.info("No alpha matrix available.")
    else:
        jump_df["RankScore"] = jump_df[["1D Alpha", "30D Alpha", "60D Alpha", "365D Alpha"]].mean(axis=1, skipna=True)
        jump_df = jump_df.sort_values("RankScore", ascending=False)
        show_df(jump_df, selected_wave, key="jump_table_fmt")
        selectable_table_jump(jump_df, key="jump_table_sel")

    st.markdown("---")

    st.subheader("🧾 All Waves Overview (Returns + Alpha + Risk lenses)")
    overview = crosswave_diagnostics_table(all_waves, mode, days=min(365, max(120, history_days)))
    if overview is None or overview.empty:
        st.info("Overview unavailable (insufficient history).")
    else:
        show_df(overview, selected_wave, key="overview_diag")

    st.markdown("---")

    st.subheader(f"📌 Selected Wave Holdings — {selected_wave}")
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

        st.markdown("#### Top 10 Holdings (clickable)")
        for _, r in top10.iterrows():
            t = str(r.get("Ticker", "")).strip()
            wgt = r.get("Weight", np.nan)
            nm = str(r.get("Name", t))
            if t:
                st.markdown(f"- **[{t}]({google_quote_url(t)})** — {nm} — **{fmt_pct(wgt)}**")

        st.markdown("#### Full Holdings")
        show_df(hold2, selected_wave, key="hold_full")

    st.markdown("---")

    st.subheader("✅ Benchmark Truth + Attribution (Engine vs Basket)")
    colA, colB = st.columns(2)

    with colA:
        st.markdown("#### Benchmark Mix (engine)")
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
        attrib = compute_alpha_attribution(selected_wave, mode=mode, days=min(365, max(120, history_days)))
        if not attrib:
            st.info("Attribution unavailable.")
        else:
            arows = []
            for k, v in attrib.items():
                if any(x in k for x in ["Return", "Alpha", "Vol", "MaxDD", "TE", "Difficulty"]):
                    arows.append({"Metric": k, "Value": fmt_pct(v)})
                elif "IR" in k or "β_" in k:
                    arows.append({"Metric": k, "Value": fmt_num(v, 2)})
                else:
                    arows.append({"Metric": k, "Value": fmt_num(v, 5)})
            st.dataframe(pd.DataFrame(arows), use_container_width=True)

    st.markdown("---")

    st.subheader("🩺 Wave Doctor (Diagnostics++)")
    wd = wave_doctor_assess(
        selected_wave,
        mode=mode,
        days=min(365, max(120, history_days)),
        alpha_warn=alpha_warn,
        te_warn=te_warn,
        beta_drift_warn=beta_drift_warn,
    )
    if not wd.get("ok", False):
        st.info(wd.get("message", "Wave Doctor unavailable."))
    else:
        m = wd["metrics"]
        mdf = pd.DataFrame(
            [
                {"Metric": "365D Return", "Value": fmt_pct(m.get("Return_365D"))},
                {"Metric": "365D Alpha", "Value": fmt_pct(m.get("Alpha_365D"))},
                {"Metric": "30D Return", "Value": fmt_pct(m.get("Return_30D"))},
                {"Metric": "30D Alpha", "Value": fmt_pct(m.get("Alpha_30D"))},
                {"Metric": "30D Alpha Captured (sum)", "Value": fmt_pct(m.get("AlphaCaptured_30D"))},
                {"Metric": "Vol (Wave)", "Value": fmt_pct(m.get("Vol_Wave"))},
                {"Metric": "Vol (Benchmark)", "Value": fmt_pct(m.get("Vol_Benchmark"))},
                {"Metric": "Tracking Error (TE)", "Value": fmt_pct(m.get("TE"))},
                {"Metric": "Information Ratio (IR)", "Value": fmt_num(m.get("IR"), 2)},
                {"Metric": "β_real", "Value": fmt_num(m.get("Beta_Real"), 2)},
                {"Metric": "β_target", "Value": fmt_num(m.get("Beta_Target"), 2)},
                {"Metric": "β_drift", "Value": fmt_num(m.get("Beta_Drift"), 2)},
                {"Metric": "MaxDD (Wave)", "Value": fmt_pct(m.get("MaxDD_Wave"))},
                {"Metric": "MaxDD (Benchmark)", "Value": fmt_pct(m.get("MaxDD_Benchmark"))},
                {"Metric": "BM Difficulty (BM - SPY)", "Value": fmt_pct(m.get("Benchmark_Difficulty_BM_minus_SPY"))},
            ]
        )
        st.dataframe(mdf, use_container_width=True)

        # audit
        aud = wd.get("audit", {})
        if aud:
            st.markdown("#### Data Quality / Coverage Audit")
            st.write(f"- Rows: {aud.get('rows')} | Null % (wave_ret): {fmt_num(aud.get('null_pct_wave_ret'),1)}% | Null % (bm_ret): {fmt_num(aud.get('null_pct_bm_ret'),1)}%")
            if aud.get("start"):
                st.write(f"- Start: {aud.get('start')} | End: {aud.get('end')}")
            if aud.get("flags"):
                st.warning("Audit flags: " + " | ".join(aud.get("flags", [])))

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
    st.caption("This does NOT change engine math. It is a sandbox overlay simulation for diagnostics only.")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        tilt_strength = st.slider("Tilt strength", 0.0, 1.0, 0.30, 0.05)
    with c2:
        vol_target = st.slider("Vol target (annual)", 0.05, 0.60, 0.20, 0.01)
    with c3:
        extra_safe = st.slider("Extra safe boost", 0.0, 0.50, 0.00, 0.01)
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
            bm_total = ret_from_nav(bm_nav, len(bm_nav)) if bm_nav is not None and len(bm_nav) > 1 else 0.0
            alpha_total = ret_total - bm_total

            st.markdown(f"**What-If Return:** {fmt_pct(ret_total)}   |   **What-If Alpha:** {fmt_pct(alpha_total)}")

            if go is not None:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=sim.index, y=sim["whatif_nav"], name="What-If NAV", mode="lines"))
                if "bm_nav" in sim.columns:
                    fig.add_trace(go.Scatter(x=sim.index, y=sim["bm_nav"], name="Benchmark NAV", mode="lines"))
                fig.update_layout(height=380, margin=dict(l=40, r=40, t=40, b=40))
                st.plotly_chart(fig, use_container_width=True)
            else:
                cols = ["whatif_nav"] + (["bm_nav"] if "bm_nav" in sim.columns else [])
                st.line_chart(sim[cols])

            st.markdown("#### What-If Diagnostics Table")
            sim_tbl = pd.DataFrame(
                [
                    {"Metric": "What-If Return", "Value": fmt_pct(ret_total)},
                    {"Metric": "What-If Benchmark Return", "Value": fmt_pct(bm_total)},
                    {"Metric": "What-If Alpha", "Value": fmt_pct(alpha_total)},
                ]
            )
            st.dataframe(sim_tbl, use_container_width=True)


# ============================================================
# TAB: Mode Proof — side-by-side + NAV overlay
# ============================================================
with tab_modeproof:
    st.subheader("🧾 Mode Separation Proof — Side-by-Side Metrics")
    st.caption("If Standard / Alpha-Minus-Beta / Private Logic look identical, this panel exposes it clearly.")

    if not all_modes:
        st.warning("Engine did not provide modes. Verify waves_engine.get_modes().")
    else:
        mp = mode_proof_table(selected_wave, all_modes, days=min(365, max(120, history_days)))
        if mp is None or mp.empty:
            st.info("Mode proof table unavailable.")
        else:
            show_df(mp.rename(columns={"Mode": "Wave"}).rename(columns={"Wave": "Mode"}), selected_wave, key="modeproof_table")  # safe formatting reuse
            st.markdown("#### NAV Overlay across Modes")
            plot_mode_nav_overlay(selected_wave, all_modes, days=min(365, max(120, history_days)), title=f"Mode NAV Overlay — {selected_wave}")

        st.markdown("---")
        st.subheader("Mode Proof — Interpretation Aids")
        st.write("- **If β_real is nearly identical across modes**, mode scaling may not be applied (or targets not set).")
        st.write("- **If TE and vol are identical**, exposures or gating might not be mode-specific.")
        st.write("- Use Rolling Diagnostics tab to see if differences emerge over time.")


# ============================================================
# TAB: Rolling Diagnostics — rolling alpha/TE/beta/vol + persistence
# ============================================================
with tab_rolling:
    st.subheader("📈 Rolling Diagnostics (Rolling Alpha / TE / Beta / Vol + Persistence)")
    st.caption("Uses engine wave_ret and bm_ret. Rolling alpha is annualized excess over BM on a rolling window.")

    hist_sel = compute_wave_history(selected_wave, mode=mode, days=min(730, max(240, history_days)))
    if hist_sel is None or hist_sel.empty:
        st.warning("History unavailable for rolling diagnostics.")
    else:
        plot_rolling_panels(hist_sel, mode=mode, window=roll_window, title_prefix=selected_wave)

        st.markdown("---")
        st.subheader("Rolling Flag Summary")
        roll = rolling_metrics(hist_sel, window=roll_window)
        if roll is None or roll.empty:
            st.info("Rolling metric series not available.")
        else:
            last = roll.tail(1)
            if not last.empty:
                la = float(last["roll_alpha"].iloc[0]) if "roll_alpha" in last.columns else float("nan")
                lte = float(last["roll_te"].iloc[0]) if "roll_te" in last.columns else float("nan")
                lb = float(last["roll_beta"].iloc[0]) if "roll_beta" in last.columns else float("nan")
                lp = float(last["alpha_persistence"].iloc[0]) if "alpha_persistence" in last.columns else float("nan")

                beta_target = get_beta_target_if_available(mode)
                drift_str, drift_flag = beta_drift_flags(lb, beta_target, thresh=beta_drift_warn)

                st.write(f"- Rolling Alpha (ann): **{fmt_pct(la)}**")
                st.write(f"- Rolling TE (ann): **{fmt_pct(lte)}**")
                st.write(f"- Rolling Beta: **{fmt_num(lb,2)}** vs target **{fmt_num(beta_target,2)}** (Δ{drift_str})")
                st.write(f"- Alpha Persistence: **{fmt_num(lp,2)}**")

                if drift_flag:
                    st.warning(f"Rolling beta drift exceeds threshold: Δβ={drift_str}.")


# ============================================================
# TAB: Correlations — correlation matrix across waves (returns)
# ============================================================
with tab_corr:
    st.subheader("🧮 Correlation Matrix Across Waves")
    st.caption("Aligned daily wave returns (engine wave_ret). Use to identify redundancy and diversify multi-wave portfolios.")

    wret_mat = build_wave_returns_matrix(all_waves, mode=mode, days=min(365, max(120, history_days)))
    if wret_mat is None or wret_mat.empty:
        st.info("Not enough aligned return history to compute correlations.")
    else:
        corr = wret_mat.corr()
        plot_corr_heatmap(corr, title=f"Wave Correlation Matrix — Mode: {mode}")

        st.markdown("#### Correlation Table (tail)")
        st.dataframe(corr, use_container_width=True)

        # Simple redundancy scan: show top correlated pairs
        st.markdown("#### Highest Correlation Pairs (redundancy scan)")
        pairs = []
        cols = corr.columns.tolist()
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                v = float(corr.iloc[i, j])
                if math.isfinite(v):
                    pairs.append((cols[i], cols[j], v))
        pairs = sorted(pairs, key=lambda x: abs(x[2]), reverse=True)[:20]
        pdf = pd.DataFrame(pairs, columns=["Wave A", "Wave B", "Corr"])
        pdf["Corr"] = pdf["Corr"].apply(lambda x: fmt_num(x, 2))
        st.dataframe(pdf, use_container_width=True)


# ============================================================
# TAB: Market Intel — macro dashboard
# ============================================================
with tab_market:
    st.subheader("🌐 Market Intel")
    st.caption("Macro dashboard (daily): SPY/QQQ/IWM/TLT/GLD/BTC/VIX/TNX. Display-only.")

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
            fig.update_layout(height=420, margin=dict(l=40, r=40, t=50, b=40), title="Indexed Prices (Start=100)")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.line_chart(mk)


# ============================================================
# TAB: Factor Decomposition — regression betas
# ============================================================
with tab_factors:
    st.subheader("🧩 Factor Decomposition (Simple Regression Betas)")
    st.caption("Uses SPY/QQQ/IWM/TLT/GLD daily returns as factor proxies. Display-only (not advice).")

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
            betas = {}

            bdf = pd.DataFrame([{"Factor": k, "Beta": v} for k, v in betas.items()])
            bdf["Beta"] = bdf["Beta"].apply(lambda x: fmt_num(x, 2))
            st.dataframe(bdf, use_container_width=True)

            st.markdown("#### Factor Notes")
            st.write("- Betas are simple OLS on daily returns; they can drift materially with regime changes.")
            st.write("- Compare this panel across modes using the Mode selector in the sidebar.")


# ============================================================
# TAB: Vector OS Insight — narrative layer
# ============================================================
with tab_vector:
    st.subheader("🤖 Vector OS Insight Layer")
    st.caption("Narrative interpretation (non-advice). Flags explain what to validate in demos.")

    wd2 = wave_doctor_assess(
        selected_wave,
        mode=mode,
        days=min(365, max(120, history_days)),
        alpha_warn=alpha_warn,
        te_warn=te_warn,
        beta_drift_warn=beta_drift_warn,
    )
    attrib2 = compute_alpha_attribution(selected_wave, mode=mode, days=min(365, max(120, history_days)))

    st.markdown("### Vector Summary")
    st.write(f"**Wave:** {selected_wave}  |  **Mode:** {mode}  |  **Regime:** {reg_now}  |  **Benchmark:** {bar_src}")

    if wd2.get("ok", False):
        m = wd2["metrics"]
        st.markdown(
            f"""
- **365D Return:** {fmt_pct(m.get("Return_365D"))}  
- **365D Alpha:** {fmt_pct(m.get("Alpha_365D"))}  
- **30D Alpha:** {fmt_pct(m.get("Alpha_30D"))}  |  **30D Alpha Captured:** {fmt_pct(m.get("AlphaCaptured_30D"))}  
- **Tracking Error:** {fmt_pct(m.get("TE"))}  |  **IR:** {fmt_num(m.get("IR"), 2)}  
- **β_real vs β_target:** {fmt_num(m.get("Beta_Real"),2)} vs {fmt_num(m.get("Beta_Target"),2)} (Δ{fmt_num(m.get("Beta_Drift"),2)})  
- **Max Drawdown:** {fmt_pct(m.get("MaxDD_Wave"))} (Wave) vs {fmt_pct(m.get("MaxDD_Benchmark"))} (BM)
"""
        )
        if wd2.get("flags"):
            st.warning("Flags: " + " | ".join(wd2["flags"]))
        if wd2.get("audit", {}).get("flags"):
            st.warning("Audit: " + " | ".join(wd2["audit"]["flags"]))

    st.markdown("### Attribution Lens")
    if attrib2:
        st.write(f"- **Engine Return:** {fmt_pct(attrib2.get('Engine Return'))}")
        st.write(f"- **Static Basket Return:** {fmt_pct(attrib2.get('Static Basket Return'))}")
        st.write(f"- **Overlay Contribution:** {fmt_pct(attrib2.get('Overlay Contribution (Engine - Static)'))}")
        st.write(f"- **Alpha vs Benchmark:** {fmt_pct(attrib2.get('Alpha vs Benchmark'))}")
        st.write(f"- **Benchmark Difficulty (BM - SPY):** {fmt_pct(attrib2.get('Benchmark Difficulty (BM - SPY)'))}")
        st.write(f"- **β_real (Wave vs BM):** {fmt_num(attrib2.get('β_real (Wave vs BM)'), 2)}")
        st.write(f"- **β_target (if available):** {fmt_num(attrib2.get('β_target (if available)'), 2)}")

    st.markdown("### Vector Guidance (Non-Advice)")
    st.write(
        "Vector suggests validating benchmark stability (Benchmark Truth), then using Mode Proof + Rolling Diagnostics to ensure mode separation is real. "
        "If 30D alpha is extreme but 365D is not, review benchmark mix drift and data coverage (Audit flags). "
        "Use the Correlation Matrix to avoid redundant waves in multi-wave portfolio construction."
    )

# ============================================================
# END PART 4 / 5
# Next: PART 5 adds extra diagnostic utilities, export tools, debug panels,
# edge-case guards, plus final footer + run-safety + optional logs links.
# ============================================================
# ============================================================
# PART 5 / 5 — Exports + Deep Debug + Hardening + Footer
# ============================================================

# ============================================================
# Export helpers (CSV download) — phone friendly
# ============================================================
def _csv_bytes(df: pd.DataFrame) -> bytes:
    if df is None:
        return b""
    try:
        return df.to_csv(index=True).encode("utf-8")
    except Exception:
        try:
            return df.to_csv(index=False).encode("utf-8")
        except Exception:
            return b""


def _safe_df_name(name: str) -> str:
    s = "".join([c if c.isalnum() or c in ("_", "-", ".") else "_" for c in str(name)])
    return s[:80] if len(s) > 80 else s


# ============================================================
# Engine Introspection (diagnostics only)
# ============================================================
def engine_capabilities() -> Dict[str, Any]:
    """
    Best-effort introspection so we can show what's available without breaking the app.
    This DOES NOT change engine behavior.
    """
    caps: Dict[str, Any] = {}
    # functions
    for fn in [
        "get_all_waves",
        "get_modes",
        "compute_history_nav",
        "get_benchmark_mix_table",
        "get_wave_holdings",
        "get_benchmark_for_wave",
        "get_benchmark_mix_for_wave",
        "get_mode_exposure",
        "get_beta_target",
        "get_version",
        "engine_version",
    ]:
        caps[f"fn:{fn}"] = callable(getattr(we, fn, None))

    # attributes / maps
    for att in [
        "BENCHMARK_WEIGHTS_STATIC",
        "MODE_BASE_EXPOSURE",
        "MODE_BETA_TARGET",
        "BETA_TARGET_BY_MODE",
        "BETA_TARGETS",
        "REGIME_EXPOSURE",
        "REGIME_GATING",
        "ENGINE_VERSION",
        "__version__",
    ]:
        try:
            obj = getattr(we, att, None)
            caps[f"attr:{att}"] = True if obj is not None else False
        except Exception:
            caps[f"attr:{att}"] = False

    # common columns expectation
    caps["expected_history_cols"] = ["wave_nav", "bm_nav", "wave_ret", "bm_ret"]
    return caps


def engine_version_string() -> str:
    for att in ["ENGINE_VERSION", "__version__"]:
        try:
            v = getattr(we, att, None)
            if v:
                return str(v)
        except Exception:
            pass
    for fn in ["get_version", "engine_version"]:
        try:
            f = getattr(we, fn, None)
            if callable(f):
                v = f()
                if v:
                    return str(v)
        except Exception:
            pass
    return "unknown"


# ============================================================
# Defensive checks
# ============================================================
def assert_history_schema(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    missing: List[str] = []
    if df is None or df.empty:
        return False, ["history empty"]
    for c in ["wave_nav", "bm_nav", "wave_ret", "bm_ret"]:
        if c not in df.columns:
            missing.append(c)
    return (len(missing) == 0), missing


def safe_align_two(a: pd.Series, b: pd.Series) -> pd.DataFrame:
    """Align two series on index; drop NaNs."""
    a = safe_series(a)
    b = safe_series(b)
    df = pd.concat([a.rename("a"), b.rename("b")], axis=1).dropna()
    return df


# ============================================================
# Extra Diagnostics Panels
#   - Coverage + Nulls + continuity
#   - Benchmark mix drift (if engine provides)
#   - Export data packs
# ============================================================
def coverage_panel(hist: pd.DataFrame, label: str = ""):
    ok, missing = assert_history_schema(hist)
    if not ok:
        st.warning(f"Coverage panel: missing/empty history for {label}. Missing: {missing}")
        return

    rows = int(hist.shape[0])
    start = str(hist.index.min()) if rows else ""
    end = str(hist.index.max()) if rows else ""
    null_w = float(pd.to_numeric(hist["wave_ret"], errors="coerce").isna().mean() * 100.0) if rows else float("nan")
    null_b = float(pd.to_numeric(hist["bm_ret"], errors="coerce").isna().mean() * 100.0) if rows else float("nan")

    # continuity: how many gaps in date index (business days heuristic)
    idx = pd.to_datetime(hist.index)
    idx = idx.sort_values()
    gaps = int(((idx.to_series().diff().dt.days.fillna(1)) > 4).sum())  # crude: >4 day gap
    st.markdown("#### Coverage + Quality")
    st.write(f"- Rows: **{rows}** | Start: **{start}** | End: **{end}**")
    st.write(f"- Null % wave_ret: **{fmt_num(null_w,1)}%** | Null % bm_ret: **{fmt_num(null_b,1)}%**")
    st.write(f"- Large date gaps (>4d): **{gaps}**")


def benchmark_mix_panel(selected_wave: str):
    st.markdown("#### Benchmark Mix Details (Engine)")
    bm = get_benchmark_mix()
    if bm is None or bm.empty:
        st.info("Benchmark mix table not available from engine.")
        return

    if "Wave" in bm.columns:
        bms = bm[bm["Wave"] == selected_wave].copy()
    else:
        bms = bm.copy()

    if bms.empty:
        st.info("No benchmark mix rows found for this wave.")
        return

    # normalize weights for display only
    if "Weight" in bms.columns:
        bms["Weight"] = pd.to_numeric(bms["Weight"], errors="coerce")
        total = float(bms["Weight"].sum()) if bms["Weight"].notna().any() else float("nan")
        if math.isfinite(total) and total > 0:
            bms["Weight_norm"] = bms["Weight"] / total
        else:
            bms["Weight_norm"] = bms["Weight"]

    show_df(bms, selected_wave, key="bm_mix_deep")


def exports_panel(selected_wave: str, mode: str, days: int):
    st.markdown("### 📤 Export Data Packs (CSV)")
    st.caption("For audit trails, screenshots, external validation. Does not modify engine math.")

    hist = compute_wave_history(selected_wave, mode=mode, days=days)
    attrib = compute_alpha_attribution(selected_wave, mode=mode, days=days)

    alpha_df = build_alpha_matrix(all_waves, mode, days=min(365, days))
    corr_df = build_wave_returns_matrix(all_waves, mode, days=min(365, days))

    c1, c2, c3 = st.columns(3)
    with c1:
        st.download_button(
            "Download: Selected Wave History",
            data=_csv_bytes(hist if hist is not None else pd.DataFrame()),
            file_name=f"{_safe_df_name(selected_wave)}_{_safe_df_name(mode)}_history.csv",
            mime="text/csv",
        )
    with c2:
        adf = pd.DataFrame([{"Metric": k, "Value": v} for k, v in (attrib.items() if attrib else [])])
        st.download_button(
            "Download: Attribution Snapshot",
            data=_csv_bytes(adf),
            file_name=f"{_safe_df_name(selected_wave)}_{_safe_df_name(mode)}_attrib.csv",
            mime="text/csv",
        )
    with c3:
        st.download_button(
            "Download: Alpha Matrix (All Waves)",
            data=_csv_bytes(alpha_df if alpha_df is not None else pd.DataFrame()),
            file_name=f"alpha_matrix_{_safe_df_name(mode)}.csv",
            mime="text/csv",
        )

    st.markdown("---")

    c4, c5 = st.columns(2)
    with c4:
        st.download_button(
            "Download: Wave Returns Matrix (Aligned)",
            data=_csv_bytes(corr_df if corr_df is not None else pd.DataFrame()),
            file_name=f"wave_returns_matrix_{_safe_df_name(mode)}.csv",
            mime="text/csv",
        )
    with c5:
        corr = corr_df.corr() if corr_df is not None and not corr_df.empty else pd.DataFrame()
        st.download_button(
            "Download: Correlation Matrix",
            data=_csv_bytes(corr),
            file_name=f"correlation_matrix_{_safe_df_name(mode)}.csv",
            mime="text/csv",
        )


# ============================================================
# Deep Debug Panel (Streamlit Cloud “redacted” errors -> show safe info)
# ============================================================
def debug_panel(selected_wave: str, mode: str, history_days: int):
    st.markdown("### 🧰 Debug / Engine Diagnostics (Safe)")
    st.caption("Shows engine capabilities + schema checks + quick sanity without leaking sensitive data.")

    caps = engine_capabilities()
    ver = engine_version_string()

    st.write(f"**Engine version:** `{ver}`")
    st.markdown("#### Capability Map")
    cap_rows = [{"Key": k, "Value": str(v)} for k, v in sorted(caps.items(), key=lambda x: x[0])]
    st.dataframe(pd.DataFrame(cap_rows), use_container_width=True, height=320)

    st.markdown("---")
    st.markdown("#### Selected Wave History Schema Check")
    hist = compute_wave_history(selected_wave, mode=mode, days=min(730, max(120, history_days)))
    if hist is None or hist.empty:
        st.error("History returned empty.")
        return

    ok, missing = assert_history_schema(hist)
    st.write(f"- Rows: **{hist.shape[0]}** | Columns: **{list(hist.columns)}**")
    if ok:
        st.success("Schema OK: wave_nav / bm_nav / wave_ret / bm_ret present.")
    else:
        st.error(f"Schema missing columns: {missing}")

    st.markdown("---")
    st.markdown("#### Quick Sanity Metrics (last row)")
    try:
        last = hist.tail(1)
        st.dataframe(last, use_container_width=True)
    except Exception:
        pass

    st.markdown("---")
    st.markdown("#### Known Streamlit Cloud Redaction Fix Tips")
    st.write("- If you see `SyntaxError: from __future__ import ...` → that import is NOT at the top of the file.")
    st.write("- If you see `unterminated string literal` → a quote was opened but not closed in the code.")
    st.write("- If you see `ValueError` from rolling metrics → usually misaligned shapes; we now guard and align internally.")

    st.markdown("---")
    st.markdown("#### One-click Self-Test (safe)")
    if st.button("Run Self-Test"):
        # Minimal self-test that should never crash the app
        try:
            _ = compute_wave_history(selected_wave, mode=mode, days=120)
            _ = compute_alpha_attribution(selected_wave, mode=mode, days=120)
            _ = build_alpha_matrix(all_waves, mode=mode, days=120)
            st.success("Self-test passed: history + attribution + alpha matrix computed.")
        except Exception as e:
            st.error(f"Self-test failed: {type(e).__name__}: {e}")


# ============================================================
# Add a hidden “Diagnostics++” expander to every page bottom
# ============================================================
with st.expander("🔧 Diagnostics++ (Coverage / Benchmark / Exports / Debug)", expanded=False):
    # Coverage panel for selected wave
    hist_dbg = compute_wave_history(selected_wave, mode=mode, days=min(730, max(120, history_days)))
    if hist_dbg is not None and not hist_dbg.empty:
        coverage_panel(hist_dbg, label=f"{selected_wave} / {mode}")
    else:
        st.info("No history available for coverage panel.")

    st.markdown("---")
    benchmark_mix_panel(selected_wave)

    st.markdown("---")
    exports_panel(selected_wave, mode, days=min(365, max(120, history_days)))

    st.markdown("---")
    debug_panel(selected_wave, mode, history_days)


# ============================================================
# Footer + safety disclaimers
# ============================================================
st.markdown("---")
st.caption(
    "WAVES Intelligence™ Institutional Console — Diagnostics++ • Display-only analytics • Not investment advice. "
    "Backtest/Sandbox figures must be labeled SANDBOX; LIVE is real-money tracked. "
    "All math from engine remains unchanged; What-If Lab is shadow simulation only."
)

# ============================================================
# END PART 5 / 5
# ============================================================