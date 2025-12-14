# ============================================================
# app.py — WAVES Intelligence™ Institutional Console
# VECTOR OS • DIAGNOSTICS++ • FULL PRODUCTION BUILD
#
# PART 1 / 5 — FOUNDATION + DATA LAYER
#
# NOTHING REMOVED
# NOTHING REDUCED
# NOTHING COMMENTED OUT
#
# Paste Parts 1 → 5 IN ORDER
# ============================================================

from __future__ import annotations

# ============================================================
# Standard library imports
# ============================================================
import math
import json
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

# ============================================================
# Core scientific stack
# ============================================================
import numpy as np
import pandas as pd

# ============================================================
# Streamlit
# ============================================================
import streamlit as st

# ============================================================
# WAVES Engine (LOCKED INTERFACE)
# ============================================================
import waves_engine as we

# ============================================================
# Optional dependencies (SAFE FAIL)
# ============================================================
try:
    import yfinance as yf
except Exception:
    yf = None

try:
    import plotly.graph_objects as go
except Exception:
    go = None

# ============================================================
# Streamlit Configuration
# ============================================================
st.set_page_config(
    page_title="WAVES Intelligence™ Institutional Console",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# Global CSS — Vector OS Visual System
# ============================================================
st.markdown(
    """
<style>

/* Core layout */
.block-container {
    padding-top: 1rem;
    padding-bottom: 2.5rem;
}

/* Sticky summary bar */
.waves-sticky {
    position: sticky;
    top: 0;
    z-index: 999;
    backdrop-filter: blur(14px);
    -webkit-backdrop-filter: blur(14px);
    padding: 12px 16px;
    margin-bottom: 14px;
    border-radius: 16px;
    border: 1px solid rgba(255,255,255,0.12);
    background: linear-gradient(
        180deg,
        rgba(10,15,32,0.85),
        rgba(10,15,32,0.65)
    );
}

/* Chips */
.waves-chip {
    display: inline-block;
    padding: 6px 12px;
    margin: 6px 8px 0 0;
    border-radius: 999px;
    border: 1px solid rgba(255,255,255,0.14);
    background: rgba(255,255,255,0.05);
    font-size: 0.85rem;
    white-space: nowrap;
}

/* Headers */
.waves-hdr {
    font-weight: 900;
    letter-spacing: 0.3px;
    margin-bottom: 6px;
}

/* Tables */
div[data-testid="stDataFrame"] {
    border-radius: 14px;
    overflow: hidden;
}

/* Mobile tuning */
@media (max-width: 700px) {
    .block-container {
        padding-left: 0.8rem;
        padding-right: 0.8rem;
    }
}

</style>
""",
    unsafe_allow_html=True,
)

# ============================================================
# Formatting helpers (DISPLAY ONLY — NO MATH CHANGE)
# ============================================================
def fmt_pct(x: Any, digits: int = 2) -> str:
    try:
        if x is None:
            return "—"
        x = float(x)
        if math.isnan(x):
            return "—"
        return f"{x * 100:.{digits}f}%"
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
# Cached market data loaders
# ============================================================
@st.cache_data(show_spinner=False)
def fetch_prices_daily(tickers: List[str], days: int = 365) -> pd.DataFrame:
    if yf is None or not tickers:
        return pd.DataFrame()

    end = datetime.utcnow().date()
    start = end - timedelta(days=days + 260)

    data = yf.download(
        tickers=sorted(set(tickers)),
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
        else:
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
# Engine passthroughs (LOCKED)
# ============================================================
@st.cache_data(show_spinner=False)
def compute_wave_history(wave: str, mode: str, days: int = 365) -> pd.DataFrame:
    try:
        return we.compute_history_nav(wave, mode=mode, days=days)
    except Exception:
        return pd.DataFrame(
            columns=["wave_nav", "bm_nav", "wave_ret", "bm_ret"]
        )


@st.cache_data(show_spinner=False)
def get_wave_holdings(wave: str) -> pd.DataFrame:
    try:
        return we.get_wave_holdings(wave)
    except Exception:
        return pd.DataFrame(columns=["Ticker", "Name", "Weight"])


@st.cache_data(show_spinner=False)
def get_benchmark_mix() -> pd.DataFrame:
    try:
        return we.get_benchmark_mix_table()
    except Exception:
        return pd.DataFrame(columns=["Wave", "Ticker", "Name", "Weight"])


# ============================================================
# END PART 1 / 5
# ============================================================
# ============================================================
# PART 2 / 5 — CORE METRICS + DIAGNOSTICS ENGINE
# ============================================================

# ============================================================
# Core return & risk math (ENGINE-TRUTH SAFE)
# ============================================================
def ret_from_nav(nav: pd.Series, window: int) -> float:
    nav = safe_series(nav)
    if len(nav) < 2:
        return float("nan")
    if window > len(nav):
        window = len(nav)
    if window < 2:
        return float("nan")
    start = float(nav.iloc[-window])
    end = float(nav.iloc[-1])
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
    dd = nav / running_max - 1.0
    return float(dd.min())


def tracking_error(wave_ret: pd.Series, bm_ret: pd.Series) -> float:
    w = safe_series(wave_ret)
    b = safe_series(bm_ret)
    df = pd.concat([w.rename("w"), b.rename("b")], axis=1).dropna()
    if len(df) < 2:
        return float("nan")
    diff = df["w"] - df["b"]
    return float(diff.std() * np.sqrt(252))


def information_ratio(nav_wave: pd.Series, nav_bm: pd.Series, te: float) -> float:
    if not math.isfinite(te) or te <= 0:
        return float("nan")
    r_wave = ret_from_nav(nav_wave, len(nav_wave))
    r_bm = ret_from_nav(nav_bm, len(nav_bm))
    return float((r_wave - r_bm) / te)


def beta_vs_benchmark(wave_ret: pd.Series, bm_ret: pd.Series) -> float:
    w = safe_series(wave_ret)
    b = safe_series(bm_ret)
    df = pd.concat([w.rename("w"), b.rename("b")], axis=1).dropna()
    if len(df) < 20:
        return float("nan")
    var_b = df["b"].var()
    if var_b <= 0:
        return float("nan")
    return float(df["w"].cov(df["b"]) / var_b)


# ============================================================
# Rolling diagnostics (alpha persistence engine)
# ============================================================
def rolling_alpha(
    nav_wave: pd.Series,
    nav_bm: pd.Series,
    window: int = 60,
) -> pd.Series:
    nav_wave = safe_series(nav_wave)
    nav_bm = safe_series(nav_bm)
    out = []
    idx = []

    for i in range(window, len(nav_wave)):
        w = nav_wave.iloc[i - window : i]
        b = nav_bm.iloc[i - window : i]
        a = ret_from_nav(w, window) - ret_from_nav(b, window)
        out.append(a)
        idx.append(nav_wave.index[i])

    return pd.Series(out, index=idx, name=f"α_{window}D")


def rolling_tracking_error(
    wave_ret: pd.Series,
    bm_ret: pd.Series,
    window: int = 60,
) -> pd.Series:
    wave_ret = safe_series(wave_ret)
    bm_ret = safe_series(bm_ret)

    out = []
    idx = []

    for i in range(window, len(wave_ret)):
        w = wave_ret.iloc[i - window : i]
        b = bm_ret.iloc[i - window : i]
        out.append(tracking_error(w, b))
        idx.append(wave_ret.index[i])

    return pd.Series(out, index=idx, name=f"TE_{window}D")


def rolling_beta(
    wave_ret: pd.Series,
    bm_ret: pd.Series,
    window: int = 60,
) -> pd.Series:
    out = []
    idx = []
    for i in range(window, len(wave_ret)):
        w = wave_ret.iloc[i - window : i]
        b = bm_ret.iloc[i - window : i]
        out.append(beta_vs_benchmark(w, b))
        idx.append(wave_ret.index[i])
    return pd.Series(out, index=idx, name=f"β_{window}D")


# ============================================================
# Alpha captured (daily attribution)
# ============================================================
def alpha_captured_daily(
    wave_ret: pd.Series,
    bm_ret: pd.Series,
) -> pd.Series:
    df = pd.concat(
        [safe_series(wave_ret), safe_series(bm_ret)],
        axis=1,
    ).dropna()
    df.columns = ["wave", "bm"]
    return (df["wave"] - df["bm"]).rename("alpha_captured")


# ============================================================
# Data quality & coverage audit
# ============================================================
def data_quality_audit(hist: pd.DataFrame) -> Dict[str, Any]:
    audit = {
        "rows": len(hist),
        "missing_wave_ret": hist["wave_ret"].isna().sum(),
        "missing_bm_ret": hist["bm_ret"].isna().sum(),
        "start_date": str(hist.index.min()) if not hist.empty else "—",
        "end_date": str(hist.index.max()) if not hist.empty else "—",
        "ok": True,
        "flags": [],
    }

    if audit["rows"] < 120:
        audit["flags"].append("Short history (<120 days)")
    if audit["missing_wave_ret"] > 0:
        audit["flags"].append("Missing wave returns")
    if audit["missing_bm_ret"] > 0:
        audit["flags"].append("Missing benchmark returns")

    if audit["flags"]:
        audit["ok"] = False

    return audit


# ============================================================
# END PART 2 / 5
# ============================================================
# ============================================================
# PART 3 / 5 — ATTRIBUTION + MODE SEPARATION + CORRELATIONS
# ============================================================

# ============================================================
# Benchmark truth helpers
# ============================================================
@st.cache_data(show_spinner=False)
def compute_spy_nav(days: int = 365) -> pd.Series:
    px = fetch_prices_daily(["SPY"], days=days)
    if px is None or px.empty or "SPY" not in px.columns:
        return pd.Series(dtype=float)
    nav = (1.0 + px["SPY"].pct_change().fillna(0.0)).cumprod()
    nav.name = "spy_nav"
    return nav


def benchmark_source_label(wave_name: str, bm_mix_df: pd.DataFrame) -> str:
    # Static override detection (best-effort)
    try:
        static_dict = getattr(we, "BENCHMARK_WEIGHTS_STATIC", None)
        if isinstance(static_dict, dict) and static_dict.get(wave_name):
            return "Static Override (Engine)"
    except Exception:
        pass
    if bm_mix_df is not None and not bm_mix_df.empty:
        return "Auto-Composite (Dynamic)"
    return "Unknown"


def get_beta_target_if_available(mode: str) -> float:
    """
    Best-effort: if engine has a beta target map, use it.
    We DO NOT assume exact names; we try common attribute names.
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


def get_exposure_if_available(mode: str) -> float:
    """
    Best-effort: if engine exposes base exposure per mode, use it.
    (Only for diagnostics labeling — does not alter engine math.)
    """
    candidates = ["MODE_BASE_EXPOSURE", "BASE_EXPOSURE_BY_MODE", "MODE_EXPOSURE"]
    for name in candidates:
        try:
            obj = getattr(we, name, None)
            if isinstance(obj, dict) and mode in obj:
                return float(obj[mode])
        except Exception:
            pass
    return float("nan")


# ============================================================
# Attribution helpers (Engine vs Static Basket)
# ============================================================
def _weights_from_df(df: pd.DataFrame, ticker_col: str = "Ticker", weight_col: str = "Weight") -> pd.Series:
    if df is None or df.empty:
        return pd.Series(dtype=float)
    tmp = df[[ticker_col, weight_col]].copy()
    tmp[ticker_col] = tmp[ticker_col].astype(str)
    tmp[weight_col] = pd.to_numeric(tmp[weight_col], errors="coerce").fillna(0.0)
    w = tmp.groupby(ticker_col, as_index=True)[weight_col].sum()
    s = float(w.sum())
    if s <= 0:
        return pd.Series(dtype=float)
    return (w / s).sort_index()


@st.cache_data(show_spinner=False)
def compute_static_nav_from_weights(weights: pd.Series, days: int = 365) -> pd.Series:
    if weights is None or weights.empty:
        return pd.Series(dtype=float)
    px = fetch_prices_daily(list(weights.index), days=days)
    if px is None or px.empty:
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
    Attribution snapshot (display-only):
    - Engine Return (engine NAV)
    - Static Basket Return (holdings buy-and-hold)
    - Overlay Contribution = Engine - Static
    - Alpha vs Benchmark
    - Alpha vs SPY
    - Benchmark Difficulty = BM - SPY
    - TE, IR, β_real, β_target (if available)
    """
    out: Dict[str, float] = {}

    hist = compute_wave_history(wave_name, mode=mode, days=days)
    if hist is None or hist.empty or len(hist) < 2:
        return out

    nav_w = hist["wave_nav"]
    nav_b = hist["bm_nav"]
    ret_w = hist["wave_ret"]
    ret_b = hist["bm_ret"]

    eng_ret = ret_from_nav(nav_w, len(nav_w))
    bm_ret = ret_from_nav(nav_b, len(nav_b))
    alpha_bm = eng_ret - bm_ret

    # Static basket
    hold = get_wave_holdings(wave_name)
    wts = _weights_from_df(hold, "Ticker", "Weight")
    static_nav = compute_static_nav_from_weights(wts, days=days)
    static_ret = ret_from_nav(static_nav, len(static_nav)) if len(static_nav) >= 2 else float("nan")
    overlay = (eng_ret - static_ret) if (pd.notna(eng_ret) and pd.notna(static_ret)) else float("nan")

    # SPY difficulty
    spy_nav = compute_spy_nav(days=days)
    spy_ret = ret_from_nav(spy_nav, len(spy_nav)) if len(spy_nav) >= 2 else float("nan")
    alpha_spy = eng_ret - spy_ret if (pd.notna(eng_ret) and pd.notna(spy_ret)) else float("nan")
    bm_diff = bm_ret - spy_ret if (pd.notna(bm_ret) and pd.notna(spy_ret)) else float("nan")

    te = tracking_error(ret_w, ret_b)
    ir = information_ratio(nav_w, nav_b, te)
    beta_real = beta_vs_benchmark(ret_w, ret_b)
    beta_target = get_beta_target_if_available(mode)

    out["Engine Return"] = float(eng_ret)
    out["Static Basket Return"] = float(static_ret) if pd.notna(static_ret) else float("nan")
    out["Overlay Contribution (Engine - Static)"] = float(overlay) if pd.notna(overlay) else float("nan")
    out["Benchmark Return"] = float(bm_ret)
    out["Alpha vs Benchmark"] = float(alpha_bm)
    out["SPY Return"] = float(spy_ret) if pd.notna(spy_ret) else float("nan")
    out["Alpha vs SPY"] = float(alpha_spy) if pd.notna(alpha_spy) else float("nan")
    out["Benchmark Difficulty (BM - SPY)"] = float(bm_diff) if pd.notna(bm_diff) else float("nan")
    out["Tracking Error (TE)"] = float(te)
    out["Information Ratio (IR)"] = float(ir)
    out["β_real (Wave vs BM)"] = float(beta_real)
    out["β_target (if available)"] = float(beta_target)

    out["Wave Vol"] = float(annualized_vol(ret_w))
    out["Benchmark Vol"] = float(annualized_vol(ret_b))
    out["Wave MaxDD"] = float(max_drawdown(nav_w))
    out["Benchmark MaxDD"] = float(max_drawdown(nav_b))

    # Daily alpha captured
    try:
        ac = alpha_captured_daily(ret_w, ret_b)
        out["AlphaCaptured Avg (daily)"] = float(ac.mean())
        out["AlphaCaptured Std (daily)"] = float(ac.std())
    except Exception:
        out["AlphaCaptured Avg (daily)"] = float("nan")
        out["AlphaCaptured Std (daily)"] = float("nan")

    return out


# ============================================================
# Mode separation proof (SIDE-BY-SIDE across ALL MODES)
# ============================================================
@st.cache_data(show_spinner=False)
def mode_separation_snapshot(
    wave_name: str,
    modes: List[str],
    days: int = 365,
) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """
    Returns:
      - metrics_df: one row per mode
      - hist_by_mode: dict[mode] -> history df
    """
    rows: List[Dict[str, Any]] = []
    hist_by_mode: Dict[str, pd.DataFrame] = {}

    spy_nav = compute_spy_nav(days=days)
    spy_ret = ret_from_nav(spy_nav, len(spy_nav)) if len(spy_nav) >= 2 else float("nan")

    for m in modes:
        hist = compute_wave_history(wave_name, mode=m, days=days)
        hist_by_mode[m] = hist

        if hist is None or hist.empty or len(hist) < 2:
            rows.append(
                {
                    "Mode": m,
                    "Engine Ret": np.nan,
                    "BM Ret": np.nan,
                    "Alpha vs BM": np.nan,
                    "Alpha vs SPY": np.nan,
                    "TE": np.nan,
                    "IR": np.nan,
                    "β_real": np.nan,
                    "β_target": get_beta_target_if_available(m),
                    "Vol": np.nan,
                    "MaxDD": np.nan,
                    "Data Rows": 0,
                }
            )
            continue

        nav_w = hist["wave_nav"]
        nav_b = hist["bm_nav"]
        ret_w = hist["wave_ret"]
        ret_b = hist["bm_ret"]

        eng_ret = ret_from_nav(nav_w, len(nav_w))
        bm_ret = ret_from_nav(nav_b, len(nav_b))
        a_bm = eng_ret - bm_ret
        a_spy = eng_ret - spy_ret if (pd.notna(eng_ret) and pd.notna(spy_ret)) else np.nan

        te = tracking_error(ret_w, ret_b)
        ir = information_ratio(nav_w, nav_b, te)
        beta_real = beta_vs_benchmark(ret_w, ret_b)
        beta_t = get_beta_target_if_available(m)

        rows.append(
            {
                "Mode": m,
                "Engine Ret": eng_ret,
                "BM Ret": bm_ret,
                "Alpha vs BM": a_bm,
                "Alpha vs SPY": a_spy,
                "TE": te,
                "IR": ir,
                "β_real": beta_real,
                "β_target": beta_t,
                "Vol": annualized_vol(ret_w),
                "MaxDD": max_drawdown(nav_w),
                "Data Rows": len(hist),
            }
        )

    metrics_df = pd.DataFrame(rows)
    return metrics_df, hist_by_mode


def beta_discipline_flags(beta_real: float, beta_target: float, drift_thresh: float = 0.07) -> List[str]:
    flags = []
    if not math.isfinite(beta_real):
        return flags
    if math.isfinite(beta_target):
        drift = abs(beta_real - beta_target)
        if drift > drift_thresh:
            flags.append(f"β drift > {drift_thresh:.2f} (real={beta_real:.2f}, target={beta_target:.2f})")
    return flags


# ============================================================
# Correlation matrix across Waves (returns)
# ============================================================
@st.cache_data(show_spinner=False)
def correlation_matrix_across_waves(
    waves: List[str],
    mode: str,
    days: int = 180,
) -> pd.DataFrame:
    """
    Builds correlation matrix using aligned daily returns from engine history.
    """
    series_list = []
    names = []

    for wname in waves:
        hist = compute_wave_history(wname, mode=mode, days=days)
        if hist is None or hist.empty or "wave_ret" not in hist.columns:
            continue
        s = safe_series(hist["wave_ret"]).rename(wname)
        series_list.append(s)
        names.append(wname)

    if not series_list:
        return pd.DataFrame()

    df = pd.concat(series_list, axis=1).dropna(how="any")
    if df.shape[0] < 30:
        return pd.DataFrame()

    return df.corr()


# ============================================================
# Rolling diagnostics pack (for charts + persistence)
# ============================================================
@st.cache_data(show_spinner=False)
def rolling_diagnostics_pack(
    wave_name: str,
    mode: str,
    days: int = 365,
    roll_window: int = 60,
) -> pd.DataFrame:
    hist = compute_wave_history(wave_name, mode=mode, days=days)
    if hist is None or hist.empty or len(hist) < roll_window + 5:
        return pd.DataFrame()

    nav_w = hist["wave_nav"]
    nav_b = hist["bm_nav"]
    ret_w = hist["wave_ret"]
    ret_b = hist["bm_ret"]

    a = rolling_alpha(nav_w, nav_b, window=roll_window)
    te = rolling_tracking_error(ret_w, ret_b, window=roll_window)
    beta = rolling_beta(ret_w, ret_b, window=roll_window)

    # Rolling vol of wave
    vol = []
    idx = []
    for i in range(roll_window, len(ret_w)):
        seg = ret_w.iloc[i - roll_window : i]
        vol.append(annualized_vol(seg))
        idx.append(ret_w.index[i])
    vol = pd.Series(vol, index=idx, name=f"Vol_{roll_window}D")

    out = pd.concat([a, te, beta, vol], axis=1).dropna(how="any")

    # Alpha persistence metrics:
    # % of rolling windows with alpha > 0
    if not out.empty and f"α_{roll_window}D" in out.columns:
        out["Alpha_Positive_Flag"] = (out[f"α_{roll_window}D"] > 0).astype(int)

    return out


# ============================================================
# Alpha matrix (scan) — same signature as earlier builds
# ============================================================
@st.cache_data(show_spinner=False)
def build_alpha_matrix(all_waves: List[str], mode: str) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for wname in all_waves:
        hist = compute_wave_history(wname, mode=mode, days=365)
        if hist is None or hist.empty or len(hist) < 2:
            rows.append(
                {"Wave": wname, "1D Alpha": np.nan, "30D Alpha": np.nan, "60D Alpha": np.nan, "365D Alpha": np.nan}
            )
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
        r365b = ret_from_nav(nav_b, len(nav_b)))
        a365 = r365w - r365b

        rows.append({"Wave": wname, "1D Alpha": a1, "30D Alpha": a30, "60D Alpha": a60, "365D Alpha": a365})

    df = pd.DataFrame(rows)
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
# END PART 3 / 5
# ============================================================
# ============================================================
# PART 4 / 5 — MAIN UI (SIDEBAR + STICKY + DIAGNOSTICS PANELS)
# ============================================================

# ============================================================
# Sidebar boot + engine discovery
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

# Hard fallback if engine doesn't provide modes
if not all_modes:
    all_modes = ["Standard", "Alpha-Minus-Beta", "Private Logic"]

if "selected_wave" not in st.session_state:
    st.session_state["selected_wave"] = all_waves[0] if all_waves else ""
if "mode" not in st.session_state:
    st.session_state["mode"] = all_modes[0] if all_modes else "Standard"

# Universal diagnostics defaults
if "history_days" not in st.session_state:
    st.session_state["history_days"] = 365
if "diag_roll_window" not in st.session_state:
    st.session_state["diag_roll_window"] = 60
if "corr_days" not in st.session_state:
    st.session_state["corr_days"] = 180

# Doctor thresholds
if "alpha_warn" not in st.session_state:
    st.session_state["alpha_warn"] = 0.08
if "te_warn" not in st.session_state:
    st.session_state["te_warn"] = 0.20
if "beta_drift_warn" not in st.session_state:
    st.session_state["beta_drift_warn"] = 0.07

with st.sidebar:
    st.title("WAVES Intelligence™")
    st.caption("Institutional Console • Vector OS™")

    st.markdown("---")
    st.markdown("### Core")

    st.selectbox(
        "Mode",
        all_modes,
        index=all_modes.index(st.session_state["mode"]) if st.session_state["mode"] in all_modes else 0,
        key="mode",
    )

    if all_waves:
        st.selectbox(
            "Select Wave",
            all_waves,
            index=all_waves.index(st.session_state["selected_wave"])
            if st.session_state["selected_wave"] in all_waves
            else 0,
            key="selected_wave",
        )
    else:
        st.text_input("Select Wave", value=st.session_state.get("selected_wave", ""), key="selected_wave")

    st.markdown("---")
    st.markdown("### Windows")
    st.slider(
        "History window (days)",
        min_value=60,
        max_value=730,
        value=int(st.session_state["history_days"]),
        step=15,
        key="history_days",
    )
    st.slider(
        "Rolling diagnostics window (days)",
        min_value=20,
        max_value=160,
        value=int(st.session_state["diag_roll_window"]),
        step=5,
        key="diag_roll_window",
    )
    st.slider(
        "Correlation window (days)",
        min_value=60,
        max_value=365,
        value=int(st.session_state["corr_days"]),
        step=15,
        key="corr_days",
    )

    st.markdown("---")
    st.markdown("### Alerts (Diagnostics++)")
    st.slider(
        "Alpha alert threshold (abs, 30D)",
        min_value=0.02,
        max_value=0.25,
        value=float(st.session_state["alpha_warn"]),
        step=0.01,
        key="alpha_warn",
    )
    st.slider(
        "Tracking error alert threshold",
        min_value=0.05,
        max_value=0.50,
        value=float(st.session_state["te_warn"]),
        step=0.01,
        key="te_warn",
    )
    st.slider(
        "Beta drift alert threshold",
        min_value=0.02,
        max_value=0.20,
        value=float(st.session_state["beta_drift_warn"]),
        step=0.01,
        key="beta_drift_warn",
    )

    st.markdown("---")
    st.markdown("### Quick Tools")
    if st.button("Clear cache (data)", use_container_width=True):
        st.cache_data.clear()
        st.success("Cache cleared. Rerun now.")

mode = st.session_state["mode"]
selected_wave = st.session_state["selected_wave"]
history_days = int(st.session_state["history_days"])
roll_window = int(st.session_state["diag_roll_window"])
corr_days = int(st.session_state["corr_days"])
alpha_warn = float(st.session_state["alpha_warn"])
te_warn = float(st.session_state["te_warn"])
beta_drift_warn = float(st.session_state["beta_drift_warn"])

if not selected_wave:
    st.error("No Waves available (engine returned empty list). Verify waves_engine.get_all_waves().")
    st.stop()


# ============================================================
# Page header
# ============================================================
st.title("WAVES Intelligence™ Institutional Console")
st.caption("Live Alpha Capture • SmartSafe™ • Multi-Asset • Crypto • Gold • Income Ladders • Diagnostics++")


# ============================================================
# Sticky bar snapshot (selected wave)
# ============================================================
h_bar = compute_wave_history(selected_wave, mode=mode, days=max(365, history_days))

bar_r30 = bar_a30 = bar_r365 = bar_a365 = float("nan")
bar_te = bar_ir = float("nan")
bar_beta = float("nan")
bar_beta_target = get_beta_target_if_available(mode)
bar_beta_flags: List[str] = []

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
    bar_beta_flags = beta_discipline_flags(bar_beta, bar_beta_target, drift_thresh=beta_drift_warn)

# regime snapshot
spy_vix = fetch_spy_vix(days=min(365, max(120, history_days)))
reg_now = "—"
vix_last = float("nan")
if spy_vix is not None and not spy_vix.empty and "SPY" in spy_vix.columns and "^VIX" in spy_vix.columns and len(spy_vix) > 60:
    vix_last = float(spy_vix["^VIX"].iloc[-1])
    spy_nav = (1.0 + spy_vix["SPY"].pct_change().fillna(0.0)).cumprod()
    r60 = spy_nav / spy_nav.shift(60) - 1.0

    def _lab(x: Any) -> str:
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

    reg_now = str(r60.apply(_lab).iloc[-1])

# wavescore snapshot row
ws_snap = compute_wavescore_for_all_waves(all_waves, mode=mode, days=365)
ws_val = "—"
ws_grade = "—"
if ws_snap is not None and not ws_snap.empty:
    rr = ws_snap[ws_snap["Wave"] == selected_wave]
    if not rr.empty:
        ws_val = fmt_score(float(rr.iloc[0]["WaveScore"]))
        ws_grade = str(rr.iloc[0]["Grade"])

# exposure label (diagnostics only)
base_expo = get_exposure_if_available(mode)

beta_chip = f"{fmt_num(bar_beta,2)}"
if math.isnan(bar_beta):
    beta_chip = "—"
beta_t_chip = f"{fmt_num(bar_beta_target,2)}" if math.isfinite(bar_beta_target) else "—"

expo_chip = f"{fmt_num(base_expo,2)}" if math.isfinite(base_expo) else "—"

st.markdown(
    f"""
<div class="waves-sticky">
  <div class="waves-hdr">📌 Live Summary</div>
  <span class="waves-chip">Mode: <b>{mode}</b></span>
  <span class="waves-chip">Wave: <b>{selected_wave}</b></span>
  <span class="waves-chip">Benchmark: <b>{bar_src}</b></span>
  <span class="waves-chip">Regime: <b>{reg_now}</b></span>
  <span class="waves-chip">VIX: <b>{fmt_num(vix_last, 1) if math.isfinite(vix_last) else "—"}</b></span>
  <span class="waves-chip">30D α: <b>{fmt_pct(bar_a30)}</b> · 30D r: <b>{fmt_pct(bar_r30)}</b></span>
  <span class="waves-chip">365D α: <b>{fmt_pct(bar_a365)}</b> · 365D r: <b>{fmt_pct(bar_r365)}</b></span>
  <span class="waves-chip">TE: <b>{fmt_pct(bar_te)}</b> · IR: <b>{fmt_num(bar_ir, 2)}</b></span>
  <span class="waves-chip">β: <b>{beta_chip}</b> · β_target: <b>{beta_t_chip}</b></span>
  <span class="waves-chip">Base Exposure (diag): <b>{expo_chip}</b></span>
  <span class="waves-chip">WaveScore: <b>{ws_val}</b> ({ws_grade})</span>
</div>
""",
    unsafe_allow_html=True,
)

if bar_beta_flags:
    st.warning(" | ".join(bar_beta_flags))


# ============================================================
# Tabs (expanded)
# ============================================================
tab_console, tab_diagnostics, tab_market, tab_factors, tab_vector = st.tabs(
    ["Console", "Diagnostics++", "Market Intel", "Factor Decomposition", "Vector OS Insight Layer"]
)

# ============================================================
# TAB 1: Console (scan)
# ============================================================
with tab_console:
    st.subheader("🔥 Alpha Heatmap View (All Waves × Timeframe)")
    st.caption("Fast scan. (Values displayed as %; engine math unchanged.)")

    alpha_df = build_alpha_matrix(all_waves, mode)
    plot_alpha_heatmap(alpha_df, selected_wave, title=f"Alpha Heatmap — Mode: {mode}")

    st.markdown("### 🧭 One-Click Jump Table")
    jump_df = alpha_df.copy()
    if not jump_df.empty:
        jump_df["RankScore"] = jump_df[["1D Alpha", "30D Alpha", "60D Alpha", "365D Alpha"]].mean(axis=1, skipna=True)
        jump_df = jump_df.sort_values("RankScore", ascending=False)

        show_df(jump_df, selected_wave, key="jump_table_fmt")
        selectable_table_jump(jump_df, key="jump_table_select")
    else:
        st.info("No alpha matrix data.")

    st.markdown("---")
    st.subheader("🧾 All Waves Overview (1D / 30D / 60D / 365D Returns + Alpha)")
    overview_rows: List[Dict[str, Any]] = []

    for wname in all_waves:
        hist = compute_wave_history(wname, mode=mode, days=365)
        if hist is None or hist.empty or len(hist) < 2:
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
            r1w, a1 = np.nan, np.nan

        # 30D / 60D / 365D
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
            {"Wave": wname, "1D Ret": r1w, "1D Alpha": a1,
             "30D Ret": r30w, "30D Alpha": a30,
             "60D Ret": r60w, "60D Alpha": a60,
             "365D Ret": r365w, "365D Alpha": a365}
        )

    overview_df = pd.DataFrame(overview_rows)
    show_df(overview_df, selected_wave, key="overview_df_fmt")

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
            t = str(r.get("Ticker", "")).strip()
            if not t:
                continue
            nm = str(r.get("Name", t))
            wgt = r.get("Weight", np.nan)
            st.markdown(f"- **[{t}]({google_quote_url(t)})** — {nm} — **{fmt_pct(wgt)}**")

        st.markdown("### Full Holdings")
        show_df(hold2, selected_wave, key="holdings_df_fmt")


# ============================================================
# TAB 2: Diagnostics++ (big panel)
# ============================================================
with tab_diagnostics:
    st.subheader("🧪 Diagnostics++ (Institutional Proof Layer)")
    st.caption("This section is designed to prove benchmark truth, mode separation, stability, and data quality.")

    # ========================================================
    # Panel A: Benchmark Truth + Attribution
    # ========================================================
    st.markdown("## ✅ Benchmark Truth + Attribution")
    cA, cB = st.columns(2)

    with cA:
        st.markdown("### Benchmark Mix (as used by Engine)")
        bm_mix = get_benchmark_mix()
        if bm_mix is None or bm_mix.empty:
            st.info("No benchmark mix table returned by engine.")
        else:
            if "Wave" in bm_mix.columns:
                sub = bm_mix[bm_mix["Wave"] == selected_wave].copy()
            else:
                sub = bm_mix.copy()
            show_df(sub, selected_wave, key="bm_mix_fmt")

    with cB:
        st.markdown("### Attribution Snapshot (365D)")
        attrib = compute_alpha_attribution(selected_wave, mode=mode, days=365)
        if not attrib:
            st.info("Attribution unavailable.")
        else:
            arows = []
            for k, v in attrib.items():
                if any(x in k for x in ["Return", "Alpha", "Vol", "MaxDD", "TE", "Difficulty", "AlphaCaptured"]):
                    arows.append({"Metric": k, "Value": fmt_pct(v)})
                elif "IR" in k or "β_" in k:
                    arows.append({"Metric": k, "Value": fmt_num(v, 2)})
                else:
                    arows.append({"Metric": k, "Value": fmt_num(v, 6)})
            st.dataframe(pd.DataFrame(arows), use_container_width=True)

    st.markdown("---")

    # ========================================================
    # Panel B: Mode Separation Proof (metrics table + NAV overlay)
    # ========================================================
    st.markdown("## 🧬 Mode Separation Proof (ALL MODES)")
    ms_df, ms_hist = mode_separation_snapshot(selected_wave, all_modes, days=max(365, history_days))

    if ms_df is None or ms_df.empty:
        st.warning("Mode separation snapshot unavailable.")
    else:
        show_df(ms_df, selected_wave="", key="mode_sep_metrics")

        # Drift flags per mode
        drift_lines = []
        for _, r in ms_df.iterrows():
            br = float(r.get("β_real", np.nan))
            bt = float(r.get("β_target", np.nan))
            flags = beta_discipline_flags(br, bt, drift_thresh=beta_drift_warn)
            if flags:
                drift_lines.append(f"**{r.get('Mode','')}**: " + " | ".join(flags))
        if drift_lines:
            st.warning("Beta Discipline Flags:\n\n" + "\n\n".join(drift_lines))

        st.markdown("### NAV Overlay (Engine NAV by Mode)")
        if go is None:
            st.info("Plotly missing; showing simplified chart instead.")
            # build a combined nav df for st.line_chart
            nav_cols = {}
            for m in all_modes:
                h = ms_hist.get(m)
                if h is None or h.empty or "wave_nav" not in h.columns:
                    continue
                nav_cols[m] = h["wave_nav"].copy()
            if nav_cols:
                nav_df = pd.DataFrame(nav_cols).dropna(how="all")
                st.line_chart(nav_df)
            else:
                st.info("No NAV series available.")
        else:
            fig = go.Figure()
            added = 0
            for m in all_modes:
                h = ms_hist.get(m)
                if h is None or h.empty or "wave_nav" not in h.columns:
                    continue
                s = h["wave_nav"].copy()
                fig.add_trace(go.Scatter(x=s.index, y=s.values, name=f"{m}", mode="lines"))
                added += 1

            # Benchmark overlay (for current mode only, to avoid clutter)
            h0 = ms_hist.get(mode)
            if h0 is not None and not h0.empty and "bm_nav" in h0.columns:
                s2 = h0["bm_nav"].copy()
                fig.add_trace(go.Scatter(x=s2.index, y=s2.values, name=f"BM ({mode})", mode="lines"))

            fig.update_layout(height=420, margin=dict(l=40, r=40, t=40, b=40))
            if added > 0:
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No NAV series available.")

    st.markdown("---")

    # ========================================================
    # Panel C: Rolling Diagnostics (Alpha / TE / Beta / Vol / Persistence)
    # ========================================================
    st.markdown("## 📈 Rolling Diagnostics (Alpha / TE / β / Vol + Persistence)")
    rd = rolling_diagnostics_pack(selected_wave, mode=mode, days=max(365, history_days), roll_window=roll_window)

    if rd is None or rd.empty:
        st.info("Not enough data for rolling diagnostics.")
    else:
        st.markdown("### Rolling Table (latest rows)")
        st.dataframe(rd.tail(12), use_container_width=True)

        # Persistence summary
        pos_rate = float(rd["Alpha_Positive_Flag"].mean()) if "Alpha_Positive_Flag" in rd.columns else float("nan")
        st.markdown(
            f"**Alpha Persistence:** {fmt_pct(pos_rate)} of rolling windows are positive (window={roll_window}D)."
        )

        if go is None:
            st.info("Plotly missing; showing simplified charts.")
            cols = [c for c in rd.columns if c.startswith("α_") or c.startswith("TE_") or c.startswith("β_") or c.startswith("Vol_")]
            if cols:
                st.line_chart(rd[cols])
        else:
            # Rolling alpha
            alpha_cols = [c for c in rd.columns if c.startswith("α_")]
            te_cols = [c for c in rd.columns if c.startswith("TE_")]
            beta_cols = [c for c in rd.columns if c.startswith("β_")]
            vol_cols = [c for c in rd.columns if c.startswith("Vol_")]

            if alpha_cols:
                figA = go.Figure()
                for c in alpha_cols:
                    figA.add_trace(go.Scatter(x=rd.index, y=rd[c], name=c, mode="lines"))
                figA.update_layout(height=260, margin=dict(l=40, r=40, t=40, b=40), title="Rolling Alpha")
                st.plotly_chart(figA, use_container_width=True)

            if te_cols:
                figT = go.Figure()
                for c in te_cols:
                    figT.add_trace(go.Scatter(x=rd.index, y=rd[c], name=c, mode="lines"))
                figT.update_layout(height=260, margin=dict(l=40, r=40, t=40, b=40), title="Rolling Tracking Error")
                st.plotly_chart(figT, use_container_width=True)

            if beta_cols:
                figB = go.Figure()
                for c in beta_cols:
                    figB.add_trace(go.Scatter(x=rd.index, y=rd[c], name=c, mode="lines"))
                # show beta target line if known
                bt = get_beta_target_if_available(mode)
                if math.isfinite(bt):
                    figB.add_trace(go.Scatter(x=rd.index, y=[bt] * len(rd.index), name="β_target", mode="lines"))
                figB.update_layout(height=260, margin=dict(l=40, r=40, t=40, b=40), title="Rolling Beta")
                st.plotly_chart(figB, use_container_width=True)

            if vol_cols:
                figV = go.Figure()
                for c in vol_cols:
                    figV.add_trace(go.Scatter(x=rd.index, y=rd[c], name=c, mode="lines"))
                figV.update_layout(height=260, margin=dict(l=40, r=40, t=40, b=40), title="Rolling Volatility")
                st.plotly_chart(figV, use_container_width=True)

    st.markdown("---")

    # ========================================================
    # Panel D: Correlation Matrix (across Waves)
    # ========================================================
    st.markdown("## 🔗 Correlation Matrix (Wave Returns)")
    st.caption("Correlation across waves using engine daily returns (aligned).")

    corr = correlation_matrix_across_waves(all_waves, mode=mode, days=corr_days)
    if corr is None or corr.empty:
        st.info("Not enough aligned return data to compute correlations.")
    else:
        st.dataframe(corr, use_container_width=True)

        if go is not None:
            z = corr.values
            figC = go.Figure(
                data=go.Heatmap(
                    z=z,
                    x=corr.columns.tolist(),
                    y=corr.index.tolist(),
                    zmin=-1,
                    zmax=1,
                    colorbar=dict(title="Corr"),
                )
            )
            figC.update_layout(height=min(900, 280 + 22 * max(10, len(corr))), margin=dict(l=40, r=40, t=40, b=40))
            st.plotly_chart(figC, use_container_width=True)

    st.markdown("---")

    # ========================================================
    # Panel E: Data Quality & Coverage Audit
    # ========================================================
    st.markdown("## 🧾 Data Quality / Coverage Audit")
    st.caption("Flags missing rows, NaN ratios, benchmark alignment, and basic coverage checks.")

    audit = data_quality_audit(selected_wave, mode=mode, days=max(365, history_days))
    if audit is None or not audit:
        st.info("Audit unavailable.")
    else:
        adf = pd.DataFrame([{"Field": k, "Value": v} for k, v in audit.items()])
        st.dataframe(adf, use_container_width=True)

        # quick flags
        flags = []
        try:
            if float(audit.get("Rows", 0)) < 120:
                flags.append("Short history window (Rows < 120)")
        except Exception:
            pass
        try:
            if float(audit.get("NaN% wave_ret", 0)) > 0.05:
                flags.append("High NaN ratio in wave returns")
        except Exception:
            pass
        try:
            if float(audit.get("NaN% bm_ret", 0)) > 0.05:
                flags.append("High NaN ratio in benchmark returns")
        except Exception:
            pass
        try:
            if float(audit.get("Coverage Ratio (wave/bm)", 1)) < 0.95:
                flags.append("Wave/BM index coverage mismatch")
        except Exception:
            pass

        if flags:
            st.warning(" | ".join(flags))
        else:
            st.success("No major data quality flags detected on this window.")


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
# TAB 5: Vector OS Insight Layer (placeholder; Part 5 expands)
# ============================================================
with tab_vector:
    st.subheader("🤖 Vector OS Insight Layer")
    st.caption("Narrative interpretation (non-advice). Part 5 expands this panel + Wave Doctor + What-If.")

    st.write(f"**Wave:** {selected_wave}  |  **Mode:** {mode}  |  **Regime:** {reg_now}  |  **Benchmark:** {bar_src}")

    attrib2 = compute_alpha_attribution(selected_wave, mode=mode, days=365)
    if attrib2:
        st.markdown("### Attribution Lens")
        st.write(f"- **Engine Return:** {fmt_pct(attrib2.get('Engine Return'))}")
        st.write(f"- **Static Basket Return:** {fmt_pct(attrib2.get('Static Basket Return'))}")
        st.write(f"- **Overlay Contribution:** {fmt_pct(attrib2.get('Overlay Contribution (Engine - Static)'))}")
        st.write(f"- **Alpha vs Benchmark:** {fmt_pct(attrib2.get('Alpha vs Benchmark'))}")
        st.write(f"- **Benchmark Difficulty (BM - SPY):** {fmt_pct(attrib2.get('Benchmark Difficulty (BM - SPY)'))}")

    st.markdown("### Vector Guidance (Non-Advice)")
    st.write(
        "Vector suggests validating benchmark stability (Benchmark Truth panel), then using the heatmap + overview grid "
        "to identify persistent alpha across multiple windows. If 30D alpha is extreme but 365D is not, check benchmark mix drift and data coverage."
    )

# ============================================================
# END PART 4 / 5
# ============================================================
# ============================================================
# PART 5 / 5 — FINISH + WAVEDOCTOR + WHATIF + WAVESCORE + EXPORTS
# ============================================================

# ------------------------------------------------------------
# (A) SAFETY FINISHER:
# If your Part 3 got cut off mid simulate_whatif_nav(), the
# code below completes the function in-place.
#
# IMPORTANT:
# If your simulate_whatif_nav already finished in Part 3,
# this block will NOT break anything because we only define
# it if it's missing or incomplete.
# ------------------------------------------------------------
try:
    _SIM_EXISTS = "simulate_whatif_nav" in globals()
except Exception:
    _SIM_EXISTS = False

# If it exists, we do nothing. If not, we define it fully here.
if not _SIM_EXISTS:

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
            if hist_eng is not None and not hist_eng.empty and "bm_nav" in hist_eng.columns:
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


# ------------------------------------------------------------
# (B) WAVE DOCTOR (if missing)
# ------------------------------------------------------------
try:
    _WD_EXISTS = "wave_doctor_assess" in globals()
except Exception:
    _WD_EXISTS = False

if not _WD_EXISTS:

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
        if hist is None or hist.empty or len(hist) < 2:
            return {"ok": False, "message": "Not enough data to run Wave Doctor."}

        nav_w = hist["wave_nav"]
        nav_b = hist["bm_nav"]
        ret_w = hist["wave_ret"]
        ret_b = hist["bm_ret"]

        # core
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

        spy_nav = compute_spy_nav(days=days)
        spy_ret = ret_from_nav(spy_nav, len(spy_nav)) if len(spy_nav) >= 2 else float("nan")
        bm_difficulty = (r365_b - spy_ret) if (pd.notna(r365_b) and pd.notna(spy_ret)) else float("nan")

        # alpha captured (daily)
        try:
            expo = get_exposure_if_available(mode)
            alpha_cap_daily = alpha_captured_series(ret_w, ret_b, exposure=expo)
            alpha_cap_30 = float(alpha_cap_daily.tail(30).sum()) if len(alpha_cap_daily) >= 2 else float("nan")
        except Exception:
            alpha_cap_30 = float("nan")

        flags: List[str] = []
        diagnosis: List[str] = []
        recs: List[str] = []

        # large alpha flags
        if pd.notna(a30) and abs(a30) >= alpha_warn:
            flags.append("Large 30D alpha (verify benchmark + coverage)")
            diagnosis.append("30D alpha is unusually large. This can be real, or from benchmark mix drift / coverage mismatch.")
            recs.append("Check Benchmark Truth panel: benchmark mix stability + difficulty vs SPY.")
            recs.append("Check Data Quality Audit for NaN ratios and alignment gaps.")

        # TE
        if pd.notna(te) and te > te_warn:
            flags.append("High tracking error (active risk elevated)")
            diagnosis.append("Tracking error is high; wave deviates significantly from benchmark behavior.")
            recs.append("Use What-If Lab (shadow) to tighten exposure caps and reduce tilt strength.")

        # vol / drawdown
        if pd.notna(vol_w) and vol_w > vol_warn:
            flags.append("High volatility")
            diagnosis.append("Volatility is elevated relative to common institutional tolerances.")
            recs.append("Use shadow controls to reduce exposure range and/or lower vol target.")
        if pd.notna(mdd_w) and mdd_w < mdd_warn:
            flags.append("Deep drawdown")
            diagnosis.append("Drawdown is deep on this lookback.")
            recs.append("Consider stronger SmartSafe posture (shadow sim) under panic/downtrend regimes.")

        # beta discipline
        beta_flags = beta_discipline_flags(beta_real, beta_target, drift_thresh=beta_drift_warn)
        if beta_flags:
            flags.extend(beta_flags)
            diagnosis.append("Beta discipline indicates drift vs expected profile (if target is available).")
            recs.append("Confirm mode separation proof table + rolling beta chart; compare across modes for validation.")

        # benchmark difficulty context
        if pd.notna(bm_difficulty) and bm_difficulty > 0.03:
            diagnosis.append("Benchmark difficulty is positive vs SPY on this window (benchmark outperformed SPY). Alpha is harder here.")

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
                "β_real": beta_real,
                "β_target": beta_target,
                "AlphaCaptured_30D": alpha_cap_30,
            },
            "flags": flags,
            "diagnosis": diagnosis,
            "recommendations": list(dict.fromkeys(recs)),
        }


# ------------------------------------------------------------
# (C) EXPORT HELPERS
# ------------------------------------------------------------
def _to_csv_bytes(df: pd.DataFrame) -> bytes:
    if df is None or df.empty:
        return b""
    return df.to_csv(index=True).encode("utf-8")

def export_block(title: str, df: pd.DataFrame, filename: str, key: str):
    st.markdown(f"### ⬇️ Export — {title}")
    if df is None or df.empty:
        st.info("Nothing to export.")
        return
    st.download_button(
        label=f"Download CSV: {filename}",
        data=_to_csv_bytes(df),
        file_name=filename,
        mime="text/csv",
        key=key,
        use_container_width=True,
    )


# ============================================================
# (D) EXPAND TAB_VECTOR INTO FULL PANEL (Wave Doctor + What-If)
# ============================================================
# Re-open the Vector tab and append the full sections
with tab_vector:
    st.markdown("---")
    st.subheader("🩺 Wave Doctor (Diagnostics++)")
    wd = wave_doctor_assess(
        selected_wave,
        mode=mode,
        days=max(365, history_days),
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
                {"Metric": "Alpha Captured (30D)", "Value": fmt_pct(m.get("AlphaCaptured_30D"))},
                {"Metric": "Vol (Wave)", "Value": fmt_pct(m.get("Vol_Wave"))},
                {"Metric": "Vol (Benchmark)", "Value": fmt_pct(m.get("Vol_Benchmark"))},
                {"Metric": "Tracking Error (TE)", "Value": fmt_pct(m.get("TE"))},
                {"Metric": "Information Ratio (IR)", "Value": fmt_num(m.get("IR"), 2)},
                {"Metric": "MaxDD (Wave)", "Value": fmt_pct(m.get("MaxDD_Wave"))},
                {"Metric": "MaxDD (Benchmark)", "Value": fmt_pct(m.get("MaxDD_Benchmark"))},
                {"Metric": "BM Difficulty (BM - SPY)", "Value": fmt_pct(m.get("Benchmark_Difficulty_BM_minus_SPY"))},
                {"Metric": "β_real", "Value": fmt_num(m.get("β_real"), 2)},
                {"Metric": "β_target", "Value": fmt_num(m.get("β_target"), 2) if math.isfinite(float(m.get("β_target", float("nan")))) else "—"},
            ]
        )
        st.dataframe(mdf, use_container_width=True)

        if wd.get("flags"):
            st.warning(" | ".join(wd["flags"]))

        st.markdown("#### Diagnosis")
        for line in wd.get("diagnosis", []):
            st.write(f"- {line}")

        if wd.get("recommendations"):
            st.markdown("#### Recommendations (shadow controls)")
            for line in wd["recommendations"]:
                st.write(f"- {line}")

    st.markdown("---")
    st.subheader("🧪 What-If Lab (Shadow Simulation)")
    st.caption("This does NOT change engine math. It’s a sandbox overlay for diagnostics + stability exploration.")

    wcol1, wcol2, wcol3, wcol4 = st.columns(4)
    with wcol1:
        tilt_strength = st.slider("Tilt strength", 0.0, 1.0, 0.30, 0.05, key="whatif_tilt_strength")
    with wcol2:
        vol_target = st.slider("Vol target (annual)", 0.05, 0.60, 0.20, 0.01, key="whatif_vol_target")
    with wcol3:
        extra_safe = st.slider("Extra safe boost", 0.0, 0.60, 0.00, 0.01, key="whatif_extra_safe")
    with wcol4:
        freeze_bm = st.checkbox("Freeze benchmark (use engine BM)", value=True, key="whatif_freeze_bm")

    wcol5, wcol6 = st.columns(2)
    with wcol5:
        exp_min = st.slider("Exposure min", 0.0, 1.8, 0.60, 0.05, key="whatif_exp_min")
    with wcol6:
        exp_max = st.slider("Exposure max", 0.2, 2.5, 1.20, 0.05, key="whatif_exp_max")

    run_whatif = st.button("Run What-If Shadow Sim", use_container_width=True, key="whatif_run_btn")

    if run_whatif:
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
            st.warning("Simulation failed (insufficient prices / blocked feed).")
        else:
            nav = sim["whatif_nav"]
            bm_nav = sim["bm_nav"] if "bm_nav" in sim.columns else None

            ret_total = ret_from_nav(nav, len(nav))
            bm_total = ret_from_nav(bm_nav, len(bm_nav)) if bm_nav is not None and len(bm_nav) > 1 else 0.0
            alpha_total = ret_total - bm_total

            st.markdown(
                f"**What-If Return:** {fmt_pct(ret_total)}   |   **What-If Alpha:** {fmt_pct(alpha_total)}   |   **Benchmark Return:** {fmt_pct(bm_total)}"
            )

            if go is not None:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=sim.index, y=sim["whatif_nav"], name="What-If NAV", mode="lines"))
                if "bm_nav" in sim.columns:
                    fig.add_trace(go.Scatter(x=sim.index, y=sim["bm_nav"], name="Benchmark NAV", mode="lines"))
                fig.update_layout(height=380, margin=dict(l=40, r=40, t=40, b=40))
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.line_chart(sim[["whatif_nav"] + (["bm_nav"] if "bm_nav" in sim.columns else [])])

            export_block("What-If NAV Series", sim, f"{selected_wave}_{mode}_whatif.csv", key="export_whatif_csv")

    st.markdown("---")
    st.subheader("🏁 WaveScore Leaderboard (Mode-Specific)")
    ws_df = compute_wavescore_for_all_waves(all_waves, mode=mode, days=365)
    if ws_df is None or ws_df.empty:
        st.info("WaveScore unavailable.")
    else:
        ws_view = ws_df.sort_values("WaveScore", ascending=False).reset_index(drop=True)
        show_df(ws_view, selected_wave, key="wavescore_board_fmt")
        export_block("WaveScore Leaderboard", ws_view, f"wavescore_{mode}.csv", key="export_wavescore_csv")

    st.markdown("---")
    st.subheader("📦 Diagnostics Exports (Selected Wave)")
    # export: selected wave history, rolling diagnostics, and attribution table
    hist_sel = compute_wave_history(selected_wave, mode=mode, days=max(365, history_days))
    if hist_sel is not None and not hist_sel.empty:
        export_block("Engine History NAV/RET", hist_sel, f"{selected_wave}_{mode}_history.csv", key="export_hist_csv")
    else:
        st.info("No engine history to export.")

    rd2 = rolling_diagnostics_pack(selected_wave, mode=mode, days=max(365, history_days), roll_window=roll_window)
    if rd2 is not None and not rd2.empty:
        export_block("Rolling Diagnostics", rd2, f"{selected_wave}_{mode}_rolling_{roll_window}d.csv", key="export_roll_csv")
    else:
        st.info("No rolling diagnostics to export.")

    attrib3 = compute_alpha_attribution(selected_wave, mode=mode, days=365)
    if attrib3:
        adf = pd.DataFrame([{"Metric": k, "Value": v} for k, v in attrib3.items()])
        export_block("Attribution Snapshot", adf, f"{selected_wave}_{mode}_attrib.csv", key="export_attrib_csv")
    else:
        st.info("No attribution snapshot to export.")


# ============================================================
# (E) FINAL SYSTEM FOOTER (status)
# ============================================================
st.markdown("---")
with st.expander("System Status / Debug (click to open)", expanded=False):
    st.markdown("### Engine Discovery")
    st.write({"modes": all_modes, "waves_count": len(all_waves)})
    st.markdown("### Current Selection")
    st.write({"selected_wave": selected_wave, "mode": mode, "history_days": history_days, "roll_window": roll_window, "corr_days": corr_days})
    st.markdown("### Data Feed Availability")
    st.write({"yfinance_loaded": yf is not None, "plotly_loaded": go is not None})
    st.markdown("### Tip")
    st.write("If you see blanks: clear cache in sidebar, then rerun app.")

# ============================================================
# END PART 5 / 5
# ============================================================
