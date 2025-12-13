# waves_engine.py — WAVES Intelligence™ Engine (Stable Core + Alpha Attribution)
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

try:
    import yfinance as yf
except Exception:
    yf = None


# ----------------------------
# Modes
# ----------------------------

MODES: List[str] = ["Standard", "Alpha-Minus-Beta", "Private Logic"]

# Base risk exposure targets (simple + stable)
MODE_BASE_EXPOSURE: Dict[str, float] = {
    "Standard": 1.00,
    "Alpha-Minus-Beta": 0.85,
    "Private Logic": 1.15,
}

# Cash / safe sleeve candidates (SmartSafe proxy)
SAFE_TICKER = "SGOV"
VIX_TICKER = "^VIX"
BASE_INDEX = "SPY"


# ----------------------------
# Data: Wave weights + Benchmarks
# ----------------------------

# You can keep using your existing wave_weights.csv if you already have it.
# Format supported:
#   Wave, Ticker, Weight
# or
#   wave, ticker, weight
WAVE_WEIGHTS_PATH = "wave_weights.csv"

# If you don’t have a CSV available for a Wave, you can hardcode it here:
FALLBACK_WAVE_WEIGHTS: Dict[str, Dict[str, float]] = {
    # Example: "Demas Fund Wave": {"BRK-B": 0.12, "VTV": 0.12, "JPM": 0.10, ...}
}

# Optional: composite benchmark mix per wave (simple placeholder; you likely already have this logic)
# If a wave is not listed, benchmark defaults to SPY 100%.
BENCHMARK_MIX: Dict[str, Dict[str, float]] = {
    # "Demas Fund Wave": {"SPY": 0.60, "VTV": 0.40},
}


def _normalize_weights(w: Dict[str, float]) -> Dict[str, float]:
    w = {k: float(v) for k, v in w.items() if pd.notna(v) and float(v) > 0}
    s = sum(w.values())
    if s <= 0:
        return {}
    return {k: v / s for k, v in w.items()}


def _load_wave_weights_csv(path: str) -> Dict[str, Dict[str, float]]:
    try:
        df = pd.read_csv(path)
    except Exception:
        return {}

    cols = {c.lower(): c for c in df.columns}
    wave_c = cols.get("wave")
    ticker_c = cols.get("ticker")
    weight_c = cols.get("weight")

    if not (wave_c and ticker_c and weight_c):
        return {}

    out: Dict[str, Dict[str, float]] = {}
    for _, r in df.iterrows():
        wv = str(r[wave_c]).strip()
        tk = str(r[ticker_c]).strip()
        try:
            wt = float(r[weight_c])
        except Exception:
            continue
        if not wv or not tk or not math.isfinite(wt) or wt <= 0:
            continue
        out.setdefault(wv, {})
        out[wv][tk] = out[wv].get(tk, 0.0) + wt

    out = {k: _normalize_weights(v) for k, v in out.items() if v}
    return out


# Load weights safely (never crash import)
_WAVE_WEIGHTS: Dict[str, Dict[str, float]] = {}
try:
    _WAVE_WEIGHTS = _load_wave_weights_csv(WAVE_WEIGHTS_PATH)
except Exception:
    _WAVE_WEIGHTS = {}

# Merge fallbacks (hardcoded wins if exists)
for wname, wmap in FALLBACK_WAVE_WEIGHTS.items():
    _WAVE_WEIGHTS[wname] = _normalize_weights(wmap)


# ----------------------------
# Public API (called by app.py)
# ----------------------------

def get_modes() -> List[str]:
    return list(MODES)


def get_all_waves() -> List[str]:
    # Always return a list, never raise
    waves = sorted([w for w in _WAVE_WEIGHTS.keys() if isinstance(w, str) and w.strip()])
    return waves


def get_wave_holdings(wave_name: str) -> pd.DataFrame:
    w = _WAVE_WEIGHTS.get(wave_name, {})
    if not w:
        return pd.DataFrame(columns=["Ticker", "Name", "Weight"])

    df = pd.DataFrame(
        [{"Ticker": t, "Name": t, "Weight": float(wt)} for t, wt in w.items()]
    ).sort_values("Weight", ascending=False)

    return df.reset_index(drop=True)


def get_benchmark_mix_table() -> pd.DataFrame:
    rows = []
    for w in get_all_waves():
        mix = BENCHMARK_MIX.get(w, {BASE_INDEX: 1.0})
        mix = _normalize_weights(mix)
        for t, wt in mix.items():
            rows.append({"Wave": w, "Ticker": t, "Name": t, "Weight": float(wt)})
    if not rows:
        return pd.DataFrame(columns=["Wave", "Ticker", "Name", "Weight"])
    return pd.DataFrame(rows)


def _download_history(tickers: List[str], start: datetime, end: datetime) -> pd.DataFrame:
    if yf is None:
        return pd.DataFrame()

    data = yf.download(
        tickers=tickers,
        start=start.date().isoformat(),
        end=end.date().isoformat(),
        interval="1d",
        auto_adjust=True,
        progress=False,
        group_by="column",
    )

    if isinstance(data.columns, pd.MultiIndex):
        # Prefer Adj Close or Close
        lvl0 = data.columns.get_level_values(0)
        if "Adj Close" in lvl0:
            data = data["Adj Close"]
        elif "Close" in lvl0:
            data = data["Close"]
        else:
            data = data[data.columns.levels[0][0]]

    if isinstance(data.columns, pd.MultiIndex):
        data = data.droplevel(0, axis=1)

    if isinstance(data, pd.Series):
        data = data.to_frame()

    data = data.sort_index().ffill().bfill()
    return data


def _compute_nav_from_weights(price_df: pd.DataFrame, weights: Dict[str, float]) -> pd.Series:
    tickers = [t for t in weights.keys() if t in price_df.columns]
    if not tickers:
        return pd.Series(dtype=float)

    w = np.array([weights[t] for t in tickers], dtype=float)
    px = price_df[tickers].copy()
    # normalize base to 1
    px0 = px.iloc[0].replace(0, np.nan)
    rel = px.divide(px0).fillna(method="ffill").fillna(method="bfill")
    nav = (rel.values @ w)
    return pd.Series(nav, index=px.index, name="nav").astype(float)


def _regime_from_return(r60: float) -> str:
    # simple & stable label; app.py calls this sometimes
    if not math.isfinite(r60):
        return "neutral"
    if r60 > 0.05:
        return "risk-on"
    if r60 < -0.05:
        return "risk-off"
    return "neutral"


def compute_static_basket_nav(wave_name: str, days: int = 365) -> pd.DataFrame:
    """
    Static basket = fixed target weights, no overlay.
    Returns df with columns: static_nav, static_ret
    """
    weights = _WAVE_WEIGHTS.get(wave_name, {})
    if not weights:
        return pd.DataFrame(columns=["static_nav", "static_ret"])

    end = datetime.utcnow()
    start = end - timedelta(days=days + 10)

    tickers = sorted(set(list(weights.keys())))
    px = _download_history(tickers, start, end)
    if px.empty:
        return pd.DataFrame(columns=["static_nav", "static_ret"])

    if len(px) > days:
        px = px.iloc[-days:]

    nav = _compute_nav_from_weights(px, weights)
    if nav.empty:
        return pd.DataFrame(columns=["static_nav", "static_ret"])

    ret = nav.pct_change().fillna(0.0)
    out = pd.DataFrame({"static_nav": nav, "static_ret": ret})
    return out


def _benchmark_weights_for_wave(wave_name: str) -> Dict[str, float]:
    mix = BENCHMARK_MIX.get(wave_name, {BASE_INDEX: 1.0})
    return _normalize_weights(mix)


def compute_history_nav(wave_name: str, mode: str = "Standard", days: int = 365) -> pd.DataFrame:
    """
    Engine NAV (dynamic overlay) vs composite benchmark NAV.
    Returns df with columns:
      wave_nav, bm_nav, wave_ret, bm_ret
    """
    weights = _WAVE_WEIGHTS.get(wave_name, {})
    if not weights or mode not in MODE_BASE_EXPOSURE:
        return pd.DataFrame(columns=["wave_nav", "bm_nav", "wave_ret", "bm_ret"])

    end = datetime.utcnow()
    start = end - timedelta(days=days + 10)

    bm_w = _benchmark_weights_for_wave(wave_name)

    # Price universe includes wave, benchmark, safe sleeve, VIX + SPY for regime
    tickers = set(weights.keys()) | set(bm_w.keys()) | {SAFE_TICKER, VIX_TICKER, BASE_INDEX}
    px = _download_history(sorted(tickers), start, end)
    if px.empty:
        return pd.DataFrame(columns=["wave_nav", "bm_nav", "wave_ret", "bm_ret"])

    if len(px) > days:
        px = px.iloc[-days:]

    # Build static base NAVs
    base_nav = _compute_nav_from_weights(px, weights)
    bm_nav = _compute_nav_from_weights(px, bm_w)

    if base_nav.empty or bm_nav.empty:
        return pd.DataFrame(columns=["wave_nav", "bm_nav", "wave_ret", "bm_ret"])

    # Overlay: a simple exposure scaler using VIX level + mode base exposure
    # (stable, won’t crash; you can make it smarter later)
    vix = px[VIX_TICKER].copy() if VIX_TICKER in px.columns else pd.Series(index=px.index, data=np.nan)
    spy = px[BASE_INDEX].copy() if BASE_INDEX in px.columns else pd.Series(index=px.index, data=np.nan)

    # 60D SPY trend for regime
    spy_ret = spy.pct_change().fillna(0.0)
    r60 = (spy / spy.shift(60) - 1.0).fillna(0.0)
    regime = r60.apply(lambda x: x)  # series

    base_expo = MODE_BASE_EXPOSURE.get(mode, 1.0)

    # VIX scaling: higher VIX -> lower exposure (down to 55% of base)
    # lower VIX -> allow full base (or slightly above for Private Logic)
    vix_clean = vix.ffill().bfill()
    vix_level = vix_clean.clip(lower=10, upper=60)

    vix_scale = 1.0 - ((vix_level - 15.0) / 45.0) * 0.45  # at 60 => -45%
    vix_scale = vix_scale.clip(lower=0.55, upper=1.05)

    # If SPY 60D is negative, shave a little more
    trend_scale = 1.0 - (regime < 0).astype(float) * 0.10

    exposure = (base_expo * vix_scale * trend_scale).clip(lower=0.40, upper=1.25)

    # Create engine NAV by mixing: (exposure * base_nav) + ((1-exposure) * safe_nav)
    safe_nav = None
    if SAFE_TICKER in px.columns:
        safe_nav = (px[SAFE_TICKER] / px[SAFE_TICKER].iloc[0]).astype(float)
    else:
        # fallback: cash-like flat NAV
        safe_nav = pd.Series(index=px.index, data=1.0, dtype=float)

    wave_nav = (base_nav.values * exposure.values) + (safe_nav.values * (1.0 - exposure.values))
    wave_nav = pd.Series(wave_nav, index=px.index, name="wave_nav").astype(float)

    # normalize both to start at 1
    wave_nav = wave_nav / wave_nav.iloc[0]
    bm_nav = bm_nav / bm_nav.iloc[0]

    wave_ret = wave_nav.pct_change().fillna(0.0)
    bm_ret = bm_nav.pct_change().fillna(0.0)

    out = pd.DataFrame(
        {
            "wave_nav": wave_nav,
            "bm_nav": bm_nav,
            "wave_ret": wave_ret,
            "bm_ret": bm_ret,
        }
    )
    return out


def compute_alpha_attribution_summary(wave_name: str, mode: str = "Standard", days: int = 365) -> dict:
    """
    Simple, practical attribution:
      Engine Return (dynamic overlay engine NAV)
      Static Basket Return (fixed holdings NAV)
      Overlay Contribution = Engine - Static
      Alpha vs Benchmark
      Benchmark Return
      SPY Return + BM difficulty
      IR
      Risk stats
    """
    hist = compute_history_nav(wave_name, mode=mode, days=days)
    static_df = compute_static_basket_nav(wave_name, days=days)

    if hist.empty or static_df.empty:
        return {}

    wave_nav = hist["wave_nav"]
    bm_nav = hist["bm_nav"]
    wave_ret = hist["wave_ret"]
    bm_ret = hist["bm_ret"]

    static_nav = static_df["static_nav"]
    # normalize static to 1
    if len(static_nav) > 0:
        static_nav = static_nav / static_nav.iloc[0]

    def total_return(nav: pd.Series) -> float:
        if nav is None or len(nav) < 2:
            return float("nan")
        return float(nav.iloc[-1] / nav.iloc[0] - 1.0)

    engine_r = total_return(wave_nav)
    bm_r = total_return(bm_nav)
    static_r = total_return(static_nav)

    overlay = engine_r - static_r
    alpha_vs_bm = engine_r - bm_r

    # SPY reference
    spy_px = None
    if yf is not None:
        end = datetime.utcnow()
        start = end - timedelta(days=days + 10)
        spy_df = _download_history([BASE_INDEX], start, end)
        if not spy_df.empty and BASE_INDEX in spy_df.columns:
            spy_px = spy_df[BASE_INDEX].iloc[-len(hist):].copy()
    if spy_px is None or len(spy_px) < 2:
        spy_r = float("nan")
    else:
        spy_nav = spy_px / spy_px.iloc[0]
        spy_r = float(spy_nav.iloc[-1] / spy_nav.iloc[0] - 1.0)

    bm_difficulty = bm_r - spy_r if (math.isfinite(bm_r) and math.isfinite(spy_r)) else float("nan")

    # Risk stats
    vol_wave = float(wave_ret.std() * np.sqrt(252)) if len(wave_ret) > 2 else float("nan")
    vol_bm = float(bm_ret.std() * np.sqrt(252)) if len(bm_ret) > 2 else float("nan")

    def max_dd(nav: pd.Series) -> float:
        if nav is None or len(nav) < 2:
            return float("nan")
        rm = nav.cummax()
        dd = nav / rm - 1.0
        return float(dd.min())

    mdd_wave = max_dd(wave_nav)
    mdd_bm = max_dd(bm_nav)

    te = float((wave_ret - bm_ret).std() * np.sqrt(252)) if len(wave_ret) > 2 else float("nan")
    ir = float(alpha_vs_bm / te) if (math.isfinite(alpha_vs_bm) and math.isfinite(te) and te > 0) else float("nan")

    return {
        "engine_return": engine_r,
        "static_return": static_r,
        "overlay_contribution": overlay,
        "alpha_vs_benchmark": alpha_vs_bm,
        "benchmark_return": bm_r,
        "spy_return": spy_r,
        "benchmark_difficulty": bm_difficulty,
        "information_ratio": ir,
        "wave_vol": vol_wave,
        "bm_vol": vol_bm,
        "wave_maxdd": mdd_wave,
        "bm_maxdd": mdd_bm,
    }