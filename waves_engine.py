# waves_engine.py — WAVES Intelligence™ Engine (Console-Compatible)
# Safe-by-default engine with:
# - get_all_waves(), get_modes()
# - compute_history_nav(wave, mode, days, overrides=None)
# - get_wave_holdings(wave, overrides=None)  (returns current target holdings)
# - get_benchmark_mix_table(wave=None)       (simple, transparent)
#
# NOTE:
# This file is designed to “just work” with the app.py I’m providing.
# It loads wave_weights.csv if present. If you have your own richer engine,
# you can merge only the new hooks (overrides + logging) later.

from __future__ import annotations

import os
import math
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import yfinance as yf
except ImportError:
    yf = None


# -----------------------------
# Config
# -----------------------------

WEIGHTS_FILE = "wave_weights.csv"
UNIVERSE_FILE = "list.csv"

MODES = ["Standard", "Alpha-Minus-Beta", "Private Logic"]

# Base exposure multipliers by mode (safe defaults)
MODE_BASE_EXPOSURE = {
    "Standard": 1.00,
    "Alpha-Minus-Beta": 0.85,
    "Private Logic": 1.15,
}

# Safe assets used for SmartSafe-style risk-off sleeves
SAFE_CANDIDATES = ["SGOV", "BIL", "SHY"]  # pick first available
VIX_TICKER = "^VIX"
SPY_TICKER = "SPY"

# VIX regime thresholds (simple + readable)
VIX_REGIMES = [
    ("Low", 0, 16),
    ("Medium", 16, 22),
    ("High", 22, 30),
    ("Stress", 30, 10_000),
]

# Default SmartSafe risk-off fractions by VIX regime
# (fraction of portfolio diverted to SAFE asset sleeve)
SMARTSAFE_BY_VIX = {
    "Low": 0.00,
    "Medium": 0.10,
    "High": 0.25,
    "Stress": 0.40,
}

# Maximum per-session override changes (safety rails)
MAX_SMARTSAFE_DELTA = 0.25   # max +/- 25% absolute change from baseline
MAX_EXPOSURE_DELTA = 0.25    # max +/- 0.25 exposure multiplier from base


# -----------------------------
# Utilities
# -----------------------------

def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def _today_utc() -> datetime:
    return datetime.utcnow()

def _read_csv_if_exists(path: str) -> pd.DataFrame:
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame()

def _coerce_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    cols = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols:
            return cols[cand.lower()]
    return None

def _normalize_weights(items: List[Tuple[str, float]]) -> List[Tuple[str, float]]:
    items = [(t, float(w)) for t, w in items if t and pd.notna(w)]
    s = sum(w for _, w in items)
    if s <= 0:
        return []
    return [(t, w / s) for t, w in items]

def _pick_safe_ticker(price_df: pd.DataFrame) -> Optional[str]:
    for t in SAFE_CANDIDATES:
        if t in price_df.columns:
            return t
    return None

def _download_history(tickers: List[str], start: datetime, end: datetime) -> pd.DataFrame:
    if yf is None:
        return pd.DataFrame()
    if not tickers:
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
    if data is None or len(getattr(data, "columns", [])) == 0:
        return pd.DataFrame()

    # Normalize to a simple price frame
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
    return data


# -----------------------------
# Data model: overrides
# -----------------------------

@dataclass
class SessionOverrides:
    # exposure multiplier additive delta relative to mode base exposure
    exposure_delta: float = 0.0
    # smartsafe delta additive to SMARTSAFE_BY_VIX baseline (clamped)
    smartsafe_delta: float = 0.0


def validate_overrides(overrides: Optional[dict]) -> SessionOverrides:
    if not overrides:
        return SessionOverrides()
    ex = float(overrides.get("exposure_delta", 0.0))
    ss = float(overrides.get("smartsafe_delta", 0.0))

    ex = float(np.clip(ex, -MAX_EXPOSURE_DELTA, MAX_EXPOSURE_DELTA))
    ss = float(np.clip(ss, -MAX_SMARTSAFE_DELTA, MAX_SMARTSAFE_DELTA))
    return SessionOverrides(exposure_delta=ex, smartsafe_delta=ss)


# -----------------------------
# Load wave weights
# -----------------------------

def _load_wave_weights() -> Dict[str, List[Tuple[str, float]]]:
    df = _read_csv_if_exists(WEIGHTS_FILE)
    if df.empty:
        return {}

    wave_col = _coerce_col(df, ["Wave", "wave", "WaveName", "wave_name"])
    ticker_col = _coerce_col(df, ["Ticker", "ticker", "Symbol", "symbol"])
    weight_col = _coerce_col(df, ["Weight", "weight", "Pct", "pct"])

    if not wave_col or not ticker_col or not weight_col:
        return {}

    weights: Dict[str, List[Tuple[str, float]]] = {}
    for _, r in df.iterrows():
        w = str(r[wave_col]).strip()
        t = str(r[ticker_col]).strip().upper()
        wt = r[weight_col]
        if not w or not t or pd.isna(wt):
            continue
        weights.setdefault(w, []).append((t, float(wt)))

    # Normalize each wave
    for k in list(weights.keys()):
        weights[k] = _normalize_weights(weights[k])

    return weights


WAVE_WEIGHTS: Dict[str, List[Tuple[str, float]]] = _load_wave_weights()


# -----------------------------
# Public API used by app.py
# -----------------------------

def get_all_waves() -> List[str]:
    return sorted(list(WAVE_WEIGHTS.keys()))

def get_modes() -> List[str]:
    return MODES.copy()

def get_wave_holdings(wave_name: str, overrides: Optional[dict] = None) -> pd.DataFrame:
    """
    Returns target holdings table (not intraday live fills).
    For the console Top-10, this is enough.
    """
    if wave_name not in WAVE_WEIGHTS:
        return pd.DataFrame(columns=["Ticker", "Name", "Weight"])

    holdings = WAVE_WEIGHTS[wave_name]
    df = pd.DataFrame(holdings, columns=["Ticker", "Weight"]).copy()
    # Names are optional; keep it simple (ticker only)
    df["Name"] = df["Ticker"]
    df = df.sort_values("Weight", ascending=False).reset_index(drop=True)
    return df[["Ticker", "Name", "Weight"]]

def get_benchmark_mix_table(wave: Optional[str] = None) -> pd.DataFrame:
    """
    Simple transparent benchmark:
    - If wave exists: benchmark = SPY 60% + VTV 40% (placeholder mix)
    - Otherwise: return a combined table for all waves.

    You can replace this with your real composite-benchmark mapping later.
    """
    def _mix_rows(w: str) -> List[dict]:
        return [
            {"Wave": w, "Ticker": "SPY", "Name": "SPDR S&P 500 ETF", "Weight": 0.60},
            {"Wave": w, "Ticker": "VTV", "Name": "Vanguard Value ETF", "Weight": 0.40},
        ]

    if wave:
        return pd.DataFrame(_mix_rows(wave))

    rows = []
    for w in get_all_waves():
        rows.extend(_mix_rows(w))
    return pd.DataFrame(rows)

def _vix_regime(vix_level: float) -> str:
    if math.isnan(vix_level):
        return "Unknown"
    for name, lo, hi in VIX_REGIMES:
        if lo <= vix_level < hi:
            return name
    return "Unknown"

def _smartsafe_fraction(regime: str, overrides: SessionOverrides) -> float:
    base = SMARTSAFE_BY_VIX.get(regime, 0.0)
    frac = base + overrides.smartsafe_delta
    return float(np.clip(frac, 0.0, 0.80))  # never exceed 80% safe sleeve

def compute_history_nav(
    wave_name: str,
    mode: str,
    days: int = 365,
    overrides: Optional[dict] = None,
) -> pd.DataFrame:
    """
    Computes wave NAV and benchmark NAV with:
    - daily rebalance (simple)
    - mode exposure multiplier
    - SmartSafe based on VIX regime (+ optional session overrides)
    Returns DataFrame indexed by date with columns:
      wave_nav, bm_nav, wave_ret, bm_ret
    """
    if wave_name not in WAVE_WEIGHTS:
        raise ValueError(f"Unknown Wave: {wave_name}")
    if mode not in MODE_BASE_EXPOSURE:
        raise ValueError(f"Unknown mode: {mode}")

    ov = validate_overrides(overrides)

    end = _today_utc()
    start = end - timedelta(days=int(days) + 10)

    wave_holdings = WAVE_WEIGHTS[wave_name]
    wave_tickers = [t for t, _ in wave_holdings]

    # Benchmark tickers (simple mix placeholder)
    bm_mix = get_benchmark_mix_table(wave_name)
    bm_tickers = bm_mix["Ticker"].astype(str).str.upper().tolist()

    tickers = sorted(list(set(wave_tickers + bm_tickers + [SPY_TICKER, VIX_TICKER] + SAFE_CANDIDATES)))

    price = _download_history(tickers, start=start, end=end)
    if price.empty:
        return pd.DataFrame(columns=["wave_nav", "bm_nav", "wave_ret", "bm_ret"])

    # Trim to days
    if len(price) > days:
        price = price.iloc[-days:]

    # Ensure required series exist
    if SPY_TICKER not in price.columns:
        return pd.DataFrame(columns=["wave_nav", "bm_nav", "wave_ret", "bm_ret"])
    if VIX_TICKER not in price.columns:
        # allow missing VIX: assume Medium
        price[VIX_TICKER] = np.nan

    safe_ticker = _pick_safe_ticker(price)

    # Daily returns
    ret = price.pct_change().fillna(0.0)

    # Static weights
    wave_w = pd.Series({t: w for t, w in wave_holdings}, dtype=float)
    wave_w = wave_w / wave_w.sum() if wave_w.sum() > 0 else wave_w

    bm_w = pd.Series(
        bm_mix.set_index("Ticker")["Weight"].astype(float).to_dict(),
        dtype=float,
    )
    bm_w = bm_w / bm_w.sum() if bm_w.sum() > 0 else bm_w

    # Build NAVs
    wave_nav = []
    bm_nav = []
    wave_nav_val = 1.0
    bm_nav_val = 1.0

    base_exposure = MODE_BASE_EXPOSURE[mode] + ov.exposure_delta
    base_exposure = float(np.clip(base_exposure, 0.50, 1.50))

    for dt in ret.index:
        vix_level = float(price.loc[dt, VIX_TICKER]) if VIX_TICKER in price.columns else float("nan")
        regime = _vix_regime(vix_level) if not math.isnan(vix_level) else "Medium"
        ss_frac = _smartsafe_fraction(regime, ov) if safe_ticker else 0.0

        # Effective wave weights = (1-ss)*base_exposure*wave + ss*safe
        w_eff = wave_w.copy()
        # scale risk sleeve
        w_eff = w_eff * (1.0 - ss_frac)
        if safe_ticker:
            w_eff[safe_ticker] = w_eff.get(safe_ticker, 0.0) + ss_frac

        # Apply exposure scaling: multiply risk sleeve contribution by base_exposure
        # Implemented by scaling returns impact (not leverage financing)
        wave_ret_day = 0.0
        for t, w in w_eff.items():
            if t not in ret.columns:
                continue
            r = float(ret.loc[dt, t])
            if t == safe_ticker:
                wave_ret_day += w * r
            else:
                wave_ret_day += w * (base_exposure * r)

        # Benchmark: simple static mix (no SmartSafe)
        bm_ret_day = 0.0
        for t, w in bm_w.items():
            if t not in ret.columns:
                continue
            bm_ret_day += float(w) * float(ret.loc[dt, t])

        wave_nav_val *= (1.0 + wave_ret_day)
        bm_nav_val *= (1.0 + bm_ret_day)

        wave_nav.append(wave_nav_val)
        bm_nav.append(bm_nav_val)

    out = pd.DataFrame(
        {
            "wave_nav": wave_nav,
            "bm_nav": bm_nav,
        },
        index=ret.index,
    )
    out["wave_ret"] = out["wave_nav"].pct_change().fillna(0.0)
    out["bm_ret"] = out["bm_nav"].pct_change().fillna(0.0)
    return out