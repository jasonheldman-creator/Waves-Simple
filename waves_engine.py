# waves_engine.py
# WAVES Intelligence™ — Vector Engine (Internal Holdings, SmartSafe™, Mode Separation)
#
# Public API:
#   - get_all_waves()
#   - get_modes()
#   - compute_history_nav(wave: str, mode: str, start=None, end=None)
#   - get_benchmark_mix_table(wave: str)
#   - get_wave_holdings(wave: str, mode: str)
#
# Notes:
# - Uses yfinance for market data. In Streamlit Cloud, ensure yfinance + pandas + numpy are installed.
# - “SmartSafe™” here is implemented as a rules-based risk-off sweep using:
#     • regime proxy = SPY vs 200D MA
#     • VIX gating thresholds
# - Mode separation is enforced via different caps + exposure scaling rules.

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

try:
    import yfinance as yf
except Exception:
    yf = None


# -----------------------------
# Config
# -----------------------------

DEFAULT_BASE_CCY = "USD"

MODES = ["Standard", "Alpha-Minus-Beta", "Private Logic"]

# A practical set of safe assets for SmartSafe™ sweeps
SMARTSAFE_TICKER = "BIL"     # cash-like proxy
SMARTSAFE_ALT = "SHY"        # short-term treasury ETF
VIX_TICKER = "^VIX"

# Regime proxy
REGIME_TICKER = "SPY"

# History window defaults
DEFAULT_START_DAYS = 420  # enough to compute 365D metrics + moving averages
MIN_START_DAYS_FOR_MA = 260

# Vol targeting settings
VOL_LOOKBACK_DAYS = 20
VOL_TARGET_STANDARD = 0.14
VOL_TARGET_AMB = 0.10
VOL_TARGET_PL = 0.18

# Momentum tilt settings
MOM_LOOKBACK_DAYS = 60
MOM_TILT_MAX = 0.12  # +/- tilt max per asset (before renormalization)

# VIX exposure scaling
VIX_LOW = 16.0
VIX_HIGH = 24.0
VIX_EXTREME = 32.0

# Mode exposure caps
MODE_GROSS_CAP = {
    "Standard": 1.00,
    "Alpha-Minus-Beta": 0.88,     # intentionally lower beta / exposure
    "Private Logic": 1.20,        # can run hotter
}

# SmartSafe sweep intensity by mode
MODE_SWEEP_INTENSITY = {
    "Standard": 0.55,
    "Alpha-Minus-Beta": 0.70,
    "Private Logic": 0.35,
}

# Benchmark defaults (fallbacks); engine also builds “auto composites”
BENCHMARK_FALLBACKS = {
    # equities
    "US_Large": {"SPY": 1.0},
    "US_Tech": {"QQQ": 1.0},
    "US_Small": {"IWM": 1.0},
    "US_Mid": {"MDY": 1.0},
    "US_Growth": {"VUG": 1.0},
    "US_Value": {"VTV": 1.0},
    "Global": {"VT": 1.0},
    "Intl_Dev": {"VEA": 1.0},
    "EM": {"VWO": 1.0},
    # sectors/themes
    "AI_Semis": {"SOXX": 1.0},
    "Clean_Energy": {"ICLN": 1.0},
    "Cyber": {"CIBR": 1.0},
    "Cloud": {"SKYY": 1.0},
    "Fintech": {"FINX": 1.0},
    "Space": {"ARKX": 1.0},
    # crypto proxy (ETF proxy if available)
    "Crypto": {"IBIT": 0.6, "ETHA": 0.4},  # will gracefully degrade if missing
    # income
    "Income": {"AGG": 0.6, "LQD": 0.4},
}

# -----------------------------
# Internal holdings universe
# -----------------------------
# IMPORTANT:
# You can refine tickers/weights anytime. This engine will:
# - normalize weights
# - dedupe tickers
# - apply mode overlays (SmartSafe, momentum, vol, VIX scaling)
#
# The names here should match what you want displayed in the Console.

WAVE_WEIGHTS: Dict[str, Dict[str, float]] = {
    "S&P 500 Wave": {
        "AAPL": 0.06, "MSFT": 0.06, "NVDA": 0.05, "AMZN": 0.04, "META": 0.03,
        "GOOGL": 0.03, "BRK-B": 0.03, "LLY": 0.02, "AVGO": 0.02, "JPM": 0.02,
        "XOM": 0.02, "UNH": 0.02, "COST": 0.02, "V": 0.02, "HD": 0.02,
        "MA": 0.02, "PG": 0.02, "KO": 0.01, "PEP": 0.01, "ADBE": 0.01,
    },
    "AI Wave": {
        "NVDA": 0.10, "MSFT": 0.08, "GOOGL": 0.06, "META": 0.06, "AMZN": 0.06,
        "AMD": 0.05, "AVGO": 0.05, "ASML": 0.05, "TSM": 0.05, "SMCI": 0.04,
        "ORCL": 0.04, "ADBE": 0.04, "CRM": 0.04, "SNOW": 0.03, "NOW": 0.03,
        "PLTR": 0.03, "DDOG": 0.02, "ANET": 0.02, "INTC": 0.02, "QCOM": 0.02
    },
    "Quantum Computing Wave": {
        "IONQ": 0.08, "RGTI": 0.06, "QBTS": 0.06, "IBM": 0.07, "GOOGL": 0.07,
        "MSFT": 0.07, "NVDA": 0.06, "AMZN": 0.05, "INTC": 0.05, "QCOM": 0.05,
        "HON": 0.04, "LMT": 0.04, "BA": 0.03, "AVGO": 0.06, "ASML": 0.06,
        "TSM": 0.05, "META": 0.05, "ORCL": 0.03, "PLTR": 0.03, "CSCO": 0.04,
    },
    "Crypto Wave": {
        "IBIT": 0.35, "ETHA": 0.20, "COIN": 0.08, "MSTR": 0.08, "HOOD": 0.05,
        "SQ": 0.05, "PYPL": 0.04, "RIOT": 0.04, "MARA": 0.04, "HUT": 0.03,
        "GLXY": 0.04, "BITF": 0.02, "CLSK": 0.03, "TSLA": 0.02, "NVDA": 0.03
    },
    "Crypto Income Wave": {
        "IBIT": 0.20, "ETHA": 0.12, "COIN": 0.06, "MSTR": 0.06,
        "JPM": 0.08, "BAC": 0.06, "GS": 0.06, "MS": 0.06,
        "BX": 0.06, "KKR": 0.05, "BLK": 0.05,
        "LQD": 0.06, "AGG": 0.06, "SHY": 0.04, "BIL": 0.04
    },
    "Future Power & Energy Wave": {
        "XOM": 0.06, "CVX": 0.05, "SLB": 0.05, "COP": 0.04, "EOG": 0.04,
        "NEE": 0.06, "ENPH": 0.04, "FSLR": 0.05, "TSLA": 0.05, "CHPT": 0.02,
        "LNG": 0.04, "OXY": 0.04, "VLO": 0.03, "ET": 0.03, "KMI": 0.03,
        "GEV": 0.04, "CAT": 0.04, "DE": 0.04, "PLUG": 0.02, "ICLN": 0.03,
    },
    "Clean Transit-Infrastructure Wave": {
        "CAT": 0.06, "DE": 0.05, "UNP": 0.05, "CSX": 0.04, "NSC": 0.04,
        "WM": 0.04, "VMC": 0.04, "MLM": 0.04, "URI": 0.04, "PWR": 0.05,
        "NUE": 0.04, "STLD": 0.04, "TSLA": 0.05, "ALB": 0.04, "CHPT": 0.02,
        "GE": 0.05, "HON": 0.04, "EMR": 0.04, "ETN": 0.05, "JCI": 0.03
    },
    "Cloud/Software Wave": {
        "MSFT": 0.09, "AMZN": 0.07, "GOOGL": 0.06, "ORCL": 0.06, "CRM": 0.06,
        "NOW": 0.06, "SNOW": 0.05, "DDOG": 0.05, "NET": 0.04, "PANW": 0.04,
        "CRWD": 0.04, "ZS": 0.03, "MDB": 0.03, "TEAM": 0.03, "ADBE": 0.05,
        "INTU": 0.04, "SHOP": 0.04, "PLTR": 0.03, "UBER": 0.03, "TWLO": 0.02
    },
    "Cybersecurity Wave": {
        "CRWD": 0.09, "PANW": 0.08, "ZS": 0.07, "FTNT": 0.07, "OKTA": 0.05,
        "S": 0.04, "NET": 0.05, "DDOG": 0.05, "CSCO": 0.06, "MSFT": 0.08,
        "AMZN": 0.05, "GOOGL": 0.05, "AKAM": 0.05, "CHKP": 0.05, "CYBR": 0.06
    },
    "Fintech Wave": {
        "SQ": 0.08, "PYPL": 0.07, "HOOD": 0.05, "SOFI": 0.05, "COIN": 0.06,
        "V": 0.08, "MA": 0.08, "AXP": 0.06, "JPM": 0.07, "BAC": 0.06,
        "GS": 0.05, "MS": 0.05, "NU": 0.05, "MELI": 0.05, "INTU": 0.04
    },
    "Small Cap Growth Wave": {
        "IWM": 0.25, "IWO": 0.20, "SOUN": 0.05, "RKLB": 0.05, "SOFI": 0.05,
        "CELH": 0.05, "PLTR": 0.05, "RIVN": 0.05, "UPST": 0.05, "AFRM": 0.05,
        "DUOL": 0.05, "CRSP": 0.05, "OKLO": 0.05, "ASTS": 0.05
    },
    "Small to Mid Cap Growth Wave": {
        "MDY": 0.22, "IWF": 0.15, "VUG": 0.15, "SHOP": 0.05, "SNOW": 0.05,
        "NOW": 0.05, "DDOG": 0.05, "NET": 0.04, "CRWD": 0.04, "UBER": 0.04,
        "PLTR": 0.04, "CELH": 0.04, "DUOL": 0.04, "ARM": 0.04, "AMD": 0.05
    },
    "Global Macro Wave": {
        "SPY": 0.22, "TLT": 0.16, "IEF": 0.10, "GLD": 0.12, "DBC": 0.08,
        "UUP": 0.06, "FXE": 0.04, "EWJ": 0.06, "EEM": 0.06, "VNQ": 0.05,
        "XLE": 0.05
    },
    "Income Wave": {
        "AGG": 0.28, "LQD": 0.20, "HYG": 0.10, "TIP": 0.10, "IEF": 0.10,
        "SHY": 0.08, "BIL": 0.06, "VNQ": 0.08
    },
    "Muni Ladder Wave": {
        "MUB": 0.45, "VTEB": 0.35, "SUB": 0.08, "SHY": 0.06, "BIL": 0.06
    },
    "Infinity Wave™": {
        "SPY": 0.18, "QQQ": 0.16, "VUG": 0.10, "VTV": 0.08, "IWM": 0.07,
        "GLD": 0.07, "TLT": 0.08, "IEF": 0.05, "VNQ": 0.05, "DBC": 0.05,
        "IBIT": 0.07, "ETHA": 0.04, "CASH": 0.00  # placeholder
    },
    "SmartSafe™ Money Market Wave": {
        "BIL": 0.70, "SHY": 0.20, "IEF": 0.10
    },
}

# Map waves to benchmark “themes” (fallbacks used if auto composite is not available)
WAVE_BENCHMARK_THEME: Dict[str, str] = {
    "S&P 500 Wave": "US_Large",
    "AI Wave": "US_Tech",
    "Quantum Computing Wave": "AI_Semis",
    "Crypto Wave": "Crypto",
    "Crypto Income Wave": "Crypto",
    "Future Power & Energy Wave": "Clean_Energy",
    "Clean Transit-Infrastructure Wave": "US_Large",
    "Cloud/Software Wave": "Cloud",
    "Cybersecurity Wave": "Cyber",
    "Fintech Wave": "Fintech",
    "Small Cap Growth Wave": "US_Small",
    "Small to Mid Cap Growth Wave": "US_Mid",
    "Global Macro Wave": "Global",
    "Income Wave": "Income",
    "Muni Ladder Wave": "Income",
    "Infinity Wave™": "Global",
    "SmartSafe™ Money Market Wave": "Income",
}


# -----------------------------
# Helpers
# -----------------------------

def _today_utc() -> datetime:
    return datetime.utcnow()


def _safe_ticker(t: str) -> str:
    # yfinance uses BRK-B not BRK.B
    return t.replace(".", "-").strip().upper()


def _dedupe_and_normalize(weights: Dict[str, float]) -> Dict[str, float]:
    agg: Dict[str, float] = {}
    for k, v in weights.items():
        k2 = _safe_ticker(k)
        if k2 == "CASH":
            continue
        agg[k2] = agg.get(k2, 0.0) + float(v)
    s = sum(max(0.0, x) for x in agg.values())
    if s <= 0:
        return {}
    return {k: max(0.0, v) / s for k, v in agg.items()}


def get_all_waves() -> List[str]:
    return sorted(list(WAVE_WEIGHTS.keys()))


def get_modes() -> List[str]:
    return list(MODES)


def _download_adj_close(tickers: List[str], start: datetime, end: datetime) -> pd.DataFrame:
    if yf is None:
        raise RuntimeError("yfinance is not available. Install yfinance in requirements.txt.")

    tickers = [_safe_ticker(t) for t in tickers if t]
    tickers = sorted(list(set(tickers)))

    if len(tickers) == 0:
        return pd.DataFrame()

    df = yf.download(
        tickers=tickers,
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
        auto_adjust=True,
        progress=False,
        group_by="column",
        threads=True,
    )

    # yfinance shape differs for 1 ticker vs many
    if isinstance(df.columns, pd.MultiIndex):
        if ("Close" in df.columns.get_level_values(0)) or ("Adj Close" in df.columns.get_level_values(0)):
            # auto_adjust=True => Close is adjusted; often returned as "Close"
            if "Close" in df.columns.get_level_values(0):
                px = df["Close"].copy()
            else:
                px = df["Adj Close"].copy()
        else:
            # fallback: take last level
            px = df.xs(df.columns.levels[0][0], axis=1, level=0)
    else:
        # single series
        px = df.copy()
        if "Close" in px.columns:
            px = px[["Close"]]
        px.columns = [tickers[0]]

    px = px.dropna(how="all")
    px = px.ffill().dropna(how="all")
    return px


def _compute_returns(px: pd.DataFrame) -> pd.DataFrame:
    rets = px.pct_change().replace([np.inf, -np.inf], np.nan)
    return rets.fillna(0.0)


def _max_drawdown(nav: pd.Series) -> float:
    if nav.empty:
        return 0.0
    roll_max = nav.cummax()
    dd = (nav / roll_max) - 1.0
    return float(dd.min())


def _ann_vol(daily_rets: pd.Series) -> float:
    if daily_rets.empty:
        return 0.0
    return float(daily_rets.std(ddof=0) * math.sqrt(252.0))


def _tracking_error(port_rets: pd.Series, bench_rets: pd.Series) -> float:
    if port_rets.empty or bench_rets.empty:
        return 0.0
    diff = (port_rets - bench_rets).dropna()
    if diff.empty:
        return 0.0
    return float(diff.std(ddof=0) * math.sqrt(252.0))


def _information_ratio(port_rets: pd.Series, bench_rets: pd.Series) -> float:
    if port_rets.empty or bench_rets.empty:
        return 0.0
    diff = (port_rets - bench_rets).dropna()
    if diff.empty:
        return 0.0
    te = diff.std(ddof=0)
    if te <= 1e-12:
        return 0.0
    # annualized active return / annualized TE
    active_ann = diff.mean() * 252.0
    te_ann = te * math.sqrt(252.0)
    return float(active_ann / te_ann)


def _beta(port_rets: pd.Series, bench_rets: pd.Series) -> float:
    x = bench_rets.dropna()
    y = port_rets.dropna()
    idx = x.index.intersection(y.index)
    if len(idx) < 30:
        return 1.0
    x = x.loc[idx].values
    y = y.loc[idx].values
    var = np.var(x)
    if var <= 1e-12:
        return 1.0
    cov = np.cov(x, y, ddof=0)[0, 1]
    return float(cov / var)


def _linear_clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


# -----------------------------
# SmartSafe™ + overlays
# -----------------------------

def _regime_risk_off(spy_px: pd.Series) -> bool:
    """
    Regime proxy: risk-off if SPY below its 200D MA.
    """
    if spy_px is None or spy_px.empty:
        return False
    if len(spy_px) < 210:
        return False
    ma200 = spy_px.rolling(200).mean()
    latest = spy_px.iloc[-1]
    latest_ma = ma200.iloc[-1]
    if pd.isna(latest_ma):
        return False
    return bool(latest < latest_ma)


def _vix_sweep_fraction(vix_last: float, mode: str) -> float:
    """
    Determine SmartSafe sweep fraction (0..1) based on VIX and mode.
    """
    base = MODE_SWEEP_INTENSITY.get(mode, 0.55)
    if vix_last >= VIX_EXTREME:
        return _linear_clip(base + 0.25, 0.0, 0.95)
    if vix_last >= VIX_HIGH:
        return _linear_clip(base + 0.15, 0.0, 0.90)
    if vix_last <= VIX_LOW:
        return _linear_clip(base - 0.25, 0.0, 0.75)
    # mid band
    return _linear_clip(base, 0.0, 0.85)


def _vix_exposure_scale(vix_last: float, mode: str) -> float:
    """
    Additional exposure scaling (gross cap multiplier) as VIX rises.
    Lower exposure for Standard/AMB more aggressively, PL less so.
    """
    if vix_last <= VIX_LOW:
        return 1.00
    if vix_last >= VIX_EXTREME:
        return {"Standard": 0.72, "Alpha-Minus-Beta": 0.65, "Private Logic": 0.85}.get(mode, 0.75)
    if vix_last >= VIX_HIGH:
        return {"Standard": 0.82, "Alpha-Minus-Beta": 0.75, "Private Logic": 0.92}.get(mode, 0.85)
    return {"Standard": 0.92, "Alpha-Minus-Beta": 0.88, "Private Logic": 0.98}.get(mode, 0.92)


def _apply_momentum_tilt(weights: Dict[str, float], rets: pd.DataFrame) -> Dict[str, float]:
    """
    Momentum tilt: tilt up assets with better 60D cumulative return, tilt down laggards.
    """
    if rets is None or rets.empty:
        return weights

    cols = [c for c in rets.columns if c in weights]
    if len(cols) < 3:
        return weights

    window = min(MOM_LOOKBACK_DAYS, len(rets))
    cum = (1.0 + rets[cols].iloc[-window:]).prod() - 1.0
    if cum.isna().all():
        return weights

    # rank to [-1, +1]
    ranks = cum.rank(pct=True)
    score = (ranks - 0.5) * 2.0  # [-1, 1]

    tilted = dict(weights)
    for t in cols:
        adj = float(score[t]) * MOM_TILT_MAX
        tilted[t] = max(0.0, tilted[t] * (1.0 + adj))

    return _dedupe_and_normalize(tilted)


def _apply_vol_target_scale(
    weights: Dict[str, float],
    port_daily_rets: pd.Series,
    mode: str,
) -> Tuple[Dict[str, float], float]:
    """
    Apply gross exposure scaling via vol targeting.
    Returns (weights_scaled, gross_scale).
    """
    if port_daily_rets is None or port_daily_rets.empty:
        return weights, 1.0

    look = min(VOL_LOOKBACK_DAYS, len(port_daily_rets))
    realized = float(port_daily_rets.iloc[-look:].std(ddof=0) * math.sqrt(252.0))
    target = {"Standard": VOL_TARGET_STANDARD, "Alpha-Minus-Beta": VOL_TARGET_AMB, "Private Logic": VOL_TARGET_PL}.get(mode, VOL_TARGET_STANDARD)
    if realized <= 1e-8:
        return weights, 1.0

    scale = _linear_clip(target / realized, 0.55, 1.35 if mode == "Private Logic" else 1.10)
    return weights, float(scale)


def get_wave_holdings(wave: str, mode: str) -> pd.DataFrame:
    """
    Returns a holdings table after applying:
      - normalization + dedupe
      - SmartSafe™ sweep (if risk-off regime and/or high VIX)
      - momentum tilt (60D)
      - vol targeting scale (gross exposure)
      - VIX exposure scaling + mode caps
    Output columns: ticker, weight, weight_effective, notes
    """
    if wave not in WAVE_WEIGHTS:
        raise KeyError(f"Unknown wave: {wave}")
    if mode not in MODES:
        raise KeyError(f"Unknown mode: {mode}")

    base = _dedupe_and_normalize(WAVE_WEIGHTS[wave])
    if not base:
        return pd.DataFrame(columns=["ticker", "weight", "weight_effective", "notes"])

    # Pull minimal data needed for overlays
    end = _today_utc() + timedelta(days=1)
    start = _today_utc() - timedelta(days=max(DEFAULT_START_DAYS, MIN_START_DAYS_FOR_MA))

    tickers_needed = list(base.keys()) + [REGIME_TICKER, VIX_TICKER, SMARTSAFE_TICKER, SMARTSAFE_ALT]
    tickers_needed = sorted(list(set(_safe_ticker(x) for x in tickers_needed)))

    px = _download_adj_close(tickers_needed, start=start, end=end)
    rets = _compute_returns(px)

    # Regime + VIX
    vix_last = float(px[VIX_TICKER].iloc[-1]) if VIX_TICKER in px.columns and len(px[VIX_TICKER]) > 0 else 20.0
    spy_series = px[REGIME_TICKER] if REGIME_TICKER in px.columns else None
    risk_off = _regime_risk_off(spy_series)

    # Momentum tilt first (on base risk-on basket)
    tilted = _apply_momentum_tilt(base, rets)

    notes = []
    eff = dict(tilted)

    # SmartSafe sweep (only if risk_off OR VIX above high)
    sweep = 0.0
    if risk_off or vix_last >= VIX_HIGH:
        sweep = _vix_sweep_fraction(vix_last, mode)
        notes.append(f"SmartSafe sweep {sweep:.0%} (risk_off={risk_off}, VIX={vix_last:.1f})")
        # move sweep fraction into cash proxy
        for t in list(eff.keys()):
            eff[t] = eff[t] * (1.0 - sweep)

        # allocate sweep to BIL/SHY split
        safe_split = {SMARTSAFE_TICKER: 0.70, SMARTSAFE_ALT: 0.30}
        for st, w in safe_split.items():
            eff[st] = eff.get(st, 0.0) + sweep * w

        eff = _dedupe_and_normalize(eff)

    # Compute preliminary portfolio returns (for vol targeting)
    # Use weights on available tickers
    cols = [c for c in rets.columns if c in eff]
    if len(cols) >= 2:
        port_rets = (rets[cols] * pd.Series({c: eff[c] for c in cols})).sum(axis=1)
    else:
        port_rets = pd.Series(dtype=float)

    # Vol targeting scale (gross exposure)
    eff2, vol_scale = _apply_vol_target_scale(eff, port_rets, mode)
    if abs(vol_scale - 1.0) > 1e-6:
        notes.append(f"Vol-target scale {vol_scale:.2f}")

    # VIX exposure scaling + mode cap
    vix_scale = _vix_exposure_scale(vix_last, mode)
    cap = MODE_GROSS_CAP.get(mode, 1.0)
    gross = vol_scale * vix_scale
    gross = min(gross, cap)

    if abs(vix_scale - 1.0) > 1e-6:
        notes.append(f"VIX exposure scale {vix_scale:.2f}")
    if gross < vol_scale * vix_scale:
        notes.append(f"Mode cap applied => gross {gross:.2f}")

    # Convert to output: weight (normalized 1.0) and weight_effective (scaled by gross)
    df = pd.DataFrame({
        "ticker": list(eff2.keys()),
        "weight": [eff2[t] for t in eff2.keys()],
    })
    df["weight_effective"] = df["weight"] * gross
    df["notes"] = "; ".join(notes) if notes else ""
    df = df.sort_values("weight", ascending=False).reset_index(drop=True)
    return df


# -----------------------------
# Benchmarks
# -----------------------------

def _auto_composite_benchmark_from_holdings(base_weights: Dict[str, float]) -> Dict[str, float]:
    """
    Simple benchmark composer:
    - If holdings are mostly mega-cap US equities => SPY heavy + QQQ tilt
    - If crypto ETFs appear => include those
    - If bonds dominate => AGG/LQD/TLT mix
    This is intentionally transparent and stable.
    """
    keys = list(base_weights.keys())

    has_crypto = any(t in keys for t in ["IBIT", "FBTC", "GBTC", "ETHA", "ETHE", "BITO"])
    has_bonds = sum(base_weights.get(t, 0.0) for t in ["AGG", "LQD", "IEF", "TLT", "SHY", "BIL", "TIP"]) > 0.35

    if has_bonds:
        return {"AGG": 0.55, "LQD": 0.25, "IEF": 0.10, "TLT": 0.10}

    if has_crypto:
        # blend with a tech growth proxy
        return {"IBIT": 0.35, "ETHA": 0.20, "QQQ": 0.45}

    # default equity composite
    return {"SPY": 0.65, "QQQ": 0.35}


def get_benchmark_mix_table(wave: str) -> pd.DataFrame:
    """
    Returns benchmark components per wave.
    """
    if wave not in WAVE_WEIGHTS:
        raise KeyError(f"Unknown wave: {wave}")
    base = _dedupe_and_normalize(WAVE_WEIGHTS[wave])
    auto_mix = _auto_composite_benchmark_from_holdings(base)

    theme = WAVE_BENCHMARK_THEME.get(wave, "US_Large")
    fallback = BENCHMARK_FALLBACKS.get(theme, {"SPY": 1.0})

    # Provide both, with "primary" = auto composite; "fallback" shown for transparency
    rows = []
    for k, v in sorted(auto_mix.items(), key=lambda x: -x[1]):
        rows.append({"type": "auto_composite", "ticker": _safe_ticker(k), "weight": float(v)})

    for k, v in sorted(fallback.items(), key=lambda x: -x[1]):
        rows.append({"type": "fallback_static", "ticker": _safe_ticker(k), "weight": float(v)})

    df = pd.DataFrame(rows)
    return df


def _get_benchmark_weights(wave: str) -> Dict[str, float]:
    # choose auto composite by default, fall back if it fails later due to missing tickers
    base = _dedupe_and_normalize(WAVE_WEIGHTS[wave])
    return _auto_composite_benchmark_from_holdings(base)


# -----------------------------
# NAV computation
# -----------------------------

@dataclass
class NavResult:
    wave: str
    mode: str
    start: datetime
    end: datetime
    nav: pd.Series
    bench_nav: pd.Series
    port_rets: pd.Series
    bench_rets: pd.Series
    meta: Dict[str, float]


def compute_history_nav(
    wave: str,
    mode: str,
    start: Optional[datetime] = None,
    end: Optional[datetime] = None,
) -> NavResult:
    """
    Compute NAV series for wave and benchmark.
    NAV starts at 100.
    """
    if wave not in WAVE_WEIGHTS:
        raise KeyError(f"Unknown wave: {wave}")
    if mode not in MODES:
        raise KeyError(f"Unknown mode: {mode}")

    end_dt = end or (_today_utc() + timedelta(days=1))
    start_dt = start or (_today_utc() - timedelta(days=DEFAULT_START_DAYS))

    # holdings after overlays (includes weight_effective gross scaling)
    h = get_wave_holdings(wave, mode)
    if h.empty:
        idx = pd.date_range(start_dt, end_dt, freq="B")
        zero = pd.Series(100.0, index=idx)
        return NavResult(wave, mode, start_dt, end_dt, zero, zero, pd.Series(0, index=idx), pd.Series(0, index=idx), {})

    port_w = {r["ticker"]: float(r["weight_effective"]) for _, r in h.iterrows()}
    # normalize to “effective gross” sum; keep gross scaling embedded (sum may be <1 or >1)
    gross = sum(abs(v) for v in port_w.values())
    if gross <= 1e-12:
        port_w = {r["ticker"]: float(r["weight"]) for _, r in h.iterrows()}
        gross = sum(abs(v) for v in port_w.values())
    port_norm = {k: v / gross for k, v in port_w.items()}  # normalized for return calc

    bench_w = _get_benchmark_weights(wave)
    bench_w = _dedupe_and_normalize(bench_w)

    tickers = sorted(list(set(list(port_norm.keys()) + list(bench_w.keys()))))
    px = _download_adj_close(tickers, start=start_dt, end=end_dt)
    if px.empty:
        idx = pd.date_range(start_dt, end_dt, freq="B")
        flat = pd.Series(100.0, index=idx)
        return NavResult(wave, mode, start_dt, end_dt, flat, flat, pd.Series(0, index=idx), pd.Series(0, index=idx), {})

    rets = _compute_returns(px)

    port_cols = [c for c in rets.columns if c in port_norm]
    bench_cols = [c for c in rets.columns if c in bench_w]

    port_rets = (rets[port_cols] * pd.Series({c: port_norm[c] for c in port_cols})).sum(axis=1) if port_cols else pd.Series(0.0, index=rets.index)
    bench_rets = (rets[bench_cols] * pd.Series({c: bench_w[c] for c in bench_cols})).sum(axis=1) if bench_cols else pd.Series(0.0, index=rets.index)

    nav = (1.0 + port_rets).cumprod() * 100.0
    bnav = (1.0 + bench_rets).cumprod() * 100.0

    # meta / risk metrics
    vol = _ann_vol(port_rets)
    mdd = _max_drawdown(nav)
    te = _tracking_error(port_rets, bench_rets)
    ir = _information_ratio(port_rets, bench_rets)
    beta = _beta(port_rets, bench_rets)

    meta = {
        "gross_exposure": float(sum(port_w.values())),
        "ann_vol": float(vol),
        "max_drawdown": float(mdd),
        "tracking_error": float(te),
        "information_ratio": float(ir),
        "beta": float(beta),
    }

    return NavResult(wave, mode, start_dt, end_dt, nav, bnav, port_rets, bench_rets, meta)


# -----------------------------
# WaveScore™ (pragmatic implementation)
# -----------------------------

def _bounded_score(x: float, lo: float, hi: float) -> float:
    if hi <= lo:
        return 0.0
    return 100.0 * _linear_clip((x - lo) / (hi - lo), 0.0, 1.0)


def compute_wavescore(
    wave: str,
    mode: str,
    navres: NavResult,
) -> Dict[str, float]:
    """
    Produces a WaveScore-like composite 0-100 plus subscores.
    This is a pragmatic, reproducible v1-style approximation suitable for dashboards.
    """
    pr = navres.port_rets
    br = navres.bench_rets
    if pr is None or pr.empty:
        return {"WaveScore": 0.0}

    # Return Quality (25): 12m alpha + IR
    alpha_12m = float((navres.nav.iloc[-1] / navres.nav.iloc[-min(len(navres.nav), 252)] - 1.0) -
                      (navres.bench_nav.iloc[-1] / navres.bench_nav.iloc[-min(len(navres.bench_nav), 252)] - 1.0)) if len(navres.nav) >= 40 else 0.0
    ir = navres.meta.get("information_ratio", 0.0)

    rq = 0.6 * _bounded_score(alpha_12m, -0.10, 0.25) + 0.4 * _bounded_score(ir, -0.5, 1.5)
    rq = float(_linear_clip(rq, 0.0, 100.0))

    # Risk Control (25): lower vol, lower MDD, tighter beta discipline
    vol = navres.meta.get("ann_vol", 0.0)
    mdd = navres.meta.get("max_drawdown", 0.0)
    beta = navres.meta.get("beta", 1.0)
    beta_target = 0.85 if mode == "Alpha-Minus-Beta" else (1.05 if mode == "Private Logic" else 1.00)

    vol_s = 100.0 - _bounded_score(vol, 0.10, 0.35)
    mdd_s = 100.0 - _bounded_score(abs(mdd), 0.08, 0.40)
    beta_pen = _linear_clip(100.0 * abs(beta - beta_target), 0.0, 20.0)  # 0..20
    rc = _linear_clip(0.45 * vol_s + 0.45 * mdd_s + 0.10 * (100.0 - 5.0 * beta_pen), 0.0, 100.0)

    # Consistency (15): hit rate vs benchmark + tail loss frequency
    diff = (pr - br).dropna()
    if len(diff) > 21:
        monthly = (1.0 + diff).resample("M").prod() - 1.0
        hit = float((monthly > 0).mean())
    else:
        hit = 0.5
    tail = float((pr < -0.03).mean()) if len(pr) > 30 else 0.15
    con = 0.7 * _bounded_score(hit, 0.35, 0.70) + 0.3 * (100.0 - _bounded_score(tail, 0.05, 0.25))
    con = float(_linear_clip(con, 0.0, 100.0))

    # Resilience (10): speed of recovery proxy (drawdown + recent trend)
    last_60 = pr.iloc[-min(len(pr), 60):]
    res = 0.5 * (100.0 - _bounded_score(abs(mdd), 0.10, 0.45)) + 0.5 * _bounded_score(float(last_60.mean() * 252.0), -0.10, 0.25)
    res = float(_linear_clip(res, 0.0, 100.0))

    # Efficiency (15): turnover proxy not available -> use TE + vol penalty as proxy
    te = navres.meta.get("tracking_error", 0.0)
    eff = 0.6 * (100.0 - _bounded_score(te, 0.05, 0.30)) + 0.4 * (100.0 - _bounded_score(vol, 0.12, 0.40))
    eff = float(_linear_clip(eff, 0.0, 100.0))

    # Transparency & Governance (10): data coverage proxy
    cov = 1.0 if len(navres.nav) >= 220 else (0.7 if len(navres.nav) >= 120 else 0.5)
    tg = 100.0 * cov

    wavescore = (0.25 * rq + 0.25 * rc + 0.15 * con + 0.10 * res + 0.15 * eff + 0.10 * tg)
    wavescore = float(_linear_clip(wavescore, 0.0, 100.0))

    return {
        "WaveScore": wavescore,
        "ReturnQuality": float(rq),
        "RiskControl": float(rc),
        "Consistency": float(con),
        "Resilience": float(res),
        "Efficiency": float(eff),
        "TransparencyGov": float(tg),
        "Alpha12m": float(alpha_12m),
        "IR": float(ir),
        "Vol": float(vol),
        "MaxDD": float(mdd),
        "TE": float(te),
        "Beta": float(beta),
        "BetaTarget": float(beta_target),
    }