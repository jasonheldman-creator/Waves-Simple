# waves_engine.py
# WAVES Intelligence™ — Vector Engine (Internal 20 Waves + Old Console API Compatible)
#
# This engine is designed to support BOTH:
#   (A) The richer “old console” app.py backup API signatures
#   (B) A mode-separated, overlay-driven Vector Engine behavior
#
# Public API (as expected by your backup console):
#   - get_all_waves() -> list[str]
#   - get_modes() -> list[str]
#   - compute_history_nav(wave_name: str, mode: str, days: int = 365) -> pd.DataFrame
#         returns df indexed by date with columns:
#           ["wave_nav","bm_nav","wave_ret","bm_ret"]
#   - get_benchmark_mix_table() -> pd.DataFrame
#         returns columns:
#           ["Wave","Ticker","Name","Weight","Type"]
#   - get_wave_holdings(wave_name: str) -> pd.DataFrame
#         returns columns:
#           ["Ticker","Name","Weight"]
#
# Optional internal helpers are present to support overlays:
#   - SmartSafe sweeps (regime + VIX)
#   - Momentum tilt (60D)
#   - Vol targeting (20D realized)
#   - VIX exposure scaling
#   - Mode caps: Standard / Alpha-Minus-Beta / Private Logic
#
# Notes:
# - Uses yfinance for daily adjusted closes.
# - If yfinance is missing, functions return empty frames (console will display warnings).

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
# Modes / Config
# -----------------------------

MODES = ["Standard", "Alpha-Minus-Beta", "Private Logic"]

REGIME_TICKER = "SPY"
VIX_TICKER = "^VIX"

SMARTSAFE_TICKER = "BIL"  # cash-like proxy
SMARTSAFE_ALT = "SHY"     # short-term treasury proxy

# History + overlays
DEFAULT_BACKFILL_DAYS = 730  # we’ll fetch extra so 200D MA + 365D stats are stable
MA_REGIME_DAYS = 200

MOM_LOOKBACK_DAYS = 60
MOM_TILT_MAX = 0.12  # per-asset tilt before renormalizing

VOL_LOOKBACK_DAYS = 20
VOL_TARGET = {"Standard": 0.14, "Alpha-Minus-Beta": 0.10, "Private Logic": 0.18}

# VIX gating thresholds
VIX_LOW = 16.0
VIX_HIGH = 24.0
VIX_EXTREME = 32.0

# Mode caps (gross exposure multiplier)
MODE_GROSS_CAP = {"Standard": 1.00, "Alpha-Minus-Beta": 0.88, "Private Logic": 1.20}

# SmartSafe intensity by mode (base)
MODE_SWEEP_INTENSITY = {"Standard": 0.55, "Alpha-Minus-Beta": 0.70, "Private Logic": 0.35}

# VIX exposure scaling by mode (reduces exposure when volatility rises)
def _vix_exposure_scale(vix_last: float, mode: str) -> float:
    if vix_last <= VIX_LOW:
        return 1.00
    if vix_last >= VIX_EXTREME:
        return {"Standard": 0.72, "Alpha-Minus-Beta": 0.65, "Private Logic": 0.85}.get(mode, 0.75)
    if vix_last >= VIX_HIGH:
        return {"Standard": 0.82, "Alpha-Minus-Beta": 0.75, "Private Logic": 0.92}.get(mode, 0.85)
    return {"Standard": 0.92, "Alpha-Minus-Beta": 0.88, "Private Logic": 0.98}.get(mode, 0.92)


def _clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _safe_ticker(t: str) -> str:
    return t.replace(".", "-").strip().upper()


def _dedupe_and_normalize(weights: Dict[str, float]) -> Dict[str, float]:
    agg: Dict[str, float] = {}
    for k, v in weights.items():
        k2 = _safe_ticker(k)
        if k2 in ("CASH", ""):
            continue
        agg[k2] = agg.get(k2, 0.0) + float(v)

    s = sum(max(0.0, x) for x in agg.values())
    if s <= 0:
        return {}
    return {k: max(0.0, v) / s for k, v in agg.items()}


# -----------------------------
# INTERNAL 20-WAVE HOLDINGS
# -----------------------------
# You can refine these later; the engine will always auto-discover all waves from this dict.

WAVE_WEIGHTS: Dict[str, Dict[str, float]] = {
    # 1
    "S&P 500 Wave": {
        "AAPL": 0.06, "MSFT": 0.06, "NVDA": 0.05, "AMZN": 0.04, "META": 0.03,
        "GOOGL": 0.03, "BRK-B": 0.03, "LLY": 0.02, "AVGO": 0.02, "JPM": 0.02,
        "XOM": 0.02, "UNH": 0.02, "COST": 0.02, "V": 0.02, "HD": 0.02,
        "MA": 0.02, "PG": 0.02, "KO": 0.01, "PEP": 0.01, "ADBE": 0.01,
    },

    # 2
    "AI Wave": {
        "NVDA": 0.10, "MSFT": 0.08, "GOOGL": 0.06, "META": 0.06, "AMZN": 0.06,
        "AMD": 0.05, "AVGO": 0.05, "ASML": 0.05, "TSM": 0.05, "SMCI": 0.04,
        "ORCL": 0.04, "ADBE": 0.04, "CRM": 0.04, "SNOW": 0.03, "NOW": 0.03,
        "PLTR": 0.03, "DDOG": 0.02, "ANET": 0.02, "INTC": 0.02, "QCOM": 0.02
    },

    # 3
    "Quantum Computing Wave": {
        "IONQ": 0.08, "RGTI": 0.06, "QBTS": 0.06, "IBM": 0.07, "GOOGL": 0.07,
        "MSFT": 0.07, "NVDA": 0.06, "AMZN": 0.05, "INTC": 0.05, "QCOM": 0.05,
        "HON": 0.04, "LMT": 0.04, "BA": 0.03, "AVGO": 0.06, "ASML": 0.06,
        "TSM": 0.05, "META": 0.05, "ORCL": 0.03, "PLTR": 0.03, "CSCO": 0.04,
    },

    # 4
    "Crypto Wave": {
        "IBIT": 0.35, "ETHA": 0.20, "COIN": 0.08, "MSTR": 0.08, "HOOD": 0.05,
        "SQ": 0.05, "PYPL": 0.04, "RIOT": 0.04, "MARA": 0.04, "HUT": 0.03,
        "BITF": 0.02, "CLSK": 0.03, "TSLA": 0.02, "NVDA": 0.03, "BLK": 0.02
    },

    # 5
    "Crypto Income Wave": {
        "IBIT": 0.20, "ETHA": 0.12, "COIN": 0.06, "MSTR": 0.06,
        "JPM": 0.08, "BAC": 0.06, "GS": 0.06, "MS": 0.06,
        "BX": 0.06, "KKR": 0.05, "BLK": 0.05,
        "LQD": 0.06, "AGG": 0.06, "SHY": 0.04, "BIL": 0.04
    },

    # 6
    "Future Power & Energy Wave": {
        "XOM": 0.06, "CVX": 0.05, "SLB": 0.05, "COP": 0.04, "EOG": 0.04,
        "NEE": 0.06, "ENPH": 0.04, "FSLR": 0.05, "TSLA": 0.05, "CHPT": 0.02,
        "LNG": 0.04, "OXY": 0.04, "VLO": 0.03, "ET": 0.03, "KMI": 0.03,
        "CAT": 0.04, "DE": 0.04, "ICLN": 0.03, "PLUG": 0.02, "GE": 0.04,
    },

    # 7
    "Clean Transit-Infrastructure Wave": {
        "CAT": 0.06, "DE": 0.05, "UNP": 0.05, "CSX": 0.04, "NSC": 0.04,
        "WM": 0.04, "VMC": 0.04, "MLM": 0.04, "URI": 0.04, "PWR": 0.05,
        "NUE": 0.04, "STLD": 0.04, "TSLA": 0.05, "ALB": 0.04, "CHPT": 0.02,
        "HON": 0.04, "EMR": 0.04, "ETN": 0.05, "JCI": 0.03, "GE": 0.04
    },

    # 8
    "Cloud/Software Wave": {
        "MSFT": 0.09, "AMZN": 0.07, "GOOGL": 0.06, "ORCL": 0.06, "CRM": 0.06,
        "NOW": 0.06, "SNOW": 0.05, "DDOG": 0.05, "NET": 0.04, "PANW": 0.04,
        "CRWD": 0.04, "ZS": 0.03, "MDB": 0.03, "TEAM": 0.03, "ADBE": 0.05,
        "INTU": 0.04, "SHOP": 0.04, "PLTR": 0.03, "UBER": 0.03, "TWLO": 0.02
    },

    # 9
    "Cybersecurity Wave": {
        "CRWD": 0.09, "PANW": 0.08, "ZS": 0.07, "FTNT": 0.07, "OKTA": 0.05,
        "NET": 0.05, "DDOG": 0.05, "CSCO": 0.06, "MSFT": 0.08, "GOOGL": 0.05,
        "AMZN": 0.05, "CHKP": 0.05, "CYBR": 0.06, "AKAM": 0.05, "S": 0.04
    },

    # 10
    "Fintech Wave": {
        "SQ": 0.08, "PYPL": 0.07, "HOOD": 0.05, "SOFI": 0.05, "COIN": 0.06,
        "V": 0.08, "MA": 0.08, "AXP": 0.06, "JPM": 0.07, "BAC": 0.06,
        "GS": 0.05, "MS": 0.05, "NU": 0.05, "MELI": 0.05, "INTU": 0.04
    },

    # 11
    "Small Cap Growth Wave": {
        "IWM": 0.25, "IWO": 0.20, "SOUN": 0.05, "RKLB": 0.05, "SOFI": 0.05,
        "CELH": 0.05, "PLTR": 0.05, "RIVN": 0.05, "UPST": 0.05, "AFRM": 0.05,
        "DUOL": 0.05, "CRSP": 0.05, "ASTS": 0.05, "OKLO": 0.05
    },

    # 12
    "Small to Mid Cap Growth Wave": {
        "MDY": 0.22, "IWF": 0.15, "VUG": 0.15, "SHOP": 0.05, "SNOW": 0.05,
        "NOW": 0.05, "DDOG": 0.05, "NET": 0.04, "CRWD": 0.04, "UBER": 0.04,
        "PLTR": 0.04, "CELH": 0.04, "DUOL": 0.04, "ARM": 0.04, "AMD": 0.05
    },

    # 13
    "Global Macro Wave": {
        "SPY": 0.22, "TLT": 0.16, "IEF": 0.10, "GLD": 0.12, "DBC": 0.08,
        "UUP": 0.06, "FXE": 0.04, "EWJ": 0.06, "EEM": 0.06, "VNQ": 0.05,
        "XLE": 0.05
    },

    # 14
    "Income Wave": {
        "AGG": 0.28, "LQD": 0.20, "HYG": 0.10, "TIP": 0.10, "IEF": 0.10,
        "SHY": 0.08, "BIL": 0.06, "VNQ": 0.08
    },

    # 15
    "Muni Ladder Wave": {
        "MUB": 0.45, "VTEB": 0.35, "SUB": 0.08, "SHY": 0.06, "BIL": 0.06
    },

    # 16
    "Infinity Wave™": {
        "SPY": 0.18, "QQQ": 0.16, "VUG": 0.10, "VTV": 0.08, "IWM": 0.07,
        "GLD": 0.07, "TLT": 0.08, "IEF": 0.05, "VNQ": 0.05, "DBC": 0.05,
        "IBIT": 0.07, "ETHA": 0.04
    },

    # 17
    "SmartSafe™ Money Market Wave": {
        "BIL": 0.70, "SHY": 0.20, "IEF": 0.10
    },

    # 18 (added)
    "Healthcare Innovation Wave": {
        "LLY": 0.08, "UNH": 0.06, "JNJ": 0.06, "ABBV": 0.06, "MRK": 0.06,
        "TMO": 0.06, "DHR": 0.06, "ISRG": 0.05, "VRTX": 0.05, "REGN": 0.05,
        "PFE": 0.04, "BMY": 0.04, "GILD": 0.04, "AMGN": 0.05, "SYK": 0.04,
        "MDT": 0.04, "HCA": 0.04, "CI": 0.04, "IQV": 0.04, "BSX": 0.04
    },

    # 19 (added)
    "Defense & Aerospace Wave": {
        "LMT": 0.10, "NOC": 0.08, "RTX": 0.08, "GD": 0.06, "BA": 0.06,
        "LHX": 0.06, "HII": 0.04, "HEI": 0.04, "TDG": 0.06, "TXT": 0.03,
        "HON": 0.05, "GE": 0.06, "ETN": 0.05, "RRX": 0.03, "BWXT": 0.04,
        "KTOS": 0.03, "AVAV": 0.03, "PLTR": 0.06, "MSFT": 0.03, "CSCO": 0.02
    },

    # 20 (added)
    "Space Economy Wave": {
        "RKLB": 0.08, "LUNR": 0.04, "ASTS": 0.06, "PLTR": 0.06, "NVDA": 0.06,
        "BA": 0.05, "LMT": 0.05, "NOC": 0.04, "RTX": 0.04, "IRDM": 0.05,
        "TSLA": 0.05, "GOOGL": 0.05, "AMZN": 0.05, "MSFT": 0.05, "AVGO": 0.04,
        "CSCO": 0.04, "HON": 0.04, "QCOM": 0.04, "INTC": 0.03, "SPCE": 0.02
    },
}

# Benchmark “themes” (fallbacks) used if composite tickers are missing
BENCHMARK_FALLBACKS: Dict[str, Dict[str, float]] = {
    "US_Large": {"SPY": 1.0},
    "US_Tech": {"QQQ": 1.0},
    "US_Small": {"IWM": 1.0},
    "US_Mid": {"MDY": 1.0},
    "Growth": {"VUG": 1.0},
    "Value": {"VTV": 1.0},
    "Global": {"VT": 1.0},
    "Income": {"AGG": 0.6, "LQD": 0.4},
    "Crypto": {"IBIT": 0.6, "ETHA": 0.4},
    "Healthcare": {"XLV": 1.0},
    "Defense": {"ITA": 1.0},
    "Space": {"ARKX": 1.0},
    "Energy": {"XLE": 1.0},
}

WAVE_BENCH_THEME: Dict[str, str] = {
    "S&P 500 Wave": "US_Large",
    "AI Wave": "US_Tech",
    "Quantum Computing Wave": "US_Tech",
    "Crypto Wave": "Crypto",
    "Crypto Income Wave": "Crypto",
    "Future Power & Energy Wave": "Energy",
    "Clean Transit-Infrastructure Wave": "US_Large",
    "Cloud/Software Wave": "US_Tech",
    "Cybersecurity Wave": "US_Tech",
    "Fintech Wave": "US_Tech",
    "Small Cap Growth Wave": "US_Small",
    "Small to Mid Cap Growth Wave": "US_Mid",
    "Global Macro Wave": "Global",
    "Income Wave": "Income",
    "Muni Ladder Wave": "Income",
    "Infinity Wave™": "Global",
    "SmartSafe™ Money Market Wave": "Income",
    "Healthcare Innovation Wave": "Healthcare",
    "Defense & Aerospace Wave": "Defense",
    "Space Economy Wave": "Space",
}


# -----------------------------
# Core fetch
# -----------------------------

def _download_adj_close(tickers: List[str], start: datetime, end: datetime) -> pd.DataFrame:
    if yf is None:
        return pd.DataFrame()

    tickers = [_safe_ticker(t) for t in tickers if t]
    tickers = sorted(list(set(tickers)))
    if not tickers:
        return pd.DataFrame()

    data = yf.download(
        tickers=tickers,
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
        interval="1d",
        auto_adjust=True,
        progress=False,
        group_by="column",
        threads=True,
    )

    # MultiIndex handling
    if isinstance(data.columns, pd.MultiIndex):
        if "Adj Close" in data.columns.get_level_values(0):
            px = data["Adj Close"].copy()
        elif "Close" in data.columns.get_level_values(0):
            px = data["Close"].copy()
        else:
            # fallback first top-level
            px = data[data.columns.levels[0][0]].copy()
    else:
        px = data.copy()
        if isinstance(px, pd.Series):
            px = px.to_frame(name=tickers[0])
        if "Close" in px.columns:
            px = px[["Close"]]
            px.columns = [tickers[0]]

    if isinstance(px.columns, pd.MultiIndex):
        px = px.droplevel(0, axis=1)

    px = px.sort_index().dropna(how="all").ffill().bfill()
    return px


def _returns(px: pd.DataFrame) -> pd.DataFrame:
    if px is None or px.empty:
        return pd.DataFrame()
    r = px.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return r


# -----------------------------
# Regime / overlays
# -----------------------------

def _risk_off(spy_px: pd.Series) -> bool:
    if spy_px is None or spy_px.empty or len(spy_px) < (MA_REGIME_DAYS + 10):
        return False
    ma = spy_px.rolling(MA_REGIME_DAYS).mean()
    if pd.isna(ma.iloc[-1]):
        return False
    return bool(spy_px.iloc[-1] < ma.iloc[-1])


def _vix_sweep_fraction(vix_last: float, mode: str) -> float:
    base = MODE_SWEEP_INTENSITY.get(mode, 0.55)
    if vix_last >= VIX_EXTREME:
        return _clip(base + 0.25, 0.0, 0.95)
    if vix_last >= VIX_HIGH:
        return _clip(base + 0.15, 0.0, 0.90)
    if vix_last <= VIX_LOW:
        return _clip(base - 0.25, 0.0, 0.75)
    return _clip(base, 0.0, 0.85)


def _apply_momentum_tilt(weights: Dict[str, float], rets: pd.DataFrame) -> Dict[str, float]:
    if rets is None or rets.empty:
        return weights
    cols = [c for c in rets.columns if c in weights]
    if len(cols) < 3:
        return weights

    window = min(MOM_LOOKBACK_DAYS, len(rets))
    cum = (1.0 + rets[cols].iloc[-window:]).prod() - 1.0
    if cum.isna().all():
        return weights

    ranks = cum.rank(pct=True)
    score = (ranks - 0.5) * 2.0  # [-1, 1]

    tilted = dict(weights)
    for t in cols:
        adj = float(score[t]) * MOM_TILT_MAX
        tilted[t] = max(0.0, tilted[t] * (1.0 + adj))

    return _dedupe_and_normalize(tilted)


def _apply_vol_target_scale(port_rets: pd.Series, mode: str) -> float:
    if port_rets is None or port_rets.empty:
        return 1.0
    look = min(VOL_LOOKBACK_DAYS, len(port_rets))
    realized = float(port_rets.iloc[-look:].std(ddof=0) * math.sqrt(252.0))
    target = VOL_TARGET.get(mode, 0.14)
    if realized <= 1e-9:
        return 1.0
    scale = target / realized
    # constrain
    return float(_clip(scale, 0.55, 1.35 if mode == "Private Logic" else 1.10))


def _auto_composite_benchmark_from_holdings(base_weights: Dict[str, float]) -> Dict[str, float]:
    keys = list(base_weights.keys())

    has_crypto = any(t in keys for t in ["IBIT", "ETHA", "GBTC", "FBTC", "BITO", "ETHE"])
    bond_share = sum(base_weights.get(t, 0.0) for t in ["AGG", "LQD", "IEF", "TLT", "SHY", "BIL", "TIP", "MUB", "VTEB"])  # rough

    if bond_share > 0.35:
        return {"AGG": 0.55, "LQD": 0.25, "IEF": 0.10, "TLT": 0.10}

    if has_crypto:
        return {"QQQ": 0.45, "IBIT": 0.35, "ETHA": 0.20}

    # default equity composite
    return {"SPY": 0.65, "QQQ": 0.35}


def _fallback_benchmark_for_wave(wave: str) -> Dict[str, float]:
    theme = WAVE_BENCH_THEME.get(wave, "US_Large")
    return BENCHMARK_FALLBACKS.get(theme, {"SPY": 1.0})


# -----------------------------
# Public API
# -----------------------------

def get_all_waves() -> List[str]:
    return sorted(list(WAVE_WEIGHTS.keys()))


def get_modes() -> List[str]:
    return list(MODES)


def get_wave_holdings(wave_name: str) -> pd.DataFrame:
    """
    Static holdings table (NO overlays). Used for attribution in the old console.
    Columns: Ticker, Name, Weight
    """
    if wave_name not in WAVE_WEIGHTS:
        return pd.DataFrame(columns=["Ticker", "Name", "Weight"])

    base = _dedupe_and_normalize(WAVE_WEIGHTS[wave_name])
    rows = []
    for t, w in sorted(base.items(), key=lambda x: -x[1]):
        rows.append({"Ticker": t, "Name": t, "Weight": float(w)})

    return pd.DataFrame(rows)


def get_benchmark_mix_table() -> pd.DataFrame:
    """
    Returns composite benchmark components per wave (plus fallback static).
    Columns: Wave, Ticker, Name, Weight, Type
    """
    rows = []
    for wave in get_all_waves():
        base = _dedupe_and_normalize(WAVE_WEIGHTS[wave])
        comp = _auto_composite_benchmark_from_holdings(base)
        comp = _dedupe_and_normalize(comp)
        fb = _dedupe_and_normalize(_fallback_benchmark_for_wave(wave))

        for t, w in sorted(comp.items(), key=lambda x: -x[1]):
            rows.append({"Wave": wave, "Ticker": _safe_ticker(t), "Name": _safe_ticker(t), "Weight": float(w), "Type": "auto_composite"})
        for t, w in sorted(fb.items(), key=lambda x: -x[1]):
            rows.append({"Wave": wave, "Ticker": _safe_ticker(t), "Name": _safe_ticker(t), "Weight": float(w), "Type": "fallback_static"})

    return pd.DataFrame(rows)


def compute_history_nav(wave_name: str, mode: str, days: int = 365) -> pd.DataFrame:
    """
    Old console compatible:
      returns df indexed by date with columns:
        wave_nav, bm_nav, wave_ret, bm_ret
    """
    if yf is None:
        return pd.DataFrame(columns=["wave_nav", "bm_nav", "wave_ret", "bm_ret"])

    if wave_name not in WAVE_WEIGHTS:
        return pd.DataFrame(columns=["wave_nav", "bm_nav", "wave_ret", "bm_ret"])

    if mode not in MODES:
        mode = "Standard"

    # fetch window: backfill for MA + stable signals
    end = datetime.utcnow().date()
    start = end - timedelta(days=days + DEFAULT_BACKFILL_DAYS)

    base = _dedupe_and_normalize(WAVE_WEIGHTS[wave_name])
    if not base:
        return pd.DataFrame(columns=["wave_nav", "bm_nav", "wave_ret", "bm_ret"])

    # composite benchmark
    bm_w = _dedupe_and_normalize(_auto_composite_benchmark_from_holdings(base))
    fb_w = _dedupe_and_normalize(_fallback_benchmark_for_wave(wave_name))

    # tickers to fetch
    tickers = set(base.keys()) | set(bm_w.keys()) | {REGIME_TICKER, VIX_TICKER, SMARTSAFE_TICKER, SMARTSAFE_ALT}
    px = _download_adj_close(list(tickers), start=datetime.combine(start, datetime.min.time()), end=datetime.combine(end + timedelta(days=1), datetime.min.time()))
    if px.empty:
        return pd.DataFrame(columns=["wave_nav", "bm_nav", "wave_ret", "bm_ret"])

    rets = _returns(px)

    # determine risk regime
    spy_series = px[REGIME_TICKER] if REGIME_TICKER in px.columns else None
    vix_last = float(px[VIX_TICKER].iloc[-1]) if VIX_TICKER in px.columns and len(px[VIX_TICKER]) else 20.0
    is_risk_off = _risk_off(spy_series)

    # --- build dynamic weights (mode-separated overlays) ---
    w_dyn = dict(base)

    # momentum tilt (60D)
    w_dyn = _apply_momentum_tilt(w_dyn, rets)

    # SmartSafe sweep if risk-off OR VIX high
    sweep = 0.0
    if is_risk_off or vix_last >= VIX_HIGH:
        sweep = _vix_sweep_fraction(vix_last, mode)
        # reduce risk basket
        for t in list(w_dyn.keys()):
            w_dyn[t] = w_dyn[t] * (1.0 - sweep)
        # allocate swept weight into safe assets
        w_dyn[SMARTSAFE_TICKER] = w_dyn.get(SMARTSAFE_TICKER, 0.0) + sweep * 0.70
        w_dyn[SMARTSAFE_ALT] = w_dyn.get(SMARTSAFE_ALT, 0.0) + sweep * 0.30
        w_dyn = _dedupe_and_normalize(w_dyn)

    # portfolio returns using normalized weights
    port_cols = [c for c in rets.columns if c in w_dyn]
    if not port_cols:
        return pd.DataFrame(columns=["wave_nav", "bm_nav", "wave_ret", "bm_ret"])

    w_vec = pd.Series({c: w_dyn[c] for c in port_cols})
    w_vec = w_vec / float(w_vec.sum()) if float(w_vec.sum()) > 0 else w_vec

    wave_ret = (rets[port_cols] * w_vec).sum(axis=1)

    # vol targeting scale (gross multiplier)
    vol_scale = _apply_vol_target_scale(wave_ret, mode)
    # vix exposure scale + mode cap
    vix_scale = _vix_exposure_scale(vix_last, mode)
    gross_cap = MODE_GROSS_CAP.get(mode, 1.0)
    gross = min(vol_scale * vix_scale, gross_cap)

    # Apply gross scaling to returns as a simple exposure model
    wave_ret = wave_ret * gross

    # benchmark returns (try composite; if missing data, fallback)
    bm_cols = [c for c in rets.columns if c in bm_w]
    if len(bm_cols) >= 1:
        bm_vec = pd.Series({c: bm_w[c] for c in bm_cols})
        bm_vec = bm_vec / float(bm_vec.sum()) if float(bm_vec.sum()) > 0 else bm_vec
        bm_ret = (rets[bm_cols] * bm_vec).sum(axis=1)
    else:
        fb_cols = [c for c in rets.columns if c in fb_w]
        if not fb_cols:
            bm_ret = pd.Series(0.0, index=rets.index)
        else:
            fb_vec = pd.Series({c: fb_w[c] for c in fb_cols})
            fb_vec = fb_vec / float(fb_vec.sum()) if float(fb_vec.sum()) > 0 else fb_vec
            bm_ret = (rets[fb_cols] * fb_vec).sum(axis=1)

    # Trim to requested window
    # Keep last `days` rows (trading days ~, but the console uses "days" as row count window)
    df = pd.DataFrame({"wave_ret": wave_ret, "bm_ret": bm_ret}).dropna(how="all").copy()
    if len(df) > days:
        df = df.iloc[-days:].copy()

    # NAV (normalized)
    df["wave_nav"] = (1.0 + df["wave_ret"].fillna(0.0)).cumprod()
    df["bm_nav"] = (1.0 + df["bm_ret"].fillna(0.0)).cumprod()

    # reorder columns
    df = df[["wave_nav", "bm_nav", "wave_ret", "bm_ret"]]
    return df