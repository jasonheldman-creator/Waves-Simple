# waves_engine.py — WAVES Intelligence™ Vector Engine
# Upgrade: Recommendations + safe apply controls + persistent overrides + full wave discovery
#
# Key goals:
#   ✅ Never lose waves: discover from wave_weights.csv + logs + internal fallback
#   ✅ Keep your existing compute_history_nav style (NAV + benchmark NAV + returns)
#   ✅ Provide wave_detail_bundle() used by app.py (df365 + attributions + diagnostics + recos)
#   ✅ Provide preview-first + optional persistent apply with hard guardrails + logging

from __future__ import annotations

import os
import json
import glob
from dataclasses import dataclass
from datetime import datetime, timedelta
from functools import lru_cache
from typing import Dict, List, Set, Tuple, Any, Optional

import numpy as np
import pandas as pd

try:
    import yfinance as yf
except Exception:
    yf = None


# ============================================================
# Global config
# ============================================================

TRADING_DAYS_PER_YEAR = 252

# Modes
MODE_BASE_EXPOSURE: Dict[str, float] = {
    "Standard": 1.00,
    "Alpha-Minus-Beta": 0.85,
    "Private Logic": 1.10,
}

MODE_EXPOSURE_CAPS: Dict[str, Tuple[float, float]] = {
    "Standard": (0.70, 1.30),
    "Alpha-Minus-Beta": (0.50, 1.00),
    "Private Logic": (0.80, 1.50),
}

# Base SmartSafe by mode (this can be overridden by regime/VIX gating)
MODE_SMARTSAFE_BASE: Dict[str, float] = {
    "Standard": 0.05,
    "Alpha-Minus-Beta": 0.20,
    "Private Logic": 0.03,
}

# Regime labels from SPY 60D
REGIME_EXPOSURE: Dict[str, float] = {
    "panic": 0.80,
    "downtrend": 0.90,
    "neutral": 1.00,
    "uptrend": 1.10,
}

REGIME_GATING: Dict[str, Dict[str, float]] = {
    "Standard": {
        "panic": 0.50,
        "downtrend": 0.30,
        "neutral": 0.10,
        "uptrend": 0.00,
    },
    "Alpha-Minus-Beta": {
        "panic": 0.75,
        "downtrend": 0.50,
        "neutral": 0.25,
        "uptrend": 0.05,
    },
    "Private Logic": {
        "panic": 0.40,
        "downtrend": 0.25,
        "neutral": 0.05,
        "uptrend": 0.00,
    },
}

PORTFOLIO_VOL_TARGET = 0.20
VIX_TICKER = "^VIX"
BTC_TICKER = "BTC-USD"

CRYPTO_YIELD_OVERLAY_APY: Dict[str, float] = {
    "Crypto Stable Yield Wave": 0.04,
    "Crypto Income & Yield Wave": 0.08,
    "Crypto High-Yield Income Wave": 0.12,
}

CRYPTO_WAVE_KEYWORD = "Crypto"

# Safe assets to use for SmartSafe returns
SAFE_CANDIDATES = [
    "SGOV", "BIL", "SHV", "SHY", "SUB", "SHM", "MUB", "IEF", "TLT", "ICSH",
    "USDC-USD", "USDT-USD", "DAI-USD", "USDP-USD"
]

# ============================================================
# Recommendations / Guardrails / Logging
# ============================================================

CONF_RANK = {"Low": 1, "Medium": 2, "High": 3}

SAFE_APPLY_LIMITS = {
    "min_confidence_to_apply": "Medium",

    "max_abs_exposure_delta": 0.10,
    "max_abs_smartsafe_delta": 0.10,

    "exposure_min": 0.00,
    "exposure_max": 1.25,
    "smartsafe_min": 0.00,
    "smartsafe_max": 0.90,

    "min_days_per_bucket": 20,
    "min_bp_per_day": 2.0,   # 2 bp/day
}

RECO_LOG_DIR = os.path.join("logs", "recommendations")
RECO_EVENTS_CSV = os.path.join(RECO_LOG_DIR, "reco_events.csv")
RECO_STATE_JSON = os.path.join(RECO_LOG_DIR, "persistent_overrides.json")


def _now_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _ensure_dirs(path: str):
    os.makedirs(path, exist_ok=True)


def _log_recommendation_event(evt: Dict[str, Any]) -> None:
    _ensure_dirs(RECO_LOG_DIR)
    df = pd.DataFrame([evt])
    if os.path.exists(RECO_EVENTS_CSV):
        df.to_csv(RECO_EVENTS_CSV, mode="a", header=False, index=False)
    else:
        df.to_csv(RECO_EVENTS_CSV, index=False)


def _key_wave_mode(wave: str, mode: str) -> str:
    return f"{wave}|{mode}"


def load_persistent_overrides() -> Dict[str, Any]:
    _ensure_dirs(RECO_LOG_DIR)
    if not os.path.exists(RECO_STATE_JSON):
        return {}
    try:
        with open(RECO_STATE_JSON, "r", encoding="utf-8") as f:
            return json.load(f) or {}
    except Exception:
        return {}


def save_persistent_overrides(state: Dict[str, Any]) -> None:
    _ensure_dirs(RECO_LOG_DIR)
    with open(RECO_STATE_JSON, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, sort_keys=True)


def persist_apply(
    wave: str,
    mode: str,
    new_exposure: float,
    new_smartsafe: float,
    *,
    reason: str,
    confidence: str,
    reco_id: str,
) -> None:
    state = load_persistent_overrides()
    k = _key_wave_mode(wave, mode)
    state[k] = {
        "exposure": float(new_exposure),
        "smartsafe": float(new_smartsafe),
        "meta": {"ts": _now_iso(), "reason": reason, "confidence": confidence, "reco_id": reco_id},
    }
    save_persistent_overrides(state)

    _log_recommendation_event({
        "ts": _now_iso(),
        "type": "persist_apply",
        "wave": wave,
        "mode": mode,
        "reco_id": reco_id,
        "confidence": confidence,
        "reason": reason,
        "new_exposure": float(new_exposure),
        "new_smartsafe": float(new_smartsafe),
    })


def persist_clear(wave: str, mode: str) -> None:
    state = load_persistent_overrides()
    k = _key_wave_mode(wave, mode)
    if k in state:
        old = state[k]
        del state[k]
        save_persistent_overrides(state)
        _log_recommendation_event({
            "ts": _now_iso(),
            "type": "persist_clear",
            "wave": wave,
            "mode": mode,
            "old_exposure": old.get("exposure"),
            "old_smartsafe": old.get("smartsafe"),
        })


def get_persistent_overrides_for(wave: str, mode: str) -> Dict[str, Any]:
    state = load_persistent_overrides()
    k = _key_wave_mode(wave, mode)
    v = state.get(k, {})
    return v if isinstance(v, dict) else {}


def _clamp(x: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, x)))


def _capped_delta(delta: float, cap: float) -> float:
    return float(max(-cap, min(cap, delta)))


def apply_recommendation_preview(
    wave: str,
    mode: str,
    current_exposure: float,
    current_smartsafe: float,
    deltas: Dict[str, float]
) -> Tuple[float, float, Dict[str, float]]:
    """
    Preview a recommendation with caps + bounds.
    Returns (new_exposure, new_smartsafe, applied_deltas)
    """
    exp_d = float(deltas.get("exposure_delta", 0.0))
    ss_d  = float(deltas.get("smartsafe_delta", 0.0))

    exp_d = _capped_delta(exp_d, SAFE_APPLY_LIMITS["max_abs_exposure_delta"])
    ss_d  = _capped_delta(ss_d,  SAFE_APPLY_LIMITS["max_abs_smartsafe_delta"])

    new_exp = _clamp(current_exposure + exp_d, SAFE_APPLY_LIMITS["exposure_min"], SAFE_APPLY_LIMITS["exposure_max"])
    new_ss  = _clamp(current_smartsafe + ss_d, SAFE_APPLY_LIMITS["smartsafe_min"], SAFE_APPLY_LIMITS["smartsafe_max"])

    applied = {"exposure_delta": new_exp - current_exposure, "smartsafe_delta": new_ss - current_smartsafe}
    return new_exp, new_ss, applied


# ============================================================
# Data structures
# ============================================================

@dataclass
class Holding:
    ticker: str
    weight: float
    name: str | None = None


# ============================================================
# Internal fallback holdings (only used if no wave_weights.csv)
# (This is safe fallback; your real lineup comes from wave_weights.csv.)
# ============================================================

WAVE_WEIGHTS_FALLBACK: Dict[str, List[Holding]] = {
    "S&P 500 Wave": [
        Holding("NVDA", 0.04), Holding("AAPL", 0.04), Holding("MSFT", 0.04), Holding("AMZN", 0.04),
        Holding("META", 0.04), Holding("GOOGL", 0.04), Holding("TSLA", 0.04), Holding("BRK-B", 0.04),
        Holding("JPM", 0.04), Holding("LLY", 0.04),
    ],
    "SmartSafe Wave": [Holding("BIL", 0.25), Holding("SHV", 0.25), Holding("SGOV", 0.25), Holding("ICSH", 0.25)],
}


# ============================================================
# wave_weights.csv loading + wave discovery
# ============================================================

_WEIGHTS_DF: Optional[pd.DataFrame] = None


def refresh_weights() -> None:
    """
    Loads wave_weights.csv into memory if present.
    Expected columns: wave, ticker, weight  (case-insensitive)
    """
    global _WEIGHTS_DF
    if not os.path.exists("wave_weights.csv"):
        _WEIGHTS_DF = None
        return

    df = pd.read_csv("wave_weights.csv")
    if df.empty:
        _WEIGHTS_DF = None
        return

    # normalize columns
    cols = {c.strip().lower(): c for c in df.columns}
    need = {"wave", "ticker", "weight"}
    if not need.issubset(set(cols.keys())):
        # try common alternatives
        # if the file is in 3-line block format, user must fix it to CSV with headers
        _WEIGHTS_DF = None
        return

    df = df.rename(columns={cols["wave"]: "wave", cols["ticker"]: "ticker", cols["weight"]: "weight"})
    df["wave"] = df["wave"].astype(str).str.strip()
    df["ticker"] = df["ticker"].astype(str).str.strip()
    df["weight"] = pd.to_numeric(df["weight"], errors="coerce").fillna(0.0)

    # clean tickers: BRK.B -> BRK-B for yfinance
    df["ticker"] = df["ticker"].str.replace("BRK.B", "BRK-B", regex=False)

    # group duplicates
    df = df.groupby(["wave", "ticker"], as_index=False)["weight"].sum()

    # drop zero/neg
    df = df[df["weight"] > 0].copy()

    _WEIGHTS_DF = df


def _weights_loaded() -> bool:
    return _WEIGHTS_DF is not None and not _WEIGHTS_DF.empty


def _get_wave_weights_df() -> pd.DataFrame:
    if _WEIGHTS_DF is None:
        refresh_weights()
    if _WEIGHTS_DF is None:
        return pd.DataFrame(columns=["wave", "ticker", "weight"])
    return _WEIGHTS_DF.copy()


def get_all_waves() -> List[str]:
    """
    Union of:
      - wave_weights.csv (preferred)
      - orphan waves from logs
      - fallback internal dictionary (never blocks)
    """
    waves = set()

    # 1) CSV
    df = _get_wave_weights_df()
    if not df.empty:
        waves.update(df["wave"].astype(str).str.strip().tolist())

    # 2) logs
    for p in glob.glob("logs/positions/*_positions_*.csv"):
        base = os.path.basename(p)
        if "_positions_" in base:
            waves.add(base.split("_positions_")[0])

    for p in glob.glob("logs/performance/*_performance_daily.csv"):
        base = os.path.basename(p)
        if base.endswith("_performance_daily.csv"):
            waves.add(base.replace("_performance_daily.csv", ""))

    # 3) fallback
    waves.update(WAVE_WEIGHTS_FALLBACK.keys())

    return sorted([w for w in waves if w and w.strip()])


def get_modes() -> List[str]:
    return list(MODE_BASE_EXPOSURE.keys())


def _normalize_weights_df(df: pd.DataFrame) -> pd.Series:
    if df is None or df.empty:
        return pd.Series(dtype=float)
    tmp = df.groupby("ticker", as_index=True)["weight"].sum().copy()
    total = float(tmp.sum())
    if total <= 0:
        return pd.Series(dtype=float)
    tmp = tmp / total
    return tmp


def _get_holdings_for_wave(wave_name: str) -> pd.Series:
    """
    Returns pd.Series indexed by ticker with normalized weights.
    """
    df = _get_wave_weights_df()
    if not df.empty:
        sub = df[df["wave"] == wave_name].copy()
        if not sub.empty:
            return _normalize_weights_df(sub)

    # fallback
    holdings = WAVE_WEIGHTS_FALLBACK.get(wave_name, [])
    if not holdings:
        return pd.Series(dtype=float)
    dff = pd.DataFrame([{"ticker": h.ticker, "weight": h.weight} for h in holdings])
    return _normalize_weights_df(dff)


def get_wave_holdings(wave_name: str) -> pd.DataFrame:
    w = _get_holdings_for_wave(wave_name)
    if w.empty:
        return pd.DataFrame(columns=["ticker", "weight"])
    out = w.sort_values(ascending=False).reset_index()
    out.columns = ["ticker", "weight"]
    return out


# ============================================================
# Benchmark logic (simple + robust)
# ============================================================

def _is_crypto_wave(wave_name: str) -> bool:
    n = wave_name.lower()
    return ("bitcoin" in n) or ("crypto" in n)


def _benchmark_for_wave(wave_name: str) -> pd.Series:
    """
    Simple, robust benchmark mapping:
      - Crypto/Bitcoin -> BTC-USD/ETH-USD mix
      - SmartSafe / cash -> SGOV/BIL
      - Otherwise -> SPY baseline, with QQQ tilt for tech waves
    """
    name = wave_name.lower()

    # crypto
    if _is_crypto_wave(wave_name):
        # Bitcoin Wave special-case
        if "bitcoin" in name:
            return pd.Series({"BTC-USD": 1.0})
        # broader crypto
        return pd.Series({"BTC-USD": 0.60, "ETH-USD": 0.40})

    # cash / safes
    if "smartsafe" in name or "cash" in name or "money market" in name or "ladder" in name:
        # muni ladder: SUB/SHM/MUB
        if "muni" in name or "tax-free" in name:
            return pd.Series({"SUB": 0.30, "SHM": 0.30, "MUB": 0.40})
        # treasury ladder
        if "treasury" in name or "t-bill" in name:
            return pd.Series({"BIL": 0.25, "SHY": 0.25, "IEF": 0.25, "TLT": 0.25})
        return pd.Series({"SGOV": 0.50, "BIL": 0.50})

    # tech-ish waves
    if "ai" in name or "cloud" in name or "software" in name or "quantum" in name or "compute" in name or "semis" in name:
        return pd.Series({"QQQ": 0.70, "SPY": 0.30})

    # default broad
    return pd.Series({"SPY": 1.0})


# ============================================================
# Price download
# ============================================================

def _download_history(tickers: List[str], days: int) -> pd.DataFrame:
    if yf is None:
        raise RuntimeError("yfinance is not available in this environment.")

    lookback_days = days + 260
    end = datetime.utcnow().date()
    start = end - timedelta(days=lookback_days)

    data = yf.download(
        tickers=sorted(list(set(tickers))),
        start=start.isoformat(),
        end=end.isoformat(),
        interval="1d",
        auto_adjust=True,
        progress=False,
        group_by="column",
    )

    if data is None or len(data) == 0:
        return pd.DataFrame()

    # handle multiindex
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
        data = data.iloc[-days:].copy()

    return data


def _regime_from_return(ret_60d: float) -> str:
    if ret_60d is None or np.isnan(ret_60d):
        return "neutral"
    if ret_60d <= -0.12:
        return "panic"
    if ret_60d <= -0.04:
        return "downtrend"
    if ret_60d < 0.06:
        return "neutral"
    return "uptrend"


def _vix_exposure_factor(vix_level: float, mode: str) -> float:
    if vix_level is None or np.isnan(vix_level) or vix_level <= 0:
        return 1.0
    if vix_level < 15:
        base = 1.15
    elif vix_level < 20:
        base = 1.05
    elif vix_level < 25:
        base = 0.95
    elif vix_level < 30:
        base = 0.85
    elif vix_level < 40:
        base = 0.75
    else:
        base = 0.60
    if mode == "Alpha-Minus-Beta":
        base -= 0.05
    elif mode == "Private Logic":
        base += 0.05
    return float(np.clip(base, 0.5, 1.3))


def _vix_safe_fraction(vix_level: float, mode: str) -> float:
    if vix_level is None or np.isnan(vix_level) or vix_level <= 0:
        return 0.0
    if vix_level < 18:
        base = 0.00
    elif vix_level < 24:
        base = 0.05
    elif vix_level < 30:
        base = 0.15
    elif vix_level < 40:
        base = 0.25
    else:
        base = 0.40
    if mode == "Alpha-Minus-Beta":
        base *= 1.5
    elif mode == "Private Logic":
        base *= 0.7
    return float(np.clip(base, 0.0, 0.8))


# ============================================================
# Core NAV computation
# ============================================================

def compute_history_nav(
    wave_name: str,
    mode: str = "Standard",
    days: int = 365,
    *,
    exposure_override: Optional[float] = None,
    smartsafe_override: Optional[float] = None,
) -> pd.DataFrame:
    """
    Returns DataFrame indexed by Date:
      wave_nav, bm_nav, wave_ret, bm_ret, alpha
    """
    if mode not in MODE_BASE_EXPOSURE:
        raise ValueError(f"Unknown mode: {mode}")

    wave_weights = _get_holdings_for_wave(wave_name)
    if wave_weights.empty:
        return pd.DataFrame(columns=["wave_nav", "bm_nav", "wave_ret", "bm_ret", "alpha"])

    bm_weights = _benchmark_for_wave(wave_name)
    bm_weights = bm_weights / bm_weights.sum()

    tickers_wave = list(wave_weights.index)
    tickers_bm = list(bm_weights.index)

    # regime base index
    base_index_ticker = "SPY"

    # pull all tickers + risk controls
    all_tickers = set(tickers_wave + tickers_bm + [base_index_ticker, VIX_TICKER, BTC_TICKER])
    all_tickers.update(SAFE_CANDIDATES)

    price_df = _download_history(sorted(all_tickers), days=days)
    if price_df.empty:
        return pd.DataFrame(columns=["wave_nav", "bm_nav", "wave_ret", "bm_ret", "alpha"])

    # returns
    ret_df = price_df.pct_change().fillna(0.0)

    # align weights to columns
    w_aligned = wave_weights.reindex(price_df.columns).fillna(0.0)
    b_aligned = bm_weights.reindex(price_df.columns).fillna(0.0)

    # benchmark return
    bm_ret = (ret_df * b_aligned).sum(axis=1)

    # regime series from SPY 60D
    if base_index_ticker in price_df.columns:
        idx_price = price_df[base_index_ticker]
    else:
        idx_price = price_df[price_df.columns[0]]
    idx_ret_60 = idx_price / idx_price.shift(60) - 1.0

    # VIX series: use BTC vol for crypto waves
    if _is_crypto_wave(wave_name) and BTC_TICKER in price_df.columns:
        btc_ret = price_df[BTC_TICKER].pct_change().fillna(0.0)
        vix_series = btc_ret.rolling(30).std() * np.sqrt(TRADING_DAYS_PER_YEAR) * 100.0
        vix_series = vix_series.ffill().bfill()
    else:
        vix_series = price_df[VIX_TICKER] if VIX_TICKER in price_df.columns else pd.Series(20.0, index=price_df.index)

    # safe ticker
    safe_ticker = None
    for t in SAFE_CANDIDATES:
        if t in price_df.columns:
            safe_ticker = t
            break
    if safe_ticker is None:
        safe_ticker = base_index_ticker

    safe_ret = ret_df[safe_ticker]

    # mode base
    base_exposure = MODE_BASE_EXPOSURE[mode]
    exp_min, exp_max = MODE_EXPOSURE_CAPS[mode]
    base_ss = MODE_SMARTSAFE_BASE.get(mode, 0.0)

    # persistent overrides (if caller didn't pass overrides)
    persist = get_persistent_overrides_for(wave_name, mode)
    if exposure_override is None and isinstance(persist, dict) and "exposure" in persist:
        exposure_override = float(persist["exposure"])
    if smartsafe_override is None and isinstance(persist, dict) and "smartsafe" in persist:
        smartsafe_override = float(persist["smartsafe"])

    # if overrides are set, clamp
    if exposure_override is not None:
        base_exposure = _clamp(float(exposure_override), SAFE_APPLY_LIMITS["exposure_min"], SAFE_APPLY_LIMITS["exposure_max"])
    if smartsafe_override is not None:
        base_ss = _clamp(float(smartsafe_override), SAFE_APPLY_LIMITS["smartsafe_min"], SAFE_APPLY_LIMITS["smartsafe_max"])

    # momentum tilts
    mom_60 = price_df / price_df.shift(60) - 1.0

    wave_ret_list: List[float] = []
    dates: List[pd.Timestamp] = []

    apy = CRYPTO_YIELD_OVERLAY_APY.get(wave_name, 0.0)
    daily_yield = apy / TRADING_DAYS_PER_YEAR if apy > 0 else 0.0

    for dt in ret_df.index:
        rets = ret_df.loc[dt]

        regime = _regime_from_return(idx_ret_60.get(dt, np.nan))
        regime_exposure = REGIME_EXPOSURE[regime]
        regime_gate = REGIME_GATING[mode][regime]

        vix_level = float(vix_series.get(dt, np.nan))
        vix_exposure = _vix_exposure_factor(vix_level, mode)
        vix_gate = _vix_safe_fraction(vix_level, mode)

        # momentum tilt weights
        mom_row = mom_60.loc[dt] if dt in mom_60.index else None
        if mom_row is not None:
            mom_clipped = mom_row.reindex(price_df.columns).fillna(0.0).clip(lower=-0.30, upper=0.30)
            tilt = 1.0 + 0.8 * mom_clipped
            eff_w = w_aligned * tilt
        else:
            eff_w = w_aligned.copy()

        eff_w = eff_w.clip(lower=0.0)
        total_risk_w = float(eff_w.sum())
        if total_risk_w > 0:
            risk_w = eff_w / total_risk_w
        else:
            risk_w = w_aligned.copy()

        port_risk_ret = float((rets * risk_w).sum())
        sret = float(safe_ret.loc[dt])

        # 20D realized vol targeting
        if len(wave_ret_list) >= 20:
            recent = np.array(wave_ret_list[-20:])
            recent_vol = recent.std() * np.sqrt(TRADING_DAYS_PER_YEAR)
        else:
            recent_vol = PORTFOLIO_VOL_TARGET

        vol_adjust = 1.0
        if recent_vol > 0:
            vol_adjust = PORTFOLIO_VOL_TARGET / recent_vol
            vol_adjust = float(np.clip(vol_adjust, 0.7, 1.3))

        raw_exposure = base_exposure * regime_exposure * vol_adjust * vix_exposure
        exposure = float(np.clip(raw_exposure, exp_min, exp_max))

        # SmartSafe mix
        safe_fraction = base_ss + regime_gate + vix_gate
        safe_fraction = float(np.clip(safe_fraction, 0.0, 0.95))
        risk_fraction = 1.0 - safe_fraction

        total_ret = safe_fraction * sret + risk_fraction * exposure * port_risk_ret

        # yield overlay for crypto income waves (not for Bitcoin Wave)
        if daily_yield != 0.0 and "bitcoin wave" not in wave_name.lower():
            total_ret += daily_yield

        # Private Logic mean reversion overlay
        if mode == "Private Logic" and len(wave_ret_list) >= 20:
            recent = np.array(wave_ret_list[-20:])
            dv = recent.std()
            if dv > 0:
                shock = 2.0 * dv
                if total_ret <= -shock:
                    total_ret = total_ret * 1.30
                elif total_ret >= shock:
                    total_ret = total_ret * 0.70

        wave_ret_list.append(float(total_ret))
        dates.append(dt)

    wave_ret = pd.Series(wave_ret_list, index=pd.Index(dates, name="Date"))
    bm_ret = bm_ret.reindex(wave_ret.index).fillna(0.0)

    wave_nav = (1.0 + wave_ret).cumprod()
    bm_nav = (1.0 + bm_ret).cumprod()
    alpha = wave_ret - bm_ret

    out = pd.DataFrame(
        {
            "wave_nav": wave_nav,
            "bm_nav": bm_nav,
            "wave_ret": wave_ret,
            "bm_ret": bm_ret,
            "alpha": alpha,
        }
    )
    out.index.name = "Date"
    return out


# ============================================================
# Summaries / bundles used by app.py
# ============================================================

def compute_multi_window_summary(wave: str, mode: str) -> Dict[str, Optional[float]]:
    """
    Returns dict of 1D/30D/60D/365D return and alpha
    """
    df = compute_history_nav(wave, mode=mode, days=365)
    if df is None or df.empty or len(df) < 2:
        return {
            "1D_return": None, "1D_alpha": None,
            "30D_return": None, "30D_alpha": None,
            "60D_return": None, "60D_alpha": None,
            "365D_return": None, "365D_alpha": None,
        }

    nav_w = df["wave_nav"]
    nav_b = df["bm_nav"]

    def _ret(nav: pd.Series, window: int) -> float:
        if len(nav) < 2:
            return np.nan
        w = min(window, len(nav))
        if w < 2:
            return np.nan
        return float(nav.iloc[-1] / nav.iloc[-w] - 1.0)

    def _alpha(window: int) -> float:
        rw = _ret(nav_w, window)
        rb = _ret(nav_b, window)
        if np.isnan(rw) or np.isnan(rb):
            return np.nan
        return float(rw - rb)

    # 1D uses last two points
    r1 = float(nav_w.iloc[-1] / nav_w.iloc[-2] - 1.0) if len(nav_w) >= 2 else np.nan
    a1 = float((nav_w.iloc[-1] / nav_w.iloc[-2] - 1.0) - (nav_b.iloc[-1] / nav_b.iloc[-2] - 1.0)) if len(nav_w) >= 2 else np.nan

    r30 = _ret(nav_w, 30)
    a30 = _alpha(30)

    r60 = _ret(nav_w, 60)
    a60 = _alpha(60)

    r365 = _ret(nav_w, len(nav_w))
    a365 = _alpha(len(nav_w))

    return {
        "1D_return": r1 if not np.isnan(r1) else None,
        "1D_alpha": a1 if not np.isnan(a1) else None,
        "30D_return": r30 if not np.isnan(r30) else None,
        "30D_alpha": a30 if not np.isnan(a30) else None,
        "60D_return": r60 if not np.isnan(r60) else None,
        "60D_alpha": a60 if not np.isnan(a60) else None,
        "365D_return": r365 if not np.isnan(r365) else None,
        "365D_alpha": a365 if not np.isnan(a365) else None,
    }


def _vol_regime_from_vix(v: float) -> str:
    if v is None or np.isnan(v):
        return "Unknown"
    if v < 18:
        return "Low"
    if v < 28:
        return "Medium"
    if v < 40:
        return "High"
    return "Stress"


def _trend_from_spy_60d(spy_ret_60d: float) -> str:
    if spy_ret_60d is None or np.isnan(spy_ret_60d):
        return "Unknown"
    if spy_ret_60d >= 0.06:
        return "Uptrend"
    if spy_ret_60d <= -0.04:
        return "Downtrend"
    return "Flat"


def _diagnostics_from_df(df: pd.DataFrame, holdings: pd.DataFrame) -> List[Dict[str, Any]]:
    diags: List[Dict[str, Any]] = []
    if df is None or df.empty or len(df) < 30:
        diags.append({"level": "WARN", "code": "NO_HISTORY", "msg": "Insufficient history for reliable attribution."})
        return diags

    # concentration checks
    if holdings is not None and not holdings.empty:
        top1 = float(holdings["weight"].iloc[0])
        top3 = float(holdings["weight"].iloc[:3].sum())
        if top1 >= 0.45:
            diags.append({"level": "WARN", "code": "TOP1_CONC", "msg": f"Top holding concentration is {top1:.0%}."})
        if top3 >= 0.75:
            diags.append({"level": "WARN", "code": "TOP3_CONC", "msg": f"Top-3 concentration is {top3:.0%}."})

    # alpha instability check
    alpha = df["alpha"].dropna()
    if len(alpha) >= 60:
        vol_alpha = float(alpha.std() * np.sqrt(TRADING_DAYS_PER_YEAR))
        if vol_alpha > 0.20:
            diags.append({"level": "WARN", "code": "ALPHA_VOL", "msg": f"Alpha volatility is high (~{vol_alpha:.0%} annualized)."})
        else:
            diags.append({"level": "PASS", "code": "ALPHA_VOL_OK", "msg": "Alpha volatility looks controlled."})
    else:
        diags.append({"level": "INFO", "code": "ALPHA_VOL_NA", "msg": "Not enough history for alpha-vol stability test."})

    return diags


def _conditional_grid(df: pd.DataFrame) -> pd.DataFrame:
    """
    Regime × Trend grid using:
      - vol regime from VIX (or BTC-vol proxy already embedded in df? we recompute via yfinance)
      - trend from SPY 60D
    We compute with SPY and VIX pulled again (lightweight).
    """
    if df is None or df.empty or len(df) < 60 or yf is None:
        return pd.DataFrame()

    idx = df.index
    end = datetime.utcnow().date()
    start = end - timedelta(days=365 + 30)

    mk = yf.download(
        tickers=["SPY", "^VIX"],
        start=start.isoformat(),
        end=end.isoformat(),
        interval="1d",
        auto_adjust=True,
        progress=False
    )

    if mk is None or len(mk) == 0:
        return pd.DataFrame()

    # normalize columns
    if isinstance(mk.columns, pd.MultiIndex):
        if "Adj Close" in mk.columns.get_level_values(0):
            mk = mk["Adj Close"]
        elif "Close" in mk.columns.get_level_values(0):
            mk = mk["Close"]
    if isinstance(mk.columns, pd.MultiIndex):
        mk = mk.droplevel(0, axis=1)

    mk = mk.sort_index().ffill().bfill()

    spy = mk["SPY"].reindex(idx).ffill().bfill()
    vix = mk["^VIX"].reindex(idx).ffill().bfill()

    spy_ret_60 = spy / spy.shift(60) - 1.0

    tmp = pd.DataFrame({
        "alpha": df["alpha"].values,
        "wave_ret": df["wave_ret"].values,
        "bm_ret": df["bm_ret"].values,
        "vix": vix.values,
        "spy_ret_60": spy_ret_60.values,
    }, index=idx)

    tmp["regime"] = tmp["vix"].map(_vol_regime_from_vix)
    tmp["trend"] = tmp["spy_ret_60"].map(_trend_from_spy_60d)

    def agg(g: pd.DataFrame) -> pd.Series:
        days = int(g.shape[0])
        mean_alpha = float(g["alpha"].mean()) if days else np.nan
        cum_alpha = float((1.0 + g["alpha"]).prod() - 1.0) if days else np.nan
        return pd.Series({
            "days": days,
            "mean_daily_alpha": mean_alpha,
            "cum_alpha": cum_alpha,
        })

    out = tmp.groupby(["regime", "trend"], as_index=False).apply(agg).reset_index(drop=True)
    return out


def _volatility_regime_attribution(df: pd.DataFrame) -> pd.DataFrame:
    """
    Volatility regime attribution table: regime -> wave_ret, bm_ret, alpha
    """
    if df is None or df.empty or len(df) < 60 or yf is None:
        return pd.DataFrame()

    idx = df.index
    end = datetime.utcnow().date()
    start = end - timedelta(days=365 + 30)

    mk = yf.download(
        tickers=["^VIX"],
        start=start.isoformat(),
        end=end.isoformat(),
        interval="1d",
        auto_adjust=True,
        progress=False
    )

    if mk is None or len(mk) == 0:
        return pd.DataFrame()

    if isinstance(mk.columns, pd.MultiIndex):
        if "Adj Close" in mk.columns.get_level_values(0):
            mk = mk["Adj Close"]
        elif "Close" in mk.columns.get_level_values(0):
            mk = mk["Close"]
    if isinstance(mk.columns, pd.MultiIndex):
        mk = mk.droplevel(0, axis=1)

    mk = mk.sort_index().ffill().bfill()
    vix = mk["^VIX"].reindex(idx).ffill().bfill()

    tmp = pd.DataFrame({
        "wave_ret": df["wave_ret"].values,
        "bm_ret": df["bm_ret"].values,
        "alpha": df["alpha"].values,
        "vix": vix.values
    }, index=idx)

    tmp["regime"] = tmp["vix"].map(_vol_regime_from_vix)

    out = tmp.groupby("regime", as_index=False).agg(
        days=("alpha", "count"),
        wave_ret=("wave_ret", "mean"),
        bm_ret=("bm_ret", "mean"),
        alpha=("alpha", "mean"),
    )
    return out.sort_values("days", ascending=False)


def _recommendations_from_grid(cond_grid: pd.DataFrame, style: str = "Balanced") -> List[Dict[str, Any]]:
    """
    Generates recos from conditional grid.
    style affects delta magnitude:
      Conservative: 0.7x
      Balanced: 1.0x
      Aggressive: 1.3x
    """
    recos: List[Dict[str, Any]] = []
    if cond_grid is None or cond_grid.empty:
        return recos

    mult = {"Conservative": 0.7, "Balanced": 1.0, "Aggressive": 1.3}.get(style, 1.0)

    g = cond_grid.copy()
    need = {"regime", "trend", "days", "mean_daily_alpha"}
    if not need.issubset(set(g.columns)):
        return recos

    benign = g[(g["days"] >= SAFE_APPLY_LIMITS["min_days_per_bucket"]) &
               (g["regime"].astype(str).str.lower().isin(["low", "medium"])) &
               (g["trend"].astype(str).str.lower().isin(["uptrend", "flat"]))].copy()

    if not benign.empty:
        bp_day = float(benign["mean_daily_alpha"].mean()) * 10000.0
        if bp_day >= SAFE_APPLY_LIMITS["min_bp_per_day"]:
            raw = (0.02 + min(0.06, bp_day / 200.0)) * mult
            recos.append({
                "id": "raise_exposure_benign",
                "title": "Increase exposure slightly in benign regimes",
                "confidence": "Medium" if bp_day < 6 else "High",
                "why": f"Low/Medium + Uptrend/Flat mean daily alpha ≈ {bp_day:.1f} bp/day "
                       f"over {int(benign['days'].sum())} days.",
                "deltas": {"exposure_delta": float(raw), "smartsafe_delta": 0.0},
            })

    stress = g[(g["days"] >= SAFE_APPLY_LIMITS["min_days_per_bucket"]) &
               (g["regime"].astype(str).str.lower().isin(["high", "stress"]))].copy()

    if not stress.empty:
        bp_day = float(stress["mean_daily_alpha"].mean()) * 10000.0
        if bp_day < -SAFE_APPLY_LIMITS["min_bp_per_day"]:
            recos.append({
                "id": "defend_in_stress",
                "title": "Add SmartSafe + reduce exposure in high-vol regimes",
                "confidence": "Medium" if bp_day > -8 else "High",
                "why": f"High/Stress regimes mean daily alpha ≈ {bp_day:.1f} bp/day "
                       f"over {int(stress['days'].sum())} days.",
                "deltas": {"exposure_delta": float(-0.05 * mult), "smartsafe_delta": float(+0.05 * mult)},
            })

    return recos


def wave_detail_bundle(
    wave_name: str,
    mode: str,
    *,
    exposure_override: Optional[float] = None,
    smartsafe_override: Optional[float] = None,
    reco_style: str = "Balanced",
) -> Dict[str, Any]:
    """
    Returns dict used by app.py:
      - df365: DataFrame(date,wave_nav,bm_nav,alpha,...) or None
      - holdings: top 10 holdings DataFrame(ticker,weight)
      - vol_attr: volatility regime attribution table
      - cond_grid: conditional grid (regime x trend)
      - cond_log_path: path logged to
      - diagnostics: list of dicts
      - recommendations: list of dicts
    """
    # holdings
    holds = get_wave_holdings(wave_name)
    holds_top10 = holds.sort_values("weight", ascending=False).head(10).copy()

    # NAV history
    df = compute_history_nav(
        wave_name,
        mode=mode,
        days=365,
        exposure_override=exposure_override,
        smartsafe_override=smartsafe_override,
    )

    df365_out = None
    if df is not None and not df.empty and len(df) >= 2:
        df365_out = df.copy()
        df365_out = df365_out.reset_index().rename(columns={"Date": "date"})

    # attribution tables
    vol_attr = _volatility_regime_attribution(df) if df is not None and not df.empty else pd.DataFrame()
    cond_grid = _conditional_grid(df) if df is not None and not df.empty else pd.DataFrame()

    # persistent logging of conditional grid
    cond_log_path = ""
    if cond_grid is not None and not cond_grid.empty:
        _ensure_dirs(os.path.join("logs", "attribution"))
        cond_log_path = os.path.join("logs", "attribution", f"{wave_name.replace('/','_')}_{mode.replace(' ','_')}_conditional.csv")
        try:
            cond_grid.to_csv(cond_log_path, index=False)
        except Exception:
            cond_log_path = ""

    diagnostics = _diagnostics_from_df(df, holds) if df is not None else [{"level": "WARN", "code": "NO_DATA", "msg": "No data."}]

    # recommendations from grid + diagnostics
    recos = _recommendations_from_grid(cond_grid, style=reco_style)

    # if diagnostics show concentration, add action-only reco (no deltas)
    diag_codes = {d.get("code") for d in diagnostics if d.get("level") == "WARN"}
    if "TOP1_CONC" in diag_codes or "TOP3_CONC" in diag_codes:
        recos.append({
            "id": "reduce_concentration_action",
            "title": "Reduce concentration risk (weights/holdings action)",
            "confidence": "High",
            "why": "Diagnostics flag high concentration. Consider broadening weights or capping top positions.",
            "deltas": {},
            "action_only": True
        })

    bundle = {
        "df365": df365_out,
        "holdings": holds_top10.rename(columns={"ticker": "ticker", "weight": "weight"}),
        "vol_attr": vol_attr,
        "cond_grid": cond_grid,
        "cond_log_path": cond_log_path,
        "diagnostics": diagnostics,
        "recommendations": recos,
    }
    return bundle