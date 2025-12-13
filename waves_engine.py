# waves_engine.py — WAVES Intelligence™ Vector Engine
# ✅ 20-Wave enforcement via wave_weights.csv
# ✅ Dynamic Strategy + VIX + SmartSafe + Auto-Custom Benchmarks
#
# Public API used by app.py:
#   • USE_FULL_WAVE_HISTORY
#   • get_all_waves()
#   • get_modes()
#   • compute_history_nav(wave_name, mode, days)
#   • get_benchmark_mix_table()
#   • get_wave_holdings(wave_name)

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from functools import lru_cache
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd

try:
    import yfinance as yf
except ImportError:  # pragma: no cover
    yf = None


# ------------------------------------------------------------
# Global config
# ------------------------------------------------------------

USE_FULL_WAVE_HISTORY: bool = False  # placeholder flag; kept for compatibility
TRADING_DAYS_PER_YEAR = 252

# ✅ REQUIRED 20-WAVE ROSTER (hard-enforced)
REQUIRED_WAVES: List[str] = [
    "AI & Cloud MegaCap Wave",
    "AI Wave",
    "Bitcoin Wave",
    "Clean Transit-Infrastructure Wave",
    "Cloud & Software Wave",
    "Crypto High-Yield Income Wave",
    "Crypto Income & Yield Wave",
    "Crypto Income Wave",
    "Crypto Stable Yield Wave",
    "Demas Fund Wave",
    "EV & Infrastructure Wave",
    "Future Energy & EV Wave",
    "Future Power & Energy Wave",
    "Gold Wave",
    "Growth Wave",
    "Income Wave",
    "Quantum Computing Wave",
    "S&P 500 Wave",
    "Small Cap Growth Wave",
    "SmartSafe Wave",
]

WAVE_WEIGHTS_CSV_PATH = "wave_weights.csv"

# Mode risk appetites & exposure caps
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

# Regime → additional exposure tilt (from SPY 60D trend)
REGIME_EXPOSURE: Dict[str, float] = {
    "panic": 0.80,
    "downtrend": 0.90,
    "neutral": 1.00,
    "uptrend": 1.10,
}

# Regime & mode → baseline SmartSafe gating fraction (portion in safe asset)
REGIME_GATING: Dict[str, Dict[str, float]] = {
    "Standard": {"panic": 0.50, "downtrend": 0.30, "neutral": 0.10, "uptrend": 0.00},
    "Alpha-Minus-Beta": {"panic": 0.75, "downtrend": 0.50, "neutral": 0.25, "uptrend": 0.05},
    "Private Logic": {"panic": 0.40, "downtrend": 0.25, "neutral": 0.05, "uptrend": 0.00},
}

PORTFOLIO_VOL_TARGET = 0.20  # ~20% annualized (default)

VIX_TICKER = "^VIX"
BTC_TICKER = "BTC-USD"  # used for crypto-VIX proxy

# Crypto yield overlays (APY assumptions per Wave)
CRYPTO_YIELD_OVERLAY_APY: Dict[str, float] = {
    "Crypto Stable Yield Wave": 0.04,
    "Crypto Income & Yield Wave": 0.08,
    "Crypto High-Yield Income Wave": 0.12,
    # NOTE: "Crypto Income Wave" intentionally no overlay unless you want it
}

CRYPTO_WAVE_KEYWORD = "Crypto"


# ------------------------------------------------------------
# Data structures
# ------------------------------------------------------------

@dataclass
class Holding:
    ticker: str
    weight: float
    name: str | None = None


@dataclass
class ETFBenchmarkCandidate:
    ticker: str
    name: str
    sector_tags: Set[str]
    cap_style: str  # "Mega", "Large", "Mid", "Small", "Crypto", "Safe", "Broad", "Gold"


# ------------------------------------------------------------
# CSV loader (source of truth)
# ------------------------------------------------------------

def _normalize_ticker(t: str) -> str:
    t = (t or "").strip()
    u = t.upper()
    # yfinance expects BRK-B not BRK.B
    if u == "BRK.B":
        return "BRK-B"
    return t


def _load_wave_weights_csv(path: str = WAVE_WEIGHTS_CSV_PATH) -> Dict[str, List[Holding]]:
    """
    Load wave_weights.csv with columns: wave,ticker,weight

    Enforces REQUIRED_WAVES to exist. If any are missing, they will be created
    with a safe placeholder holding (SPY 1.0) so the console NEVER drops waves.
    """
    try:
        df = pd.read_csv(path)
    except Exception:
        df = pd.DataFrame(columns=["wave", "ticker", "weight"])

    if df.empty:
        waves: Dict[str, List[Holding]] = {}
    else:
        df = df.rename(columns={c: c.strip().lower() for c in df.columns})
        # accept common variants
        if "wave" not in df.columns and "name" in df.columns:
            df["wave"] = df["name"]
        if "ticker" not in df.columns and "symbol" in df.columns:
            df["ticker"] = df["symbol"]

        df = df[["wave", "ticker", "weight"]].copy()
        df["wave"] = df["wave"].astype(str).str.strip()
        df["ticker"] = df["ticker"].astype(str).str.strip().apply(_normalize_ticker)
        df["weight"] = pd.to_numeric(df["weight"], errors="coerce").fillna(0.0)

        # drop junk rows
        df = df[(df["wave"] != "") & (df["ticker"] != "") & (df["weight"] > 0)]

        waves = {}
        for wave, g in df.groupby("wave", sort=False):
            holdings = [Holding(row["ticker"], float(row["weight"]), None) for _, row in g.iterrows()]
            waves[wave] = holdings

    # ✅ enforce the 20-wave roster (never drop)
    for w in REQUIRED_WAVES:
        if w not in waves or not waves[w]:
            waves[w] = [Holding("SPY", 1.0, "SPDR S&P 500 ETF (placeholder)")]

    return waves


# The runtime holdings dictionary is ALWAYS the CSV version (with enforcement)
WAVE_WEIGHTS: Dict[str, List[Holding]] = _load_wave_weights_csv()


# ------------------------------------------------------------
# Static benchmarks (fallback / overrides)
# ------------------------------------------------------------

BENCHMARK_WEIGHTS_STATIC: Dict[str, List[Holding]] = {
    "S&P 500 Wave": [Holding("SPY", 1.0, "SPDR S&P 500 ETF")],
    "Growth Wave": [Holding("QQQ", 1.0, "Invesco QQQ Trust")],
    "Income Wave": [Holding("VYM", 0.5, "Vanguard High Dividend Yield ETF"), Holding("HDV", 0.5, "iShares Core High Dividend ETF")],
    "SmartSafe Wave": [Holding("SGOV", 0.5, "iShares 0-3 Month Treasury Bond ETF"), Holding("BIL", 0.5, "SPDR Bloomberg 1-3 Month T-Bill ETF")],
    "Bitcoin Wave": [Holding("BTC-USD", 1.0, "Bitcoin")],
    "Gold Wave": [Holding("GLD", 0.5, "SPDR Gold Shares"), Holding("IAU", 0.5, "iShares Gold Trust")],
    "Demas Fund Wave": [Holding("SPY", 0.6, "SPDR S&P 500 ETF"), Holding("VTV", 0.4, "Vanguard Value ETF")],
}


# ------------------------------------------------------------
# ETF / Crypto benchmark candidate library
# ------------------------------------------------------------

ETF_CANDIDATES: List[ETFBenchmarkCandidate] = [
    ETFBenchmarkCandidate("SPY", "SPDR S&P 500 ETF", {"Broad", "Large", "Mega"}, "Large"),
    ETFBenchmarkCandidate("QQQ", "Invesco QQQ Trust", {"Tech", "Growth", "Mega"}, "Mega"),
    ETFBenchmarkCandidate("VGT", "Vanguard Information Technology ETF", {"Tech"}, "Large"),
    ETFBenchmarkCandidate("XLK", "Technology Select Sector SPDR", {"Tech"}, "Large"),
    ETFBenchmarkCandidate("SMH", "VanEck Semiconductor ETF", {"Tech", "Semis"}, "Large"),
    ETFBenchmarkCandidate("SOXX", "iShares Semiconductor ETF", {"Tech", "Semis"}, "Large"),
    ETFBenchmarkCandidate("IGV", "iShares Expanded Tech-Software Sector ETF", {"Tech", "Software"}, "Large"),
    ETFBenchmarkCandidate("WCLD", "WisdomTree Cloud Computing Fund", {"Tech", "Software", "Cloud"}, "Mid"),
    ETFBenchmarkCandidate("XLE", "Energy Select Sector SPDR Fund", {"Energy"}, "Large"),
    ETFBenchmarkCandidate("ICLN", "iShares Global Clean Energy ETF", {"Energy", "Clean"}, "Mid"),
    ETFBenchmarkCandidate("PAVE", "Global X U.S. Infrastructure Development ETF", {"Industrials", "Infrastructure"}, "Mid"),
    ETFBenchmarkCandidate("XLI", "Industrial Select Sector SPDR Fund", {"Industrials"}, "Large"),
    ETFBenchmarkCandidate("IWO", "iShares Russell 2000 Growth ETF", {"Small", "Growth"}, "Small"),
    ETFBenchmarkCandidate("VBK", "Vanguard Small-Cap Growth ETF", {"Small", "Growth"}, "Small"),
    ETFBenchmarkCandidate("IWP", "iShares Russell Mid-Cap Growth ETF", {"Mid", "Growth"}, "Mid"),
    ETFBenchmarkCandidate("MDY", "SPDR S&P MidCap 400 ETF Trust", {"Mid"}, "Mid"),
    ETFBenchmarkCandidate("BITO", "ProShares Bitcoin Strategy ETF", {"Crypto"}, "Crypto"),
    ETFBenchmarkCandidate("BIL", "SPDR Bloomberg 1-3 Month T-Bill ETF", {"Safe"}, "Safe"),
    ETFBenchmarkCandidate("SGOV", "iShares 0-3 Month Treasury Bond ETF", {"Safe"}, "Safe"),
    ETFBenchmarkCandidate("SUB", "iShares Short-Term National Muni Bond ETF", {"Safe"}, "Safe"),
    ETFBenchmarkCandidate("SHM", "SPDR Nuveen Short-Term Municipal Bond ETF", {"Safe"}, "Safe"),
    ETFBenchmarkCandidate("MUB", "iShares National Muni Bond ETF", {"Safe"}, "Safe"),
    ETFBenchmarkCandidate("GLD", "SPDR Gold Shares", {"Gold", "Safe"}, "Gold"),
    ETFBenchmarkCandidate("IAU", "iShares Gold Trust", {"Gold", "Safe"}, "Gold"),
    ETFBenchmarkCandidate("VTV", "Vanguard Value ETF", {"Broad", "Large"}, "Large"),
]


# ------------------------------------------------------------
# Public API
# ------------------------------------------------------------

def get_all_waves() -> list[str]:
    """
    ✅ Always return exactly the REQUIRED_WAVES roster (in that order),
    as long as they exist in WAVE_WEIGHTS (they will, due to enforcement).
    """
    return list(REQUIRED_WAVES)


def get_modes() -> list[str]:
    return list(MODE_BASE_EXPOSURE.keys())


def _normalize_weights(holdings: List[Holding]) -> pd.Series:
    if not holdings:
        return pd.Series(dtype=float)
    df = pd.DataFrame([{"ticker": _normalize_ticker(h.ticker), "weight": h.weight} for h in holdings])
    df = df.groupby("ticker", as_index=False)["weight"].sum()
    total = float(df["weight"].sum())
    if total <= 0:
        return pd.Series(dtype=float)
    df["weight"] = df["weight"] / total
    return df.set_index("ticker")["weight"]


def _download_history(tickers: list[str], days: int) -> pd.DataFrame:
    if yf is None:
        raise RuntimeError("yfinance is not available in this environment.")
    lookback_days = days + 260
    end = datetime.utcnow().date()
    start = end - timedelta(days=lookback_days)
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
    return data


def _map_sector_name(raw_sector: str | None) -> str:
    if not raw_sector:
        return "Unknown"
    s = raw_sector.lower()
    if "information technology" in s or "technology" in s:
        return "Tech"
    if "semiconductor" in s:
        return "Semis"
    if "software" in s:
        return "Software"
    if "energy" in s:
        return "Energy"
    if "industrial" in s:
        return "Industrials"
    if "real estate" in s:
        return "RealEstate"
    if "financial" in s:
        return "Financials"
    if "health" in s:
        return "HealthCare"
    if "communication" in s:
        return "Comm"
    if "consumer" in s:
        return "Consumer"
    return "Other"


def _cap_style_from_mcap(mcap: float | None) -> str:
    if mcap is None or np.isnan(mcap) or mcap <= 0:
        return "Unknown"
    if mcap >= 2e11:
        return "Mega"
    if mcap >= 2e10:
        return "Large"
    if mcap >= 5e9:
        return "Mid"
    return "Small"


@lru_cache(maxsize=256)
def _get_ticker_meta(ticker: str) -> tuple[str, float]:
    ticker = _normalize_ticker(ticker)
    if ticker.endswith("-USD"):
        if ticker in {"USDC-USD", "USDT-USD", "DAI-USD", "USDP-USD", "sDAI-USD"}:
            return ("Safe", np.nan)
        return ("Crypto", np.nan)
    if ticker in {"BIL", "SGOV", "SHV", "SHY", "SUB", "SHM", "MUB", "IEF", "TLT", "LQD", "ICSH"}:
        return ("Safe", np.nan)
    if ticker in {"GLD", "IAU"}:
        return ("Gold", np.nan)
    if yf is None:
        return ("Unknown", np.nan)
    try:
        info = yf.Ticker(ticker).info
    except Exception:
        return ("Unknown", np.nan)
    sector = info.get("sector")
    mcap = info.get("marketCap")
    return (_map_sector_name(sector), float(mcap) if mcap is not None else np.nan)


def _derive_wave_exposure(wave_name: str) -> tuple[Dict[str, float], str]:
    holdings = WAVE_WEIGHTS.get(wave_name, [])
    if not holdings:
        return {}, "Unknown"
    weights = _normalize_weights(holdings)
    sector_weights: Dict[str, float] = {}
    cap_votes: Dict[str, float] = {}
    for h in holdings:
        t = _normalize_ticker(h.ticker)
        if t not in weights.index:
            continue
        w = float(weights[t])
        sector, mcap = _get_ticker_meta(t)
        sector_weights[sector] = sector_weights.get(sector, 0.0) + w
        if sector in {"Crypto", "Safe", "Gold"}:
            style = sector
        else:
            style = _cap_style_from_mcap(mcap)
        cap_votes[style] = cap_votes.get(style, 0.0) + w
    total = sum(sector_weights.values())
    if total > 0:
        for k in list(sector_weights.keys()):
            sector_weights[k] /= total
    cap_style = max(cap_votes.items(), key=lambda kv: kv[1])[0] if cap_votes else "Unknown"
    return sector_weights, cap_style


def _score_etf_candidate(etf: ETFBenchmarkCandidate, sector_weights: Dict[str, float], cap_style: str) -> float:
    score = 0.0
    for s, w in sector_weights.items():
        if s in etf.sector_tags:
            score += w
        if s == "Tech" and "Tech" in etf.sector_tags:
            score += 0.3 * w
        if s == "Energy" and "Energy" in etf.sector_tags:
            score += 0.3 * w
        if s == "Industrials" and "Industrials" in etf.sector_tags:
            score += 0.3 * w
        if s == "Crypto" and "Crypto" in etf.sector_tags:
            score += 0.5 * w
        if s == "Safe" and "Safe" in etf.sector_tags:
            score += 0.5 * w
        if s == "Gold" and "Gold" in etf.sector_tags:
            score += 0.5 * w
    if cap_style == etf.cap_style:
        score += 0.10
    elif cap_style in {"Mega", "Large"} and etf.cap_style in {"Mega", "Large"}:
        score += 0.05
    elif cap_style in {"Mid", "Small"} and etf.cap_style in {"Mid", "Small"}:
        score += 0.05
    return score


@lru_cache(maxsize=64)
def get_auto_benchmark_holdings(wave_name: str) -> List[Holding]:
    # Prefer explicit static benchmark if defined
    if wave_name in BENCHMARK_WEIGHTS_STATIC:
        return BENCHMARK_WEIGHTS_STATIC[wave_name]

    sector_weights, cap_style = _derive_wave_exposure(wave_name)
    if not sector_weights:
        return [Holding("SPY", 1.0, "SPDR S&P 500 ETF")]

    scores = []
    for etf in ETF_CANDIDATES:
        s = _score_etf_candidate(etf, sector_weights, cap_style)
        if s > 0.0:
            scores.append((etf, s))
    if not scores:
        return [Holding("SPY", 1.0, "SPDR S&P 500 ETF")]

    scores.sort(key=lambda x: x[1], reverse=True)
    top = scores[:4]

    if len(top) == 1:
        etf, _ = top[0]
        return [Holding(etf.ticker, 1.0, etf.name)]

    total_score = sum(s for _, s in top)
    if total_score <= 0:
        return [Holding("SPY", 1.0, "SPDR S&P 500 ETF")]

    holdings: List[Holding] = []
    for etf, s in top:
        w = float(s / total_score)
        holdings.append(Holding(etf.ticker, w, etf.name))
    return holdings


def _regime_from_return(ret_60d: float) -> str:
    if np.isnan(ret_60d):
        return "neutral"
    if ret_60d <= -0.12:
        return "panic"
    if ret_60d <= -0.04:
        return "downtrend"
    if ret_60d < 0.06:
        return "neutral"
    return "uptrend"


def _vix_exposure_factor(vix_level: float, mode: str) -> float:
    if np.isnan(vix_level) or vix_level <= 0:
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
    if np.isnan(vix_level) or vix_level <= 0:
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


# ------------------------------------------------------------
# Core compute_history_nav
# ------------------------------------------------------------

def compute_history_nav(wave_name: str, mode: str = "Standard", days: int = 365) -> pd.DataFrame:
    """
    Compute Wave & Benchmark NAV + daily returns over a given window.

    Returns DataFrame indexed by Date:
        ['wave_nav', 'bm_nav', 'wave_ret', 'bm_ret']
    """
    if wave_name not in WAVE_WEIGHTS:
        raise ValueError(f"Unknown Wave: {wave_name}")
    if mode not in MODE_BASE_EXPOSURE:
        raise ValueError(f"Unknown mode: {mode}")

    wave_holdings = WAVE_WEIGHTS[wave_name]

    # Use static benchmark if provided, else auto-benchmark
    bm_holdings = BENCHMARK_WEIGHTS_STATIC.get(wave_name)
    if not bm_holdings:
        bm_holdings = get_auto_benchmark_holdings(wave_name)

    wave_weights = _normalize_weights(wave_holdings)
    bm_weights = _normalize_weights(bm_holdings)

    tickers_wave = list(wave_weights.index)
    tickers_bm = list(bm_weights.index)

    base_index_ticker = "SPY"
    safe_candidates = ["SGOV", "BIL", "SHV", "ICSH", "SHY", "SUB", "SHM", "MUB", "USDC-USD", "USDT-USD", "DAI-USD", "USDP-USD"]

    all_tickers = set(tickers_wave + tickers_bm)
    all_tickers.add(base_index_ticker)
    all_tickers.add(VIX_TICKER)
    all_tickers.add(BTC_TICKER)
    all_tickers.update(safe_candidates)

    all_tickers = sorted({ _normalize_ticker(t) for t in all_tickers if t })
    if not all_tickers:
        return pd.DataFrame(columns=["wave_nav", "bm_nav", "wave_ret", "bm_ret"], dtype=float)

    price_df = _download_history(all_tickers, days=days)
    if price_df.empty:
        return pd.DataFrame(columns=["wave_nav", "bm_nav", "wave_ret", "bm_ret"], dtype=float)
    if len(price_df) > days:
        price_df = price_df.iloc[-days:]

    ret_df = price_df.pct_change().fillna(0.0)

    wave_weights_aligned = wave_weights.reindex(price_df.columns).fillna(0.0)
    bm_weights_aligned = bm_weights.reindex(price_df.columns).fillna(0.0)

    bm_ret_series = (ret_df * bm_weights_aligned).sum(axis=1)

    # Base index for regime detection
    if base_index_ticker in price_df.columns:
        idx_price = price_df[base_index_ticker]
    else:
        fallback_ticker = tickers_bm[0] if tickers_bm else (tickers_wave[0] if tickers_wave else price_df.columns[0])
        idx_price = price_df[fallback_ticker]
    idx_ret_60d = idx_price / idx_price.shift(60) - 1.0
    mom_60 = price_df / price_df.shift(60) - 1.0

    # VIX or crypto-VIX proxy (BTC vol) for crypto/Bitcoin Waves
    wave_is_crypto = ((CRYPTO_WAVE_KEYWORD in wave_name) or ("Bitcoin" in wave_name))
    if wave_is_crypto and BTC_TICKER in price_df.columns:
        btc_ret = price_df[BTC_TICKER].pct_change().fillna(0.0)
        rolling_vol = btc_ret.rolling(30).std() * np.sqrt(TRADING_DAYS_PER_YEAR) * 100.0
        vix_level_series = rolling_vol.reindex(price_df.index).ffill().bfill()
    else:
        if VIX_TICKER in price_df.columns:
            vix_level_series = price_df[VIX_TICKER].copy()
        else:
            vix_level_series = pd.Series(20.0, index=price_df.index)

    # Safe asset
    safe_ticker = None
    for t in safe_candidates:
        if t in price_df.columns:
            safe_ticker = t
            break
    if safe_ticker is None:
        safe_ticker = base_index_ticker
    safe_ret_series = ret_df[safe_ticker]

    # -------------------------
    # Simple per-wave tuning
    # -------------------------
    tilt_strength = 0.80
    vol_target = PORTFOLIO_VOL_TARGET
    extra_safe_boost = 0.00  # additive safe fraction

    if wave_name == "Demas Fund Wave":
        tilt_strength = 0.45
        vol_target = 0.15
        extra_safe_boost = 0.03

    mode_base_exposure = MODE_BASE_EXPOSURE[mode]
    exp_min, exp_max = MODE_EXPOSURE_CAPS[mode]

    wave_ret_list: List[float] = []
    dates: List[pd.Timestamp] = []

    # Yield overlay
    apy = CRYPTO_YIELD_OVERLAY_APY.get(wave_name, 0.0)
    daily_yield = apy / TRADING_DAYS_PER_YEAR if apy > 0 else 0.0

    for dt in ret_df.index:
        rets = ret_df.loc[dt]

        regime = _regime_from_return(idx_ret_60d.get(dt, np.nan))
        regime_exposure = REGIME_EXPOSURE[regime]
        regime_gate = REGIME_GATING[mode][regime]

        vix_level = float(vix_level_series.get(dt, np.nan))
        vix_exposure = _vix_exposure_factor(vix_level, mode)
        vix_gate = _vix_safe_fraction(vix_level, mode)

        # Momentum tilt
        mom_row = mom_60.loc[dt] if dt in mom_60.index else None
        if mom_row is not None:
            mom_series = mom_row.reindex(price_df.columns).fillna(0.0)
            mom_clipped = mom_series.clip(lower=-0.30, upper=0.30)
            tilt_factor = 1.0 + tilt_strength * mom_clipped
            effective_weights = wave_weights_aligned * tilt_factor
        else:
            effective_weights = wave_weights_aligned.copy()

        effective_weights = effective_weights.clip(lower=0.0)

        risk_weight_total = float(effective_weights.sum())
        if risk_weight_total > 0:
            risk_weights = effective_weights / risk_weight_total
        else:
            risk_weights = wave_weights_aligned.copy()

        portfolio_risk_ret = float((rets * risk_weights).sum())
        safe_ret = float(safe_ret_series.loc[dt])

        # 20D realized vol for vol-targeting
        if len(wave_ret_list) >= 20:
            recent = np.array(wave_ret_list[-20:])
            recent_vol = recent.std() * np.sqrt(TRADING_DAYS_PER_YEAR)
        else:
            recent_vol = vol_target

        vol_adjust = 1.0
        if recent_vol > 0:
            vol_adjust = vol_target / recent_vol
            vol_adjust = float(np.clip(vol_adjust, 0.7, 1.3))

        raw_exposure = mode_base_exposure * regime_exposure * vol_adjust * vix_exposure
        exposure = float(np.clip(raw_exposure, exp_min, exp_max))

        safe_fraction = regime_gate + vix_gate + extra_safe_boost
        safe_fraction = float(np.clip(safe_fraction, 0.0, 0.95))
        risk_fraction = 1.0 - safe_fraction

        base_total_ret = safe_fraction * safe_ret + risk_fraction * exposure * portfolio_risk_ret
        total_ret = base_total_ret

        # Crypto income Waves: add assumed APY overlay (Bitcoin does NOT get this unless you add it)
        if daily_yield != 0.0:
            total_ret += daily_yield

        # Private Logic mean-reversion overlay
        if mode == "Private Logic" and len(wave_ret_list) >= 20:
            recent = np.array(wave_ret_list[-20:])
            daily_vol = recent.std()
            if daily_vol > 0:
                shock_threshold = 2.0 * daily_vol
                if base_total_ret <= -shock_threshold:
                    total_ret = base_total_ret * 1.30
                elif base_total_ret >= shock_threshold:
                    total_ret = base_total_ret * 0.70

        wave_ret_list.append(total_ret)
        dates.append(dt)

    wave_ret_series = pd.Series(wave_ret_list, index=pd.Index(dates, name="Date"))
    bm_ret_series = bm_ret_series.reindex(wave_ret_series.index).fillna(0.0)

    wave_nav = (1.0 + wave_ret_series).cumprod()
    bm_nav = (1.0 + bm_ret_series).cumprod()

    out = pd.DataFrame({"wave_nav": wave_nav, "bm_nav": bm_nav, "wave_ret": wave_ret_series, "bm_ret": bm_ret_series})
    out.index.name = "Date"
    return out


def get_benchmark_mix_table() -> pd.DataFrame:
    rows = []
    for wave in get_all_waves():
        holdings = BENCHMARK_WEIGHTS_STATIC.get(wave)
        if not holdings:
            holdings = get_auto_benchmark_holdings(wave)
        weights = _normalize_weights(holdings)
        for h in holdings:
            t = _normalize_ticker(h.ticker)
            if t not in weights.index:
                continue
            rows.append({"Wave": wave, "Ticker": t, "Name": h.name or "", "Weight": float(weights[t])})
    if not rows:
        return pd.DataFrame(columns=["Wave", "Ticker", "Name", "Weight"])
    df = pd.DataFrame(rows).sort_values(["Wave", "Weight"], ascending=[True, False])
    return df


def get_wave_holdings(wave_name: str) -> pd.DataFrame:
    holdings = WAVE_WEIGHTS.get(wave_name, [])
    if not holdings:
        return pd.DataFrame(columns=["Ticker", "Name", "Weight"])
    weights = _normalize_weights(holdings)
    rows = []
    for h in holdings:
        t = _normalize_ticker(h.ticker)
        if t not in weights.index:
            continue
        rows.append({"Ticker": t, "Name": h.name or "", "Weight": float(weights[t])})
    df = pd.DataFrame(rows).drop_duplicates(subset=["Ticker"]).sort_values("Weight", ascending=False)
    return df