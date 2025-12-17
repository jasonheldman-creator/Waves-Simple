# waves_engine.py — WAVES Intelligence™ Vector Engine (v17.1)
# Dynamic Strategy + VIX + SmartSafe + Auto-Custom Benchmarks
#
# NEW in v17.1:
#   • Adds a "shadow" simulator: simulate_history_nav(... overrides ...)
#     - Does NOT alter baseline compute_history_nav
#   • Adds diagnostics helpers:
#     - get_latest_diagnostics(...)
#     - get_parameter_defaults(...)
#
# NOTE:
#   This engine is "mobile-friendly" and does not require CSVs.
#   It uses internal holdings and an auto-constructed composite benchmark system.

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from functools import lru_cache
from typing import Dict, List, Set, Optional, Any, Tuple

import numpy as np
import pandas as pd

try:
    import yfinance as yf
except ImportError:  # pragma: no cover
    yf = None

# ------------------------------------------------------------
# Global config
# ------------------------------------------------------------

USE_FULL_WAVE_HISTORY: bool = False  # compatibility flag

TRADING_DAYS_PER_YEAR = 252

# Mode risk appetites & exposure caps
MODE_BASE_EXPOSURE: Dict[str, float] = {
    "Standard": 1.00,
    "Alpha-Minus-Beta": 0.85,
    "Private Logic": 1.10,
}

MODE_EXPOSURE_CAPS: Dict[str, tuple[float, float]] = {
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

PORTFOLIO_VOL_TARGET = 0.20  # 20% annualized default

VIX_TICKER = "^VIX"
BTC_TICKER = "BTC-USD"  # used for crypto "VIX proxy"

# Crypto yield overlays (APY assumptions per Wave)
CRYPTO_YIELD_OVERLAY_APY: Dict[str, float] = {
    "Crypto Stable Yield Wave": 0.04,
    "Crypto Income & Yield Wave": 0.08,
    "Crypto High-Yield Income Wave": 0.12,
}

CRYPTO_WAVE_KEYWORD = "Crypto"

# Default per-wave tuning (can be overridden in shadow simulation)
DEFAULT_TILT_STRENGTH = 0.80
DEFAULT_EXTRA_SAFE_BOOST = 0.00

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
# Internal Wave holdings (20 Waves)
# ------------------------------------------------------------

WAVE_WEIGHTS: Dict[str, List[Holding]] = {
    # 1) Added to reach 20 waves
    "S&P 500 Wave": [
        Holding("SPY", 1.00, "SPDR S&P 500 ETF"),
    ],

    # Core US equity Waves
    "US MegaCap Core Wave": [
        Holding("AAPL", 0.07, "Apple Inc."),
        Holding("MSFT", 0.07, "Microsoft Corp."),
        Holding("AMZN", 0.05, "Amazon.com Inc."),
        Holding("GOOGL", 0.04, "Alphabet Inc. (Class A)"),
        Holding("META", 0.03, "Meta Platforms Inc."),
        Holding("NVDA", 0.06, "NVIDIA Corp."),
        Holding("BRK-B", 0.04, "Berkshire Hathaway Inc. (B)"),
        Holding("UNH", 0.03, "UnitedHealth Group Inc."),
        Holding("JPM", 0.03, "JPMorgan Chase & Co."),
        Holding("XOM", 0.03, "Exxon Mobil Corp."),
    ],
    "AI & Cloud MegaCap Wave": [
        Holding("NVDA", 0.12, "NVIDIA Corp."),
        Holding("MSFT", 0.10, "Microsoft Corp."),
        Holding("GOOGL", 0.08, "Alphabet Inc. (Class A)"),
        Holding("META", 0.07, "Meta Platforms Inc."),
        Holding("AVGO", 0.08, "Broadcom Inc."),
        Holding("ADBE", 0.07, "Adobe Inc."),
        Holding("AMD", 0.08, "Advanced Micro Devices Inc."),
        Holding("CRM", 0.06, "Salesforce Inc."),
        Holding("ORCL", 0.06, "Oracle Corp."),
        Holding("INTC", 0.06, "Intel Corp."),
    ],
    "Next-Gen Compute & Semis Wave": [
        Holding("IBM", 0.10, "International Business Machines Corp."),
        Holding("MSFT", 0.08, "Microsoft Corp."),
        Holding("GOOGL", 0.08, "Alphabet Inc. (Class A)"),
        Holding("NVDA", 0.10, "NVIDIA Corp."),
        Holding("AMZN", 0.08, "Amazon.com Inc."),
        Holding("QCOM", 0.08, "Qualcomm Inc."),
        Holding("INTC", 0.08, "Intel Corp."),
        Holding("TSM", 0.10, "Taiwan Semiconductor Manufacturing"),
        Holding("ADBE", 0.07, "Adobe Inc."),
        Holding("SNOW", 0.07, "Snowflake Inc."),
    ],
    "Future Energy & EV Wave": [
        Holding("XLE", 0.12, "Energy Select Sector SPDR"),
        Holding("ICLN", 0.10, "iShares Global Clean Energy"),
        Holding("ENPH", 0.08, "Enphase Energy Inc."),
        Holding("NEE", 0.10, "NextEra Energy Inc."),
        Holding("FSLR", 0.08, "First Solar Inc."),
        Holding("TSLA", 0.10, "Tesla Inc."),
        Holding("RUN", 0.07, "Sunrun Inc."),
        Holding("BP", 0.08, "BP plc"),
        Holding("CVX", 0.10, "Chevron Corp."),
        Holding("PLUG", 0.07, "Plug Power Inc."),
    ],
    "EV & Infrastructure Wave": [
        Holding("TSLA", 0.12, "Tesla Inc."),
        Holding("NIO", 0.08, "NIO Inc."),
        Holding("GM", 0.08, "General Motors"),
        Holding("F", 0.08, "Ford Motor Co."),
        Holding("CAT", 0.08, "Caterpillar Inc."),
        Holding("UNP", 0.08, "Union Pacific Corp."),
        Holding("VMC", 0.08, "Vulcan Materials"),
        Holding("MLM", 0.08, "Martin Marietta Materials"),
        Holding("XLI", 0.16, "Industrial Select Sector SPDR"),
        Holding("PAVE", 0.16, "Global X U.S. Infrastructure Development"),
    ],
    "US Small-Cap Disruptors Wave": [
        Holding("IWO", 0.30, "iShares Russell 2000 Growth ETF"),
        Holding("VBK", 0.30, "Vanguard Small-Cap Growth ETF"),
        Holding("ARKK", 0.10, "ARK Innovation ETF"),
        Holding("ZS", 0.10, "Zscaler Inc."),
        Holding("DDOG", 0.10, "Datadog Inc."),
        Holding("NET", 0.10, "Cloudflare Inc."),
    ],
    "US Mid/Small Growth & Semis Wave": [
        Holding("IWP", 0.30, "iShares Russell Mid-Cap Growth ETF"),
        Holding("MDY", 0.30, "SPDR S&P MidCap 400 ETF"),
        Holding("IWO", 0.20, "iShares Russell 2000 Growth ETF"),
        Holding("SMH", 0.20, "VanEck Semiconductor ETF"),
    ],

    # Demas Fund Wave (value/quality + defensive tilt)
    "Demas Fund Wave": [
        Holding("BRK-B", 0.12, "Berkshire Hathaway (B)"),
        Holding("JPM", 0.10, "JPMorgan Chase & Co."),
        Holding("XOM", 0.10, "Exxon Mobil Corp."),
        Holding("CVX", 0.08, "Chevron Corp."),
        Holding("PG", 0.10, "Procter & Gamble"),
        Holding("KO", 0.08, "Coca-Cola"),
        Holding("JNJ", 0.10, "Johnson & Johnson"),
        Holding("UNH", 0.10, "UnitedHealth Group"),
        Holding("WMT", 0.10, "Walmart"),
        Holding("VTV", 0.12, "Vanguard Value ETF"),
    ],

    # Crypto growth Wave
    "Multi-Cap Crypto Growth Wave": [
        Holding("BTC-USD", 0.25, "Bitcoin"),
        Holding("ETH-USD", 0.20, "Ethereum"),
        Holding("SOL-USD", 0.10, "Solana"),
        Holding("AVAX-USD", 0.07, "Avalanche"),
        Holding("ADA-USD", 0.07, "Cardano"),
        Holding("MATIC-USD", 0.06, "Polygon"),
        Holding("LINK-USD", 0.06, "Chainlink"),
        Holding("DOT-USD", 0.06, "Polkadot"),
        Holding("ATOM-USD", 0.05, "Cosmos"),
        Holding("GRT-USD", 0.03, "The Graph"),
        Holding("AAVE-USD", 0.03, "Aave"),
        Holding("UNI-USD", 0.02, "Uniswap"),
    ],
    "Bitcoin Wave": [Holding("BTC-USD", 1.00, "Bitcoin")],

    "Crypto Stable Yield Wave": [
        Holding("USDC-USD", 0.35, "USD Coin"),
        Holding("USDT-USD", 0.30, "Tether"),
        Holding("DAI-USD", 0.20, "Dai"),
        Holding("USDP-USD", 0.10, "Pax Dollar"),
        Holding("sDAI-USD", 0.03, "Savings Dai (proxy)"),
        Holding("stETH-USD", 0.02, "Lido Staked Ether"),
    ],
    "Crypto Income & Yield Wave": [
        Holding("USDC-USD", 0.25, "USD Coin"),
        Holding("USDT-USD", 0.20, "Tether"),
        Holding("DAI-USD", 0.15, "Dai"),
        Holding("USDP-USD", 0.05, "Pax Dollar"),
        Holding("stETH-USD", 0.10, "Lido Staked Ether"),
        Holding("AAVE-USD", 0.10, "Aave"),
        Holding("MKR-USD", 0.05, "Maker"),
        Holding("ETH-USD", 0.05, "Ethereum"),
        Holding("BTC-USD", 0.05, "Bitcoin"),
    ],
    "Crypto High-Yield Income Wave": [
        Holding("stETH-USD", 0.15, "Lido Staked Ether"),
        Holding("AAVE-USD", 0.15, "Aave"),
        Holding("MKR-USD", 0.10, "Maker"),
        Holding("UNI-USD", 0.10, "Uniswap"),
        Holding("GRT-USD", 0.10, "The Graph"),
        Holding("LINK-USD", 0.10, "Chainlink"),
        Holding("ETH-USD", 0.15, "Ethereum"),
        Holding("BTC-USD", 0.15, "Bitcoin"),
    ],

    # SmartSafe Waves
    "SmartSafe Treasury Cash Wave": [
        Holding("BIL", 0.50, "SPDR Bloomberg 1-3 Month T-Bill ETF"),
        Holding("SGOV", 0.50, "iShares 0-3 Month Treasury Bond ETF"),
    ],
    "SmartSafe Tax-Free Money Market Wave": [
        Holding("SUB", 0.40, "iShares Short-Term National Muni Bond ETF"),
        Holding("SHM", 0.40, "SPDR Nuveen Short-Term Municipal Bond ETF"),
        Holding("MUB", 0.20, "iShares National Muni Bond ETF"),
    ],

    "Gold Wave": [
        Holding("GLD", 0.70, "SPDR Gold Shares"),
        Holding("IAU", 0.30, "iShares Gold Trust"),
    ],

    "Infinity Multi-Asset Growth Wave": [
        Holding("SPY", 0.20, "SPDR S&P 500 ETF"),
        Holding("QQQ", 0.20, "Invesco QQQ Trust"),
        Holding("VGT", 0.10, "Vanguard Information Technology ETF"),
        Holding("SMH", 0.10, "VanEck Semiconductor ETF"),
        Holding("ICLN", 0.10, "iShares Global Clean Energy ETF"),
        Holding("PAVE", 0.10, "Global X U.S. Infrastructure Development"),
        Holding("IWO", 0.10, "iShares Russell 2000 Growth ETF"),
        Holding("BTC-USD", 0.05, "Bitcoin"),
        Holding("ETH-USD", 0.05, "Ethereum"),
    ],

    "Vector Treasury Ladder Wave": [
        Holding("BIL", 0.25, "SPDR Bloomberg 1-3 Month T-Bill ETF"),
        Holding("SHY", 0.20, "iShares 1-3 Year Treasury Bond ETF"),
        Holding("IEF", 0.20, "iShares 7-10 Year Treasury Bond ETF"),
        Holding("TLT", 0.20, "iShares 20+ Year Treasury Bond ETF"),
        Holding("LQD", 0.15, "iShares iBoxx $ Investment Grade Corporate Bond ETF"),
    ],
    "Vector Muni Ladder Wave": [
        Holding("SUB", 0.25, "iShares Short-Term National Muni Bond ETF"),
        Holding("SHM", 0.20, "SPDR Nuveen Short-Term Municipal Bond ETF"),
        Holding("MUB", 0.25, "iShares National Muni Bond ETF"),
        Holding("TFI", 0.15, "SPDR Nuveen Bloomberg Municipal Bond ETF"),
        Holding("HYD", 0.15, "VanEck High-Yield Muni ETF"),
    ],
}

# ------------------------------------------------------------
# Static benchmarks (fallback / overrides)
# ------------------------------------------------------------

BENCHMARK_WEIGHTS_STATIC: Dict[str, List[Holding]] = {
    "S&P 500 Wave": [Holding("SPY", 1.0, "SPDR S&P 500 ETF")],

    "US MegaCap Core Wave": [Holding("SPY", 1.0, "SPDR S&P 500 ETF")],
    "AI & Cloud MegaCap Wave": [
        Holding("QQQ", 0.60, "Invesco QQQ Trust"),
        Holding("IGV", 0.40, "iShares Expanded Tech-Software"),
    ],
    "Next-Gen Compute & Semis Wave": [
        Holding("QQQ", 0.50, "Invesco QQQ Trust"),
        Holding("SMH", 0.50, "VanEck Semiconductor ETF"),
    ],
    "Future Energy & EV Wave": [
        Holding("XLE", 0.50, "Energy Select Sector SPDR"),
        Holding("ICLN", 0.50, "iShares Global Clean Energy"),
    ],
    "EV & Infrastructure Wave": [
        Holding("PAVE", 0.60, "Global X U.S. Infrastructure Development"),
        Holding("XLI", 0.40, "Industrial Select Sector SPDR"),
    ],
    "US Small-Cap Disruptors Wave": [
        Holding("IWO", 0.50, "iShares Russell 2000 Growth ETF"),
        Holding("VBK", 0.50, "Vanguard Small-Cap Growth ETF"),
    ],
    "US Mid/Small Growth & Semis Wave": [
        Holding("IWP", 0.50, "iShares Russell Mid-Cap Growth ETF"),
        Holding("IWO", 0.50, "iShares Russell 2000 Growth ETF"),
    ],

    "Demas Fund Wave": [
        Holding("SPY", 0.60, "SPDR S&P 500 ETF"),
        Holding("VTV", 0.40, "Vanguard Value ETF"),
    ],

    "Multi-Cap Crypto Growth Wave": [
        Holding("BTC-USD", 0.50, "Bitcoin"),
        Holding("ETH-USD", 0.35, "Ethereum"),
        Holding("SOL-USD", 0.15, "Solana"),
    ],
    "Bitcoin Wave": [Holding("BTC-USD", 1.00, "Bitcoin")],

    "Crypto Stable Yield Wave": [
        Holding("USDC-USD", 0.25, "USD Coin"),
        Holding("USDT-USD", 0.25, "Tether"),
        Holding("DAI-USD", 0.25, "Dai"),
        Holding("USDP-USD", 0.25, "Pax Dollar"),
    ],
    "Crypto Income & Yield Wave": [
        Holding("USDC-USD", 0.25, "USD Coin"),
        Holding("USDT-USD", 0.25, "Tether"),
        Holding("DAI-USD", 0.25, "Dai"),
        Holding("USDP-USD", 0.25, "Pax Dollar"),
    ],
    "Crypto High-Yield Income Wave": [
        Holding("BTC-USD", 0.40, "Bitcoin"),
        Holding("ETH-USD", 0.40, "Ethereum"),
        Holding("stETH-USD", 0.20, "Lido Staked Ether"),
    ],

    "SmartSafe Treasury Cash Wave": [
        Holding("BIL", 0.50, "SPDR Bloomberg 1-3 Month T-Bill"),
        Holding("SGOV", 0.50, "iShares 0-3 Month Treasury Bond ETF"),
    ],
    "SmartSafe Tax-Free Money Market Wave": [
        Holding("SUB", 0.50, "iShares Short-Term National Muni Bond ETF"),
        Holding("SHM", 0.50, "SPDR Nuveen Short-Term Municipal Bond ETF"),
    ],
    "Gold Wave": [
        Holding("GLD", 0.50, "SPDR Gold Shares"),
        Holding("IAU", 0.50, "iShares Gold Trust"),
    ],
    "Infinity Multi-Asset Growth Wave": [
        Holding("SPY", 0.40, "SPDR S&P 500 ETF"),
        Holding("QQQ", 0.40, "Invesco QQQ Trust"),
        Holding("BTC-USD", 0.20, "Bitcoin"),
    ],
    "Vector Treasury Ladder Wave": [
        Holding("BIL", 0.25, "SPDR Bloomberg 1-3 Month T-Bill ETF"),
        Holding("SHY", 0.25, "iShares 1-3 Year Treasury Bond ETF"),
        Holding("IEF", 0.25, "iShares 7-10 Year Treasury Bond ETF"),
        Holding("TLT", 0.25, "iShares 20+ Year Treasury Bond ETF"),
    ],
    "Vector Muni Ladder Wave": [
        Holding("SUB", 0.30, "iShares Short-Term National Muni Bond ETF"),
        Holding("SHM", 0.30, "SPDR Nuveen Short-Term Municipal Bond ETF"),
        Holding("MUB", 0.40, "iShares National Muni Bond ETF"),
    ],
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
    return sorted(WAVE_WEIGHTS.keys())


def get_modes() -> list[str]:
    return list(MODE_BASE_EXPOSURE.keys())


def get_parameter_defaults(wave_name: str, mode: str) -> Dict[str, float]:
    # Simple defaults + a per-wave tweak example (Demas)
    tilt_strength = DEFAULT_TILT_STRENGTH
    vol_target = PORTFOLIO_VOL_TARGET
    extra_safe_boost = DEFAULT_EXTRA_SAFE_BOOST

    if wave_name == "Demas Fund Wave":
        tilt_strength = 0.45
        vol_target = 0.15
        extra_safe_boost = 0.03

    exp_min, exp_max = MODE_EXPOSURE_CAPS.get(mode, (0.70, 1.30))
    base_exposure = MODE_BASE_EXPOSURE.get(mode, 1.0)

    return {
        "tilt_strength": float(tilt_strength),
        "vol_target": float(vol_target),
        "extra_safe_boost": float(extra_safe_boost),
        "exp_min": float(exp_min),
        "exp_max": float(exp_max),
        "base_exposure": float(base_exposure),
    }


def _normalize_weights(holdings: List[Holding]) -> pd.Series:
    if not holdings:
        return pd.Series(dtype=float)
    df = pd.DataFrame([{"ticker": h.ticker, "weight": h.weight} for h in holdings])
    df = df.groupby("ticker", as_index=False)["weight"].sum()
    total = df["weight"].sum()
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
    if ticker.endswith("-USD"):
        if ticker in {"USDC-USD", "USDT-USD", "DAI-USD", "USDP-USD", "sDAI-USD"}:
            return ("Safe", np.nan)
        return ("Crypto", np.nan)
    if ticker in {"BIL", "SGOV", "SHV", "SHY", "SUB", "SHM", "MUB", "IEF", "TLT", "LQD", "TFI", "HYD"}:
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
        if h.ticker not in weights.index:
            continue
        w = float(weights[h.ticker])
        sector, mcap = _get_ticker_meta(h.ticker)
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
    # Explicit override: Bitcoin benchmarks to spot BTC
    if wave_name == "Bitcoin Wave":
        return BENCHMARK_WEIGHTS_STATIC.get(wave_name, [])

    sector_weights, cap_style = _derive_wave_exposure(wave_name)
    if not sector_weights:
        return BENCHMARK_WEIGHTS_STATIC.get(wave_name, [])
    scores = []
    for etf in ETF_CANDIDATES:
        s = _score_etf_candidate(etf, sector_weights, cap_style)
        if s > 0.0:
            scores.append((etf, s))
    if not scores:
        return BENCHMARK_WEIGHTS_STATIC.get(wave_name, [])
    scores.sort(key=lambda x: x[1], reverse=True)
    top = scores[:4]
    if len(top) == 1:
        etf, _ = top[0]
        return [Holding(etf.ticker, 1.0, etf.name)]
    total_score = sum(s for _, s in top)
    if total_score <= 0:
        return BENCHMARK_WEIGHTS_STATIC.get(wave_name, [])
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


def _compute_core(
    wave_name: str,
    mode: str = "Standard",
    days: int = 365,
    overrides: Optional[Dict[str, Any]] = None,
    shadow: bool = False,
) -> pd.DataFrame:
    """
    Core engine calculator used by:
      - compute_history_nav()  [baseline]
      - simulate_history_nav() [shadow overrides]

    overrides keys (optional):
      - tilt_strength: float
      - vol_target: float
      - extra_safe_boost: float
      - base_exposure_mult: float
      - exp_min: float
      - exp_max: float
      - freeze_benchmark: bool   (use static benchmark only)
    """
    if wave_name not in WAVE_WEIGHTS:
        raise ValueError(f"Unknown Wave: {wave_name}")
    if mode not in MODE_BASE_EXPOSURE:
        raise ValueError(f"Unknown mode: {mode}")

    ov = overrides or {}

    # Holdings
    wave_holdings = WAVE_WEIGHTS[wave_name]

    # Benchmark selection
    freeze_benchmark = bool(ov.get("freeze_benchmark", False))
    if freeze_benchmark:
        bm_holdings = BENCHMARK_WEIGHTS_STATIC.get(wave_name, [])
    else:
        bm_holdings = BENCHMARK_WEIGHTS_STATIC.get(wave_name)
        if not bm_holdings:
            bm_holdings = get_auto_benchmark_holdings(wave_name) or BENCHMARK_WEIGHTS_STATIC.get(wave_name, [])

    wave_weights = _normalize_weights(wave_holdings)
    bm_weights = _normalize_weights(bm_holdings)

    tickers_wave = list(wave_weights.index)
    tickers_bm = list(bm_weights.index)

    base_index_ticker = "SPY"
    safe_candidates = ["SGOV", "BIL", "SHY", "SUB", "SHM", "MUB", "USDC-USD", "USDT-USD", "DAI-USD", "USDP-USD"]

    all_tickers = set(tickers_wave + tickers_bm)
    all_tickers.add(base_index_ticker)
    all_tickers.add(VIX_TICKER)
    all_tickers.add(BTC_TICKER)
    all_tickers.update(safe_candidates)

    all_tickers = sorted(all_tickers)
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

    # Defaults (plus Demas tweak)
    defaults = get_parameter_defaults(wave_name, mode)
    tilt_strength = float(ov.get("tilt_strength", defaults["tilt_strength"]))
    vol_target = float(ov.get("vol_target", defaults["vol_target"]))
    extra_safe_boost = float(ov.get("extra_safe_boost", defaults["extra_safe_boost"]))

    base_exposure_mult = float(ov.get("base_exposure_mult", 1.0))
    mode_base_exposure = float(MODE_BASE_EXPOSURE[mode]) * base_exposure_mult

    exp_min = float(ov.get("exp_min", defaults["exp_min"]))
    exp_max = float(ov.get("exp_max", defaults["exp_max"]))

    wave_ret_list: List[float] = []
    dates: List[pd.Timestamp] = []

    # Yield overlay
    apy = CRYPTO_YIELD_OVERLAY_APY.get(wave_name, 0.0)
    daily_yield = apy / TRADING_DAYS_PER_YEAR if apy > 0 else 0.0

    # Diagnostics series (optional)
    diag_rows = []

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

        risk_weight_total = effective_weights.sum()
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

        if recent_vol > 0:
            vol_adjust = vol_target / recent_vol
            vol_adjust = float(np.clip(vol_adjust, 0.7, 1.3))
        else:
            vol_adjust = 1.0

        raw_exposure = mode_base_exposure * regime_exposure * vol_adjust * vix_exposure
        exposure = float(np.clip(raw_exposure, exp_min, exp_max))

        safe_fraction = regime_gate + vix_gate + extra_safe_boost
        safe_fraction = float(np.clip(safe_fraction, 0.0, 0.95))
        risk_fraction = 1.0 - safe_fraction

        base_total_ret = safe_fraction * safe_ret + risk_fraction * exposure * portfolio_risk_ret
        total_ret = base_total_ret

        # Crypto income Waves: add assumed APY overlay (Bitcoin does NOT get this)
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

        if shadow:
            diag_rows.append(
                {
                    "Date": dt,
                    "regime": regime,
                    "vix": vix_level,
                    "safe_fraction": safe_fraction,
                    "exposure": exposure,
                    "vol_adjust": vol_adjust,
                    "vix_exposure": vix_exposure,
                    "vix_gate": vix_gate,
                    "regime_gate": regime_gate,
                }
            )

    wave_ret_series = pd.Series(wave_ret_list, index=pd.Index(dates, name="Date"))
    bm_ret_series = bm_ret_series.reindex(wave_ret_series.index).fillna(0.0)

    wave_nav = (1.0 + wave_ret_series).cumprod()
    bm_nav = (1.0 + bm_ret_series).cumprod()

    out = pd.DataFrame(
        {"wave_nav": wave_nav, "bm_nav": bm_nav, "wave_ret": wave_ret_series, "bm_ret": bm_ret_series}
    )
    out.index.name = "Date"

    if shadow and diag_rows:
        diag_df = pd.DataFrame(diag_rows).set_index("Date")
        out.attrs["diagnostics"] = diag_df

    return out


# ------------------------------------------------------------
# Baseline API (unchanged behavior)
# ------------------------------------------------------------

def compute_history_nav(wave_name: str, mode: str = "Standard", days: int = 365) -> pd.DataFrame:
    """
    Baseline (official) engine output.
    """
    return _compute_core(wave_name=wave_name, mode=mode, days=days, overrides=None, shadow=False)


def simulate_history_nav(wave_name: str, mode: str = "Standard", days: int = 365, overrides: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    """
    Shadow "What-If" simulation. Never overwrites baseline.
    Returns the same columns, plus diagnostics in out.attrs["diagnostics"] if available.
    """
    return _compute_core(wave_name=wave_name, mode=mode, days=days, overrides=overrides or {}, shadow=True)


def get_benchmark_mix_table() -> pd.DataFrame:
    rows = []
    for wave in get_all_waves():
        holdings = BENCHMARK_WEIGHTS_STATIC.get(wave)
        if not holdings:
            holdings = get_auto_benchmark_holdings(wave) or BENCHMARK_WEIGHTS_STATIC.get(wave, [])
        weights = _normalize_weights(holdings)
        for h in holdings:
            if h.ticker not in weights.index:
                continue
            rows.append({"Wave": wave, "Ticker": h.ticker, "Name": h.name or "", "Weight": float(weights[h.ticker])})
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
        if h.ticker not in weights.index:
            continue
        rows.append({"Ticker": h.ticker, "Name": h.name or "", "Weight": float(weights[h.ticker])})
    df = pd.DataFrame(rows).drop_duplicates(subset=["Ticker"]).sort_values("Weight", ascending=False)
    return df


# ------------------------------------------------------------
# Diagnostics helpers
# ------------------------------------------------------------

def _annualized_vol(daily_ret: pd.Series) -> float:
    if daily_ret is None or len(daily_ret) < 2:
        return float("nan")
    return float(daily_ret.std() * np.sqrt(TRADING_DAYS_PER_YEAR))


def _max_drawdown(nav: pd.Series) -> float:
    if nav is None or len(nav) < 2:
        return float("nan")
    running_max = nav.cummax()
    dd = (nav / running_max) - 1.0
    return float(dd.min())


def _tracking_error(daily_wave: pd.Series, daily_bm: pd.Series) -> float:
    if daily_wave is None or daily_bm is None:
        return float("nan")
    if len(daily_wave) != len(daily_bm) or len(daily_wave) < 2:
        return float("nan")
    diff = (daily_wave - daily_bm).dropna()
    if len(diff) < 2:
        return float("nan")
    return float(diff.std() * np.sqrt(TRADING_DAYS_PER_YEAR))


def _compute_total_return(nav: pd.Series) -> float:
    if nav is None or len(nav) < 2:
        return float("nan")
    start = float(nav.iloc[0])
    end = float(nav.iloc[-1])
    if start <= 0:
        return float("nan")
    return float(end / start - 1.0)


def get_latest_diagnostics(wave_name: str, mode: str = "Standard", days: int = 365) -> Dict[str, Any]:
    """
    Runs baseline compute_history_nav and produces a structured snapshot + suggestions.
    """
    out: Dict[str, Any] = {"wave": wave_name, "mode": mode, "days": days}

    hist = compute_history_nav(wave_name, mode=mode, days=days)
    if hist is None or hist.empty or len(hist) < 50:
        out["ok"] = False
        out["message"] = "Not enough data to compute diagnostics."
        return out

    nav_w = hist["wave_nav"]
    nav_b = hist["bm_nav"]
    ret_w = hist["wave_ret"]
    ret_b = hist["bm_ret"]

    total_w = _compute_total_return(nav_w)
    total_b = _compute_total_return(nav_b)
    alpha = total_w - total_b

    vol_w = _annualized_vol(ret_w)
    vol_b = _annualized_vol(ret_b)
    te = _tracking_error(ret_w, ret_b)

    mdd_w = _max_drawdown(nav_w)
    mdd_b = _max_drawdown(nav_b)

    # IR proxy (excess / TE)
    ir = float("nan")
    if te and not np.isnan(te) and te > 0:
        ir = float((alpha) / te)

    # Heuristic “what might be going on”
    suggestions: List[str] = []
    flags: List[str] = []

    if not np.isnan(alpha) and alpha < 0:
        flags.append("Alpha is negative vs benchmark over the window.")
        suggestions.append("Check benchmark mix (Benchmark Mix table) — if the auto benchmark shifted, alpha can swing even if raw return looks good.")
        suggestions.append("Use What-If: toggle 'Freeze benchmark' to compare dynamic benchmark vs static baseline for stability.")

    if not np.isnan(te) and te > 0.25:
        flags.append("Tracking error is high (active intensity).")
        suggestions.append("Consider tightening exposure cap max (What-If slider) or lowering tilt strength slightly to reduce whipsaw.")

    if not np.isnan(mdd_w) and mdd_w < -0.25:
        flags.append("Max drawdown is deep.")
        suggestions.append("Consider increasing extra safe boost slightly or lowering vol target (What-If).")

    if not np.isnan(vol_w) and not np.isnan(vol_b) and vol_w > vol_b * 1.25:
        flags.append("Wave vol is much higher than benchmark vol.")
        suggestions.append("Consider lowering vol target and/or lowering base exposure multiplier in this mode.")

    # Add context baseline numbers
    out.update(
        {
            "ok": True,
            "return_wave": float(total_w),
            "return_benchmark": float(total_b),
            "alpha": float(alpha),
            "vol_wave": float(vol_w),
            "vol_benchmark": float(vol_b),
            "tracking_error": float(te),
            "information_ratio": float(ir),
            "maxdd_wave": float(mdd_w),
            "maxdd_benchmark": float(mdd_b),
            "flags": flags,
            "suggestions": suggestions,
        }
    )
    return out
    
    # ============================================================
# ADD-ON: Canonical History Provider for the Console
# Provides: compute_history_nav(wave_name, mode="Standard", days=365)
#
# Safe design:
#   1) If you already have any internal history function, we try it first.
#   2) Else we read wave_history.csv (recommended immediate solution).
#   3) Else (optional) we can synthesize history from yfinance using a benchmark mix
#      (best-effort; non-fatal if yfinance not available).
# ============================================================

from __future__ import annotations

import os
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List

import numpy as np
import pandas as pd

# Optional yfinance (do NOT hard-require)
try:
    import yfinance as yf  # type: ignore
except Exception:
    yf = None


def _to_dt_index(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize a dataframe to have a datetime index if possible."""
    if df is None or df.empty:
        return df
    out = df.copy()
    for dc in ["date", "Date", "timestamp", "Timestamp", "datetime", "Datetime"]:
        if dc in out.columns:
            out[dc] = pd.to_datetime(out[dc], errors="coerce")
            out = out.dropna(subset=[dc]).set_index(dc)
            break
    out.index = pd.to_datetime(out.index, errors="coerce")
    out = out[~out.index.isna()].sort_index()
    return out


def _standardize_history_engine(df: pd.DataFrame) -> pd.DataFrame:
    """
    Output spec the app expects:
      index = datetime
      columns include:
        wave_nav, bm_nav (required)
        wave_ret, bm_ret (optional; will be derived if missing)
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=["wave_nav", "bm_nav", "wave_ret", "bm_ret"])

    out = _to_dt_index(df)

    # Normalize column names
    ren = {}
    for c in out.columns:
        low = str(c).strip().lower()
        if low in ["wave_nav", "nav_wave", "portfolio_nav", "nav", "wavevalue", "wave value"]:
            ren[c] = "wave_nav"
        elif low in ["bm_nav", "bench_nav", "benchmark_nav", "benchmark", "benchmark value", "bm value"]:
            ren[c] = "bm_nav"
        elif low in ["wave_ret", "ret_wave", "portfolio_ret", "return", "wave_return", "wave return"]:
            ren[c] = "wave_ret"
        elif low in ["bm_ret", "ret_bm", "benchmark_ret", "bm_return", "benchmark_return", "benchmark return"]:
            ren[c] = "bm_ret"
    out = out.rename(columns=ren)

    # Coerce numeric
    for col in ["wave_nav", "bm_nav", "wave_ret", "bm_ret"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    # Derive returns if missing
    if "wave_ret" not in out.columns and "wave_nav" in out.columns:
        out["wave_ret"] = out["wave_nav"].pct_change()
    if "bm_ret" not in out.columns and "bm_nav" in out.columns:
        out["bm_ret"] = out["bm_nav"].pct_change()

    # Keep only canonical columns
    cols = [c for c in ["wave_nav", "bm_nav", "wave_ret", "bm_ret"] if c in out.columns]
    out = out[cols].dropna(how="all")

    return out


def _read_wave_history_csv(path: str = "wave_history.csv") -> pd.DataFrame:
    """Reads wave_history.csv if present. Non-fatal if missing."""
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
        df.columns = [str(c).strip() for c in df.columns]
        return df
    except Exception:
        return pd.DataFrame()


def _history_from_csv(wave_name: str, mode: str, days: int, path: str = "wave_history.csv") -> pd.DataFrame:
    """
    Expected CSV columns (minimum):
      date, wave, mode, wave_nav, bm_nav
    Also supported aliases:
      wave_name, wavename, risk_mode, strategy_mode, benchmark_nav, bm_nav
    """
    raw = _read_wave_history_csv(path)
    if raw is None or raw.empty:
        return pd.DataFrame(columns=["wave_nav", "bm_nav", "wave_ret", "bm_ret"])

    df = raw.copy()
    df.columns = [str(c).strip() for c in df.columns]

    # Identify key columns
    wave_cols = [c for c in df.columns if c.lower() in ["wave", "wave_name", "wavename"]]
    mode_cols = [c for c in df.columns if c.lower() in ["mode", "risk_mode", "strategy_mode"]]
    date_cols = [c for c in df.columns if c.lower() in ["date", "timestamp", "datetime"]]

    wc = wave_cols[0] if wave_cols else None
    mc = mode_cols[0] if mode_cols else None
    dc = date_cols[0] if date_cols else None

    # Filter
    if wc:
        df[wc] = df[wc].astype(str)
        df = df[df[wc] == str(wave_name)]
    if mc:
        df[mc] = df[mc].astype(str)
        df = df[df[mc].str.lower() == str(mode).lower()]

    # Date index
    if dc:
        df[dc] = pd.to_datetime(df[dc], errors="coerce")
        df = df.dropna(subset=[dc]).sort_values(dc).set_index(dc)

    out = _standardize_history_engine(df)
    if len(out) > int(days):
        out = out.iloc[-int(days):]
    return out


def _yf_download_prices(tickers: List[str], days: int) -> pd.DataFrame:
    """Best-effort daily adjusted close download. Returns empty df if yf unavailable."""
    if yf is None or not tickers:
        return pd.DataFrame()
    end = datetime.utcnow().date()
    start = end - timedelta(days=int(days) + 520)
    try:
        data = yf.download(
            tickers=sorted(list(set([t.upper().strip() for t in tickers if str(t).strip()]))),
            start=start.isoformat(),
            end=end.isoformat(),
            interval="1d",
            auto_adjust=True,
            progress=False,
            group_by="column",
        )
        if data is None or getattr(data, "empty", True):
            return pd.DataFrame()

        # Flatten yfinance output
        if isinstance(getattr(data, "columns", None), pd.MultiIndex):
            if "Adj Close" in data.columns.get_level_values(0):
                data = data["Adj Close"]
            elif "Close" in data.columns.get_level_values(0):
                data = data["Close"]
            else:
                data = data[data.columns.levels[0][0]]

        if isinstance(getattr(data, "columns", None), pd.MultiIndex):
            data = data.droplevel(0, axis=1)

        if isinstance(data, pd.Series):
            data = data.to_frame()

        data = data.sort_index().ffill().bfill()

        if len(data) > int(days):
            data = data.iloc[-int(days):]

        return data
    except Exception:
        return pd.DataFrame()


def _compute_nav_from_prices(price_df: pd.DataFrame, weights: Dict[str, float]) -> pd.Series:
    """Create a normalized NAV series from price_df and weights."""
    if price_df is None or price_df.empty:
        return pd.Series(dtype=float)

    # Normalize weights
    w = {str(k).upper().strip(): float(v) for k, v in (weights or {}).items() if k is not None}
    if not w:
        return pd.Series(dtype=float)

    # Align tickers
    cols = [c for c in price_df.columns if str(c).upper().strip() in w]
    if not cols:
        return pd.Series(dtype=float)

    sub = price_df[cols].copy()
    sub.columns = [str(c).upper().strip() for c in sub.columns]
    sub = sub.ffill().bfill()

    # Normalize each ticker to 1.0 at start
    base = sub.iloc[0].replace(0, np.nan)
    norm = sub.divide(base, axis=1)

    # Weighted sum
    total_w = float(sum([w.get(c, 0.0) for c in norm.columns]))
    if total_w <= 0:
        return pd.Series(dtype=float)

    nav = pd.Series(0.0, index=norm.index)
    for c in norm.columns:
        nav = nav + norm[c] * (w.get(c, 0.0) / total_w)

    return nav.astype(float)


def _get_benchmark_mix_table_safe() -> pd.DataFrame:
    """
    Tries to use an existing engine function if present.
    If your engine already has get_benchmark_mix_table(), we use it.
    Otherwise, we try benchmark_mix.csv as a fallback.
    """
    # If you already implemented this elsewhere, prefer it
    try:
        if "get_benchmark_mix_table" in globals() and callable(globals()["get_benchmark_mix_table"]):
            df = globals()["get_benchmark_mix_table"]()
            if isinstance(df, pd.DataFrame):
                return df
    except Exception:
        pass

    # Fallback: benchmark_mix.csv
    for p in ["benchmark_mix.csv", "benchmark_mix_table.csv"]:
        if os.path.exists(p):
            try:
                df = pd.read_csv(p)
                df.columns = [str(c).strip() for c in df.columns]
                return df
            except Exception:
                continue

    return pd.DataFrame(columns=["Wave", "Ticker", "Name", "Weight"])


def compute_history_nav(wave_name: str, mode: str = "Standard", days: int = 365) -> pd.DataFrame:
    """
    Canonical history provider used by the Streamlit console.

    Returns a DataFrame with:
      index datetime
      wave_nav, bm_nav, wave_ret, bm_ret

    Source order:
      1) If you already have an internal history function, try it.
      2) wave_history.csv filtered by wave+mode
      3) yfinance synth (benchmark-only; best-effort) -> still returns bm_nav + placeholder wave_nav if possible
    """
    w = str(wave_name)
    m = str(mode)
    d = int(days)

    # --------------------------------------------------------
    # 1) Try any internal history function you might already have
    # --------------------------------------------------------
    internal_candidates = [
        "get_history_nav",
        "get_wave_history",
        "history_nav",
        "compute_nav_history",
        "compute_history",
    ]
    for fn in internal_candidates:
        try:
            if fn in globals() and callable(globals()[fn]):
                f = globals()[fn]
                try:
                    df = f(w, mode=m, days=d)
                except TypeError:
                    df = f(w, m, d)
                df = _standardize_history_engine(df)
                if df is not None and not df.empty:
                    return df
        except Exception:
            continue

    # --------------------------------------------------------
    # 2) CSV fallback (recommended immediate solution)
    # --------------------------------------------------------
    df_csv = _history_from_csv(w, m, d, path="wave_history.csv")
    if df_csv is not None and not df_csv.empty:
        return df_csv

    # --------------------------------------------------------
    # 3) Best-effort yfinance synth using benchmark mix ONLY
    #    (non-fatal; returns empty if no benchmark mix)
    # --------------------------------------------------------
    bm_mix = _get_benchmark_mix_table_safe()
    if isinstance(bm_mix, pd.DataFrame) and not bm_mix.empty and "Wave" in bm_mix.columns:
        try:
            rows = bm_mix[bm_mix["Wave"].astype(str) == w].copy()
            if "Ticker" in rows.columns and "Weight" in rows.columns and not rows.empty:
                rows["Ticker"] = rows["Ticker"].astype(str).str.upper().str.strip()
                rows["Weight"] = pd.to_numeric(rows["Weight"], errors="coerce").fillna(0.0)

                weights = {r["Ticker"]: float(r["Weight"]) for _, r in rows.iterrows() if float(r["Weight"]) > 0}
                tickers = list(weights.keys())

                prices = _yf_download_prices(tickers, d)
                bm_nav = _compute_nav_from_prices(prices, weights)

                if bm_nav is not None and len(bm_nav) >= 2:
                    out = pd.DataFrame(index=bm_nav.index)
                    out["bm_nav"] = bm_nav
                    # Placeholder wave_nav if we only have benchmark; keeps console alive but signals missing wave history
                    out["wave_nav"] = np.nan
                    out["bm_ret"] = out["bm_nav"].pct_change()
                    out["wave_ret"] = np.nan
                    out = out.dropna(how="all")
                    return out
        except Exception:
            pass

    # Nothing found
    return pd.DataFrame(columns=["wave_nav", "bm_nav", "wave_ret", "bm_ret"])