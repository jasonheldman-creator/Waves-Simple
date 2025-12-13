# waves_engine.py — WAVES Intelligence™ Vector Engine
# Dynamic Strategy + VIX + SmartSafe + Auto-Custom Benchmarks
#
# Mobile-friendly:
#   • No terminal, no CLI, no CSV requirement for core behavior.
#   • Internal Wave holdings + ETF / crypto candidate library.
#
# Strategy side:
#   • Momentum tilts (60D)
#   • Volatility targeting (20D realized vol)
#   • Regime gating (SPY 60D trend)
#   • VIX-based exposure scaling (or BTC-vol-based for crypto)
#   • SmartSafe™ sweep based on VIX & regime
#   • Mode-specific exposure caps (Standard, Alpha-Minus-Beta, Private Logic)
#   • Private Logic™ mean-reversion overlay
#
# Benchmark side:
#   • Auto-constructed composite benchmarks based on Wave exposures.
#   • Fallback static benchmark weights per Wave.
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
from typing import Dict, List, Set

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

PORTFOLIO_VOL_TARGET = 0.20  # ~20% annualized

VIX_TICKER = "^VIX"
BTC_TICKER = "BTC-USD"  # used for crypto-VIX proxy

# Crypto yield overlays (APY assumptions per Wave)
CRYPTO_YIELD_OVERLAY_APY: Dict[str, float] = {
    "Crypto Stable Yield Wave": 0.04,
    "Crypto Income & Yield Wave": 0.08,
    "Crypto High-Yield Income Wave": 0.12,
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
# Internal Wave holdings
# ------------------------------------------------------------

WAVE_WEIGHTS: Dict[str, List[Holding]] = {
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

    # ✅ DEMAS FUND (simple add — edit later anytime)
    "Demas Fund Wave": [
        Holding("SPY", 0.40, "SPDR S&P 500 ETF"),
        Holding("QQQ", 0.30, "Invesco QQQ Trust"),
        Holding("TLT", 0.15, "iShares 20+ Year Treasury Bond ETF"),
        Holding("GLD", 0.10, "SPDR Gold Shares"),
        Holding("BIL", 0.05, "SPDR Bloomberg 1-3 Month T-Bill ETF"),
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

    # ✅ DEMAS benchmark (simple, clean)
    "Demas Fund Wave": [
        Holding("SPY", 0.60, "SPDR S&P 500 ETF"),
        Holding("TLT", 0.25, "iShares 20+ Year Treasury Bond ETF"),
        Holding("GLD", 0.15, "SPDR Gold Shares"),
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
]

# ------------------------------------------------------------
# Public API
# ------------------------------------------------------------

def get_all_waves() -> list[str]:
    return sorted(WAVE_WEIGHTS.keys())


def get_modes() -> list[str]:
    return list(MODE_BASE_EXPOSURE.keys())


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
    if ticker in {"BIL", "SGOV", "SHV", "SHY", "SUB", "SHM", "MUB", "IEF", "TLT", "LQD"}:
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


def compute_history_nav(wave_name: str, mode: str = "Standard", days: int = 365) -> pd.DataFrame:
    if wave_name not in WAVE_WEIGHTS:
        raise ValueError(f"Unknown Wave: {wave_name}")
    if mode not in MODE_BASE_EXPOSURE:
        raise ValueError(f"Unknown mode: {mode}")

    wave_holdings = WAVE_WEIGHTS[wave_name]
    bm_holdings = get_auto_benchmark_holdings(wave_name) or BENCHMARK_WEIGHTS_STATIC.get(wave_name, [])

    wave_weights = _normalize_weights(wave_holdings)
    bm_weights = _normalize_weights(bm_holdings)

    tickers_wave = list(wave_weights.index)
    tickers_bm = list(bm_weights.index)

    base_index_ticker = "SPY"
    safe_candidates = ["SGOV", "BIL", "SHY", "SUB", "SHM", "MUB",
                       "USDC-USD", "USDT-USD", "DAI-USD", "USDP-USD"]

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

    if base_index_ticker in price_df.columns:
        idx_price = price_df[base_index_ticker]
    else:
        fallback_ticker = tickers_bm[0] if tickers_bm else (tickers_wave[0] if tickers_wave else price_df.columns[0])
        idx_price = price_df[fallback_ticker]

    idx_ret_60d = idx_price / idx_price.shift(60) - 1.0
    mom_60 = price_df / price_df.shift(60) - 1.0

    wave_is_crypto = (CRYPTO_WAVE_KEYWORD in wave_name) or ("Bitcoin" in wave_name)

    if wave_is_crypto and BTC_TICKER in price_df.columns:
        btc_ret = price_df[BTC_TICKER].pct_change().fillna(0.0)
        rolling_vol = btc_ret.rolling(30).std() * np.sqrt(TRADING_DAYS_PER_YEAR) * 100.0
        vix_level_series = rolling_vol.reindex(price_df.index).ffill().bfill()
    else:
        vix_level_series = price_df[VIX_TICKER].copy() if VIX_TICKER in price_df.columns else pd.Series(20.0, index=price_df.index)

    safe_ticker = next((t for t in safe_candidates if t in price_df.columns), None) or base_index_ticker
    safe_ret_series = ret_df[safe_ticker]

    mode_base_exposure = MODE_BASE_EXPOSURE[mode]
    exp_min, exp_max = MODE_EXPOSURE_CAPS[mode]

    wave_ret_list: List[float] = []
    dates: List[pd.Timestamp] = []

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

        mom_row = mom_60.loc[dt] if dt in mom_60.index else None
        if mom_row is not None:
            mom_series = mom_row.reindex(price_df.columns).fillna(0.0)
            mom_clipped = mom_series.clip(lower=-0.30, upper=0.30)
            tilt_factor = 1.0 + 0.8 * mom_clipped
            effective_weights = wave_weights_aligned * tilt_factor
        else:
            effective_weights = wave_weights_aligned.copy()

        effective_weights = effective_weights.clip(lower=0.0)
        risk_weight_total = effective_weights.sum()
        risk_weights = (effective_weights / risk_weight_total) if risk_weight_total > 0 else wave_weights_aligned.copy()

        portfolio_risk_ret = float((rets * risk_weights).sum())
        safe_ret = float(safe_ret_series.loc[dt])

        if len(wave_ret_list) >= 20:
            recent = np.array(wave_ret_list[-20:])
            recent_vol = recent.std() * np.sqrt(TRADING_DAYS_PER_YEAR)
        else:
            recent_vol = PORTFOLIO_VOL_TARGET

        vol_adjust = 1.0
        if recent_vol > 0:
            vol_adjust = PORTFOLIO_VOL_TARGET / recent_vol
            vol_adjust = float(np.clip(vol_adjust, 0.7, 1.3))

        raw_exposure = mode_base_exposure * regime_exposure * vol_adjust * vix_exposure
        exposure = float(np.clip(raw_exposure, exp_min, exp_max))

        safe_fraction = float(np.clip(regime_gate + vix_gate, 0.0, 0.95))
        risk_fraction = 1.0 - safe_fraction

        base_total_ret = safe_fraction * safe_ret + risk_fraction * exposure * portfolio_risk_ret
        total_ret = base_total_ret

        if daily_yield != 0.0:
            total_ret += daily_yield

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
        auto_holdings = get_auto_benchmark_holdings(wave)
        holdings = auto_holdings or BENCHMARK_WEIGHTS_STATIC.get(wave, [])
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