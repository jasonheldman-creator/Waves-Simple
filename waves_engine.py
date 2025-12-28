# waves_engine.py — WAVES Intelligence™ Vector Engine (v17.1)
# Dynamic Strategy + VIX + SmartSafe + Auto-Custom Benchmarks
#
# NEW in v17.1:
#   • Adds a "shadow" simulator: simulate_history_nav(... overrides ...)
#     - Does NOT alter baseline compute_history_nav (architectural stability)
#   • Adds diagnostics helpers (predictable system-feedback):
#     - get_latest_diagnostics(...)
#     - get_parameter_defaults(...)
#
# NOTE:
#   This engine is "mobile-friendly" and does not require CSVs (flexible onboarding).
#   It uses internal holdings and an auto-constructed composite benchmark system (governance-native architecture).

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from functools import lru_cache
from typing import Dict, List, Set, Optional, Any, Tuple
import time
import json
import os

import numpy as np
import pandas as pd

try:
    import yfinance as yf
except ImportError:  # pragma: no cover
    yf = None

# Import ticker diagnostics
try:
    from helpers.ticker_diagnostics import (
        FailedTickerReport, 
        FailureType, 
        categorize_error,
        get_diagnostics_tracker
    )
    DIAGNOSTICS_AVAILABLE = True
except ImportError:
    DIAGNOSTICS_AVAILABLE = False

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

# Income wave tickers (for income-specific overlays)
TNX_TICKER = "^TNX"  # 10-year Treasury (rates/duration proxy)
HYG_TICKER = "HYG"   # High Yield Corporate Bond ETF (credit risk proxy)
LQD_TICKER = "LQD"   # Investment Grade Corporate Bond ETF (credit quality proxy)

# Crypto yield overlays (APY assumptions per Wave)
CRYPTO_YIELD_OVERLAY_APY: Dict[str, float] = {
    "Crypto Income Wave": 0.08,  # Crypto Income Wave with CSE basket
}

CRYPTO_WAVE_KEYWORD = "Crypto"

# Ticker normalization and aliases
# Maps known ticker variants to their canonical form for yfinance
TICKER_ALIASES: Dict[str, str] = {
    # Berkshire Hathaway Class B
    "BRK.B": "BRK-B",
    "BRK/B": "BRK-B",
    "BRKB": "BRK-B",
    
    # Brown-Forman Class B
    "BF.B": "BF-B",
    "BF/B": "BF-B",
    "BFB": "BF-B",
    
    # Crypto tickers - ensure USD pairs for yfinance
    "BTC": "BTC-USD",
    "ETH": "ETH-USD",
    "SOL": "SOL-USD",
    "ADA": "ADA-USD",
    "DOT": "DOT-USD",
    "AVAX": "AVAX-USD",
    "MATIC": "MATIC-USD",
    "UNI": "UNI-USD",
    "LINK": "LINK-USD",
    "AAVE": "AAVE-USD",
    "MKR": "MKR-USD",
    "SNX": "SNX-USD",
    "CRV": "CRV-USD",
    "COMP": "COMP-USD",
    "XRP": "XRP-USD",
    "BNB": "BNB-USD",
    "ATOM": "ATOM-USD",
    "NEAR": "NEAR-USD",
    "APT": "APT-USD",
    "ARB": "ARB-USD",
    "OP": "OP-USD",
    "IMX": "IMX-USD",
    "MNT": "MNT-USD",
    "STX": "STX-USD",
    "ICP": "ICP-USD",
    "FET": "FET-USD",
    "OCEAN": "OCEAN-USD",
    "AGIX": "AGIX-USD",
    "RENDER": "RENDER-USD",
    "TAO": "TAO-USD",
    "INJ": "INJ-USD",
    "LDO": "LDO-USD",
    "CAKE": "CAKE-USD",
    "stETH": "stETH-USD",
}

# Crypto-specific overlay configurations
# These are DISTINCT from equity VIX-based overlays

# Crypto Trend/Momentum Regime thresholds (for growth waves)
CRYPTO_TREND_MOMENTUM_THRESHOLDS = {
    "strong_uptrend": 0.15,     # 15% trend = strong bullish
    "uptrend": 0.05,            # 5% trend = bullish
    "downtrend": -0.05,         # -5% trend = bearish
    "strong_downtrend": -0.15,  # -15% trend = strong bearish
}

# Crypto Trend/Momentum exposure multipliers
CRYPTO_TREND_EXPOSURE = {
    "strong_uptrend": 1.15,     # +15% exposure in strong uptrend
    "uptrend": 1.05,            # +5% exposure in uptrend
    "neutral": 1.00,            # baseline exposure
    "downtrend": 0.90,          # -10% exposure in downtrend
    "strong_downtrend": 0.75,   # -25% exposure in strong downtrend
}

# Crypto Volatility State thresholds (annualized, for realized vol)
CRYPTO_VOL_THRESHOLDS = {
    "extreme_compression": 0.30,  # < 30% annualized = extreme compression
    "compression": 0.50,          # < 50% = compression
    "normal": 0.80,               # 50-80% = normal
    "expansion": 1.20,            # 80-120% = expansion
    # > 120% = extreme expansion
}

# Crypto Volatility State exposure adjustments
CRYPTO_VOL_EXPOSURE = {
    "extreme_compression": 1.10,   # +10% in extreme compression (anticipate breakout)
    "compression": 1.05,           # +5% in compression
    "normal": 1.00,                # baseline
    "expansion": 0.90,             # -10% in expansion
    "extreme_expansion": 0.75,     # -25% in extreme expansion
}

# Crypto Liquidity/Market Structure thresholds (volume-based)
CRYPTO_LIQUIDITY_THRESHOLDS = {
    "strong_volume": 1.50,     # Volume 50%+ above average = strong
    "normal_volume": 0.80,     # Volume 80%-150% of average = normal
    # Below 80% = weak volume
}

# Crypto Liquidity exposure adjustments (for all crypto waves)
CRYPTO_LIQUIDITY_EXPOSURE = {
    "strong_volume": 1.05,     # +5% on strong volume confirmation
    "normal_volume": 1.00,     # baseline
    "weak_volume": 0.90,       # -10% on weak volume/illiquidity
}

# Crypto Income Wave: Yield Stability Overlay
# Conservative exposure controls for capital preservation
CRYPTO_INCOME_SAFE_FRACTION = {
    "baseline": 0.20,          # 20% baseline safe allocation
    "stress_boost": 0.30,      # +30% to safe during stress (total 50%)
}

CRYPTO_INCOME_EXPOSURE_CAP = {
    "max_exposure": 0.90,      # Cap at 90% exposure (conservative)
    "min_exposure": 0.60,      # Floor at 60% exposure
}

# Crypto Income: Drawdown Guard thresholds (more aggressive than equity income)
CRYPTO_INCOME_DRAWDOWN_THRESHOLDS = {
    "minor": -0.05,           # -5% drawdown (crypto is more volatile)
    "moderate": -0.10,        # -10% drawdown
    "severe": -0.15,          # -15% drawdown
}

# Crypto Income: Drawdown Guard safe allocation boosts
CRYPTO_INCOME_DRAWDOWN_SAFE_BOOST = {
    "minor": 0.15,            # 15% to safe on minor drawdown
    "moderate": 0.25,         # 25% to safe on moderate drawdown
    "severe": 0.40,           # 40% to safe on severe drawdown
}

# Crypto Income: Liquidity Gate thresholds (more conservative than crypto growth)
CRYPTO_INCOME_LIQUIDITY_THRESHOLDS = {
    "strong_volume": 1.30,     # Volume 30%+ above average = strong (tighter than growth)
    "normal_volume": 0.90,     # Volume 90%-130% of average = normal (tighter than growth)
    # Below 90% = weak volume (stricter than growth's 80%)
}

# Crypto Income: Liquidity exposure adjustments (more conservative)
CRYPTO_INCOME_LIQUIDITY_EXPOSURE = {
    "strong_volume": 1.03,     # +3% on strong volume (less aggressive than growth)
    "normal_volume": 1.00,     # baseline
    "weak_volume": 0.85,       # -15% on weak volume (more defensive than growth)
}

# Default per-wave tuning (can be overridden in shadow simulation)
DEFAULT_TILT_STRENGTH = 0.80
DEFAULT_EXTRA_SAFE_BOOST = 0.00

# ------------------------------------------------------------
# Income Wave Overlay Configurations (Non-Crypto Income Waves)
# ------------------------------------------------------------

# A) Rates/Duration Regime thresholds (based on TNX trend)
INCOME_RATES_REGIME_THRESHOLDS = {
    "rising_fast": 0.10,      # +10% TNX = rising fast
    "rising": 0.03,           # +3% TNX = rising
    "stable": -0.03,          # -3% to +3% = stable
    "falling": -0.10,         # -10% TNX = falling
    # < -10% = falling_fast
}

# Rates regime exposure adjustments (reduce duration in rising rate environment)
INCOME_RATES_EXPOSURE = {
    "rising_fast": 0.80,      # -20% exposure in fast rising rates
    "rising": 0.90,           # -10% exposure in rising rates
    "stable": 1.00,           # baseline in stable rates
    "falling": 1.05,          # +5% exposure in falling rates
    "falling_fast": 1.10,     # +10% exposure in fast falling rates
}

# B) Credit/Risk Regime thresholds (HYG vs LQD relative strength)
INCOME_CREDIT_REGIME_THRESHOLDS = {
    "risk_on": 0.02,          # HYG outperforming LQD by 2%+ = risk on
    "neutral": -0.02,         # Within 2% = neutral
    # < -2% = risk off
}

# Credit regime exposure adjustments (shift to quality in risk-off)
INCOME_CREDIT_EXPOSURE = {
    "risk_on": 1.05,          # +5% exposure when credit spreads tight
    "neutral": 1.00,          # baseline
    "risk_off": 0.90,         # -10% exposure when credit spreads widen
}

# Credit regime safe allocation boosts (increase quality allocation)
INCOME_CREDIT_SAFE_BOOST = {
    "risk_on": 0.00,          # No boost in risk-on
    "neutral": 0.05,          # 5% to safe in neutral
    "risk_off": 0.15,         # 15% to safe in risk-off (quality focus)
}

# C) Carry/Drawdown Guard thresholds
INCOME_DRAWDOWN_THRESHOLDS = {
    "minor": -0.03,           # -3% drawdown
    "moderate": -0.05,        # -5% drawdown
    "severe": -0.08,          # -8% drawdown
}

# Drawdown protection adjustments
INCOME_DRAWDOWN_SAFE_BOOST = {
    "minor": 0.10,            # 10% to safe on minor drawdown
    "moderate": 0.20,         # 20% to safe on moderate drawdown
    "severe": 0.30,           # 30% to safe on severe drawdown
}

# D) Turnover Discipline settings
INCOME_MIN_REBALANCE_DAYS = 5        # Minimum days between rebalances
INCOME_MAX_TURNOVER_PER_PERIOD = 0.20  # Max 20% turnover unless strong signals

# ------------------------------------------------------------
# Strategy Configuration System
# ------------------------------------------------------------

@dataclass
class StrategyConfig:
    """Configuration for individual strategy contribution."""
    enabled: bool = True
    weight: float = 1.0  # Contribution weight (0.0 to 1.0+)
    min_impact: float = 0.0  # Minimum allowed impact
    max_impact: float = 1.0  # Maximum allowed impact

@dataclass
class StrategyContribution:
    """Per-strategy contribution to final exposure decision."""
    name: str
    exposure_impact: float  # Multiplier effect on exposure (0.5 to 1.5)
    safe_fraction_impact: float  # Additive contribution to safe fraction (0.0 to 1.0)
    risk_state: str  # "risk-on", "risk-off", or "neutral"
    enabled: bool = True
    metadata: Dict[str, Any] = None  # Optional diagnostic data
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

# Default strategy configurations (can be overridden per wave/mode)
DEFAULT_STRATEGY_CONFIGS: Dict[str, StrategyConfig] = {
    "momentum": StrategyConfig(enabled=True, weight=1.0, min_impact=0.0, max_impact=2.0),
    "trend_confirmation": StrategyConfig(enabled=True, weight=0.5, min_impact=0.0, max_impact=1.0),
    "relative_strength": StrategyConfig(enabled=True, weight=0.3, min_impact=0.0, max_impact=1.0),
    "volatility_targeting": StrategyConfig(enabled=True, weight=1.0, min_impact=0.7, max_impact=1.3),
    "regime_detection": StrategyConfig(enabled=True, weight=1.0, min_impact=0.8, max_impact=1.1),
    "vix_overlay": StrategyConfig(enabled=True, weight=1.0, min_impact=0.5, max_impact=1.3),
    "smartsafe": StrategyConfig(enabled=True, weight=1.0, min_impact=0.0, max_impact=0.95),
    "mode_constraint": StrategyConfig(enabled=True, weight=1.0, min_impact=0.5, max_impact=1.5),
    # Crypto-specific overlays (only active for crypto waves)
    "crypto_trend_momentum": StrategyConfig(enabled=True, weight=1.0, min_impact=0.75, max_impact=1.15),
    "crypto_volatility": StrategyConfig(enabled=True, weight=1.0, min_impact=0.75, max_impact=1.10),
    "crypto_liquidity": StrategyConfig(enabled=True, weight=1.0, min_impact=0.90, max_impact=1.05),
    "crypto_income_stability": StrategyConfig(enabled=True, weight=1.0, min_impact=0.60, max_impact=0.90),
    "crypto_income_drawdown_guard": StrategyConfig(enabled=True, weight=1.0, min_impact=0.0, max_impact=0.40),
    "crypto_income_liquidity_gate": StrategyConfig(enabled=True, weight=1.0, min_impact=0.85, max_impact=1.03),
    # Income-specific overlays (only active for non-crypto income waves)
    "income_rates_regime": StrategyConfig(enabled=True, weight=1.0, min_impact=0.80, max_impact=1.10),
    "income_credit_regime": StrategyConfig(enabled=True, weight=1.0, min_impact=0.90, max_impact=1.05),
    "income_drawdown_guard": StrategyConfig(enabled=True, weight=1.0, min_impact=0.0, max_impact=0.30),
    "income_turnover_discipline": StrategyConfig(enabled=True, weight=1.0, min_impact=0.0, max_impact=1.0),
}

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
    
    # Russell 3000 Wave - Broad US equity market index reference
    "Russell 3000 Wave": [
        Holding("IWV", 1.00, "iShares Russell 3000 ETF"),
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

    # New Equity Waves (Wave Universe Refactor)
    "Small Cap Growth Wave": [
        Holding("IWO", 0.40, "iShares Russell 2000 Growth ETF"),
        Holding("VBK", 0.30, "Vanguard Small-Cap Growth ETF"),
        Holding("VTWO", 0.15, "Vanguard Russell 2000 ETF"),
        Holding("IJH", 0.15, "iShares Core S&P Mid-Cap ETF"),
    ],
    "Small to Mid Cap Growth Wave": [
        Holding("IJH", 0.25, "iShares Core S&P Mid-Cap ETF"),
        Holding("IWP", 0.25, "iShares Russell Mid-Cap Growth ETF"),
        Holding("MDY", 0.20, "SPDR S&P MidCap 400 ETF"),
        Holding("IWO", 0.15, "iShares Russell 2000 Growth ETF"),
        Holding("VBK", 0.15, "Vanguard Small-Cap Growth ETF"),
    ],
    "Future Power & Energy Wave": [
        Holding("ICLN", 0.15, "iShares Global Clean Energy ETF"),
        Holding("TAN", 0.15, "Invesco Solar ETF"),
        Holding("NEE", 0.12, "NextEra Energy Inc."),
        Holding("ENPH", 0.10, "Enphase Energy Inc."),
        Holding("FSLR", 0.10, "First Solar Inc."),
        Holding("XLE", 0.10, "Energy Select Sector SPDR"),
        Holding("PLUG", 0.08, "Plug Power Inc."),
        Holding("BE", 0.08, "Bloom Energy Corp."),
        Holding("RUN", 0.07, "Sunrun Inc."),
        Holding("SEDG", 0.05, "SolarEdge Technologies Inc."),
    ],
    "Quantum Computing Wave": [
        Holding("IONQ", 0.20, "IonQ Inc."),
        Holding("RGTI", 0.15, "Rigetti Computing Inc."),
        Holding("QBTS", 0.15, "D-Wave Quantum Inc."),
        Holding("IBM", 0.15, "International Business Machines Corp."),
        Holding("GOOGL", 0.10, "Alphabet Inc. (Class A)"),
        Holding("MSFT", 0.10, "Microsoft Corp."),
        Holding("NVDA", 0.10, "NVIDIA Corp."),
        Holding("AMD", 0.05, "Advanced Micro Devices Inc."),
    ],
    "Clean Transit-Infrastructure Wave": [
        Holding("TSLA", 0.15, "Tesla Inc."),
        Holding("RIVN", 0.08, "Rivian Automotive Inc."),
        Holding("LCID", 0.07, "Lucid Group Inc."),
        Holding("GM", 0.10, "General Motors"),
        Holding("F", 0.10, "Ford Motor Co."),
        Holding("PAVE", 0.15, "Global X U.S. Infrastructure Development"),
        Holding("XLI", 0.15, "Industrial Select Sector SPDR"),
        Holding("CAT", 0.08, "Caterpillar Inc."),
        Holding("UNP", 0.07, "Union Pacific Corp."),
        Holding("CHPT", 0.05, "ChargePoint Holdings Inc."),
    ],

    # Income Wave (equity income wave)
    "Income Wave": [
        Holding("VYM", 0.25, "Vanguard High Dividend Yield ETF"),
        Holding("SCHD", 0.25, "Schwab U.S. Dividend Equity ETF"),
        Holding("DVY", 0.20, "iShares Select Dividend ETF"),
        Holding("HDV", 0.15, "iShares Core High Dividend ETF"),
        Holding("NOBL", 0.15, "ProShares S&P 500 Dividend Aristocrats ETF"),
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

    # Crypto Waves - Six Wave Lineup (5 Growth + 1 Income)
    # Growth Wave 1: Layer 1 Smart Contract Platforms
    "Crypto L1 Growth Wave": [
        Holding("ETH-USD", 0.30, "Ethereum"),
        Holding("SOL-USD", 0.20, "Solana"),
        Holding("AVAX-USD", 0.12, "Avalanche"),
        Holding("ADA-USD", 0.10, "Cardano"),
        Holding("DOT-USD", 0.10, "Polkadot"),
        Holding("NEAR-USD", 0.08, "NEAR Protocol"),
        Holding("APT-USD", 0.05, "Aptos"),
        Holding("ATOM-USD", 0.05, "Cosmos"),
    ],
    
    # Growth Wave 2: DeFi & Infrastructure
    "Crypto DeFi Growth Wave": [
        Holding("UNI-USD", 0.20, "Uniswap"),
        Holding("AAVE-USD", 0.18, "Aave"),
        Holding("LINK-USD", 0.15, "Chainlink"),
        Holding("MKR-USD", 0.12, "Maker"),
        Holding("CRV-USD", 0.10, "Curve DAO"),
        Holding("INJ-USD", 0.10, "Injective"),
        Holding("SNX-USD", 0.08, "Synthetix"),
        Holding("COMP-USD", 0.07, "Compound"),
    ],
    
    # Growth Wave 3: Layer 2 Scaling Solutions
    "Crypto L2 Growth Wave": [
        Holding("MATIC-USD", 0.30, "Polygon"),
        Holding("ARB-USD", 0.25, "Arbitrum"),
        Holding("OP-USD", 0.20, "Optimism"),
        Holding("IMX-USD", 0.12, "Immutable X"),
        Holding("MNT-USD", 0.08, "Mantle"),
        Holding("STX-USD", 0.05, "Stacks"),
    ],
    
    # Growth Wave 4: AI & Compute
    "Crypto AI Growth Wave": [
        Holding("TAO-USD", 0.25, "Bittensor"),
        Holding("RENDER-USD", 0.20, "Render Token"),
        Holding("FET-USD", 0.18, "Fetch.ai"),
        Holding("ICP-USD", 0.15, "Internet Computer"),
        Holding("OCEAN-USD", 0.12, "Ocean Protocol"),
        Holding("AGIX-USD", 0.10, "SingularityNET"),
    ],
    
    # Growth Wave 5: Broad Market Multi-Cap
    "Crypto Broad Growth Wave": [
        Holding("BTC-USD", 0.30, "Bitcoin"),
        Holding("ETH-USD", 0.25, "Ethereum"),
        Holding("BNB-USD", 0.10, "Binance Coin"),
        Holding("SOL-USD", 0.10, "Solana"),
        Holding("XRP-USD", 0.08, "XRP"),
        Holding("ADA-USD", 0.07, "Cardano"),
        Holding("AVAX-USD", 0.05, "Avalanche"),
        Holding("DOT-USD", 0.05, "Polkadot"),
    ],
    
    # Income Wave: Staking & Yield Generation
    "Crypto Income Wave": [
        Holding("ETH-USD", 0.30, "Ethereum"),
        Holding("stETH-USD", 0.15, "Lido Staked Ether"),
        Holding("LDO-USD", 0.15, "Lido DAO"),
        Holding("AAVE-USD", 0.12, "Aave"),
        Holding("MKR-USD", 0.10, "Maker"),
        Holding("UNI-USD", 0.08, "Uniswap"),
        Holding("CAKE-USD", 0.05, "PancakeSwap"),
        Holding("CRV-USD", 0.05, "Curve DAO"),
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
# Wave ID Mapping - Canonical Identifiers
# ------------------------------------------------------------
# Each wave has a unique, immutable wave_id in snake_case format.
# This serves as the canonical identifier throughout the system.
# display_name is used only for UI presentation.

WAVE_ID_REGISTRY: Dict[str, str] = {
    # wave_id -> display_name mapping
    "sp500_wave": "S&P 500 Wave",
    "russell_3000_wave": "Russell 3000 Wave",
    "us_megacap_core_wave": "US MegaCap Core Wave",
    "ai_cloud_megacap_wave": "AI & Cloud MegaCap Wave",
    "next_gen_compute_semis_wave": "Next-Gen Compute & Semis Wave",
    "future_energy_ev_wave": "Future Energy & EV Wave",
    "ev_infrastructure_wave": "EV & Infrastructure Wave",
    "us_small_cap_disruptors_wave": "US Small-Cap Disruptors Wave",
    "us_mid_small_growth_semis_wave": "US Mid/Small Growth & Semis Wave",
    "small_cap_growth_wave": "Small Cap Growth Wave",
    "small_to_mid_cap_growth_wave": "Small to Mid Cap Growth Wave",
    "future_power_energy_wave": "Future Power & Energy Wave",
    "quantum_computing_wave": "Quantum Computing Wave",
    "clean_transit_infrastructure_wave": "Clean Transit-Infrastructure Wave",
    "income_wave": "Income Wave",
    "demas_fund_wave": "Demas Fund Wave",
    "crypto_l1_growth_wave": "Crypto L1 Growth Wave",
    "crypto_defi_growth_wave": "Crypto DeFi Growth Wave",
    "crypto_l2_growth_wave": "Crypto L2 Growth Wave",
    "crypto_ai_growth_wave": "Crypto AI Growth Wave",
    "crypto_broad_growth_wave": "Crypto Broad Growth Wave",
    "crypto_income_wave": "Crypto Income Wave",
    "smartsafe_treasury_cash_wave": "SmartSafe Treasury Cash Wave",
    "smartsafe_tax_free_money_market_wave": "SmartSafe Tax-Free Money Market Wave",
    "gold_wave": "Gold Wave",
    "infinity_multi_asset_growth_wave": "Infinity Multi-Asset Growth Wave",
    "vector_treasury_ladder_wave": "Vector Treasury Ladder Wave",
    "vector_muni_ladder_wave": "Vector Muni Ladder Wave",
}

# Reverse mapping: display_name -> wave_id
DISPLAY_NAME_TO_WAVE_ID: Dict[str, str] = {v: k for k, v in WAVE_ID_REGISTRY.items()}

# Add legacy/alias mappings for backward compatibility with wave_history.csv
# These map old display names to their corresponding wave_ids
LEGACY_DISPLAY_NAME_ALIASES: Dict[str, str] = {
    "Growth Wave": "small_to_mid_cap_growth_wave",  # Historical alias
    "Small-Mid Cap Growth Wave": "small_to_mid_cap_growth_wave",  # Hyphenated variant
}

# Merge legacy aliases into DISPLAY_NAME_TO_WAVE_ID
DISPLAY_NAME_TO_WAVE_ID.update(LEGACY_DISPLAY_NAME_ALIASES)

# For backward compatibility, also map WAVE_WEIGHTS keys to wave_ids
WAVE_WEIGHTS_TO_WAVE_ID: Dict[str, str] = DISPLAY_NAME_TO_WAVE_ID.copy()

# ------------------------------------------------------------
# Strategy Family Registry
# ------------------------------------------------------------
# Canonical strategy family assignment for each wave.
# Determines which overlay logic applies to each wave.

STRATEGY_FAMILY_REGISTRY: Dict[str, str] = {
    # Equity Growth Waves
    "sp500_wave": "equity_growth",
    "russell_3000_wave": "equity_growth",
    "us_megacap_core_wave": "equity_growth",
    "ai_cloud_megacap_wave": "equity_growth",
    "next_gen_compute_semis_wave": "equity_growth",
    "future_energy_ev_wave": "equity_growth",
    "ev_infrastructure_wave": "equity_growth",
    "us_small_cap_disruptors_wave": "equity_growth",
    "us_mid_small_growth_semis_wave": "equity_growth",
    "small_cap_growth_wave": "equity_growth",
    "small_to_mid_cap_growth_wave": "equity_growth",
    "future_power_energy_wave": "equity_growth",
    "quantum_computing_wave": "equity_growth",
    "clean_transit_infrastructure_wave": "equity_growth",
    "demas_fund_wave": "equity_growth",
    "infinity_multi_asset_growth_wave": "equity_growth",
    
    # Equity Income Waves
    "income_wave": "equity_income",
    "smartsafe_treasury_cash_wave": "equity_income",
    "smartsafe_tax_free_money_market_wave": "equity_income",
    "vector_treasury_ladder_wave": "equity_income",
    "vector_muni_ladder_wave": "equity_income",
    
    # Crypto Growth Waves
    "crypto_l1_growth_wave": "crypto_growth",
    "crypto_defi_growth_wave": "crypto_growth",
    "crypto_l2_growth_wave": "crypto_growth",
    "crypto_ai_growth_wave": "crypto_growth",
    "crypto_broad_growth_wave": "crypto_growth",
    
    # Crypto Income Wave
    "crypto_income_wave": "crypto_income",
    
    # Special Waves
    "gold_wave": "special",
}

def get_strategy_family(wave_name: str) -> str:
    """
    Get the strategy family for a given wave name.
    
    Args:
        wave_name: Display name of the wave
        
    Returns:
        Strategy family string: "equity_growth", "equity_income", "crypto_growth", "crypto_income", or "special"
        Defaults to "equity_growth" for backward compatibility if wave not found.
    """
    # Get wave_id from display name
    wave_id = DISPLAY_NAME_TO_WAVE_ID.get(wave_name)
    if wave_id:
        return STRATEGY_FAMILY_REGISTRY.get(wave_id, "equity_growth")
    
    # Fallback: try to infer from wave name for backward compatibility
    if "Crypto" in wave_name:
        if wave_name == "Crypto Income Wave":
            return "crypto_income"
        return "crypto_growth"
    elif _is_income_wave(wave_name):
        return "equity_income"
    
    return "equity_growth"

# ------------------------------------------------------------
# Static benchmarks (fallback / overrides)
# ------------------------------------------------------------

BENCHMARK_WEIGHTS_STATIC: Dict[str, List[Holding]] = {
    "S&P 500 Wave": [Holding("SPY", 1.0, "SPDR S&P 500 ETF")],
    "Russell 3000 Wave": [Holding("IWV", 1.0, "iShares Russell 3000 ETF")],

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

    # New Equity Waves benchmarks
    "Small Cap Growth Wave": [
        Holding("IWO", 0.50, "iShares Russell 2000 Growth ETF"),
        Holding("VBK", 0.50, "Vanguard Small-Cap Growth ETF"),
    ],
    "Small to Mid Cap Growth Wave": [
        Holding("IWP", 0.50, "iShares Russell Mid-Cap Growth ETF"),
        Holding("IWO", 0.50, "iShares Russell 2000 Growth ETF"),
    ],
    "Future Power & Energy Wave": [
        Holding("ICLN", 0.50, "iShares Global Clean Energy ETF"),
        Holding("XLE", 0.50, "Energy Select Sector SPDR"),
    ],
    "Quantum Computing Wave": [
        Holding("QQQ", 0.60, "Invesco QQQ Trust"),
        Holding("VGT", 0.40, "Vanguard Information Technology ETF"),
    ],
    "Clean Transit-Infrastructure Wave": [
        Holding("PAVE", 0.60, "Global X U.S. Infrastructure Development"),
        Holding("XLI", 0.40, "Industrial Select Sector SPDR"),
    ],
    
    # Income Wave benchmark
    "Income Wave": [
        Holding("VYM", 0.50, "Vanguard High Dividend Yield ETF"),
        Holding("SCHD", 0.50, "Schwab U.S. Dividend Equity ETF"),
    ],

    "Demas Fund Wave": [
        Holding("SPY", 0.60, "SPDR S&P 500 ETF"),
        Holding("VTV", 0.40, "Vanguard Value ETF"),
    ],

    # ------------------------------------------------------------
    # CRYPTO BENCHMARK COMPOSITES (GOVERNED, READ-ONLY)
    # ------------------------------------------------------------
    # Per requirements: Fixed, governed composites. Avoid dynamic optimization.
    # Ensure transparency and reproducibility. Labeled as "Governed Composite."
    # Snapshot-stable across attribution windows.
    
    # 1. Crypto Broad Market Benchmark (large + mid cap)
    "Crypto Broad Market Benchmark": [
        Holding("BTC-USD", 0.45, "Bitcoin"),
        Holding("ETH-USD", 0.30, "Ethereum"),
        Holding("BNB-USD", 0.08, "Binance Coin"),
        Holding("XRP-USD", 0.08, "XRP"),
        Holding("SOL-USD", 0.05, "Solana"),
        Holding("ADA-USD", 0.04, "Cardano"),
    ],
    
    # 2. Crypto Growth Benchmark (L1s, L2s, infrastructure, AI, gaming)
    "Crypto Growth Benchmark": [
        Holding("ETH-USD", 0.30, "Ethereum"),
        Holding("SOL-USD", 0.15, "Solana"),
        Holding("AVAX-USD", 0.10, "Avalanche"),
        Holding("DOT-USD", 0.10, "Polkadot"),
        Holding("LINK-USD", 0.10, "Chainlink"),
        Holding("MATIC-USD", 0.08, "Polygon"),
        Holding("ATOM-USD", 0.07, "Cosmos"),
        Holding("NEAR-USD", 0.05, "NEAR Protocol"),
        Holding("APT-USD", 0.05, "Aptos"),
    ],
    
    # 3. Crypto Income/Yield Benchmark (staking, fee-generating assets)
    "Crypto Income Benchmark": [
        Holding("ETH-USD", 0.35, "Ethereum"),
        Holding("AAVE-USD", 0.20, "Aave"),
        Holding("UNI-USD", 0.15, "Uniswap"),
        Holding("LDO-USD", 0.10, "Lido DAO"),
        Holding("stETH-USD", 0.10, "Lido Staked Ether"),
        Holding("MKR-USD", 0.10, "Maker"),
    ],
    
    # 4. Crypto Defensive Benchmark (BTC-like / settlement assets)
    "Crypto Defensive Benchmark": [
        Holding("BTC-USD", 0.85, "Bitcoin"),
        Holding("BCH-USD", 0.10, "Bitcoin Cash"),
        Holding("LTC-USD", 0.05, "Litecoin"),
    ],
    
    # 5. Crypto DeFi Benchmark (DEXs, lending, derivatives infrastructure)
    "Crypto DeFi Benchmark": [
        Holding("UNI-USD", 0.25, "Uniswap"),
        Holding("AAVE-USD", 0.25, "Aave"),
        Holding("CAKE-USD", 0.15, "PancakeSwap"),
        Holding("CRV-USD", 0.12, "Curve DAO"),
        Holding("INJ-USD", 0.10, "Injective"),
        Holding("SNX-USD", 0.08, "Synthetix"),
        Holding("COMP-USD", 0.05, "Compound"),
    ],
    
    # ------------------------------------------------------------
    # CRYPTO WAVES (Allocate only from crypto instruments)
    # ------------------------------------------------------------
    
    # Crypto Waves → Governed Crypto Benchmarks
    # Per requirements: Reference crypto-specific benchmark composites
    # Use existing exposure, regime, and risk-control logic (unaltered)
    
    # Growth Wave 1: Layer 1 Smart Contract Platforms
    "Crypto L1 Growth Wave": [
        Holding("ETH-USD", 0.30, "Ethereum"),
        Holding("SOL-USD", 0.15, "Solana"),
        Holding("AVAX-USD", 0.10, "Avalanche"),
        Holding("DOT-USD", 0.10, "Polkadot"),
        Holding("LINK-USD", 0.10, "Chainlink"),
        Holding("MATIC-USD", 0.08, "Polygon"),
        Holding("ATOM-USD", 0.07, "Cosmos"),
        Holding("NEAR-USD", 0.05, "NEAR Protocol"),
        Holding("APT-USD", 0.05, "Aptos"),
    ],
    
    # Growth Wave 2: DeFi & Infrastructure
    "Crypto DeFi Growth Wave": [
        Holding("UNI-USD", 0.25, "Uniswap"),
        Holding("AAVE-USD", 0.25, "Aave"),
        Holding("CAKE-USD", 0.15, "PancakeSwap"),
        Holding("CRV-USD", 0.12, "Curve DAO"),
        Holding("INJ-USD", 0.10, "Injective"),
        Holding("SNX-USD", 0.08, "Synthetix"),
        Holding("COMP-USD", 0.05, "Compound"),
    ],
    
    # Growth Wave 3: Layer 2 Scaling Solutions
    "Crypto L2 Growth Wave": [
        Holding("MATIC-USD", 0.35, "Polygon"),
        Holding("ARB-USD", 0.25, "Arbitrum"),
        Holding("OP-USD", 0.20, "Optimism"),
        Holding("IMX-USD", 0.12, "Immutable X"),
        Holding("MNT-USD", 0.05, "Mantle"),
        Holding("STX-USD", 0.03, "Stacks"),
    ],
    
    # Growth Wave 4: AI & Compute
    "Crypto AI Growth Wave": [
        Holding("TAO-USD", 0.25, "Bittensor"),
        Holding("RENDER-USD", 0.20, "Render Token"),
        Holding("FET-USD", 0.18, "Fetch.ai"),
        Holding("ICP-USD", 0.15, "Internet Computer"),
        Holding("OCEAN-USD", 0.12, "Ocean Protocol"),
        Holding("AGIX-USD", 0.10, "SingularityNET"),
    ],
    
    # Growth Wave 5: Broad Market Multi-Cap
    "Crypto Broad Growth Wave": [
        Holding("BTC-USD", 0.45, "Bitcoin"),
        Holding("ETH-USD", 0.30, "Ethereum"),
        Holding("BNB-USD", 0.08, "Binance Coin"),
        Holding("XRP-USD", 0.08, "XRP"),
        Holding("SOL-USD", 0.05, "Solana"),
        Holding("ADA-USD", 0.04, "Cardano"),
    ],

    # Crypto Income Wave benchmark (uses CSE basket - Crypto Income Benchmark)
    "Crypto Income Wave": [
        Holding("ETH-USD", 0.35, "Ethereum"),
        Holding("AAVE-USD", 0.20, "Aave"),
        Holding("UNI-USD", 0.15, "Uniswap"),
        Holding("LDO-USD", 0.10, "Lido DAO"),
        Holding("stETH-USD", 0.10, "Lido Staked Ether"),
        Holding("MKR-USD", 0.10, "Maker"),
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
    
    # Crypto benchmark candidates (governed composites)
    ETFBenchmarkCandidate("BTC-USD", "Bitcoin", {"Crypto", "Defensive", "Settlement"}, "Crypto"),
    ETFBenchmarkCandidate("ETH-USD", "Ethereum", {"Crypto", "Growth", "L1"}, "Crypto"),
    ETFBenchmarkCandidate("BITO", "ProShares Bitcoin Strategy ETF", {"Crypto"}, "Crypto"),
    
    # Safe assets
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
    """Get all wave display names (for backward compatibility)."""
    return sorted(WAVE_WEIGHTS.keys())


def get_all_wave_ids() -> list[str]:
    """Get all canonical wave_ids."""
    return sorted(WAVE_ID_REGISTRY.keys())


def get_all_waves_universe() -> dict:
    """
    Get the complete wave universe from the canonical registry.
    
    This is the SINGLE SOURCE OF TRUTH for all waves in the system.
    All parts of the application MUST use this function instead of ad-hoc lists.
    
    Returns:
        Dictionary containing:
        - 'waves': List of all wave display names (28 total)
        - 'wave_ids': List of all canonical wave_ids (28 total)
        - 'count': Total number of waves
        - 'source': Source identifier ('wave_registry')
        - 'version': Registry version for cache busting
    
    Raises:
        ValueError: If WAVE_WEIGHTS and WAVE_ID_REGISTRY are inconsistent
    
    Example:
        >>> universe = get_all_waves_universe()
        >>> print(f"Total waves: {universe['count']}")
        >>> for wave in universe['waves']:
        ...     print(wave)
    """
    waves = sorted(WAVE_WEIGHTS.keys())
    wave_ids = sorted(WAVE_ID_REGISTRY.keys())
    
    # Validate consistency between registries
    if len(waves) != len(wave_ids):
        warnings = validate_wave_id_registry()
        raise ValueError(
            f"Inconsistent registry state: WAVE_WEIGHTS has {len(waves)} waves "
            f"but WAVE_ID_REGISTRY has {len(wave_ids)} wave_ids. "
            f"Validation warnings: {warnings}"
        )
    
    return {
        'waves': waves,
        'wave_ids': wave_ids,
        'count': len(wave_ids),  # Use wave_ids count for consistency
        'source': 'wave_registry',
        'version': 1  # Increment when registry structure changes
    }


def get_wave_id_from_display_name(display_name: str) -> Optional[str]:
    """
    Convert display_name to wave_id.
    
    Args:
        display_name: Human-readable wave name (e.g., "S&P 500 Wave")
        
    Returns:
        wave_id (e.g., "sp500_wave") or None if not found
    """
    return DISPLAY_NAME_TO_WAVE_ID.get(display_name)


def get_display_name_from_wave_id(wave_id: str) -> Optional[str]:
    """
    Convert wave_id to display_name.
    
    Args:
        wave_id: Canonical wave identifier (e.g., "sp500_wave")
        
    Returns:
        display_name (e.g., "S&P 500 Wave") or None if not found
    """
    return WAVE_ID_REGISTRY.get(wave_id)


def validate_wave_id_registry() -> List[str]:
    """
    Validate wave_id registry for duplicates and missing mappings.
    
    Returns:
        List of warning messages (empty if all valid)
    """
    warnings = []
    
    # Check for duplicate wave_ids
    wave_ids = list(WAVE_ID_REGISTRY.keys())
    if len(wave_ids) != len(set(wave_ids)):
        duplicates = [wid for wid in wave_ids if wave_ids.count(wid) > 1]
        warnings.append(f"Duplicate wave_ids found: {set(duplicates)}")
    
    # Check for duplicate display_names
    display_names = list(WAVE_ID_REGISTRY.values())
    if len(display_names) != len(set(display_names)):
        duplicates = [dn for dn in display_names if display_names.count(dn) > 1]
        warnings.append(f"Duplicate display_names found: {set(duplicates)}")
    
    # Check that all WAVE_WEIGHTS keys have a wave_id mapping
    for wave_name in WAVE_WEIGHTS.keys():
        if wave_name not in DISPLAY_NAME_TO_WAVE_ID:
            warnings.append(f"Wave '{wave_name}' in WAVE_WEIGHTS has no wave_id mapping")
    
    # Check that all wave_ids have corresponding WAVE_WEIGHTS entries
    for wave_id, display_name in WAVE_ID_REGISTRY.items():
        if display_name not in WAVE_WEIGHTS:
            warnings.append(f"wave_id '{wave_id}' -> '{display_name}' has no WAVE_WEIGHTS entry")
    
    return warnings


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


def _normalize_ticker(ticker: str) -> str:
    """
    Normalize ticker symbol to handle common formatting issues.
    Uses TICKER_ALIASES for known symbol variants.
    
    Args:
        ticker: Original ticker symbol
        
    Returns:
        Normalized ticker symbol
    """
    if not ticker:
        return ticker
    
    # Convert to uppercase and strip whitespace
    normalized = ticker.strip().upper()
    
    # Check if ticker has a known alias
    if normalized in TICKER_ALIASES:
        return TICKER_ALIASES[normalized]
    
    # Replace dots with hyphens (e.g., BRK.B → BRK-B) for any ticker not in aliases
    normalized = normalized.replace('.', '-')
    
    # Check again after dot replacement
    if normalized in TICKER_ALIASES:
        return TICKER_ALIASES[normalized]
    
    return normalized


def _is_rate_limit_error(error: Exception) -> bool:
    """
    Check if an error is a rate limit error.
    
    Args:
        error: Exception to check
        
    Returns:
        True if error is rate-limit related
    """
    error_str = str(error).lower()
    return any(keyword in error_str for keyword in ['rate limit', 'too many requests', '429', 'quota'])


def _retry_with_backoff(func, max_retries: int = 3, initial_delay: float = 1.0, backoff_factor: float = 2.0):
    """
    Retry a function with exponential backoff.
    
    Args:
        func: Function to retry
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay in seconds
        backoff_factor: Multiplier for delay between retries
        
    Returns:
        Result of the function call
        
    Raises:
        Last exception if all retries fail
    """
    delay = initial_delay
    last_exception = None
    
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            last_exception = e
            
            # Check if it's a rate limit error (don't retry immediately)
            if _is_rate_limit_error(e):
                delay = max(delay * 2, 5.0)  # Longer delay for rate limits
            
            if attempt < max_retries - 1:
                time.sleep(delay)
                delay *= backoff_factor
    
    # All retries failed, raise the last exception
    raise last_exception


def _log_diagnostics_to_json(failures: Dict[str, str], wave_id: Optional[str] = None, wave_name: Optional[str] = None):
    """
    Log failed ticker diagnostics to a JSON file.
    
    Args:
        failures: Dict mapping tickers to error messages
        wave_id: Optional wave identifier
        wave_name: Optional wave display name
    """
    if not failures:
        return
    
    # Create logs directory if it doesn't exist
    log_dir = "logs/diagnostics"
    os.makedirs(log_dir, exist_ok=True)
    
    # Create timestamped log entry
    timestamp = datetime.now().isoformat()
    log_entry = {
        "timestamp": timestamp,
        "wave_id": wave_id or "unknown",
        "wave_name": wave_name or "unknown",
        "failures": []
    }
    
    # Categorize each failure
    for ticker, error_msg in failures.items():
        failure_type, suggested_fix = categorize_error(error_msg, ticker) if DIAGNOSTICS_AVAILABLE else (None, "")
        
        log_entry["failures"].append({
            "ticker_original": ticker,
            "ticker_normalized": _normalize_ticker(ticker),
            "error_message": error_msg,
            "failure_type": failure_type.value if failure_type else "UNKNOWN_ERROR",
            "suggested_fix": suggested_fix,
            "is_fatal": True
        })
    
    # Append to log file (one file per day)
    date_str = datetime.now().strftime('%Y%m%d')
    log_file = os.path.join(log_dir, f"failed_tickers_{date_str}.json")
    
    # Load existing entries if file exists
    entries = []
    if os.path.exists(log_file):
        try:
            with open(log_file, 'r') as f:
                entries = json.load(f)
        except (json.JSONDecodeError, IOError):
            entries = []
    
    # Append new entry
    entries.append(log_entry)
    
    # Write back to file
    try:
        with open(log_file, 'w') as f:
            json.dump(entries, f, indent=2)
    except IOError as e:
        print(f"Warning: Failed to write diagnostics log: {e}")


def _download_history(tickers: list[str], days: int, wave_id: Optional[str] = None, wave_name: Optional[str] = None) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """
    Download historical price data with per-ticker isolation and graceful error handling.
    
    Enhanced Features:
    - Retry logic with exponential backoff for transient failures
    - Ticker normalization (e.g., BRK.B → BRK-B) to handle common issues
    - Diagnostics tracking with categorized failure types
    - Explicit handling for delisted tickers (permanent failures)
    - API throttling/rate limiting protection
    - JSON-based logging of failed tickers with timestamps
    - LRU cache with 15-minute TTL to prevent redundant API calls
    - Per-ticker error isolation - partial success is acceptable
    - Falls back to individual ticker fetching if batch fails
    - Returns partial data instead of failing completely
    
    Args:
        tickers: List of ticker symbols to download
        days: Number of days of historical data to fetch
        wave_id: Optional wave identifier for diagnostics tracking
        wave_name: Optional wave display name for diagnostics tracking
    
    Returns:
        Tuple of (prices_df, failures_dict):
        - prices_df: DataFrame with dates as index and tickers as columns
        - failures_dict: Dict mapping failed tickers to error reasons
    """
    failures = {}
    
    if yf is None:
        print("Error: yfinance is not available in this environment.")
        failures = {ticker: "yfinance not available" for ticker in tickers}
        # Log diagnostics
        _log_diagnostics_to_json(failures, wave_id, wave_name)
        # Track in diagnostics if available
        if DIAGNOSTICS_AVAILABLE:
            tracker = get_diagnostics_tracker()
            for ticker in tickers:
                failure_type, suggested_fix = categorize_error(failures[ticker], ticker)
                report = FailedTickerReport(
                    ticker_original=ticker,
                    ticker_normalized=_normalize_ticker(ticker),
                    wave_id=wave_id,
                    wave_name=wave_name,
                    source="yfinance",
                    failure_type=failure_type,
                    error_message=failures[ticker],
                    is_fatal=True,
                    suggested_fix=suggested_fix
                )
                tracker.record_failure(report)
        return pd.DataFrame(), failures
    
    # Normalize tickers before fetching
    normalized_tickers = []
    for ticker in tickers:
        normalized = _normalize_ticker(ticker)
        normalized_tickers.append(normalized)
    
    # Remove duplicates while preserving order
    seen = set()
    unique_normalized = []
    for ticker in normalized_tickers:
        if ticker not in seen:
            seen.add(ticker)
            unique_normalized.append(ticker)
    
    lookback_days = days + 260
    end = datetime.utcnow().date()
    start = end - timedelta(days=lookback_days)
    
    # Try batch download first with retry logic
    try:
        def _batch_download():
            return yf.download(
                tickers=unique_normalized,
                start=start.isoformat(),
                end=end.isoformat(),
                interval="1d",
                auto_adjust=True,
                progress=False,
                group_by="column",
            )
        
        # Use retry with backoff for batch download
        data = _retry_with_backoff(_batch_download, max_retries=3, initial_delay=1.0)
        
        if data is None or len(data) == 0:
            # Fall back to individual ticker fetching
            print(f"Warning: Batch download returned no data, trying individual tickers")
            return _download_history_individually(unique_normalized, start, end, wave_id, wave_name)
        
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
        
        # Check if we got at least some data
        if data.empty:
            print(f"Warning: No price data after normalization, trying individual tickers")
            return _download_history_individually(unique_normalized, start, end, wave_id, wave_name)
        
        # Track which tickers failed in batch download
        available_tickers = set(data.columns)
        for ticker in unique_normalized:
            if ticker not in available_tickers:
                failures[ticker] = "Not in batch result"
        
        # Log diagnostics if there are failures
        if failures:
            _log_diagnostics_to_json(failures, wave_id, wave_name)
            # Track in diagnostics if available
            if DIAGNOSTICS_AVAILABLE:
                tracker = get_diagnostics_tracker()
                for ticker, error_msg in failures.items():
                    failure_type, suggested_fix = categorize_error(error_msg, ticker)
                    report = FailedTickerReport(
                        ticker_original=ticker,
                        ticker_normalized=ticker,  # Already normalized
                        wave_id=wave_id,
                        wave_name=wave_name,
                        source="yfinance",
                        failure_type=failure_type,
                        error_message=error_msg,
                        is_fatal=False,  # Might succeed in individual download
                        suggested_fix=suggested_fix
                    )
                    tracker.record_failure(report)
        
        return data, failures
        
    except Exception as e:
        # Graceful degradation on rate limits or other errors
        # Try individual ticker fetching as fallback
        error_msg = f"Batch download failed: {str(e)}"
        print(f"Warning: yfinance batch download failed, trying individual tickers: {str(e)}")
        
        # Check if this is a rate limit error
        if _is_rate_limit_error(e):
            # Add a delay before trying individual downloads
            time.sleep(5.0)
        
        return _download_history_individually(unique_normalized, start, end, wave_id, wave_name)


def _download_history_individually(tickers: list[str], start, end, wave_id: Optional[str] = None, wave_name: Optional[str] = None) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """
    Download price data one ticker at a time for maximum resilience.
    
    Enhanced Features:
    - Retry logic with exponential backoff for each ticker
    - Batch delays to prevent API rate limiting
    - Diagnostics tracking for all failure types
    - Explicit categorization of permanent vs transient failures
    - JSON-based logging of failures
    
    Args:
        tickers: List of ticker symbols (should already be normalized)
        start: Start date
        end: End date
        wave_id: Optional wave identifier for diagnostics tracking
        wave_name: Optional wave display name for diagnostics tracking
        
    Returns:
        Tuple of (prices_df, failures_dict):
        - prices_df: DataFrame with dates as index and tickers as columns
        - failures_dict: Dict mapping failed tickers to error reasons
    """
    if yf is None:
        failures = {ticker: "yfinance not available" for ticker in tickers}
        # Log diagnostics
        _log_diagnostics_to_json(failures, wave_id, wave_name)
        # Track in diagnostics if available
        if DIAGNOSTICS_AVAILABLE:
            tracker = get_diagnostics_tracker()
            for ticker in tickers:
                failure_type, suggested_fix = categorize_error(failures[ticker], ticker)
                report = FailedTickerReport(
                    ticker_original=ticker,
                    ticker_normalized=ticker,
                    wave_id=wave_id,
                    wave_name=wave_name,
                    source="yfinance",
                    failure_type=failure_type,
                    error_message=failures[ticker],
                    is_fatal=True,
                    suggested_fix=suggested_fix
                )
                tracker.record_failure(report)
        return pd.DataFrame(), failures
    
    all_prices = {}
    failures = {}
    
    # Get diagnostics tracker if available
    tracker = get_diagnostics_tracker() if DIAGNOSTICS_AVAILABLE else None
    
    # Add batch processing with delays to reduce API stress
    batch_size = 10
    batch_delay = 0.5  # 0.5 seconds between batches
    
    for i, ticker in enumerate(tickers):
        # Add delay between batches to prevent rate limiting
        if i > 0 and i % batch_size == 0:
            time.sleep(batch_delay)
        
        try:
            def _download_single():
                return yf.download(
                    tickers=ticker,
                    start=start.isoformat(),
                    end=end.isoformat(),
                    interval="1d",
                    auto_adjust=True,
                    progress=False,
                )
            
            # Use retry with backoff for each ticker
            data = _retry_with_backoff(_download_single, max_retries=3, initial_delay=1.0)
            
            if data is None or data.empty:
                error_msg = "Empty data returned"
                failures[ticker] = error_msg
                
                # Track in diagnostics
                if tracker:
                    failure_type, suggested_fix = categorize_error(error_msg, ticker)
                    is_fatal = failure_type.value in ["SYMBOL_INVALID", "PROVIDER_EMPTY"]
                    report = FailedTickerReport(
                        ticker_original=ticker,
                        ticker_normalized=ticker,  # Already normalized
                        wave_id=wave_id,
                        wave_name=wave_name,
                        source="yfinance",
                        failure_type=failure_type,
                        error_message=error_msg,
                        is_fatal=is_fatal,
                        suggested_fix=suggested_fix
                    )
                    tracker.record_failure(report)
                continue
            
            if 'Close' in data.columns:
                all_prices[ticker] = data['Close']
            elif 'Adj Close' in data.columns:
                all_prices[ticker] = data['Adj Close']
            else:
                error_msg = "No Close column"
                failures[ticker] = error_msg
                
                # Track in diagnostics
                if tracker:
                    failure_type, suggested_fix = categorize_error(error_msg, ticker)
                    report = FailedTickerReport(
                        ticker_original=ticker,
                        ticker_normalized=ticker,  # Already normalized
                        wave_id=wave_id,
                        wave_name=wave_name,
                        source="yfinance",
                        failure_type=failure_type,
                        error_message=error_msg,
                        is_fatal=True,
                        suggested_fix=suggested_fix
                    )
                    tracker.record_failure(report)
                continue
                
        except Exception as e:
            error_msg = f"Download error: {str(e)}"
            failures[ticker] = error_msg
            
            # Track in diagnostics
            if tracker:
                failure_type, suggested_fix = categorize_error(error_msg, ticker)
                # Determine if this is a fatal error
                is_fatal = failure_type.value not in ["RATE_LIMIT", "NETWORK_TIMEOUT"]
                report = FailedTickerReport(
                    ticker_original=ticker,
                    ticker_normalized=ticker,  # Already normalized
                    wave_id=wave_id,
                    wave_name=wave_name,
                    source="yfinance",
                    failure_type=failure_type,
                    error_message=error_msg,
                    is_fatal=is_fatal,
                    suggested_fix=suggested_fix
                )
                tracker.record_failure(report)
            continue
    
    if failures:
        print(f"Warning: {len(failures)} ticker(s) failed to download: {list(failures.keys())[:5]}{'...' if len(failures) > 5 else ''}")
        # Log diagnostics to JSON
        _log_diagnostics_to_json(failures, wave_id, wave_name)
    
    if not all_prices:
        print("Error: No tickers successfully downloaded")
        return pd.DataFrame(), failures
    
    # Build DataFrame
    prices_df = pd.DataFrame(all_prices)
    prices_df = prices_df.sort_index().ffill().bfill()
    
    print(f"Successfully downloaded {len(all_prices)}/{len(tickers)} tickers")
    return prices_df, failures


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
    # Explicit override: Bitcoin Wave uses BTC as difficulty reference benchmark
    # (not passive beta or static asset allocation guidance)
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


# ------------------------------------------------------------
# Crypto-specific helper functions
# ------------------------------------------------------------

def _is_crypto_wave(wave_name: str) -> bool:
    """Determine if a wave is a crypto wave (distinct from equity waves)."""
    return CRYPTO_WAVE_KEYWORD in wave_name or wave_name == "Bitcoin Wave"


def _is_crypto_growth_wave(wave_name: str) -> bool:
    """Determine if a wave is a crypto growth wave."""
    if not _is_crypto_wave(wave_name):
        return False
    # Crypto Income Wave is NOT a growth wave
    return "Income" not in wave_name


def _is_crypto_income_wave(wave_name: str) -> bool:
    """Determine if a wave is the crypto income wave."""
    return wave_name == "Crypto Income Wave"


def _is_income_wave(wave_name: str) -> bool:
    """
    Determine if a wave is a non-crypto Income/Dividend/Yield/Defensive/Bond-like wave.
    Income waves use distinct income-first overlays instead of equity growth overlays.
    """
    # List of non-crypto income waves
    income_waves = {
        "Income Wave",
        "Vector Treasury Ladder Wave",
        "Vector Muni Ladder Wave",
        "SmartSafe Treasury Cash Wave",
        "SmartSafe Tax-Free Money Market Wave",
    }
    return wave_name in income_waves


def _crypto_trend_regime(trend_60d: float) -> str:
    """
    Classify crypto trend regime based on 60-day momentum.
    Uses crypto-specific thresholds (higher volatility than equities).
    """
    if np.isnan(trend_60d):
        return "neutral"
    if trend_60d >= CRYPTO_TREND_MOMENTUM_THRESHOLDS["strong_uptrend"]:
        return "strong_uptrend"
    if trend_60d >= CRYPTO_TREND_MOMENTUM_THRESHOLDS["uptrend"]:
        return "uptrend"
    if trend_60d <= CRYPTO_TREND_MOMENTUM_THRESHOLDS["strong_downtrend"]:
        return "strong_downtrend"
    if trend_60d <= CRYPTO_TREND_MOMENTUM_THRESHOLDS["downtrend"]:
        return "downtrend"
    return "neutral"


def _crypto_volatility_state(realized_vol: float) -> str:
    """
    Classify crypto volatility state based on realized volatility.
    Returns: extreme_compression, compression, normal, expansion, extreme_expansion
    """
    if np.isnan(realized_vol) or realized_vol <= 0:
        return "normal"
    if realized_vol < CRYPTO_VOL_THRESHOLDS["extreme_compression"]:
        return "extreme_compression"
    if realized_vol < CRYPTO_VOL_THRESHOLDS["compression"]:
        return "compression"
    if realized_vol < CRYPTO_VOL_THRESHOLDS["normal"]:
        return "normal"
    if realized_vol < CRYPTO_VOL_THRESHOLDS["expansion"]:
        return "expansion"
    return "extreme_expansion"


def _crypto_liquidity_state(volume_ratio: float) -> str:
    """
    Classify crypto liquidity/market structure state.
    volume_ratio = current volume / average volume
    Returns: strong_volume, normal_volume, weak_volume
    """
    if np.isnan(volume_ratio) or volume_ratio <= 0:
        return "normal_volume"
    if volume_ratio >= CRYPTO_LIQUIDITY_THRESHOLDS["strong_volume"]:
        return "strong_volume"
    if volume_ratio >= CRYPTO_LIQUIDITY_THRESHOLDS["normal_volume"]:
        return "normal_volume"
    return "weak_volume"


def _crypto_trend_momentum_overlay(trend_60d: float) -> tuple[float, float, str]:
    """
    Crypto Trend/Momentum Regime Overlay (for growth waves).
    Returns: (exposure_multiplier, safe_fraction_impact, regime)
    """
    regime = _crypto_trend_regime(trend_60d)
    exposure = CRYPTO_TREND_EXPOSURE.get(regime, 1.0)
    
    # Safe fraction impact: increase safe allocation in downtrends
    safe_impact = 0.0
    if regime == "strong_downtrend":
        safe_impact = 0.30  # 30% to safe in strong downtrend
    elif regime == "downtrend":
        safe_impact = 0.15  # 15% to safe in downtrend
    
    return (float(exposure), float(safe_impact), regime)


def _crypto_volatility_overlay(realized_vol: float) -> tuple[float, str]:
    """
    Crypto Volatility State Overlay (for growth waves).
    Returns: (exposure_multiplier, vol_state)
    """
    vol_state = _crypto_volatility_state(realized_vol)
    exposure = CRYPTO_VOL_EXPOSURE.get(vol_state, 1.0)
    return (float(exposure), vol_state)


def _crypto_liquidity_overlay(volume_ratio: float) -> tuple[float, str]:
    """
    Crypto Liquidity/Market Structure Overlay (for all crypto waves).
    Returns: (exposure_multiplier, liquidity_state)
    """
    liquidity_state = _crypto_liquidity_state(volume_ratio)
    exposure = CRYPTO_LIQUIDITY_EXPOSURE.get(liquidity_state, 1.0)
    return (float(exposure), liquidity_state)


def _crypto_income_stability_overlay(realized_vol: float, trend_60d: float) -> tuple[float, float]:
    """
    Crypto Income Wave: Yield Stability Overlay.
    Prioritizes capital preservation with conservative exposure controls.
    Returns: (capped_exposure_multiplier, safe_fraction_boost)
    """
    # Start with conservative baseline
    exposure = 0.80  # 80% baseline exposure (conservative)
    safe_boost = CRYPTO_INCOME_SAFE_FRACTION["baseline"]
    
    # Detect stress conditions
    is_high_vol = realized_vol > 0.80 if not np.isnan(realized_vol) else False
    is_downtrend = trend_60d < -0.05 if not np.isnan(trend_60d) else False
    
    # Apply stress adjustments
    if is_high_vol or is_downtrend:
        exposure *= 0.75  # Further reduce exposure in stress
        safe_boost += CRYPTO_INCOME_SAFE_FRACTION["stress_boost"]
    
    # Apply conservative caps
    exposure = float(np.clip(
        exposure,
        CRYPTO_INCOME_EXPOSURE_CAP["min_exposure"],
        CRYPTO_INCOME_EXPOSURE_CAP["max_exposure"]
    ))
    safe_boost = float(np.clip(safe_boost, 0.0, 0.60))
    
    return (exposure, safe_boost)


def _crypto_income_drawdown_guard(wave_ret_list: list[float]) -> tuple[float, str]:
    """
    Crypto Income Wave: Drawdown Guard Overlay.
    Monitors drawdowns and increases safe allocation during sharp declines.
    Returns: (safe_fraction_boost, stress_state)
    """
    if len(wave_ret_list) < 10:
        return (0.0, "normal")
    
    # Calculate drawdown from recent peak
    recent_returns = np.array(wave_ret_list[-30:])  # Last 30 days
    cumulative = (1 + recent_returns).cumprod()
    running_max = np.maximum.accumulate(cumulative)
    drawdown = (cumulative[-1] / running_max[-1]) - 1.0
    
    # Classify stress state
    if drawdown <= CRYPTO_INCOME_DRAWDOWN_THRESHOLDS["severe"]:
        stress_state = "severe"
        safe_boost = CRYPTO_INCOME_DRAWDOWN_SAFE_BOOST["severe"]
    elif drawdown <= CRYPTO_INCOME_DRAWDOWN_THRESHOLDS["moderate"]:
        stress_state = "moderate"
        safe_boost = CRYPTO_INCOME_DRAWDOWN_SAFE_BOOST["moderate"]
    elif drawdown <= CRYPTO_INCOME_DRAWDOWN_THRESHOLDS["minor"]:
        stress_state = "minor"
        safe_boost = CRYPTO_INCOME_DRAWDOWN_SAFE_BOOST["minor"]
    else:
        stress_state = "normal"
        safe_boost = 0.0
    
    return (float(safe_boost), stress_state)


def _crypto_income_liquidity_gate(volume_ratio: float) -> tuple[float, str]:
    """
    Crypto Income Wave: Liquidity Gate Overlay (more conservative than growth).
    Returns: (exposure_multiplier, liquidity_state)
    """
    if np.isnan(volume_ratio) or volume_ratio <= 0:
        return (1.0, "normal_volume")
    
    if volume_ratio >= CRYPTO_INCOME_LIQUIDITY_THRESHOLDS["strong_volume"]:
        liquidity_state = "strong_volume"
        exposure = CRYPTO_INCOME_LIQUIDITY_EXPOSURE["strong_volume"]
    elif volume_ratio >= CRYPTO_INCOME_LIQUIDITY_THRESHOLDS["normal_volume"]:
        liquidity_state = "normal_volume"
        exposure = CRYPTO_INCOME_LIQUIDITY_EXPOSURE["normal_volume"]
    else:
        liquidity_state = "weak_volume"
        exposure = CRYPTO_INCOME_LIQUIDITY_EXPOSURE["weak_volume"]
    
    return (float(exposure), liquidity_state)


# ------------------------------------------------------------
# Income wave overlay functions (non-crypto income waves)
# ------------------------------------------------------------

def _rates_duration_regime(tnx_trend: float) -> str:
    """
    Classify rates/duration regime based on TNX (10-year Treasury) trend.
    Returns: rising_fast, rising, stable, falling, falling_fast
    """
    if np.isnan(tnx_trend):
        return "stable"
    if tnx_trend >= INCOME_RATES_REGIME_THRESHOLDS["rising_fast"]:
        return "rising_fast"
    if tnx_trend >= INCOME_RATES_REGIME_THRESHOLDS["rising"]:
        return "rising"
    if tnx_trend >= INCOME_RATES_REGIME_THRESHOLDS["stable"]:
        return "stable"  # Between -0.03 and +0.03
    if tnx_trend >= INCOME_RATES_REGIME_THRESHOLDS["falling"]:
        return "falling"  # Between -0.10 and -0.03
    return "falling_fast"  # Below -0.10


def _rates_duration_overlay(tnx_trend: float) -> tuple[float, str]:
    """
    Income Wave: Rates/Duration Regime Overlay.
    Monitors TNX trends to calibrate duration sensitivity.
    Returns: (exposure_multiplier, regime)
    """
    regime = _rates_duration_regime(tnx_trend)
    exposure = INCOME_RATES_EXPOSURE.get(regime, 1.0)
    return (float(exposure), regime)


def _credit_risk_regime(hyg_lqd_spread: float) -> str:
    """
    Classify credit/risk regime based on HYG vs LQD relative performance.
    Positive spread = HYG outperforming (risk on)
    Negative spread = LQD outperforming (risk off)
    Returns: risk_on, neutral, risk_off
    """
    if np.isnan(hyg_lqd_spread):
        return "neutral"
    if hyg_lqd_spread >= INCOME_CREDIT_REGIME_THRESHOLDS["risk_on"]:
        return "risk_on"
    if hyg_lqd_spread <= INCOME_CREDIT_REGIME_THRESHOLDS["neutral"]:
        return "risk_off"
    return "neutral"


def _credit_risk_overlay(hyg_lqd_spread: float) -> tuple[float, float, str]:
    """
    Income Wave: Credit/Risk Regime Overlay.
    Evaluates credit risk via HYG vs LQD to assess credit stress.
    Returns: (exposure_multiplier, safe_fraction_boost, regime)
    """
    regime = _credit_risk_regime(hyg_lqd_spread)
    exposure = INCOME_CREDIT_EXPOSURE.get(regime, 1.0)
    safe_boost = INCOME_CREDIT_SAFE_BOOST.get(regime, 0.0)
    return (float(exposure), float(safe_boost), regime)


def _drawdown_guard_overlay(current_nav: float, peak_nav: float, recent_vol: float, vol_threshold: float = 0.15) -> tuple[float, str]:
    """
    Income Wave: Carry + Drawdown Guard Overlay.
    Protects against drawdowns and volatility spikes.
    Returns: (safe_fraction_boost, stress_state)
    """
    # Calculate current drawdown
    if peak_nav > 0:
        drawdown = (current_nav / peak_nav) - 1.0
    else:
        drawdown = 0.0
    
    # Check volatility spike
    is_vol_spike = recent_vol > vol_threshold if not np.isnan(recent_vol) else False
    
    # Determine stress level and safe boost
    safe_boost = 0.0
    stress_state = "normal"
    
    if drawdown <= INCOME_DRAWDOWN_THRESHOLDS["severe"] or is_vol_spike:
        safe_boost = INCOME_DRAWDOWN_SAFE_BOOST["severe"]
        stress_state = "severe"
    elif drawdown <= INCOME_DRAWDOWN_THRESHOLDS["moderate"]:
        safe_boost = INCOME_DRAWDOWN_SAFE_BOOST["moderate"]
        stress_state = "moderate"
    elif drawdown <= INCOME_DRAWDOWN_THRESHOLDS["minor"]:
        safe_boost = INCOME_DRAWDOWN_SAFE_BOOST["minor"]
        stress_state = "minor"
    
    return (float(safe_boost), stress_state)


def _calculate_price_return(price_series: pd.Series, dt: pd.Timestamp, periods: int) -> float:
    """
    Helper function to calculate price return over specified periods.
    Returns NaN if insufficient data or date not in index.
    
    Args:
        price_series: Price time series
        dt: Current date
        periods: Number of periods to look back
        
    Returns:
        Return as decimal (e.g., 0.05 = 5%) or NaN if insufficient data
    """
    if len(price_series) < periods:
        return np.nan
    
    ret_series = price_series / price_series.shift(periods) - 1.0
    
    if dt not in ret_series.index:
        return np.nan
    
    ret_val = ret_series.loc[dt]
    return float(ret_val) if not np.isnan(ret_val) else np.nan


# ------------------------------------------------------------
# Strategy-specific computation functions
# ------------------------------------------------------------

def _compute_trend_confirmation_strategy(
    price_df: pd.DataFrame,
    base_index_ticker: str,
    dt: pd.Timestamp,
    config: StrategyConfig
) -> StrategyContribution:
    """
    Trend confirmation: validates regime with multiple timeframes (20D, 60D, 120D).
    Returns exposure multiplier based on trend alignment.
    """
    if not config.enabled:
        return StrategyContribution(
            name="trend_confirmation",
            exposure_impact=1.0,
            safe_fraction_impact=0.0,
            risk_state="neutral",
            enabled=False
        )
    
    if base_index_ticker not in price_df.columns:
        return StrategyContribution(
            name="trend_confirmation",
            exposure_impact=1.0,
            safe_fraction_impact=0.0,
            risk_state="neutral",
            metadata={"reason": "no_index_data"}
        )
    
    idx_price = price_df[base_index_ticker]
    
    # Calculate multiple timeframe trends
    ret_20d = idx_price / idx_price.shift(20) - 1.0 if len(idx_price) >= 20 else pd.Series()
    ret_60d = idx_price / idx_price.shift(60) - 1.0 if len(idx_price) >= 60 else pd.Series()
    ret_120d = idx_price / idx_price.shift(120) - 1.0 if len(idx_price) >= 120 else pd.Series()
    
    trend_scores = []
    metadata = {}
    
    # 20D trend (short-term)
    if dt in ret_20d.index and not np.isnan(ret_20d.loc[dt]):
        r20 = float(ret_20d.loc[dt])
        metadata["ret_20d"] = r20
        if r20 > 0.02:
            trend_scores.append(1.0)
        elif r20 < -0.02:
            trend_scores.append(-1.0)
        else:
            trend_scores.append(0.0)
    
    # 60D trend (medium-term)
    if dt in ret_60d.index and not np.isnan(ret_60d.loc[dt]):
        r60 = float(ret_60d.loc[dt])
        metadata["ret_60d"] = r60
        if r60 > 0.06:
            trend_scores.append(1.0)
        elif r60 < -0.04:
            trend_scores.append(-1.0)
        else:
            trend_scores.append(0.0)
    
    # 120D trend (long-term)
    if dt in ret_120d.index and not np.isnan(ret_120d.loc[dt]):
        r120 = float(ret_120d.loc[dt])
        metadata["ret_120d"] = r120
        if r120 > 0.10:
            trend_scores.append(1.0)
        elif r120 < -0.08:
            trend_scores.append(-1.0)
        else:
            trend_scores.append(0.0)
    
    if not trend_scores:
        return StrategyContribution(
            name="trend_confirmation",
            exposure_impact=1.0,
            safe_fraction_impact=0.0,
            risk_state="neutral",
            metadata={"reason": "insufficient_data"}
        )
    
    # Average trend alignment
    avg_trend = sum(trend_scores) / len(trend_scores)
    metadata["avg_trend_score"] = avg_trend
    metadata["trend_alignment"] = len([s for s in trend_scores if s == trend_scores[0]]) == len(trend_scores)
    
    # Exposure impact based on trend alignment
    if avg_trend >= 0.7:  # Strong uptrend across timeframes
        exposure_impact = 1.0 + (0.05 * config.weight)  # Slight boost
        risk_state = "risk-on"
        safe_impact = 0.0
    elif avg_trend <= -0.7:  # Strong downtrend across timeframes
        exposure_impact = 1.0 - (0.10 * config.weight)  # Reduce exposure
        risk_state = "risk-off"
        safe_impact = 0.05 * config.weight  # Add to safe
    else:  # Mixed or neutral
        exposure_impact = 1.0
        risk_state = "neutral"
        safe_impact = 0.0
    
    exposure_impact = float(np.clip(exposure_impact, config.min_impact, config.max_impact))
    safe_impact = float(np.clip(safe_impact, 0.0, config.max_impact))
    
    return StrategyContribution(
        name="trend_confirmation",
        exposure_impact=exposure_impact,
        safe_fraction_impact=safe_impact,
        risk_state=risk_state,
        metadata=metadata
    )


def _compute_relative_strength_strategy(
    wave_weights: pd.Series,
    bm_weights: pd.Series,
    ret_df: pd.DataFrame,
    dt: pd.Timestamp,
    config: StrategyConfig
) -> StrategyContribution:
    """
    Relative strength: compare wave holdings performance vs benchmark over 20D/60D.
    Adjusts exposure based on relative outperformance.
    """
    if not config.enabled:
        return StrategyContribution(
            name="relative_strength",
            exposure_impact=1.0,
            safe_fraction_impact=0.0,
            risk_state="neutral",
            enabled=False
        )
    
    metadata = {}
    
    # Calculate 20D cumulative returns for wave and benchmark
    if len(ret_df) < 20:
        return StrategyContribution(
            name="relative_strength",
            exposure_impact=1.0,
            safe_fraction_impact=0.0,
            risk_state="neutral",
            metadata={"reason": "insufficient_history"}
        )
    
    # Get last 20 days of returns up to dt
    dt_idx = ret_df.index.get_loc(dt) if dt in ret_df.index else -1
    if dt_idx < 20:
        return StrategyContribution(
            name="relative_strength",
            exposure_impact=1.0,
            safe_fraction_impact=0.0,
            risk_state="neutral",
            metadata={"reason": "insufficient_lookback"}
        )
    
    ret_20d = ret_df.iloc[dt_idx-19:dt_idx+1]
    
    # Wave cumulative return
    wave_weights_aligned = wave_weights.reindex(ret_20d.columns).fillna(0.0)
    wave_ret_20d = ((ret_20d * wave_weights_aligned).sum(axis=1) + 1.0).prod() - 1.0
    
    # Benchmark cumulative return
    bm_weights_aligned = bm_weights.reindex(ret_20d.columns).fillna(0.0)
    bm_ret_20d = ((ret_20d * bm_weights_aligned).sum(axis=1) + 1.0).prod() - 1.0
    
    relative_strength = wave_ret_20d - bm_ret_20d
    metadata["wave_ret_20d"] = float(wave_ret_20d)
    metadata["bm_ret_20d"] = float(bm_ret_20d)
    metadata["relative_strength_20d"] = float(relative_strength)
    
    # Exposure impact based on relative strength
    if relative_strength > 0.03:  # Outperforming by 3%+
        exposure_impact = 1.0 + (0.03 * config.weight)  # Slight increase
        risk_state = "risk-on"
        safe_impact = 0.0
    elif relative_strength < -0.03:  # Underperforming by 3%+
        exposure_impact = 1.0 - (0.03 * config.weight)  # Slight decrease
        risk_state = "risk-off"
        safe_impact = 0.02 * config.weight
    else:  # Neutral performance
        exposure_impact = 1.0
        risk_state = "neutral"
        safe_impact = 0.0
    
    exposure_impact = float(np.clip(exposure_impact, config.min_impact, config.max_impact))
    safe_impact = float(np.clip(safe_impact, 0.0, config.max_impact))
    
    return StrategyContribution(
        name="relative_strength",
        exposure_impact=exposure_impact,
        safe_fraction_impact=safe_impact,
        risk_state=risk_state,
        metadata=metadata
    )


def _aggregate_strategy_contributions(
    contributions: List[StrategyContribution],
    mode_base_exposure: float,
    exp_min: float,
    exp_max: float
) -> Tuple[float, float, str, Dict[str, Any]]:
    """
    Aggregate all strategy contributions into final exposure and safe fraction.
    
    Returns:
        - final_exposure_multiplier: combined exposure multiplier from all strategies
        - final_safe_fraction: combined safe fraction contribution
        - aggregated_risk_state: overall risk-on/off state
        - attribution: per-strategy contribution breakdown
    """
    # Separate multiplicative (exposure) and additive (safe) contributions
    exposure_multipliers = []
    safe_fractions = []
    risk_states = []
    attribution = {}
    
    for contrib in contributions:
        if not contrib.enabled:
            continue
        
        exposure_multipliers.append(contrib.exposure_impact)
        safe_fractions.append(contrib.safe_fraction_impact)
        risk_states.append(contrib.risk_state)
        
        attribution[contrib.name] = {
            "exposure_impact": contrib.exposure_impact,
            "safe_impact": contrib.safe_fraction_impact,
            "risk_state": contrib.risk_state,
            "metadata": contrib.metadata
        }
    
    # Combine exposure multipliers (multiplicative)
    combined_exposure = np.prod(exposure_multipliers) if exposure_multipliers else 1.0
    
    # Combine safe fractions (additive, capped)
    combined_safe = sum(safe_fractions) if safe_fractions else 0.0
    combined_safe = float(np.clip(combined_safe, 0.0, 0.95))
    
    # Determine overall risk state (majority vote)
    risk_on_count = sum(1 for s in risk_states if s == "risk-on")
    risk_off_count = sum(1 for s in risk_states if s == "risk-off")
    
    if risk_off_count > risk_on_count:
        aggregated_risk_state = "risk-off"
    elif risk_on_count > risk_off_count:
        aggregated_risk_state = "risk-on"
    else:
        aggregated_risk_state = "neutral"
    
    attribution["_summary"] = {
        "combined_exposure_multiplier": float(combined_exposure),
        "combined_safe_fraction": float(combined_safe),
        "aggregated_risk_state": aggregated_risk_state,
        "active_strategies": len([c for c in contributions if c.enabled])
    }
    
    return float(combined_exposure), float(combined_safe), aggregated_risk_state, attribution


def _compute_core(
    wave_name: str,
    mode: str = "Standard",
    days: int = 365,
    overrides: Optional[Dict[str, Any]] = None,
    shadow: bool = False,
    price_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Core engine calculator used by:
      - compute_history_nav()  [baseline]
      - simulate_history_nav() [shadow overrides]

    Args:
        wave_name: Name of the Wave
        mode: Operating mode (Standard, Alpha-Minus-Beta, Private Logic)
        days: History window
        overrides: Optional dict with parameter overrides
        shadow: Whether to include diagnostics in result.attrs
        price_df: Optional pre-fetched price DataFrame (date index, ticker columns).
                  If provided, uses this instead of calling _download_history().
    
    Overrides keys (optional):
      - tilt_strength: float
      - vol_target: float
      - extra_safe_boost: float
      - base_exposure_mult: float
      - exp_min: float
      - exp_max: float
      - freeze_benchmark: bool   (use static benchmark only)
    """
    if wave_name not in WAVE_WEIGHTS:
        print(f"Error: Unknown Wave: {wave_name}")
        # Return empty DataFrame instead of raising
        return pd.DataFrame(columns=["wave_nav", "bm_nav", "wave_ret", "bm_ret"], dtype=float)
    if mode not in MODE_BASE_EXPOSURE:
        print(f"Error: Unknown mode: {mode}")
        # Return empty DataFrame instead of raising
        return pd.DataFrame(columns=["wave_nav", "bm_nav", "wave_ret", "bm_ret"], dtype=float)

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
    
    # Income wave specific tickers
    income_tickers = [TNX_TICKER, HYG_TICKER, LQD_TICKER]

    all_tickers = set(tickers_wave + tickers_bm)
    all_tickers.add(base_index_ticker)
    all_tickers.add(VIX_TICKER)
    all_tickers.add(BTC_TICKER)
    all_tickers.update(safe_candidates)
    all_tickers.update(income_tickers)  # Add income-specific tickers

    all_tickers = sorted(all_tickers)
    if not all_tickers:
        return pd.DataFrame(columns=["wave_nav", "bm_nav", "wave_ret", "bm_ret"], dtype=float)

    # Use provided price_df if available, otherwise download
    failed_tickers = {}
    # Get wave_id for diagnostics tracking
    wave_id = get_wave_id_from_display_name(wave_name)
    
    if price_df is None:
        price_df, failed_tickers = _download_history(all_tickers, days=days, wave_id=wave_id, wave_name=wave_name)
    else:
        # Filter to needed tickers and ensure we have the data
        available_tickers = [t for t in all_tickers if t in price_df.columns]
        if available_tickers:
            price_df = price_df[available_tickers].copy()
        else:
            # Fallback to download if no tickers are available in provided price_df
            price_df, failed_tickers = _download_history(all_tickers, days=days, wave_id=wave_id, wave_name=wave_name)
    
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

    # Strategy configurations (can be overridden)
    strategy_configs = ov.get("strategy_configs", DEFAULT_STRATEGY_CONFIGS.copy())
    
    # Diagnostics series (optional)
    diag_rows = []
    attribution_rows = []  # New: per-day strategy attribution

    for dt in ret_df.index:
        rets = ret_df.loc[dt]

        # ===== Individual Strategy Computations =====
        
        # Detect wave type
        is_crypto = _is_crypto_wave(wave_name)
        is_crypto_growth = _is_crypto_growth_wave(wave_name)
        is_crypto_income = _is_crypto_income_wave(wave_name)
        is_income = _is_income_wave(wave_name)
        
        # 1. Regime detection strategy (EQUITY GROWTH ONLY - disabled for crypto and income)
        if not is_crypto and not is_income:
            regime = _regime_from_return(idx_ret_60d.get(dt, np.nan))
            regime_exposure = REGIME_EXPOSURE[regime]
            regime_gate = REGIME_GATING[mode][regime]
            regime_risk_state = "risk-off" if regime in ("panic", "downtrend") else ("risk-on" if regime == "uptrend" else "neutral")
            
            regime_contrib = StrategyContribution(
                name="regime_detection",
                exposure_impact=regime_exposure,
                safe_fraction_impact=regime_gate,
                risk_state=regime_risk_state,
                enabled=strategy_configs.get("regime_detection", DEFAULT_STRATEGY_CONFIGS["regime_detection"]).enabled,
                metadata={"regime": regime}
            )
        else:
            # Crypto and income waves: disable equity regime detection
            regime_contrib = StrategyContribution(
                name="regime_detection",
                exposure_impact=1.0,
                safe_fraction_impact=0.0,
                risk_state="neutral",
                enabled=False,
                metadata={"note": "disabled_for_crypto_and_income"}
            )

        # 2. VIX overlay strategy (EQUITY GROWTH ONLY - disabled for crypto and income)
        if not is_crypto and not is_income:
            vix_level = float(vix_level_series.get(dt, np.nan))
            vix_exposure = _vix_exposure_factor(vix_level, mode)
            vix_gate = _vix_safe_fraction(vix_level, mode)
            vix_risk_state = "risk-off" if vix_level >= 25 else ("risk-on" if vix_level < 18 else "neutral")
            
            vix_contrib = StrategyContribution(
                name="vix_overlay",
                exposure_impact=vix_exposure,
                safe_fraction_impact=vix_gate,
                risk_state=vix_risk_state,
                enabled=strategy_configs.get("vix_overlay", DEFAULT_STRATEGY_CONFIGS["vix_overlay"]).enabled,
                metadata={"vix_level": vix_level}
            )
        else:
            # Crypto and income waves: disable VIX overlay
            vix_contrib = StrategyContribution(
                name="vix_overlay",
                exposure_impact=1.0,
                safe_fraction_impact=0.0,
                risk_state="neutral",
                enabled=False,
                metadata={"note": "disabled_for_crypto_and_income"}
            )

        # 3. Momentum strategy (weight tilting)
        mom_row = mom_60.loc[dt] if dt in mom_60.index else None
        if mom_row is not None:
            mom_series = mom_row.reindex(price_df.columns).fillna(0.0)
            mom_clipped = mom_series.clip(lower=-0.30, upper=0.30)
            tilt_factor = 1.0 + tilt_strength * mom_clipped
            effective_weights = wave_weights_aligned * tilt_factor
            momentum_enabled = strategy_configs.get("momentum", DEFAULT_STRATEGY_CONFIGS["momentum"]).enabled
        else:
            effective_weights = wave_weights_aligned.copy()
            momentum_enabled = False

        effective_weights = effective_weights.clip(lower=0.0)

        risk_weight_total = effective_weights.sum()
        if risk_weight_total > 0:
            risk_weights = effective_weights / risk_weight_total
        else:
            risk_weights = wave_weights_aligned.copy()

        # Note: Momentum affects weights, not exposure multiplier directly
        # It's already applied to effective_weights above
        momentum_contrib = StrategyContribution(
            name="momentum",
            exposure_impact=1.0,  # Already in weights
            safe_fraction_impact=0.0,
            risk_state="neutral",
            enabled=momentum_enabled,
            metadata={"tilt_strength": tilt_strength}
        )

        portfolio_risk_ret = float((rets * risk_weights).sum())
        safe_ret = float(safe_ret_series.loc[dt])

        # 4. Volatility targeting strategy
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
        
        vol_contrib = StrategyContribution(
            name="volatility_targeting",
            exposure_impact=vol_adjust,
            safe_fraction_impact=0.0,
            risk_state="neutral",
            enabled=strategy_configs.get("volatility_targeting", DEFAULT_STRATEGY_CONFIGS["volatility_targeting"]).enabled,
            metadata={"recent_vol": recent_vol, "vol_target": vol_target}
        )

        # 5. Trend confirmation strategy (new)
        trend_config = strategy_configs.get("trend_confirmation", DEFAULT_STRATEGY_CONFIGS["trend_confirmation"])
        trend_contrib = _compute_trend_confirmation_strategy(
            price_df, base_index_ticker, dt, trend_config
        )

        # 6. Relative strength strategy (new)
        rs_config = strategy_configs.get("relative_strength", DEFAULT_STRATEGY_CONFIGS["relative_strength"])
        rs_contrib = _compute_relative_strength_strategy(
            wave_weights_aligned, bm_weights_aligned, ret_df, dt, rs_config
        )

        # 7. SmartSafe strategy (combines regime + vix gating + extra boost)
        smartsafe_gate = extra_safe_boost  # extra_safe_boost is the user-configurable part
        smartsafe_contrib = StrategyContribution(
            name="smartsafe",
            exposure_impact=1.0,
            safe_fraction_impact=smartsafe_gate,
            risk_state="neutral",
            enabled=strategy_configs.get("smartsafe", DEFAULT_STRATEGY_CONFIGS["smartsafe"]).enabled,
            metadata={"extra_safe_boost": extra_safe_boost}
        )

        # 8. Mode constraint strategy
        mode_contrib = StrategyContribution(
            name="mode_constraint",
            exposure_impact=mode_base_exposure,
            safe_fraction_impact=0.0,
            risk_state="neutral",
            enabled=strategy_configs.get("mode_constraint", DEFAULT_STRATEGY_CONFIGS["mode_constraint"]).enabled,
            metadata={"mode": mode, "base_exposure": mode_base_exposure}
        )

        # ===== CRYPTO-SPECIFIC OVERLAYS (only active for crypto waves) =====
        
        # 9. Crypto Trend/Momentum Regime Overlay (for crypto growth waves only)
        if is_crypto_growth:
            trend_60d_val = float(idx_ret_60d.get(dt, np.nan))
            crypto_trend_exp, crypto_trend_safe, crypto_trend_regime = _crypto_trend_momentum_overlay(trend_60d_val)
            crypto_trend_risk_state = "risk-off" if "downtrend" in crypto_trend_regime else ("risk-on" if "uptrend" in crypto_trend_regime else "neutral")
            
            crypto_trend_contrib = StrategyContribution(
                name="crypto_trend_momentum",
                exposure_impact=crypto_trend_exp,
                safe_fraction_impact=crypto_trend_safe,
                risk_state=crypto_trend_risk_state,
                enabled=strategy_configs.get("crypto_trend_momentum", DEFAULT_STRATEGY_CONFIGS["crypto_trend_momentum"]).enabled,
                metadata={"crypto_regime": crypto_trend_regime, "trend_60d": trend_60d_val}
            )
        else:
            crypto_trend_contrib = StrategyContribution(
                name="crypto_trend_momentum",
                exposure_impact=1.0,
                safe_fraction_impact=0.0,
                risk_state="neutral",
                enabled=False,
                metadata={"note": "not_crypto_growth_wave"}
            )
        
        # 10. Crypto Volatility State Overlay (for crypto growth waves only)
        if is_crypto_growth:
            # Calculate realized volatility for crypto
            if len(wave_ret_list) >= 30:
                recent_crypto = np.array(wave_ret_list[-30:])
                crypto_realized_vol = recent_crypto.std() * np.sqrt(TRADING_DAYS_PER_YEAR)
            else:
                crypto_realized_vol = 0.60  # Default crypto vol assumption
            
            crypto_vol_exp, crypto_vol_state = _crypto_volatility_overlay(crypto_realized_vol)
            
            crypto_vol_contrib = StrategyContribution(
                name="crypto_volatility",
                exposure_impact=crypto_vol_exp,
                safe_fraction_impact=0.0,
                risk_state="neutral",
                enabled=strategy_configs.get("crypto_volatility", DEFAULT_STRATEGY_CONFIGS["crypto_volatility"]).enabled,
                metadata={"vol_state": crypto_vol_state, "realized_vol": crypto_realized_vol}
            )
        else:
            crypto_vol_contrib = StrategyContribution(
                name="crypto_volatility",
                exposure_impact=1.0,
                safe_fraction_impact=0.0,
                risk_state="neutral",
                enabled=False,
                metadata={"note": "not_crypto_growth_wave"}
            )
        
        # 11. Crypto Liquidity/Market Structure Overlay (for ALL crypto waves)
        if is_crypto:
            # Calculate volume ratio (simplified - using portfolio return magnitude as proxy)
            # In production, would use actual volume data
            if len(wave_ret_list) >= 20:
                recent_rets = np.array(wave_ret_list[-20:])
                avg_abs_ret = np.abs(recent_rets).mean()
                current_abs_ret = abs(portfolio_risk_ret)
                volume_ratio = current_abs_ret / avg_abs_ret if avg_abs_ret > 0 else 1.0
            else:
                volume_ratio = 1.0
            
            crypto_liq_exp, crypto_liq_state = _crypto_liquidity_overlay(volume_ratio)
            
            crypto_liquidity_contrib = StrategyContribution(
                name="crypto_liquidity",
                exposure_impact=crypto_liq_exp,
                safe_fraction_impact=0.0,
                risk_state="neutral",
                enabled=strategy_configs.get("crypto_liquidity", DEFAULT_STRATEGY_CONFIGS["crypto_liquidity"]).enabled,
                metadata={"liquidity_state": crypto_liq_state, "volume_ratio": volume_ratio}
            )
        else:
            crypto_liquidity_contrib = StrategyContribution(
                name="crypto_liquidity",
                exposure_impact=1.0,
                safe_fraction_impact=0.0,
                risk_state="neutral",
                enabled=False,
                metadata={"note": "not_crypto_wave"}
            )
        
        # 12. Crypto Income Stability Overlay (for Crypto Income Wave only)
        if is_crypto_income:
            # Get metrics for income stability assessment
            trend_60d_val = float(idx_ret_60d.get(dt, np.nan))
            if len(wave_ret_list) >= 30:
                recent_crypto = np.array(wave_ret_list[-30:])
                crypto_realized_vol = recent_crypto.std() * np.sqrt(TRADING_DAYS_PER_YEAR)
            else:
                crypto_realized_vol = 0.60
            
            crypto_income_exp, crypto_income_safe = _crypto_income_stability_overlay(crypto_realized_vol, trend_60d_val)
            
            crypto_income_contrib = StrategyContribution(
                name="crypto_income_stability",
                exposure_impact=crypto_income_exp,
                safe_fraction_impact=crypto_income_safe,
                risk_state="neutral",
                enabled=strategy_configs.get("crypto_income_stability", DEFAULT_STRATEGY_CONFIGS["crypto_income_stability"]).enabled,
                metadata={"realized_vol": crypto_realized_vol, "trend_60d": trend_60d_val}
            )
        else:
            crypto_income_contrib = StrategyContribution(
                name="crypto_income_stability",
                exposure_impact=1.0,
                safe_fraction_impact=0.0,
                risk_state="neutral",
                enabled=False,
                metadata={"note": "not_crypto_income_wave"}
            )
        
        # 12a. Crypto Income Drawdown Guard Overlay (for Crypto Income Wave only)
        if is_crypto_income:
            crypto_income_dd_safe, crypto_income_dd_state = _crypto_income_drawdown_guard(wave_ret_list)
            
            crypto_income_drawdown_contrib = StrategyContribution(
                name="crypto_income_drawdown_guard",
                exposure_impact=1.0,
                safe_fraction_impact=crypto_income_dd_safe,
                risk_state="risk-off" if crypto_income_dd_state in ("moderate", "severe") else "neutral",
                enabled=strategy_configs.get("crypto_income_drawdown_guard", DEFAULT_STRATEGY_CONFIGS["crypto_income_drawdown_guard"]).enabled,
                metadata={"stress_state": crypto_income_dd_state, "safe_boost": crypto_income_dd_safe}
            )
        else:
            crypto_income_drawdown_contrib = StrategyContribution(
                name="crypto_income_drawdown_guard",
                exposure_impact=1.0,
                safe_fraction_impact=0.0,
                risk_state="neutral",
                enabled=False,
                metadata={"note": "not_crypto_income_wave"}
            )
        
        # 12b. Crypto Income Liquidity Gate Overlay (for Crypto Income Wave only)
        if is_crypto_income:
            # Calculate volume ratio (simplified - using portfolio return magnitude as proxy)
            if len(wave_ret_list) >= 20:
                recent_rets = np.array(wave_ret_list[-20:])
                avg_abs_ret = np.abs(recent_rets).mean()
                current_abs_ret = abs(portfolio_risk_ret)
                volume_ratio = current_abs_ret / avg_abs_ret if avg_abs_ret > 0 else 1.0
            else:
                volume_ratio = 1.0
            
            crypto_income_liq_exp, crypto_income_liq_state = _crypto_income_liquidity_gate(volume_ratio)
            
            crypto_income_liquidity_contrib = StrategyContribution(
                name="crypto_income_liquidity_gate",
                exposure_impact=crypto_income_liq_exp,
                safe_fraction_impact=0.0,
                risk_state="neutral",
                enabled=strategy_configs.get("crypto_income_liquidity_gate", DEFAULT_STRATEGY_CONFIGS["crypto_income_liquidity_gate"]).enabled,
                metadata={"liquidity_state": crypto_income_liq_state, "volume_ratio": volume_ratio}
            )
        else:
            crypto_income_liquidity_contrib = StrategyContribution(
                name="crypto_income_liquidity_gate",
                exposure_impact=1.0,
                safe_fraction_impact=0.0,
                risk_state="neutral",
                enabled=False,
                metadata={"note": "not_crypto_income_wave"}
            )

        # ===== INCOME-SPECIFIC OVERLAYS (only active for non-crypto income waves) =====
        
        # 13. Income Rates/Duration Regime Overlay (for income waves only)
        if is_income:
            # Calculate TNX (10-year Treasury) trend using helper
            if TNX_TICKER in price_df.columns:
                tnx_trend_val = _calculate_price_return(price_df[TNX_TICKER], dt, 60)
            else:
                tnx_trend_val = np.nan
            
            income_rates_exp, income_rates_regime = _rates_duration_overlay(tnx_trend_val)
            income_rates_risk_state = "risk-off" if "rising" in income_rates_regime else ("risk-on" if "falling" in income_rates_regime else "neutral")
            
            income_rates_contrib = StrategyContribution(
                name="income_rates_regime",
                exposure_impact=income_rates_exp,
                safe_fraction_impact=0.0,
                risk_state=income_rates_risk_state,
                enabled=strategy_configs.get("income_rates_regime", DEFAULT_STRATEGY_CONFIGS["income_rates_regime"]).enabled,
                metadata={"rates_regime": income_rates_regime, "tnx_trend_60d": tnx_trend_val}
            )
        else:
            income_rates_contrib = StrategyContribution(
                name="income_rates_regime",
                exposure_impact=1.0,
                safe_fraction_impact=0.0,
                risk_state="neutral",
                enabled=False,
                metadata={"note": "not_income_wave"}
            )
        
        # 14. Income Credit/Risk Regime Overlay (for income waves only)
        if is_income:
            # Calculate HYG vs LQD spread (relative performance) using helper
            if HYG_TICKER in price_df.columns and LQD_TICKER in price_df.columns:
                hyg_val = _calculate_price_return(price_df[HYG_TICKER], dt, 20)
                lqd_val = _calculate_price_return(price_df[LQD_TICKER], dt, 20)
                
                if not np.isnan(hyg_val) and not np.isnan(lqd_val):
                    hyg_lqd_spread = hyg_val - lqd_val
                else:
                    hyg_lqd_spread = np.nan
            else:
                hyg_lqd_spread = np.nan
            
            income_credit_exp, income_credit_safe, income_credit_regime = _credit_risk_overlay(hyg_lqd_spread)
            income_credit_risk_state = "risk-on" if income_credit_regime == "risk_on" else ("risk-off" if income_credit_regime == "risk_off" else "neutral")
            
            income_credit_contrib = StrategyContribution(
                name="income_credit_regime",
                exposure_impact=income_credit_exp,
                safe_fraction_impact=income_credit_safe,
                risk_state=income_credit_risk_state,
                enabled=strategy_configs.get("income_credit_regime", DEFAULT_STRATEGY_CONFIGS["income_credit_regime"]).enabled,
                metadata={"credit_regime": income_credit_regime, "hyg_lqd_spread": hyg_lqd_spread}
            )
        else:
            income_credit_contrib = StrategyContribution(
                name="income_credit_regime",
                exposure_impact=1.0,
                safe_fraction_impact=0.0,
                risk_state="neutral",
                enabled=False,
                metadata={"note": "not_income_wave"}
            )
        
        # 15. Income Carry + Drawdown Guard Overlay (for income waves only)
        if is_income:
            # Calculate current drawdown and recent volatility
            if len(wave_ret_list) >= 1:
                wave_nav_series = pd.Series(wave_ret_list, index=dates)
                wave_nav_cumulative = (1.0 + wave_nav_series).cumprod()
                peak_nav = wave_nav_cumulative.max()
                current_nav = wave_nav_cumulative.iloc[-1]
                
                # Calculate recent volatility
                if len(wave_ret_list) >= 20:
                    recent_rets = np.array(wave_ret_list[-20:])
                    recent_vol = recent_rets.std() * np.sqrt(TRADING_DAYS_PER_YEAR)
                else:
                    recent_vol = 0.05  # Default low vol for income
            else:
                peak_nav = 1.0
                current_nav = 1.0
                recent_vol = 0.05
            
            income_dd_safe, income_dd_state = _drawdown_guard_overlay(current_nav, peak_nav, recent_vol)
            income_dd_risk_state = "risk-off" if income_dd_state in ("moderate", "severe") else "neutral"
            
            income_drawdown_contrib = StrategyContribution(
                name="income_drawdown_guard",
                exposure_impact=1.0,
                safe_fraction_impact=income_dd_safe,
                risk_state=income_dd_risk_state,
                enabled=strategy_configs.get("income_drawdown_guard", DEFAULT_STRATEGY_CONFIGS["income_drawdown_guard"]).enabled,
                metadata={"stress_state": income_dd_state, "recent_vol": recent_vol, "drawdown": (current_nav / peak_nav - 1.0) if peak_nav > 0 else 0.0}
            )
        else:
            income_drawdown_contrib = StrategyContribution(
                name="income_drawdown_guard",
                exposure_impact=1.0,
                safe_fraction_impact=0.0,
                risk_state="neutral",
                enabled=False,
                metadata={"note": "not_income_wave"}
            )
        
        # 16. Income Turnover Discipline Overlay (for income waves only)
        # Note: Turnover discipline is primarily enforced through weight stability
        # This overlay serves as a placeholder for future turnover monitoring
        if is_income:
            income_turnover_contrib = StrategyContribution(
                name="income_turnover_discipline",
                exposure_impact=1.0,
                safe_fraction_impact=0.0,
                risk_state="neutral",
                enabled=strategy_configs.get("income_turnover_discipline", DEFAULT_STRATEGY_CONFIGS["income_turnover_discipline"]).enabled,
                metadata={"note": "low_turnover_enforced"}
            )
        else:
            income_turnover_contrib = StrategyContribution(
                name="income_turnover_discipline",
                exposure_impact=1.0,
                safe_fraction_impact=0.0,
                risk_state="neutral",
                enabled=False,
                metadata={"note": "not_income_wave"}
            )

        # ===== Aggregate All Strategy Contributions =====
        all_contributions = [
            mode_contrib,
            regime_contrib,
            vix_contrib,
            momentum_contrib,
            vol_contrib,
            trend_contrib,
            rs_contrib,
            smartsafe_contrib,
            crypto_trend_contrib,
            crypto_vol_contrib,
            crypto_liquidity_contrib,
            crypto_income_contrib,
            crypto_income_drawdown_contrib,
            crypto_income_liquidity_contrib,
            income_rates_contrib,
            income_credit_contrib,
            income_drawdown_contrib,
            income_turnover_contrib,
        ]

        # Aggregate strategies into final exposure and safe fraction
        combined_exposure_mult, combined_safe_add, agg_risk_state, strategy_attribution = _aggregate_strategy_contributions(
            all_contributions, mode_base_exposure, exp_min, exp_max
        )

        # Apply aggregated exposure (clipped to min/max)
        raw_exposure = combined_exposure_mult
        exposure = float(np.clip(raw_exposure, exp_min, exp_max))

        # Apply aggregated safe fraction (clipped to 0-95%)
        safe_fraction = combined_safe_add
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
            # Legacy diagnostics (backward compatible)
            # For crypto and income waves, regime/vix may not be defined
            diag_regime = regime_contrib.metadata.get("regime", "n/a") if not is_crypto and not is_income else "n/a"
            diag_vix = vix_contrib.metadata.get("vix_level", np.nan) if not is_crypto and not is_income else np.nan
            diag_vix_exposure = vix_contrib.exposure_impact if not is_crypto and not is_income else 1.0
            diag_vix_gate = vix_contrib.safe_fraction_impact if not is_crypto and not is_income else 0.0
            diag_regime_gate = regime_contrib.safe_fraction_impact if not is_crypto and not is_income else 0.0
            
            diag_rows.append(
                {
                    "Date": dt,
                    "regime": diag_regime,
                    "vix": diag_vix,
                    "safe_fraction": safe_fraction,
                    "exposure": exposure,
                    "vol_adjust": vol_adjust,
                    "vix_exposure": diag_vix_exposure,
                    "vix_gate": diag_vix_gate,
                    "regime_gate": diag_regime_gate,
                    "aggregated_risk_state": agg_risk_state,
                    # Add crypto-specific diagnostics
                    "is_crypto": is_crypto,
                    "crypto_trend_regime": crypto_trend_contrib.metadata.get("crypto_regime", "n/a") if is_crypto_growth else "n/a",
                    "crypto_vol_state": crypto_vol_contrib.metadata.get("vol_state", "n/a") if is_crypto_growth else "n/a",
                    "crypto_liq_state": crypto_liquidity_contrib.metadata.get("liquidity_state", "n/a") if is_crypto else "n/a",
                    # Add income-specific diagnostics
                    "is_income": is_income,
                    "income_rates_regime": income_rates_contrib.metadata.get("rates_regime", "n/a") if is_income else "n/a",
                    "income_credit_regime": income_credit_contrib.metadata.get("credit_regime", "n/a") if is_income else "n/a",
                    "income_stress_state": income_drawdown_contrib.metadata.get("stress_state", "n/a") if is_income else "n/a",
                    # Add crypto income-specific diagnostics
                    "is_crypto_income": is_crypto_income,
                    "crypto_income_stress_state": crypto_income_drawdown_contrib.metadata.get("stress_state", "n/a") if is_crypto_income else "n/a",
                    "crypto_income_liq_state": crypto_income_liquidity_contrib.metadata.get("liquidity_state", "n/a") if is_crypto_income else "n/a",
                    # Strategy family tagging (using canonical registry)
                    "strategy_family": get_strategy_family(wave_name),
                }
            )
            
            # New: Strategy-level attribution
            attribution_rows.append({
                "Date": dt,
                "strategy_attribution": strategy_attribution
            })

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
        
        # Add strategy attribution to attrs
        if attribution_rows:
            out.attrs["strategy_attribution"] = attribution_rows

    return out


# ------------------------------------------------------------
# Baseline API (unchanged behavior)
# ------------------------------------------------------------

def compute_history_nav(wave_name: str, mode: str = "Standard", days: int = 365, include_diagnostics: bool = False, price_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Baseline (official) engine output with comprehensive error handling.
    
    Args:
        wave_name: name of the Wave
        mode: operating mode (Standard, Alpha-Minus-Beta, Private Logic)
        days: history window
        include_diagnostics: if True, include per-day VIX/regime/exposure diagnostics in result.attrs["diagnostics"]
        price_df: Optional pre-fetched price DataFrame (date index, ticker columns). If provided, skips yfinance download.
        
    Returns:
        DataFrame with wave_nav, bm_nav, wave_ret, bm_ret columns.
        If include_diagnostics=True, also includes diagnostics DataFrame in attrs["diagnostics"].
        Returns empty DataFrame on error to prevent crashes.
    """
    try:
        return _compute_core(wave_name=wave_name, mode=mode, days=days, overrides=None, shadow=include_diagnostics, price_df=price_df)
    except Exception as e:
        # Comprehensive error handling - never crash the app
        print(f"Error in compute_history_nav for {wave_name}: {str(e)}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame(columns=["wave_nav", "bm_nav", "wave_ret", "bm_ret"], dtype=float)


def simulate_history_nav(wave_name: str, mode: str = "Standard", days: int = 365, overrides: Optional[Dict[str, Any]] = None, price_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Shadow "What-If" simulation. Never overwrites baseline.
    Returns the same columns, plus diagnostics in out.attrs["diagnostics"] if available.
    
    Args:
        price_df: Optional pre-fetched price DataFrame (date index, ticker columns). If provided, skips yfinance download.
    """
    return _compute_core(wave_name=wave_name, mode=mode, days=days, overrides=overrides or {}, shadow=True, price_df=price_df)


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


def get_vix_regime_diagnostics(wave_name: str, mode: str = "Standard", days: int = 365) -> pd.DataFrame:
    """
    Get detailed VIX/regime overlay diagnostics showing exposure scaling per day.
    
    Returns DataFrame with columns:
        - Date (index): trading date
        - regime: regime state (panic/downtrend/neutral/uptrend)
        - vix: VIX level (or crypto vol proxy)
        - safe_fraction: portion allocated to safe assets (0..1)
        - exposure: final exposure used in return calculation (0..1+)
        - vol_adjust: volatility targeting adjustment factor
        - vix_exposure: VIX-driven exposure adjustment factor
        - vix_gate: VIX-driven safe allocation boost
        - regime_gate: regime-driven safe allocation boost
        
    This function proves that VIX/regime overlay is actively affecting returns
    by showing the exact exposure scaling applied each day.
    
    Example usage:
        >>> diag = get_vix_regime_diagnostics("US MegaCap Core Wave", "Standard", 365)
        >>> # Filter to high VIX periods
        >>> stress_periods = diag[diag["vix"] >= 25]
        >>> print(f"Avg exposure during stress: {stress_periods['exposure'].mean():.2%}")
        >>> # Compare to low VIX periods
        >>> calm_periods = diag[diag["vix"] < 20]
        >>> print(f"Avg exposure during calm: {calm_periods['exposure'].mean():.2%}")
    
    Args:
        wave_name: name of the Wave
        mode: operating mode (Standard, Alpha-Minus-Beta, Private Logic)
        days: history window
        
    Returns:
        DataFrame indexed by Date with diagnostic columns
    """
    result = _compute_core(wave_name=wave_name, mode=mode, days=days, overrides=None, shadow=True)
    
    if result.empty:
        return pd.DataFrame(columns=[
            "regime", "vix", "safe_fraction", "exposure", 
            "vol_adjust", "vix_exposure", "vix_gate", "regime_gate"
        ])
    
    # Extract diagnostics from attrs
    diag_df = result.attrs.get("diagnostics")
    if diag_df is None or diag_df.empty:
        return pd.DataFrame(columns=[
            "regime", "vix", "safe_fraction", "exposure",
            "vol_adjust", "vix_exposure", "vix_gate", "regime_gate"
        ])
    
    return diag_df


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


def get_strategy_attribution(wave_name: str, mode: str = "Standard", days: int = 365) -> Dict[str, Any]:
    """
    Get per-strategy attribution showing how each strategy contributed to exposure and returns.
    
    This function provides diagnostic-level visibility into strategy impacts without changing
    the UI. It shows:
    - Per-strategy exposure impact (multiplier effect)
    - Per-strategy safe fraction impact (additive contribution)
    - Per-strategy risk state (risk-on/off/neutral)
    - Which strategies were dormant vs. active
    - Aggregated summary statistics
    
    Args:
        wave_name: name of the Wave
        mode: operating mode (Standard, Alpha-Minus-Beta, Private Logic)
        days: history window
        
    Returns:
        Dictionary with:
        - "summary": Overall statistics across all days
        - "daily_attribution": List of per-day strategy contributions
        - "strategy_stats": Per-strategy statistics over the period
        
    Example:
        >>> attr = get_strategy_attribution("US MegaCap Core Wave", "Standard", 365)
        >>> # Check which strategies were most impactful
        >>> print(attr["summary"]["most_impactful_strategies"])
        >>> # See which days had risk-off triggers
        >>> risk_off_days = [d for d in attr["daily_attribution"] 
        ...                  if d["aggregated_risk_state"] == "risk-off"]
    """
    # Run shadow simulation to get attribution data
    result = simulate_history_nav(wave_name=wave_name, mode=mode, days=days, overrides={})
    
    if result.empty:
        return {
            "ok": False,
            "message": "No data available for attribution",
            "summary": {},
            "daily_attribution": [],
            "strategy_stats": {}
        }
    
    # Extract attribution from attrs
    attribution_rows = result.attrs.get("strategy_attribution", [])
    
    if not attribution_rows:
        return {
            "ok": False,
            "message": "Strategy attribution not available (requires shadow simulation)",
            "summary": {},
            "daily_attribution": [],
            "strategy_stats": {}
        }
    
    # Process attribution data
    strategy_impacts = {}
    risk_state_counts = {"risk-on": 0, "risk-off": 0, "neutral": 0}
    daily_summaries = []
    
    for row in attribution_rows:
        dt = row["Date"]
        attr = row["strategy_attribution"]
        
        # Extract summary
        summary = attr.get("_summary", {})
        agg_state = summary.get("aggregated_risk_state", "neutral")
        risk_state_counts[agg_state] = risk_state_counts.get(agg_state, 0) + 1
        
        daily_summaries.append({
            "date": str(dt),
            "combined_exposure_multiplier": summary.get("combined_exposure_multiplier", 1.0),
            "combined_safe_fraction": summary.get("combined_safe_fraction", 0.0),
            "aggregated_risk_state": agg_state,
            "active_strategies": summary.get("active_strategies", 0),
            "strategies": {k: v for k, v in attr.items() if k != "_summary"}
        })
        
        # Accumulate per-strategy statistics
        for strat_name, strat_data in attr.items():
            if strat_name == "_summary":
                continue
            
            if strat_name not in strategy_impacts:
                strategy_impacts[strat_name] = {
                    "exposure_impacts": [],
                    "safe_impacts": [],
                    "risk_on_count": 0,
                    "risk_off_count": 0,
                    "neutral_count": 0,
                    "enabled_count": 0,
                    "dormant_count": 0
                }
            
            exp_impact = strat_data.get("exposure_impact", 1.0)
            safe_impact = strat_data.get("safe_impact", 0.0)
            risk_state = strat_data.get("risk_state", "neutral")
            
            strategy_impacts[strat_name]["exposure_impacts"].append(exp_impact)
            strategy_impacts[strat_name]["safe_impacts"].append(safe_impact)
            
            if risk_state == "risk-on":
                strategy_impacts[strat_name]["risk_on_count"] += 1
            elif risk_state == "risk-off":
                strategy_impacts[strat_name]["risk_off_count"] += 1
            else:
                strategy_impacts[strat_name]["neutral_count"] += 1
            
            # Check if strategy was dormant (no meaningful impact)
            if abs(exp_impact - 1.0) < 0.001 and abs(safe_impact) < 0.001:
                strategy_impacts[strat_name]["dormant_count"] += 1
            else:
                strategy_impacts[strat_name]["enabled_count"] += 1
    
    # Compute per-strategy summary statistics
    strategy_stats = {}
    for strat_name, impacts in strategy_impacts.items():
        exp_impacts = impacts["exposure_impacts"]
        safe_impacts = impacts["safe_impacts"]
        
        strategy_stats[strat_name] = {
            "avg_exposure_impact": float(np.mean(exp_impacts)) if exp_impacts else 1.0,
            "max_exposure_impact": float(np.max(exp_impacts)) if exp_impacts else 1.0,
            "min_exposure_impact": float(np.min(exp_impacts)) if exp_impacts else 1.0,
            "avg_safe_impact": float(np.mean(safe_impacts)) if safe_impacts else 0.0,
            "max_safe_impact": float(np.max(safe_impacts)) if safe_impacts else 0.0,
            "risk_on_days": impacts["risk_on_count"],
            "risk_off_days": impacts["risk_off_count"],
            "neutral_days": impacts["neutral_count"],
            "active_days": impacts["enabled_count"],
            "dormant_days": impacts["dormant_count"],
            "activity_rate": float(impacts["enabled_count"]) / len(exp_impacts) if exp_impacts else 0.0
        }
    
    # Identify most impactful strategies
    strategy_importance = []
    for strat_name, stats in strategy_stats.items():
        # Importance score: deviation from neutral (1.0 for exposure, 0.0 for safe)
        exp_deviation = abs(stats["avg_exposure_impact"] - 1.0)
        safe_contribution = stats["avg_safe_impact"]
        importance = exp_deviation + safe_contribution
        
        strategy_importance.append((strat_name, importance, stats["activity_rate"]))
    
    strategy_importance.sort(key=lambda x: x[1], reverse=True)
    
    # Overall summary
    total_days = len(attribution_rows)
    summary = {
        "wave_name": wave_name,
        "mode": mode,
        "days": days,
        "total_trading_days": total_days,
        "risk_on_days": risk_state_counts.get("risk-on", 0),
        "risk_off_days": risk_state_counts.get("risk-off", 0),
        "neutral_days": risk_state_counts.get("neutral", 0),
        "risk_off_percentage": float(risk_state_counts.get("risk-off", 0)) / total_days if total_days > 0 else 0.0,
        "most_impactful_strategies": [
            {"name": name, "importance_score": float(score), "activity_rate": float(rate)}
            for name, score, rate in strategy_importance[:5]
        ],
        "dormant_strategies": [
            name for name, stats in strategy_stats.items()
            if stats["activity_rate"] < 0.1  # Less than 10% active
        ]
    }
    
    return {
        "ok": True,
        "summary": summary,
        "daily_attribution": daily_summaries,
        "strategy_stats": strategy_stats
    }


# ------------------------------------------------------------
# Module Initialization - Validate Wave ID Registry
# ------------------------------------------------------------

def _log_wave_id_warnings():
    """Log warnings for wave_id registry validation at module import."""
    warnings = validate_wave_id_registry()
    if warnings:
        import sys
        print("⚠️  WAVE_ID_REGISTRY Validation Warnings:", file=sys.stderr)
        for warning in warnings:
            print(f"  - {warning}", file=sys.stderr)
    else:
        # Silent success - no warnings to log
        pass

# Run validation on import
_log_wave_id_warnings()

    