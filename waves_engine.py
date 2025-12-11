"""
waves_engine.py — WAVES Intelligence™ Vector Engine (Stage 5+)

Features
--------
- Internal WAVE_WEIGHTS (no CSV dependency).
- Three risk modes:
    • Standard
    • Alpha-Minus-Beta (de-risked via lower return scaling)
    • Private Logic (enhanced via higher return scaling)
- Optional Full_Wave_History.csv (Date/Wave/NAV), else live/simulated prices.
- Robust price fetching with yfinance + simulator fallback.
- Daily performance logging:
    • logs/performance/<Wave>_performance_daily.csv
      with Date, Wave, Mode, NAV, Return, CumReturn.
- Composite Benchmarks (RESTORED):
    • Each Wave has its own custom benchmark portfolio of 1–3 ETFs.
    • Alpha is computed vs these composite benchmarks.
- Benchmark Helpers:
    • get_benchmark_wave_for(wave_name) -> benchmark portfolio name.
    • get_benchmark_composition(benchmark_name) -> dict of {ETF: weight}.
"""

import os
import datetime as dt
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

try:
    import yfinance as yf
except ImportError:
    yf = None  # App will fall back to simulator.


# ============================================================
# INTERNAL MASTER WEIGHTS (no CSV required)
# ============================================================

WAVE_WEIGHTS: Dict[str, Dict[str, float]] = {
    "AI Wave": {
        "NVDA": 0.15,
        "MSFT": 0.10,
        "GOOGL": 0.10,
        "AMD": 0.10,
        "PLTR": 0.10,
        "META": 0.10,
        "CRWD": 0.10,
        "SNOW": 0.10,
        "TSLA": 0.05,
        "AVGO": 0.10,
    },
    "Cloud & Software Wave": {
        "MSFT": 0.20,
        "CRM": 0.10,
        "ADBE": 0.10,
        "NOW": 0.10,
        "GOOGL": 0.10,
        "AMZN": 0.20,
        "PANW": 0.10,
        "DDOG": 0.10,
    },
    "Crypto Income Wave": {
        "COIN": 0.33,
        "MSTR": 0.33,
        "RIOT": 0.34,
    },
    "Future Power & Energy Wave": {
        "NEE": 0.10,
        "DUK": 0.10,
        "ENPH": 0.10,
        "PSX": 0.10,
        "SLB": 0.10,
        "CVX": 0.10,
        "HAL": 0.10,
        "LIN": 0.10,
        "MPC": 0.10,
        "XOM": 0.10,
    },
    "Small Cap Growth Wave": {
        "PLTR": 0.10,
        "ROKU": 0.10,
        "UPST": 0.10,
        "DOCN": 0.10,
        "FSLY": 0.10,
        "AI": 0.10,
        "SMCI": 0.10,
        "TTD": 0.10,
        "AFRM": 0.10,
        "PATH": 0.10,
    },
    "Quantum Computing Wave": {
        "IBM": 0.25,
        "IONQ": 0.25,
        "NVDA": 0.25,
        "AMD": 0.25,
    },
    "Clean Transit-Infrastructure Wave": {
        "TSLA": 0.25,
        "NIO": 0.15,
        "BLNK": 0.15,
        "CHPT": 0.15,
        "RIVN": 0.15,
        "F": 0.15,
    },
    "S&P 500 Wave": {
        "AAPL": 0.06,
        "MSFT": 0.06,
        "AMZN": 0.06,
        "NVDA": 0.06,
        "GOOGL": 0.06,
        "META": 0.06,
        "TSLA": 0.06,
        "BRK.B": 0.06,
        "LLY": 0.06,
        "JPM": 0.06,
    },
    "Income Wave": {
        "HDV": 0.20,
        "SCHD": 0.20,
        "JEPI": 0.20,
        "JEPQ": 0.20,
        "PFF": 0.20,
    },
    "SmartSafe Wave": {
        "BIL": 0.50,
        "SHV": 0.50,
    },
}

# ============================================================
# COMPOSITE BENCHMARK PORTFOLIOS (RESTORED)
# ============================================================

BENCHMARK_WEIGHTS: Dict[str, Dict[str, float]] = {
    # AI & Tech
    "AI Benchmark": {      # AI Wave benchmark
        "QQQ": 0.50,
        "IGV": 0.50,
    },
    "Cloud Benchmark": {   # Cloud & Software Wave benchmark
        "IGV": 0.60,
        "VGT": 0.40,
    },
    "Quantum Benchmark": { # Quantum Computing Wave benchmark
        "QQQ": 0.70,
        "VGT": 0.30,
    },

    # Crypto
    "Crypto Benchmark": {  # Crypto Income Wave benchmark
        "BITO": 0.50,
        "COIN": 0.50,
    },

    # Thematic / Sector
    "Future Power Benchmark": {  # Future Power & Energy Wave benchmark
        "XLE": 0.60,
        "ICLN": 0.40,
    },
    "Clean Transit Benchmark": {  # Clean Transit-Infrastructure Wave benchmark
        "PBW": 0.70,
        "ICLN": 0.30,
    },

    # Size / Style
    "Small Cap Benchmark": {   # Small Cap Growth Wave benchmark
        "IWM": 0.70,
        "VTWO": 0.30,
    },

    # Income & Cash
    "Income Benchmark": {      # Income Wave benchmark
        "SCHD": 0.50,
        "HDV": 0.50,
    },
    "SmartSafe Benchmark": {   # SmartSafe Wave benchmark
        "BIL": 1.00,
    },

    # Broad Market
    "S&P 500 Benchmark": {     # S&P 500 Wave benchmark
        "SPY": 1.00,
    },
}

# Maps each Wave to its composite benchmark portfolio
BENCHMARK_MAP: Dict[str, str] = {
    "AI Wave": "AI Benchmark",
    "Cloud & Software Wave": "Cloud Benchmark",
    "Crypto Income Wave": "Crypto Benchmark",
    "Future Power & Energy Wave": "Future Power Benchmark",
    "Small Cap Growth Wave": "Small Cap Benchmark",
    "Quantum Computing Wave": "Quantum Benchmark",
    "Clean Transit-Infrastructure Wave": "Clean Transit Benchmark",
    "Income Wave": "Income Benchmark",
    "SmartSafe Wave": "SmartSafe Benchmark",
    "S&P 500 Wave": "S&P 500 Benchmark",
}

FULL_HISTORY_FILE = "Full_Wave_History.csv"  # optional, Date/Wave/NAV

DEFAULT_LOOKBACK_DAYS = 365
SHORT_LOOKBACK_DAYS = 30

# Mode behaviour: these scale daily returns (not NAV directly)
MODE_MULTIPLIERS = {
    "Standard": 1.0,
    "Alpha-Minus-Beta": 0.80,   # ~20% de-risked vs Standard
    "Private Logic": 1.15,      # modestly enhanced
}


def get_benchmark_wave_for(wave_name: str) -> str:
    """Return the benchmark portfolio name for a given Wave."""
    return BENCHMARK_MAP.get(wave_name, "S&P 500 Benchmark")


def get_benchmark_composition(benchmark_name: str) -> Dict[str, float]:
    """
    Return the ETF composition for a given benchmark portfolio as {ticker: weight}.
    """
    return BENCHMARK_WEIGHTS.get(benchmark_name, {}).copy()