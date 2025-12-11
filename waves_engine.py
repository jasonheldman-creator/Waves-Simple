# waves_engine.py — WAVES Intelligence™ Vector Engine (Mobile-Friendly)
#
# Key features:
#   • Internal WAVE_WEIGHTS (no CSV required by default)
#   • Composite ETF benchmarks for each Wave
#   • USE_FULL_WAVE_HISTORY flag (but app uses 365D windows)
#   • compute_history_nav(...) returns NAV + daily returns for Wave & benchmark
#   • Mode multipliers: Standard, Alpha-Minus-Beta, Private Logic
#   • Helper utilities for benchmark mix + top-10 holdings

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List

import numpy as np
import pandas as pd

try:
    import yfinance as yf
except ImportError:  # pragma: no cover - runtime environment responsibility
    yf = None

# ------------------------------------------------------------
# Config
# ------------------------------------------------------------

USE_FULL_WAVE_HISTORY: bool = False  # mobile-friendly; app uses 365D rolling

TRADING_DAYS_PER_YEAR = 252

# Mode multipliers (can be tuned)
MODE_MULTIPLIERS: Dict[str, float] = {
    "Standard": 1.0,
    "Alpha-Minus-Beta": 0.8,
    "Private Logic": 1.15,
}

# ------------------------------------------------------------
# Wave & Benchmark definitions
# NOTE:
#   • These are illustrative; you can tune tickers/weights later.
#   • Weights do NOT have to be exactly 10 holdings per Wave; top-10
#     screen will just show everything provided.
# ------------------------------------------------------------

@dataclass
class Holding:
    ticker: str
    weight: float
    name: str | None = None


# Wave Holdings (example internal config)
WAVE_WEIGHTS: Dict[str, List[Holding]] = {
    "S&P 500 Wave": [
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
    "AI Wave": [
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
    "Quantum Computing Wave": [
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
    "Future Power & Energy Wave": [
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
    "Clean Transit-Infrastructure Wave": [
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
    "Small Cap Growth Wave": [
        Holding("IWO", 0.30, "iShares Russell 2000 Growth ETF"),
        Holding("VBK", 0.30, "Vanguard Small-Cap Growth ETF"),
        Holding("ARKK", 0.10, "ARK Innovation ETF"),
        Holding("ZS", 0.10, "Zscaler Inc."),
        Holding("DDOG", 0.10, "Datadog Inc."),
        Holding("NET", 0.10, "Cloudflare Inc."),
    ],
    "Small to Mid Cap Growth Wave": [
        Holding("IWP", 0.30, "iShares Russell Mid-Cap Growth ETF"),
        Holding("MDY", 0.30, "SPDR S&P MidCap 400 ETF"),
        Holding("IWO", 0.20, "iShares Russell 2000 Growth ETF"),
        Holding("SMH", 0.20, "VanEck Semiconductor ETF"),
    ],
    "Crypto Income Wave": [
        Holding("BTC-USD", 0.40, "Bitcoin (USD)"),
        Holding("ETH-USD", 0.30, "Ethereum (USD)"),
        Holding("MSTR", 0.10, "MicroStrategy Inc."),
        Holding("COIN", 0.10, "Coinbase Global Inc."),
        Holding("BITO", 0.10, "ProShares Bitcoin Strategy ETF"),
    ],
    "SmartSafe Money Market Wave": [
        Holding("BIL", 0.50, "SPDR Bloomberg 1-3 Month T-Bill ETF"),
        Holding("SGOV", 0.50, "iShares 0-3 Month Treasury Bond ETF"),
    ],
}

# Composite benchmarks (ETF mixes)
BENCHMARK_WEIGHTS: Dict[str, List[Holding]] = {
    "S&P 500 Wave": [Holding("SPY", 1.0, "SPDR S&P 500 ETF")],
    "AI Wave": [
        Holding("QQQ", 0.60, "Invesco QQQ Trust"),
        Holding("SMH", 0.40, "VanEck Semiconductor ETF"),
    ],
    "Quantum Computing Wave": [
        Holding("QQQ", 0.50, "Invesco QQQ Trust"),
        Holding("SMH", 0.25, "VanEck Semiconductor ETF"),
        Holding("IBM", 0.25, "International Business Machines Corp."),
    ],
    "Future Power & Energy Wave": [
        Holding("XLE", 0.50, "Energy Select Sector SPDR"),
        Holding("ICLN", 0.50, "iShares Global Clean Energy"),
    ],
    "Clean Transit-Infrastructure Wave": [
        Holding("PAVE", 0.60, "Global X U.S. Infrastructure Development"),
        Holding("XLI", 0.40, "Industrial Select Sector SPDR"),
    ],
    "Small Cap Growth Wave": [
        Holding("IWO", 0.50, "iShares Russell 2000 Growth ETF"),
        Holding("VBK", 0.50, "Vanguard Small-Cap Growth ETF"),
    ],
    "Small to Mid Cap Growth Wave": [
        Holding("IWP", 0.50, "iShares Russell Mid-Cap Growth ETF"),
        Holding("IWO", 0.50, "iShares Russell 2000 Growth ETF"),
    ],
    "Crypto Income Wave": [
        Holding("BTC-USD", 0.50, "Bitcoin (USD)"),
        Holding("ETH-USD", 0.50, "Ethereum (USD)"),
    ],
    "SmartSafe Money Market Wave": [
        Holding("BIL", 0.50, "SPDR Bloomberg 1-3 Month T-Bill"),
        Holding("SGOV", 0.50, "iShares 0-3 Month Treasury Bond ETF"),
    ],
}


# ------------------------------------------------------------
# Public API
# ------------------------------------------------------------

def get_all_waves() -> list[str]:
    """Return sorted list of Wave names."""
    return sorted(WAVE_WEIGHTS.keys())


def get_modes() -> list[str]:
    """Return list of available modes (Standard, AMB, PL)."""
    return list(MODE_MULTIPLIERS.keys())


def _normalize_weights(holdings: List[Holding]) -> pd.Series:
    """Return normalized weight Series indexed by ticker."""
    if not holdings:
        return pd.Series(dtype=float)

    df = pd.DataFrame(
        [{"ticker": h.ticker, "weight": h.weight} for h in holdings]
    )
    df = df.groupby("ticker", as_index=False)["weight"].sum()
    total = df["weight"].sum()
    if total <= 0:
        return pd.Series(dtype=float)
    df["weight"] = df["weight"] / total
    return df.set_index("ticker")["weight"]


def _download_history(tickers: list[str], days: int) -> pd.DataFrame:
    """
    Download daily adjusted close prices for given tickers.

    Returns DataFrame indexed by Date with columns per ticker.
    """
    if yf is None:
        raise RuntimeError(
            "yfinance is not available in this environment. "
            "Please ensure yfinance is installed."
        )

    # Add a small buffer to ensure we have enough history
    lookback_days = max(days + 10, days)
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

    # yfinance sometimes returns a multi-index (column level: field, ticker)
    if isinstance(data.columns, pd.MultiIndex):
        if "Adj Close" in data.columns.get_level_values(0):
            data = data["Adj Close"]
        elif "Close" in data.columns.get_level_values(0):
            data = data["Close"]
        else:
            # fallback: pick first level
            top = data.columns.levels[0][0]
            data = data[top]

    if isinstance(data, pd.Series):
        data = data.to_frame()

    data = data.sort_index()
    data = data.ffill().bfill()  # basic cleaning
    return data


def compute_history_nav(
    wave_name: str,
    mode: str = "Standard",
    days: int = 365,
) -> pd.DataFrame:
    """
    Compute Wave & Benchmark NAV and daily returns over a given window.

    Returns DataFrame indexed by Date with columns:
        ['wave_nav', 'bm_nav', 'wave_ret', 'bm_ret']

    Notes:
        • Wave returns scaled by MODE_MULTIPLIERS[mode].
        • Benchmark always uses implicit "Standard" (no mode multiplier).
    """
    if wave_name not in WAVE_WEIGHTS:
        raise ValueError(f"Unknown Wave: {wave_name}")

    if mode not in MODE_MULTIPLIERS:
        raise ValueError(f"Unknown mode: {mode}")

    wave_holdings = WAVE_WEIGHTS[wave_name]
    bm_holdings = BENCHMARK_WEIGHTS.get(wave_name, [])

    wave_weights = _normalize_weights(wave_holdings)
    bm_weights = _normalize_weights(bm_holdings)

    tickers_wave = list(wave_weights.index)
    tickers_bm = list(bm_weights.index)
    all_tickers = sorted(set(tickers_wave + tickers_bm))

    if not all_tickers:
        # Degenerate case: no tickers configured
        return pd.DataFrame(
            columns=["wave_nav", "bm_nav", "wave_ret", "bm_ret"], dtype=float
        )

    price_df = _download_history(all_tickers, days=days)
    if price_df.empty:
        return pd.DataFrame(
            columns=["wave_nav", "bm_nav", "wave_ret", "bm_ret"], dtype=float
        )

    # Restrict to requested window (last N days)
    if len(price_df) > days:
        price_df = price_df.iloc[-days:]

    # Daily returns
    ret_df = price_df.pct_change().dropna(how="all")

    # Align weight vectors to available columns
    wave_weights_aligned = wave_weights.reindex(price_df.columns).fillna(0.0)
    bm_weights_aligned = bm_weights.reindex(price_df.columns).fillna(0.0)

    wave_ret = (ret_df * wave_weights_aligned).sum(axis=1)
    bm_ret = (ret_df * bm_weights_aligned).sum(axis=1)

    # Apply mode multiplier to Wave returns only
    multiplier = MODE_MULTIPLIERS.get(mode, 1.0)
    wave_ret = wave_ret * multiplier

    # Compute NAV (start at 1.0)
    wave_nav = (1.0 + wave_ret).cumprod()
    bm_nav = (1.0 + bm_ret).cumprod()

    out = pd.DataFrame(
        {
            "wave_nav": wave_nav,
            "bm_nav": bm_nav,
            "wave_ret": wave_ret,
            "bm_ret": bm_ret,
        }
    )
    out.index.name = "Date"
    return out


def get_benchmark_mix_table() -> pd.DataFrame:
    """
    Return a table summarizing the ETF mix used for each Wave's benchmark.

    Columns:
        • Wave
        • Ticker
        • Name
        • Weight
    """
    rows = []
    for wave, holdings in BENCHMARK_WEIGHTS.items():
        if not holdings:
            continue
        weights = _normalize_weights(holdings)
        for h in holdings:
            if h.ticker not in weights.index:
                continue
            rows.append(
                {
                    "Wave": wave,
                    "Ticker": h.ticker,
                    "Name": h.name or "",
                    "Weight": float(weights[h.ticker]),
                }
            )

    if not rows:
        return pd.DataFrame(columns=["Wave", "Ticker", "Name", "Weight"])

    df = pd.DataFrame(rows)
    df = df.sort_values(["Wave", "Weight"], ascending=[True, False])
    return df


def get_wave_holdings(wave_name: str) -> pd.DataFrame:
    """
    Return top holdings for a given Wave as a DataFrame with columns:
        ['Ticker', 'Name', 'Weight']
    """
    holdings = WAVE_WEIGHTS.get(wave_name, [])
    if not holdings:
        return pd.DataFrame(columns=["Ticker", "Name", "Weight"])

    weights = _normalize_weights(holdings)

    rows = []
    for h in holdings:
        if h.ticker not in weights.index:
            continue
        rows.append(
            {
                "Ticker": h.ticker,
                "Name": h.name or "",
                "Weight": float(weights[h.ticker]),
            }
        )

    df = pd.DataFrame(rows)
    df = df.drop_duplicates(subset=["Ticker"])
    df = df.sort_values("Weight", ascending=False)
    return df