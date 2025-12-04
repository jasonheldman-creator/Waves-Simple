"""
waves_equity_universe_v2.py
WAVES Intelligence™ – Equity Waves Engine (10 equity waves)

This module is designed to be used both:
- directly from the command line, and
- as the backend engine for the Streamlit app (app.py).

It:
- Defines 10 Equity Waves in WAVES_CONFIG
- Loads each Wave's holdings from a published Google Sheets CSV URL
- Normalizes portfolio weights
- Pulls recent prices via yfinance
- Computes a simple Wave daily return vs. a benchmark ETF
- Returns a summary DataFrame via run_equity_waves()
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional
import datetime as dt
import io

import pandas as pd
import yfinance as yf
import requests


# -------------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------------


@dataclass
class WaveConfig:
    """Configuration for a single Equity Wave."""
    code: str                 # short ID (e.g., "SPX")
    name: str                 # human-readable Wave name
    benchmark: str            # benchmark ETF ticker (SPY, QQQ, etc.)
    holdings_csv_url: str     # published CSV URL for holdings
    default_notional: float = 1_000_000.0  # hypothetical capital for NAV calc


# 👇 EDIT THESE URLs as you build each Wave.
# SPX already points to your Google Sheets master CSV.
WAVES_CONFIG: List[WaveConfig] = [
    WaveConfig(
        code="SPX",
        name="S&P 500 Core Equity Wave",
        benchmark="SPY",
        holdings_csv_url=(
            "https://docs.google.com/spreadsheets/d/e/"
            "2PACX-1vT7VpPdWSUSyZP9CVXZwTgqx7a7mMD2aQMRqSESqZgiagh8wSeEm3RAWHvLlWmJtLqYrqj7UVjQIpq9/"
            "pub?gid=711820877&single=true&output=csv"
        ),
    ),
    WaveConfig(
        code="USMKT",
        name="Total US Market Equity Wave",
        benchmark="VTI",
        holdings_csv_url="",  # TODO: paste CSV URL
    ),
    WaveConfig(
        code="LGRW",
        name="US Large Growth Equity Wave",
        benchmark="QQQ",
        holdings_csv_url="",  # TODO
    ),
    WaveConfig(
        code="SCG",
        name="US Small Cap Growth Wave",
        benchmark="IWO",
        holdings_csv_url="",  # TODO
    ),
    WaveConfig(
        code="SMID",
        name="US Small–Mid Growth Wave",
        benchmark="IJT",
        holdings_csv_url="",  # TODO
    ),
    WaveConfig(
        code="AITECH",
        name="AI & Innovation Equity Wave",
        benchmark="QQQ",
        holdings_csv_url="",  # TODO
    ),
    WaveConfig(
        code="ROBO",
        name="Automation & Robotics Equity Wave",
        benchmark="BOTZ",
        holdings_csv_url="",  # TODO
    ),
    WaveConfig(
        code="ENERGYF",
        name="Future Power & Energy Wave",
        benchmark="ICLN",
        holdings_csv_url="",  # TODO
    ),
    WaveConfig(
        code="EQINC",
        name="Global Equity Income Wave",
        benchmark="SCHD",
        holdings_csv_url="",  # TODO
    ),
    WaveConfig(
        code="INTL",
        name="International + EM Equity Wave",
        benchmark="VEA",
        holdings_csv_url="",  # TODO
    ),
]


# -------------------------------------------------------------------
# HELPER FUNCTIONS
# -------------------------------------------------------------------


def load_holdings_from_csv(url: str) -> pd.DataFrame:
    """
    Load holdings from a published CSV.

    Requirements:
    - Must contain at least a 'Ticker' or 'Symbol' column.
    - If it contains a 'Weight' / 'Weight (%)' column, that will be used.
      Otherwise, equal-weights are assigned.

    Returns a DataFrame with columns: ['Ticker', 'Weight'] and weights normalized to sum to 1.0
    """
    if not url:
        raise ValueError("Holdings CSV URL is empty. Please set holdings_csv_url in WAVES_CONFIG.")

    resp = requests.get(url)
    resp.raise_for_status()

    df = pd.read_csv(io.StringIO(resp.text))
    df.columns = [c.strip() for c in df.columns]

    # Find ticker column
    ticker_col: Optional[str] = None
    for candidate in ("Ticker", "Symbol", "ticker", "symbol"):
        if candidate in df.columns:
            ticker_col = candidate
            break
    if ticker_col is None:
        raise ValueError("No Ticker/Symbol column found in holdings CSV.")

    df.rename(columns={ticker_col: "Ticker"}, inplace=True)

    # Try to find weight column
    weight_col: Optional[str] = None
    for candidate in ("Weight", "Weight (%)", "Weight%", "Index Weight"):
        if candidate in df.columns:
            weight_col = candidate
            break

    if weight_col is None:
        # Equal weight if nothing provided
        n = len(df)
        df["Weight"] = 1.0 / max(n, 1)
    else:
        w = df[weight_col].astype(str).str.replace("%", "", regex=False)
        w = w.replace("", "0").astype(float)
        if w.max() > 1.5:  # assume it's expressed in percent
            w = w / 100.0
        df["Weight"] = w

    # Normalize weights to sum to 1
    total_w = df["Weight"].sum()
    if total_w <= 0:
        raise ValueError("Total portfolio weight <= 0 in holdings CSV.")
    df["Weight"] = df["Weight"] / total_w

    return df[["Ticker", "Weight"]]


def _download_close_prices(tickers: List[str]) -> pd.DataFrame:
    """
    Download recent adjusted close prices for a list of tickers.
    Returns a DataFrame with last 2 rows of Close prices (index = date, columns = tickers).
    """
    unique_tickers = sorted(set(tickers))
    if not unique_tickers:
        raise ValueError("No tickers provided to _download_close_prices.")

    data = yf.download(
        tickers=" ".join(unique_tickers),
        period="5d",
        interval="1d",
        auto_adjust=True,
        progress=False,
        threads=True,
    )

    # Handle single vs multi-ticker shapes
    if isinstance(data.columns, pd.MultiIndex):
        close = data["Close"]
    else:
        # Single ticker: create a 2D frame with the ticker as the column name
        if "Close" not in data.columns:
            raise ValueError("No 'Close' column found in price data.")
        close = data[["Close"]]
        close.columns = [unique_tickers[0]]

    # We only need last 2 days
    close = close.tail(2)
    return close


def compute_wave_nav(wave: WaveConfig, holdings: pd.DataFrame) -> dict:
    """
    Compute a simple Wave NAV and daily return vs benchmark.

    - Wave return = sum(weight_i * daily_return_i)
    - NAV = default_notional * (1 + wave_return)
    - Alpha = wave_return - benchmark_return
    """
    tickers = holdings["Ticker"].astype(str).str.upper().tolist()
    weights = holdings["Weight"].values

    # Wave prices & returns
    prices = _download_close_prices(tickers)
    last = prices.iloc[-1]
    prev = prices.iloc[-2]
    rets = (last - prev) / prev
    rets = rets.reindex([t.upper() for t in prices.columns]).fillna(0.0)

    # Map returns back to holdings vector
    returns_vec = rets.reindex(tickers).fillna(0.0).values
    wave_ret = float((weights * returns_vec).sum())
    nav = wave.default_notional * (1.0 + wave_ret)

    # Benchmark
    bench_prices = _download_close_prices([wave.benchmark])
    b_last = bench_prices.iloc[-1, 0]
    b_prev = bench_prices.iloc[-2, 0]
    bench_ret = float((b_last - b_prev) / b_prev)

    return {
        "code": wave.code,
        "name": wave.name,
        "benchmark": wave.benchmark,
        "nav": nav,
        "wave_return": wave_ret,
        "benchmark_return": bench_ret,
        "alpha": wave_ret - bench_ret,
    }


# -------------------------------------------------------------------
# PUBLIC ENTRYPOINT FOR STREAMLIT (and CLI)
# -------------------------------------------------------------------


def run_equity_waves() -> pd.DataFrame:
    """
    Run all configured equity waves and return a summary DataFrame.

    Columns:
        ['code', 'name', 'benchmark', 'nav',
         'wave_return', 'benchmark_return', 'alpha', 'run_date']
    """
    results = []
    run_date = dt.date.today().isoformat()

    for wave in WAVES_CONFIG:
        if not wave.holdings_csv_url:
            # Skip waves that don't have a holdings URL yet
            continue

        try:
            holdings = load_holdings_from_csv(wave.holdings_csv_url)
            stats = compute_wave_nav(wave, holdings)
            stats["run_date"] = run_date
            results.append(stats)
        except Exception as e:
            # In production you'd log this properly; for now we just record the error.
            results.append({
                "code": wave.code,
                "name": wave.name,
                "benchmark": wave.benchmark,
                "nav": float("nan"),
                "wave_return": float("nan"),
                "benchmark_return": float("nan"),
                "alpha": float("nan"),
                "run_date": run_date,
                "error": str(e),
            })

    if not results:
        return pd.DataFrame(
            columns=[
                "code",
                "name",
                "benchmark",
                "nav",
                "wave_return",
                "benchmark_return",
                "alpha",
                "run_date",
            ]
        )

    df = pd.DataFrame(results)
    # Ensure standard column order when there are no errors
    base_cols = ["code", "name", "benchmark", "nav", "wave_return",
                 "benchmark_return", "alpha", "run_date"]
    extra_cols = [c for c in df.columns if c not in base_cols]
    df = df[base_cols + extra_cols]
    return df.sort_values("code")


# -------------------------------------------------------------------
# CLI MODE
# -------------------------------------------------------------------

if __name__ == "__main__":
    print("WAVES Intelligence™ – Equity Waves Engine")
    print(f"Run date: {dt.date.today().isoformat()}\n")

    summary = run_equity_waves()
    if summary.empty:
        print("No waves processed – check holdings_csv_url in WAVES_CONFIG.")
    else:
        pd.set_option("display.float_format", lambda x: f"{x:,.4f}")
        print(summary.to_string(index=False))