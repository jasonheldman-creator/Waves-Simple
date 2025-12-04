"""
waves_equity_waves.py
WAVES Intelligence™ – Equity Waves Engine (10 equity waves version)

This script is designed for GitHub use. It:

- Defines 10 equity Waves in a config block
- Loads each Wave's holdings from a published Google Sheets CSV URL
- Fetches latest prices via yfinance
- Computes a simple Wave NAV and daily return
- Computes benchmark ETF performance for the same day
- Prints a summary table for all Waves

You can run it as:
    python waves_equity_waves.py

Prereqs (pip):
    pip install pandas yfinance requests
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional
import datetime as dt

import pandas as pd
import yfinance as yf
import requests


# -------------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------------

@dataclass
class WaveConfig:
    """Configuration for a single Equity Wave."""
    code: str                 # short ID (used in logs, filenames, etc.)
    name: str                 # human-readable Wave name
    benchmark: str            # benchmark ETF ticker (SPY, IWM, etc.)
    holdings_csv_url: str     # published CSV URL for holdings (Ticker, Weight% or similar)
    default_notional: float = 1_000_000  # hypothetical capital for NAV calc


# 🔧 Paste your published CSV URLs for each Wave here.
# For now, I’ve used placeholder URLs ("") – you’ll paste the real ones
# like the one you just sent:
#   https://docs.google.com/spreadsheets/d/.../pub?gid=...&single=true&output=csv
WAVES_CONFIG: List[WaveConfig] = [
    WaveConfig(
        code="SPX",
        name="S&P 500 Core Equity Wave",
        benchmark="SPY",
        holdings_csv_url="",  # TODO: paste S&P 500 Wave holdings CSV URL
    ),
    WaveConfig(
        code="USMKT",
        name="Total US Market Equity Wave",
        benchmark="VTI",
        holdings_csv_url="",  # TODO: paste Total Market Wave holdings CSV URL
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
# CORE FUNCTIONS
# -------------------------------------------------------------------

def load_holdings_from_csv(url: str) -> pd.DataFrame:
    """
    Load holdings from a published CSV.

    Requirements:
    - Must contain at least a 'Ticker' column.
    - If it contains a 'Weight' / 'Weight (%)' column, that will be used.
      Otherwise, equal-weights are assigned.
    """
    if not url:
        raise ValueError("Holdings CSV URL is empty. Please set holdings_csv_url in config.")

    resp = requests.get(url)
    resp.raise_for_status()

    df = pd.read_csv(pd.compat.StringIO(resp.text))
    # Normalize column names
    df.columns = [c.strip() for c in df.columns]

    # Find ticker column
    ticker_col = None
    for candidate in ["Ticker", "Symbol", "ticker", "symbol"]:
        if candidate in df.columns:
            ticker_col = candidate
            break
    if ticker_col is None:
        raise ValueError("No Ticker/Symbol column found in holdings CSV.")

    df = df[[ticker_col] + [c for c in df.columns if c != ticker_col]]
    df.rename(columns={ticker_col: "Ticker"}, inplace=True)

    # Try to find weight column
    weight_col: Optional[str] = None
    for candidate in ["Weight", "Weight (%)", "Weight%", "Index Weight"]:
        if candidate in df.columns:
            weight_col = candidate
            break

    if weight_col is None:
        # Equal weight if nothing provided
        n = len(df)
        df["Weight"] = 1 / n
    else:
        w = df[weight_col].astype(str).str.replace("%", "").astype(float)
        if w.max() > 1.5:  # assume it's percentage
            w = w / 100.0
        df["Weight"] = w

    # Normalize weights to sum to 1
    df["Weight"] = df["Weight"] / df["Weight"].sum()

    return df[["Ticker", "Weight"]]


def fetch_prices(tickers: List[str]) -> pd.Series:
    """
    Fetch latest close prices for a list of tickers using yfinance.

    Returns a Series indexed by ticker.
    """
    data = yf.download(
        tickers=" ".join(sorted(set(tickers))),
        period="5d",
        interval="1d",
        auto_adjust=True,
        progress=False,
        threads=True,
    )
    # yfinance returns multiindex if multiple tickers
    if isinstance(data.columns, pd.MultiIndex):
        close = data["Close"].iloc[-1]
    else:
        close = data["Close"].iloc[-1:]
    close.index = [t.upper() for t in close.index]
    return close


def compute_wave_nav(wave: WaveConfig, holdings: pd.DataFrame, prices: pd.Series) -> dict:
    """
    Compute a simple Wave NAV and daily return vs benchmark.

    For now:
    - Position value = default_notional * weight
    - NAV = sum(position values)
    - Daily return is approximated using today's vs yesterday's price.
      (We fetch 5 days and compute last 2 bars.)
    """
    # Align holdings with prices
    tickers = holdings["Ticker"].str.upper()
    weights = holdings["Weight"].values

    # Re-fetch with 2 columns per ticker for last 2 days
    hist = yf.download(
        tickers=" ".join(sorted(set(tickers))),
        period="5d",
        interval="1d",
        auto_adjust=True,
        progress=False,
        threads=True,
    )

    if isinstance(hist.columns, pd.MultiIndex):
        close = hist["Close"].iloc[-2:]
    else:
        close = hist[["Close"]].iloc[-2:]

    # Compute per-ticker returns (last day vs previous)
    last = close.iloc[-1]
    prev = close.iloc[-2]
    rets = (last - prev) / prev
    rets.index = [t.upper() for t in rets.index]

    # Map returns back to holdings
    rets_vec = rets.reindex(tickers).fillna(0.0).values

    # Wave return = sum(weight * ret)
    wave_ret = float((weights * rets_vec).sum())

    # NAV (simple) = default_notional * (1 + wave_ret)
    nav = wave.default_notional * (1 + wave_ret)

    # Benchmark return
    bench_hist = yf.download(
        tickers=wave.benchmark,
        period="5d",
        interval="1d",
        auto_adjust=True,
        progress=False,
        threads=True,
    )
    bench_close = bench_hist["Close"].iloc[-2:]
    bench_ret = float((bench_close.iloc[-1] - bench_close.iloc[-2]) / bench_close.iloc[-2])

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
# MAIN
# -------------------------------------------------------------------

def main() -> None:
    print("WAVES Intelligence™ – Equity Waves Engine\n")
    print(f"Run date: {dt.date.today().isoformat()}\n")

    results = []
    for wave in WAVES_CONFIG:
        if not wave.holdings_csv_url:
            print(f"⚠️  Skipping {wave.code} – no holdings_csv_url set yet.")
            continue

        print(f"Loading holdings for {wave.code} – {wave.name} ...")
        holdings = load_holdings_from_csv(wave.holdings_csv_url)
        print(f"  Loaded {len(holdings)} positions.")

        tickers = holdings["Ticker"].tolist()
        # Fetching prices happens inside compute_wave_nav.

        stats = compute_wave_nav(wave, holdings, None)
        results.append(stats)
        print(
            f"  WaveReturn: {stats['wave_return']:.4%}  "
            f"Bench({wave.benchmark}): {stats['benchmark_return']:.4%}  "
            f"Alpha: {stats['alpha']:.4%}\n"
        )

    if not results:
        print("\nNo Waves processed (all holdings_csv_url are empty). "
              "Add URLs in WAVES_CONFIG at the top of the file.")
        return

    df = pd.DataFrame(results)
    df = df[
        ["code", "name", "benchmark", "nav", "wave_return", "benchmark_return", "alpha"]
    ].sort_values("code")

    print("\n=== Equity Waves Summary ===\n")
    print(df.to_string(index=False, float_format=lambda x: f"{x:,.4f}"))


if __name__ == "__main__":
    main()
    # ... keep all your existing imports, WaveConfig, WAVES_CONFIG, etc ...

import pandas as pd
import datetime as dt


def run_equity_waves() -> pd.DataFrame:
    """
    Run all configured equity waves and return a summary DataFrame.
    This is what the Streamlit app will call.
    """
    results = []
    run_date = dt.date.today().isoformat()

    for wave in WAVES_CONFIG:
        if not wave.holdings_csv_url:
            # Skip waves without a holdings URL configured
            continue

        holdings = load_holdings_from_csv(wave.holdings_csv_url)
        stats = compute_wave_nav(wave, holdings, None)
        stats["run_date"] = run_date
        results.append(stats)

    if not results:
        return pd.DataFrame(
            columns=["code", "name", "benchmark", "nav", "wave_return", "benchmark_return", "alpha", "run_date"]
        )

    df = pd.DataFrame(results)
    df = df[
        ["code", "name", "benchmark", "nav", "wave_return", "benchmark_return", "alpha", "run_date"]
    ].sort_values("code")
    return df


if __name__ == "__main__":
    # Keep a CLI runner for debugging
    df = run_equity_waves()
    if df.empty:
        print("No waves processed – check holdings_csv_url in WAVES_CONFIG.")
    else:
        print(df.to_string(index=False, float_format=lambda x: f"{x:,.4f}"))