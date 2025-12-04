"""
waves_equity_universe_v2.py
WAVES Intelligence™ – Equity Waves Engine (10 equity waves)

This module is used both:
- directly from the command line, and
- as the backend engine for the Streamlit app (app.py).

Each Wave:
- Has its own holdings source (Google Sheets CSV URL or local CSV path)
- Uses 'Ticker' and either 'Weight'/'Weight (%)' or 'Market Value' to derive weights
- Computes a simple 1-day Wave return vs benchmark ETF
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional
import datetime as dt
import io
import os
from urllib.parse import urlparse

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
    holdings_src: str         # Google Sheets CSV URL OR local CSV path
    default_notional: float = 1_000_000.0  # hypothetical capital for NAV calc


# 👇 EDIT THESE paths/URLs as you build each Wave.

# NOTE: SPX is already wired to your Master_Stock_Sheet published CSV.
# The others are placeholders: fill them in with each Wave's own CSV later.
WAVES_CONFIG: List[WaveConfig] = [
    WaveConfig(
        code="SPX",
        name="S&P 500 Core Equity Wave",
        benchmark="SPY",
        holdings_src=(
            "https://docs.google.com/spreadsheets/d/e/"
            "2PACX-1vT7VpPdWSUSyZP9CVXZwTgqx7a7mMD2aQMRqSESqZgiagh8wSeEm3RAWHvLlWmJtLqYrqj7UVjQIpq9/"
            "pub?gid=711820877&single=true&output=csv"
        ),
    ),
    WaveConfig(
        code="USMKT",
        name="Total US Market Equity Wave",
        benchmark="VTI",
        holdings_src="",  # TODO: set a CSV URL/path for this Wave
    ),
    WaveConfig(
        code="LGRW",
        name="US Large Growth Equity Wave",
        benchmark="QQQ",
        holdings_src="",  # TODO
    ),
    WaveConfig(
        code="SCG",
        name="US Small Cap Growth Wave",
        benchmark="IWO",
        holdings_src="",  # TODO
    ),
    WaveConfig(
        code="SMID",
        name="US Small–Mid Growth Wave",
        benchmark="IJT",
        holdings_src="",  # TODO
    ),
    WaveConfig(
        code="AITECH",
        name="AI & Innovation Equity Wave",
        benchmark="QQQ",
        holdings_src="",  # TODO
    ),
    WaveConfig(
        code="ROBO",
        name="Automation & Robotics Equity Wave",
        benchmark="BOTZ",
        holdings_src="",  # TODO
    ),
    WaveConfig(
        code="ENERGYF",
        name="Future Power & Energy Wave",
        benchmark="ICLN",
        holdings_src="",  # TODO
    ),
    WaveConfig(
        code="EQINC",
        name="Global Equity Income Wave",
        benchmark="SCHD",
        holdings_src="",  # TODO
    ),
    WaveConfig(
        code="INTL",
        name="International + EM Equity Wave",
        benchmark="VEA",
        holdings_src="",  # TODO
    ),
]


# -------------------------------------------------------------------
# LOADING + NORMALIZATION
# -------------------------------------------------------------------


def _is_url(path_or_url: str) -> bool:
    parsed = urlparse(path_or_url)
    return parsed.scheme in ("http", "https")


def _read_csv(path_or_url: str) -> pd.DataFrame:
    if _is_url(path_or_url):
        resp = requests.get(path_or_url)
        resp.raise_for_status()
        return pd.read_csv(io.StringIO(resp.text))
    else:
        if not os.path.exists(path_or_url):
            raise FileNotFoundError(f"CSV file not found: {path_or_url}")
        return pd.read_csv(path_or_url)


def load_holdings(path_or_url: str) -> pd.DataFrame:
    """
    Load holdings from a CSV (URL or local path).

    Requirements:
    - Must contain a ticker column:
        'Ticker', 'Symbol', 'ticker', 'symbol', 'TICKER', or 'SYMBOL'
    - EITHER:
        - 'Weight' or 'Weight (%)' column, OR
        - 'Market Value' column to compute weights from.

    Returns:
        DataFrame with at least ['Ticker', 'Weight'].
        Weights are normalized to sum to 1.0.
    """
    if not path_or_url:
        raise ValueError("holdings_src is empty – set it in WAVES_CONFIG.")

    df = _read_csv(path_or_url)
    original_cols = list(df.columns)
    # Keep a copy of the unmodified columns for charts
    df.columns = [c.strip() for c in df.columns]

    # ---- Ticker detection ---- #
    ticker_col: Optional[str] = None
    for candidate in ("Ticker", "Symbol", "ticker", "symbol", "TICKER", "SYMBOL"):
        if candidate in df.columns:
            ticker_col = candidate
            break
    if ticker_col is None:
        raise ValueError(
            f"No Ticker/Symbol column found in holdings CSV. Columns seen: {original_cols}"
        )

    df.rename(columns={ticker_col: "Ticker"}, inplace=True)
    df["Ticker"] = df["Ticker"].astype(str).str.upper()

    # ---- Weight detection ---- #
    weight_col: Optional[str] = None
    for candidate in ("Weight", "Weight (%)", "Weight%", "Index Weight"):
        if candidate in df.columns:
            weight_col = candidate
            break

    if weight_col is not None:
        w = df[weight_col].astype(str).str.replace("%", "", regex=False)
        w = w.replace("", "0").astype(float)
        if w.max() > 1.5:  # treat as percent if larger than 150 bps
            w = w / 100.0
        df["Weight"] = w
    else:
        # Fall back to Market Value
        if "Market Value" not in df.columns:
            raise ValueError(
                "No Weight/Weight (%) or Market Value column found in holdings CSV."
            )
        mv = df["Market Value"].astype(str).str.replace(",", "", regex=False)
        mv = mv.replace("", "0").astype(float)
        total_mv = mv.sum()
        if total_mv <= 0:
            raise ValueError("Total Market Value <= 0; cannot derive weights.")
        df["Weight"] = mv / total_mv

    # Normalize weights
    total_w = df["Weight"].sum()
    if total_w <= 0:
        raise ValueError("Total portfolio weight <= 0 after normalization.")
    df["Weight"] = df["Weight"] / total_w

    return df


# -------------------------------------------------------------------
# PRICING + RETURNS
# -------------------------------------------------------------------


def _download_close_prices(tickers: List[str]) -> pd.DataFrame:
    """
    Download recent adjusted close prices for a list of tickers.
    Returns last 2 rows of Close prices (index=date, columns=tickers).
    """
    unique_tickers = sorted(set(tickers))
    if not unique_tickers:
        raise ValueError("No tickers provided to _download_close_prices().")

    data = yf.download(
        tickers=" ".join(unique_tickers),
        period="5d",
        interval="1d",
        auto_adjust=True,
        progress=False,
        threads=True,
    )

    if isinstance(data.columns, pd.MultiIndex):
        close = data["Close"]
    else:
        if "Close" not in data.columns:
            raise ValueError("No 'Close' column found in price data.")
        close = data[["Close"]]
        close.columns = [unique_tickers[0]]

    return close.tail(2)


def compute_wave_nav(wave: WaveConfig, holdings: pd.DataFrame) -> dict:
    """
    Compute a simple Wave NAV and 1-day return vs benchmark.

    Wave return = sum(weight_i * daily_return_i)
    NAV = default_notional * (1 + wave_return)
    Alpha = wave_return - benchmark_return
    """
    tickers = holdings["Ticker"].tolist()
    weights = holdings["Weight"].values

    prices = _download_close_prices(tickers)
    last = prices.iloc[-1]
    prev = prices.iloc[-2]
    rets = (last - prev) / prev
    rets.index = [t.upper() for t in prices.columns]

    # Align returns with holdings
    returns_vec = rets.reindex(holdings["Ticker"]).fillna(0.0).values
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
# PUBLIC ENTRYPOINT
# -------------------------------------------------------------------


def run_equity_waves() -> pd.DataFrame:
    """
    Run all Waves that have holdings_src configured and return a summary DataFrame.

    Columns:
        code, name, benchmark, nav, wave_return, benchmark_return, alpha, run_date, error?
    """
    results = []
    run_date = dt.date.today().isoformat()

    for wave in WAVES_CONFIG:
        if not wave.holdings_src:
            continue

        try:
            holdings = load_holdings(wave.holdings_src)
            stats = compute_wave_nav(wave, holdings)
            stats["run_date"] = run_date
            results.append(stats)
        except Exception as e:
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
    base_cols = ["code", "name", "benchmark", "nav",
                 "wave_return", "benchmark_return", "alpha", "run_date"]
    extra_cols = [c for c in df.columns if c not in base_cols]
    return df[base_cols + extra_cols].sort_values("code")


# -------------------------------------------------------------------
# CLI MODE
# -------------------------------------------------------------------


if __name__ == "__main__":
    print("WAVES Intelligence™ – Equity Waves Engine")
    print(f"Run date: {dt.date.today().isoformat()}\n")

    summary = run_equity_waves()
    if summary.empty:
        print("No waves processed – set holdings_src URLs/paths in WAVES_CONFIG.")
    else:
        pd.set_option("display.float_format", lambda x: f"{x:,.4f}")
        print(summary.to_string(index=False))