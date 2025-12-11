"""
waves_engine.py — WAVES Intelligence™ Simple Engine (12-Wave Edition)

Purpose
-------
Lightweight engine used by the Streamlit console to:

• Load wave definitions from wave_weights.csv
• Discover available Waves
• Build live snapshots with current prices
• Optionally load history from Full_Wave_History.csv (if present)

Expected CSVs in repo root:
    wave_weights.csv
        wave,ticker,weight

    Full_Wave_History.csv  (optional; built by history engine)
        Date,Wave,Position,Weight,Ticker,Price,MarketValue,...
"""

import os
import datetime as dt
from typing import List, Dict, Optional, Tuple

import pandas as pd
import numpy as np

try:
    import yfinance as yf
except ImportError:  # Safety: console should have this via requirements.txt
    yf = None


# ---------------------------------------------------------------------------
# Core loaders
# ---------------------------------------------------------------------------

def _load_wave_weights(csv_path: str = "wave_weights.csv") -> pd.DataFrame:
    """
    Load and clean wave_weights.csv into a standard DataFrame.

    Expected columns in the CSV (any case is OK):
        wave, ticker, weight
    """
    if not os.path.exists(csv_path):
        raise ValueError(f"{csv_path} not found in repo root")

    df = pd.read_csv(csv_path)

    # Normalise column names to lowercase
    df.columns = [c.strip().lower() for c in df.columns]

    # Map any near-matches onto the canonical names
    rename = {}
    colset = set(df.columns)

    # If user used different labels, rescue them here
    for c in list(df.columns):
        cl = c.lower()
        if "wave" in cl and "wave" not in colset:
            rename[c] = "wave"
        if "ticker" in cl and "ticker" not in colset:
            rename[c] = "ticker"
        if "weight" in cl and "weight" not in colset:
            rename[c] = "weight"

    if rename:
        df = df.rename(columns=rename)
        df.columns = [c.strip().lower() for c in df.columns]
        colset = set(df.columns)

    expected_cols = {"wave", "ticker", "weight"}
    missing = expected_cols - colset
    if missing:
        raise ValueError(f"wave_weights.csv missing required columns: {missing}")

    # Clean types
    df["wave"] = df["wave"].astype(str).str.strip()
    df["ticker"] = df["ticker"].astype(str).str.strip().str.upper()
    df["weight"] = pd.to_numeric(df["weight"], errors="coerce")

    # Validate weights
    if df["weight"].isna().any():
        bad_rows = df[df["weight"].isna()]
        raise ValueError(
            "Some rows in wave_weights.csv have non-numeric weights.\n"
            f"Offending rows:\n{bad_rows}"
        )

    # Normalise weights within each wave so they sum to 1.0
    totals = df.groupby("wave")["weight"].transform("sum")
    # Avoid divide-by-zero in case of weird input
    totals = totals.replace(0, np.nan)
    df["weight"] = df["weight"] / totals

    return df


# ---------------------------------------------------------------------------
# Public API used by app.py
# ---------------------------------------------------------------------------

def get_available_waves() -> List[str]:
    """
    Return a sorted list of available wave names from wave_weights.csv.
    """
    df = _load_wave_weights()
    waves = sorted(df["wave"].unique().tolist())
    return waves


def get_wave_holdings(wave: str) -> pd.DataFrame:
    """
    Return holdings for a single Wave from wave_weights.csv.

    Columns:
        wave, ticker, weight
    """
    df = _load_wave_weights()
    wave = wave.strip()
    sub = df[df["wave"] == wave].copy()
    if sub.empty:
        raise ValueError(f"No holdings found for wave '{wave}' in wave_weights.csv")
    return sub.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Live price & snapshot helpers
# ---------------------------------------------------------------------------

def _fetch_prices(tickers: List[str]) -> pd.Series:
    """
    Fetch latest prices for a list of tickers using yfinance.

    Returns a pandas Series indexed by ticker.
    """
    if yf is None:
        # Fallback: everything = 1.0 so app at least renders
        return pd.Series(1.0, index=tickers, name="price")

    tickers = [t.strip().upper() for t in tickers]
    data = yf.download(
        tickers=" ".join(tickers),
        period="2d",
        interval="1d",
        progress=False,
        group_by="ticker",
    )

    prices = {}
    for t in tickers:
        try:
            # yfinance shape is annoying; handle both single & multi ticker
            if isinstance(data.columns, pd.MultiIndex):
                last_close = data[t]["Close"].dropna().iloc[-1]
            else:
                last_close = data["Close"].dropna().iloc[-1]
            prices[t] = float(last_close)
        except Exception:
            # If anything fails, just use 1.0 so UI doesn't crash
            prices[t] = 1.0

    return pd.Series(prices, name="price")


def get_wave_snapshot(wave: str) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Build a live snapshot for a single Wave.

    Returns:
        positions_df, summary_dict

    positions_df columns:
        wave, ticker, weight, price, market_value, allocation

    summary_dict keys:
        "as_of" (ISO date string),
        "wave",
        "nav"   (sum of market_value, assuming starting capital = 1.0),
    """
    holdings = get_wave_holdings(wave)
    tickers = holdings["ticker"].tolist()

    prices = _fetch_prices(tickers)
    holdings = holdings.merge(
        prices.rename("price"),
        left_on="ticker",
        right_index=True,
        how="left",
    )

    # price fallback already handled in _fetch_prices
    holdings["market_value"] = holdings["weight"] * holdings["price"]
    # For now, allocation is just equal to weight
    holdings["allocation"] = holdings["weight"]

    nav = float(holdings["market_value"].sum())

    summary = {
        "as_of": dt.date.today().isoformat(),
        "wave": wave,
        "nav": nav,
    }

    return holdings, summary


def get_all_wave_snapshots() -> Dict[str, Dict[str, object]]:
    """
    Convenience helper: build snapshots for all Waves.

    Returns:
        {
          "S&P 500 Wave": {
              "positions": <DataFrame>,
              "summary": {"as_of": ..., "wave": ..., "nav": ...}
          },
          ...
        }
    """
    result: Dict[str, Dict[str, object]] = {}
    for w in get_available_waves():
        try:
            positions, summary = get_wave_snapshot(w)
            result[w] = {"positions": positions, "summary": summary}
        except Exception as e:
            # Don't let one bad wave kill the app; report minimal info
            result[w] = {
                "positions": pd.DataFrame(),
                "summary": {
                    "as_of": dt.date.today().isoformat(),
                    "wave": w,
                    "nav": float("nan"),
                    "error": str(e),
                },
            }
    return result


# ---------------------------------------------------------------------------
# Optional history loader (for Full_Wave_History.csv)
# ---------------------------------------------------------------------------

def load_wave_history(
    wave: str,
    csv_path: str = "Full_Wave_History.csv",
) -> pd.DataFrame:
    """
    Load historical NAV/positions for a single Wave from Full_Wave_History.csv,
    if that file exists. This is used by the console for 60D / 1Y charts.

    Returns an empty DataFrame if the file is missing.
    """
    if not os.path.exists(csv_path):
        # History not built yet; caller can handle empty DF
        return pd.DataFrame()

    df = pd.read_csv(csv_path)

    # Normalise column names
    df.columns = [c.strip() for c in df.columns]

    if "Wave" not in df.columns:
        # Unexpected format; return empty rather than crash the UI
        return pd.DataFrame()

    sub = df[df["Wave"].astype(str).str.strip() == wave.strip()].copy()
    return sub.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Simple self-test (optional; not used by Streamlit)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Available Waves:")
    print(get_available_waves())

    demo_wave = get_available_waves()[0]
    print(f"\nDemo snapshot for: {demo_wave}")
    positions_df, summary = get_wave_snapshot(demo_wave)
    print(summary)
    print(positions_df.head())