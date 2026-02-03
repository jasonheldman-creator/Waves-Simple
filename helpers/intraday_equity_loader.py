import os
from datetime import datetime, timezone
from typing import Dict, Optional, List

import pandas as pd


# ============================================================================
# INTRADAY EQUITY PRICE LOADER (ALPACA, SAFE)
# ============================================================================

def _alpaca_client():
    """
    Lazily construct Alpaca client.
    Returns None if credentials are missing.
    """
    key = os.getenv("ALPACA_API_KEY_ID")
    secret = os.getenv("ALPACA_API_SECRET_KEY")

    if not key or not secret:
        return None

    try:
        from alpaca.data.historical import StockHistoricalDataClient
        return StockHistoricalDataClient(key, secret)
    except Exception:
        return None


def fetch_intraday_equity_prices(
    tickers: List[str],
) -> Dict[str, pd.Series]:
    """
    Fetch latest intraday prices for equities.

    RULES:
    - Uses most recent two bars per symbol
    - Normalized UTC timestamps
    - Never raises
    """

    client = _alpaca_client()
    if client is None:
        return {}

    try:
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame

        now = datetime.now(timezone.utc)

        request = StockBarsRequest(
            symbol_or_symbols=tickers,
            timeframe=TimeFrame.Minute,
            start=now.replace(hour=0, minute=0, second=0),
            end=now,
            limit=2,
        )

        bars = client.get_stock_bars(request)

        out: Dict[str, pd.Series] = {}

        for symbol, df in bars.data.items():
            if df is None or df.empty:
                continue

            closes = df["close"].dropna()
            if len(closes) < 1:
                continue

            closes.index = pd.to_datetime(closes.index, utc=True)
            out[symbol] = closes.sort_index()

        return out

    except Exception:
        return {}


# ============================================================================
# CACHE MERGE (NON-DESTRUCTIVE)
# ============================================================================

def merge_intraday_into_price_book(
    price_book: pd.DataFrame,
    intraday_prices: Dict[str, pd.Series],
) -> pd.DataFrame:
    """
    Merge intraday prices into existing price book.

    RULES:
    - Never deletes historical data
    - Appends only newer timestamps
    - Column-safe
    """

    if price_book is None or price_book.empty:
        return price_book

    updated = price_book.copy()

    for symbol, series in intraday_prices.items():
        if series is None or series.empty:
            continue

        if symbol not in updated.columns:
            continue

        last_ts = updated[symbol].dropna().index.max()

        for ts, price in series.items():
            if last_ts is None or ts > last_ts:
                updated.loc[ts, symbol] = float(price)

    return updated.sort_index()


# ============================================================================
# PUBLIC ENTRY POINT (USED BY WORKFLOW)
# ============================================================================

def update_intraday_equity_cache(
    price_book: pd.DataFrame,
) -> pd.DataFrame:
    """
    Main entry point.

    - Detects equity symbols
    - Pulls intraday prices
    - Merges safely
    """

    if price_book is None or price_book.empty:
        return price_book

    # Heuristic: equities are uppercase tickers (SPY, AAPL, etc.)
    equity_symbols = [
        c for c in price_book.columns
        if c.isupper() and len(c) <= 5
    ]

    if not equity_symbols:
        return price_book

    intraday = fetch_intraday_equity_prices(equity_symbols)
    if not intraday:
        return price_book

    return merge_intraday_into_price_book(price_book, intraday)