# ============================================================================
# INTRADAY-AWARE PRICE BOOK (24/7 — INSTITUTIONAL SAFE)
# ============================================================================

import pandas as pd
from datetime import datetime, timezone
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

# NOTE: expects these env vars to already exist (they usually do in your app)
# ALPACA_API_KEY
# ALPACA_SECRET_KEY

def inject_intraday_prices(price_book: pd.DataFrame) -> pd.DataFrame:
    """
    Overlays latest intraday prices on top of EOD price_book.

    Rules:
    - Never deletes history
    - Replaces ONLY the most recent row
    - Works after-hours & pre-market
    - Crypto unaffected (already live)
    - Never hard-fails
    """

    if price_book is None or price_book.empty:
        return price_book

    try:
        client = StockHistoricalDataClient()

        symbols = [
            c for c in price_book.columns
            if not c.upper().startswith("CRYPTO_")
        ]

        if not symbols:
            return price_book

        request = StockBarsRequest(
            symbol_or_symbols=symbols,
            timeframe=TimeFrame.Minute,
            limit=1,
            feed="sip",  # includes after-hours
        )

        bars = client.get_stock_bars(request).df

        if bars is None or bars.empty:
            return price_book

        latest_prices = (
            bars
            .reset_index()
            .groupby("symbol")["close"]
            .last()
            .to_dict()
        )

        # Clone to avoid mutating original
        pb = price_book.copy()

        latest_index = pb.index[-1]

        for symbol, price in latest_prices.items():
            if symbol in pb.columns and pd.notna(price):
                pb.loc[latest_index, symbol] = float(price)

        return pb

    except Exception:
        # Absolute rule: NEVER block the app
        return price_book