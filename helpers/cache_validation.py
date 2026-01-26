import os
import pandas as pd
from datetime import datetime, timedelta


def validate_trading_day_freshness(cache_path, max_market_gap=5):
    if not os.path.exists(cache_path):
        return False, "Cache file missing"

    cache = pd.read_parquet(cache_path)

    latest_date = cache.index.max()
    today = datetime.now().normalize()

    delta = today - latest_date
    if delta.days > max_market_gap:
        return False, f"Cache is stale. Latest date is {latest_date}, {delta.days} days old."

    return True, "Cache is fresh"


def validate_required_tickers(cache, required_tickers):
    missing = [ticker for ticker in required_tickers if ticker not in cache.columns]
    if missing:
        return False, f"Missing tickers: {missing}"
    return True, "All required tickers are present."