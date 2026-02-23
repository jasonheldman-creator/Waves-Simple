"""Market news stub - no live API calls."""


def ensure_schema(x):
    """Ensure a consistent return schema of {"data": <payload>, "meta": {}}.

    If x is already a dict with a "data" key it is returned unchanged.
    All other values are wrapped so every helper return is dict-accessible.
    """
    if isinstance(x, dict) and "data" in x:
        return x
    return {"data": x, "meta": {}}


def fetch_market_news(tickers=None, max_items=10):
    """Return dict with schema {"data": {"items": list, "timestamp": None}, "meta": {}}.

    Stub implementation returns an empty items list with no timestamp.
    """
    return ensure_schema({"items": [], "timestamp": None})
