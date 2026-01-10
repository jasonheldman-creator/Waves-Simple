"""
V3 ADD-ON: Bottom Ticker (Institutional Rail) - Data Sources

Handles all data fetching for the bottom ticker with exception handling and fallbacks.
Enhanced with circuit breaker and persistent cache for resilience.

NOTE: This module can work without Streamlit, but will use Streamlit caching if available.
"""

import os
import json
import logging
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Set, Any

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------
# Optional Streamlit support
# --------------------------------------------------------------------
try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False

# --------------------------------------------------------------------
# Optional resilience infrastructure
# --------------------------------------------------------------------
try:
    from .circuit_breaker import get_circuit_breaker
    from .persistent_cache import get_persistent_cache
    RESILIENCE_AVAILABLE = True
except ImportError:
    RESILIENCE_AVAILABLE = False


# ====================================================================
# BLOCKLIST — SINGLE SOURCE OF TRUTH (KEEP THIS ONE ONLY)
# ====================================================================

BLOCKLIST_TICKERS: Set[str] = {
    "COMP-USD",
    "ALT-USD",
    "IMX-USD",
    "MNT-USD",
    "TAO-USD",
    "APT-USD",
}


# ====================================================================
# TICKER NORMALIZATION (USED EVERYWHERE)
# ====================================================================

def normalize_ticker(raw: Any) -> Optional[str]:
    """
    Normalize tickers coming from CSVs / universes / inputs.

    - Strips whitespace
    - Removes leading '$'
    - Uppercases
    - Drops blocklisted or invalid tickers
    """
    if raw is None:
        return None

    t = str(raw).strip().upper()
    if not t:
        return None

    if t.startswith("$"):
        t = t[1:].strip()

    if not t:
        return None

    if t in BLOCKLIST_TICKERS:
        logger.warning(f"[TICKER BLOCKED] {t}")
        return None

    return t


# ====================================================================
# CONDITIONAL CACHE DECORATOR
# ====================================================================

def conditional_cache(ttl=300):
    def decorator(func):
        if STREAMLIT_AVAILABLE:
            return st.cache_data(ttl=ttl)(func)
        return func
    return decorator


# ====================================================================
# SECTION 1: HOLDINGS / UNIVERSE TICKERS
# ====================================================================

@conditional_cache(ttl=300)
def get_wave_holdings_tickers(
    max_tickers: int = 60,
    active_only: bool = True
) -> List[str]:
    """
    Load tickers from universal_universe.csv (single source of truth)
    """
    try:
        base_dir = os.path.dirname(os.path.dirname(__file__))
        universe_path = os.path.join(base_dir, "universal_universe.csv")

        if not os.path.exists(universe_path):
            return default_tickers()

        df = pd.read_csv(universe_path)

        if "status" in df.columns:
            df = df[df["status"] == "active"]

        if "ticker" not in df.columns:
            return default_tickers()

        tickers = [normalize_ticker(t) for t in df["ticker"].tolist()]
        tickers = [t for t in tickers if t]

        return tickers[:max_tickers] if max_tickers else tickers

    except Exception as e:
        logger.error(f"Universe load failure: {e}")
        return default_tickers()


def default_tickers() -> List[str]:
    return [
        "AAPL", "MSFT", "NVDA", "GOOGL", "META",
        "TSLA", "JPM", "V", "WMT", "JNJ"
    ]


# ====================================================================
# SECTION 2: PRICE DATA
# ====================================================================

def _fetch_price_internal(ticker: str) -> Dict[str, Optional[float]]:
    import yfinance as yf

    ticker = normalize_ticker(ticker)
    if ticker is None:
        return {"price": None, "change_pct": None, "success": False}

    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="2d")

        if hist.empty or len(hist) < 2:
            return {"price": None, "change_pct": None, "success": False}

        current = float(hist["Close"].iloc[-1])
        prev = float(hist["Close"].iloc[-2])
        change = ((current - prev) / prev) * 100 if prev else 0.0

        return {"price": current, "change_pct": change, "success": True}

    except Exception:
        return {"price": None, "change_pct": None, "success": False}


@conditional_cache(ttl=600)
def get_ticker_price_data(ticker: str) -> Dict[str, Optional[float]]:
    ticker = normalize_ticker(ticker)
    if ticker is None:
        return {"price": None, "change_pct": None, "success": False}

    if RESILIENCE_AVAILABLE:
        try:
            cache = get_persistent_cache()
            key = f"price:{ticker}"
            cached = cache.get(key)
            if cached:
                return cached
        except Exception:
            pass

        try:
            cb = get_circuit_breaker("yfinance", failure_threshold=5, recovery_timeout=60)
            success, result, _ = cb.call(_fetch_price_internal, ticker)

            if success and result:
                try:
                    cache.set(key, result, ttl=600)
                except Exception:
                    pass
                return result

            return {"price": None, "change_pct": None, "success": False}

        except Exception:
            return {"price": None, "change_pct": None, "success": False}

    return _fetch_price_internal(ticker)


# ====================================================================
# SECTION 3: EARNINGS
# ====================================================================

@conditional_cache(ttl=3600)
def get_earnings_date(ticker: str) -> Optional[str]:
    import yfinance as yf

    ticker = normalize_ticker(ticker)
    if ticker is None:
        return None

    try:
        cal = yf.Ticker(ticker).calendar
        if cal is not None and "Earnings Date" in cal.index:
            d = cal.loc["Earnings Date"]
            if hasattr(d, "strftime"):
                return d.strftime("%Y-%m-%d")
        return None
    except Exception:
        return None


# ====================================================================
# SECTION 4: FED / MACRO
# ====================================================================

@conditional_cache(ttl=86400)
def get_fed_indicators() -> Dict[str, Optional[str]]:
    now = datetime.now()

    fomc_dates = [
        datetime(2025, 1, 29),
        datetime(2025, 3, 19),
        datetime(2025, 5, 7),
        datetime(2025, 6, 18),
        datetime(2025, 7, 30),
        datetime(2025, 9, 17),
        datetime(2025, 10, 29),
        datetime(2025, 12, 10),
    ]

    next_fomc = next((d for d in fomc_dates if d > now), None)

    return {
        "fed_funds_rate": "4.25–4.50%",
        "next_fomc_date": next_fomc.strftime("%Y-%m-%d") if next_fomc else None,
        "cpi_latest": "Dec 2024",
        "jobs_latest": "Dec 2024",
    }


# ====================================================================
# SECTION 5: SYSTEM HEALTH
# ====================================================================

def test_ticker_fetch(ticker: str = "AAPL") -> Dict[str, Any]:
    import time

    start = time.time()
    data = get_ticker_price_data(ticker)
    latency = (time.time() - start) * 1000

    return {
        "ticker": ticker,
        "success": bool(data.get("success")),
        "latency_ms": round(latency, 2),
        "data": data,
        "timestamp": datetime.now().isoformat(),
    }