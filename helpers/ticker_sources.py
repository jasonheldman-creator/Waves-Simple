"""
V3/V4 COMPAT: Bottom Ticker (Institutional Rail) - Data Sources

Goals:
- Keep ONE canonical BLOCKLIST_TICKERS (no duplicates)
- Provide normalize_ticker() used across the app
- Provide compatibility functions that other modules likely import:
  - filter_and_normalize_tickers
  - get_wave_holdings_tickers
  - get_ticker_price_data
  - get_earnings_date
  - get_fed_indicators
  - get_waves_status
  - load_events_cache / save_events_cache / update_cache_with_current_data
  - get_ticker_health_status / test_ticker_fetch
- Use Streamlit cache if available, otherwise no-op.
- Use resilience helpers (circuit_breaker/persistent_cache) if available, otherwise degrade gracefully.

NOTE: This module can work without Streamlit, but will use Streamlit caching if available.
"""

import os
import json
import logging
import pandas as pd
from datetime import datetime
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
# TICKER NORMALIZATION
# ====================================================================

def normalize_ticker(raw: Any) -> Optional[str]:
    """
    Normalize tickers coming from CSVs / universes / inputs.

    - Strips whitespace
    - Removes leading '$'
    - Uppercases
    - Drops blocklisted or invalid tickers (returns None)
    """
    if raw is None:
        return None

    try:
        t = str(raw).strip().upper()
    except Exception:
        return None

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


def filter_and_normalize_tickers(tickers: List[Any]) -> List[str]:
    """
    Compatibility helper: normalize a list of tickers and drop blocklisted/invalid/dupes.
    """
    seen: Set[str] = set()
    out: List[str] = []
    for raw in tickers or []:
        t = normalize_ticker(raw)
        if not t:
            continue
        if t in seen:
            continue
        seen.add(t)
        out.append(t)
    return out


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

def _default_tickers() -> List[str]:
    return ["AAPL", "MSFT", "NVDA", "GOOGL", "META", "TSLA", "JPM", "V", "WMT", "JNJ"]


@conditional_cache(ttl=300)
def get_wave_holdings_tickers(
    max_tickers: int = 60,
    top_n_per_wave: int = 5,   # kept for compat, not strictly used here
    active_waves_only: bool = True
) -> List[str]:
    """
    Extract tickers from universal_universe.csv (preferred).
    Falls back to defaults if file missing/unreadable.
    """
    try:
        base_dir = os.path.dirname(os.path.dirname(__file__))
        universe_path = os.path.join(base_dir, "universal_universe.csv")

        if not os.path.exists(universe_path):
            logger.warning(f"universal_universe.csv not found at {universe_path}")
            return _default_tickers()

        df = pd.read_csv(universe_path)

        if "status" in df.columns:
            df = df[df["status"] == "active"]

        if "ticker" not in df.columns:
            logger.warning("universal_universe.csv missing 'ticker' column")
            return _default_tickers()

        tickers = df["ticker"].dropna().tolist()
        tickers = filter_and_normalize_tickers(tickers)

        if not tickers:
            return _default_tickers()

        return tickers[:max_tickers] if max_tickers else tickers

    except Exception as e:
        logger.error(f"get_wave_holdings_tickers failed: {e}")
        return _default_tickers()


# ====================================================================
# SECTION 2: PRICE DATA
# ====================================================================

def _fetch_ticker_price_data_internal(ticker: str) -> Dict[str, Optional[float]]:
    """
    One-shot yfinance fetch (no retries in here).
    """
    import yfinance as yf

    t = normalize_ticker(ticker)
    if t is None:
        return {"price": None, "change_pct": None, "success": False}

    try:
        stock = yf.Ticker(t)
        hist = stock.history(period="2d")

        if hist is None or hist.empty or len(hist) < 2:
            return {"price": None, "change_pct": None, "success": False}

        current = float(hist["Close"].iloc[-1])
        prev = float(hist["Close"].iloc[-2])
        change = ((current - prev) / prev) * 100 if prev else 0.0

        return {"price": current, "change_pct": change, "success": True}
    except Exception:
        logger.debug(f"yfinance fetch failed for {t}: {e}")
        return {"price": None, "change_pct": None, "success": False}


@conditional_cache(ttl=600)
def get_ticker_price_data(ticker: str) -> Dict[str, Optional[float]]:
    """
    Cached price lookup with optional persistent cache + circuit breaker.
    """
    t = normalize_ticker(ticker)
    if t is None:
        return {"price": None, "change_pct": None, "success": False}

    failure = {"price": None, "change_pct": None, "success": False}

    # Persistent cache first
    if RESILIENCE_AVAILABLE:
        try:
            cache = get_persistent_cache()
            key = f"ticker_price:{t}"
            cached = cache.get(key)
            if cached is not None:
                return cached
        except Exception:
            pass

        # Circuit breaker guarded call
        try:
            cb = get_circuit_breaker("yfinance_ticker", failure_threshold=5, recovery_timeout=60)
            success, result, _err = cb.call(_fetch_ticker_price_data_internal, t)

            if success and result:
                try:
                    cache = get_persistent_cache()
                    cache.set(f"ticker_price:{t}", result, ttl=600)
                except Exception:
                    pass
                return result

            return failure
        except Exception:
            return failure

    # No resilience available
    return _fetch_ticker_price_data_internal(t)


# ====================================================================
# SECTION 3: EARNINGS
# ====================================================================

@conditional_cache(ttl=3600)
def get_earnings_date(ticker: str) -> Optional[str]:
    """
    Best-effort earnings date via yfinance.calendar
    """
    import yfinance as yf

    t = normalize_ticker(ticker)
    if t is None:
        return None

    try:
        cal = yf.Ticker(t).calendar
        if cal is not None and not cal.empty and "Earnings Date" in cal.index:
            d = cal.loc["Earnings Date"]
            if hasattr(d, "strftime"):
                return d.strftime("%Y-%m-%d")
            if isinstance(d, str):
                return d
        return None
    except Exception:
        return None


# ====================================================================
# SECTION 4: FED / MACRO
# ====================================================================

@conditional_cache(ttl=86400)
def get_fed_indicators() -> Dict[str, Optional[str]]:
    """
    Hardcoded (safe) macro placeholders — avoids paid APIs.
    """
    try:
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
    except Exception:
        return {"fed_funds_rate": "N/A", "next_fomc_date": None, "cpi_latest": "N/A", "jobs_latest": "N/A"}


# ====================================================================
# SECTION 5: WAVES INTERNAL STATUS
# ====================================================================

def get_waves_status() -> Dict[str, str]:
    """
    Minimal internal status for UI.
    """
    try:
        current_time = datetime.now().strftime("%H:%M:%S")
        if STREAMLIT_AVAILABLE:
            waves_loaded = "ACTIVE" if st.session_state.get("wave_universe") else "LOADING"
        else:
            waves_loaded = "N/A"
        return {"system_status": "ONLINE", "last_update": current_time, "waves_status": waves_loaded}
    except Exception:
        return {"system_status": "ONLINE", "last_update": "N/A", "waves_status": "N/A"}


# ====================================================================
# SECTION 6: CACHE MANAGEMENT (events_cache.json)
# ====================================================================

def load_events_cache() -> Dict:
    try:
        base_dir = os.path.dirname(os.path.dirname(__file__))
        cache_path = os.path.join(base_dir, "data", "events_cache.json")
        if os.path.exists(cache_path):
            with open(cache_path, "r") as f:
                return json.load(f)
        return {}
    except Exception:
        return {}


def save_events_cache(cache_data: Dict) -> bool:
    try:
        base_dir = os.path.dirname(os.path.dirname(__file__))
        cache_path = os.path.join(base_dir, "data", "events_cache.json")
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump(cache_data, f, indent=2)
        return True
    except Exception:
        return False


def update_cache_with_current_data() -> None:
    try:
        fed_data = get_fed_indicators()
        waves_status = get_waves_status()
        cache_data = {
            "last_updated": datetime.now().isoformat(),
            "fed_indicators": fed_data,
            "waves_status": waves_status,
        }
        save_events_cache(cache_data)
    except Exception:
        pass


# ====================================================================
# SECTION 7: DATA HEALTH TRACKING
# ====================================================================

def get_ticker_health_status() -> Dict[str, Any]:
    """
    Compatibility health function used by some diagnostics panels.
    """
    health: Dict[str, Any] = {
        "timestamp": datetime.now().isoformat(),
        "resilience_available": RESILIENCE_AVAILABLE,
        "circuit_breakers": {},
        "cache_stats": {},
        "overall_status": "unknown",
    }

    if not RESILIENCE_AVAILABLE:
        health["overall_status"] = "healthy"
        return health

    # Circuit breaker states (best effort)
    try:
        from .circuit_breaker import get_all_circuit_states
        states = get_all_circuit_states()
        health["circuit_breakers"] = states

        open_count = 0
        for _name, state in (states or {}).items():
            if isinstance(state, dict) and state.get("state") == "open":
                open_count += 1

        health["overall_status"] = "degraded" if open_count > 0 else "healthy"
    except Exception as e:
        health["circuit_breakers"] = {"error": str(e)}
        health["overall_status"] = "unknown"

    # Persistent cache stats (best effort)
    try:
        cache = get_persistent_cache()
        if hasattr(cache, "get_stats"):
            health["cache_stats"] = cache.get_stats()
    except Exception as e:
        health["cache_stats"] = {"error": str(e)}

    return health


def test_ticker_fetch(ticker: str = "AAPL") -> Dict[str, Any]:
    import time

    result = {
        "ticker": ticker,
        "timestamp": datetime.now().isoformat(),
        "success": False,
        "latency_ms": 0,
        "data": None,
        "error": None,
    }

    try:
        start = time.time()
        data = get_ticker_price_data(ticker)
        latency = (time.time() - start) * 1000
        result["latency_ms"] = round(latency, 2)
        result["data"] = data
        result["success"] = bool(data.get("success", False))
    except Exception as e:
        result["error"] = str(e)
        result["success"] = False

    return result