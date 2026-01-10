"""
ticker_sources.py (helpers)

Data sources for the bottom ticker / quick spot quotes.

Goals:
- NEVER break the app if Yahoo/yfinance fails
- Normalize tickers (strip whitespace, remove leading '$', uppercase)
- Avoid known-bad Yahoo symbols that cause repeated 404 / quoteSummary errors
- Keep logic self-contained and safe if Streamlit or resilience modules are missing
"""

from __future__ import annotations

import os
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Set

import pandas as pd

logger = logging.getLogger(__name__)

# -----------------------------
# Optional Streamlit caching
# -----------------------------
try:
    import streamlit as st  # type: ignore
    STREAMLIT_AVAILABLE = True
except Exception:
    STREAMLIT_AVAILABLE = False

def conditional_cache(ttl: int = 300):
    """Use st.cache_data if available; otherwise no-op decorator."""
    def decorator(func):
        if STREAMLIT_AVAILABLE:
            try:
                return st.cache_data(ttl=ttl)(func)  # type: ignore
            except Exception:
                return func
        return func
    return decorator

# -----------------------------
# Optional resilience helpers
# -----------------------------
try:
    from .circuit_breaker import get_circuit_breaker  # type: ignore
    from .persistent_cache import get_persistent_cache  # type: ignore
    RESILIENCE_AVAILABLE = True
except Exception:
    RESILIENCE_AVAILABLE = False


# ============================================================================
# QUICK FIX: Normalize + block known-bad tickers
# ============================================================================

# These symbols are frequently invalid on Yahoo/yfinance in this app context and
# cause repeated HTTP 404 / quoteSummary errors. We do NOT want them spamming logs
# or triggering retry/circuit behavior.
BLOCKLIST_TICKERS: Set[str] = {
    "COMP-USD",
    "ALT-USD",
    "IMX-USD",
    "MNT-USD",
    "TAO-USD",
    "APT-USD",
}

def normalize_ticker(raw: Any) -> Optional[str]:
    """
    Normalize tickers coming from CSVs / universes / inputs.
    - Strips whitespace
    - Removes leading '$' (e.g., '$APT-USD' -> 'APT-USD')
    - Uppercases result
    - Returns None if empty/invalid or blocklisted
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

def _is_probably_crypto_pair(t: str) -> bool:
    # Most of your problematic ones are like APT-USD, COMP-USD etc.
    # We treat any *-USD as crypto pair in the bottom ticker context.
    return t.endswith("-USD")

def filter_and_normalize_tickers(tickers: List[Any]) -> List[str]:
    """Normalize tickers and drop duplicates / invalid / blocklisted."""
    seen: Set[str] = set()
    out: List[str] = []
    for raw in tickers:
        t = normalize_ticker(raw)
        if not t:
            continue
        if t in seen:
            continue
        seen.add(t)
        out.append(t)
    return out


# ============================================================================
# SECTION 1: Holdings Data Extraction (for bottom ticker list)
# ============================================================================

@conditional_cache(ttl=300)
def get_wave_holdings_tickers(
    max_tickers: int = 60,
    active_only: bool = True,
) -> List[str]:
    """
    Pull a safe ticker list for the bottom tape.

    IMPORTANT: This does NOT drive your canonical price cache.
    This is just the bottom ticker / spot-quote ribbon.
    """
    try:
        base_dir = os.path.dirname(os.path.dirname(__file__))

        # Canonical universe (as used by your repo)
        universe_path = os.path.join(base_dir, "universal_universe.csv")
        if not os.path.exists(universe_path):
            logger.warning(f"universal_universe.csv not found: {universe_path}")
            return _default_tickers()

        df = pd.read_csv(universe_path)

        # Keep active only if the column exists
        if active_only and "status" in df.columns:
            df = df[df["status"].astype(str).str.lower() == "active"]

        if "ticker" not in df.columns:
            logger.warning("universal_universe.csv missing 'ticker' column")
            return _default_tickers()

        tickers = df["ticker"].dropna().astype(str).tolist()
        tickers = filter_and_normalize_tickers(tickers)

        # Bottom ticker should NOT include crypto pairs (they are causing 404 spam)
        tickers = [t for t in tickers if not _is_probably_crypto_pair(t)]
        tickers = tickers[:max_tickers] if max_tickers else tickers

        if not tickers:
            return _default_tickers()

        logger.info(f"Loaded {len(tickers)} bottom-ticker symbols from universal_universe.csv")
        return tickers

    except Exception as e:
        logger.warning(f"get_wave_holdings_tickers failed: {e}")
        return _default_tickers()

def _default_tickers() -> List[str]:
    return ["SPY", "QQQ", "IWM", "AAPL", "MSFT", "NVDA", "GOOGL", "META", "AMZN", "TSLA"]


# ============================================================================
# SECTION 2: PRICE DATA (spot quote)
# ============================================================================

def _fetch_ticker_price_data_internal(ticker: str) -> Dict[str, Optional[float]]:
    """
    One-shot yfinance fetch (no retries in here).
    """
    import yfinance as yf  # type: ignore

    t = normalize_ticker(ticker)
    if t is None:
        return {"price": None, "change_pct": None, "success": False}

    # Hard stop on crypto pairs to prevent endless 404 spam in logs
    if _is_probably_crypto_pair(t):
        logger.warning(f"[TICKER BLOCKED - CRYPTO PAIR] {t}")
        return {"price": None, "change_pct": None, "success": False}

    try:
        stock = yf.Ticker(t)
        hist = stock.history(period="2d")

        if hist is None or hist.empty or len(hist) < 2:
            return {"price": None, "change_pct": None, "success": False}

        current = float(hist["Close"].iloc[-1])
        prev = float(hist["Close"].iloc[-2])
        change_pct = ((current - prev) / prev) * 100.0 if prev else 0.0

        return {"price": current, "change_pct": change_pct, "success": True}

    except Exception as e:
        # Do NOT raise; just fail quietly
        logger.debug(f"yfinance fetch failed for {t}: {e}")
        return {"price": None, "change_pct": None, "success": False}


@conditional_cache(ttl=600)
def get_ticker_price_data(ticker: str) -> Dict[str, Optional[float]]:
    """
    Spot quote with optional circuit breaker + persistent cache.
    Never throw; never loop.
    """
    t = normalize_ticker(ticker)
    if t is None:
        return {"price": None, "change_pct": None, "success": False}

    if _is_probably_crypto_pair(t):
        logger.warning(f"[TICKER BLOCKED - CRYPTO PAIR] {t}")
        return {"price": None, "change_pct": None, "success": False}

    # Try persistent cache if available
    if RESILIENCE_AVAILABLE:
        try:
            cache = get_persistent_cache()
            key = f"ticker_price:{t}"
            cached = cache.get(key)
            if cached is not None:
                return cached
        except Exception:
            pass

        try:
            cb = get_circuit_breaker("yfinance_ticker", failure_threshold=5, recovery_timeout=60)
            success, result, _err = cb.call(_fetch_ticker_price_data_internal, t)
            if success and isinstance(result, dict):
                try:
                    cache = get_persistent_cache()
                    cache.set(f"ticker_price:{t}", result, ttl=600)
                except Exception:
                    pass
                return result
            return {"price": None, "change_pct": None, "success": False}
        except Exception:
            return {"price": None, "change_pct": None, "success": False}

    # Non-resilient fallback
    return _fetch_ticker_price_data_internal(t)


# ============================================================================
# SECTION 3: EVENTS CACHE (optional)
# ============================================================================

def load_events_cache() -> Dict[str, Any]:
    try:
        base_dir = os.path.dirname(os.path.dirname(__file__))
        cache_path = os.path.join(base_dir, "data", "events_cache.json")
        if os.path.exists(cache_path):
            with open(cache_path, "r") as f:
                return json.load(f)
        return {}
    except Exception:
        return {}

def save_events_cache(cache_data: Dict[str, Any]) -> bool:
    try:
        base_dir = os.path.dirname(os.path.dirname(__file__))
        cache_path = os.path.join(base_dir, "data", "events_cache.json")
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump(cache_data, f, indent=2)
        return True
    except Exception:
        return False


# ============================================================================
# SECTION 4: HEALTH / TEST
# ============================================================================

def test_ticker_fetch(ticker: str = "AAPL") -> Dict[str, Any]:
    import time
    start = time.time()
    data = get_ticker_price_data(ticker)
    latency = (time.time() - start) * 1000.0
    return {
        "ticker": ticker,
        "timestamp": datetime.now().isoformat(),
        "latency_ms": round(latency, 2),
        "success": bool(data.get("success", False)),
        "data": data,
    }