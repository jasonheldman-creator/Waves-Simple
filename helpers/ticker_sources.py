"""
ticker_sources.py (Helpers)

Bottom Ticker / Institutional Rail data sources + lightweight resilience.

Goals:
- Never crash the app due to a bad ticker or transient yfinance failure
- Normalize tickers (trim, uppercase, strip leading '$')
- Blocklist known-bad / noisy tickers so they don't spam logs or break fetches
- Provide optional Streamlit caching when available
"""

from __future__ import annotations

import os
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Set

import pandas as pd

logger = logging.getLogger(__name__)

# ---- Optional Streamlit cache ----
try:
    import streamlit as st  # type: ignore
    STREAMLIT_AVAILABLE = True
except Exception:
    STREAMLIT_AVAILABLE = False

# ---- Optional resilience helpers (best-effort; must never fail import) ----
try:
    from .circuit_breaker import get_circuit_breaker  # type: ignore
    from .persistent_cache import get_persistent_cache  # type: ignore
    RESILIENCE_AVAILABLE = True
except Exception:
    RESILIENCE_AVAILABLE = False


# ============================================================================
# QUICK FIX: Normalize + block known-bad tickers
# ============================================================================

# Keep this EXACT name — other modules may reference it.
BLOCKLIST_TICKERS: Set[str] = {
    # These have been repeatedly failing / causing yfinance noise
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


def filter_and_normalize_tickers(tickers: List[Any]) -> List[str]:
    """Normalize tickers, drop invalids, de-dupe while preserving order."""
    seen: Set[str] = set()
    out: List[str] = []
    for x in tickers:
        t = normalize_ticker(x)
        if not t:
            continue
        if t in seen:
            continue
        seen.add(t)
        out.append(t)
    return out


def conditional_cache(ttl: int = 300):
    """
    Decorator that uses Streamlit caching if available, otherwise no-op.
    """
    def decorator(func):
        if STREAMLIT_AVAILABLE:
            return st.cache_data(ttl=ttl)(func)  # type: ignore
        return func
    return decorator


# ============================================================================
# SECTION 1: Holdings Data Extraction
# ============================================================================

def _default_tickers() -> List[str]:
    return ["AAPL", "MSFT", "NVDA", "GOOGL", "META", "TSLA", "JPM", "V", "WMT", "JNJ"]


@conditional_cache(ttl=300)
def get_wave_holdings_tickers(
    max_tickers: int = 60,
    active_waves_only: bool = True,
) -> List[str]:
    """
    Extract tickers from universal_universe.csv (preferred) with safe fallbacks.

    This must NEVER raise — it feeds UI elements and health checks.
    """
    try:
        base_dir = os.path.dirname(os.path.dirname(__file__))
        universe_path = os.path.join(base_dir, "universal_universe.csv")

        active_wave_ids: Set[str] = set()
        if active_waves_only:
            try:
                registry_path = os.path.join(base_dir, "data", "wave_registry.csv")
                if os.path.exists(registry_path):
                    reg = pd.read_csv(registry_path)
                    if "active" in reg.columns and "wave_id" in reg.columns:
                        active_wave_ids = set(
                            reg[reg["active"] == True]["wave_id"].astype(str).str.strip().tolist()
                        )
            except Exception as e:
                logger.warning(f"Active-wave filtering disabled (registry read failed): {e}")
                active_waves_only = False

        if os.path.exists(universe_path):
            df = pd.read_csv(universe_path)

            # If a 'status' column exists, keep active tickers only
            if "status" in df.columns:
                df = df[df["status"].astype(str).str.lower() == "active"]

            if "ticker" not in df.columns:
                return _default_tickers()

            # Prefer tickers that belong to waves (if present)
            tickers: List[Any]
            if "index_membership" in df.columns:
                wave_rows = df["index_membership"].astype(str).str.contains("WAVE_", case=False, na=False)

                if active_waves_only and active_wave_ids:
                    # Keep only rows that mention one of the active wave ids
                    def row_is_active(m: Any) -> bool:
                        if pd.isna(m):
                            return False
                        s = str(m).upper()
                        # Rough match; we just want to exclude junk
                        for wid in active_wave_ids:
                            if wid and wid.upper() in s:
                                return True
                        return False

                    wave_rows = wave_rows & df["index_membership"].apply(row_is_active)

                tickers = df.loc[wave_rows, "ticker"].tolist()
                tickers = filter_and_normalize_tickers(tickers)
                if tickers:
                    return tickers[:max_tickers] if max_tickers else tickers

            # Otherwise: all tickers
            tickers = filter_and_normalize_tickers(df["ticker"].tolist())
            if tickers:
                return tickers[:max_tickers] if max_tickers else tickers

        return _default_tickers()
    except Exception as e:
        logger.warning(f"get_wave_holdings_tickers failed, using defaults: {e}")
        return _default_tickers()


# ============================================================================
# SECTION 2: Market Price Data
# ============================================================================

def _fetch_ticker_price_data_internal(ticker: str) -> Dict[str, Optional[float]]:
    """
    One-shot yfinance fetch (no retries here).
    Must NEVER raise.
    """
    try:
        import yfinance as yf  # type: ignore

        t = normalize_ticker(ticker)
        if t is None:
            return {"price": None, "change_pct": None, "success": False}

        stock = yf.Ticker(t)

        # Prefer 2d history (more reliable than .info)
        hist = stock.history(period="2d")
        if hist is None or hist.empty or len(hist) < 2:
            return {"price": None, "change_pct": None, "success": False}

        current = float(hist["Close"].iloc[-1])
        prev = float(hist["Close"].iloc[-2])
        change_pct = ((current - prev) / prev) * 100.0 if prev else 0.0

        return {"price": current, "change_pct": change_pct, "success": True}
    except Exception:
        return {"price": None, "change_pct": None, "success": False}


@conditional_cache(ttl=600)
def get_ticker_price_data(ticker: str) -> Dict[str, Optional[float]]:
    """
    Get current price + daily % change with best-effort resilience.
    Must NEVER raise.
    """
    t = normalize_ticker(ticker)
    if t is None:
        return {"price": None, "change_pct": None, "success": False}

    failure = {"price": None, "change_pct": None, "success": False}

    # Persistent cache first (if available)
    if RESILIENCE_AVAILABLE:
        try:
            cache = get_persistent_cache()
            cached = cache.get(f"ticker_price:{t}")
            if cached is not None:
                return cached
        except Exception:
            pass

    # Circuit breaker (if available)
    if RESILIENCE_AVAILABLE:
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
            return failure
        except Exception:
            return failure

    # No resilience available
    return _fetch_ticker_price_data_internal(t)


# ============================================================================
# SECTION 3: Earnings Data
# ============================================================================

@conditional_cache(ttl=3600)
def get_earnings_date(ticker: str) -> Optional[str]:
    """
    Best-effort earnings date fetch.
    Must NEVER raise.
    """
    t = normalize_ticker(ticker)
    if t is None:
        return None

    try:
        import yfinance as yf  # type: ignore

        stock = yf.Ticker(t)
        cal = stock.calendar
        if cal is None or getattr(cal, "empty", True):
            return None

        # yfinance calendar shape varies; be defensive
        if "Earnings Date" in cal.index:
            val = cal.loc["Earnings Date"]
            if hasattr(val, "strftime"):
                return val.strftime("%Y-%m-%d")
            if isinstance(val, str):
                return val
        return None
    except Exception:
        return None


# ============================================================================
# SECTION 4: Fed/Macro Indicators (static placeholders)
# ============================================================================

@conditional_cache(ttl=86400)
def get_fed_indicators() -> Dict[str, Optional[str]]:
    """
    Lightweight placeholders (no external API).
    """
    try:
        current_rate = "4.25-4.50%"
        return {
            "fed_funds_rate": current_rate,
            "next_fomc_date": None,
            "cpi_latest": "N/A",
            "jobs_latest": "N/A",
        }
    except Exception:
        return {"fed_funds_rate": "N/A", "next_fomc_date": None, "cpi_latest": "N/A", "jobs_latest": "N/A"}


# ============================================================================
# SECTION 5: WAVES Internal Status
# ============================================================================

def get_waves_status() -> Dict[str, str]:
    try:
        now = datetime.now().strftime("%H:%M:%S")
        if STREAMLIT_AVAILABLE:
            waves_loaded = "ACTIVE" if st.session_state.get("wave_universe") else "LOADING"  # type: ignore
        else:
            waves_loaded = "N/A"
        return {"system_status": "ONLINE", "last_update": now, "waves_status": waves_loaded}
    except Exception:
        return {"system_status": "ONLINE", "last_update": "N/A", "waves_status": "N/A"}


# ============================================================================
# SECTION 6: Cache Management (events_cache.json)
# ============================================================================

def load_events_cache() -> Dict:
    try:
        base_dir = os.path.dirname(os.path.dirname(__file__))
        p = os.path.join(base_dir, "data", "events_cache.json")
        if os.path.exists(p):
            with open(p, "r") as f:
                return json.load(f)
        return {}
    except Exception:
        return {}


def save_events_cache(cache_data: Dict) -> bool:
    try:
        base_dir = os.path.dirname(os.path.dirname(__file__))
        p = os.path.join(base_dir, "data", "events_cache.json")
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w") as f:
            json.dump(cache_data, f, indent=2)
        return True
    except Exception:
        return False


def update_cache_with_current_data() -> None:
    try:
        cache_data = {
            "last_updated": datetime.now().isoformat(),
            "fed_indicators": get_fed_indicators(),
            "waves_status": get_waves_status(),
        }
        save_events_cache(cache_data)
    except Exception:
        pass


# ============================================================================
# SECTION 7: Data Health Tracking
# ============================================================================

def get_ticker_health_status() -> Dict[str, Any]:
    """
    Must NEVER raise.
    """
    health: Dict[str, Any] = {
        "timestamp": datetime.now().isoformat(),
        "resilience_available": RESILIENCE_AVAILABLE,
        "circuit_breakers": {},
        "cache_stats": {},
        "overall_status": "unknown",
    }

    try:
        if RESILIENCE_AVAILABLE:
            try:
                from .circuit_breaker import get_all_circuit_states  # type: ignore
                health["circuit_breakers"] = get_all_circuit_states()
                open_count = sum(
                    1 for _k, v in health["circuit_breakers"].items()
                    if isinstance(v, dict) and v.get("state") == "open"
                )
                health["overall_status"] = "degraded" if open_count > 0 else "healthy"
            except Exception as e:
                health["circuit_breakers"] = {"error": str(e)}
                health["overall_status"] = "unknown"

            try:
                cache = get_persistent_cache()
                health["cache_stats"] = cache.get_stats()
            except Exception as e:
                health["cache_stats"] = {"error": str(e)}
        else:
            health["overall_status"] = "healthy"
    except Exception as e:
        health["overall_status"] = "unknown"
        health["error"] = str(e)

    return health


def test_ticker_fetch(ticker: str = "AAPL") -> Dict[str, Any]:
    import time

    result: Dict[str, Any] = {
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
        result["latency_ms"] = round((time.time() - start) * 1000, 2)
        result["data"] = data
        result["success"] = bool(data.get("success", False)) if isinstance(data, dict) else False
    except Exception as e:
        result["error"] = str(e)
        result["success"] = False

    return result