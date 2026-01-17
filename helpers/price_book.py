"""
PRICE_BOOK - Single Canonical Price Source of Truth

This module provides THE authoritative price data loader for the entire application.
All price data must flow through this module to ensure consistency and prevent
"two truths" problems.

Key Principles:
- ONE canonical cache file: data/cache/prices_cache.parquet
- NO implicit fetching - all fetches are explicit and controlled
- PRICE_BOOK is a DataFrame: index=dates, columns=tickers, values=close prices
- All readiness, health, execution, and diagnostics use the SAME PRICE_BOOK
"""

import os
import logging
from datetime import datetime
from typing import List, Optional, Dict, Any

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

# ============================================================================
# LIVE / NETWORK FETCH CONTROL (SINGLE SOURCE OF TRUTH)
# ============================================================================
LIVE_DATA_ENABLED = os.getenv("LIVE_DATA_ENABLED", "false").lower() == "true"

# Canonical fetch gate used everywhere
ALLOW_NETWORK_FETCH = LIVE_DATA_ENABLED

# ============================================================================
# Canonical cache configuration
# ============================================================================
CACHE_DIR = "data/cache"
CACHE_FILE = "prices_cache.parquet"
CANONICAL_CACHE_PATH = os.path.join(CACHE_DIR, CACHE_FILE)

# ============================================================================
# Cache health thresholds
# ============================================================================
_logger = logging.getLogger(__name__)

try:
    PRICE_CACHE_OK_DAYS = int(os.getenv("PRICE_CACHE_OK_DAYS", "14"))
except (ValueError, TypeError):
    _logger.warning("Invalid PRICE_CACHE_OK_DAYS, using default 14")
    PRICE_CACHE_OK_DAYS = 14

try:
    PRICE_CACHE_DEGRADED_DAYS = int(os.getenv("PRICE_CACHE_DEGRADED_DAYS", "30"))
except (ValueError, TypeError):
    _logger.warning("Invalid PRICE_CACHE_DEGRADED_DAYS, using default 30")
    PRICE_CACHE_DEGRADED_DAYS = 30

if PRICE_CACHE_DEGRADED_DAYS <= PRICE_CACHE_OK_DAYS:
    _logger.warning("Invalid cache thresholds, resetting to defaults")
    PRICE_CACHE_OK_DAYS = 14
    PRICE_CACHE_DEGRADED_DAYS = 30

CRITICAL_MISSING_THRESHOLD = 0.5
STALE_DAYS_THRESHOLD = PRICE_CACHE_DEGRADED_DAYS
DEGRADED_DAYS_THRESHOLD = PRICE_CACHE_OK_DAYS

# ============================================================================
# Import price_loader helpers
# ============================================================================
try:
    from helpers.price_loader import (
        load_cache,
        save_cache,
        collect_required_tickers,
        normalize_ticker,
        deduplicate_tickers,
        load_or_fetch_prices,
        get_cache_info,
        check_cache_readiness,
        CACHE_PATH,
    )
except ImportError:
    logger.error("FAILED importing helpers.price_loader", exc_info=True)
    raise


# ============================================================================
# Fallback helpers
# ============================================================================
def _load_cache_fallback() -> pd.DataFrame:
    if not os.path.exists(CANONICAL_CACHE_PATH):
        logger.warning(f"Cache file not found: {CANONICAL_CACHE_PATH}")
        return pd.DataFrame()

    try:
        return pd.read_parquet(CANONICAL_CACHE_PATH)
    except Exception as e:
        logger.error(f"Failed loading cache directly: {e}")
        return pd.DataFrame()


def _deduplicate_tickers_fallback(tickers: List[str]) -> List[str]:
    return sorted(set(tickers))


# ============================================================================
# PRICE_BOOK access
# ============================================================================
def get_price_book(
    active_tickers: Optional[List[str]] = None,
    mode: str = "Standard",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """
    Load the canonical PRICE_BOOK from cache only.
    NEVER fetches from the network.
    """

    logger.info("PRICE_BOOK: loading canonical cache")

    cache_df = load_cache() if load_cache else _load_cache_fallback()

    if cache_df is None or cache_df.empty:
        logger.warning("PRICE_BOOK is empty")
        df = pd.DataFrame(columns=active_tickers or [])
        df.index.name = "Date"
        return df

    if start_date:
        cache_df = cache_df[cache_df.index >= pd.to_datetime(start_date)]
    if end_date:
        cache_df = cache_df[cache_df.index <= pd.to_datetime(end_date)]

    if active_tickers:
        dedupe = deduplicate_tickers if deduplicate_tickers else _deduplicate_tickers_fallback
        tickers = dedupe(active_tickers)

        for t in tickers:
            if t not in cache_df.columns:
                cache_df[t] = np.nan

        cache_df = cache_df[tickers]

    return cache_df


def get_price_book_meta(price_book: pd.DataFrame) -> Dict[str, Any]:
    if price_book is None or price_book.empty:
        return {
            "date_min": None,
            "date_max": None,
            "rows": 0,
            "cols": 0,
            "tickers_count": 0,
            "tickers": [],
            "is_empty": True,
            "cache_path": CANONICAL_CACHE_PATH,
        }

    return {
        "date_min": price_book.index[0].strftime("%Y-%m-%d"),
        "date_max": price_book.index[-1].strftime("%Y-%m-%d"),
        "rows": len(price_book),
        "cols": len(price_book.columns),
        "tickers_count": len(price_book.columns),
        "tickers": sorted(price_book.columns.tolist()),
        "is_empty": False,
        "cache_path": CANONICAL_CACHE_PATH,
    }


# ============================================================================
# Explicit cache rebuild (ONLY place network is allowed)
# ============================================================================
def rebuild_price_cache(
    active_only: bool = True,
    force_user_initiated: bool = False,
) -> Dict[str, Any]:
    """
    Explicit user-triggered cache rebuild.
    """

    if not ALLOW_NETWORK_FETCH and not force_user_initiated:
        return {
            "allowed": False,
            "success": False,
            "message": "Live fetching disabled (LIVE_DATA_ENABLED=false)",
        }

    required = collect_required_tickers(active_only=active_only)
    prices = load_or_fetch_prices(required, force_fetch=True)

    cache_info = get_cache_info() if get_cache_info else {}

    return {
        "allowed": True,
        "success": not prices.empty,
        "tickers_requested": len(required),
        "tickers_fetched": len(prices.columns),
        "date_max": cache_info.get("last_updated"),
        "cache_info": cache_info,
    }


# ============================================================================
# Singleton PRICE_BOOK
# ============================================================================
_PRICE_BOOK_CACHE: Optional[pd.DataFrame] = None
_PRICE_BOOK_LOADED = False


def get_price_book_singleton(force_reload: bool = False) -> pd.DataFrame:
    global _PRICE_BOOK_CACHE, _PRICE_BOOK_LOADED

    if force_reload or not _PRICE_BOOK_LOADED:
        _PRICE_BOOK_CACHE = get_price_book()
        _PRICE_BOOK_LOADED = True

    return _PRICE_BOOK_CACHE


PRICE_BOOK = get_price_book_singleton


# ============================================================================
# System health diagnostics
# ============================================================================
def compute_system_health(price_book: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
    if price_book is None:
        price_book = get_price_book_singleton()

    required = collect_required_tickers(active_only=True)
    cached = [] if price_book.empty else price_book.columns.tolist()

    missing = sorted(set(required) - set(cached))
    coverage_pct = 100.0 if not required else (1 - len(missing) / len(required)) * 100

    readiness = check_cache_readiness(active_only=True) if check_cache_readiness else {}
    days_stale = readiness.get("days_stale", 0)

    if price_book.empty or days_stale > STALE_DAYS_THRESHOLD:
        status = "STALE"
        emoji = "❌"
    elif missing:
        status = "DEGRADED"
        emoji = "⚠️"
    else:
        status = "OK"
        emoji = "✅"

    return {
        "health_status": status,
        "health_emoji": emoji,
        "missing_count": len(missing),
        "total_required": len(required),
        "coverage_pct": coverage_pct,
        "days_stale": days_stale,
        "details": f"{status} — {coverage_pct:.1f}% coverage",
    }