"""
PRICE_BOOK — Canonical Price Source of Truth

This module is the ONLY authoritative source of price data for the system.
All pricing, readiness, diagnostics, governance, and execution intelligence
must flow through this module.

Core Guarantees:
- ONE canonical EOD cache: data/cache/prices_cache.parquet
- Optional intraday overlay (read-only, non-authoritative)
- NO implicit network access
- Deterministic, reproducible outputs
- Stable public API (UI-safe + backward compatible)
"""

import os
import logging
import warnings
from typing import List, Optional, Dict, Any
from pathlib import Path

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

# =============================================================================
# ENVIRONMENT & FETCH CONTROL
# =============================================================================
LIVE_DATA_ENABLED = os.getenv("LIVE_DATA_ENABLED", "false").lower() == "true"
ALLOW_NETWORK_FETCH = LIVE_DATA_ENABLED

# -----------------------------------------------------------------------------
# LEGACY BACKWARD-COMPATIBILITY ALIAS (DO NOT REMOVE)
# -----------------------------------------------------------------------------
PRICE_FETCH_ENABLED = ALLOW_NETWORK_FETCH
warnings.warn(
    "PRICE_FETCH_ENABLED is deprecated and will be removed in a future release. "
    "Please use ALLOW_NETWORK_FETCH instead.",
    DeprecationWarning,
    stacklevel=2,
)

# =============================================================================
# CACHE CONFIG
# =============================================================================
CACHE_DIR = "data/cache"

EOD_CACHE_FILE = "prices_cache.parquet"
INTRADAY_CACHE_FILE = "intraday_equity_prices.parquet"

EOD_CACHE_PATH = os.path.join(CACHE_DIR, EOD_CACHE_FILE)
INTRADAY_CACHE_PATH = os.path.join(CACHE_DIR, INTRADAY_CACHE_FILE)

CANONICAL_CACHE_PATH = EOD_CACHE_PATH  # governance anchor

# =============================================================================
# HEALTH THRESHOLDS
# =============================================================================
PRICE_CACHE_OK_DAYS = int(os.getenv("PRICE_CACHE_OK_DAYS", "14"))
PRICE_CACHE_DEGRADED_DAYS = int(os.getenv("PRICE_CACHE_DEGRADED_DAYS", "30"))

if PRICE_CACHE_DEGRADED_DAYS <= PRICE_CACHE_OK_DAYS:
    logger.warning("Invalid cache thresholds — resetting to defaults")
    PRICE_CACHE_OK_DAYS = 14
    PRICE_CACHE_DEGRADED_DAYS = 30

STALE_DAYS_THRESHOLD = PRICE_CACHE_DEGRADED_DAYS

# =============================================================================
# DEPENDENCIES (LOW-LEVEL IO ONLY)
# =============================================================================
try:
    from helpers.price_loader import (
        load_cache,
        collect_required_tickers,
        deduplicate_tickers,
        load_or_fetch_prices,
        get_cache_info,
        check_cache_readiness,
    )
except ImportError:
    logger.exception("Failed importing helpers.price_loader")
    raise

# =============================================================================
# INTERNAL FALLBACKS (SAFE MODE)
# =============================================================================
def _read_parquet_safe(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        return pd.read_parquet(path)
    except Exception as e:
        logger.warning(f"Failed reading {path}: {e}")
        return pd.DataFrame()


def _dedupe_fallback(tickers: List[str]) -> List[str]:
    return sorted(set(tickers))


# =============================================================================
# REQUIRED TICKERS (SINGLE SOURCE)
# =============================================================================
def _get_required_tickers(active_only: bool = True) -> List[str]:
    tickers = collect_required_tickers(active_only=active_only)
    dedupe = deduplicate_tickers or _dedupe_fallback
    return dedupe(tickers)


# =============================================================================
# PUBLIC / LEGACY API (DO NOT BREAK)
# =============================================================================
def get_active_required_tickers(active_only: bool = True) -> List[str]:
    return _get_required_tickers(active_only=active_only)


def get_required_tickers(active_only: bool = True) -> List[str]:
    return _get_required_tickers(active_only=active_only)


def get_required_tickers_active_waves() -> List[str]:
    return _get_required_tickers(active_only=True)


# =============================================================================
# PRICE BOOK ACCESS (READ-ONLY, PRIORITY-AWARE)
# =============================================================================
def _load_intraday_cache() -> pd.DataFrame:
    df = _read_parquet_safe(INTRADAY_CACHE_PATH)
    if not df.empty:
        logger.info("PRICE_BOOK: using intraday cache")
    return df


def _load_eod_cache() -> pd.DataFrame:
    df = load_cache() if load_cache else _read_parquet_safe(EOD_CACHE_PATH)
    if not df.empty:
        logger.info("PRICE_BOOK: using EOD canonical cache")
    return df


def get_price_book(
    active_tickers: Optional[List[str]] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """
    Returns the best available price book.

    Priority:
    1. Intraday cache (if present)
    2. Canonical EOD cache
    3. Empty DataFrame (safe mode)
    """
    df = _load_intraday_cache()
    if df.empty:
        df = _load_eod_cache()

    if df is None or df.empty:
        out = pd.DataFrame(columns=active_tickers or [])
        out.index.name = "Date"
        return out

    if start_date:
        df = df[df.index >= pd.to_datetime(start_date)]
    if end_date:
        df = df[df.index <= pd.to_datetime(end_date)]

    if active_tickers:
        dedupe = deduplicate_tickers or _dedupe_fallback
        tickers = dedupe(active_tickers)
        for t in tickers:
            if t not in df.columns:
                df[t] = np.nan
        df = df[tickers]

    return df


# =============================================================================
# METADATA
# =============================================================================
def get_price_book_meta(price_book: Optional[pd.DataFrame]) -> Dict[str, Any]:
    if price_book is None or price_book.empty:
        return {
            "is_empty": True,
            "rows": 0,
            "cols": 0,
            "tickers": [],
            "cache_path": CANONICAL_CACHE_PATH,
        }

    return {
        "is_empty": False,
        "rows": len(price_book),
        "cols": len(price_book.columns),
        "date_min": price_book.index[0].strftime("%Y-%m-%d"),
        "date_max": price_book.index[-1].strftime("%Y-%m-%d"),
        "tickers": sorted(price_book.columns.tolist()),
        "cache_path": CANONICAL_CACHE_PATH,
    }


# =============================================================================
# REALITY PANEL COMPATIBILITY
# =============================================================================
def compute_missing_and_extra_tickers(price_book: pd.DataFrame) -> Dict[str, Any]:
    required = set(_get_required_tickers(active_only=True))
    cached = set() if price_book is None or price_book.empty else set(price_book.columns)

    missing = sorted(required - cached)
    extra = sorted(cached - required)

    return {
        "missing_tickers": missing,
        "extra_tickers": extra,
        "missing_count": len(missing),
        "extra_count": len(extra),
        "required_count": len(required),
        "cached_count": len(cached),
    }


# =============================================================================
# EXPLICIT CACHE REBUILD (NETWORK ENTRY POINT)
# =============================================================================
def rebuild_price_cache(
    active_only: bool = True,
    force_user_initiated: bool = False,
) -> Dict[str, Any]:
    if not ALLOW_NETWORK_FETCH and not force_user_initiated:
        return {
            "allowed": False,
            "success": False,
            "message": "LIVE_DATA_ENABLED=false — network fetch blocked",
        }

    required = _get_required_tickers(active_only=active_only)
    prices = load_or_fetch_prices(required, force_fetch=True)
    cache_info = get_cache_info() or {}

    return {
        "allowed": True,
        "success": not prices.empty,
        "tickers_requested": len(required),
        "tickers_fetched": len(prices.columns),
        "last_updated": cache_info.get("last_updated"),
    }


# =============================================================================
# SINGLETON (SYSTEM-WIDE)
# =============================================================================
_PRICE_BOOK_CACHE: Optional[pd.DataFrame] = None
_PRICE_BOOK_LOADED = False


def get_price_book_singleton(force_reload: bool = False) -> pd.DataFrame:
    global _PRICE_BOOK_CACHE, _PRICE_BOOK_LOADED
    if force_reload or not _PRICE_BOOK_LOADED:
        _PRICE_BOOK_CACHE = get_price_book()
        _PRICE_BOOK_LOADED = True
    return _PRICE_BOOK_CACHE


PRICE_BOOK = get_price_book_singleton


# =============================================================================
# SYSTEM HEALTH
# =============================================================================
def compute_system_health(price_book: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
    if price_book is None:
        price_book = get_price_book_singleton()

    required = _get_required_tickers(active_only=True)
    cached = [] if price_book.empty else price_book.columns.tolist()

    missing = sorted(set(required) - set(cached))
    coverage_pct = 100.0 if not required else (1 - len(missing) / len(required)) * 100

    readiness = check_cache_readiness(active_only=True) if check_cache_readiness else {}
    days_stale = readiness.get("days_stale", 0)

    if price_book.empty:
        status, emoji = "EMPTY", "⚪"
    elif days_stale > STALE_DAYS_THRESHOLD:
        status, emoji = "STALE", "❌"
    elif missing:
        status, emoji = "DEGRADED", "⚠️"
    else:
        status, emoji = "OK", "✅"

    return {
        "health_status": status,
        "health_emoji": emoji,
        "coverage_pct": coverage_pct,
        "missing_count": len(missing),
        "days_stale": days_stale,
        "details": f"{status} — {coverage_pct:.1f}% coverage",
    }


# =============================================================================
# EXPORT SAFETY
# =============================================================================
__all__ = [
    "LIVE_DATA_ENABLED",
    "ALLOW_NETWORK_FETCH",
    "PRICE_FETCH_ENABLED",
    "get_price_book",
    "get_price_book_singleton",
    "get_active_required_tickers",
    "get_required_tickers",
    "get_required_tickers_active_waves",
    "compute_missing_and_extra_tickers",
    "compute_system_health",
    "rebuild_price_cache",
    "PRICE_BOOK",
]