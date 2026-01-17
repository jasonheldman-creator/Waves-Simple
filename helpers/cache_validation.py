"""
Cache Validation Module

This module provides robust validation for the price cache pipeline with:
- Trading-day freshness validation using SPY as reference
- Required symbol validation with ALL/ANY group semantics
- Cache integrity checks
- No-change detection for fresh/stale scenarios
"""

import os
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any

import pandas as pd

logger = logging.getLogger(__name__)

# Required symbol groups
REQUIRED_SYMBOLS_ALL = ["SPY", "QQQ", "IWM"]
REQUIRED_SYMBOLS_VIX_ANY = ["^VIX", "VIXY", "VXX"]
REQUIRED_SYMBOLS_TBILL_ANY = ["BIL", "SHY"]


def fetch_spy_trading_days(calendar_days: int = 10) -> Tuple[Optional[datetime], List[datetime]]:
    try:
        import yfinance as yf

        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=calendar_days)

        spy_data = yf.download(
            tickers="SPY",
            start=start_date.strftime("%Y-%m-%d"),
            end=(end_date + timedelta(days=1)).strftime("%Y-%m-%d"),
            auto_adjust=True,
            progress=False,
            timeout=15,
        )

        if spy_data.empty:
            return None, []

        trading_days = sorted(pd.to_datetime(spy_data.index).to_pydatetime().tolist())
        return trading_days[-1], trading_days

    except Exception as e:
        logger.error(f"Failed to fetch SPY trading days: {e}")
        return None, []


def get_cache_max_date(cache_path: str) -> Optional[datetime]:
    try:
        if not os.path.exists(cache_path):
            return None

        df = pd.read_parquet(cache_path)
        if df.empty:
            return None

        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)

        max_date = df.index.max()
        return max_date.to_pydatetime() if hasattr(max_date, "to_pydatetime") else max_date

    except Exception as e:
        logger.error(f"Failed reading cache max date: {e}")
        return None


def validate_trading_day_freshness(
    cache_path: str,
    max_market_feed_gap_days: int = 5,
    allow_metadata_bootstrap: bool = False,
) -> Dict[str, Any]:
    result = {
        "valid": False,
        "today": datetime.utcnow(),
        "last_trading_day": None,
        "cache_max_date": None,
        "spy_max_date": None,
        "error": None,
    }

    last_trading_day, trading_days = fetch_spy_trading_days()
    cache_max_date = get_cache_max_date(cache_path)

    result["last_trading_day"] = last_trading_day
    result["cache_max_date"] = cache_max_date

    # 🚨 CRITICAL FIX — bootstrap spy_max_date if missing
    if last_trading_day is None and cache_max_date is not None and allow_metadata_bootstrap:
        logger.warning("BOOTSTRAP: Using cache_max_date as spy_max_date")
        last_trading_day = cache_max_date

    result["spy_max_date"] = last_trading_day

    if last_trading_day is None or cache_max_date is None:
        result["error"] = "Missing trading day or cache date"
        return result

    market_gap = (result["today"].date() - last_trading_day.date()).days
    if market_gap > max_market_feed_gap_days:
        result["error"] = "Market data feed appears stale"
        return result

    cache_date = cache_max_date.date()
    trading_date = last_trading_day.date()

    if cache_date == trading_date:
        result["valid"] = True
        return result

    if trading_days:
        trading_dates = [d.date() for d in trading_days]
        if len(trading_dates) >= 2 and cache_date == trading_dates[-2]:
            result["valid"] = True
            logger.warning("Cache is 1 trading session behind (allowed)")
            return result

    result["error"] = f"Cache date {cache_date} != trading date {trading_date}"
    return result


def validate_required_symbols(cache_path: str) -> Dict[str, Any]:
    result = {"valid": False, "error": None}

    try:
        df = pd.read_parquet(cache_path)
        symbols = set(df.columns)

        missing_all = [s for s in REQUIRED_SYMBOLS_ALL if s not in symbols]
        if missing_all:
            result["error"] = f"Missing ALL symbols: {missing_all}"
            return result

        if not any(s in symbols for s in REQUIRED_SYMBOLS_VIX_ANY):
            result["error"] = "Missing VIX proxy"
            return result

        if not any(s in symbols for s in REQUIRED_SYMBOLS_TBILL_ANY):
            result["error"] = "Missing T-bill proxy"
            return result

        result["valid"] = True
        return result

    except Exception as e:
        result["error"] = str(e)
        return result


def validate_cache_integrity(cache_path: str) -> Dict[str, Any]:
    result = {"valid": False, "error": None}

    if not os.path.exists(cache_path):
        result["error"] = "Cache file missing"
        return result

    if os.path.getsize(cache_path) == 0:
        result["error"] = "Cache file empty"
        return result

    try:
        df = pd.read_parquet(cache_path)
        if df.empty or len(df.columns) == 0:
            result["error"] = "Cache contains no symbols"
            return result

        result["valid"] = True
        return result

    except Exception as e:
        result["error"] = str(e)
        return result


def check_for_changes(repo_path: str = ".") -> Dict[str, Any]:
    import subprocess

    try:
        status = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=repo_path, text=True
        )
        diff = subprocess.check_output(
            ["git", "diff", "--stat"], cwd=repo_path, text=True
        )
        return {
            "has_changes": bool(status.strip()),
            "git_status": status.strip(),
            "git_diff_stat": diff.strip(),
        }
    except Exception:
        return {"has_changes": False, "git_status": "", "git_diff_stat": ""}


def validate_no_change_logic(cache_fresh: bool, has_changes: bool) -> Dict[str, Any]:
    if cache_fresh:
        return {
            "should_commit": has_changes,
            "should_succeed": True,
            "message": "Fresh cache",
        }

    if has_changes:
        return {
            "should_commit": True,
            "should_succeed": True,
            "message": "Stale but updated",
        }

    return {
        "should_commit": False,
        "should_succeed": False,
        "message": "Stale and unchanged",
    }