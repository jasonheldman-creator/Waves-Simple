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
REQUIRED_SYMBOLS_ALL = ["SPY", "QQQ", "IWM"]  # ALL must be present
REQUIRED_SYMBOLS_VIX_ANY = ["^VIX", "VIXY", "VXX"]  # At least ONE must be present
REQUIRED_SYMBOLS_TBILL_ANY = ["BIL", "SHY"]  # At least ONE must be present


def fetch_spy_trading_days(calendar_days: int = 10) -> Tuple[Optional[datetime], List[datetime]]:
    """
    Fetch SPY prices for the last N calendar days to determine trading days.
    
    Args:
        calendar_days: Number of calendar days to look back (default: 10)
        
    Returns:
        Tuple of (last_trading_day, all_trading_days):
        - last_trading_day: Most recent trading day from SPY data (or None if unavailable)
        - all_trading_days: List of all trading days in the period
    """
    try:
        import yfinance as yf
        
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=calendar_days)
        
        logger.info(f"Fetching SPY prices from {start_date.date()} to {end_date.date()}")
        
        # Fetch SPY data
        spy_data = yf.download(
            tickers="SPY",
            start=start_date.strftime('%Y-%m-%d'),
            end=(end_date + timedelta(days=1)).strftime('%Y-%m-%d'),
            auto_adjust=True,
            progress=False,
            timeout=15
        )
        
        if spy_data.empty:
            logger.error("No SPY data returned from yfinance")
            return None, []
        
        # Extract trading days from index
        trading_days = [pd.Timestamp(dt).to_pydatetime() for dt in spy_data.index]
        
        # Ensure trading days are in ascending chronological order
        trading_days = sorted(trading_days)
        
        if not trading_days:
            logger.error("No trading days found in SPY data")
            return None, []
        
        last_trading_day = max(trading_days)
        
        logger.info(f"Found {len(trading_days)} trading days, last: {last_trading_day.date()}")
        
        return last_trading_day, trading_days
        
    except ImportError:
        logger.error("yfinance not available - cannot fetch SPY data")
        return None, []
    except Exception as e:
        logger.error(f"Error fetching SPY trading days: {e}")
        return None, []


def get_cache_max_date(cache_path: str) -> Optional[datetime]:
    """
    Get the maximum date from the cache parquet file.
    
    Args:
        cache_path: Path to the parquet cache file
        
    Returns:
        Maximum date in the cache (or None if unavailable)
    """
    try:
        if not os.path.exists(cache_path):
            logger.error(f"Cache file does not exist: {cache_path}")
            return None
        
        # Read parquet file
        cache_df = pd.read_parquet(cache_path)
        
        if cache_df.empty:
            logger.error("Cache file is empty")
            return None
        
        # Ensure index is datetime
        if not isinstance(cache_df.index, pd.DatetimeIndex):
            cache_df.index = pd.to_datetime(cache_df.index)
        
        # Get max date
        max_date = cache_df.index.max()
        
        # Convert to datetime if it's a Timestamp
        if hasattr(max_date, 'to_pydatetime'):
            max_date = max_date.to_pydatetime()
        
        logger.info(f"Cache max date: {max_date.date()}")
        
        return max_date
        
    except Exception as e:
        logger.error(f"Error reading cache max date: {e}")
        return None


def validate_trading_day_freshness(
    cache_path: str,
    max_market_feed_gap_days: int = 5
) -> Dict[str, Any]:
    """
    Validate that the cache is up-to-date with the latest trading day.
    
    This performs validation with 1-session tolerance:
    - Fetches SPY prices for last 10 calendar days
    - Computes last_trading_day = max(date_index_of_SPY)
    - Computes cache_max_date from parquet file
    - PASSES if cache_max_date == last_trading_day
    - PASSES if cache_max_date is within 1 trading session before last_trading_day
    - FAILS if cache_max_date is more than 1 trading session behind
    - FAILS with "Market data feed likely broken" if today - last_trading_day > max_market_feed_gap_days
    
    Args:
        cache_path: Path to the cache parquet file
        max_market_feed_gap_days: Maximum gap (in calendar days) before market feed is considered broken
        
    Returns:
        Dictionary with:
        - valid: bool - Whether validation passed
        - today: datetime - Current date
        - last_trading_day: datetime or None - Most recent SPY trading day
        - cache_max_date: datetime or None - Most recent date in cache
        - sessions_behind: int or None - Number of trading sessions cache is behind
        - delta_days: int or None - Difference between cache_max_date and last_trading_day
        - market_feed_gap_days: int or None - Days between today and last_trading_day
        - error: str or None - Error message if validation failed
    """
    result = {
        'valid': False,
        'today': datetime.now(),
        'last_trading_day': None,
        'cache_max_date': None,
        'delta_days': None,
        'sessions_behind': None,
        'market_feed_gap_days': None,
        'error': None
    }
    
    logger.info("=" * 70)
    logger.info("TRADING-DAY FRESHNESS VALIDATION")
    logger.info("=" * 70)
    
    # Step 1: Fetch SPY trading days
    last_trading_day, trading_days = fetch_spy_trading_days(calendar_days=10)
    result['last_trading_day'] = last_trading_day
    
    if last_trading_day is None:
        result['error'] = "Failed to fetch SPY trading days"
        logger.error(f"✗ FAIL: {result['error']}")
        return result
    
    # Step 2: Sanity check - market data feed health
    market_feed_gap = (result['today'] - last_trading_day).days
    result['market_feed_gap_days'] = market_feed_gap
    
    if market_feed_gap > max_market_feed_gap_days:
        result['error'] = f"Market data feed likely broken: {market_feed_gap} days since last trading day (>{max_market_feed_gap_days} days)"
        logger.error(f"✗ FAIL: {result['error']}")
        return result
    
    # Step 3: Get cache max date
    cache_max_date = get_cache_max_date(cache_path)
    result['cache_max_date'] = cache_max_date
    
    if cache_max_date is None:
        result['error'] = "Failed to read cache max date"
        logger.error(f"✗ FAIL: {result['error']}")
        return result
    
    # Step 4: Compare cache_max_date with last_trading_day
    # Normalize both to date only (ignore time component)
    cache_date_only = cache_max_date.date()
    trading_date_only = last_trading_day.date()
    
    delta_days = (cache_date_only - trading_date_only).days
    result['delta_days'] = delta_days
    
    logger.info(f"Today: {result['today'].date()}")
    logger.info(f"Last trading day: {trading_date_only}")
    logger.info(f"Cache max date: {cache_date_only}")
    logger.info(f"Delta (cache - trading): {delta_days} days")
    logger.info(f"Market feed gap: {market_feed_gap} days")
    
    # Check if cache is current or within 1 trading session tolerance
    # Normalize trading_days to date objects for comparison
    trading_dates = [pd.Timestamp(dt).date() for dt in trading_days]
    
    # Allow tolerance: cache may be 1 trading day behind last trading day
prev_trading_date_only = trading_dates[-2] if len(trading_dates) >= 2 else trading_date_only

# PASS if cache matches last trading day OR previous trading day
if cache_date_only == trading_date_only or cache_date_only == prev_trading_date_only:
    result['valid'] = True

    if cache_date_only == prev_trading_date_only and cache_date_only != trading_date_only:
        logger.warning("⚠ PASS: Cache is 1 trading day behind (allowed tolerance).")
    else:
        logger.info("✓ PASS: Cache is fresh and matches last trading day.")

    logger.info("=" * 70)
    return result
    
    # Check if cache_max_date is within the trading days list
    if cache_date_only not in trading_dates:
        result['error'] = f"Cache max date ({cache_date_only}) is not a valid trading day"
        logger.error(f"✗ FAIL: {result['error']}")
        return result
    
    # Get the index of last_trading_day and cache_max_date in the trading days
    # Note: trading_dates is in ascending chronological order (earlier dates have lower indices)
    try:
        cache_index = trading_dates.index(cache_date_only)
        last_trading_index = trading_dates.index(trading_date_only)
    except ValueError as e:
        result['error'] = f"Error finding trading day indices: {e}"
        logger.error(f"✗ FAIL: {result['error']}")
        return result
    
    # Calculate sessions behind (lower index = earlier date in ascending order)
    sessions_behind = last_trading_index - cache_index
    result['sessions_behind'] = sessions_behind
    
    # Allow cache to be up to 1 trading session behind
    if 0 <= sessions_behind <= 1:
        result['valid'] = True
        if sessions_behind == 0:
            logger.info("✓ PASS: Cache is fresh and up-to-date with latest trading day")
        else:
            # sessions_behind == 1: PASS but log as WARNING
            logger.warning(f"⚠ WARNING: Cache is 1 trading session behind (still PASSING)")
            logger.info(f"✓ PASS: Cache is within tolerance ({sessions_behind} trading session behind)")
        logger.info("=" * 70)
        return result
    
    # Cache is more than 1 session behind or ahead (which shouldn't happen)
    if sessions_behind < 0:
        result['error'] = f"Cache max date ({cache_date_only}) is ahead of last trading day ({trading_date_only})"
    else:
        result['error'] = f"Cache max date ({cache_date_only}) is {sessions_behind} trading sessions behind last trading day ({trading_date_only})"
    logger.error(f"✗ FAIL: {result['error']}")
    return result


def validate_required_symbols(cache_path: str) -> Dict[str, Any]:
    """
    Validate that the cache contains all required symbols.
    
    Required symbols are organized into groups with ALL/ANY semantics:
    - ALL group: SPY, QQQ, IWM (all must be present)
    - ANY group 1: ^VIX, VIXY, or VXX (at least one must be present)
    - ANY group 2: BIL or SHY (at least one must be present)
    
    Args:
        cache_path: Path to the cache parquet file
        
    Returns:
        Dictionary with:
        - valid: bool - Whether validation passed
        - symbols_in_cache: list - All symbols in cache
        - missing_all_group: list - Missing symbols from ALL group
        - present_vix_group: list - Present symbols from VIX ANY group
        - present_tbill_group: list - Present symbols from T-bill ANY group
        - error: str or None - Error message if validation failed
    """
    result = {
        'valid': False,
        'symbols_in_cache': [],
        'missing_all_group': [],
        'present_vix_group': [],
        'present_tbill_group': [],
        'error': None
    }
    
    logger.info("=" * 70)
    logger.info("REQUIRED SYMBOLS VALIDATION")
    logger.info("=" * 70)
    
    try:
        # Read cache to get symbol list
        if not os.path.exists(cache_path):
            result['error'] = f"Cache file does not exist: {cache_path}"
            logger.error(f"✗ FAIL: {result['error']}")
            return result
        
        cache_df = pd.read_parquet(cache_path)
        
        if cache_df.empty:
            result['error'] = "Cache file is empty"
            logger.error(f"✗ FAIL: {result['error']}")
            return result
        
        # Get symbols from cache
        symbols_in_cache = set(cache_df.columns)
        result['symbols_in_cache'] = sorted(list(symbols_in_cache))
        
        logger.info(f"Symbols in cache: {len(symbols_in_cache)}")
        
        # Validate ALL group (SPY, QQQ, IWM)
        missing_all = [sym for sym in REQUIRED_SYMBOLS_ALL if sym not in symbols_in_cache]
        result['missing_all_group'] = missing_all
        
        if missing_all:
            result['error'] = f"ALL group validation failed - missing symbols: {missing_all}"
            logger.error(f"✗ FAIL: {result['error']}")
            logger.error(f"  Required (ALL): {REQUIRED_SYMBOLS_ALL}")
            logger.error(f"  Missing: {missing_all}")
            return result
        
        logger.info(f"✓ ALL group present: {REQUIRED_SYMBOLS_ALL}")
        
        # Validate VIX ANY group (^VIX, VIXY, or VXX)
        present_vix = [sym for sym in REQUIRED_SYMBOLS_VIX_ANY if sym in symbols_in_cache]
        result['present_vix_group'] = present_vix
        
        if not present_vix:
            result['error'] = f"VIX ANY group validation failed - none of {REQUIRED_SYMBOLS_VIX_ANY} present"
            logger.error(f"✗ FAIL: {result['error']}")
            logger.error(f"  Required (ANY): {REQUIRED_SYMBOLS_VIX_ANY}")
            logger.error(f"  Present: none")
            return result
        
        logger.info(f"✓ VIX ANY group present: {present_vix} (from {REQUIRED_SYMBOLS_VIX_ANY})")
        
        # Validate T-bill ANY group (BIL or SHY)
        present_tbill = [sym for sym in REQUIRED_SYMBOLS_TBILL_ANY if sym in symbols_in_cache]
        result['present_tbill_group'] = present_tbill
        
        if not present_tbill:
            result['error'] = f"T-bill ANY group validation failed - none of {REQUIRED_SYMBOLS_TBILL_ANY} present"
            logger.error(f"✗ FAIL: {result['error']}")
            logger.error(f"  Required (ANY): {REQUIRED_SYMBOLS_TBILL_ANY}")
            logger.error(f"  Present: none")
            return result
        
        logger.info(f"✓ T-bill ANY group present: {present_tbill} (from {REQUIRED_SYMBOLS_TBILL_ANY})")
        
        # All validations passed
        result['valid'] = True
        logger.info("✓ PASS: All required symbols present")
        logger.info("=" * 70)
        
        return result
        
    except Exception as e:
        result['error'] = f"Error validating required symbols: {e}"
        logger.error(f"✗ FAIL: {result['error']}")
        return result


def validate_cache_integrity(cache_path: str) -> Dict[str, Any]:
    """
    Validate cache file integrity.
    
    Checks:
    - File exists
    - File size > 0
    - Symbol count > 0
    
    Args:
        cache_path: Path to the cache parquet file
        
    Returns:
        Dictionary with:
        - valid: bool - Whether validation passed
        - file_exists: bool - File exists
        - file_size_bytes: int - File size in bytes
        - symbol_count: int - Number of symbols
        - error: str or None - Error message if validation failed
    """
    result = {
        'valid': False,
        'file_exists': False,
        'file_size_bytes': 0,
        'symbol_count': 0,
        'error': None
    }
    
    logger.info("=" * 70)
    logger.info("CACHE INTEGRITY VALIDATION")
    logger.info("=" * 70)
    
    # Check file exists
    result['file_exists'] = os.path.exists(cache_path)
    logger.info(f"File exists: {result['file_exists']}")
    
    if not result['file_exists']:
        result['error'] = f"Cache file does not exist: {cache_path}"
        logger.error(f"✗ FAIL: {result['error']}")
        return result
    
    # Check file size
    try:
        file_size = os.path.getsize(cache_path)
        result['file_size_bytes'] = file_size
        logger.info(f"File size: {file_size:,} bytes ({file_size / (1024*1024):.2f} MB)")
        
        if file_size == 0:
            result['error'] = "Cache file is empty (0 bytes)"
            logger.error(f"✗ FAIL: {result['error']}")
            return result
        
    except Exception as e:
        result['error'] = f"Error checking file size: {e}"
        logger.error(f"✗ FAIL: {result['error']}")
        return result
    
    # Check symbol count
    try:
        cache_df = pd.read_parquet(cache_path)
        
        if cache_df.empty:
            result['error'] = "Cache dataframe is empty"
            logger.error(f"✗ FAIL: {result['error']}")
            return result
        
        result['symbol_count'] = len(cache_df.columns)
        logger.info(f"Symbol count: {result['symbol_count']}")
        
        if result['symbol_count'] == 0:
            result['error'] = "Cache has no symbols"
            logger.error(f"✗ FAIL: {result['error']}")
            return result
        
    except Exception as e:
        result['error'] = f"Error reading cache: {e}"
        logger.error(f"✗ FAIL: {result['error']}")
        return result
    
    # All checks passed
    result['valid'] = True
    logger.info("✓ PASS: Cache integrity verified")
    logger.info("=" * 70)
    
    return result


def check_for_changes(repo_path: str = ".") -> Dict[str, Any]:
    """
    Check if there are uncommitted changes in the git repository.
    
    Args:
        repo_path: Path to the git repository (default: current directory)
        
    Returns:
        Dictionary with:
        - has_changes: bool - Whether there are uncommitted changes
        - git_status: str - Output of git status
        - git_diff_stat: str - Output of git diff --stat
    """
    import subprocess
    
    result = {
        'has_changes': False,
        'git_status': '',
        'git_diff_stat': ''
    }
    
    try:
        # Get git status
        status_output = subprocess.check_output(
            ['git', 'status', '--porcelain'],
            cwd=repo_path,
            text=True,
            stderr=subprocess.STDOUT
        )
        result['git_status'] = status_output.strip()
        
        # Get git diff stat
        diff_output = subprocess.check_output(
            ['git', 'diff', '--stat'],
            cwd=repo_path,
            text=True,
            stderr=subprocess.STDOUT
        )
        result['git_diff_stat'] = diff_output.strip()
        
        # Check if there are changes
        result['has_changes'] = bool(result['git_status'])
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Error checking git changes: {e}")
    except Exception as e:
        logger.error(f"Unexpected error checking git changes: {e}")
    
    return result


def validate_no_change_logic(
    cache_freshness_valid: bool,
    has_changes: bool
) -> Dict[str, Any]:
    """
    Validate the no-change logic for the pipeline.
    
    Rules:
    - Fresh + unchanged → SUCCESS (no commit needed)
    - Fresh + changed → SUCCESS (commit changes)
    - Stale + unchanged → FAIL (data is stale and wasn't updated)
    - Stale + changed → SUCCESS (commit updated data)
    
    Args:
        cache_freshness_valid: Whether cache is fresh (from validate_trading_day_freshness)
        has_changes: Whether there are git changes
        
    Returns:
        Dictionary with:
        - should_commit: bool - Whether to commit changes
        - should_succeed: bool - Whether workflow should succeed
        - message: str - Human-readable message
    """
    result = {
        'should_commit': False,
        'should_succeed': False,
        'message': ''
    }
    
    logger.info("=" * 70)
    logger.info("NO-CHANGE LOGIC VALIDATION")
    logger.info("=" * 70)
    logger.info(f"Cache fresh: {cache_freshness_valid}")
    logger.info(f"Has changes: {has_changes}")
    
    if cache_freshness_valid and not has_changes:
        # Fresh + unchanged → SUCCESS (no commit)
        result['should_commit'] = False
        result['should_succeed'] = True
        result['message'] = "Fresh but unchanged — no commit needed"
        logger.info(f"✓ {result['message']}")
    elif cache_freshness_valid and has_changes:
        # Fresh + changed → SUCCESS (commit)
        result['should_commit'] = True
        result['should_succeed'] = True
        result['message'] = "Fresh and changed — committing updates"
        logger.info(f"✓ {result['message']}")
    elif not cache_freshness_valid and not has_changes:
        # Stale + unchanged → FAIL
        result['should_commit'] = False
        result['should_succeed'] = False
        result['message'] = "Stale + unchanged"
        logger.error(f"✗ FAIL: {result['message']}")
    else:
        # Stale + changed → SUCCESS (commit)
        result['should_commit'] = True
        result['should_succeed'] = True
        result['message'] = "Stale but changed — committing updates"
        logger.info(f"✓ {result['message']}")
    
    logger.info("=" * 70)
    
    return result
