import os
import pandas as pd
from datetime import datetime, timedelta

# Required symbol groups for validation
REQUIRED_SYMBOLS_ALL = ['SPY', 'QQQ', 'IWM']  # All must be present
REQUIRED_SYMBOLS_VIX_ANY = ['^VIX', 'VIXY', 'VXX']  # At least one must be present
REQUIRED_SYMBOLS_TBILL_ANY = ['BIL', 'SHY']  # At least one must be present


def fetch_spy_trading_days(calendar_days=10):
    """
    Fetch SPY prices to determine recent trading days.
    
    Args:
        calendar_days: Number of calendar days to look back
        
    Returns:
        Tuple of (last_trading_day: datetime, trading_days: list)
    """
    try:
        import yfinance as yf
        end_date = datetime.now()
        start_date = end_date - timedelta(days=calendar_days)
        spy = yf.download('SPY', start=start_date, end=end_date, progress=False)
        
        if spy.empty:
            return None, []
        
        trading_days = spy.index.tolist()
        # Convert to datetime if they're Timestamp objects
        trading_days = [pd.Timestamp(d).to_pydatetime() for d in trading_days]
        last_trading_day = trading_days[-1] if trading_days else None
        
        return last_trading_day, trading_days
    except Exception:
        # Fallback: return None, []
        return None, []


def get_cache_max_date(cache_path):
    """
    Get the maximum date from the cache file.
    
    Args:
        cache_path: Path to cache parquet file
        
    Returns:
        Maximum date in cache (as datetime) or None if cache doesn't exist
    """
    if not os.path.exists(cache_path):
        return None
    
    try:
        cache = pd.read_parquet(cache_path)
        if cache.empty:
            return None
        max_date = cache.index.max()
        # Convert to datetime if it's a Timestamp
        if hasattr(max_date, 'to_pydatetime'):
            return max_date.to_pydatetime()
        return max_date
    except Exception:
        return None


def validate_trading_day_freshness(cache_path, max_market_feed_gap_days=5):
    """
    Validate that cache is up-to-date with latest trading day.
    
    Args:
        cache_path: Path to cache file
        max_market_feed_gap_days: Maximum allowed gap in days for market data feed
        
    Returns:
        Dictionary with validation results
    """
    result = {
        'valid': False,
        'error': None,
        'cache_max_date': None,
        'last_trading_day': None,
        'delta_days': None,
        'market_feed_gap_days': None,
        'today': None,
    }
    
    result['today'] = datetime.now()
    
    if not os.path.exists(cache_path):
        result['error'] = "Cache file missing"
        return result
    
    try:
        # Get cache max date
        cache_max_date = get_cache_max_date(cache_path)
        if cache_max_date is None:
            result['error'] = "Cache is empty"
            return result
        
        result['cache_max_date'] = cache_max_date
        
        # Fetch SPY trading days
        last_trading_day, _ = fetch_spy_trading_days(calendar_days=10)
        
        if last_trading_day is None:
            # Fallback to simple date comparison if SPY fetch fails
            today = datetime.now()
            delta = today - cache_max_date
            result['delta_days'] = delta.days
            
            if delta.days > max_market_feed_gap_days:
                result['error'] = f"Cache is stale. Latest date is {cache_max_date.date()}, {delta.days} days old."
            else:
                result['valid'] = True
            return result
        
        result['last_trading_day'] = last_trading_day
        
        # Calculate delta between cache and last trading day
        delta = cache_max_date - last_trading_day
        result['delta_days'] = delta.days
        
        # Calculate market feed gap
        today = datetime.now()
        market_feed_gap = (today - last_trading_day).days
        result['market_feed_gap_days'] = market_feed_gap
        
        # Validate
        if market_feed_gap > max_market_feed_gap_days:
            result['error'] = f"Market data feed likely broken (last trading day is {market_feed_gap} days old)"
            return result
        
        # Cache should be at or within 1 trading session of last trading day
        if delta.days <= 0 and abs(delta.days) <= 1:
            result['valid'] = True
        else:
            result['error'] = f"Cache is {abs(delta.days)} trading sessions behind"
        
        return result
        
    except Exception as e:
        result['error'] = f"Validation error: {e}"
        return result


def validate_required_tickers(cache, required_tickers):
    """
    Validate that required tickers are present in cache.
    
    Args:
        cache: DataFrame with ticker columns
        required_tickers: List of required ticker symbols
        
    Returns:
        Tuple of (is_valid: bool, message: str)
    """
    missing = [ticker for ticker in required_tickers if ticker not in cache.columns]
    if missing:
        return False, f"Missing tickers: {missing}"
    return True, "All required tickers are present."


def validate_required_symbols(cache_path):
    """
    Validate required symbols with ALL/ANY group semantics.
    
    Args:
        cache_path: Path to cache file
        
    Returns:
        Dictionary with validation results
    """
    result = {
        'valid': False,
        'error': None,
        'missing_all': [],
        'missing_all_group': [],
        'vix_present': False,
        'tbill_present': False,
        'present_vix_group': [],
        'present_tbill_group': [],
        'symbols_in_cache': [],
    }
    
    if not os.path.exists(cache_path):
        result['error'] = "Cache file missing"
        return result
    
    try:
        cache = pd.read_parquet(cache_path)
        columns = cache.columns.tolist()
        result['symbols_in_cache'] = columns
        
        # Check ALL group (all must be present)
        missing_all = [sym for sym in REQUIRED_SYMBOLS_ALL if sym not in columns]
        result['missing_all'] = missing_all
        result['missing_all_group'] = missing_all
        if missing_all:
            result['error'] = f"Missing required symbols from ALL group: {missing_all}"
            return result
        
        # Check VIX ANY group (at least one must be present)
        present_vix = [sym for sym in REQUIRED_SYMBOLS_VIX_ANY if sym in columns]
        result['present_vix_group'] = present_vix
        result['vix_present'] = len(present_vix) > 0
        
        if not result['vix_present']:
            result['error'] = f"No VIX symbols found from VIX ANY group. Need at least one of: {REQUIRED_SYMBOLS_VIX_ANY}"
            return result
        
        # Check T-bill ANY group (at least one must be present)
        present_tbill = [sym for sym in REQUIRED_SYMBOLS_TBILL_ANY if sym in columns]
        result['present_tbill_group'] = present_tbill
        result['tbill_present'] = len(present_tbill) > 0
        
        if not result['tbill_present']:
            result['error'] = f"No T-bill symbols found from T-bill ANY group. Need at least one of: {REQUIRED_SYMBOLS_TBILL_ANY}"
            return result
        
        result['valid'] = True
        return result
    except Exception as e:
        result['error'] = f"Error validating symbols: {e}"
        return result


def validate_cache_integrity(cache_path):
    """
    Validate cache file exists, has size > 0, and has symbols.
    
    Args:
        cache_path: Path to cache file
        
    Returns:
        Dictionary with validation results
    """
    result = {
        'valid': False,
        'error': None,
        'file_exists': False,
        'file_size': 0,
        'file_size_bytes': 0,
        'num_rows': 0,
        'num_symbols': 0,
        'symbol_count': 0,
    }
    
    if not os.path.exists(cache_path):
        result['error'] = "Cache file does not exist"
        return result
    
    result['file_exists'] = True
    file_size = os.path.getsize(cache_path)
    result['file_size'] = file_size
    result['file_size_bytes'] = file_size
    
    if file_size == 0:
        result['error'] = "Cache file is empty (0 bytes)"
        return result
    
    try:
        cache = pd.read_parquet(cache_path)
        result['num_rows'] = len(cache)
        result['num_symbols'] = len(cache.columns)
        result['symbol_count'] = len(cache.columns)
        
        if cache.empty:
            result['error'] = "Cache has no data"
            return result
        if len(cache.columns) == 0:
            result['error'] = "Cache has no symbol columns"
            return result
        
        result['valid'] = True
        return result
    except Exception as e:
        result['error'] = f"Error reading cache: {e}"
        return result


def check_for_changes():
    """
    Check if there are git changes (stub for compatibility).
    
    Returns:
        bool: True if changes detected
    """
    # This is a stub - real implementation would check git status
    return False


def validate_no_change_logic(cache_freshness_valid, has_changes):
    """
    Determine commit/success logic based on cache state and changes.
    
    Args:
        cache_freshness_valid: Whether cache is fresh
        has_changes: Whether there are changes to commit
        
    Returns:
        Dictionary with 'should_commit', 'should_succeed', and 'message' keys
    """
    if cache_freshness_valid and not has_changes:
        return {
            'should_commit': False,
            'should_succeed': True,
            'message': 'Fresh but unchanged — no commit needed'
        }
    elif cache_freshness_valid and has_changes:
        return {
            'should_commit': True,
            'should_succeed': True,
            'message': 'Fresh and changed — committing updates'
        }
    elif not cache_freshness_valid and not has_changes:
        return {
            'should_commit': False,
            'should_succeed': False,
            'message': 'Stale + unchanged'
        }
    else:  # not fresh and has changes
        return {
            'should_commit': True,
            'should_succeed': True,
            'message': 'Stale but changed — committing updates'
        }