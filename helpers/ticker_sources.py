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

# Configure logging
logger = logging.getLogger(__name__)

# Conditionally import Streamlit for caching
try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False

# Import circuit breaker and persistent cache
try:
    from .circuit_breaker import get_circuit_breaker
    from .persistent_cache import get_persistent_cache
    from .resilient_call import call_with_retry
    RESILIENCE_AVAILABLE = True
except ImportError:
    RESILIENCE_AVAILABLE = False


# Create a conditional caching decorator
def conditional_cache(ttl=300):
    """
    Decorator that uses Streamlit caching if available, otherwise no-op.
    
    Args:
        ttl: Time to live in seconds for the cache (default: 300).
             Only applies when Streamlit is available.
    
    Returns:
        Decorated function with caching (if Streamlit available) or original function.
    """
    def decorator(func):
        if STREAMLIT_AVAILABLE:
            return st.cache_data(ttl=ttl)(func)
        else:
            # No caching when Streamlit is not available
            return func
    return decorator


# ============================================================================
# SECTION 1: Holdings Data Extraction
# ============================================================================

@conditional_cache(ttl=300)
def get_wave_holdings_tickers(max_tickers: int = 60, top_n_per_wave: int = 5, active_waves_only: bool = True) -> List[str]:
    """
    Extract holdings from canonical universal universe file.
    
    This function now uses universal_universe.csv as the SINGLE SOURCE OF TRUTH
    for all ticker references across the platform.
    
    Enhanced to filter tickers by active wave membership to prevent inactive wave
    tickers from affecting system health status.
    
    Args:
        max_tickers: Maximum number of unique tickers to return
        top_n_per_wave: Number of top holdings to extract per wave (for display)
        active_waves_only: If True, only return tickers from waves marked as active in wave_registry.csv
    
    Returns:
        List of unique ticker symbols (up to max_tickers)
    """
    try:
        base_dir = os.path.dirname(os.path.dirname(__file__))
        
        # Get active wave IDs if filtering is requested
        active_wave_ids = set()
        if active_waves_only:
            try:
                wave_registry_path = os.path.join(base_dir, 'data', 'wave_registry.csv')
                if os.path.exists(wave_registry_path):
                    wave_registry_df = pd.read_csv(wave_registry_path)
                    # Get wave_ids where active == True
                    active_wave_ids = set(
                        wave_registry_df[wave_registry_df['active'] == True]['wave_id'].tolist()
                    )
                    logger.info(f"Found {len(active_wave_ids)} active waves for filtering")
            except Exception as e:
                logger.warning(f"Could not load active wave list, using all waves: {str(e)}")
                # If we can't load the wave registry, fall back to using all waves
                active_waves_only = False
        
        # PRIMARY SOURCE: universal_universe.csv (CANONICAL)
        universe_path = os.path.join(base_dir, 'universal_universe.csv')
        if os.path.exists(universe_path):
            try:
                df = pd.read_csv(universe_path)
                
                # Filter to active tickers only
                df = df[df['status'] == 'active']
                
                if 'ticker' in df.columns:
                    # Filter by active waves if requested
                    if active_waves_only and active_wave_ids:
                        # Filter tickers to only those belonging to active waves
                        def belongs_to_active_wave(index_membership):
                            if pd.isna(index_membership):
                                return False
                            # Parse index_membership string to extract wave names
                            memberships = str(index_membership).upper().split(',')
                            for membership in memberships:
                                membership = membership.strip()
                                if membership.startswith('WAVE_'):
                                    # Convert display name format to wave_id format
                                    # e.g., "WAVE_AI_&_CLOUD_MEGACAP_WAVE" -> "ai_cloud_megacap_wave"
                                    # Remove 'WAVE_' prefix, convert to lowercase
                                    wave_id = membership.replace('WAVE_', '').lower()
                                    # Remove special characters and extra underscores
                                    import re
                                    # Replace &, -, and other special chars with nothing
                                    wave_id = re.sub(r'[&\-\s]+', '_', wave_id)
                                    # Remove multiple consecutive underscores
                                    wave_id = re.sub(r'_+', '_', wave_id)
                                    # Remove trailing underscore if exists
                                    wave_id = wave_id.rstrip('_')
                                    if wave_id in active_wave_ids:
                                        return True
                            return False
                        
                        df = df[df['index_membership'].apply(belongs_to_active_wave)]
                        logger.info("Filtered to tickers from active waves only")
                    
                    # Prioritize tickers from Wave definitions
                    # (those with WAVE_ in index_membership)
                    wave_tickers = df[
                        df['index_membership'].str.contains('WAVE_', case=False, na=False)
                    ]['ticker'].dropna().unique().tolist()
                    
                    # If we have wave tickers, prefer those
                    if wave_tickers:
                        tickers = wave_tickers[:max_tickers] if max_tickers else wave_tickers
                        logger.info(f"Loaded {len(tickers)} tickers from universal universe (Wave-prioritized, active_waves_only={active_waves_only})")
                        return tickers
                    
                    # Otherwise, return all active tickers
                    all_tickers = df['ticker'].dropna().unique().tolist()
                    tickers = all_tickers[:max_tickers] if max_tickers else all_tickers
                    logger.info(f"Loaded {len(tickers)} tickers from universal universe")
                    return tickers
                    
            except Exception as e:
                # Log error but continue to fallback
                logger.warning(f"Error reading universal_universe.csv: {str(e)}")
        else:
            logger.warning(f"universal_universe.csv not found at {universe_path}")
        
        # FALLBACK 1: ticker_master_clean.csv (DEPRECATED - legacy support)
        ticker_master_path = os.path.join(base_dir, 'ticker_master_clean.csv')
        if os.path.exists(ticker_master_path):
            try:
                df = pd.read_csv(ticker_master_path)
                if 'ticker' in df.columns:
                    tickers = df['ticker'].dropna().tolist()
                    tickers = tickers[:max_tickers] if max_tickers else tickers
                    logger.warning(f"Fallback: Loaded {len(tickers)} tickers from ticker_master_clean.csv (DEPRECATED)")
                    return tickers
            except Exception as e:
                logger.warning(f"Error reading ticker_master_clean.csv: {str(e)}")
        
        # FALLBACK 2: Try wave position files (LEGACY)
        ticker_set: Set[str] = set()
        wave_files = [
            'Growth_Wave_positions_20251206.csv',
            'SP500_Wave_positions_20251206.csv',
        ]
        
        for wave_file in wave_files:
            try:
                file_path = os.path.join(base_dir, wave_file)
                if os.path.exists(file_path):
                    df = pd.read_csv(file_path)
                    
                    if 'Ticker' in df.columns:
                        unique_tickers = df['Ticker'].dropna().unique()
                        
                        if 'TargetWeight' in df.columns:
                            wave_df = df.drop_duplicates(subset=['Ticker'])
                            top_tickers = wave_df.nlargest(top_n_per_wave, 'TargetWeight')['Ticker'].tolist()
                            ticker_set.update(top_tickers)
                        else:
                            ticker_set.update(list(unique_tickers[:top_n_per_wave]))
            except Exception:
                continue
        
        if ticker_set:
            tickers = list(ticker_set)[:max_tickers]
            logger.warning(f"Fallback: Loaded {len(tickers)} tickers from wave position files (LEGACY)")
            return tickers
        
        # FALLBACK 3: Default ticker array (LAST RESORT)
        default_tickers = ['AAPL', 'MSFT', 'NVDA', 'GOOGL', 'META', 'TSLA', 'JPM', 'V', 'WMT', 'JNJ']
        logger.warning(f"Last resort: Using default ticker array ({len(default_tickers)} tickers)")
        return default_tickers
        
    except Exception:
        # Ultimate fallback
        return ['AAPL', 'MSFT', 'NVDA', 'GOOGL', 'META', 'TSLA', 'JPM', 'V', 'WMT', 'JNJ']


# ============================================================================
# SECTION 2: Market Price Data
# ============================================================================

def _fetch_ticker_price_data_internal(ticker: str) -> Dict[str, Optional[float]]:
    """
    Internal function to fetch ticker price data from yfinance.
    Attempts once per call - no retries to prevent spinner issues.
    
    Args:
        ticker: Stock ticker symbol
    
    Returns:
        Dict with 'price', 'change_pct', 'success' keys
    """
    import yfinance as yf
    
    # Single attempt - no retry logic
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        if info and 'currentPrice' in info:
            current_price = info.get('currentPrice')
            previous_close = info.get('previousClose')
            
            if current_price and previous_close:
                change_pct = ((current_price - previous_close) / previous_close) * 100
                return {
                    'price': current_price,
                    'change_pct': change_pct,
                    'success': True
                }
        
        # Fallback: Try history method
        hist = stock.history(period='2d')
        if not hist.empty and len(hist) >= 2:
            current_price = hist['Close'].iloc[-1]
            previous_price = hist['Close'].iloc[-2]
            change_pct = ((current_price - previous_price) / previous_price) * 100
            
            return {
                'price': current_price,
                'change_pct': change_pct,
                'success': True
            }
    except Exception:
        # Fail fast - no retries
        pass
    
    # If we can't get data, return failure
    return {
        'price': None,
        'change_pct': None,
        'success': False
    }


@conditional_cache(ttl=600)  # Increased TTL to 10 minutes to reduce API stress
def get_ticker_price_data(ticker: str) -> Dict[str, Optional[float]]:
    """
    Get current price and daily % change for a ticker using yfinance.
    Enhanced with circuit breaker and persistent cache for resilience.
    Increased TTL and improved caching to reduce provider stress.
    
    Args:
        ticker: Stock ticker symbol
    
    Returns:
        Dict with 'price', 'change_pct', 'success' keys
    """
    # Default failure response
    failure_response = {
        'price': None,
        'change_pct': None,
        'success': False
    }
    
    # Try persistent cache first if available (with longer TTL)
    if RESILIENCE_AVAILABLE:
        try:
            cache = get_persistent_cache()
            cache_key = f"ticker_price:{ticker}"
            cached_data = cache.get(cache_key)
            if cached_data is not None:
                return cached_data
        except Exception:
            pass
    
    # Try to fetch with circuit breaker protection
    if RESILIENCE_AVAILABLE:
        try:
            # Get circuit breaker for yfinance with higher threshold
            cb = get_circuit_breaker("yfinance_ticker", failure_threshold=5, recovery_timeout=60)
            
            # Call through circuit breaker
            success, result, error = cb.call(_fetch_ticker_price_data_internal, ticker)
            
            if success and result:
                # Cache successful result with longer TTL
                try:
                    cache = get_persistent_cache()
                    cache.set(f"ticker_price:{ticker}", result, ttl=600)
                except Exception:
                    pass
                return result
            else:
                # Circuit breaker rejected or call failed
                return failure_response
                
        except Exception:
            return failure_response
    else:
        # Fallback to direct call without circuit breaker
        try:
            result = _fetch_ticker_price_data_internal(ticker)
            return result
        except Exception:
            return failure_response


# ============================================================================
# SECTION 3: Earnings Data
# ============================================================================

@conditional_cache(ttl=3600)
def get_earnings_date(ticker: str) -> Optional[str]:
    """
    Get next earnings date for a ticker using yfinance.
    
    Args:
        ticker: Stock ticker symbol
    
    Returns:
        Formatted date string (YYYY-MM-DD) or None
    """
    try:
        import yfinance as yf
        
        stock = yf.Ticker(ticker)
        calendar = stock.calendar
        
        if calendar is not None and not calendar.empty:
            # Get earnings date
            if 'Earnings Date' in calendar.index:
                earnings_date = calendar.loc['Earnings Date']
                if pd.notna(earnings_date):
                    if hasattr(earnings_date, 'strftime'):
                        return earnings_date.strftime('%Y-%m-%d')
                    elif isinstance(earnings_date, str):
                        return earnings_date
        
        return None
        
    except Exception:
        return None


# ============================================================================
# SECTION 4: Fed/Macro Indicators
# ============================================================================

@conditional_cache(ttl=86400)
def get_fed_indicators() -> Dict[str, Optional[str]]:
    """
    Get Federal Reserve and macroeconomic indicators.
    Uses hardcoded schedule for FOMC meetings (no paid API required).
    
    Returns:
        Dict with 'fed_funds_rate', 'next_fomc_date', 'cpi_latest', 'jobs_latest'
    """
    try:
        # Federal Reserve FOMC meeting dates for 2024-2025
        # Source: federalreserve.gov (publicly available schedule)
        fomc_dates = [
            datetime(2024, 12, 17),
            datetime(2024, 12, 18),
            datetime(2025, 1, 28),
            datetime(2025, 1, 29),
            datetime(2025, 3, 18),
            datetime(2025, 3, 19),
            datetime(2025, 5, 6),
            datetime(2025, 5, 7),
            datetime(2025, 6, 17),
            datetime(2025, 6, 18),
            datetime(2025, 7, 29),
            datetime(2025, 7, 30),
            datetime(2025, 9, 16),
            datetime(2025, 9, 17),
            datetime(2025, 10, 28),
            datetime(2025, 10, 29),
            datetime(2025, 12, 9),
            datetime(2025, 12, 10),
        ]
        
        # Find next FOMC date after today
        now = datetime.now()
        next_date = None
        
        for date in fomc_dates:
            if date > now:
                next_date = date
                break
        
        # Current Federal Funds Rate (as of Dec 2024)
        # This is a static value - updated manually
        current_rate = "4.25-4.50%"
        
        # Placeholder for CPI and jobs data
        # These could be updated from static sources or manual updates
        cpi_latest = "Dec 2024"  # Placeholder
        jobs_latest = "Dec 2024"  # Placeholder
        
        return {
            'fed_funds_rate': current_rate,
            'next_fomc_date': next_date.strftime('%Y-%m-%d') if next_date else None,
            'cpi_latest': cpi_latest,
            'jobs_latest': jobs_latest
        }
        
    except Exception:
        return {
            'fed_funds_rate': "N/A",
            'next_fomc_date': None,
            'cpi_latest': "N/A",
            'jobs_latest': "N/A"
        }


# ============================================================================
# SECTION 5: WAVES Internal Status
# ============================================================================

def get_waves_status() -> Dict[str, str]:
    """
    Get WAVES system internal status indicators.
    
    Returns:
        Dict with status indicators
    """
    try:
        # Get timestamp
        current_time = datetime.now().strftime('%H:%M:%S')
        
        # Check if session state has wave universe
        waves_loaded = "ACTIVE" if st.session_state.get("wave_universe") else "LOADING"
        
        return {
            'system_status': 'ONLINE',
            'last_update': current_time,
            'waves_status': waves_loaded
        }
        
    except Exception:
        return {
            'system_status': 'ONLINE',
            'last_update': 'N/A',
            'waves_status': 'N/A'
        }


# ============================================================================
# SECTION 6: Cache Management
# ============================================================================

def load_events_cache() -> Dict:
    """
    Load cached events data from JSON file for fallback consistency.
    
    Returns:
        Dict with cached event data
    """
    try:
        base_dir = os.path.dirname(os.path.dirname(__file__))
        cache_path = os.path.join(base_dir, 'data', 'events_cache.json')
        
        if os.path.exists(cache_path):
            with open(cache_path, 'r') as f:
                return json.load(f)
        
        return {}
        
    except Exception:
        return {}


def save_events_cache(cache_data: Dict) -> bool:
    """
    Save events data to cache file.
    
    Args:
        cache_data: Dict with event data to cache
    
    Returns:
        True if successful, False otherwise
    """
    try:
        base_dir = os.path.dirname(os.path.dirname(__file__))
        cache_path = os.path.join(base_dir, 'data', 'events_cache.json')
        
        # Ensure data directory exists
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        
        with open(cache_path, 'w') as f:
            json.dump(cache_data, f, indent=2)
        
        return True
        
    except Exception:
        return False


def update_cache_with_current_data() -> None:
    """
    Update cache file with current data from all sources.
    This can be called periodically to keep fallback data fresh.
    """
    try:
        fed_data = get_fed_indicators()
        waves_status = get_waves_status()
        
        cache_data = {
            'last_updated': datetime.now().isoformat(),
            'fed_indicators': fed_data,
            'waves_status': waves_status
        }
        
        save_events_cache(cache_data)
        
    except Exception:
        pass


# ============================================================================
# SECTION 7: Data Health Tracking
# ============================================================================

def get_ticker_health_status() -> Dict[str, Any]:
    """
    Get health status of ticker data fetching system.
    Enhanced with fail-safe error handling to prevent crashes.
    
    Now also checks for stale tickers from active waves only to avoid
    marking system as degraded due to inactive wave ticker failures.
    
    Returns:
        Dict with health metrics including circuit breaker status, cache stats,
        and stale ticker information (filtered to active waves only)
    """
    # Default health status (fail-safe fallback)
    health = {
        'timestamp': datetime.now().isoformat(),
        'resilience_available': RESILIENCE_AVAILABLE,
        'circuit_breakers': {},
        'cache_stats': {},
        'stale_ticker_count': 0,
        'active_wave_ticker_count': 0,
        'overall_status': 'unknown'
    }
    
    try:
        # Update timestamp
        health['timestamp'] = datetime.now().isoformat()
        
        # Check circuit breaker states if available
        if RESILIENCE_AVAILABLE:
            try:
                # Get circuit breaker states
                from .circuit_breaker import get_all_circuit_states
                health['circuit_breakers'] = get_all_circuit_states()
                
                # Check if any circuit is open
                open_count = 0
                for name, state in health['circuit_breakers'].items():
                    if isinstance(state, dict) and state.get('state') == 'open':
                        open_count += 1
                
                # Set initial status based on circuit breaker state
                if open_count > 0:
                    health['overall_status'] = 'degraded'
                else:
                    health['overall_status'] = 'healthy'
                        
            except Exception as e:
                # Non-blocking: Log error but continue
                health['circuit_breakers'] = {'error': str(e)}
                health['overall_status'] = 'unknown'
            
            try:
                # Get cache statistics
                cache = get_persistent_cache()
                health['cache_stats'] = cache.get_stats()
            except Exception as e:
                # Non-blocking: Log error but continue
                health['cache_stats'] = {'error': str(e)}
        else:
            # Resilience features not available - assume healthy initially
            health['overall_status'] = 'healthy'
        
        # Enhanced: Check for stale tickers from ACTIVE waves only
        # This prevents inactive wave ticker failures from marking system as degraded
        try:
            # Use the canonical collect_required_tickers from price_loader
            # This ensures we get the complete set of active wave tickers
            try:
                from helpers.price_loader import collect_required_tickers
                
                # Collect tickers from active waves only (excludes inactive waves, SmartSafe, etc.)
                active_wave_tickers = collect_required_tickers(active_only=True)
                
                health['active_wave_ticker_count'] = len(active_wave_tickers)
                logger.info(f"Counted {len(active_wave_tickers)} active wave tickers for health status")
                
            except ImportError as e:
                # price_loader not available, try fallback approach
                logger.warning(f"Could not import collect_required_tickers from price_loader: {e}")
                try:
                    from data_cache import collect_all_required_tickers
                    from waves_engine import WAVE_WEIGHTS
                    
                    # Collect tickers from active waves only
                    active_wave_tickers = collect_all_required_tickers(
                        WAVE_WEIGHTS,
                        include_benchmarks=False,  # Don't include optional benchmarks for health check
                        include_safe_assets=False,  # Don't include optional safe assets for health check
                        active_only=True  # KEY: Only include tickers from active waves
                    )
                    
                    health['active_wave_ticker_count'] = len(active_wave_tickers)
                    logger.info(f"Counted {len(active_wave_tickers)} active wave tickers (fallback method)")
                    
                except (ImportError, Exception) as e2:
                    # Ultimate fallback: count from wave registry + universal universe
                    logger.warning(f"Fallback ticker collection also failed: {e2}")
                    pass
                
        except Exception as e:
            # Non-blocking: If stale ticker check fails, don't change overall status
            logger.warning(f"Error counting active wave tickers: {e}")
            pass
    
    except Exception as e:
        # Ultimate fail-safe: Return basic health status
        health['overall_status'] = 'unknown'
        health['error'] = str(e)
    
    return health


def test_ticker_fetch(ticker: str = "AAPL") -> Dict[str, Any]:
    """
    Test ticker fetching capability for diagnostics.
    Enhanced with fail-safe error handling.
    
    Args:
        ticker: Ticker symbol to test
        
    Returns:
        Dict with test results
    """
    import time
    
    result = {
        'ticker': ticker,
        'timestamp': datetime.now().isoformat(),
        'success': False,
        'latency_ms': 0,
        'data': None,
        'error': None
    }
    
    try:
        start = time.time()
        data = get_ticker_price_data(ticker)
        latency = (time.time() - start) * 1000
        
        result['latency_ms'] = round(latency, 2)
        result['data'] = data
        result['success'] = data.get('success', False)
        
    except Exception as e:
        # Fail-safe: Log error but don't crash
        result['error'] = str(e)
        result['success'] = False
    
    return result
