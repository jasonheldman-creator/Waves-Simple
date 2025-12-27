"""
V3 ADD-ON: Bottom Ticker (Institutional Rail) - Data Sources
Handles all data fetching for the bottom ticker with exception handling and fallbacks.
Enhanced with circuit breaker and persistent cache for resilience.
"""

import os
import json
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Set, Any
import streamlit as st

# Import circuit breaker and persistent cache
try:
    from .circuit_breaker import get_circuit_breaker
    from .persistent_cache import get_persistent_cache
    from .resilient_call import call_with_retry
    RESILIENCE_AVAILABLE = True
except ImportError:
    RESILIENCE_AVAILABLE = False


# ============================================================================
# SECTION 1: Holdings Data Extraction
# ============================================================================

@st.cache_data(ttl=300)
def get_wave_holdings_tickers(max_tickers: int = 60, top_n_per_wave: int = 5) -> List[str]:
    """
    Extract holdings from wave position files to create ticker universe.
    
    Args:
        max_tickers: Maximum number of unique tickers to return
        top_n_per_wave: Number of top holdings to extract per wave
    
    Returns:
        List of unique ticker symbols (up to max_tickers)
    """
    try:
        base_dir = os.path.dirname(os.path.dirname(__file__))
        ticker_set: Set[str] = set()
        
        # List of wave position files to check
        wave_files = [
            'Growth_Wave_positions_20251206.csv',
            'SP500_Wave_positions_20251206.csv',
            # Add more wave position files as they become available
        ]
        
        for wave_file in wave_files:
            try:
                file_path = os.path.join(base_dir, wave_file)
                if os.path.exists(file_path):
                    df = pd.read_csv(file_path)
                    
                    if 'Ticker' in df.columns:
                        # Get unique tickers (since files have duplicates)
                        unique_tickers = df['Ticker'].dropna().unique()
                        
                        # Get top N tickers by weight if available
                        if 'TargetWeight' in df.columns:
                            wave_df = df.drop_duplicates(subset=['Ticker'])
                            top_tickers = wave_df.nlargest(top_n_per_wave, 'TargetWeight')['Ticker'].tolist()
                            ticker_set.update(top_tickers)
                        else:
                            # Just take first N unique tickers
                            ticker_set.update(list(unique_tickers[:top_n_per_wave]))
                        
            except Exception:
                # Skip this wave file if there's an error
                continue
        
        # Fallback to Master_Stock_Sheet if no wave holdings found
        if len(ticker_set) == 0:
            try:
                master_sheet_path = os.path.join(base_dir, 'Master_Stock_Sheet.csv')
                if os.path.exists(master_sheet_path):
                    df = pd.read_csv(master_sheet_path)
                    if 'Ticker' in df.columns:
                        if 'Weight' in df.columns:
                            top_tickers = df.nlargest(top_n_per_wave * 2, 'Weight')['Ticker'].dropna().tolist()
                            ticker_set.update(top_tickers)
                        else:
                            ticker_set.update(df['Ticker'].dropna().head(top_n_per_wave * 2).tolist())
            except Exception:
                pass
        
        # Final fallback: default array
        if len(ticker_set) == 0:
            ticker_set = {'AAPL', 'MSFT', 'NVDA', 'GOOGL', 'META', 'TSLA', 'JPM', 'V', 'WMT', 'JNJ'}
        
        # Convert to list and limit to max_tickers
        ticker_list = list(ticker_set)[:max_tickers]
        
        return ticker_list
        
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


@st.cache_data(ttl=600)  # Increased TTL to 10 minutes to reduce API stress
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

@st.cache_data(ttl=3600)
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

@st.cache_data(ttl=86400)
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
    
    Returns:
        Dict with health metrics including circuit breaker status and cache stats
    """
    # Default health status (fail-safe fallback)
    health = {
        'timestamp': datetime.now().isoformat(),
        'resilience_available': RESILIENCE_AVAILABLE,
        'circuit_breakers': {},
        'cache_stats': {},
        'overall_status': 'unknown'
    }
    
    try:
        # Update timestamp
        health['timestamp'] = datetime.now().isoformat()
        
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
                
                # Set status based on circuit breaker state
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
            # Resilience features not available - assume healthy
            health['overall_status'] = 'healthy'
    
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
