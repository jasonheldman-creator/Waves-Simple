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


# ============================================================================
# QUICK FIX: Normalize + block known-bad tickers
# ============================================================================

# Permanently ignore these (user request: delete forever)
BLOCKLIST_TICKERS: Set[str] = {
    "COMP-USD",
    "ALT-USD",
    "IMX-USD",
    "MNT-USD",
    "TAO-USD",
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


# Create a conditional caching decorator
def conditional_cache(ttl=300):
    """
    Decorator that uses Streamlit caching if available, otherwise no-op.
    """
    def decorator(func):
        if STREAMLIT_AVAILABLE:
            return st.cache_data(ttl=ttl)(func)
        return func
    return decorator


# ============================================================================
# SECTION 1: Holdings Data Extraction
# ============================================================================

@conditional_cache(ttl=300)
def get_wave_holdings_tickers(
    max_tickers: int = 60,
    top_n_per_wave: int = 5,
    active_waves_only: bool = True
) -> List[str]:
    """
    Extract holdings from canonical universal universe file.

    This function now uses universal_universe.csv as the SINGLE SOURCE OF TRUTH
    for all ticker references across the platform.

    Enhanced to filter tickers by active wave membership to prevent inactive wave
    tickers from affecting system health status.
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
                    active_wave_ids = set(
                        wave_registry_df[wave_registry_df['active'] == True]['wave_id'].tolist()
                    )
                    logger.info(f"Found {len(active_wave_ids)} active waves for filtering")
            except Exception as e:
                logger.warning(f"Could not load active wave list, using all waves: {str(e)}")
                active_waves_only = False

        # PRIMARY SOURCE: universal_universe.csv (CANONICAL)
        universe_path = os.path.join(base_dir, 'universal_universe.csv')
        if os.path.exists(universe_path):
            try:
                df = pd.read_csv(universe_path)

                # Filter to active tickers only (if column exists)
                if 'status' in df.columns:
                    df = df[df['status'] == 'active']

                if 'ticker' in df.columns:
                    # Filter by active waves if requested
                    if active_waves_only and active_wave_ids and 'index_membership' in df.columns:
                        def belongs_to_active_wave(index_membership):
                            if pd.isna(index_membership):
                                return False
                            memberships = str(index_membership).upper().split(',')
                            for membership in memberships:
                                membership = membership.strip()
                                if membership.startswith('WAVE_'):
                                    wave_id = membership.replace('WAVE_', '').lower()
                                    import re
                                    wave_id = re.sub(r'[&\-\s]+', '_', wave_id)
                                    wave_id = re.sub(r'_+', '_', wave_id)
                                    wave_id = wave_id.rstrip('_')
                                    if wave_id in active_wave_ids:
                                        return True
                            return False

                        df = df[df['index_membership'].apply(belongs_to_active_wave)]
                        logger.info("Filtered to tickers from active waves only")

                    # Prefer wave tickers first
                    if 'index_membership' in df.columns:
                        wave_tickers = df[
                            df['index_membership'].str.contains('WAVE_', case=False, na=False)
                        ]['ticker'].dropna().unique().tolist()
                    else:
                        wave_tickers = []

                    # Normalize + blocklist
                    wave_tickers = [normalize_ticker(t) for t in wave_tickers]
                    wave_tickers = [t for t in wave_tickers if t]

                    if wave_tickers:
                        tickers = wave_tickers[:max_tickers] if max_tickers else wave_tickers
                        logger.info(f"Loaded {len(tickers)} tickers from universal universe (Wave-prioritized)")
                        return tickers

                    # Otherwise, all active tickers
                    all_tickers = df['ticker'].dropna().unique().tolist()
                    all_tickers = [normalize_ticker(t) for t in all_tickers]
                    all_tickers = [t for t in all_tickers if t]
                    tickers = all_tickers[:max_tickers] if max_tickers else all_tickers
                    logger.info(f"Loaded {len(tickers)} tickers from universal universe")
                    return tickers

            except Exception as e:
                logger.warning(f"Error reading universal_universe.csv: {str(e)}")
        else:
            logger.warning(f"universal_universe.csv not found at {universe_path}")

        # LAST RESORT default tickers
        default_tickers = ['AAPL', 'MSFT', 'NVDA', 'GOOGL', 'META', 'TSLA', 'JPM', 'V', 'WMT', 'JNJ']
        return default_tickers

    except Exception:
        return ['AAPL', 'MSFT', 'NVDA', 'GOOGL', 'META', 'TSLA', 'JPM', 'V', 'WMT', 'JNJ']


# ============================================================================
# SECTION 2: Market Price Data
# ============================================================================

def _fetch_ticker_price_data_internal(ticker: str) -> Dict[str, Optional[float]]:
    """
    Internal function to fetch ticker price data from yfinance.
    Attempts once per call - no retries to prevent spinner issues.
    """
    import yfinance as yf

    ticker = normalize_ticker(ticker)
    if ticker is None:
        return {'price': None, 'change_pct': None, 'success': False}

    try:
        stock = yf.Ticker(ticker)
        info = stock.info

        if info and 'currentPrice' in info:
            current_price = info.get('currentPrice')
            previous_close = info.get('previousClose')

            if current_price and previous_close:
                change_pct = ((current_price - previous_close) / previous_close) * 100
                return {'price': current_price, 'change_pct': change_pct, 'success': True}

        hist = stock.history(period='2d')
        if not hist.empty and len(hist) >= 2:
            current_price = float(hist['Close'].iloc[-1])
            previous_price = float(hist['Close'].iloc[-2])
            if previous_price != 0:
                change_pct = ((current_price - previous_price) / previous_price) * 100
            else:
                change_pct = 0.0
            return {'price': current_price, 'change_pct': change_pct, 'success': True}

    except Exception:
        pass

    return {'price': None, 'change_pct': None, 'success': False}


@conditional_cache(ttl=600)
def get_ticker_price_data(ticker: str) -> Dict[str, Optional[float]]:
    """
    Get current price and daily % change for a ticker using yfinance.
    Enhanced with circuit breaker and persistent cache for resilience.
    Increased TTL and improved caching to reduce provider stress.
    """
    ticker = normalize_ticker(ticker)
    if ticker is None:
        return {'price': None, 'change_pct': None, 'success': False}

    failure_response = {'price': None, 'change_pct': None, 'success': False}

    # Try persistent cache first if available
    if RESILIENCE_AVAILABLE:
        try:
            cache = get_persistent_cache()
            cache_key = f"ticker_price:{ticker}"
            cached_data = cache.get(cache_key)
            if cached_data is not None:
                return cached_data
        except Exception:
            pass

    if RESILIENCE_AVAILABLE:
        try:
            cb = get_circuit_breaker("yfinance_ticker", failure_threshold=5, recovery_timeout=60)
            success, result, _error = cb.call(_fetch_ticker_price_data_internal, ticker)

            if success and result:
                try:
                    cache = get_persistent_cache()
                    cache.set(f"ticker_price:{ticker}", result, ttl=600)
                except Exception:
                    pass
                return result

            return failure_response

        except Exception:
            return failure_response

    # Fallback without resilience
    try:
        return _fetch_ticker_price_data_internal(ticker)
    except Exception:
        return failure_response


# ============================================================================
# SECTION 3: Earnings Data
# ============================================================================

@conditional_cache(ttl=3600)
def get_earnings_date(ticker: str) -> Optional[str]:
    """
    Get next earnings date for a ticker using yfinance.
    """
    ticker = normalize_ticker(ticker)
    if ticker is None:
        return None

    try:
        import yfinance as yf

        stock = yf.Ticker(ticker)
        calendar = stock.calendar

        if calendar is not None and not calendar.empty:
            if 'Earnings Date' in calendar.index:
                earnings_date = calendar.loc['Earnings Date']
                if pd.notna(earnings_date):
                    if hasattr(earnings_date, 'strftime'):
                        return earnings_date.strftime('%Y-%m-%d')
                    if isinstance(earnings_date, str):
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
    """
    try:
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

        now = datetime.now()
        next_date = None
        for date in fomc_dates:
            if date > now:
                next_date = date
                break

        current_rate = "4.25-4.50%"
        cpi_latest = "Dec 2024"
        jobs_latest = "Dec 2024"

        return {
            'fed_funds_rate': current_rate,
            'next_fomc_date': next_date.strftime('%Y-%m-%d') if next_date else None,
            'cpi_latest': cpi_latest,
            'jobs_latest': jobs_latest
        }

    except Exception:
        return {'fed_funds_rate': "N/A", 'next_fomc_date': None, 'cpi_latest': "N/A", 'jobs_latest': "N/A"}


# ============================================================================
# SECTION 5: WAVES Internal Status
# ============================================================================

def get_waves_status() -> Dict[str, str]:
    """
    Get WAVES system internal status indicators.
    """
    try:
        current_time = datetime.now().strftime('%H:%M:%S')
        if STREAMLIT_AVAILABLE:
            waves_loaded = "ACTIVE" if st.session_state.get("wave_universe") else "LOADING"
        else:
            waves_loaded = "N/A"

        return {'system_status': 'ONLINE', 'last_update': current_time, 'waves_status': waves_loaded}

    except Exception:
        return {'system_status': 'ONLINE', 'last_update': 'N/A', 'waves_status': 'N/A'}


# ============================================================================
# SECTION 6: Cache Management
# ============================================================================

def load_events_cache() -> Dict:
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
    try:
        base_dir = os.path.dirname(os.path.dirname(__file__))
        cache_path = os.path.join(base_dir, 'data', 'events_cache.json')
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, 'w') as f:
            json.dump(cache_data, f, indent=2)
        return True
    except Exception:
        return False


def update_cache_with_current_data() -> None:
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
    """
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
        health['timestamp'] = datetime.now().isoformat()

        if RESILIENCE_AVAILABLE:
            try:
                from .circuit_breaker import get_all_circuit_states
                health['circuit_breakers'] = get_all_circuit_states()

                open_count = 0
                for _name, state in health['circuit_breakers'].items():
                    if isinstance(state, dict) and state.get('state') == 'open':
                        open_count += 1

                health['overall_status'] = 'degraded' if open_count > 0 else 'healthy'
            except Exception as e:
                health['circuit_breakers'] = {'error': str(e)}
                health['overall_status'] = 'unknown'

            try:
                cache = get_persistent_cache()
                health['cache_stats'] = cache.get_stats()
            except Exception as e:
                health['cache_stats'] = {'error': str(e)}
        else:
            health['overall_status'] = 'healthy'

        # Count active wave tickers (best effort)
        try:
            from helpers.price_loader import collect_required_tickers
            active_wave_tickers = collect_required_tickers(active_only=True)
            health['active_wave_ticker_count'] = len(active_wave_tickers)
        except Exception:
            pass

    except Exception as e:
        health['overall_status'] = 'unknown'
        health['error'] = str(e)

    return health


def test_ticker_fetch(ticker: str = "AAPL") -> Dict[str, Any]:
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
        result['success'] = bool(data.get('success', False))
    except Exception as e:
        result['error'] = str(e)
        result['success'] = False

    return result