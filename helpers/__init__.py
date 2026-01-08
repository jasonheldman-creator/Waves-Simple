"""
Helpers Package - Pure Utility Functions

This package provides computational helpers and utilities that do not depend on Streamlit.
For Streamlit-specific UI components, import explicitly:
    - from helpers.ticker_rail import render_bottom_ticker_v3
    - from helpers.data_health_panel import render_data_health_panel

Core utilities (non-UI) are safe to import without Streamlit installed.
"""

# Non-UI utility functions - safe to import without Streamlit
from .ticker_sources import (
    get_wave_holdings_tickers,
    get_ticker_price_data,
    get_earnings_date,
    get_fed_indicators,
    get_waves_status,
    get_ticker_health_status,
    test_ticker_fetch
)

try:
    from .crypto_volatility_overlay import compute_crypto_overlay
    CRYPTO_VOLATILITY_OVERLAY_AVAILABLE = True
except ImportError:
    CRYPTO_VOLATILITY_OVERLAY_AVAILABLE = False

try:
    from .circuit_breaker import get_circuit_breaker, get_all_circuit_states
    from .persistent_cache import get_persistent_cache
    RESILIENCE_FEATURES_AVAILABLE = True
except ImportError:
    RESILIENCE_FEATURES_AVAILABLE = False

__all__ = [
    # Ticker data functions (non-UI)
    'get_wave_holdings_tickers',
    'get_ticker_price_data',
    'get_earnings_date',
    'get_fed_indicators',
    'get_waves_status',
    'get_ticker_health_status',
    'test_ticker_fetch',
]

if RESILIENCE_FEATURES_AVAILABLE:
    __all__.extend([
        'get_circuit_breaker',
        'get_all_circuit_states',
        'get_persistent_cache',
    ])

if CRYPTO_VOLATILITY_OVERLAY_AVAILABLE:
    __all__.append('compute_crypto_overlay')

# NOTE: Streamlit UI components (ticker_rail, data_health_panel) must be imported explicitly:
#   from helpers.ticker_rail import render_bottom_ticker_v3
#   from helpers.data_health_panel import render_data_health_panel, render_compact_health_indicator

