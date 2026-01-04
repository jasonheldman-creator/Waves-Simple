"""
V3 ADD-ON: Bottom Ticker (Institutional Rail) - Helpers Package
Enhanced with resilience features: circuit breaker, persistent cache, and health monitoring.
"""

from .ticker_rail import render_bottom_ticker_v3
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
    from .circuit_breaker import get_circuit_breaker, get_all_circuit_states
    from .persistent_cache import get_persistent_cache
    from .data_health_panel import render_data_health_panel, render_compact_health_indicator
    RESILIENCE_FEATURES_AVAILABLE = True
except ImportError:
    RESILIENCE_FEATURES_AVAILABLE = False

__all__ = [
    'render_bottom_ticker_v3',
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
        'render_data_health_panel',
        'render_compact_health_indicator',
    ])

