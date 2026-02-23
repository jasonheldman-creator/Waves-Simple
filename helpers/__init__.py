"""
Helpers Package - Pure Utility Functions

This package provides computational helpers and utilities that do not depend on Streamlit.

For Streamlit-specific UI components, import explicitly:
    - from helpers.ticker_rail import render_bottom_ticker_v3
    - from helpers.data_health_panel import render_data_health_panel

Core utilities (non-UI) are safe to import without Streamlit installed.
"""

# ============================================================
# CORE NON-UI UTILITIES
# ============================================================

from .ticker_sources import (
    get_wave_holdings_tickers,
    get_ticker_price_data,
    get_earnings_date,
    get_fed_indicators,
    get_waves_status,
    get_ticker_health_status,
    test_ticker_fetch,
)

# ✅ REQUIRED FOR CI — expose benchmark computation
from .wave_performance import compute_portfolio_composite_benchmark


# ============================================================
# OPTIONAL RESILIENCE FEATURES
# ============================================================

try:
    from .circuit_breaker import get_circuit_breaker, get_all_circuit_states
    from .persistent_cache import get_persistent_cache

    RESILIENCE_FEATURES_AVAILABLE = True
except ImportError:
    RESILIENCE_FEATURES_AVAILABLE = False


# ============================================================
# PUBLIC EXPORTS
# ============================================================

__all__ = [
    # ticker data utilities
    "get_wave_holdings_tickers",
    "get_ticker_price_data",
    "get_earnings_date",
    "get_fed_indicators",
    "get_waves_status",
    "get_ticker_health_status",
    "test_ticker_fetch",

    # ✅ expose composite benchmark for CI + integrations
    "compute_portfolio_composite_benchmark",
]

if RESILIENCE_FEATURES_AVAILABLE:
    __all__.extend(
        [
            "get_circuit_breaker",
            "get_all_circuit_states",
            "get_persistent_cache",
        ]
    )


# ============================================================
# NOTES
# ============================================================

# Streamlit UI components must be imported explicitly:
#
#   from helpers.ticker_rail import render_bottom_ticker_v3
#   from helpers.data_health_panel import (
#       render_data_health_panel,
#       render_compact_health_indicator,
#   )
#
# This prevents Streamlit from being required during CI testing.