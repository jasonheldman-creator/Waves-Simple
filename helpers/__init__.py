"""
Helpers Package - Pure Utility Functions

This package provides computational helpers and utilities that do not depend
on Streamlit.

For Streamlit-specific UI components, import explicitly:
    - from helpers.ticker_rail import render_bottom_ticker_v3
    - from helpers.data_health_panel import render_data_health_panel

Core utilities (non-UI) are safe to import without Streamlit installed.
"""

# ---------------------------------------------------------------------
# Core non-UI utility functions (safe for CI + pytest environments)
# ---------------------------------------------------------------------

from .ticker_sources import (
    get_wave_holdings_tickers,
    get_ticker_price_data,
    get_earnings_date,
    get_fed_indicators,
    get_waves_status,
    get_ticker_health_status,
    test_ticker_fetch,
)

# ✅ REQUIRED FOR CI — exposes benchmark function to test runner
from .wave_performance import compute_portfolio_composite_benchmark


# ---------------------------------------------------------------------
# Optional resilience utilities
# These may not exist in minimal CI environments
# ---------------------------------------------------------------------

try:
    from .circuit_breaker import get_circuit_breaker, get_all_circuit_states
    from .persistent_cache import get_persistent_cache

    RESILIENCE_FEATURES_AVAILABLE = True
except ImportError:
    RESILIENCE_FEATURES_AVAILABLE = False


# ---------------------------------------------------------------------
# Public package exports
# ---------------------------------------------------------------------

__all__ = [
    # Ticker data utilities
    "get_wave_holdings_tickers",
    "get_ticker_price_data",
    "get_earnings_date",
    "get_fed_indicators",
    "get_waves_status",
    "get_ticker_health_status",
    "test_ticker_fetch",

    # ✅ Composite benchmark (CI requirement)
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


# ---------------------------------------------------------------------
# Notes
# ---------------------------------------------------------------------
# Streamlit UI components MUST be imported explicitly:
#
#   from helpers.ticker_rail import render_bottom_ticker_v3
#   from helpers.data_health_panel import (
#       render_data_health_panel,
#       render_compact_health_indicator,
#   )
#
# This keeps CI environments lightweight and prevents Streamlit import errors.