"""
Helpers Package — Pure Utility Functions

This package exposes computational helpers that are safe to import
without Streamlit installed.

UI components must be imported explicitly.
"""

# --------------------------------------------------
# Core ticker utilities (safe imports)
# --------------------------------------------------

from .ticker_sources import (
    get_wave_holdings_tickers,
    get_ticker_price_data,
    get_earnings_date,
    get_fed_indicators,
    get_waves_status,
    get_ticker_health_status,
    test_ticker_fetch,
)

# --------------------------------------------------
# Portfolio benchmark utilities (CI REQUIRED EXPORT)
# --------------------------------------------------

try:
    from .wave_performance import compute_portfolio_composite_benchmark
    PORTFOLIO_BENCHMARK_AVAILABLE = True
except Exception:
    PORTFOLIO_BENCHMARK_AVAILABLE = False

# --------------------------------------------------
# Optional resilience utilities
# --------------------------------------------------

try:
    from .circuit_breaker import get_circuit_breaker, get_all_circuit_states
    from .persistent_cache import get_persistent_cache
    RESILIENCE_FEATURES_AVAILABLE = True
except Exception:
    RESILIENCE_FEATURES_AVAILABLE = False

# --------------------------------------------------
# Public exports
# --------------------------------------------------

__all__ = [
    "get_wave_holdings_tickers",
    "get_ticker_price_data",
    "get_earnings_date",
    "get_fed_indicators",
    "get_waves_status",
    "get_ticker_health_status",
    "test_ticker_fetch",
]

if PORTFOLIO_BENCHMARK_AVAILABLE:
    __all__.append("compute_portfolio_composite_benchmark")

if RESILIENCE_FEATURES_AVAILABLE:
    __all__.extend([
        "get_circuit_breaker",
        "get_all_circuit_states",
        "get_persistent_cache",
    ])