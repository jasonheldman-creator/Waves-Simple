"""
Test that helpers package can be imported without Streamlit.

This test ensures CI robustness by verifying that the helpers package
does not require Streamlit at import time. Streamlit is only required
when actually using UI-rendering functions.
"""

import sys
import pytest


def test_helpers_import_without_streamlit():
    """Test that helpers can be imported without Streamlit installed."""
    # Ensure streamlit is not available
    if 'streamlit' in sys.modules:
        del sys.modules['streamlit']
    
    # This should work without Streamlit
    import helpers
    
    # Verify we can access utility functions
    assert hasattr(helpers, 'get_wave_holdings_tickers')
    assert hasattr(helpers, 'get_ticker_price_data')
    assert hasattr(helpers, 'get_earnings_date')
    assert hasattr(helpers, 'get_fed_indicators')
    assert hasattr(helpers, 'get_waves_status')


def test_explicit_ticker_rail_import_without_streamlit():
    """Test that ticker_rail module can be imported without Streamlit."""
    # Ensure streamlit is not available
    if 'streamlit' in sys.modules:
        del sys.modules['streamlit']
    
    # This should work - import the module
    from helpers import ticker_rail
    
    # Verify the function exists
    assert hasattr(ticker_rail, 'render_bottom_ticker_v3')
    assert callable(ticker_rail.render_bottom_ticker_v3)


def test_ticker_rail_function_requires_streamlit():
    """Test that calling ticker_rail functions without Streamlit gives helpful error."""
    # Ensure streamlit is not available
    if 'streamlit' in sys.modules:
        del sys.modules['streamlit']
    
    from helpers.ticker_rail import render_bottom_ticker_v3
    
    # Calling the function without Streamlit should raise RuntimeError
    with pytest.raises(RuntimeError) as exc_info:
        render_bottom_ticker_v3()
    
    # Verify the error message is helpful
    assert "Streamlit is required" in str(exc_info.value)
    assert "pip install streamlit" in str(exc_info.value)


def test_data_health_panel_import_without_streamlit():
    """Test that data_health_panel module can be imported without Streamlit."""
    # Ensure streamlit is not available
    if 'streamlit' in sys.modules:
        del sys.modules['streamlit']
    
    # This should work - import the module
    from helpers import data_health_panel
    
    # Verify the functions exist
    assert hasattr(data_health_panel, 'render_data_health_panel')
    assert hasattr(data_health_panel, 'render_compact_health_indicator')


def test_ticker_sources_conditional_caching():
    """Test that ticker_sources works with conditional caching."""
    from helpers.ticker_sources import STREAMLIT_AVAILABLE, conditional_cache
    
    # Create a test function with conditional caching
    @conditional_cache(ttl=60)
    def test_func():
        return "test_value"
    
    # Should work regardless of Streamlit availability
    result = test_func()
    assert result == "test_value"
    
    # Verify STREAMLIT_AVAILABLE is set correctly
    assert isinstance(STREAMLIT_AVAILABLE, bool)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
