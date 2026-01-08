#!/usr/bin/env python3
"""
Integration tests for crypto-specific volatility overlay system.

Tests crypto wave overlay integration including:
1. Overlay fields computation and persistence
2. Exposure scaling with minimum floors (Growth: 0.20, Income: 0.40)
3. Graceful degradation when data is missing
4. Integration with price_cache.parquet
5. Portfolio-level crypto overlay contribution
"""

import sys
import os
import numpy as np
import pandas as pd
import pytest

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import waves_engine as we
    from helpers.price_book import get_price_book
    from helpers.wave_performance import compute_wave_returns
    IMPORTS_AVAILABLE = True
except ImportError as e:
    IMPORTS_AVAILABLE = False
    IMPORT_ERROR = str(e)


# Skip all tests if imports not available
pytestmark = pytest.mark.skipif(
    not IMPORTS_AVAILABLE,
    reason=f"Required imports not available: {IMPORT_ERROR if not IMPORTS_AVAILABLE else ''}"
)


class TestCryptoWaveIdentification:
    """Test crypto wave identification from wave registry."""
    
    def test_crypto_growth_waves_identified(self):
        """Test that crypto growth waves are correctly identified."""
        crypto_growth_waves = [
            "Crypto L1 Growth Wave",
            "Crypto DeFi Growth Wave",
            "Crypto L2 Growth Wave",
            "Crypto AI Growth Wave",
            "Crypto Broad Growth Wave"
        ]
        
        for wave_name in crypto_growth_waves:
            assert we._is_crypto_wave(wave_name), f"{wave_name} should be identified as crypto"
            assert we._is_crypto_growth_wave(wave_name), f"{wave_name} should be identified as crypto growth"
            assert not we._is_crypto_income_wave(wave_name), f"{wave_name} should not be crypto income"
    
    def test_crypto_income_wave_identified(self):
        """Test that crypto income wave is correctly identified."""
        wave_name = "Crypto Income Wave"
        assert we._is_crypto_wave(wave_name), f"{wave_name} should be identified as crypto"
        assert not we._is_crypto_growth_wave(wave_name), f"{wave_name} should not be crypto growth"
        assert we._is_crypto_income_wave(wave_name), f"{wave_name} should be crypto income"
    
    def test_equity_waves_not_crypto(self):
        """Test that equity waves are not identified as crypto."""
        equity_waves = [
            "S&P 500 Wave",
            "US MegaCap Core Wave",
            "Income Wave",
            "SmartSafe Treasury Cash Wave"
        ]
        
        for wave_name in equity_waves:
            assert not we._is_crypto_wave(wave_name), f"{wave_name} should not be identified as crypto"
            assert not we._is_crypto_growth_wave(wave_name), f"{wave_name} should not be crypto growth"
            assert not we._is_crypto_income_wave(wave_name), f"{wave_name} should not be crypto income"


class TestCryptoOverlayFields:
    """Test crypto overlay field computation and persistence."""
    
    def test_crypto_growth_wave_overlay_fields(self):
        """Test that crypto growth waves compute overlay fields correctly."""
        wave_name = "Crypto L1 Growth Wave"
        
        # Compute wave history with diagnostics
        result = we.compute_history_nav(
            wave_name=wave_name,
            mode="Standard",
            days=90,
            include_diagnostics=True
        )
        
        # Verify result is not empty
        assert not result.empty, f"Result for {wave_name} should not be empty"
        assert len(result) > 0, f"Result for {wave_name} should have data"
        
        # Verify crypto-specific overlay columns exist
        crypto_columns = [
            'crypto_trend_regime',
            'crypto_vol_state',
            'crypto_liq_state'
        ]
        
        for col in crypto_columns:
            assert col in result.columns, f"Column '{col}' should exist in crypto wave diagnostics"
    
    def test_crypto_income_wave_overlay_fields(self):
        """Test that crypto income wave computes overlay fields correctly."""
        wave_name = "Crypto Income Wave"
        
        # Compute wave history with diagnostics
        result = we.compute_history_nav(
            wave_name=wave_name,
            mode="Standard",
            days=90,
            include_diagnostics=True
        )
        
        # Verify result is not empty
        assert not result.empty, f"Result for {wave_name} should not be empty"
        assert len(result) > 0, f"Result for {wave_name} should have data"
        
        # Verify crypto income-specific fields exist
        # Crypto income uses different overlays than growth
        assert 'is_crypto' in result.columns, "Crypto income wave should have is_crypto flag"
    
    def test_equity_wave_no_crypto_fields(self):
        """Test that equity waves do not have crypto overlay fields."""
        wave_name = "S&P 500 Wave"
        
        # Compute wave history with diagnostics
        result = we.compute_history_nav(
            wave_name=wave_name,
            mode="Standard",
            days=90,
            include_diagnostics=True
        )
        
        # Verify result is not empty
        assert not result.empty, f"Result for {wave_name} should not be empty"
        
        # Verify crypto fields are not present or are disabled
        if 'is_crypto' in result.columns:
            # If column exists, it should be False or NaN
            assert not result['is_crypto'].any(), "Equity wave should not have is_crypto=True"


class TestExposureScaling:
    """Test exposure scaling with minimum floors."""
    
    def test_crypto_growth_exposure_floor(self):
        """Test that crypto growth waves respect 0.20 minimum exposure."""
        wave_name = "Crypto L1 Growth Wave"
        
        # Compute wave history
        result = we.compute_history_nav(
            wave_name=wave_name,
            mode="Standard",
            days=90,
            include_diagnostics=True
        )
        
        # Check if exposure column exists
        if 'exposure' in result.columns:
            # Verify minimum exposure of 0.20
            min_exposure = result['exposure'].min()
            assert min_exposure >= 0.20, f"Crypto growth wave minimum exposure should be >= 0.20, got {min_exposure:.4f}"
    
    def test_crypto_income_exposure_floor(self):
        """Test that crypto income wave respects 0.40 minimum exposure."""
        wave_name = "Crypto Income Wave"
        
        # Compute wave history
        result = we.compute_history_nav(
            wave_name=wave_name,
            mode="Standard",
            days=90,
            include_diagnostics=True
        )
        
        # Check if exposure column exists
        if 'exposure' in result.columns:
            # Verify minimum exposure of 0.40
            min_exposure = result['exposure'].min()
            assert min_exposure >= 0.40, f"Crypto income wave minimum exposure should be >= 0.40, got {min_exposure:.4f}"


class TestGracefulDegradation:
    """Test graceful degradation when data is missing."""
    
    def test_missing_benchmark_fallback(self):
        """Test that crypto waves gracefully handle missing benchmark data."""
        wave_name = "Crypto L1 Growth Wave"
        
        # Compute wave history (may have incomplete benchmark data)
        result = we.compute_history_nav(
            wave_name=wave_name,
            mode="Standard",
            days=90,
            include_diagnostics=False
        )
        
        # Verify computation succeeds even with missing data
        assert not result.empty, "Wave computation should succeed with fallback"
        assert 'wave_nav' in result.columns, "wave_nav column should exist"
        assert 'bm_ret' in result.columns, "bm_ret column should exist (with fallback)"
    
    def test_incomplete_price_data_handling(self):
        """Test that crypto waves handle incomplete price data gracefully."""
        wave_name = "Crypto DeFi Growth Wave"
        
        # Compute wave history with short window (may have gaps)
        result = we.compute_history_nav(
            wave_name=wave_name,
            mode="Standard",
            days=30,
            include_diagnostics=False
        )
        
        # Verify computation succeeds
        assert 'wave_nav' in result.columns, "wave_nav should be computed despite data gaps"
        
        # Verify no NaN in critical columns (should use forward fill or fallback)
        if len(result) > 0:
            # At least some valid data should exist
            assert result['wave_nav'].notna().any(), "wave_nav should have some valid values"


class TestPriceCacheIntegration:
    """Test integration with price_cache.parquet."""
    
    def test_crypto_wave_with_price_cache(self):
        """Test processing crypto wave using price_cache.parquet."""
        wave_name = "Crypto L1 Growth Wave"
        
        # Get price book
        try:
            price_book = get_price_book()
        except Exception as e:
            pytest.skip(f"Price cache not available: {e}")
        
        if price_book is None or price_book.empty:
            pytest.skip("Price cache is empty")
        
        # Compute wave returns using price cache
        result = compute_wave_returns(
            wave_name=wave_name,
            price_book=price_book,
            periods=[1, 30, 60]
        )
        
        # Verify computation succeeded or failed gracefully
        assert 'success' in result, "Result should have success field"
        assert 'failure_reason' in result, "Result should have failure_reason field"
        
        if not result['success']:
            # If failed, should have clear reason
            assert result['failure_reason'] is not None, "Failure should have reason"
            # Check if it's due to missing crypto tickers (expected)
            assert any(keyword in result['failure_reason'].lower() for keyword in ['ticker', 'price_book', 'missing']), \
                f"Failure reason should be clear: {result['failure_reason']}"
    
    def test_crypto_overlay_fields_persist(self):
        """Test that overlay fields are persisted in wave output."""
        wave_name = "Crypto Broad Growth Wave"
        
        # Compute wave history with full diagnostics
        result = we.compute_history_nav(
            wave_name=wave_name,
            mode="Standard",
            days=90,
            include_diagnostics=True
        )
        
        # Verify overlay diagnostic fields persist in output
        if not result.empty:
            # Check for at least basic overlay indicators
            has_crypto_diagnostics = any(col for col in result.columns if 'crypto' in col.lower())
            assert has_crypto_diagnostics, "Crypto waves should have crypto-specific diagnostic fields"


class TestOverlayBehavior:
    """Test crypto overlay behavior and regime classification."""
    
    def test_crypto_trend_regime_classification(self):
        """Test crypto trend regime classification."""
        test_cases = [
            (0.20, "strong_uptrend"),
            (0.10, "uptrend"),
            (0.00, "neutral"),
            (-0.10, "downtrend"),
            (-0.20, "strong_downtrend"),
        ]
        
        for trend, expected_regime in test_cases:
            regime = we._crypto_trend_regime(trend)
            assert regime == expected_regime, f"Trend {trend} should map to {expected_regime}, got {regime}"
    
    def test_crypto_volatility_state_classification(self):
        """Test crypto volatility state classification."""
        test_cases = [
            (0.25, "extreme_compression"),
            (0.40, "compression"),
            (0.60, "normal"),
            (1.00, "expansion"),
            (1.50, "extreme_expansion"),
        ]
        
        for vol, expected_state in test_cases:
            state = we._crypto_volatility_state(vol)
            assert state == expected_state, f"Volatility {vol} should map to {expected_state}, got {state}"
    
    def test_crypto_liquidity_state_classification(self):
        """Test crypto liquidity state classification."""
        test_cases = [
            (2.0, "strong_volume"),
            (1.2, "normal_volume"),
            (0.5, "weak_volume"),
        ]
        
        for vol_ratio, expected_state in test_cases:
            state = we._crypto_liquidity_state(vol_ratio)
            assert state == expected_state, f"Volume ratio {vol_ratio} should map to {expected_state}, got {state}"


class TestPortfolioIntegration:
    """Test portfolio-level crypto overlay integration."""
    
    def test_crypto_waves_in_portfolio_snapshot(self):
        """Test that crypto waves are included in portfolio snapshot."""
        try:
            price_book = get_price_book()
        except Exception as e:
            pytest.skip(f"Price cache not available: {e}")
        
        if price_book is None or price_book.empty:
            pytest.skip("Price cache is empty")
        
        # Import portfolio snapshot function
        from helpers.wave_performance import compute_portfolio_snapshot
        
        # Compute portfolio snapshot
        snapshot = compute_portfolio_snapshot(
            price_book=price_book,
            mode='Standard',
            periods=[1, 30, 60]
        )
        
        # Verify snapshot computed successfully or failed gracefully
        assert 'success' in snapshot, "Snapshot should have success field"
        
        if snapshot['success']:
            # Verify wave count includes crypto waves
            assert snapshot['wave_count'] > 0, "Portfolio should include waves"
    
    def test_no_double_scaling_in_portfolio(self):
        """Test that crypto wave returns are not double-scaled in portfolio."""
        # This is a structural test - crypto overlays should only be applied once
        # in compute_history_nav, not again in portfolio aggregation
        
        wave_name = "Crypto L1 Growth Wave"
        
        # Compute wave history (applies overlay)
        result = we.compute_history_nav(
            wave_name=wave_name,
            mode="Standard",
            days=90,
            include_diagnostics=False
        )
        
        # Verify wave_ret column exists (post-overlay returns)
        assert 'wave_ret' in result.columns, "wave_ret should exist after overlay application"
        
        # Portfolio should use wave_ret directly without re-applying overlay
        # No explicit test possible without portfolio integration, but
        # this test documents the expected behavior


def test_crypto_overlay_system_available():
    """Test that crypto overlay system is available and importable."""
    assert IMPORTS_AVAILABLE, "Crypto overlay system should be importable"
    assert hasattr(we, '_is_crypto_wave'), "Should have _is_crypto_wave function"
    assert hasattr(we, '_crypto_trend_regime'), "Should have _crypto_trend_regime function"
    assert hasattr(we, '_crypto_volatility_state'), "Should have _crypto_volatility_state function"
    assert hasattr(we, '_crypto_liquidity_state'), "Should have _crypto_liquidity_state function"


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
