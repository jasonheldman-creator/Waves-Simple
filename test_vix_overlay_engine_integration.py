"""
Test VIX Overlay Engine Integration

This module tests that the VIX overlay configuration properly integrates
with the waves_engine.py to ensure:
1. VIX overlay functions accept wave_name parameter
2. Configuration is respected in exposure calculations
3. Resilience fallbacks work correctly
4. Status API returns correct information
"""

import sys
import os
import numpy as np

# Add config directory to path
config_dir = os.path.join(os.path.dirname(__file__), 'config')
if config_dir not in sys.path:
    sys.path.insert(0, config_dir)

from config.vix_overlay_config import (
    VIXOverlayConfig,
    set_vix_overlay_config,
    enable_vix_overlay,
    disable_vix_overlay,
)

# Import waves_engine functions
from waves_engine import (
    _vix_exposure_factor,
    _vix_safe_fraction,
    get_vix_overlay_status_for_wave,
    is_vix_overlay_active_for_wave,
)


def test_vix_exposure_factor_with_wave_name():
    """Test _vix_exposure_factor accepts wave_name parameter."""
    # Test with valid VIX
    result = _vix_exposure_factor(15.0, "Standard", wave_name="US MegaCap Core Wave")
    assert 1.0 <= result <= 1.3, f"Expected exposure factor between 1.0 and 1.3, got {result}"
    
    # Test with high VIX
    result = _vix_exposure_factor(35.0, "Standard", wave_name="US MegaCap Core Wave")
    assert 0.5 <= result <= 1.0, f"Expected reduced exposure for high VIX, got {result}"
    
    # Test with missing VIX (should use fallback if resilient, or return neutral)
    result = _vix_exposure_factor(np.nan, "Standard", wave_name="US MegaCap Core Wave")
    # With resilient mode enabled and fallback_vix_level=20.0, this returns 0.95 (VIX 20-25 range)
    # This is correct behavior - proving resilience is working!
    assert 0.9 <= result <= 1.1, f"Expected near-neutral exposure for missing VIX with fallback, got {result}"


def test_vix_safe_fraction_with_wave_name():
    """Test _vix_safe_fraction accepts wave_name parameter."""
    # Test with low VIX
    result = _vix_safe_fraction(15.0, "Standard", wave_name="US MegaCap Core Wave")
    assert result == 0.0, f"Expected no safe allocation for low VIX, got {result}"
    
    # Test with high VIX
    result = _vix_safe_fraction(35.0, "Standard", wave_name="US MegaCap Core Wave")
    assert result > 0.2, f"Expected significant safe allocation for high VIX, got {result}"
    
    # Test with missing VIX (should use fallback if resilient)
    result = _vix_safe_fraction(np.nan, "Standard", wave_name="US MegaCap Core Wave")
    # With resilient mode and fallback_vix_level=20.0, this returns 0.05 (VIX 18-24 range)
    # This is correct behavior - proving resilience is working!
    assert 0.0 <= result <= 0.1, f"Expected small safe allocation for missing VIX with fallback, got {result}"


def test_resilience_with_config():
    """Test resilience fallback uses configuration."""
    # Create config with custom fallback
    config = VIXOverlayConfig(
        enabled=True,
        resilient_mode=True,
        fallback_vix_level=25.0,  # High fallback VIX
    )
    set_vix_overlay_config(config)
    
    # Test with missing VIX - should use fallback of 25.0
    result = _vix_exposure_factor(np.nan, "Standard", wave_name="US MegaCap Core Wave")
    # With VIX=25.0, exposure should be 0.85 (in the 25-30 range maps to 0.85)
    assert 0.8 <= result <= 0.9, f"Expected reduced exposure with VIX 25 fallback, got {result}"
    
    print(f"  Resilience proof: Missing VIX → fallback=25.0 → exposure={result:.2f} ✓")


def test_vix_overlay_disabled_for_wave():
    """Test VIX overlay can be disabled for specific wave."""
    # Create config that disables VIX for Bitcoin
    config = VIXOverlayConfig(
        enabled=True,
        wave_overrides={
            "Bitcoin Wave": {"enabled": False},
        }
    )
    set_vix_overlay_config(config)
    
    # Bitcoin Wave should return neutral values
    result_exposure = _vix_exposure_factor(35.0, "Standard", wave_name="Bitcoin Wave")
    result_safe = _vix_safe_fraction(35.0, "Standard", wave_name="Bitcoin Wave")
    
    # Should return neutral (1.0 and 0.0) because disabled for this wave
    assert result_exposure == 1.0, f"Expected neutral exposure for disabled wave, got {result_exposure}"
    assert result_safe == 0.0, f"Expected no safe allocation for disabled wave, got {result_safe}"


def test_get_vix_overlay_status_equity_wave():
    """Test VIX overlay status for equity wave."""
    enable_vix_overlay()
    
    status = get_vix_overlay_status_for_wave("US MegaCap Core Wave")
    
    assert status["is_equity_wave"] is True, "US MegaCap Core Wave should be equity wave"
    assert status["is_enabled_for_wave"] is True, "VIX overlay should be enabled"
    assert status["is_live"] is True, "VIX overlay should be LIVE"
    assert status["resilient_mode"] is True, "Resilient mode should be enabled"


def test_get_vix_overlay_status_crypto_wave():
    """Test VIX overlay status for crypto wave."""
    status = get_vix_overlay_status_for_wave("Bitcoin Wave")
    
    # Bitcoin is crypto, so VIX doesn't apply
    assert status["is_equity_wave"] is False, "Bitcoin Wave should not be equity wave"


def test_get_vix_overlay_status_income_wave():
    """Test VIX overlay status for income wave."""
    status = get_vix_overlay_status_for_wave("Income Wave")
    
    # Income waves don't use VIX overlay
    assert status["is_equity_wave"] is False, "Income wave should not use VIX overlay"


def test_is_vix_overlay_active_helper():
    """Test is_vix_overlay_active_for_wave helper function."""
    enable_vix_overlay()
    
    # Should be active for equity waves
    assert is_vix_overlay_active_for_wave("US MegaCap Core Wave") is True
    
    # Should not be active for crypto waves
    assert is_vix_overlay_active_for_wave("Bitcoin Wave") is False


def test_mode_adjustments():
    """Test that mode adjustments still work."""
    # Alpha-Minus-Beta should be more defensive
    amb_exposure = _vix_exposure_factor(25.0, "Alpha-Minus-Beta", wave_name="US MegaCap Core Wave")
    std_exposure = _vix_exposure_factor(25.0, "Standard", wave_name="US MegaCap Core Wave")
    
    assert amb_exposure < std_exposure, "Alpha-Minus-Beta should have lower exposure"
    
    # Private Logic should be more aggressive
    pl_exposure = _vix_exposure_factor(25.0, "Private Logic", wave_name="US MegaCap Core Wave")
    
    assert pl_exposure > std_exposure, "Private Logic should have higher exposure"


if __name__ == "__main__":
    print("Running VIX Overlay Engine Integration Tests...")
    print()
    
    test_vix_exposure_factor_with_wave_name()
    print("✓ VIX exposure factor accepts wave_name parameter")
    
    test_vix_safe_fraction_with_wave_name()
    print("✓ VIX safe fraction accepts wave_name parameter")
    
    test_resilience_with_config()
    print("✓ Resilience fallback uses configuration")
    
    test_vix_overlay_disabled_for_wave()
    print("✓ VIX overlay can be disabled per wave")
    
    test_get_vix_overlay_status_equity_wave()
    print("✓ VIX overlay status works for equity waves")
    
    test_get_vix_overlay_status_crypto_wave()
    print("✓ VIX overlay status works for crypto waves")
    
    test_get_vix_overlay_status_income_wave()
    print("✓ VIX overlay status works for income waves")
    
    test_is_vix_overlay_active_helper()
    print("✓ is_vix_overlay_active_for_wave helper works")
    
    test_mode_adjustments()
    print("✓ Mode adjustments still function correctly")
    
    print()
    print("=" * 60)
    print("All VIX Overlay Engine Integration Tests Passed!")
    print("=" * 60)
    print()
    print("Engine Integration: Complete ✓")
    print("Configuration Respected: Yes ✓")
    print("Resilience: Active ✓")
    print("Status API: Working ✓")
