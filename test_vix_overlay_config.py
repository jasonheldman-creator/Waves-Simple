"""
Test VIX Overlay Configuration

This module tests the VIX overlay configuration system to ensure:
1. Toggle functionality works correctly
2. Per-wave overrides function as expected
3. Resilience settings are properly applied
4. Configuration can be saved and loaded
"""

import sys
import os

# Add config directory to path
config_dir = os.path.join(os.path.dirname(__file__), 'config')
if config_dir not in sys.path:
    sys.path.insert(0, config_dir)

from config.vix_overlay_config import (
    VIXOverlayConfig,
    get_vix_overlay_config,
    set_vix_overlay_config,
    enable_vix_overlay,
    disable_vix_overlay,
    is_vix_overlay_live,
    get_vix_overlay_status,
    get_vix_overlay_strategy_config,
    DEFAULT_VIX_OVERLAY_CONFIG
)


def test_default_config_is_live():
    """Test that VIX overlay is LIVE by default."""
    config = VIXOverlayConfig()
    assert config.enabled is True, "VIX overlay should be enabled by default"
    assert config.resilient_mode is True, "Resilient mode should be enabled by default"


def test_toggle_global_enable_disable():
    """Test global enable/disable toggle."""
    # Create a fresh config
    config = VIXOverlayConfig()
    
    # Test enabling
    config.enabled = True
    assert config.enabled is True
    assert config.is_enabled_for_wave("US MegaCap Core Wave") is True
    
    # Test disabling
    config.enabled = False
    assert config.enabled is False
    assert config.is_enabled_for_wave("US MegaCap Core Wave") is False


def test_wave_specific_override():
    """Test per-wave configuration overrides."""
    config = VIXOverlayConfig(
        enabled=True,
        wave_overrides={
            "Bitcoin Wave": {"enabled": False},
            "AI & Cloud MegaCap Wave": {"enabled": True},
        }
    )
    
    # Global is enabled
    assert config.enabled is True
    
    # Bitcoin Wave is disabled via override
    assert config.is_enabled_for_wave("Bitcoin Wave") is False
    
    # AI & Cloud MegaCap Wave is explicitly enabled
    assert config.is_enabled_for_wave("AI & Cloud MegaCap Wave") is True
    
    # Wave without override uses global setting
    assert config.is_enabled_for_wave("US MegaCap Core Wave") is True


def test_fallback_vix_level():
    """Test fallback VIX level configuration."""
    config = VIXOverlayConfig(
        fallback_vix_level=18.0,
        wave_overrides={
            "Bitcoin Wave": {"fallback_vix_level": 25.0},
        }
    )
    
    # Global fallback
    assert config.get_fallback_vix("US MegaCap Core Wave") == 18.0
    
    # Wave-specific fallback
    assert config.get_fallback_vix("Bitcoin Wave") == 25.0


def test_diagnostics_logging_config():
    """Test diagnostics logging configuration."""
    config = VIXOverlayConfig(
        log_diagnostics=False,
        wave_overrides={
            "US MegaCap Core Wave": {"log_diagnostics": True},
        }
    )
    
    # Global logging is off
    assert config.should_log_diagnostics("AI & Cloud MegaCap Wave") is False
    
    # Wave-specific logging is on
    assert config.should_log_diagnostics("US MegaCap Core Wave") is True


def test_singleton_config_instance():
    """Test singleton configuration instance."""
    # Get config instance
    config1 = get_vix_overlay_config()
    config2 = get_vix_overlay_config()
    
    # Should be same instance
    assert config1 is config2


def test_enable_disable_functions():
    """Test convenience enable/disable functions."""
    # Set initial state
    config = VIXOverlayConfig(enabled=False)
    set_vix_overlay_config(config)
    
    # Test enable
    enable_vix_overlay()
    assert is_vix_overlay_live() is True
    
    # Test disable
    disable_vix_overlay()
    assert is_vix_overlay_live() is False
    
    # Re-enable for other tests
    enable_vix_overlay()


def test_get_vix_overlay_status():
    """Test status information retrieval."""
    config = VIXOverlayConfig(
        enabled=True,
        resilient_mode=True,
        fallback_vix_level=20.0,
        min_data_points=10,
        wave_overrides={"Bitcoin Wave": {"enabled": False}}
    )
    set_vix_overlay_config(config)
    
    status = get_vix_overlay_status()
    
    assert status["is_live"] is True
    assert status["resilient_mode"] is True
    assert status["fallback_vix_level"] == 20.0
    assert status["min_data_points"] == 10
    assert "Bitcoin Wave" in status["wave_overrides"]


def test_strategy_config_integration():
    """Test integration with waves_engine strategy system."""
    config = VIXOverlayConfig(
        enabled=True,
        resilient_mode=True,
        fallback_vix_level=18.5,
        wave_overrides={
            "Bitcoin Wave": {
                "enabled": False,
                "fallback_vix_level": 22.0,
                "log_diagnostics": True
            }
        }
    )
    set_vix_overlay_config(config)
    
    # Test for wave without override
    strategy_config = get_vix_overlay_strategy_config("US MegaCap Core Wave")
    assert strategy_config["enabled"] is True
    assert strategy_config["resilient_mode"] is True
    assert strategy_config["fallback_vix_level"] == 18.5
    assert strategy_config["log_diagnostics"] is False
    
    # Test for wave with override
    strategy_config = get_vix_overlay_strategy_config("Bitcoin Wave")
    assert strategy_config["enabled"] is False
    assert strategy_config["fallback_vix_level"] == 22.0
    assert strategy_config["log_diagnostics"] is True


def test_resilient_mode_configuration():
    """Test resilient mode configuration."""
    # Resilient mode enabled
    config1 = VIXOverlayConfig(resilient_mode=True)
    assert config1.resilient_mode is True
    
    # Resilient mode disabled
    config2 = VIXOverlayConfig(resilient_mode=False)
    assert config2.resilient_mode is False


def test_min_data_points_configuration():
    """Test minimum data points configuration."""
    config = VIXOverlayConfig(min_data_points=20)
    assert config.min_data_points == 20


def test_default_config_values():
    """Test default configuration values are sensible."""
    config = DEFAULT_VIX_OVERLAY_CONFIG
    
    # Should be LIVE by default
    assert config.enabled is True
    
    # Should be resilient by default
    assert config.resilient_mode is True
    
    # Fallback VIX should be neutral (around 20)
    assert 15.0 <= config.fallback_vix_level <= 25.0
    
    # Should require minimal data points
    assert config.min_data_points >= 5


if __name__ == "__main__":
    # Run tests
    print("Running VIX Overlay Configuration Tests...")
    print()
    
    test_default_config_is_live()
    print("✓ Default config is LIVE")
    
    test_toggle_global_enable_disable()
    print("✓ Global enable/disable toggle works")
    
    test_wave_specific_override()
    print("✓ Wave-specific overrides work")
    
    test_fallback_vix_level()
    print("✓ Fallback VIX level configuration works")
    
    test_diagnostics_logging_config()
    print("✓ Diagnostics logging configuration works")
    
    test_singleton_config_instance()
    print("✓ Singleton config instance works")
    
    test_enable_disable_functions()
    print("✓ Enable/disable convenience functions work")
    
    test_get_vix_overlay_status()
    print("✓ Status information retrieval works")
    
    test_strategy_config_integration()
    print("✓ Strategy config integration works")
    
    test_resilient_mode_configuration()
    print("✓ Resilient mode configuration works")
    
    test_min_data_points_configuration()
    print("✓ Minimum data points configuration works")
    
    test_default_config_values()
    print("✓ Default config values are sensible")
    
    print()
    print("=" * 60)
    print("All VIX Overlay Configuration Tests Passed!")
    print("=" * 60)
    print()
    print("VIX Overlay Status: LIVE ✓")
    print("Configuration System: Operational ✓")
    print("Toggle Functionality: Working ✓")
    print("Per-Wave Overrides: Supported ✓")
    print("Resilience: Enabled ✓")
