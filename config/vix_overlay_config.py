"""
VIX Overlay Configuration Module

This module provides configuration settings for the VIX/Regime Overlay functionality
in the WAVES Intelligence™ system. The VIX overlay dynamically adjusts equity exposure
based on market volatility (VIX) and trend regime.

Configuration Structure:
- Global toggle for enabling/disabling VIX overlay
- Per-wave overrides
- Resilience settings for missing data
- Exposure and safe fraction parameters
"""

from typing import Dict, Optional, Any
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class VIXOverlayConfig:
    """
    Configuration for VIX Overlay functionality.
    
    Attributes:
        enabled: Global toggle for VIX overlay (True = LIVE, False = disabled)
        resilient_mode: Handle missing VIX data gracefully (True = use fallbacks)
        fallback_vix_level: Default VIX level when data is missing (neutral level)
        min_data_points: Minimum data points required to enable overlay
        wave_overrides: Per-wave specific overrides
        log_diagnostics: Enable detailed logging of VIX overlay decisions
    """
    enabled: bool = True  # VIX Overlay is LIVE by default
    resilient_mode: bool = True  # Gracefully handle missing data
    fallback_vix_level: float = 20.0  # Neutral VIX level for fallback
    min_data_points: int = 10  # Minimum days of data required
    wave_overrides: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    log_diagnostics: bool = False  # Set to True for detailed logging
    
    def is_enabled_for_wave(self, wave_name: str) -> bool:
        """
        Check if VIX overlay is enabled for a specific wave.
        
        Args:
            wave_name: Name of the wave
            
        Returns:
            True if enabled for this wave, False otherwise
        """
        # Check wave-specific override first
        if wave_name in self.wave_overrides:
            wave_config = self.wave_overrides[wave_name]
            if "enabled" in wave_config:
                return bool(wave_config["enabled"])
        
        # Fall back to global setting
        return self.enabled
    
    def get_fallback_vix(self, wave_name: str) -> float:
        """
        Get fallback VIX level for a specific wave.
        
        Args:
            wave_name: Name of the wave
            
        Returns:
            Fallback VIX level (default: 20.0)
        """
        # Check wave-specific override first
        if wave_name in self.wave_overrides:
            wave_config = self.wave_overrides[wave_name]
            if "fallback_vix_level" in wave_config:
                return float(wave_config["fallback_vix_level"])
        
        # Fall back to global setting
        return self.fallback_vix_level
    
    def should_log_diagnostics(self, wave_name: str) -> bool:
        """
        Check if diagnostics logging is enabled for a specific wave.
        
        Args:
            wave_name: Name of the wave
            
        Returns:
            True if diagnostics logging enabled
        """
        # Check wave-specific override first
        if wave_name in self.wave_overrides:
            wave_config = self.wave_overrides[wave_name]
            if "log_diagnostics" in wave_config:
                return bool(wave_config["log_diagnostics"])
        
        # Fall back to global setting
        return self.log_diagnostics


# Default global configuration
DEFAULT_VIX_OVERLAY_CONFIG = VIXOverlayConfig(
    enabled=True,  # VIX Overlay is LIVE
    resilient_mode=True,
    fallback_vix_level=20.0,
    min_data_points=10,
    wave_overrides={
        # Example wave-specific overrides (can be expanded)
        # "Crypto Income Wave": {"enabled": False},  # Disable for crypto income
        # "Bitcoin Wave": {"enabled": False},  # Disable for Bitcoin
    },
    log_diagnostics=False
)


# Configuration singleton instance
_config_instance: Optional[VIXOverlayConfig] = None


def get_vix_overlay_config() -> VIXOverlayConfig:
    """
    Get the current VIX overlay configuration.
    
    Returns:
        VIXOverlayConfig instance
    """
    global _config_instance
    if _config_instance is None:
        _config_instance = DEFAULT_VIX_OVERLAY_CONFIG
    return _config_instance


def set_vix_overlay_config(config: VIXOverlayConfig) -> None:
    """
    Set the VIX overlay configuration.
    
    Args:
        config: New VIXOverlayConfig instance
    """
    global _config_instance
    _config_instance = config
    logger.info(f"VIX overlay configuration updated: enabled={config.enabled}, resilient_mode={config.resilient_mode}")


def enable_vix_overlay() -> None:
    """Enable VIX overlay globally (make it LIVE)."""
    config = get_vix_overlay_config()
    config.enabled = True
    logger.info("VIX overlay enabled (LIVE)")


def disable_vix_overlay() -> None:
    """Disable VIX overlay globally."""
    config = get_vix_overlay_config()
    config.enabled = False
    logger.warning("VIX overlay disabled")


def is_vix_overlay_live() -> bool:
    """
    Check if VIX overlay is currently LIVE.
    
    Returns:
        True if VIX overlay is enabled globally
    """
    return get_vix_overlay_config().enabled


# VIX Overlay Status Information
def get_vix_overlay_status() -> Dict[str, Any]:
    """
    Get comprehensive VIX overlay status information.
    
    Returns:
        Dictionary with status information including:
        - is_live: bool
        - resilient_mode: bool
        - fallback_vix_level: float
        - min_data_points: int
        - wave_overrides: dict
    """
    config = get_vix_overlay_config()
    return {
        "is_live": config.enabled,
        "resilient_mode": config.resilient_mode,
        "fallback_vix_level": config.fallback_vix_level,
        "min_data_points": config.min_data_points,
        "wave_overrides": config.wave_overrides,
        "log_diagnostics": config.log_diagnostics,
    }


# Convenience function for integration with waves_engine
def get_vix_overlay_strategy_config(wave_name: str) -> Dict[str, Any]:
    """
    Get VIX overlay configuration for integration with waves_engine strategy system.
    
    Args:
        wave_name: Name of the wave
        
    Returns:
        Dictionary with enabled status and parameters
    """
    config = get_vix_overlay_config()
    return {
        "enabled": config.is_enabled_for_wave(wave_name),
        "resilient_mode": config.resilient_mode,
        "fallback_vix_level": config.get_fallback_vix(wave_name),
        "log_diagnostics": config.should_log_diagnostics(wave_name),
    }
