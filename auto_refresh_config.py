"""
Auto-Refresh Configuration Module
==================================

Centralized configuration for WAVES Streamlit Console auto-refresh functionality.
This module provides:
- Default settings for auto-refresh behavior
- Configurable refresh intervals
- Error handling policies
- Performance safeguards

Design Principles:
- Auto-refresh is enabled by default for live decision support
- Refresh interval is set to 30 seconds to balance real-time updates with performance
- Errors automatically pause auto-refresh to maintain system stability
- Caching behavior is preserved to avoid unnecessary recomputations
"""

# ============================================================================
# AUTO-REFRESH SETTINGS
# ============================================================================

# Default auto-refresh state (True = enabled by default for live decision support)
DEFAULT_AUTO_REFRESH_ENABLED = False

# Default refresh interval in milliseconds (30000ms = 30 seconds)

# Available refresh interval options (in seconds)
REFRESH_INTERVAL_OPTIONS = {
    "30 seconds": 30000,
    "1 minute": 60000,
    "2 minutes": 120000,
    "5 minutes": 300000,
}

# Minimum allowed refresh interval (in milliseconds)
# Set to 15 seconds to prevent excessive refresh rates
MIN_REFRESH_INTERVAL_MS = 15000

# Maximum allowed refresh interval (in milliseconds)
# Set to 10 minutes to ensure reasonably fresh data
MAX_REFRESH_INTERVAL_MS = 600000

# ============================================================================
# ERROR HANDLING SETTINGS
# ============================================================================

# Auto-pause auto-refresh on errors
AUTO_PAUSE_ON_ERROR = True

# Number of consecutive errors before disabling auto-refresh permanently
MAX_CONSECUTIVE_ERRORS = 3

# Error cooldown period (in seconds)
# After an error, wait this long before attempting another auto-refresh
ERROR_COOLDOWN_SECONDS = 120

# ============================================================================
# PERFORMANCE SAFEGUARDS
# ============================================================================

# Scope of refresh: Which data should be updated during auto-refresh
# Set to False to exclude heavy computations
REFRESH_SCOPE = {
    "live_analytics": True,      # Live analytics and metrics
    "overlays": True,             # VIX overlays and regime detection
    "attribution": True,          # Alpha attribution components
    "diagnostics": True,          # System diagnostics
    "summary_metrics": True,      # Summary statistics
    "backtests": False,           # Historical backtests (excluded)
    "simulations": False,         # Heavy simulations (excluded)
    "reports": False,             # Report generation (excluded)
}

# Cache Time-To-Live (TTL) settings
# These values ensure caching protections remain in place during auto-refresh
CACHE_TTL_SECONDS = {
    "wave_data": 60,              # Wave historical data
    "market_data": 60,            # Market indicators
    "calculations": 60,           # Computed metrics
    "universe": 300,              # Wave universe (5 minutes)
}

# ============================================================================
# UI DISPLAY SETTINGS
# ============================================================================

# Show detailed auto-refresh status in UI
SHOW_REFRESH_STATUS = True

# Show last refresh timestamp
SHOW_LAST_REFRESH_TIME = True

# Show refresh interval in UI
SHOW_REFRESH_INTERVAL = True

# Use visual indicators (colors, icons) for status
USE_VISUAL_INDICATORS = True

# Status display format
STATUS_FORMAT = {
    "enabled": "🟢 ON",
    "disabled": "🔴 OFF",
    "paused": "🟡 PAUSED",
    "error": "⚠️ ERROR",
}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_default_settings():
    """
    Get default auto-refresh settings as a dictionary.
    
    Returns:
        dict: Default settings for auto-refresh functionality
    """
    return {
        "enabled": DEFAULT_AUTO_REFRESH_ENABLED,
        "interval_ms": DEFAULT_REFRESH_INTERVAL_MS,
        "auto_pause_on_error": AUTO_PAUSE_ON_ERROR,
        "max_consecutive_errors": MAX_CONSECUTIVE_ERRORS,
        "error_cooldown_seconds": ERROR_COOLDOWN_SECONDS,
        "refresh_scope": REFRESH_SCOPE.copy(),
        "cache_ttl": CACHE_TTL_SECONDS.copy(),
    }


def validate_refresh_interval(interval_ms):
    """
    Validate and clamp refresh interval to allowed range.
    
    Args:
        interval_ms (int): Refresh interval in milliseconds
        
    Returns:
        int: Validated interval (clamped to min/max range)
    """
    if interval_ms < MIN_REFRESH_INTERVAL_MS:
        return MIN_REFRESH_INTERVAL_MS
    if interval_ms > MAX_REFRESH_INTERVAL_MS:
        return MAX_REFRESH_INTERVAL_MS
    return interval_ms


def get_interval_display_name(interval_ms):
    """
    Get human-readable display name for a refresh interval.
    
    Args:
        interval_ms (int): Refresh interval in milliseconds
        
    Returns:
        str: Human-readable interval name (e.g., "1 minute")
    """
    # Check if interval matches a predefined option
    for name, value in REFRESH_INTERVAL_OPTIONS.items():
        if value == interval_ms:
            return name
    
    # Otherwise, format as seconds or minutes
    interval_sec = interval_ms / 1000
    if interval_sec < 60:
        return f"{interval_sec:.0f} seconds"
    else:
        interval_min = interval_sec / 60
        return f"{interval_min:.1f} minutes"


def should_refresh_component(component_name):
    """
    Check if a specific component should be refreshed during auto-refresh.
    
    Args:
        component_name (str): Name of the component (e.g., "live_analytics", "backtests")
        
    Returns:
        bool: True if component should be refreshed, False otherwise
    """
    return REFRESH_SCOPE.get(component_name, False)


# ============================================================================
# VERSION INFO
# ============================================================================

__version__ = "1.0.0"
__author__ = "WAVES Intelligence Team"
__description__ = "Auto-refresh configuration for WAVES Streamlit Console"
