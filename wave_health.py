"""
wave_health.py

Backward-compatible Wave Health diagnostics module.
This file provides safe, read-only diagnostic stubs expected by app.py.
"""

def get_wave_health_summary(*args, **kwargs):
    """
    Backward-compatible wave health accessor.

    Args:
        *args: Ignored (compatibility)
        **kwargs: Ignored

    Returns:
        dict: Stable diagnostic structure
    """
    return {
        "status": "unknown",
        "health_score": None,
        "risk_flags": [],
        "issues": [],
        "notes": "Wave health diagnostics not available"
    }