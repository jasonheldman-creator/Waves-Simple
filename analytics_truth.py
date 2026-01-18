"""
analytics_truth.py

Canonical TruthFrame accessor.
Provides backward-compatible get_truth_frame interface.
"""

def get_truth_frame(*args, safe_mode=False, **kwargs):
    """
    Backward-compatible TruthFrame accessor.

    Args:
        *args: Ignored (compatibility)
        safe_mode (bool): Accepted for compatibility with app.py
        **kwargs: Ignored

    Returns:
        TruthFrame object or safe default placeholder.
    """
    try:
        # If TruthFrame is defined elsewhere/imported later,
        # this will return it safely.
        return TruthFrame
    except NameError:
        # Safe fallback to prevent app crash
        return {
            "status": "unavailable",
            "reason": "TruthFrame not initialized",
            "safe_mode": safe_mode
        }