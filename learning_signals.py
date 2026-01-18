"""
learning_signals.py

Diagnostic-only learning signal detection.
Read-only system. No trading logic.
"""

def detect_learning_signals(truth_frame=None, *args, **kwargs):
    """
    Backward-compatible learning signal detector.

    Args:
        truth_frame: Optional TruthFrame instance
        *args, **kwargs: Ignored (compatibility)

    Returns:
        dict: Stable diagnostic structure expected by app.py
    """

    # Safe default if TruthFrame is unavailable
    if truth_frame is None:
        return {
            "signals_detected": False,
            "signal_strength": 0.0,
            "details": [],
            "notes": "TruthFrame not provided"
        }

    # Diagnostic-only placeholder (no execution logic)
    return {
        "signals_detected": False,
        "signal_strength": 0.0,
        "details": [],
        "notes": "Learning signal engine initialized (diagnostic mode)"
    }