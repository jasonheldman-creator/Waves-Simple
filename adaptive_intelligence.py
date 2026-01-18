"""
Adaptive Intelligence — Compatibility Layer

This module provides backward-compatible interfaces expected by app.py.
No trading logic is executed here. All outputs are diagnostic-safe defaults.
"""

def analyze_regime_intelligence(*args, **kwargs):
    """
    Backward compatibility wrapper for regime intelligence analysis.
    Accepts any arguments passed by app.py.
    """
    return {
        "current_regime": "neutral",
        "aligned_waves": 0,
        "total_waves": 0,
        "alignment_pct": 0.0,
        "regime_description": "Regime intelligence diagnostics available"
    }


def get_wave_health_summary(*args, **kwargs):
    """
    Backward compatibility wrapper for wave health diagnostics.
    Accepts any arguments passed by app.py.
    """
    return {
        "status": "available",
        "health_score": None,
        "risk_flags": [],
        "notes": "Wave health diagnostics initialized"
    }


def detect_learning_signals(*args, **kwargs):
    """
    Backward compatibility wrapper for learning signals.
    Accepts any arguments passed by app.py.
    """
    return {
        "signals_detected": False,
        "signal_strength": 0.0,
        "details": [],
        "notes": "Learning signals not available"
    }