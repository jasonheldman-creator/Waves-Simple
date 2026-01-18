"""
Adaptive Intelligence — Compatibility Layer

This module provides backward-compatible interfaces expected by app.py.
No trading logic is executed here. All outputs are diagnostic-safe defaults.
"""

def analyze_regime_intelligence(data=None):
    """
    Backward compatibility wrapper for regime intelligence analysis.

    Returns a stable, schema-correct structure expected by app.py.
    No execution logic or trading behavior is modified.
    """
    return {
        "current_regime": "neutral",
        "aligned_waves": 0,
        "total_waves": 0,
        "alignment_pct": 0.0,
        "regime_description": "Regime intelligence diagnostics available"
    }


def get_wave_health_summary(wave_id=None, truth_frame=None):
    """
    Backward compatibility wrapper for wave health diagnostics.

    Args:
        wave_id: Optional wave identifier
        truth_frame: Optional TruthFrame data

    Returns:
        dict: Stable diagnostic structure expected by app.py
    """
    return {
        "status": "available",
        "health_score": None,
        "risk_flags": [],
        "notes": "Wave health diagnostics initialized"
    }