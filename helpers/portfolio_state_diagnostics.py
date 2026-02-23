# helpers/portfolio_state_diagnostics.py
"""
Temporary deployment-safe diagnostics stub.

Purpose:
Prevents Streamlit Cloud crashes when diagnostics
module is unavailable. Returns empty diagnostics
until full implementation is restored.
"""

def get_wave_diagnostics(wave_name=None):
    """
    Safe fallback diagnostics.
    Returns empty structure so UI continues rendering.
    """
    return {
        "status": "unavailable",
        "message": "Diagnostics module temporarily stubbed for deployment stability.",
        "wave": wave_name,
        "data": None
    }