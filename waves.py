"""
waves.py — import-safe recovery scaffold (backward compatible)

This module is designed to be SAFE under Streamlit import semantics
and compatible with a frozen app.py entrypoint.

Key guarantees:
• No execution at import time
• Accepts BOTH legacy and new initialize_waves() signatures
• Compatible with keyword arguments used by app.py
• Safe to reload and redeploy
"""

# -----------------------------
# Safe module-level defaults
# -----------------------------

truth_df = None
unique_wave_ids = []


# -----------------------------
# Public initializer (COMPATIBLE)
# -----------------------------

def initialize_waves(
    _truth_df=None,
    _unique_wave_ids=None,
    **kwargs
):
    """
    Initialize wave structures safely.

    Accepts BOTH:
        initialize_waves(truth_df=..., unique_wave_ids=...)
    AND:
        initialize_waves(_truth_df, _unique_wave_ids)

    This ensures compatibility with frozen app.py.
    """

    global truth_df, unique_wave_ids

    # --- Normalize inputs (keyword takes precedence) ---
    truth_df = kwargs.get("truth_df", _truth_df)
    unique_wave_ids = kwargs.get("unique_wave_ids", _unique_wave_ids)

    # --- Hard validation ---
    if truth_df is None:
        raise ValueError("initialize_waves(): truth_df is required")

    if unique_wave_ids is None:
        raise ValueError("initialize_waves(): unique_wave_ids is required")

    unique_wave_ids = list(unique_wave_ids)

    # --- Ensure waves container exists ---
    if not hasattr(truth_df, "waves") or truth_df.waves is None:
        truth_df.waves = {}

    # --- Initialize missing waves ---
    for wave_id in unique_wave_ids:
        if wave_id not in truth_df.waves:
            truth_df.waves[wave_id] = {
                "health": {
                    "score": None,
                    "