"""
waves.py — import-safe recovery scaffold

This file previously executed logic at import time using
undefined globals (unique_wave_ids, truth_df), which caused
Streamlit startup failures.

All execution is now gated behind explicit function calls.
"""

# -----------------------------
# Safe defaults (IMPORT SAFE)
# -----------------------------

unique_wave_ids = []
truth_df = None


# -----------------------------
# Public initializer
# -----------------------------

def initialize_waves(_truth_df, _unique_wave_ids):
    """
    Initialize wave entries inside truth_df safely.

    Parameters:
        _truth_df: object with attribute `.waves` (dict-like)
        _unique_wave_ids: iterable of wave IDs
    """

    global truth_df, unique_wave_ids
    truth_df = _truth_df
    unique_wave_ids = list(_unique_wave_ids)

    if not hasattr(truth_df, "waves"):
        truth_df.waves = {}

    for wave_id in unique_wave_ids:
        if wave_id not in truth_df.waves:
            truth_df.waves[wave_id] = {
                "health": {
                    "score": None,
                    "alpha": None,
                    "beta_drift": None,
                    "volatility": None,
                    "exposure": None
                },
                "regime_alignment": "neutral",
                "learning_signals": []
            }

    return truth_df.waves


# -----------------------------
# Import confirmation hook
# -----------------------------

def _import_check():
    return "waves module imported safely"


# -----------------------------
# No execution at import time
# -----------------------------

if __name__ == "__main__":
    print("waves.py loaded directly (no execution performed)")