"""
waves.py — import-safe recovery scaffold (compat-safe)

This module is designed to be safely imported by Streamlit.
It accepts BOTH positional and keyword-based initialization
to remain compatible with existing app.py calls.
"""

# -----------------------------
# Safe defaults (IMPORT SAFE)
# -----------------------------

unique_wave_ids = []
truth_df = None


# -----------------------------
# Public initializer (COMPATIBLE)
# -----------------------------

def initialize_waves(
    _truth_df=None,
    _unique_wave_ids=None,
    *,
    truth_df=None,
    unique_wave_ids=None
):
    """
    Initialize wave entries inside truth_df safely.

    Accepts BOTH:
      initialize_waves(truth_df, unique_wave_ids)
      initialize_waves(truth_df=..., unique_wave_ids=...)

    This keeps compatibility with existing app.py.
    """

    global truth_df as _global_truth_df
    global unique_wave_ids as _global_wave_ids

    # Normalize inputs (keyword args win)
    resolved_truth_df = truth_df if truth_df is not None else _truth_df
    resolved_wave_ids = (
        unique_wave_ids if unique_wave_ids is not None else _unique_wave_ids
    )

    if resolved_truth_df is None:
        raise ValueError("initialize_waves: truth_df is required")

    if resolved_wave_ids is None:
        raise ValueError("initialize_waves: unique_wave_ids is required")

    _global_truth_df = resolved_truth_df
    _global_wave_ids = list(resolved_wave_ids)

    if not hasattr(_global_truth_df, "waves"):
        _global_truth_df.waves = {}

    for wave_id in _global_wave_ids:
        if wave_id not in _global_truth_df.waves:
            _global_truth_df.waves[wave_id] = {
                "health": {
                    "score": None,
                    "alpha": None,
                    "beta_drift": None,
                    "volatility": None,
                    "exposure": None,
                },
                "regime_alignment": "neutral",
                "learning_signals": [],
            }

    return _global_truth_df.waves


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