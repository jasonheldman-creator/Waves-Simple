"""
waves.py — import-safe recovery scaffold (FULL REPLACEMENT)

This module is designed to be:
- Safe at import time (no execution)
- Compatible with legacy and new callers
- Robust to positional OR keyword arguments

It intentionally does NOT execute logic on import.
"""

# -----------------------------
# Safe defaults (IMPORT SAFE)
# -----------------------------

unique_wave_ids = []
truth_df = None


# -----------------------------
# Public initializer
# -----------------------------

def initialize_waves(*args, **kwargs):
    """
    Initialize wave entries inside truth_df safely.

    Supports ALL of the following call styles:
        initialize_waves(truth_df, unique_wave_ids)
        initialize_waves(truth_df=..., unique_wave_ids=...)
        initialize_waves(_truth_df=..., _unique_wave_ids=...)

    This is required because app.py cannot be modified.
    """

    global truth_df, unique_wave_ids

    # --- Resolve arguments safely ---
    if args:
        _truth_df = args[0]
        _unique_wave_ids = args[1] if len(args) > 1 else []
    else:
        _truth_df = (
            kwargs.get("truth_df")
            or kwargs.get("_truth_df")
        )
        _unique_wave_ids = (
            kwargs.get("unique_wave_ids")
            or kwargs.get("_unique_wave_ids")
            or []
        )

    # --- Validation ---
    if _truth_df is None:
        raise ValueError("initialize_waves called with truth_df=None")

    truth_df = _truth_df
    unique_wave_ids = list(_unique_wave_ids)

    # --- Ensure waves container exists ---
    if not hasattr(truth_df, "waves") or truth_df.waves is None:
        truth_df.waves = {}

    # --- Initialize waves safely ---
    for wave_id in unique_wave_ids:
        if wave_id not in truth_df.waves:
            truth_df.waves[wave_id] = {
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