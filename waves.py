"""
waves.py — compatibility-safe initializer

This version is fully compatible with the frozen app.py call signature.
It accepts both positional and keyword arguments safely.
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
    Initialize wave entries safely.

    Compatible with app.py calling:
        initialize_waves(truth_df=..., unique_wave_ids=...)

    Also compatible with positional usage.
    """

    global truth_df, unique_wave_ids

    # --- Extract arguments safely ---
    if "truth_df" in kwargs:
        truth_df = kwargs["truth_df"]
    elif "_truth_df" in kwargs:
        truth_df = kwargs["_truth_df"]
    elif len(args) >= 1:
        truth_df = args[0]
    else:
        raise ValueError("initialize_waves: truth_df not provided")

    if "unique_wave_ids" in kwargs:
        unique_wave_ids = list(kwargs["unique_wave_ids"])
    elif "_unique_wave_ids" in kwargs:
        unique_wave_ids = list(kwargs["_unique_wave_ids"])
    elif len(args) >= 2:
        unique_wave_ids = list(args[1])
    else:
        unique_wave_ids = []

    # --- Initialize structure ---
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