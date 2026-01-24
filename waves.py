"""
waves.py — import-safe recovery scaffold (COMPATIBLE FIX)

This version is fully backward-compatible with existing app.py calls.
It safely accepts keyword arguments (truth_df, unique_wave_ids) without
executing logic at import time.
"""

# -----------------------------
# Safe defaults (IMPORT SAFE)
# -----------------------------

unique_wave_ids = []
truth_df = None


# -----------------------------
# Public initializer (COMPATIBLE)
# -----------------------------

def initialize_waves(*args, **kwargs):
    """
    Initialize wave entries inside truth_df safely.

    Accepts:
        initialize_waves(truth_df=..., unique_wave_ids=...)
        initialize_waves(_truth_df, _unique_wave_ids)

    This keeps app.py untouched and prevents signature mismatch errors.
    """

    global truth_df, unique_wave_ids

    # --- Handle keyword usage (expected by app.py) ---
    if "truth_df" in kwargs:
        truth_df = kwargs.get("truth_df")
        unique_wave_ids = list(kwargs.get("unique_wave_ids", []))

    # --- Handle positional fallback (defensive) ---
    elif len(args) >= 2:
        truth_df = args[0]
        unique_wave_ids = list(args[1])

    else:
        raise TypeError("initialize_waves requires truth_df and unique_wave_ids")

    # --- Initialize container safely ---
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