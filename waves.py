"""
waves.py — import-safe, caller-compatible wave initializer

This module is hardened to support legacy and frozen callers
(including app.py) without requiring changes upstream.
"""

# -----------------------------
# Safe defaults (IMPORT SAFE)
# -----------------------------

unique_wave_ids = []
truth_df = None


# -----------------------------
# Public initializer (HARDENED)
# -----------------------------

def initialize_waves(*args, **kwargs):
    """
    Initialize wave entries inside truth_df safely.

    Supports ALL of the following call styles:

    initialize_waves(truth_df, wave_ids)
    initialize_waves(_truth_df=..., _unique_wave_ids=...)
    initialize_waves(truth_df=..., unique_wave_ids=...)

    This is required because app.py cannot be modified.
    """

    global truth_df, unique_wave_ids

    # --- Resolve truth_df ---
    if "truth_df" in kwargs:
        truth_df = kwargs["truth_df"]
    elif "_truth_df" in kwargs:
        truth_df = kwargs["_truth_df"]
    elif len(args) >= 1:
        truth_df = args[0]
    else:
        raise ValueError("initialize_waves: truth_df not provided")

    # --- Resolve wave IDs ---
    if "unique_wave_ids" in kwargs:
        unique_wave_ids = list(kwargs["unique_wave_ids"])
    elif "_unique_wave_ids" in kwargs:
        unique_wave_ids = list(kwargs["_unique_wave_ids"])
    elif len(args) >= 2:
        unique_wave_ids = list(args[1])
    else:
        unique_wave_ids = []

    # --- Initialize container ---
    if not hasattr(truth_df, "waves") or truth_df.waves is None:
        truth_df.waves = {}

    # --- Initialize waves ---
    for wave_id in unique_wave_ids:
        truth_df.waves.setdefault(
            wave_id,
            {
                "health": {
                    "score": None,
                    "alpha": None,
                    "beta_drift": None,
                    "volatility": None,
                    "exposure": None,
                },
                "regime_alignment": "neutral",
                "learning_signals": [],
            },
        )

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