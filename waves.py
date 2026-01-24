"""
waves.py — import-safe, app.py-compatible scaffold

This module is designed to be:
• Import-safe (no execution at import time)
• Compatible with frozen app.py
• Keyword-argument tolerant
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

    Accepts BOTH positional and keyword arguments to remain
    compatible with legacy app.py calls.

    Supported:
        initialize_waves(truth_df, unique_wave_ids)
        initialize_waves(truth_df=..., unique_wave_ids=...)
    """

    global truth_df, unique_wave_ids

    # --- Resolve arguments safely ---
    if args:
        if len(args) >= 1:
            truth_df = args[0]
        if len(args) >= 2:
            unique_wave_ids = list(args[1])
    else:
        truth_df = kwargs.get("truth_df")
        unique_wave_ids = list(kwargs.get("unique_wave_ids", []))

    # --- Guardrails ---
    if truth_df is None:
        raise ValueError("initialize_waves: truth_df is None")

    if not hasattr(truth_df, "waves"):
        truth_df.waves = {}

    # --- Initialize wave shells ---
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
    