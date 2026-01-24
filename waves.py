"""
waves.py — import-safe recovery scaffold (FINAL)

Compatible with app.py calling:
    initialize_waves(truth_df=..., unique_wave_ids=...)

Safe for Streamlit import and redeploy.
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
    backward-compatible with locked app.py.
    """

    global truth_df, unique_wave_ids

    # --- Accept keyword arguments (PRIMARY PATH) ---
    if "truth_df" in kwargs:
        truth_df = kwargs.get("truth_df")
        unique_wave_ids = list(kwargs.get("unique_wave_ids", []))

    # --- Accept positional arguments (FALLBACK) ---
    elif len(args) >= 2:
        truth_df = args[0]
        unique_wave_ids = list(args[1])

    else:
        raise TypeError(
            "initialize_waves requires truth_df and unique_wave_ids"
        )

    # --- Ensure container exists ---
    if not hasattr(truth_df, "waves"):
        truth_df.waves = {}

    # --- Initialize waves ---
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