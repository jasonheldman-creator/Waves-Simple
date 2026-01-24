"""
waves.py — import-safe recovery scaffold (keyword-compatible)

This module is intentionally import-safe and designed to be compatible
with existing app.py calls that pass keyword arguments.

NO execution occurs at import time.
"""

# -----------------------------
# Safe defaults (IMPORT SAFE)
# -----------------------------

unique_wave_ids = []
truth_df = None


# -----------------------------
# Public initializer
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

    Supports BOTH positional and keyword arguments to remain compatible
    with existing app.py usage.

    Accepted:
      initialize_waves(truth_df=..., unique_wave_ids=...)
      initialize_waves(_truth_df, _unique_wave_ids)
    """

    global truth_df as _global_truth_df
    global unique_wave_ids as _global_wave_ids

    # Resolve arguments (keyword wins over positional)
    resolved_truth_df = truth_df if truth_df is not None else _truth_df
    resolved_wave_ids = (
        list(unique_wave_ids)
        if unique_wave_ids is not None
        else list(_unique_wave_ids or [])
    )

    if resolved_truth_df is None:
        raise ValueError("initialize_waves: truth_df is required")

    # Store globals (for downstream reads, not execution)
    _global_truth_df = resolved_truth_df
    _global_wave_ids = resolved_wave_ids

    # Ensure waves container exists
    if not hasattr(resolved_truth_df, "waves"):
        resolved_truth_df.waves = {}

    # Initialize wave records
    for wave_id in resolved_wave_ids:
        if wave_id not in resolved_truth_df.waves:
            resolved_truth_df.waves[wave_id] = {
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

    return resolved_truth_df.waves


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