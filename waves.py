"""
waves.py — import-safe recovery scaffold (COMPATIBILITY FIX)

This module must remain import-safe and must be compatible with
legacy app.py calls that pass keyword arguments:

    initialize_waves(truth_df=..., unique_wave_ids=...)

DO NOT execute logic at import time.
"""

from typing import Any, Iterable

# -----------------------------
# Safe defaults (IMPORT SAFE)
# -----------------------------

truth_df = None
unique_wave_ids = []


# -----------------------------
# Public initializer (BACKWARD COMPATIBLE)
# -----------------------------

def initialize_waves(
    _truth_df: Any = None,
    _unique_wave_ids: Iterable[str] | None = None,
    *,
    truth_df: Any = None,
    unique_wave_ids: Iterable[str] | None = None,
):
    """
    Initialize wave entries safely.

    Supports BOTH call styles:
      - initialize_waves(truth_df=..., unique_wave_ids=...)
      - initialize_waves(_truth_df, _unique_wave_ids)

    This is REQUIRED because app.py cannot be modified.
    """

    global truth_df as _global_truth_df
    global unique_wave_ids as _global_unique_wave_ids

    # Resolve arguments (keyword args take precedence)
    resolved_truth_df = truth_df if truth_df is not None else _truth_df
    resolved_wave_ids = (
        unique_wave_ids if unique_wave_ids is not None else _unique_wave_ids
    )

    if resolved_truth_df is None:
        raise ValueError("initialize_waves: truth_df is required")

    if resolved_wave_ids is None:
        raise ValueError("initialize_waves: unique_wave_ids is required")

    _global_truth_df = resolved_truth_df
    _global_unique_wave_ids = list(resolved_wave_ids)

    # Ensure waves container exists
    if not hasattr(_global_truth_df, "waves") or _global_truth_df.waves is None:
        _global_truth_df.waves = {}

    # Initialize wave entries
    for wave_id in _global_unique_wave_ids:
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
    return "waves module imported safely (compatible)"


# -----------------------------
# No execution at import time
# -----------------------------

if __name__ == "__main__":
    print("waves.py loaded directly (no execution performed)")