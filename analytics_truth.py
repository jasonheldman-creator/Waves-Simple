"""
analytics_truth.py

Canonical TruthFrame provider for WAVES Intelligence.

This module defines the minimal, read-only TruthFrame used by the
application for diagnostics, monitoring, and UI rendering.

NO trading logic
NO execution logic
NO side effects

This file intentionally provides stable, schema-complete defaults.
"""

from datetime import datetime


class TruthFrame:
    """
    Minimal canonical TruthFrame.

    Acts as the single source of diagnostic truth for the application.
    All values are intentionally safe defaults unless populated elsewhere.
    """

    def __init__(self):
        self.generated_at = datetime.utcnow().isoformat() + "Z"
        self.mode = "diagnostic"
        self.status = "initialized"

        # Portfolio-level metrics (intentionally None by default)
        self.portfolio = {
            "1D": None,
            "30D": None,
            "60D": None,
            "365D": None,
            "alpha_1D": None,
            "alpha_30D": None,
            "alpha_60D": None,
            "alpha_365D": None,
        }

        # Wave-level diagnostics (optional, empty by default)
        self.waves = {}

    # ------------------------------------------------------------------
    # Compatibility shims (DataFrame-like behavior expected by app.py)
    # ------------------------------------------------------------------

    @property
    def empty(self):
        """
        Pandas-compatibility shim.

        Some legacy code checks `truth_frame.empty`.
        TruthFrame is considered empty when no waves are present.
        """
        return len(self.waves) == 0

    def __bool__(self):
        """
        TruthFrame should always evaluate as True to avoid
        accidental falsy checks breaking execution.
        """
        return True

    # ------------------------------------------------------------------

    def summary(self):
        """
        Lightweight summary for UI or logging.
        """
        return {
            "status": self.status,
            "generated_at": self.generated_at,
            "wave_count": len(self.waves),
        }


def get_truth_frame(*args, safe_mode=False, **kwargs):
    """
    Backward-compatible TruthFrame accessor.

    Args:
        safe_mode (bool): Accepted for compatibility with app.py (unused)

    Returns:
        TruthFrame instance
    """
    return TruthFrame()


def compute_portfolio_snapshot_from_truth(truth_frame=None):
    """
    Minimal, diagnostic-safe portfolio snapshot builder.

    Restores the interface expected by app.py and test files.
    NO calculations are performed.

    Returns:
        dict: Schema-complete portfolio snapshot
    """
    tf = truth_frame or get_truth_frame()

    return {
        # --- metadata ---
        "computed_at_utc": tf.generated_at,
        "mode": tf.mode,
        "status": tf.status,

        # --- returns ---
        "return_1d": tf.portfolio.get("1D"),
        "return_30d": tf.portfolio.get("30D"),
        "return_60d": tf.portfolio.get("60D"),
        "return_365d": tf.portfolio.get("365D"),

        # --- alpha ---
        "alpha_1d": tf.portfolio.get("alpha_1D"),
        "alpha_30d": tf.portfolio.get("alpha_30D"),
        "alpha_60d": tf.portfolio.get("alpha_60D"),
        "alpha_365d": tf.portfolio.get("alpha_365D"),

        # --- diagnostics ---
        "wave_count": len(tf.waves),
        "waves": tf.waves,
    }