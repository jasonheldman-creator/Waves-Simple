"""
analytics_truth.py

Canonical TruthFrame provider for WAVES Intelligence.

Read-only, diagnostic-safe.
NO trading logic
NO execution logic
NO side effects

This file intentionally provides stable, schema-complete defaults
to satisfy app.py, tests, and UI expectations.
"""

from datetime import datetime


class TruthFrame:
    """
    Minimal canonical TruthFrame.

    Acts as the single source of diagnostic truth for the application.
    """

    def __init__(self):
        self.generated_at = datetime.utcnow().isoformat() + "Z"
        self.mode = "diagnostic"
        self.status = "initialized"

        # Portfolio-level metrics (safe defaults)
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

        # Wave-level diagnostics (empty by default)
        self.waves = {}

    @property
    def empty(self):
        """
        Compatibility property expected by UI and guards.
        TruthFrame is always considered empty until populated.
        """
        return True

    def summary(self):
        return {
            "status": self.status,
            "generated_at": self.generated_at,
            "wave_count": len(self.waves),
        }


def get_truth_frame(*args, safe_mode=False, **kwargs):
    """
    Backward-compatible TruthFrame accessor.
    """
    return TruthFrame()


def compute_portfolio_snapshot_from_truth(truth_frame=None):
    """
    Diagnostic-safe portfolio snapshot builder.

    Restores interface expected by app.py and tests.
    No calculations performed.
    """
    tf = truth_frame or get_truth_frame()

    return {
        "computed_at_utc": tf.generated_at,
        "mode": tf.mode,
        "status": tf.status,

        "return_1d": tf.portfolio.get("1D"),
        "return_30d": tf.portfolio.get("30D"),
        "return_60d": tf.portfolio.get("60D"),
        "return_365d": tf.portfolio.get("365D"),

        "alpha_1d": tf.portfolio.get("alpha_1D"),
        "alpha_30d": tf.portfolio.get("alpha_30D"),
        "alpha_60d": tf.portfolio.get("alpha_60D"),
        "alpha_365d": tf.portfolio.get("alpha_365D"),

        "wave_count": len(tf.waves),
        "waves": tf.waves,
    }