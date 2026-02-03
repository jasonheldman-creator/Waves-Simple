"""
analytics_truth.py

Canonical TruthFrame provider for WAVES Intelligence.

READ-ONLY
DIAGNOSTIC-SAFE
NO trading logic
NO execution logic
NO side effects

Purpose:
Expose the TRUE sources of alpha for IC review.
TruthFrame must reflect reality — not erase or blur it.
"""

from datetime import datetime
import os
import pandas as pd


ATTRIBUTION_PATH = "data/alpha_attribution_snapshot.csv"


class TruthFrame:
    """
    Canonical TruthFrame.

    Acts as the single source of diagnostic truth for the application.
    """

    def __init__(self):
        self.generated_at = datetime.utcnow().isoformat() + "Z"
        self.mode = "diagnostic"
        self.status = "initialized"

        # Portfolio-level metrics (unchanged, optional)
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

        # Wave-level attribution truth (IC-facing)
        self.waves = {}

        self._load_attribution_truth()

    # ---------------------------------------------------------
    # Attribution truth loading (READ-ONLY)
    # ---------------------------------------------------------

    def _load_attribution_truth(self):
        """
        Load wave-level attribution drivers from
        data/alpha_attribution_snapshot.csv.

        No recomputation. No scaling. No inference.
        """
        if not os.path.exists(ATTRIBUTION_PATH):
            self.status = "no_attribution_snapshot"
            return

        try:
            df = pd.read_csv(ATTRIBUTION_PATH)
        except Exception:
            self.status = "attribution_snapshot_unreadable"
            return

        if df.empty:
            self.status = "attribution_snapshot_empty"
            return

        for _, r in df.iterrows():
            wave_id = r.get("wave_id")
            if wave_id is None or str(wave_id).strip() == "":
                continue

            self.waves[wave_id] = {
                "wave_name": r.get("wave_name"),

                # -------------------------------------------------
                # TRUE SOURCES OF ALPHA (IC CANON)
                # -------------------------------------------------

                # A) Market / beta structure
                "alpha_market": r.get("alpha_market"),

                # B) Momentum / trend capture
                "alpha_momentum": r.get("alpha_momentum"),

                # C) Volatility / VIX / convexity control
                "alpha_volatility": r.get("alpha_vix"),

                # D) Rotation / allocation intelligence
                "alpha_rotation": r.get("alpha_rotation"),

                # E) Stock selection (idiosyncratic alpha)
                "alpha_stock_selection": r.get("alpha_stock_selection"),

                # -------------------------------------------------
                # Reconciliation only
                # -------------------------------------------------
                "alpha_total": r.get("alpha_total"),
            }

        self.status = "attribution_loaded"

    # ---------------------------------------------------------
    # Compatibility flags
    # ---------------------------------------------------------

    @property
    def empty(self):
        """
        TruthFrame is empty ONLY if no wave attribution exists.
        """
        return len(self.waves) == 0

    def summary(self):
        return {
            "status": self.status,
            "generated_at": self.generated_at,
            "wave_count": len(self.waves),
        }


# -------------------------------------------------------------
# Public accessors (backward compatible)
# -------------------------------------------------------------

def get_truth_frame(*args, safe_mode=False, **kwargs):
    """
    Backward-compatible TruthFrame accessor.
    """
    return TruthFrame()


def compute_portfolio_snapshot_from_truth(truth_frame=None):
    """
    Diagnostic-safe portfolio snapshot builder.

    TruthFrame now exposes full IC-grade attribution drivers.
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