"""
alpha_attribution_adapter.py

Governance-safe alpha attribution adapter.

Purpose
-------
Convert REAL, already-computed wave returns and diagnostics into a
fully reconciled, six-source alpha attribution that explains overlay alpha
WITHOUT fabricating returns or altering economics.

This adapter NEVER changes returns.
It only decomposes observed overlay alpha.
"""

from typing import Dict, List
import pandas as pd
import numpy as np


ALPHA_SOURCES = [
    "selection",
    "momentum",
    "volatility",
    "beta",
    "allocation",
    "residual",
]


class AlphaAttributionAdapter:
    def __init__(
        self,
        *,
        wave_return: float,
        raw_wave_return: float,
        benchmark_return: float,
        diagnostics: Dict[str, float],
        horizon_days: int,
    ):
        """
        Parameters
        ----------
        wave_return : float
            Strategy-adjusted wave return (WITH overlays)
        raw_wave_return : float
            Raw wave return (NO overlays, pure selection)
        benchmark_return : float
            Composite benchmark return
        diagnostics : dict
            diagnostics from compute_history_nav attrs
        horizon_days : int
            Attribution horizon (e.g. 30, 60, 365)
        """

        self.wave_return = float(wave_return)
        self.raw_wave_return = float(raw_wave_return)
        self.benchmark_return = float(benchmark_return)
        self.diagnostics = diagnostics or {}
        self.horizon_days = horizon_days

        # Core alphas
        self.total_alpha = self.wave_return - self.benchmark_return
        self.selection_alpha = self.raw_wave_return - self.benchmark_return
        self.overlay_alpha = self.wave_return - self.raw_wave_return

    # ------------------------------------------------------------------
    # PUBLIC API
    # ------------------------------------------------------------------

    def build_attribution_row(self) -> Dict[str, float]:
        """
        Returns a fully reconciled attribution dictionary for this horizon.
        """

        overlay_components = self._decompose_overlay_alpha()

        row = {
            f"alpha_total_{self.horizon_days}d": self.total_alpha,
            f"alpha_selection_{self.horizon_days}d": self.selection_alpha,
        }

        for source, value in overlay_components.items():
            row[f"alpha_{source}_{self.horizon_days}d"] = value

        # Reconciliation check (for sanity & debugging)
        row[f"alpha_overlay_{self.horizon_days}d"] = self.overlay_alpha
        row[f"alpha_residual_{self.horizon_days}d"] = overlay_components["residual"]

        return row

    # ------------------------------------------------------------------
    # INTERNAL LOGIC
    # ------------------------------------------------------------------

    def _decompose_overlay_alpha(self) -> Dict[str, float]:
        """
        Allocate observed overlay alpha across causal sources
        using diagnostics as weights (NOT returns).
        """

        if abs(self.overlay_alpha) < 1e-12:
            return {k: 0.0 for k in ALPHA_SOURCES}

        weights = self._compute_overlay_weights()
        allocated = {}

        used = 0.0
        for source in ALPHA_SOURCES[:-1]:
            allocated[source] = self.overlay_alpha * weights[source]
            used += allocated[source]

        # Residual guarantees reconciliation
        allocated["residual"] = self.overlay_alpha - used

        return allocated

    def _compute_overlay_weights(self) -> Dict[str, float]:
        """
        Convert diagnostics into normalized allocation weights.
        """

        raw_weights = {
            "momentum": abs(self._get("tilt_factor")),
            "volatility": abs(self._get("vix_exposure")) + abs(self._get("vol_adjust")),
            "beta": abs(self._get("exposure") - 1.0),
            "allocation": abs(self._get("safe_fraction")),
            "selection": 0.0,  # selection handled separately
            "residual": 0.0,
        }

        total = sum(raw_weights.values())

        # Fallback: equal allocation if diagnostics are flat
        if total <= 0.0:
            equal = 1.0 / (len(ALPHA_SOURCES) - 1)
            return {
                "momentum": equal,
                "volatility": equal,
                "beta": equal,
                "allocation": equal,
                "selection": 0.0,
                "residual": 0.0,
            }

        return {
            k: (v / total if k != "selection" else 0.0)
            for k, v in raw_weights.items()
        }

    def _get(self, key: str, default: float = 0.0) -> float:
        try:
            return float(self.diagnostics.get(key, default))
        except Exception:
            return default