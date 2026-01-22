"""
Portfolio Snapshot Helper

Builds portfolio-level returns and alpha across
1D, 30D, 60D, and 365D horizons from the canonical TruthFrame.

SAFE:
- Read-only
- Never raises
- No Streamlit dependencies
"""

from typing import Dict, Any
import pandas as pd


HORIZONS = {
    "1D": 1,
    "30D": 30,
    "60D": 60,
    "365D": 365,
}


def _safe_sum(values):
    try:
        return float(sum(v for v in values if v is not None))
    except Exception:
        return 0.0


def build_portfolio_snapshot_from_truthframe(
    truthframe: Dict[str, Any]
) -> pd.DataFrame:
    """
    Build a portfolio snapshot table from TruthFrame.

    Returns DataFrame with:
    - Horizon
    - Return
    - Alpha
    """

    rows = []

    waves = truthframe.get("waves", {})
    if not isinstance(waves, dict) or not waves:
        return pd.DataFrame()

    for label in HORIZONS.keys():
        returns = []
        alphas = []

        for wave_data in waves.values():
            perf = wave_data.get("performance", {})
            alpha = wave_data.get("alpha", {})

            returns.append(perf.get(f"return_{label}", 0.0))
            alphas.append(alpha.get(f"alpha_{label}", alpha.get("total", 0.0)))

        rows.append(
            {
                "Horizon": label,
                "Return": _safe_sum(returns),
                "Alpha": _safe_sum(alphas),
            }
        )

    return pd.DataFrame(rows)