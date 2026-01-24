"""
attribution_engine.py — SAFE attribution orchestration layer

Purpose:
- Consume alpha/beta computation logic
- Operate ONLY when explicitly called
- Never execute at import time
- Never touch Streamlit directly

This module is intentionally defensive.
"""

from typing import Dict, Any

# Import is safe because alpha_beta.py is import-safe
from alpha_beta import compute_alpha_beta


# ==========================================================
# PUBLIC API
# ==========================================================

def build_alpha_beta_attribution(
    truthframe: Dict[str, Any],
    horizon: str = "365D"
) -> Dict[str, Any]:
    """
    Build alpha/beta attribution across all waves.

    Parameters:
        truthframe: dict containing wave data
        horizon: time horizon (e.g. "1D", "30D", "60D", "365D")

    Returns:
        dict keyed by wave_id with alpha/beta metrics
    """

    results = {}

    if not truthframe or "waves" not in truthframe:
        return results

    waves = truthframe.get("waves", {})

    for wave_id, wave_data in waves.items():
        try:
            performance = wave_data.get("performance", {})
            horizon_data = performance.get(horizon, {})

            portfolio_return = horizon_data.get("return")
            benchmark_return = horizon_data.get("benchmark_return")

            metrics = compute_alpha_beta(
                portfolio_return=portfolio_return,
                benchmark_return=benchmark_return
            )

            results[wave_id] = {
                "horizon": horizon,
                "alpha": metrics["alpha"],
                "beta": metrics["beta"],
                "portfolio_return": portfolio_return,
                "benchmark_return": benchmark_return,
            }

        except Exception:
            # HARD RULE: attribution must never crash caller
            results[wave_id] = {
                "horizon": horizon,
                "alpha": None,
                "beta": None,
                "portfolio_return": None,
                "benchmark_return": None,
                "error": "attribution_failed",
            }

    return results


# ==========================================================
# IMPORT SAFETY CONFIRMATION
# ==========================================================

def _import_check():
    return "attribution_engine imported safely"


if __name__ == "__main__":
    print("attribution_engine.py loaded directly (no execution)")