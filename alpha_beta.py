"""
alpha_beta.py — SAFE alpha / beta attribution module

This module is:
• Import-safe (no execution at import time)
• Stateless by default
• Defensive against missing or malformed data

Nothing in this file will execute unless explicitly called.
"""

from typing import Dict, Any, Optional


# -------------------------------------------------------------------
# Public API
# -------------------------------------------------------------------

def compute_alpha_beta(
    wave_returns: Dict[str, float],
    benchmark_returns: Dict[str, float],
) -> Dict[str, Optional[float]]:
    """
    Compute simple alpha and beta attribution.

    Parameters
    ----------
    wave_returns : dict
        Mapping of horizon -> wave return (e.g. {"30D": 0.04})
    benchmark_returns : dict
        Mapping of horizon -> benchmark return

    Returns
    -------
    dict
        {
            "alpha": float | None,
            "beta": float | None
        }
    """

    try:
        if not wave_returns or not benchmark_returns:
            return {"alpha": None, "beta": None}

        # Use overlapping horizons only
        common_keys = set(wave_returns.keys()) & set(benchmark_returns.keys())
        if not common_keys:
            return {"alpha": None, "beta": None}

        # Simple averages (safe, transparent, debuggable)
        wave_avg = sum(float(wave_returns[k]) for k in common_keys) / len(common_keys)
        bench_avg = sum(float(benchmark_returns[k]) for k in common_keys) / len(common_keys)

        alpha = wave_avg - bench_avg

        # Naive beta proxy (can be upgraded later)
        beta = None
        if bench_avg != 0:
            beta = wave_avg / bench_avg

        return {
            "alpha": round(alpha, 6),
            "beta": round(beta, 6) if beta is not None else None,
        }

    except Exception:
        # HARD RULE: never raise
        return {"alpha": None, "beta": None}


# -------------------------------------------------------------------
# Diagnostic helper (optional)
# -------------------------------------------------------------------

def module_healthcheck() -> str:
    """
    Lightweight confirmation hook.
    Safe to call from recovery / diagnostics views.
    """
    return "alpha_beta module loaded safely"