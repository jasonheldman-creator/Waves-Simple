import numpy as np
from typing import List, Dict, Any, Iterable


def compute_wave_returns_pipeline(
    wave_data: Iterable[Any],
    horizons: List[Any],
) -> List[Dict[str, Any]]:
    """
    Compute wave returns, benchmark returns, and alpha independently
    for each horizon.

    Design rules (IMPORTANT — do not regress):
    - Horizons are computed independently
    - Insufficient data → emit NaN (NOT exclusion)
    - A wave is excluded ONLY if ALL horizons are unavailable
    - Output is one row per wave per horizon

    This function is intentionally horizon-tolerant to avoid
    accidental wave exclusion and DEGRADED system state.
    """

    results: List[Dict[str, Any]] = []

    for wave in wave_data:
        any_horizon_available = False
        wave_rows: List[Dict[str, Any]] = []

        for horizon in horizons:
            if has_sufficient_data(wave, horizon):
                wave_return = calculate_wave_return(wave, horizon)
                benchmark_return = calculate_benchmark_return(wave, horizon)
                alpha = wave_return - benchmark_return
                any_horizon_available = True
            else:
                wave_return = np.nan
                benchmark_return = np.nan
                alpha = np.nan

            wave_rows.append(
                {
                    "wave": wave,
                    "horizon": horizon,
                    "wave_return": wave_return,
                    "benchmark_return": benchmark_return,
                    "alpha": alpha,
                }
            )

        # Exclude wave ONLY if no horizons had sufficient data
        if not any_horizon_available:
            continue

        results.extend(wave_rows)

    return results