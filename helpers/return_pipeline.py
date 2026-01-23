import numpy as np


def compute_wave_returns_pipeline(wave_data, horizons):
    """
    Compute wave returns, benchmark returns, and alpha independently
    for each horizon.

    Rules:
    - Each horizon is computed independently
    - Insufficient data → emit NaN (not exclusion)
    - A wave is excluded ONLY if ALL horizons are unavailable
    - Output is one row per wave per horizon
    """

    results = []

    for wave in wave_data:
        any_horizon_available = False
        wave_rows = []

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

            wave_rows.append({
                "wave": wave,
                "horizon": horizon,
                "wave_return": wave_return,
                "benchmark_return": benchmark_return,
                "alpha": alpha,
            })

        # Exclude wave ONLY if no horizons had sufficient data
        if not any_horizon_available:
            continue

        results.extend(wave_rows)

    return results