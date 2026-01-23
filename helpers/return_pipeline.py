def compute_wave_returns_pipeline(wave_data, horizons):
    results = []
    for wave in wave_data:
        wave_returns = {'wave_return': None, 'benchmark_return': None, 'alpha': None}
        all_horizons_available = True

        for horizon in horizons:
            # Check if there is sufficient data for the horizon
            if has_sufficient_data(wave, horizon):
                # Calculate returns and update the results
                wave_returns['wave_return'] = calculate_wave_return(wave, horizon)
                wave_returns['benchmark_return'] = calculate_benchmark_return(wave, horizon)
                wave_returns['alpha'] = wave_returns['wave_return'] - wave_returns['benchmark_return']
            else:
                # Emit NaN for insufficient data
                wave_returns['wave_return'] = np.nan
                wave_returns['benchmark_return'] = np.nan
                wave_returns['alpha'] = np.nan
                all_horizons_available = False

        # If all horizons are available, include the wave
        if not all_horizons_available:
            if is_any_data_available(wave_returns):
                results.append(wave_returns)
            # If no data available for any horizon, exclude the wave completely
        else:
            results.append(wave_returns)

    return results

# Ensure integrity of existing functions and comments remain unchanged.