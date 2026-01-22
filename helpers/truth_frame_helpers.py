def normalize_horizons(horizons: dict) -> dict:
    """Ensure that the dictionary contains keys '1D', '30D', '60D', and '365D'. If any of these keys are missing, default their values to 0.0. Existing keys and values must be preserved unchanged."""
    for key in ['1D', '30D', '60D', '365D']:
        if key not in horizons:
            horizons[key] = 0.0
    return horizons
