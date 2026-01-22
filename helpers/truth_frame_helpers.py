def normalize_horizons(horizons):
    # Ensure the horizons contain valid values for 1D, 30D, 60D, and 365D
    defaults = {'1D': 0.0, '30D': 0.0, '60D': 0.0, '365D': 0.0}
    normalized_horizons = {key: horizons.get(key, defaults[key]) for key in defaults.keys()}
    return normalized_horizons

# Call this function just before TruthFrame emission
# Assuming 'wave' is the wave instance and it has a method that emits the TruthFrame
wave_horizons = wave.get_horizons()   # This line depends on the actual structure of your code
normalized_horizons = normalize_horizons(wave_horizons)
wave.emit_truth_frame(normalized_horizons)  # Replace with your actual emit method where necessary
