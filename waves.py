for wave_id in unique_wave_ids:
    if wave_id not in truth_df.waves:
        truth_df.waves[wave_id] = {
            "health": {
                "score": None,
                "alpha": None,
                "beta_drift": None,
                "volatility": None,
                "exposure": None
            },
            "regime_alignment": 'neutral',
            "learning_signals": []
        }