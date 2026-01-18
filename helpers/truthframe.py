class TruthFrame:
    def __init__(self):
        # Existing initialization code
        self.truth_df = ... # Assuming this initializes a DataFrame object
        self._build_waves()  # Invoking the new method

    def _build_waves(self):
        waves_universe = waves_engine.get_all_waves_universe()
        self.truth_df.waves = []
        for wave in waves_universe:
            wave_data = {
                'wave_id': wave['wave_id'],
                'health': wave['health'],
                'regime_alignment': wave['regime_alignment'],
                'learning_signals': wave['learning_signals']
            }
            self.truth_df.waves.append(wave_data)