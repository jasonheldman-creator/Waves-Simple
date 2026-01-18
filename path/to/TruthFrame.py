class TruthFrame:
    def __init__(self):
        # Portfolio-level metrics
        self.one_day_return = None
        self.thirty_day_return = None
        self.sixty_day_return = None
        self.three_sixty_five_day_return = None
        self.alpha_metrics = {
            '1_day': None,
            '30_day': None,
            '60_day': None,
            '365_day': None
        }

        # Waves structure
        self.waves = []  # list of structured objects

        # Regime Intelligence Object
        self.regime = {
            'current_regime': None,
            'aligned_waves': [],
            'misaligned_waves': []
        }

        # Default Attributes
        self.portfolio = None  # Placeholder for portfolio object
        self.learning_signals = []  # empty learning signals list
        self.status = 'initialized'  # default status

    # Add additional methods and logic as per requirements