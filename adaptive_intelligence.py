def detect_learning_signals(truth_df):
    """Add your docstring here"""
    # Ensure truth_df is a DataFrame
    import pandas as pd
    if isinstance(truth_df, list):
        truth_df = pd.DataFrame(truth_df)
    signals = []
    # ... (rest of the function logic remains unchanged) ...
