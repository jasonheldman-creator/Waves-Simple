def detect_learning_signals(truth_df):
    """Add your docstring here"""
    import pandas as pd

    # Ensure truth_df is a DataFrame exactly once, right after get_truth_frame.
    truth_df = get_truth_frame(safe_mode=True)
    if isinstance(truth_df, list):
        truth_df = pd.DataFrame(truth_df)

    signals = []
    # ... (rest of the function logic remains unchanged) ...