import pandas as pd

# Assume other necessary imports and functions are here

def generate_live_snapshot_csv():
    # Call compute_history_nav with include_diagnostics=True
    result = compute_history_nav(include_diagnostics=True)
    
    # Extract diagnostics
    diagnostics = result.attrs["diagnostics"]

    # Prepare the data for live_snapshot.csv including new fields
    new_row = {
        'vix': diagnostics['vix'],
        'regime': diagnostics['regime'],
        'tilt_factor': diagnostics['tilt_factor'],
        'vix_exposure': diagnostics['vix_exposure'],
        'vol_adjust': diagnostics['vol_adjust'],
        'safe_fraction': diagnostics['safe_fraction'],
        'exposure': diagnostics['exposure'],
        'aggregated_risk_state': diagnostics['aggregated_risk_state']
    }

    # Assume there's existing logic to append new_row to a DataFrame and save to CSV
    output_df = pd.DataFrame([new_row])  # Example of how to create a DataFrame
    output_df.to_csv('live_snapshot.csv', mode='a', header=False, index=False)

# Assume the function may be called elsewhere