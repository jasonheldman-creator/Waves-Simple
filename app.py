# Format alpha strings (handle NaN values from TruthFrame)

    alpha_1d_str = f"{alpha_1d:+.2%}" if alpha_1d is not None and not pd.isna(alpha_1d) else "N/A"
    alpha_30d_str = f"{alpha_30d:+.2%}" if alpha_30d is not None and not pd.isna(alpha_30d) else "N/A"
    alpha_60d_str = f"{alpha_60d:+.2%}" if alpha_60d is not None and not pd.isna(alpha_60d) else "N/A"
    alpha_365d_str = f"{alpha_365d:+.2%}" if alpha_365d is not None and not pd.isna(alpha_365d) else "N/A"

# Remaining content of app.py... (unchanged)