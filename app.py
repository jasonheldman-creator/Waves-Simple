@st.cache_data(show_spinner=True)
def load_wave_weights():
    """
    Load Wave allocations from wave_weights.csv.

    Expected format (long):

        Ticker,Wave,Weight
        NVDA,SP500_Wave,0.1486
        AAPL,SP500_Wave,0.1388
        ...

    Comment lines starting with '#' are ignored.
    """
    if not os.path.exists(WAVE_WEIGHTS_CSV):
        return None, f"[WAVE ERROR] '{WAVE_WEIGHTS_CSV}' not found."

    try:
        # 'comment' tells pandas to ignore any line starting with '#'
        df = pd.read_csv(WAVE_WEIGHTS_CSV, comment="#")
    except Exception as e:
        return None, f"[WAVE ERROR] Failed to read '{WAVE_WEIGHTS_CSV}': {e}"

    if df.empty:
        return None, f"[WAVE ERROR] '{WAVE_WEIGHTS_CSV}' is empty."

    # We expect exactly these columns
    required_cols = {"Ticker", "Wave", "Weight"}
    missing = required_cols - set(df.columns)
    if missing:
        return None, (
            f"[WAVE ERROR] Missing column(s) {missing} in '{WAVE_WEIGHTS_CSV}'. "
            f"Found columns: {list(df.columns)}"
        )

    df_long = df.copy()

    # Ensure numeric weights
    df_long["Weight"] = pd.to_numeric(df_long["Weight"], errors="coerce").fillna(0.0)

    # Drop zero or negative weights
    df_long = df_long[df_long["Weight"] > 0]

    if df_long.empty:
        return None, f"[WAVE ERROR] All weights in '{WAVE_WEIGHTS_CSV}' are zero."

    # Normalize weights within each Wave so they sum to 1.0
    total_by_wave = df_long.groupby("Wave")["Weight"].transform("sum")
    total_by_wave = total_by_wave.replace(0, 1.0)
    df_long["Weight"] = df_long["Weight"] / total_by_wave

    return df_long, None
