def load_weights(csv_path: str) -> pd.DataFrame:
    """
    Ultra-safe loader for wave_weights.csv.

    Expected headers (case-insensitive, any order):
        Wave, Ticker, Weight
    """

    if not os.path.exists(csv_path):
        st.error(f"weights file not found at: {csv_path}")
        return pd.DataFrame()

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        st.error(f"Could not read weights file `{csv_path}`: {e}")
        return pd.DataFrame()

    # DEBUG: show columns
    st.write("DEBUG — wave_weights.csv columns:", list(df.columns))

    # Build lowercase column map
    col_map = {c.strip().lower(): c for c in df.columns}

    required = ["wave", "ticker", "weight"]
    missing = [r for r in required if r not in col_map]

    if missing:
        st.error(f"wave_weights.csv missing required columns: {missing}")
        return pd.DataFrame()

    # Select original columns via the map
    df = df[[col_map["wave"], col_map["ticker"], col_map["weight"]]].copy()
    df.columns = ["wave", "ticker", "weight"]

    # FIXED — correct string operations
    df["wave"] = df["wave"].astype(str).str.strip()
    df["ticker"] = df["ticker"].astype(str).str.strip().str.upper()

    # Weight must be numeric
    df["weight"] = pd.to_numeric(df["weight"], errors="coerce").fillna(0.0)

    # Normalize weights per wave
    df["weight"] = df.groupby("wave")["weight"].transform(
        lambda x: x / x.sum() if x.sum() != 0 else x
    )

    # Drop any invalid rows
    df = df[(df["ticker"] != "") & (df["weight"] > 0)]

    if df.empty:
        st.error("wave_weights.csv has no usable rows after cleaning.")

    return df
