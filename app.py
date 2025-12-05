import pandas as pd
import streamlit as st

# Set correct weights CSV name
WEIGHTS_FILE = "WaveWeight-Sheet1.csv - Sheet1.csv"
# -------------------------------------------------------------------
# LOAD WAVE WEIGHTS (auto-fix malformed CSV)
# -------------------------------------------------------------------
try:
    raw = pd.read_csv(WEIGHTS_FILE, header=None)
except Exception as e:
    st.error(f"Unable to load weights file: {WEIGHTS_FILE}\n\n{e}")
    st.stop()

# Clean quotes
raw = raw.applymap(lambda x: str(x).replace('"',''))

# Split the single-column row into Wave, Ticker, Weight
split_cols = raw[0].str.split(",", expand=True)
split_cols.columns = ["Wave", "Ticker", "Weight"]

# Convert weight to float
split_cols["Weight"] = split_cols["Weight"].astype(float)

weights = split_cols