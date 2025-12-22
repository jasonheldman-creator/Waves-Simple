import os
import pandas as pd
import streamlit as st
from some_module import bootstrap_wave_history

# On app startup, check if wave_history.csv exists, is empty, or has only a header
wave_history_path = 'data/wave_history.csv'
def is_wave_history_invalid(filepath):
    if not os.path.exists(filepath):
        return True
    try:
        df = pd.read_csv(filepath)
        if df.empty or (len(df.columns) > 0 and len(df) == 0):  # Handles header-only file
            return True
    except Exception:
        return True
    return False

if is_wave_history_invalid(wave_history_path):
    bootstrap_wave_history()
    st.warning("Using synthetic wave history seed data due to invalid or missing wave_history.csv.")

# Your existing app logic goes here...
# Example Plotly chart rendering inside a loop
for wave in waves_list:  # Assuming waves_list is defined somewhere in your app
    wave_id = wave['id']  # Extract a unique identifier for the wave
    st.plotly_chart(wave['chart'], key=f"plot_{wave_id}_main")  # Unique key for each plot