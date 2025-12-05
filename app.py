import streamlit as st
import pandas as pd

st.set_page_config(page_title="Waves Intelligence – Portfolio Wave Console", layout="wide")

st.title("WAVES INTELLIGENCE™ – PORTFOLIO WAVE CONSOLE")
st.caption("Equity Waves only – benchmark-aware, AI-directed, multi-mode demo.")

# -------------------------------------------------------------------
# FILE NAMES — EXACT MATCH TO YOUR GITHUB FILES
# -------------------------------------------------------------------
UNIVERSE_FILE = "Master_Stock_Sheet.csv - Sheet5.csv"
WEIGHTS_FILE = "WaveWeight-Sheet1.csv - Sheet1.csv"
LIST_FILE = "list.csv"

# -------------------------------------------------------------------
# LOAD UNIVERSE
# -------------------------------------------------------------------
try:
    universe = pd.read_csv(UNIVERSE_FILE)
except:
    st.error(
        f"""
        Universe file not found.

        I looked for:

        • **{UNIVERSE_FILE}**

        Please confirm the exact file name in your repo.
        """
    )
    st.stop()

# -------------------------------------------------------------------
# LOAD WAVE WEIGHTS
# -------------------------------------------------------------------
try:
    weights = pd.read_csv(WEIGHTS_FILE)
except:
    st.error(
        f"""
        Wave weight file not found.

        I looked for:

        • **{WEIGHTS_FILE}**

        Make sure the uploaded file name matches exactly.
        """
    )
    st.stop()

# -------------------------------------------------------------------
# LOAD LIST CSV (symbol map)
# -------------------------------------------------------------------
try:
    symbol_list = pd.read_csv(LIST_FILE)
except:
    st.error(
        f"""
        List file not found.

        I looked for:

        • **{LIST_FILE}**
        """
    )
    st.stop()

# -------------------------------------------------------------------
# CLEAN WEIGHTS FILE (remove quotes)
# -------------------------------------------------------------------
weights.columns = [c.replace('"','') for c in weights.columns]
for col in weights.columns:
    weights[col] = weights[col].astype(str).str.replace('"','')

# Force proper column names
weights.columns = ["Wave", "Ticker", "Weight"]

# Weight column to float
weights["Weight"] = weights["Weight"].astype(float)

# -------------------------------------------------------------------
# UI – wave selector
# -------------------------------------------------------------------
waves = sorted(weights["Wave"].unique())
selected_wave = st.selectbox("Select a Wave to view:", waves)

wave_df = weights[weights["Wave"] == selected_wave]

st.subheader(f"Wave Holdings – {selected_wave}")
st.dataframe(wave_df)

# -------------------------------------------------------------------
# Join with universe to show full company details
# (works automatically with your Master_Stock_Sheet.csv - Sheet5.csv)
# -------------------------------------------------------------------
if "Ticker" in universe.columns:
    merged = wave_df.merge(universe, on="Ticker", how="left")
    st.subheader("Merged With Universe (company data)")
    st.dataframe(merged)
else:
    st.warning("Ticker column not found in universe file — cannot merge.")

st.success("Console loaded successfully.")