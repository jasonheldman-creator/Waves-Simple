import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="WAVES Intelligence – Portfolio Wave Console", layout="wide")

# ------------------------------------------------------------
# 1. Load the master universe (5,000-stock basket)
# ------------------------------------------------------------

UNIVERSE_FILE = "Master_Stock_Sheet5.csv"   # <<<<<<<<<<<<<< THIS IS NOW THE ONLY FILE IT LOOKS FOR

@st.cache_data
def load_universe():
    try:
        df = pd.read_csv(UNIVERSE_FILE)
        return df
    except Exception as e:
        st.error(f"Cannot load {UNIVERSE_FILE}. Make sure it is uploaded to the repo root.")
        st.stop()

universe_df = load_universe()

# ------------------------------------------------------------
# 2. Load Wave snapshot file (your positions)
# ------------------------------------------------------------

uploaded_file = st.sidebar.file_uploader("Upload Wave snapshot CSV (your holdings)", type=["csv"])

def load_wave_snapshot(uploaded_file):
    try:
        df = pd.read_csv(uploaded_file)
        return df
    except:
        return None

wave_df = None
if uploaded_file:
    wave_df = load_wave_snapshot(uploaded_file)

# ------------------------------------------------------------
# 3. Sidebar Controls
# ------------------------------------------------------------

st.sidebar.markdown("### Select Wave")
wave_list = [
    "S&P 500 Wave",
    "Small-Cap Growth Wave",
    "Small/Mid-Cap Value Wave",
    "Future Power & Energy Wave",
    "Equity Income Wave",
    "RWA Income Wave",
    "Crypto Income Wave* (not supported here)",
]
selected_wave = st.sidebar.selectbox("", wave_list, index=0)

st.sidebar.markdown("### Mode")
mode = st.sidebar.radio("", ["Standard", "Alpha-Minus Beta", "Private Logic"])

# ------------------------------------------------------------
# 4. Main Screen
# ------------------------------------------------------------

st.title("WAVES INTELLIGENCE™ – PORTFOLIO WAVE CONSOLE")
st.subheader(f"{selected_wave} (LIVE Demo)")

if wave_df is None:
    st.warning("Upload a Wave snapshot CSV to continue.")
    st.stop()

# ---------------------------
# Build Top-10 holdings
# ---------------------------

if "Weight" not in wave_df.columns:
    st.error("Snapshot file must contain a 'Weight' column.")
    st.stop()

wave_df = wave_df.sort_values("Weight", ascending=False)
top10 = wave_df.head(10)

left, right = st.columns([2,2])

with left:
    st.markdown("### Top 10 holdings")
    st.dataframe(top10[["Ticker", "Weight"]])

with right:
    st.markdown("### Top-10 by Wave weight")
    st.bar_chart(top10.set_index("Ticker")["Weight"])

# ---------------------------
# Sector Allocation (if exists)
# ---------------------------

st.markdown("### Sector allocation")

if "Sector" in wave_df.columns:
    sector_alloc = wave_df.groupby("Sector")["Weight"].sum()
    st.bar_chart(sector_alloc)
else:
    st.info("Add a 'Sector' column to your snapshot to view sector allocation.")

# ---------------------------
# Weight Decay Curve
# ---------------------------

st.markdown("### Weight decay curve")

st.line_chart(wave_df["Weight"].reset_index(drop=True))

# ---------------------------
# Universe Diagnostics
# ---------------------------

st.markdown("### Universe diagnostics")
st.write(f"Universe loaded with **{len(universe_df):,} stocks**.")