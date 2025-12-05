import streamlit as st
import pandas as pd

# --------------------------------------------------------------------
# File names – must match EXACTLY the files in your repo root
# --------------------------------------------------------------------
UNIVERSE_FILE = "Master_Stock_Sheet.csv - Sheet5.csv"
WEIGHTS_FILE  = "WaveWeight-Sheet1.csv - Sheet1.csv"

# --------------------------------------------------------------------
# Page setup
# --------------------------------------------------------------------
st.set_page_config(
    page_title="WAVES Intelligence – Portfolio Wave Console",
    layout="wide",
)

st.title("WAVES INTELLIGENCE™ – PORTFOLIO WAVE CONSOLE")
st.caption("Equity Waves only – benchmark-aware, AI-directed, multi-mode demo.")

# --------------------------------------------------------------------
# Data loading helpers
# --------------------------------------------------------------------
@st.cache_data
def load_universe(path: str) -> pd.DataFrame:
    """
    Load the master stock universe from the CSV exported from Google Sheets.
    Needs at least a 'Ticker' column, optionally 'Sector'.
    """
    df = pd.read_csv(path, on_bad_lines="skip", low_memory=False)

    if "Ticker" not in df.columns:
        raise ValueError(f"Universe file {path} must contain a 'Ticker' column.")

    # Normalize ticker text
    df["Ticker"] = df["Ticker"].astype(str).str.strip().str.upper()
    return df


@st.cache_data
def load_weights(path: str) -> pd.DataFrame:
    """
    Load Wave / Ticker / Weight mappings.
    Your file looks like:
        "Wave,Ticker,Weight"
        "S&P 500 Wave,NVDA,0.10"
        ...
    so we read it with header=None and drop the first row.
    """
    raw = pd.read_csv(
        path,
        header=None,
        names=["Wave", "Ticker", "Weight"],
        quotechar='"',
        on_bad_lines="skip",
    )

    # Drop the header row that was read as data ("Wave,Ticker,Weight")
    mask_header = raw["Wave"].astype(str).str.strip().str.lower() == "wave"
    weights = raw[~mask_header].copy()

    # Clean up text and types
    weights["Wave"] = weights["Wave"].astype(str).str.strip()
    weights["Ticker"] = weights["Ticker"].astype(str).str.strip().str.upper()
    weights["Weight"] = pd.to_numeric(weights["Weight"], errors="coerce")

    # Drop any rows without a ticker or weight
    weights = weights.dropna(subset=["Ticker", "Weight"])

    if weights.empty:
        raise ValueError(f"No usable rows found in weights file {path}.")

    return weights


# --------------------------------------------------------------------
# Try to load both files – stop early with a clear message if anything fails
# --------------------------------------------------------------------
try:
    universe_df = load_universe(UNIVERSE_FILE)
except Exception as e:
    st.error(
        f"❌ Cannot load universe file **{UNIVERSE_FILE}**.\n\n"
        f"Make sure that exact file is in the repo root.\n\nError:\n{e}"
    )
    st.stop()

try:
    weights_df = load_weights(WEIGHTS_FILE)
except Exception as e:
    st.error(
        f"❌ Cannot load weights file **{WEIGHTS_FILE}**.\n\n"
        f"Make sure that exact file is in the repo root.\n\nError:\n{e}"
    )
    st.stop()

# --------------------------------------------------------------------
# Sidebar controls
# --------------------------------------------------------------------
st.sidebar.header("Data source")
st.sidebar.success("Using Master_Stock_Sheet + WaveWeight mappings from repo CSVs.")

available_waves = sorted(weights_df["Wave"].unique())
selected_wave = st.sidebar.selectbox("Select Wave", available_waves)

mode = st.sidebar.radio(
    "Mode",
    ["Standard", "Alpha-Minus-Beta", "Private Logic™"],
    index=0,
)

st.sidebar.caption("Modes are demo flags only in this build (no risk overlay yet).")

# --------------------------------------------------------------------
# Build holdings for the selected Wave
# --------------------------------------------------------------------
wave_holdings = weights_df[weights_df["Wave"] == selected_wave].copy()

if wave_holdings.empty:
    st.warning(f"No holdings defined for **{selected_wave}** in {WEIGHTS_FILE}.")
    st.stop()

# Join in sector info from the universe if available
join_cols = ["Ticker"]
if "Sector" in universe_df.columns:
    join_cols.append("Sector")

merged = wave_holdings.merge(
    universe_df[join_cols],
    on="Ticker",
    how="left",
)

# Sort by Wave weight
merged = merged.sort_values("Weight", ascending=False)

# --------------------------------------------------------------------
# Layout: top 10 table + bar chart
# --------------------------------------------------------------------
st.subheader(f"{selected_wave} – {mode} mode (LIVE demo)")

col1, col2 = st.columns([1.2, 1])

with col1:
    st.markdown("#### Top 10 holdings (by Wave weight)")
    display_cols = ["Ticker", "Weight"]
    if "Sector" in merged.columns:
        display_cols.append("Sector")
    st.dataframe(
        merged[display_cols].head(10),
        use_container_width=True,
    )

with col2:
    st.markdown("#### Top-10 by Wave weight")
    top10 = merged.head(10).set_index("Ticker")
    st.bar_chart(top10["Weight"])

# --------------------------------------------------------------------
# Simple analytics strip
# --------------------------------------------------------------------
total_holdings = len(wave_holdings)
largest_position = wave_holdings["Weight"].max()
weight_sum = wave_holdings["Weight"].sum()

st.markdown("### Wave analytics")

m1, m2, m3 = st.columns(3)
m1.metric("Total holdings", total_holdings)
m2.metric("Largest position (raw weight)", f"{largest_position:.2f}")
m3.metric("Sum of Wave weights", f"{weight_sum:.2f}")

st.caption(
    "Demo build: All 9 US equity Waves are driven off a single Master Stock Sheet "
    "universe file and a WaveWeight mapping file."
)