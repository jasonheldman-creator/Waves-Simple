import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

# ------------------------------------------------------------
# SAFE CSV READER (bulletproof for messy Google Sheets exports)
# ------------------------------------------------------------
def safe_read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(
            path,
            engine="python",       # handles misaligned rows
            on_bad_lines="skip",   # prevents crashes
        )
    except Exception as e:
        st.error(f"Unable to read CSV '{path}': {e}")
        return pd.DataFrame()


# ------------------------------------------------------------
# LOAD MASTER STOCK SHEET
# ------------------------------------------------------------
def load_universe(path: Path) -> pd.DataFrame:
    df = safe_read_csv(path)

    if df.empty:
        st.error("Master_Stock_Sheet.csv could not be loaded.")
        return df

    # Normalize columns
    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(r"[^a-z0-9]+", "_", regex=True)
    )

    # Rename columns safely
    rename_map = {}
    for col in df.columns:
        if col in ["ticker", "symbol"]:
            rename_map[col] = "Ticker"
        elif col in ["company", "name", "security"]:
            rename_map[col] = "Company"
        elif col in ["weight", "index_weight", "wgt"]:
            rename_map[col] = "Weight"
        elif col in ["sector", "gics_sector"]:
            rename_map[col] = "Sector"
        elif col in ["market_value", "marketvalue", "mv"]:
            rename_map[col] = "MarketValue"
        elif col in ["price", "last_price"]:
            rename_map[col] = "Price"

    df = df.rename(columns=rename_map)

    # Ensure required fields exist
    for col in ["Ticker", "Company", "Weight"]:
        if col not in df.columns:
            df[col] = np.nan

    # Optional fields
    for col in ["Sector", "MarketValue", "Price"]:
        if col not in df.columns:
            df[col] = np.nan

    # Clean types
    df["Ticker"] = df["Ticker"].astype(str).str.upper().str.strip()
    df["Weight"] = pd.to_numeric(df["Weight"], errors="coerce")

    # Drop rows missing ticker or weight
    df = df.dropna(subset=["Ticker", "Weight"])

    # Final order
    keep = ["Ticker", "Company", "Sector", "Weight", "MarketValue", "Price"]
    df = df[[c for c in keep if c in df.columns]]

    return df


# ------------------------------------------------------------
# LOAD WAVE WEIGHTS
# ------------------------------------------------------------
def load_wave_weights(path: Path) -> pd.DataFrame:
    df = safe_read_csv(path)

    if df.empty:
        st.error("wave_weights.csv could not be loaded.")
        return df

    df.columns = (
        df.columns.astype(str)
        .str.lower()
        .str.strip()
        .str.replace(r"[^a-z0-9]+", "_", regex=True)
    )

    rename_map = {}
    for col in df.columns:
        if col in ["ticker", "symbol"]:
            rename_map[col] = "Ticker"
        elif col == "wave":
            rename_map[col] = "Wave"
        elif col in ["weight", "wgt", "alloc"]:
            rename_map[col] = "Weight"

    df = df.rename(columns=rename_map)

    # Ensure required columns
    for col in ["Ticker", "Wave", "Weight"]:
        if col not in df.columns:
            df[col] = np.nan

    df["Ticker"] = df["Ticker"].astype(str).str.upper().str.strip()
    df["Wave"] = df["Wave"].astype(str).str.strip()
    df["Weight"] = pd.to_numeric(df["Weight"], errors="coerce")

    df = df.dropna(subset=["Ticker", "Wave", "Weight"])

    # Normalize weights within wave
    df["Weight"] = df.groupby("Wave")["Weight"].transform(
        lambda x: x / x.sum() if x.sum() else x
    )

    return df


# ------------------------------------------------------------
# STREAMLIT APP
# ------------------------------------------------------------
st.set_page_config(page_title="WAVES Intelligence – S&P Wave Console", layout="wide")
st.title("WAVES Intelligence – S&P Wave Console")

# Load CSVs
universe = load_universe(Path("Master_Stock_Sheet.csv"))
weights = load_wave_weights(Path("wave_weights.csv"))

if universe.empty or weights.empty:
    st.stop()

# Wave selection
waves = sorted(weights["Wave"].unique())
selected_wave = st.sidebar.selectbox("Choose Wave", waves)

wave_df = weights[weights["Wave"] == selected_wave]
merged = wave_df.merge(universe, on="Ticker", how="left")

st.subheader(f"Wave: {selected_wave}")

col1, col2, col3 = st.columns(3)
col1.metric("Holdings", len(merged))
col2.metric("Total Weight", f"{merged['Weight'].sum():.4f}")
col3.metric("Missing Universe Rows", merged['Company'].isna().sum())

st.write("### Holdings")
st.dataframe(
    merged[["Ticker", "Company", "Sector", "Weight", "Price", "MarketValue"]],
    use_container_width=True,
)

st.write("### Top 20 Holdings")
top20 = (
    merged.sort_values("Weight", ascending=False)
    .head(20)
    .set_index("Ticker")["Weight"]
)
st.bar_chart(top20)