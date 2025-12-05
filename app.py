import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

UNIVERSE_PATH = Path("Master_Stock_Sheet.csv")
WEIGHTS_PATH = Path("wave_weights.csv")

# ------------------------------
# Robust loaders
# ------------------------------

def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(r"[^a-z0-9]+", "_", regex=True)
    )
    return df


def load_universe(path: Path = UNIVERSE_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = _normalize_columns(df)

    col_map = {}
    for col in df.columns:
        if col in ["ticker", "symbol"]:
            col_map[col] = "Ticker"
        elif col in ["company", "company_name", "security", "name"]:
            col_map[col] = "Company"
        elif col in ["weight", "index_weight", "wgt"]:
            col_map[col] = "Weight"
        elif col in ["sector", "gics_sector"]:
            col_map[col] = "Sector"
        elif col in ["market_value", "marketvalue", "mv"]:
            col_map[col] = "MarketValue"
        elif col in ["price", "last_price"]:
            col_map[col] = "Price"

    df = df.rename(columns=col_map)

    # Required
    required = ["Ticker", "Company", "Weight"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Universe file missing required columns: {missing}. "
            f"Found columns: {list(df.columns)}"
        )

    # Optional
    for opt in ["Sector", "MarketValue", "Price"]:
        if opt not in df.columns:
            df[opt] = np.nan

    # Clean types
    df["Ticker"] = df["Ticker"].astype(str).str.strip().str.upper()
    df["Company"] = df["Company"].astype(str).str.strip()

    for col in ["Weight", "MarketValue", "Price"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop rows with no ticker or weight
    df = df.dropna(subset=["Ticker", "Weight"])
    df = df[df["Ticker"] != ""]

    # Reorder
    df = df[["Ticker", "Company", "Sector", "Weight", "MarketValue", "Price"]]

    return df


def load_wave_weights(path: Path = WEIGHTS_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = _normalize_columns(df)

    col_map = {}
    for col in df.columns:
        if col in ["ticker", "symbol"]:
            col_map[col] = "Ticker"
        elif col in ["wave", "portfolio", "strategy"]:
            col_map[col] = "Wave"
        elif col in ["weight", "wgt", "alloc"]:
            col_map[col] = "Weight"

    df = df.rename(columns=col_map)

    required = ["Ticker", "Wave", "Weight"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"wave_weights file missing required columns: {missing}. "
            f"Found columns: {list(df.columns)}"
        )

    df["Ticker"] = df["Ticker"].astype(str).str.strip().str.upper()
    df["Wave"] = df["Wave"].astype(str).str.strip()

    df["Weight"] = pd.to_numeric(df["Weight"], errors="coerce")
    df = df.dropna(subset=["Ticker", "Wave", "Weight"])

    # If weights look like percentages (>1), convert to decimals
    if df["Weight"].max() > 1.5:
        df["Weight"] = df["Weight"] / 100.0

    # Normalize weights within each wave
    df["Weight"] = df.groupby("Wave")["Weight"].transform(
        lambda x: x / x.sum() if x.sum() != 0 else x
    )

    df = df[["Ticker", "Wave", "Weight"]]
    return df


# Cached accessors for Streamlit
@st.cache_data
def get_universe():
    return load_universe()


@st.cache_data
def get_wave_weights():
    return load_wave_weights()


# ------------------------------
# App layout
# ------------------------------

st.set_page_config(page_title="WAVES – S&P Wave", layout="wide")
st.title("WAVES Intelligence – S&P Wave Console")

# Try loading data and show clear error if something is wrong
try:
    universe_df = get_universe()
    weights_df = get_wave_weights()
except Exception as e:
    st.error(f"Error loading CSV files: {e}")
    st.stop()

# Wave selector
available_waves = sorted(weights_df["Wave"].unique())
default_wave = available_waves[0] if available_waves else None
selected_wave = st.sidebar.selectbox("Select Wave", available_waves, index=0)

# Filter for this wave and merge with universe
wave_holdings = weights_df[weights_df["Wave"] == selected_wave].copy()
merged = wave_holdings.merge(
    universe_df,
    on="Ticker",
    how="left",
    suffixes=("_wave", "_universe"),
)

# Flag missing universe data
missing_universe = merged["Company"].isna().sum()

st.subheader(f"Wave: {selected_wave}")

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Number of Holdings", len(merged))
with col2:
    st.metric("Weight Sum (should be 1.0)", f"{merged['Weight'].sum():.4f}")
with col3:
    st.metric("Tickers Missing from Universe", int(missing_universe))

# Display holdings table
display_cols = ["Ticker", "Company", "Sector", "Weight", "Price", "MarketValue"]
existing_display_cols = [c for c in display_cols if c in merged.columns]

st.markdown("### Holdings")
st.dataframe(
    merged[existing_display_cols].sort_values("Weight", ascending=False),
    use_container_width=True,
)

# Simple weight chart
st.markdown("### Top 20 Holdings by Weight")
top20 = (
    merged.sort_values("Weight", ascending=False)
    .head(20)
    .set_index("Ticker")["Weight"]
)
st.bar_chart(top20)

# Debug / raw file inspect (optional)
with st.expander("Raw data preview (debug)"):
    st.write("Universe file (first 10 rows):")
    st.dataframe(universe_df.head(10))
    st.write("Wave weights file (first 10 rows):")
    st.dataframe(weights_df.head(10))