import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

# ------------------------------------------------------------
# BASIC PAGE SETUP
# ------------------------------------------------------------
st.set_page_config(page_title="WAVES Intelligence – S&P Wave Console", layout="wide")
st.title("WAVES Intelligence – S&P Wave Console")


# ------------------------------------------------------------
# HELPER: very forgiving CSV reader
# ------------------------------------------------------------
def safe_read_csv(path: Path) -> pd.DataFrame:
    """
    Try hard to read a CSV even if it has weird delimiters or bad rows.
    """
    if not path.exists():
        st.error(f"CSV file not found: {path.name}")
        return pd.DataFrame()

    # Try automatic delimiter detection first
    for params in [
        {"sep": None, "engine": "python"},  # auto-detect delimiter
        {"sep": ",", "engine": "python"},   # fallback: standard comma CSV
    ]:
        try:
            df = pd.read_csv(
                path,
                on_bad_lines="skip",  # skip malformed lines instead of crashing
                **params,
            )
            return df
        except Exception:
            continue

    st.error(f"Could not read CSV: {path.name}")
    return pd.DataFrame()


# ------------------------------------------------------------
# LOAD MASTER UNIVERSE
# ------------------------------------------------------------
def load_universe() -> pd.DataFrame:
    path = Path("Master_Stock_Sheet.csv")
    df = safe_read_csv(path)
    if df.empty:
        return df

    # Normalize column names
    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(r"[^a-z0-9]+", "_", regex=True)
    )

    # Map columns to standard names
    rename_map = {}
    for col in df.columns:
        if col in ["ticker", "symbol"]:
            rename_map[col] = "Ticker"
        elif col in ["company", "company_name", "security", "name"]:
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

    # Ensure columns exist
    if "Ticker" not in df.columns:
        df["Ticker"] = np.nan
    if "Company" not in df.columns:
        df["Company"] = ""
    if "Weight" not in df.columns:
        df["Weight"] = np.nan
    if "Sector" not in df.columns:
        df["Sector"] = ""
    if "MarketValue" not in df.columns:
        df["MarketValue"] = np.nan
    if "Price" not in df.columns:
        df["Price"] = np.nan

    # Clean values
    df["Ticker"] = df["Ticker"].astype(str).str.upper().str.strip()
    df["Company"] = df["Company"].astype(str).str.strip()
    df["Sector"] = df["Sector"].astype(str).str.strip()

    df["Weight"] = pd.to_numeric(df["Weight"], errors="coerce")
    df["MarketValue"] = pd.to_numeric(df["MarketValue"], errors="coerce")
    df["Price"] = pd.to_numeric(df["Price"], errors="coerce")

    # Keep rows with a ticker (even if weight is missing)
    df = df[df["Ticker"] != ""]

    # Final column order
    df = df[["Ticker", "Company", "Sector", "Weight", "MarketValue", "Price"]]
    return df


# ------------------------------------------------------------
# LOAD WAVE WEIGHTS
# ------------------------------------------------------------
def load_wave_weights() -> pd.DataFrame:
    path = Path("wave_weights.csv")
    df = safe_read_csv(path)
    if df.empty:
        return df

    # Normalize column names
    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(r"[^a-z0-9]+", "_", regex=True)
    )

    rename_map = {}
    for col in df.columns:
        if col in ["ticker", "symbol"]:
            rename_map[col] = "Ticker"
        elif col in ["wave", "portfolio", "strategy"]:
            rename_map[col] = "Wave"
        elif col in ["weight", "wgt", "alloc"]:
            rename_map[col] = "Weight"

    df = df.rename(columns=rename_map)

    # Ensure required columns
    if "Ticker" not in df.columns:
        df["Ticker"] = np.nan
    if "Weight" not in df.columns:
        df["Weight"] = np.nan
    if "Wave" not in df.columns:
        # If no wave column, assume everything is the S&P wave
        df["Wave"] = "SP500_Wave"

    df["Ticker"] = df["Ticker"].astype(str).str.upper().str.strip()
    df["Wave"] = df["Wave"].astype(str).str.strip()
    df["Weight"] = pd.to_numeric(df["Weight"], errors="coerce")

    # Keep rows with a ticker + weight
    df = df.dropna(subset=["Ticker", "Weight"])
    df = df[df["Ticker"] != ""]

    # Normalize weights within each wave so they sum to 1.0
    df["Weight"] = df.groupby("Wave")["Weight"].transform(
        lambda x: x / x.sum() if x.sum() else x
    )

    df = df[["Ticker", "Wave", "Weight"]]
    return df


# ------------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------------
universe_df = load_universe()
weights_df = load_wave_weights()

with st.expander("Debug: CSV status"):
    st.write("Universe file rows:", len(universe_df))
    st.write("Wave weights file rows:", len(weights_df))
    st.write("Universe columns:", list(universe_df.columns))
    st.write("Wave weights columns:", list(weights_df.columns))
    st.write("Universe preview:", universe_df.head(5))
    st.write("Weights preview:", weights_df.head(5))

# If either is empty, stop so we don't crash
if universe_df.empty or weights_df.empty:
    st.warning("One or both CSV files have no usable data rows. Check the debug box above.")
    st.stop()


# ------------------------------------------------------------
# APP UI
# ------------------------------------------------------------
waves = sorted(weights_df["Wave"].unique())
selected_wave = st.sidebar.selectbox("Choose Wave", waves)

wave_slice = weights_df[weights_df["Wave"] == selected_wave]
merged = wave_slice.merge(universe_df, on="Ticker", how="left")

st.subheader(f"Wave: {selected_wave}")

c1, c2, c3 = st.columns(3)
c1.metric("Holdings", len(merged))
c2.metric("Total Weight", f"{merged['Weight'].sum():.4f}")
c3.metric("Missing company info", int(merged["Company"].isna().sum()))

st.markdown("### Holdings")
cols = ["Ticker", "Company", "Sector", "Weight", "Price", "MarketValue"]
cols = [c for c in cols if c in merged.columns]
st.dataframe(
    merged[cols].sort_values("Weight", ascending=False),
    use_container_width=True,
)

st.markdown("### Top 20 by Weight")
top20 = (
    merged.sort_values("Weight", ascending=False)
    .head(20)
    .set_index("Ticker")["Weight"]
)
st.bar_chart(top20)