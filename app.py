with st.expander("SHOW RAW FILE CONTENTS"):
    try:
        st.code(Path("Master_Stock_Sheet.csv").read_text()[:2000])
    except:
        st.write("Could not read Master_Stock_Sheet.csv")

    try:
        st.code(Path("wave_weights.csv").read_text()[:2000])
    except:
        st.write("Could not read wave_weights.csv")

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

st.set_page_config(page_title="WAVES Intelligence – S&P Wave Console", layout="wide")
st.title("WAVES Intelligence – S&P Wave Console")

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def safe_read_csv(path: Path) -> pd.DataFrame:
    """Very forgiving CSV reader for messy Google Sheets exports."""
    try:
        return pd.read_csv(
            path,
            engine="python",       # more tolerant
            on_bad_lines="skip",   # skip malformed rows
        )
    except Exception as e:
        st.error(f"Unable to read CSV '{path.name}': {e}")
        return pd.DataFrame()


def find_csv(hints) -> Path | None:
    """
    Find the first .csv file in the folder whose name contains ALL hint strings.
    Example: hints=['master','stock'] will match 'Master_Stock_Sheet.csv'.
    """
    hints = [h.lower() for h in hints]
    candidates = []
    for p in Path(".").glob("*.csv"):
        name = p.name.lower()
        if all(h in name for h in hints):
            candidates.append(p)
    if not candidates:
        return None
    return sorted(candidates)[0]


# ------------------------------------------------------------
# Load master universe
# ------------------------------------------------------------

def load_universe() -> pd.DataFrame:
    path = find_csv(["master", "stock"])
    if path is None:
        st.error("Could not find your master stock CSV (expecting something like 'Master_Stock_Sheet.csv').")
        return pd.DataFrame()

    df = safe_read_csv(path)
    if df.empty:
        st.error(f"Master stock file '{path.name}' loaded but has 0 rows.")
        return df

    # Normalize columns
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

    # Ensure required + optional columns exist
    for col in ["Ticker", "Company", "Weight"]:
        if col not in df.columns:
            df[col] = np.nan
    for col in ["Sector", "MarketValue", "Price"]:
        if col not in df.columns:
            df[col] = np.nan

    # Clean types
    df["Ticker"] = df["Ticker"].astype(str).str.upper().str.strip()
    df["Company"] = df["Company"].astype(str).str.strip()
    df["Weight"] = pd.to_numeric(df["Weight"], errors="coerce")
    df["MarketValue"] = pd.to_numeric(df["MarketValue"], errors="coerce")
    df["Price"] = pd.to_numeric(df["Price"], errors="coerce")

    # Drop bad rows
    df = df.dropna(subset=["Ticker", "Weight"])
    df = df[df["Ticker"] != ""]

    # Final column order
    keep = ["Ticker", "Company", "Sector", "Weight", "MarketValue", "Price"]
    df = df[keep]

    return df


# ------------------------------------------------------------
# Load wave weights
# ------------------------------------------------------------

def load_wave_weights() -> pd.DataFrame:
    # Try to find anything with "wave" and "weight" in the name
    path = find_csv(["wave", "weight"])
    if path is None:
        st.error("Could not find your wave weights CSV (expecting something like 'wave_weights.csv').")
        return pd.DataFrame()

    df = safe_read_csv(path)
    if df.empty:
        st.error(f"Wave weights file '{path.name}' loaded but has 0 rows.")
        return df

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

    # Handle missing columns
    if "Ticker" not in df.columns:
        df["Ticker"] = np.nan
    if "Weight" not in df.columns:
        df["Weight"] = np.nan

    # If there is no Wave column at all, assume the whole file is one wave: SP500_Wave
    if "Wave" not in df.columns:
        df["Wave"] = "SP500_Wave"

    df["Ticker"] = df["Ticker"].astype(str).str.upper().str.strip()
    df["Wave"] = df["Wave"].astype(str).str.strip()
    df["Weight"] = pd.to_numeric(df["Weight"], errors="coerce")

    df = df.dropna(subset=["Ticker", "Wave", "Weight"])
    df = df[df["Ticker"] != ""]

    # Normalize weights within each wave (so they sum to 1)
    df["Weight"] = df.groupby("Wave")["Weight"].transform(
        lambda x: x / x.sum() if x.sum() else x
    )

    df = df[["Ticker", "Wave", "Weight"]]
    return df


# ------------------------------------------------------------
# Load data
# ------------------------------------------------------------

universe_df = load_universe()
weights_df = load_wave_weights()

with st.expander("Debug: CSV status"):
    st.write("Universe file rows:", len(universe_df))
    st.write("Wave weights file rows:", len(weights_df))
    st.write("Universe columns:", list(universe_df.columns))
    st.write("Wave weights columns:", list(weights_df.columns))

if universe_df.empty or weights_df.empty:
    st.stop()

# ------------------------------------------------------------
# App UI
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
