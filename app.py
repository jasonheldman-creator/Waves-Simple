import os
import textwrap
import pandas as pd
import streamlit as st

# -----------------------------------------------------------------------------
# CONFIG
# -----------------------------------------------------------------------------
TARGET_WAVE_NAME = "S&P 500 Wave"   # the Wave we want to display

st.set_page_config(
    page_title="WAVES INTELLIGENCE™ – Portfolio Wave Console",
    layout="wide",
)

# -----------------------------------------------------------------------------
# HELPERS TO FIND FILES
# -----------------------------------------------------------------------------
def find_csv(prefixes, default_name):
    """
    Look for a CSV file in the repo root.

    1) If default_name exists, use it.
    2) Otherwise, pick the first file that starts with any of `prefixes`
       and ends with `.csv`.
    """
    files = os.listdir(".")
    # 1) exact match
    if default_name in files:
        return default_name, files

    # 2) fuzzy match
    for f in files:
        lower = f.lower()
        for p in prefixes:
            if lower.startswith(p.lower()) and lower.endswith(".csv"):
                return f, files

    # 3) nothing found
    return None, files


@st.cache_data(show_spinner=False)
def load_universe():
    """
    Load the S&P 500 universe from a Master_Stock_Sheet*.csv file.
    Expected columns (case-insensitive):
        Ticker, Company, Weight, Sector
    """
    universe_file, files = find_csv(
        prefixes=["master_stock_sheet"],
        default_name="Master_Stock_Sheet.csv",
    )

    if universe_file is None:
        raise FileNotFoundError(
            "Could not find a universe CSV. "
            "Expected something like 'Master_Stock_Sheet.csv'. "
            f"Files in working directory: {files}"
        )

    df = pd.read_csv(universe_file)

    # normalize column names
    df.columns = [c.strip() for c in df.columns]

    # make sure required columns exist
    required = {"Ticker", "Company", "Weight", "Sector"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Universe file '{universe_file}' is missing columns {missing}. "
            f"Found columns: {list(df.columns)}"
        )

    # Drop duplicate tickers: keep the first occurrence
    df = df.drop_duplicates(subset="Ticker", keep="first")

    # rename Weight -> IndexWeight so we can also have WaveWeight later
    df = df.rename(columns={"Weight": "IndexWeight"})

    return universe_file, df


@st.cache_data(show_spinner=False)
def load_wave_weights():
    """
    Load wave weights from a wave_weights*.csv or WaveWeight*.csv file.
    Expected columns (case-insensitive):
        Wave, Ticker, Weight
    """
    weights_file, files = find_csv(
        prefixes=["wave_weights", "waveweight"],
        default_name="wave_weights.csv",
    )

    if weights_file is None:
        raise FileNotFoundError(
            "Could not find a weights CSV. "
            "Expected something like 'wave_weights.csv'. "
            f"Files in working directory: {files}"
        )

    df = pd.read_csv(weights_file)

    # normalize columns
    df.columns = [c.strip() for c in df.columns]

    required = {"Wave", "Ticker", "Weight"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Weights file '{weights_file}' must contain columns {required}. "
            f"Found: {list(df.columns)}"
        )

    # clean up values
    df["Wave"] = df["Wave"].astype(str).str.strip()
    df["Ticker"] = df["Ticker"].astype(str).str.strip().str.upper()

    # coerce weight to float
    df["Weight"] = pd.to_numeric(df["Weight"], errors="coerce")
    df = df.dropna(subset=["Weight"])

    return weights_file, df


def build_sp500_wave_view(universe_df, weights_df):
    """
    Join S&P 500 Wave weights with the index universe.
    Returns a DataFrame with columns:
        Ticker, Company, Sector, IndexWeight, WaveWeight
    """

    # filter just the S&P 500 Wave (case-insensitive)
    mask = weights_df["Wave"].str.strip().str.lower() == TARGET_WAVE_NAME.lower()
    sp_weights = weights_df.loc[mask].copy()

    if sp_weights.empty:
        raise ValueError(
            f"No rows found in weights file for Wave = '{TARGET_WAVE_NAME}'. "
            "Check the exact spelling in the 'Wave' column."
        )

    # merge with universe on Ticker
    merged = pd.merge(
        sp_weights,
        universe_df,
        on="Ticker",
        how="left",
        suffixes=("_wave", "_idx"),
    )

    # rename for clarity
    if "Weight_wave" in merged.columns:
        merged = merged.rename(columns={"Weight_wave": "WaveWeight"})
    else:
        # if pandas didn't create the suffix for some reason
        merged = merged.rename(columns={"Weight": "WaveWeight"})

    # just in case: ensure IndexWeight present
    if "IndexWeight" not in merged.columns and "Weight_idx" in merged.columns:
        merged = merged.rename(columns={"Weight_idx": "IndexWeight"})

    # if any companies are missing from the universe, mark them
    merged["Company"] = merged["Company"].fillna("(not found in universe)")
    merged["Sector"] = merged["Sector"].fillna("Unknown")

    # sort by WaveWeight
    merged = merged.sort_values("WaveWeight", ascending=False)

    # select & order columns for display
    cols = ["Ticker", "Company", "Sector"]
    if "IndexWeight" in merged.columns:
        cols.append("IndexWeight")
    cols.append("WaveWeight")

    return merged[cols]


# -----------------------------------------------------------------------------
# UI
# -----------------------------------------------------------------------------
st.markdown("## WAVES INTELLIGENCE™ – PORTFOLIO WAVE CONSOLE")
st.markdown(
    textwrap.dedent(
        f"""
        **{TARGET_WAVE_NAME} (LIVE Demo)**  
        Mode: Standard – demo only; in production, this Wave would drive overlays,
        SmartSafe™, and rebalancing logic.
        """
    )
)

# Load data with nice error messages
try:
    universe_file, universe_df = load_universe()
except Exception as e:
    st.error(f"Universe file error:\n\n{e}")
    st.stop()

try:
    weights_file, weights_df = load_wave_weights()
except Exception as e:
    st.error(f"Wave weights file error:\n\n{e}")
    st.stop()

# Build S&P 500 Wave view
try:
    wave_view = build_sp500_wave_view(universe_df, weights_df)
except Exception as e:
    st.error(f"Wave construction error:\n\n{e}")
    st.stop()

total_holdings = len(wave_view)
st.markdown(f"**Total holdings in {TARGET_WAVE_NAME}: {total_holdings}**")

# -----------------------------------------------------------------------------
# TOP-10 TABLE + CHART
# -----------------------------------------------------------------------------
st.markdown("### Top-10 holdings (by Wave weight)")

top10 = wave_view.head(10).reset_index(drop=True)

# display table
st.dataframe(top10, use_container_width=True)

# simple bar chart of top-10
chart_data = top10.set_index("Ticker")["WaveWeight"]
st.bar_chart(chart_data)

# -----------------------------------------------------------------------------
# SECTOR ALLOCATION
# -----------------------------------------------------------------------------
st.markdown("### Sector allocation (by Wave weight)")

sector_alloc = (
    wave_view.groupby("Sector", dropna=False)["WaveWeight"]
    .sum()
    .reset_index()
    .sort_values("WaveWeight", ascending=False)
)

st.dataframe(sector_alloc, use_container_width=True)

st.bar_chart(
    sector_alloc.set_index("Sector")["WaveWeight"]
)

# -----------------------------------------------------------------------------
# DEBUG / DIAGNOSTICS PANEL
# -----------------------------------------------------------------------------
with st.expander("Diagnostics (for Jason only)"):
    st.write("Universe file used:", universe_file)
    st.write("Weights file used:", weights_file)

    st.write("Universe sample:")
    st.dataframe(universe_df.head(10))

    st.write("Weights sample:")
    st.dataframe(weights_df.head(10))