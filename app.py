import os
import pandas as pd
import streamlit as st

# ---------------------------
# Config
# ---------------------------

APP_TITLE = "WAVES INTELLIGENCE™ – PORTFOLIO WAVE CONSOLE"
TARGET_WAVE_NAME = "S&P 500 Wave"

# Universe filenames we will accept (first one that exists wins)
UNIVERSE_CANDIDATES = [
    "Master_Stock_Sheet.csv",
    "Master_Stock_Sheet - Sheet5.csv",
    "Master_Stock_Sheet.csv - Sheet5.csv",
    "Master_Stock_Sheet5.csv",
]

# Weights filenames we will accept (first one that exists wins)
WEIGHTS_CANDIDATES = [
    "wave_weights.csv",
    "WaveWeight-Sheet1.csv - Sheet1.csv",
    "WaveWeight-Sheet1.csv",
]


# ---------------------------
# Helper functions
# ---------------------------

def find_existing_file(candidates):
    """Return the first existing filename from the list, or None."""
    for name in candidates:
        if os.path.exists(name):
            return name
    return None


def load_universe():
    """Load the master stock universe CSV and normalize column names."""
    universe_file = find_existing_file(UNIVERSE_CANDIDATES)
    if not universe_file:
        raise FileNotFoundError(
            f"No universe file found. Looked for: {UNIVERSE_CANDIDATES}"
        )

    df = pd.read_csv(universe_file)

    # Normalize column names to stripped lower-case with no spaces
    norm_map = {c: c.strip().lower().replace(" ", "") for c in df.columns}

    # Try to locate basic columns
    ticker_col = None
    company_col = None
    sector_col = None

    for orig, norm in norm_map.items():
        if norm in ("ticker", "symbol"):
            ticker_col = orig
        elif norm in ("company", "name", "security"):
            company_col = orig
        elif norm in ("sector",):
            sector_col = orig

    if ticker_col is None:
        raise ValueError(
            f"Universe file {universe_file} is missing a Ticker-like column."
        )

    # Rename the ones we found to standard names
    rename_dict = {}
    if ticker_col:
        rename_dict[ticker_col] = "Ticker"
    if company_col:
        rename_dict[company_col] = "Company"
    if sector_col:
        rename_dict[sector_col] = "Sector"

    df = df.rename(columns=rename_dict)

    # Keep only the useful columns
    keep_cols = ["Ticker"]
    if "Company" in df.columns:
        keep_cols.append("Company")
    if "Sector" in df.columns:
        keep_cols.append("Sector")

    df = df[keep_cols].drop_duplicates(subset=["Ticker"])

    return df, universe_file


def load_weights():
    """Load the wave weights CSV and normalize to ['Wave','Ticker','Weight']."""
    weights_file = find_existing_file(WEIGHTS_CANDIDATES)
    if not weights_file:
        raise FileNotFoundError(
            f"No weights file found. Looked for: {WEIGHTS_CANDIDATES}"
        )

    df = pd.read_csv(weights_file)
    
    # Handle case where CSV has all columns quoted as a single field
    # (e.g., "Wave,Ticker,Weight" as one column)
    if len(df.columns) == 1 and ',' in df.columns[0]:
        # The header and data are in a single quoted column, need to re-parse
        col_name = df.columns[0]
        if ',' in col_name:
            # Split the header to get proper column names
            new_cols = [c.strip() for c in col_name.split(',')]
            # Split each row's data
            split_data = df[col_name].str.split(',', expand=True)
            split_data.columns = new_cols
            df = split_data

    # Normalize column names
    norm_map = {c: c.strip().lower().replace(" ", "") for c in df.columns}

    ticker_col = None
    wave_col = None
    weight_col = None

    for orig, norm in norm_map.items():
        if norm in ("ticker", "symbol"):
            ticker_col = orig
        elif norm in ("wave", "wavename"):
            wave_col = orig
        elif norm in ("weight", "waveweight", "wgt"):
            weight_col = orig

    missing = []
    if ticker_col is None:
        missing.append("Ticker")
    if wave_col is None:
        missing.append("Wave")
    if weight_col is None:
        missing.append("Weight")

    if missing:
        raise ValueError(
            f"Weights file {weights_file} is missing required columns: {missing}. "
            f"Found columns: {list(df.columns)}"
        )

    # Rename to standard names
    df = df.rename(columns={
        ticker_col: "Ticker",
        wave_col: "Wave",
        weight_col: "Weight"
    })

    # Drop fully empty rows
    df = df.dropna(subset=["Ticker", "Wave"])

    # Coerce Weight to numeric, fill missing with 0
    df["Weight"] = pd.to_numeric(df["Weight"], errors="coerce").fillna(0.0)

    return df, weights_file


def build_sp500_view(universe, weights):
    """Join universe + weights for the S&P 500 Wave and aggregate duplicates."""
    w = weights[weights["Wave"] == TARGET_WAVE_NAME].copy()

    if w.empty:
        # Fallback: if there are no explicit weights for S&P 500 Wave,
        # just take the first 10 tickers from the universe equally weighted.
        demo = universe.head(10).copy()
        demo["Weight"] = 1.0 / len(demo)
        return demo.sort_values("Weight", ascending=False)

    # Join with universe to get Company / Sector, but keep all tickers from weights
    merged = w.merge(universe, on="Ticker", how="left")

    # Handle duplicates: group by Ticker and sum weights
    agg = (
        merged
        .groupby("Ticker", as_index=False)
        .agg({
            "Weight": "sum",
            "Company": "first",
            "Sector": "first"
        })
    )

    # Normalize weights to sum to 1 (in case they don't)
    total_w = agg["Weight"].sum()
    if total_w > 0:
        agg["Weight"] = agg["Weight"] / total_w

    # Sort by weight and return
    agg = agg.sort_values("Weight", ascending=False)

    return agg


# ---------------------------
# Streamlit app
# ---------------------------

def main():
    st.set_page_config(page_title=APP_TITLE, layout="wide")

    st.title(APP_TITLE)
    st.subheader(f"{TARGET_WAVE_NAME} (LIVE Demo)")
    st.caption(
        "Mode: Standard – demo only; in production this Wave would drive overlays, "
        "SmartSafe™, and rebalancing logic."
    )

    # Load data with robust error handling
    try:
        universe, uni_file = load_universe()
        weights, w_file = load_weights()
    except Exception as e:
        st.error(f"❌ Data load error:\n\n{e}")
        st.stop()

    st.info(
        f"Using **universe file**: `{uni_file}`  \n"
        f"Using **weights file**: `{w_file}`"
    )

    sp500_view = build_sp500_view(universe, weights)

    total_holdings = len(sp500_view)
    st.markdown(f"### Total holdings: {total_holdings}")

    # Top-10 table + chart
    top10 = sp500_view.head(10).copy()
    top10_display = top10[["Ticker", "Company", "Sector", "Weight"]]

    col1, col2 = st.columns([2, 2])

    with col1:
        st.markdown("### Top-10 holdings (by Wave weight)")
        st.dataframe(top10_display, use_container_width=True)

    with col2:
        st.markdown("### Top-10 by Wave weight – chart")
        st.bar_chart(
            top10.set_index("Ticker")["Weight"]
        )

    # Sector allocation
    if "Sector" in sp500_view.columns:
        st.markdown("### Sector allocation")
        sector = (
            sp500_view
            .fillna({"Sector": "Unknown"})
            .groupby("Sector", as_index=False)["Weight"]
            .sum()
            .sort_values("Weight", ascending=False)
        )
        st.dataframe(sector, use_container_width=True)
        st.bar_chart(
            sector.set_index("Sector")["Weight"]
        )
    else:
        st.caption("Sector data not found in universe file; skipping sector allocation.")


if __name__ == "__main__":
    main()