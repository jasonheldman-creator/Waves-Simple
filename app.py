import os
from pathlib import Path

import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------

# These MUST match the filenames shown in your GitHub repo exactly
UNIVERSE_FILE = "Master_Stock_Sheet.csv - Sheet5.csv"
WEIGHTS_FILE = "WaveWeight-Sheet1.csv - Sheet1.csv"

TARGET_WAVE_NAME = "S&P 500 Wave"

# ---------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------


def list_working_dir() -> list[str]:
    """Return a sorted list of files so we can show helpful errors."""
    return sorted([p.name for p in Path(".").iterdir()])


def load_csv_safely(path: str, required_cols: list[str], label: str) -> pd.DataFrame:
    """
    Load a CSV and make sure it has the required columns.
    If anything is wrong, show a clear error in the app and stop.
    """
    if not Path(path).exists():
        st.error(
            f"❌ {label} file `{path}` not found.\n\n"
            f"Files I can see here:\n`{list_working_dir()}`"
        )
        st.stop()

    try:
        df = pd.read_csv(path)
    except Exception as e:
        st.error(f"❌ Could not read {label} file `{path}`:\n\n{e}")
        st.stop()

    # Handle possible 'Wave Name' vs 'Wave'
    if "Wave" in required_cols and "Wave" not in df.columns and "Wave Name" in df.columns:
        df = df.rename(columns={"Wave Name": "Wave"})

    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        st.error(
            f"❌ {label} file `{path}` is missing columns {missing}.\n\n"
            f"Columns found: `{list(df.columns)}`"
        )
        st.stop()

    return df


def build_sp500_wave(universe: pd.DataFrame, weights: pd.DataFrame) -> pd.DataFrame:
    """
    Build the S&P 500 Wave holdings:
    - filter weights to the S&P 500 Wave
    - de-duplicate tickers by summing weights
    - merge in company/sector info from the universe
    """

    # Filter to the target Wave
    wave_mask = weights["Wave"].str.contains(TARGET_WAVE_NAME, case=False, na=False)
    sp_w = weights.loc[wave_mask].copy()

    if sp_w.empty:
        st.error(
            f"❌ No rows found in weights file for Wave name containing "
            f"`{TARGET_WAVE_NAME}`.\n\n"
            f"Unique Wave names in file: `{sorted(weights['Wave'].dropna().unique())}`"
        )
        st.stop()

    # De-duplicate tickers: some may appear multiple times, we combine them
    sp_w = (
        sp_w.groupby("Ticker", as_index=False)
        .agg({"Weight": "sum"})
        .sort_values("Weight", ascending=False)
    )

    # Make sure universe has unique tickers before merge
    if "Ticker" not in universe.columns:
        st.error(
            "❌ Universe file does not contain a `Ticker` column.\n\n"
            f"Columns found: `{list(universe.columns)}`"
        )
        st.stop()

    universe_unique = universe.drop_duplicates(subset=["Ticker"]).copy()

    # Try to detect reasonable company / sector columns
    name_col = None
    for cand in ["Name", "Company", "Security", "Holding", "Description"]:
        if cand in universe_unique.columns:
            name_col = cand
            break

    sector_col = None
    for cand in ["Sector", "GICS_Sector", "Industry"]:
        if cand in universe_unique.columns:
            sector_col = cand
            break

    merged = sp_w.merge(universe_unique, on="Ticker", how="left")

    # Normalise columns for display
    merged = merged.rename(columns={"Weight": "WaveWeight"})
    if name_col:
        merged = merged.rename(columns={name_col: "Company"})
    else:
        merged["Company"] = ""

    if sector_col:
        merged = merged.rename(columns={sector_col: "Sector"})
    else:
        merged["Sector"] = "None"

    # Order columns nicely
    display_cols = ["Ticker", "Company", "Sector", "WaveWeight"]
    extra_cols = [c for c in merged.columns if c not in display_cols]
    merged = merged[display_cols + extra_cols]

    return merged


# ---------------------------------------------------------------------
# STREAMLIT APP
# ---------------------------------------------------------------------


def main():
    st.set_page_config(
        page_title="WAVES INTELLIGENCE – S&P 500 Wave Console",
        layout="wide",
    )

    st.title("WAVES INTELLIGENCE™ – PORTFOLIO WAVE CONSOLE")
    st.subheader("S&P 500 Wave (LIVE Demo)")
    st.caption(
        "Mode: Standard – demo only; in production this Wave would drive overlays, "
        "SmartSafe™, and rebalancing logic."
    )

    # Load data
    universe = load_csv_safely(
        UNIVERSE_FILE,
        required_cols=["Ticker"],
        label="Universe",
    )

    weights = load_csv_safely(
        WEIGHTS_FILE,
        required_cols=["Wave", "Ticker", "Weight"],
        label="Wave weights",
    )

    # Build S&P 500 Wave
    sp500 = build_sp500_wave(universe, weights)

    total_holdings = len(sp500)
    st.markdown(f"**Total holdings:** {total_holdings}")

    # Top-10 table & chart
    top10 = sp500.sort_values("WaveWeight", ascending=False).head(10)

    col_table, col_chart = st.columns([2, 1])

    with col_table:
        st.markdown("### Top-10 holdings (by Wave weight)")
        st.dataframe(
            top10[["Ticker", "Company", "Sector", "WaveWeight"]],
            use_container_width=True,
        )

    with col_chart:
        st.markdown("### Top-10 by Wave weight – chart")
        st.bar_chart(
            top10.set_index("Ticker")["WaveWeight"],
            use_container_width=True,
        )

    # Sector allocation
    st.markdown("### Sector allocation")
    sector_alloc = (
        sp500.groupby("Sector", as_index=False)["WaveWeight"].sum().sort_values(
            "WaveWeight", ascending=False
        )
    )

    st.dataframe(sector_alloc, use_container_width=True)

    st.bar_chart(
        sector_alloc.set_index("Sector")["WaveWeight"],
        use_container_width=True,
    )

    # Footer
    st.caption(
        "Demo only – holdings and weights are static snapshots derived from "
        "your Master_Stock_Sheet and WaveWeight files."
    )


if __name__ == "__main__":
    main()