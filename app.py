import pathlib
import sys

import pandas as pd
import streamlit as st


# -----------------------------
# Configuration
# -----------------------------

# Name of the master holdings CSV in the repo root
UNIVERSE_FILE = "Master_Stock_Sheet5.csv"  # change if your file name is different

TOP_N = 10  # how many holdings to show in the table / chart


# -----------------------------
# Data loading helpers
# -----------------------------

@st.cache_data
def load_universe(universe_path: str) -> pd.DataFrame:
    """
    Load the master holdings universe and clean it so that:
      - Column names are standardized (Ticker, Name, Sector, Weight)
      - Tickers are upper-cased / stripped
      - Duplicate tickers are removed by aggregating their weights

    The goal is: **one row per ticker** in the final DataFrame.
    """
    df = pd.read_csv(universe_path)

    # Strip whitespace from column names
    df.columns = [c.strip() for c in df.columns]

    # ---- Ticker column ----
    ticker_col = None
    for c in df.columns:
        if c.lower() == "ticker":
            ticker_col = c
            break
    if ticker_col is None:
        raise ValueError("Universe file must contain a 'Ticker' column.")

    if ticker_col != "Ticker":
        df = df.rename(columns={ticker_col: "Ticker"})

    # Standardize tickers
    df["Ticker"] = df["Ticker"].astype(str).str.upper().str.strip()

    # ---- Name column (Company) ----
    if "Name" not in df.columns:
        for candidate in ["Company", "Security", "Issuer"]:
            if candidate in df.columns:
                df = df.rename(columns={candidate: "Name"})
                break

    # If still missing, create a placeholder Name column
    if "Name" not in df.columns:
        df["Name"] = df["Ticker"]

    # ---- Sector column ----
    if "Sector" not in df.columns:
        for candidate in ["GICS Sector", "Industry", "Sector Name"]:
            if candidate in df.columns:
                df = df.rename(columns={candidate: "Sector"})
                break

    if "Sector" not in df.columns:
        df["Sector"] = "Unknown"

    # ---- Weight column ----
    weight_col = None
    for c in df.columns:
        if c.lower().startswith("weight"):
            weight_col = c
            break

    if weight_col is None:
        # If no weight is provided, assign equal weights
        df["Weight"] = 1.0 / len(df)
    else:
        if weight_col != "Weight":
            df = df.rename(columns={weight_col: "Weight"})

    # Make sure Weight is numeric
    df["Weight"] = pd.to_numeric(df["Weight"], errors="coerce").fillna(0.0)

    # -----------------------------
    # 🚫 Fix duplicates: aggregate by ticker
    # -----------------------------
    # If NVDA (for example) appears multiple times, we:
    #   - sum the weights
    #   - keep the first non-null Name / Sector
    grouped = (
        df
        .groupby("Ticker", as_index=False)
        .agg({
            "Name": "first",
            "Sector": "first",
            "Weight": "sum"
        })
    )

    # Sort by descending weight
    grouped = grouped.sort_values("Weight", ascending=False).reset_index(drop=True)

    return grouped


def get_universe() -> pd.DataFrame:
    """
    Locate and load the universe CSV, with friendly error
    messages if something is wrong.
    """
    universe_path = pathlib.Path(UNIVERSE_FILE)

    if not universe_path.exists():
        st.error(
            f"❌ Universe file **{UNIVERSE_FILE}** not found in the app folder.\n\n"
            "Make sure the CSV is in the repo root and the file name here matches "
            "the name on GitHub exactly."
        )
        st.stop()

    try:
        return load_universe(str(universe_path))
    except Exception as e:
        st.error(
            f"❌ Could not load or parse **{UNIVERSE_FILE}**.\n\n"
            f"Error: `{e}`"
        )
        st.stop()


# -----------------------------
# UI helpers
# -----------------------------

def render_top_holdings(universe: pd.DataFrame) -> None:
    """Show top-N holdings table and bar chart."""
    st.markdown("### Top 10 holdings (by Wave weight)")

    top = universe.head(TOP_N).copy()
    top_display = top.rename(
        columns={
            "Name": "Company",
            "Weight": "WaveWeight"
        }
    )

    # Nice, compact table
    st.dataframe(
        top_display[["Ticker", "Company", "Sector", "WaveWeight"]],
        use_container_width=True,
        hide_index=True,
    )

    # Simple bar chart: WaveWeight by ticker
    st.markdown("### Top-10 by Wave weight – chart")
    chart_data = top_display.set_index("Ticker")[["WaveWeight"]]
    st.bar_chart(chart_data)


def render_sector_allocation(universe: pd.DataFrame) -> None:
    """Show sector allocation based on summed weights."""
    st.markdown("### Sector allocation")

    sector_df = (
        universe
        .groupby("Sector", as_index=False)["Weight"]
        .sum()
        .sort_values("Weight", ascending=False)
    )

    sector_df = sector_df.rename(columns={"Weight": "WeightSum"})

    st.dataframe(
        sector_df,
        use_container_width=True,
        hide_index=True,
    )

    # Bar chart for sectors
    if not sector_df.empty:
        chart_data = sector_df.set_index("Sector")[["WeightSum"]]
        st.bar_chart(chart_data)


# -----------------------------
# Main app
# -----------------------------

def main() -> None:
    st.set_page_config(
        page_title="WAVES INTELLIGENCE – S&P 500 Wave Console",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    st.markdown(
        """
        # WAVES INTELLIGENCE™ – PORTFOLIO WAVE CONSOLE  

        **S&P 500 Wave (LIVE Demo)**  
        Mode: Standard – demo only; in production, this Wave would drive overlays, SmartSafe™, and rebalancing logic.
        """,
        unsafe_allow_html=True,
    )

    # Load & clean the universe
    universe = get_universe()

    # Summary
    st.markdown(f"**Total holdings:** {len(universe)}")

    # Layout: top holdings + sector view stacked
    render_top_holdings(universe)
    st.markdown("---")
    render_sector_allocation(universe)


if __name__ == "__main__":
    # Streamlit runs this file as a script; guard just in case
    main()