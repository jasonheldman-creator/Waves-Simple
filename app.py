# app.py
import os
import pandas as pd
import streamlit as st

# ---------- CONFIG ----------
UNIVERSE_FILE = "Master_Stock_Sheet - Sheet5.csv"   # <-- EXACT FILE NAME

st.set_page_config(
    page_title="WAVES INTELLIGENCE – PORTFOLIO WAVE CONSOLE",
    layout="wide",
)

# ---------- HELPERS ----------

@st.cache_data
def load_universe(path: str) -> pd.DataFrame:
    """Load the S&P 500 universe from CSV and do basic cleaning."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Universe file not found: {path}")

    df = pd.read_csv(path)

    # Normalize column names
    df.columns = [c.strip() for c in df.columns]

    # We need at least a Ticker column
    ticker_col = None
    for cand in ["Ticker", "ticker", "Symbol", "symbol"]:
        if cand in df.columns:
            ticker_col = cand
            break

    if ticker_col is None:
        raise ValueError(
            f"Could not find a Ticker column in {path}. "
            f"Found columns: {list(df.columns)}"
        )

    # Optional columns
    name_col = None
    for cand in ["Name", "Company", "Security Name", "name"]:
        if cand in df.columns:
            name_col = cand
            break

    sector_col = None
    for cand in ["Sector", "GICS Sector", "sector"]:
        if cand in df.columns:
            sector_col = cand
            break

    # Make a working dataframe
    work = df.copy()

    # Clean tickers
    work[ticker_col] = (
        work[ticker_col]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    # Drop blank tickers
    work = work[work[ticker_col] != ""].copy()

    # Determine base weight column, if any
    weight_col = None
    for cand in ["Weight", "weight", "IndexWeight", "Index Weight", "PortfolioWeight"]:
        if cand in work.columns:
            weight_col = cand
            break

    if weight_col is None:
        # Equal-weight if there is no explicit weight column
        work["__base_weight"] = 1.0
        weight_col = "__base_weight"
    else:
        # Coerce to numeric, fall back to equal weight if all fail
        work[weight_col] = pd.to_numeric(work[weight_col], errors="coerce")
        if work[weight_col].notna().sum() == 0:
            work[weight_col] = 1.0

    # Group by ticker to remove duplicates (sum weights)
    agg_dict = {weight_col: "sum"}
    if name_col is not None:
        agg_dict[name_col] = "first"
    if sector_col is not None:
        agg_dict[sector_col] = "first"

    grouped = (
        work.groupby(ticker_col, as_index=False)
        .agg(agg_dict)
        .rename(columns={ticker_col: "Ticker"})
    )

    # Final columns
    if name_col is not None and name_col in grouped.columns:
        grouped = grouped.rename(columns={name_col: "Company"})
    else:
        grouped["Company"] = ""

    if sector_col is not None and sector_col in grouped.columns:
        grouped = grouped.rename(columns={sector_col: "Sector"})
    else:
        grouped["Sector"] = "None"

    # Normalize weights to 1.0
    total_weight = grouped[weight_col].sum()
    if total_weight == 0 or pd.isna(total_weight):
        grouped["WaveWeight"] = 1.0 / len(grouped)
    else:
        grouped["WaveWeight"] = grouped[weight_col] / total_weight

    # Sort by weight descending
    grouped = grouped.sort_values("WaveWeight", ascending=False).reset_index(drop=True)

    return grouped[["Ticker", "Company", "Sector", "WaveWeight"]]


def render_wave_view(holdings: pd.DataFrame) -> None:
    """Render the S&P 500 Wave view in Streamlit."""
    st.title("WAVES INTELLIGENCE™ – PORTFOLIO WAVE CONSOLE")

    st.markdown(
        """
        **S&P 500 Wave (LIVE Demo)**  
        Mode: Standard – demo only; in production, this Wave would drive overlays,
        SmartSafe™, and rebalancing logic.
        """
    )

    st.markdown(f"**Total holdings:** {len(holdings)}")

    col_left, col_right = st.columns([3, 2])

    # ---- Top 10 table ----
    top10 = holdings.head(10).copy()
    top10["WaveWeight"] = (top10["WaveWeight"] * 100).round(2)

    with col_left:
        st.subheader("Top-10 holdings (by Wave weight)")
        st.dataframe(
            top10.rename(columns={"WaveWeight": "WaveWeight (%)"}),
            use_container_width=True,
        )

    # ---- Top 10 chart ----
    with col_right:
        st.subheader("Top-10 by Wave weight – chart")
        chart_data = top10.set_index("Ticker")["WaveWeight"]
        st.bar_chart(chart_data)

    # ---- Sector allocation ----
    st.subheader("Sector allocation")
    sector_alloc = (
        holdings.groupby("Sector", as_index=False)["WaveWeight"].sum()
        .sort_values("WaveWeight", ascending=False)
        .reset_index(drop=True)
    )
    sector_alloc["WaveWeight"] = (sector_alloc["WaveWeight"] * 100).round(2)
    st.dataframe(
        sector_alloc.rename(columns={"WaveWeight": "Weight (%)"}),
        use_container_width=True,
    )


# ---------- MAIN APP ----------

def main():
    try:
        holdings = load_universe(UNIVERSE_FILE)
    except FileNotFoundError as e:
        st.error(
            f"Universe file not found.\n\n"
            f"Expected file: `{UNIVERSE_FILE}` in the repo root.\n\n{e}"
        )
        # Show what files DO exist to help debugging
        st.write("Files in working directory:")
        st.write(os.listdir("."))
        return
    except Exception as e:
        st.error(
            f"Unable to load universe file `{UNIVERSE_FILE}`.\n\n"
            f"Error: {e}"
        )
        return

    render_wave_view(holdings)


if __name__ == "__main__":
    main()