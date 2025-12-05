import streamlit as st
import pandas as pd
import plotly.express as px

# -------------------------------------------------------------------
# FILE NAMES – must match EXACTLY what you see in the repo
# -------------------------------------------------------------------
UNIVERSE_FILE = "Master_Stock_Sheet.csv - Sheet5.csv"
WEIGHTS_FILE = "WaveWeight-Sheet1.csv - Sheet1.csv"

st.set_page_config(
    page_title="WAVES Intelligence – Portfolio Wave Console",
    layout="wide",
)

# -------------------------------------------------------------------
# DATA LOADERS
# -------------------------------------------------------------------
@st.cache_data
def load_universe() -> pd.DataFrame:
    """Load the master stock universe (Sheet5)."""
    df = pd.read_csv(UNIVERSE_FILE)

    # Basic sanity check
    required_cols = ["Ticker", "Weight"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"Universe file `{UNIVERSE_FILE}` is missing columns: {missing}"
        )

    return df


@st.cache_data
def load_weights() -> pd.DataFrame:
    """
    Load the wave weights file.

    Handles two formats:

    1) Proper CSV:
        Wave,Ticker,Weight
        S&P 500 Wave,NVDA,0.10
        ...

    2) "Single column" CSV (what you have now):
        "Wave,Ticker,Weight"
        "S&P 500 Wave,NVDA,0.10"
        ...

       In this case, everything is in one column and we need to split it.
    """
    df = pd.read_csv(WEIGHTS_FILE)
    expected = ["Wave", "Ticker", "Weight"]

    # Case 1: already a clean CSV
    if all(col in df.columns for col in expected):
        weights = df[expected].copy()
        weights["Weight"] = pd.to_numeric(weights["Weight"], errors="coerce")
        weights = weights.dropna(subset=["Ticker", "Weight"])
        return weights

    # Case 2: single column like "Wave,Ticker,Weight"
    if len(df.columns) == 1:
        # Re-read with no header so we can see every row
        raw = pd.read_csv(WEIGHTS_FILE, header=None, names=["raw"])
        raw = raw.dropna()

        # Strip quotes and whitespace
        raw["raw"] = raw["raw"].astype(str).str.strip().str.replace('"', "")

        # Drop the header row "Wave,Ticker,Weight"
        raw = raw[raw["raw"] != "Wave,Ticker,Weight"]

        # Split into three parts
        parts = raw["raw"].str.split(",", expand=True)

        if parts.shape[1] != 3:
            raise ValueError(
                f"Could not split weights into 3 columns. Got {parts.shape[1]} columns."
            )

        parts.columns = expected
        parts["Weight"] = pd.to_numeric(parts["Weight"], errors="coerce")
        parts = parts.dropna(subset=["Ticker", "Weight"])
        return parts

    # Anything else: tell us what went wrong
    raise ValueError(
        f"Wave weights file must contain columns {expected}. "
        f"Found: {list(df.columns)}"
    )


# -------------------------------------------------------------------
# MAIN APP
# -------------------------------------------------------------------
def main():
    st.markdown(
        "## WAVES INTELLIGENCE™ – PORTFOLIO WAVE CONSOLE\n"
        "Equity Waves only – benchmark-aware, AI-directed, multi-mode demo."
    )

    # Try loading data with friendly error messages
    try:
        universe = load_universe()
    except Exception as e:
        st.error(
            f"❌ Cannot load universe file `{UNIVERSE_FILE}`.\n\n"
            f"Error: {e}"
        )
        st.stop()

    try:
        weights = load_weights()
    except Exception as e:
        st.error(
            f"❌ Cannot load weights file `{WEIGHTS_FILE}`.\n\n"
            f"Make sure it has Wave / Ticker / Weight data.\n\n"
            f"Error: {e}"
        )
        st.stop()

    # ----------------------------------------------------------------
    # SIDEBAR – wave selector + mode
    # ----------------------------------------------------------------
    st.sidebar.header("Data source")
    st.sidebar.write(f"Universe file: `{UNIVERSE_FILE}`")
    st.sidebar.write(f"Weights file: `{WEIGHTS_FILE}`")

    wave_names = sorted(weights["Wave"].unique())
    selected_wave = st.sidebar.selectbox("Select Wave", wave_names)

    mode = st.sidebar.radio(
        "Mode",
        ["Standard", "Alpha-Minus-Beta", "Private Logic™"],
        index=0,
    )

    # ----------------------------------------------------------------
    # FILTER FOR SELECTED WAVE
    # ----------------------------------------------------------------
    wave_positions = weights[weights["Wave"] == selected_wave].copy()

    if wave_positions.empty:
        st.warning(f"No holdings found in weights file for **{selected_wave}**.")
        st.stop()

    # Join to universe for company / sector / price info
    wave_positions = wave_positions.merge(
        universe,
        on="Ticker",
        how="left",
        suffixes=("", "_universe"),
    )

    # Normalise weights to 100%
    wave_positions["Weight"] = pd.to_numeric(
        wave_positions["Weight"], errors="coerce"
    )
    wave_positions = wave_positions.dropna(subset=["Weight"])
    total_w = wave_positions["Weight"].sum()
    if total_w == 0:
        st.error(f"All weights are zero for **{selected_wave}**.")
        st.stop()

    wave_positions["NormWeight"] = wave_positions["Weight"] / total_w

    # ----------------------------------------------------------------
    # HEADER
    # ----------------------------------------------------------------
    st.markdown(
        f"### {selected_wave} (LIVE Demo)\n"
        f"Mode: **{mode}** – equities only; in production this flag would drive overlays, "
        f"SmartSafe™, and rebalancing."
    )

    # ----------------------------------------------------------------
    # TOP-10 TABLE
    # ----------------------------------------------------------------
    st.markdown("#### Top 10 holdings – ranked by Wave weight")

    top10 = wave_positions.nlargest(10, "NormWeight").copy()

    # Columns to show if present
    display_cols = []
    for col in ["Ticker", "Company", "Sector", "Price", "NormWeight"]:
        if col in top10.columns:
            display_cols.append(col)

    if "NormWeight" in display_cols:
        top10["Weight %"] = (top10["NormWeight"] * 100).round(2)
        display_cols = [
            col if col != "NormWeight" else "Weight %"
            for col in display_cols
        ]

    # Rename NormWeight column for display
    rename_map = {"NormWeight": "Weight %"}
    top10 = top10.rename(columns=rename_map)

    st.dataframe(top10[display_cols], use_container_width=True)

    # ----------------------------------------------------------------
    # TOP-10 BAR CHART
    # ----------------------------------------------------------------
    if {"Ticker", "NormWeight"} <= set(wave_positions.columns):
        chart_data = top10.sort_values("NormWeight", ascending=False)
        fig = px.bar(
            chart_data,
            x="Ticker",
            y="NormWeight",
            labels={"NormWeight": "Weight"},
        )
        fig.update_yaxes(tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)

    # ----------------------------------------------------------------
    # SUMMARY STATS
    # ----------------------------------------------------------------
    st.markdown("#### Wave stats")
    st.write(f"Positions in this Wave: **{len(wave_positions)}**")
    st.write(f"Unique tickers: **{wave_positions['Ticker'].nunique()}**")


if __name__ == "__main__":
    main()