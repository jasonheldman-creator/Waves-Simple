import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from pathlib import Path

# --------------------------------------------------
# CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="WAVES INTELLIGENCE™ – PORTFOLIO WAVE CONSOLE",
    layout="wide",
)

UNIVERSE_FILE = "Master_Stock_Sheet5.csv"
WAVE_WEIGHTS_FILE = "WaveWeight-Sheet1.csv - Sheet1.csv"

# --------------------------------------------------
# HELPERS
# --------------------------------------------------


@st.cache_data(show_spinner=True)
def load_universe(path: str) -> pd.DataFrame:
    """Load the master US equity universe."""
    if not Path(path).exists():
        st.error(
            f"Universe file **{path}** not found in the repo root.\n\n"
            f"Make sure `Master_Stock_Sheet5.csv` is uploaded to the Waves-Simple folder."
        )
        return pd.DataFrame()

    df = pd.read_csv(path)

    # Normalize column names
    df.columns = df.columns.str.strip()

    # Try to standardize likely column names
    col_map = {}
    for col in df.columns:
        low = col.lower()
        if low.startswith("ticker"):
            col_map[col] = "Ticker"
        elif low.startswith("company") or low.startswith("name"):
            col_map[col] = "Company"
        elif low.startswith("sector"):
            col_map[col] = "Sector"
        elif low.startswith("weight"):
            col_map[col] = "IndexWeight"
        elif low.startswith("price"):
            col_map[col] = "Price"
        elif "market value" in low or low.startswith("market"):
            col_map[col] = "MarketValue"

    df = df.rename(columns=col_map)

    # Keep only useful columns, but don't fail if some are missing
    keep_cols = [c for c in ["Ticker", "Company", "Sector", "IndexWeight", "Price", "MarketValue"] if c in df.columns]
    if keep_cols:
        df = df[keep_cols]

    # Drop rows without tickers
    if "Ticker" in df.columns:
        df = df[df["Ticker"].notna()]
        df["Ticker"] = df["Ticker"].astype(str).str.strip().str.upper()

    return df


@st.cache_data(show_spinner=True)
def load_wave_weights(path: str) -> pd.DataFrame:
    """Load Wave -> Ticker -> Weight mapping."""
    if not Path(path).exists():
        st.error(
            f"Wave weights file **{path}** not found in the repo root.\n\n"
            f"Make sure your Google-Sheets export is uploaded with this exact name."
        )
        return pd.DataFrame(columns=["Wave", "Ticker", "Weight"])

    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()

    # Normalize column names
    col_map = {}
    for col in df.columns:
        low = col.lower()
        if low.startswith("wave"):
            col_map[col] = "Wave"
        elif low.startswith("ticker"):
            col_map[col] = "Ticker"
        elif low.startswith("weight"):
            col_map[col] = "Weight"
    df = df.rename(columns=col_map)

    required = {"Wave", "Ticker", "Weight"}
    if not required.issubset(df.columns):
        st.error(
            f"`{path}` must contain at least these columns: Wave, Ticker, Weight.\n"
            f"Found columns: {list(df.columns)}"
        )
        return pd.DataFrame(columns=["Wave", "Ticker", "Weight"])

    df["Wave"] = df["Wave"].astype(str).str.strip()
    df["Ticker"] = df["Ticker"].astype(str).str.strip().str.upper()
    df["Weight"] = pd.to_numeric(df["Weight"], errors="coerce").fillna(0.0)

    # Drop zero-weight rows
    df = df[df["Weight"] > 0]

    return df


def apply_mode_weights(weights: pd.Series, mode: str) -> pd.Series:
    """
    Adjust weights based on mode and renormalize so they still sum to 1.
    This is display-only (simulating equity exposure per mode).
    """
    if weights.empty:
        return weights

    if mode == "Standard":
        adj = weights.values
    elif mode == "Alpha-Minus-Beta":
        # Slightly lower equity risk (simulate some SmartSafe overlay)
        adj = weights.values * 0.8
    elif mode == "Private Logic":
        # Higher active risk / concentration
        # Raise to a power >1 then renormalize to push into leaders
        adj = np.power(weights.values, 1.2)
    else:
        adj = weights.values

    # Renormalize to sum to 1
    total = adj.sum()
    if total > 0:
        adj = adj / total
    return pd.Series(adj, index=weights.index)


def build_wave_slice(
    universe: pd.DataFrame, weights: pd.DataFrame, wave_name: str, mode: str
) -> pd.DataFrame:
    """Join universe with wave weights for a specific wave and mode."""
    w = weights[weights["Wave"] == wave_name].copy()
    if w.empty:
        return pd.DataFrame()

    # Merge with universe for descriptive fields
    if "Ticker" not in universe.columns:
        return pd.DataFrame()

    merged = pd.merge(
        w,
        universe,
        on="Ticker",
        how="left",
        suffixes=("", "_universe"),
    )

    # Apply mode scaling
    merged["BaseWeight"] = merged["Weight"]
    merged["Weight"] = apply_mode_weights(merged["Weight"], mode)

    # Sort by weight descending
    merged = merged.sort_values("Weight", ascending=False).reset_index(drop=True)

    return merged


def format_pct(x: float) -> str:
    return f"{x * 100:.1f}%"


# --------------------------------------------------
# MAIN UI
# --------------------------------------------------

def main():
    st.markdown(
        "<h2 style='margin-bottom:0;'>WAVES INTELLIGENCE™ – PORTFOLIO WAVE CONSOLE</h2>",
        unsafe_allow_html=True,
    )
    st.caption("Equity Waves only – benchmark-aware, AI-directed, multi-mode demo.")

    # -------- Load data --------
    universe = load_universe(UNIVERSE_FILE)
    weights = load_wave_weights(WAVE_WEIGHTS_FILE)

    if universe.empty or weights.empty:
        st.stop()

    # -------- Sidebar: Wave & Mode selection --------
    sidebar = st.sidebar
    sidebar.header("Data source")

    sidebar.write("Using:")
    sidebar.markdown(f"- **Universe:** `{UNIVERSE_FILE}`")
    sidebar.markdown(f"- **Wave weights:** `{WAVE_WEIGHTS_FILE}`")

    # Wave select
    wave_names = sorted(weights["Wave"].unique())
    selected_wave = sidebar.selectbox("Select Wave", wave_names, index=0)

    # Mode select
    mode = sidebar.radio(
        "Mode",
        options=["Standard", "Alpha-Minus-Beta", "Private Logic"],
        index=0,
    )

    # -------- Build selected Wave view --------
    wave_df = build_wave_slice(universe, weights, selected_wave, mode)

    st.markdown(
        f"### {selected_wave} (LIVE Demo)\n"
        f"Mode: **{mode}** – equities only; in production, this mode flag would drive "
        f"risk overlays, SmartSafe™, and rebalancing.",
    )

    if wave_df.empty:
        st.warning(
            f"No holdings found for **{selected_wave}** in `{WAVE_WEIGHTS_FILE}`.\n\n"
            f"Check that the Wave column values match exactly."
        )
        st.stop()

    # -------- Metrics row --------
    total_holdings = len(wave_df)
    largest_pos = wave_df.iloc[0]
    largest_name = largest_pos.get("Ticker", "")
    largest_weight = largest_pos.get("Weight", 0.0)

    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.metric("Total holdings", f"{total_holdings}")
    col_b.metric("Largest position", f"{largest_name}", format_pct(largest_weight))
    col_c.metric("Equity exposure", "100%", "Mode-adjusted")
    col_d.metric("Cash / SmartSafe", "0%", "Demo only")

    # -------- Top-10 holdings --------
    top10 = wave_df.head(10).copy()
    display_cols = ["Ticker", "Company", "Sector", "Weight"]
    display_cols = [c for c in display_cols if c in top10.columns]

    st.subheader("Top 10 holdings")
    st.caption("Ranked by Wave weight (mode-adjusted).")

    st.dataframe(
        top10[display_cols].assign(WeightPct=top10["Weight"].apply(format_pct)),
        use_container_width=True,
        hide_index=True,
    )

    # -------- Charts row --------
    chart_col, sector_col = st.columns(2)

    with chart_col:
        st.subheader("Top-10 by Wave weight")
        fig = px.bar(
            top10,
            x="Ticker",
            y="Weight",
            labels={"Weight": "Weight"},
        )
        fig.update_layout(
            yaxis_tickformat=".0%",
            xaxis_title="Ticker",
            yaxis_title="Weight (mode-adjusted)",
            margin=dict(l=10, r=10, t=30, b=10),
        )
        st.plotly_chart(fig, use_container_width=True)

    with sector_col:
        st.subheader("Sector allocation")
        if "Sector" in wave_df.columns:
            sector_slice = (
                wave_df.groupby("Sector", dropna=True)["Weight"]
                .sum()
                .sort_values(ascending=False)
                .reset_index()
            )
            if not sector_slice.empty:
                fig2 = px.pie(
                    sector_slice,
                    names="Sector",
                    values="Weight",
                )
                fig2.update_layout(margin=dict(l=10, r=10, t=30, b=10))
                st.plotly_chart(fig2, use_container_width=True)
            else:
                st.info("No sector data available for this Wave.")
        else:
            st.info("Add a **Sector** column to your universe file to see sector allocation.")

    # -------- Console notes --------
    st.markdown("---")
    st.markdown(
        """
**Console notes**

- This demo uses a single master universe (**Master_Stock_Sheet5.csv**) shared by all Waves.  
- Individual Waves and weights are defined in your Google Sheets export  
  (**WaveWeight-Sheet1.csv – Sheet1.csv**).  
- Modes (**Standard / Alpha-Minus-Beta / Private Logic**) rescale and re-concentrate weights for
  demonstration; in production they would also control SmartSafe™ overlays and risk limits.
        """
    )


if __name__ == "__main__":
    main()