import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------

st.set_page_config(
    page_title="WAVES Intelligence – Portfolio Wave Console",
    layout="wide",
)

# IMPORTANT: this must match the exact filename in your repo root
UNIVERSE_FILE = "Master_Stock_Sheet.csv - Sheet5.csv"

REQUIRED_UNIVERSE_COLUMNS = ["Ticker", "Company", "Weight"]  # Sector is optional


# ------------------------------------------------------------
# DATA LOADERS
# ------------------------------------------------------------

@st.cache_data(show_spinner=True)
def load_universe(path: str) -> pd.DataFrame:
    """Load the master universe (Sheet5) and normalize weights."""
    df = pd.read_csv(path)

    # Basic column cleanup (strip spaces from names)
    df.columns = [c.strip() for c in df.columns]

    missing = [c for c in REQUIRED_UNIVERSE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Universe CSV is missing required column(s): {missing}. "
            f"Found columns: {list(df.columns)}"
        )

    # Coerce weight numeric
    df["Weight"] = pd.to_numeric(df["Weight"], errors="coerce")
    df = df.dropna(subset=["Weight"])

    # Normalize weights to 1.0
    total_w = df["Weight"].sum()
    if total_w <= 0:
        raise ValueError("Total weight in universe file is zero or negative.")

    df["NormWeight"] = df["Weight"] / total_w

    # Optional fields
    if "Sector" not in df.columns:
        df["Sector"] = "Unknown"

    if "Price" not in df.columns:
        df["Price"] = np.nan

    return df


# ------------------------------------------------------------
# MAIN APP
# ------------------------------------------------------------

def main():
    st.title("WAVES INTELLIGENCE™ – PORTFOLIO WAVE CONSOLE")

    st.caption("Equity Waves only – benchmark-aware, AI-directed, multi-mode demo.")

    # Try to load the universe file
    try:
        universe = load_universe(UNIVERSE_FILE)
    except FileNotFoundError:
        st.error(
            f"Universe file **{UNIVERSE_FILE}** not found in the repo root.\n\n"
            "Make sure that exact CSV is uploaded to the GitHub repository."
        )
        st.stop()
    except ValueError as e:
        st.error(
            f"Unable to use universe file **{UNIVERSE_FILE}**.\n\nError: {e}"
        )
        st.stop()
    except Exception as e:
        st.error(
            f"Unexpected error loading **{UNIVERSE_FILE}**.\n\nError: {e}"
        )
        st.stop()

    # Sidebar controls
    st.sidebar.header("Wave controls")

    # For tomorrow's demo we run a single S&P Wave from this universe
    wave_name = st.sidebar.selectbox(
        "Select Wave",
        options=["S&P 500 Wave"],
        index=0,
    )

    mode = st.sidebar.radio(
        "Mode",
        options=["Standard", "Alpha-Minus-Beta", "Private Logic™"],
        index=0,
    )

    st.markdown(
        f"### {wave_name} (LIVE Demo)\n"
        f"**Mode:** {mode} – equities only; "
        f"in production this mode flag would drive overlays, SmartSafe™, and rebalancing."
    )

    wave_df = universe.copy()

    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------
    total_holdings = len(wave_df)
    top10_weight = wave_df.nlargest(10, "NormWeight")["NormWeight"].sum() * 100
    largest_weight = wave_df["NormWeight"].max() * 100

    c1, c2, c3 = st.columns(3)
    c1.metric("Total holdings", f"{total_holdings:,}")
    c2.metric("Weight in Top-10", f"{top10_weight:.1f}%")
    c3.metric("Largest single position", f"{largest_weight:.2f}%")

    # --------------------------------------------------------
    # TOP-10 HOLDINGS TABLE & CHART
    # --------------------------------------------------------
    st.markdown("#### Top 10 holdings – ranked by Wave weight")

    top10 = wave_df.nlargest(10, "NormWeight").copy()

    # Build a clean table with no duplicate column names
    cols = []
    for c in ["Ticker", "Company", "Sector", "Price"]:
        if c in top10.columns:
            cols.append(c)

    top10["Weight %"] = (top10["NormWeight"] * 100).round(2)
    cols.append("Weight %")

    top10_display = top10[cols].copy()

    tcol, ccol = st.columns([2, 3])

    with tcol:
        st.dataframe(top10_display, use_container_width=True)

    with ccol:
        fig_top = px.bar(
            top10,
            x="Ticker",
            y="NormWeight",
            title="Top-10 by Wave weight",
        )
        fig_top.update_layout(
            xaxis_title="Ticker",
            yaxis_title="Weight",
            showlegend=False,
        )
        fig_top.update_traces(hovertemplate="%{x}: %{y:.4f}")
        st.plotly_chart(fig_top, use_container_width=True)

    # --------------------------------------------------------
    # SECTOR ALLOCATION
    # --------------------------------------------------------
    st.markdown("#### Sector allocation")

    sector_weights = (
        wave_df.groupby("Sector", as_index=False)["NormWeight"].sum()
    )
    sector_weights["Weight %"] = (sector_weights["NormWeight"] * 100).round(2)

    if len(sector_weights) > 1:
        fig_sector = px.bar(
            sector_weights.sort_values("NormWeight", ascending=False),
            x="Sector",
            y="NormWeight",
            title="Sector allocation by Wave weight",
        )
        fig_sector.update_layout(
            xaxis_title="Sector",
            yaxis_title="Weight",
            showlegend=False,
        )
        fig_sector.update_traces(hovertemplate="%{x}: %{y:.4f}")
        st.plotly_chart(fig_sector, use_container_width=True)
    else:
        st.info("Sector information not available in the universe file.")

    # --------------------------------------------------------
    # WEIGHT DECAY CURVE
    # --------------------------------------------------------
    st.markdown("#### Weight decay curve")

    weight_series = (
        wave_df["NormWeight"].sort_values(ascending=False).reset_index(drop=True)
    )
    weight_df = pd.DataFrame(
        {"Rank": np.arange(1, len(weight_series) + 1), "Weight": weight_series}
    )

    fig_decay = px.line(
        weight_df,
        x="Rank",
        y="Weight",
        title="Weight decay (position rank vs. weight)",
    )
    fig_decay.update_layout(
        xaxis_title="Position rank (1 = largest)",
        yaxis_title="Weight",
        showlegend=False,
    )
    fig_decay.update_traces(hovertemplate="Rank %{x}: %{y:.6f}")
    st.plotly_chart(fig_decay, use_container_width=True)


if __name__ == "__main__":
    main()