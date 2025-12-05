import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------
st.set_page_config(
    page_title="WAVES Intelligence – Portfolio Wave Console",
    layout="wide",
)

# These MUST match the filenames in your GitHub repo root exactly
UNIVERSE_FILE = "Master_Stock_Sheet.csv - Sheet5.csv"
WEIGHTS_FILE = "WaveWeight-Sheet1.csv - Sheet1.csv"

# ---------------------------------------------------------------------
# DATA LOADERS
# ---------------------------------------------------------------------
@st.cache_data
def load_universe(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Try to normalize column names a bit
    cols = {c.strip(): c.strip() for c in df.columns}
    df.rename(columns=cols, inplace=True)
    # Make sure key columns exist
    required = ["Ticker"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Universe missing columns: {missing}")
    return df


@st.cache_data
def load_wave_weights(path: str) -> pd.DataFrame:
    # Some exports quote the whole line – let pandas figure it out
    df = pd.read_csv(path)
    # Try to handle both "Wave" and "Wave Name" header variants
    rename_map = {}
    for col in df.columns:
        col_stripped = str(col).strip()
        if col_stripped.lower() in ("wave", "wave name"):
            rename_map[col] = "Wave"
        elif col_stripped.lower() == "ticker":
            rename_map[col] = "Ticker"
        elif col_stripped.lower() == "weight":
            rename_map[col] = "Weight"
    df.rename(columns=rename_map, inplace=True)

    required = ["Wave", "Ticker", "Weight"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Wave weights file must contain columns {required}. "
            f"Found: {list(df.columns)}"
        )

    # Clean up values
    df["Wave"] = df["Wave"].astype(str).str.strip().str.replace('"', "")
    df["Ticker"] = df["Ticker"].astype(str).str.strip().str.replace('"', "")
    # Convert weight like "0.10" or "0,10" to float
    df["Weight"] = (
        df["Weight"]
        .astype(str)
        .str.replace('"', "")
        .str.replace(",", ".", regex=False)
        .astype(float)
    )
    return df


def get_wave_holdings(universe: pd.DataFrame,
                      weights: pd.DataFrame,
                      wave_name: str) -> pd.DataFrame:
    w = weights[weights["Wave"] == wave_name].copy()
    if w.empty:
        return pd.DataFrame()

    # Normalise to 100% inside the Wave
    w["Weight"] = w["Weight"].astype(float)
    total = w["Weight"].sum()
    if total == 0:
        w["NormWeight"] = 0.0
    else:
        w["NormWeight"] = w["Weight"] / total

    # Merge with universe to pull in Sector / Company / etc if present
    u = universe.copy()
    # Try to standardise a couple of common columns
    # (from your master sheet screenshots)
    if "Company" not in u.columns and "Name" in u.columns:
        u.rename(columns={"Name": "Company"}, inplace=True)

    merged = pd.merge(
        w,
        u,
        on="Ticker",
        how="left",
        suffixes=("", "_universe"),
    )
    return merged


# ---------------------------------------------------------------------
# MAIN APP
# ---------------------------------------------------------------------
def main():
    st.markdown(
        "<h1 style='margin-bottom:0.25rem;'>WAVES INTELLIGENCE™ – PORTFOLIO WAVE CONSOLE</h1>",
        unsafe_allow_html=True,
    )
    st.caption("Equity Waves only – benchmark-aware, AI-directed, multi-mode demo.")

    # ----------------- Load data with friendly errors -----------------
    try:
        universe = load_universe(UNIVERSE_FILE)
    except Exception as e:
        st.error(
            f"❌ Universe file not found or invalid.\n\n"
            f"Expected file: **{UNIVERSE_FILE}** in the repo root.\n\n"
            f"Error: `{e}`"
        )
        st.stop()

    try:
        wave_weights = load_wave_weights(WEIGHTS_FILE)
    except Exception as e:
        st.error(
            f"❌ Cannot load weights file **{WEIGHTS_FILE}**.\n\n"
            f"Make sure it’s a clean CSV with columns `Wave, Ticker, Weight`.\n\n"
            f"Error: `{e}`"
        )
        st.stop()

    # If we got here, both files loaded
    wave_names = sorted(wave_weights["Wave"].unique())

    # ----------------------- Sidebar controls -------------------------
    with st.sidebar:
        st.subheader("Data source")

        st.write("Universe file:")
        st.code(UNIVERSE_FILE, language="text")

        st.write("Wave weights file:")
        st.code(WEIGHTS_FILE, language="text")

        st.markdown("---")

        selected_wave = st.selectbox("Select Wave", wave_names)

        st.subheader("Mode")
        mode = st.radio(
            "Mode",
            options=["Standard", "Alpha-Minus-Beta", "Private Logic™"],
            index=0,
        )

    # ------------------ Compute current Wave view ---------------------
    holdings = get_wave_holdings(universe, wave_weights, selected_wave)

    if holdings.empty:
        st.warning(
            f"No holdings found for Wave **{selected_wave}** "
            f"in weights file **{WEIGHTS_FILE}**."
        )
        st.stop()

    # Convenience columns
    if "Sector" not in holdings.columns:
        holdings["Sector"] = "Unknown"

    # Sort by weight
    holdings_sorted = holdings.sort_values("NormWeight", ascending=False)

    # Summary stats
    total_holdings = len(holdings_sorted)
    largest_w = holdings_sorted["NormWeight"].max()
    largest_ticker = holdings_sorted.iloc[0]["Ticker"]

    # Just placeholders for now – we’ll wire real alpha/cash later
    equity_vs_cash = "100% / 0%"
    alpha_capture = "n/a"

    # ------------------------- Headline block -------------------------
    st.markdown(
        f"### {selected_wave} (LIVE Demo)  \n"
        f"*Mode: **{mode}*** – equities only."
    )

    # --------------------- Top-10 & chart row -------------------------
    left, right = st.columns([1.1, 1.4])

    with left:
        st.markdown("#### Top 10 holdings")
        top10 = holdings_sorted.head(10).copy()
        top10_display = top10[["Ticker", "Company", "Sector", "NormWeight"]].copy()
        top10_display["Weight %"] = (top10_display["NormWeight"] * 100).round(2)
        top10_display = top10_display.drop(columns=["NormWeight"])
        st.dataframe(
            top10_display,
            use_container_width=True,
            hide_index=True,
        )

    with right:
        st.markdown("#### Top-10 by Wave weight")
        fig_top10 = px.bar(
            top10,
            x="Ticker",
            y="NormWeight",
            labels={"NormWeight": "Weight"},
        )
        fig_top10.update_layout(
            margin=dict(l=0, r=0, t=10, b=0),
            yaxis_tickformat=".1%",
        )
        st.plotly_chart(fig_top10, use_container_width=True)

        # Quick numeric stats under the chart
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Total holdings", f"{total_holdings}")
        with c2:
            st.metric("Largest position", f"{largest_ticker} – {largest_w:0.2%}")
        with c3:
            st.metric("Equity vs Cash", equity_vs_cash)

    # ---------------------- Sector allocation row ---------------------
    st.markdown("---")
    col1, col2 = st.columns([1.3, 1])

    with col1:
        st.markdown("#### Sector allocation")
        sector = (
            holdings_sorted.groupby("Sector")["NormWeight"]
            .sum()
            .reset_index()
            .sort_values("NormWeight", ascending=False)
        )
        fig_sector = px.bar(
            sector,
            x="Sector",
            y="NormWeight",
            labels={"NormWeight": "Weight"},
        )
        fig_sector.update_layout(
            margin=dict(l=0, r=0, t=10, b=0),
            yaxis_tickformat=".1%",
        )
        st.plotly_chart(fig_sector, use_container_width=True)

    with col2:
        st.markdown("#### Mode overview")
        if mode == "Standard":
            st.write(
                "- Target: benchmark-aware, fully invested.\n"
                "- Uses raw Wave weights from the master stock universe.\n"
                "- In production, this mode would control risk overlays and rebalancing."
            )
        elif mode == "Alpha-Minus-Beta":
            st.write(
                "- Target: reduced beta with downside protection.\n"
                "- In production, this mode would dial back equity exposure and "
                "increase SmartSafe / cash.\n"
                "- Top-line view is the same here; risk engine would live behind it."
            )
        else:  # Private Logic™
            st.write(
                "- Proprietary leadership / momentum logic.\n"
                "- In production, this mode would rotate into leaders and manage "
                "position sizes more aggressively.\n"
                "- Console here focuses on holdings; trade engine would run off-screen."
            )

    st.markdown("---")
    st.caption(
        "Demo only – no orders are routed from this console.  "
        "All analytics are calculated from the uploaded universe and Wave weights CSVs."
    )


if __name__ == "__main__":
    main()