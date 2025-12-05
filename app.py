import streamlit as st
import pandas as pd

# ---------- CONFIG ----------
UNIVERSE_FILE = "Master_Stock_Sheet.csv"          # S&P universe
WEIGHTS_FILE = "WaveWeight-Sheet1.csv"           # Wave / Ticker / Weight
TARGET_WAVE = "S&P 500 Wave"                     # must match text in the Wave column

st.set_page_config(
    page_title="WAVES INTELLIGENCE™ – PORTFOLIO WAVE CONSOLE",
    layout="wide",
)

# ---------- HELPERS ----------

@st.cache_data
def load_universe(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Normalize column names
    df.columns = [c.strip() for c in df.columns]

    # Expect at least Ticker, Company, Sector
    if "Ticker" not in df.columns:
        raise ValueError(f"Universe file {path} must contain a 'Ticker' column.")

    # Some sheets use 'Name' instead of 'Company'
    if "Company" not in df.columns and "Name" in df.columns:
        df = df.rename(columns={"Name": "Company"})

    # Deduplicate by ticker so merge can’t create multiple rows per symbol
    df = df.drop_duplicates(subset="Ticker", keep="first")

    return df


@st.cache_data
def load_weights(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Normalize column names
    df.columns = [c.strip() for c in df.columns]

    # Accept either 'Wave' or 'Wave Name'
    if "Wave" not in df.columns and "Wave Name" in df.columns:
        df = df.rename(columns={"Wave Name": "Wave"})

    required = {"Wave", "Ticker", "Weight"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Weights file {path} must contain columns {sorted(required)}. "
            f"Found: {list(df.columns)}"
        )

    # Keep only the S&P 500 wave rows
    df = df[df["Wave"] == TARGET_WAVE].copy()

    # Make sure Weight is numeric
    df["Weight"] = pd.to_numeric(df["Weight"], errors="coerce").fillna(0.0)

    # If for some reason the weights file has duplicate tickers for the same wave,
    # collapse them to a single row per ticker.
    df = (
        df.groupby(["Wave", "Ticker"], as_index=False)["Weight"]
        .sum()
    )

    return df


def build_holdings(universe: pd.DataFrame, weights: pd.DataFrame) -> pd.DataFrame:
    # Merge and then *re-dedupe* by Wave+Ticker just to be bullet-proof
    merged = weights.merge(universe, on="Ticker", how="left")

    # If universe had multiple rows per ticker for any reason, this guarantees
    # we only show each ticker once.
    merged = (
        merged
        .sort_values(["Ticker"])  # deterministic
        .drop_duplicates(subset=["Wave", "Ticker"], keep="first")
    )

    # Optional: nice column order
    cols = []
    for c in ["Ticker", "Company", "Sector", "Weight"]:
        if c in merged.columns:
            cols.append(c)
    cols += [c for c in merged.columns if c not in cols]

    return merged[cols]


# ---------- APP BODY ----------

st.markdown("## WAVES INTELLIGENCE™ – PORTFOLIO WAVE CONSOLE")
st.markdown("**S&P 500 Wave (LIVE Demo)**")
st.caption(
    "Mode: Standard – demo only; in production, this Wave would drive overlays, "
    "SmartSafe™, and rebalancing logic."
)

# Load data with error handling
try:
    universe_df = load_universe(UNIVERSE_FILE)
except Exception as e:
    st.error(f"Universe file error for `{UNIVERSE_FILE}`:\n\n{e}")
    st.stop()

try:
    weights_df = load_weights(WEIGHTS_FILE)
except Exception as e:
    st.error(f"Weights file error for `{WEIGHTS_FILE}`:\n\n{e}")
    st.stop()

if weights_df.empty:
    st.warning(f"No rows found in weights file for Wave = '{TARGET_WAVE}'.")
    st.stop()

holdings = build_holdings(universe_df, weights_df)

# ---------- TOP-10 TABLE & CHART ----------

st.markdown("### Top-10 holdings (by Wave weight)")

top10 = holdings.sort_values("Weight", ascending=False).head(10).reset_index(drop=True)

# Show as nice table
st.dataframe(top10, use_container_width=True)

# Simple bar chart of weights
if not top10.empty:
    chart_data = top10[["Ticker", "Weight"]].set_index("Ticker")
    st.bar_chart(chart_data)

# ---------- OPTIONAL: BASIC SECTOR BREAKDOWN ----------

if "Sector" in holdings.columns:
    st.markdown("### Sector allocation")
    sector = (
        holdings
        .groupby("Sector", as_index=False)["Weight"]
        .sum()
        .sort_values("Weight", ascending=False)
    )
    st.dataframe(sector, use_container_width=True)