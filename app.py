import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

# ------------------------------------------------------------
# PAGE SETUP
# ------------------------------------------------------------
st.set_page_config(
    page_title="WAVES Intelligence – S&P Wave Console",
    layout="wide"
)
st.title("WAVES Intelligence – S&P Wave Console")

# ------------------------------------------------------------
# SAFE CSV READER
# ------------------------------------------------------------
def safe_read_csv(path: Path) -> pd.DataFrame:
    """Forgiving CSV reader that won't crash on bad lines."""
    try:
        return pd.read_csv(path, engine="python", on_bad_lines="skip")
    except Exception as e:
        st.error(f"Error reading {path.name}: {e}")
        return pd.DataFrame()


# ------------------------------------------------------------
# LOAD MASTER UNIVERSE (Master_Stock_Sheet.csv)
# ------------------------------------------------------------
def load_universe(path: Path = Path("Master_Stock_Sheet.csv")) -> pd.DataFrame:
    df = safe_read_csv(path)
    if df.empty:
        return df

    # Normalize column names
    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(r"[^a-z0-9]+", "_", regex=True)
    )

    # Map to standard names
    rename_map = {}
    for col in df.columns:
        if col in ["ticker", "symbol"]:
            rename_map[col] = "Ticker"
        elif col in ["company", "company_name", "name", "security"]:
            rename_map[col] = "Company"
        elif col in ["sector", "gics_sector"]:
            rename_map[col] = "Sector"
        elif col in ["weight", "index_weight", "wgt"]:
            rename_map[col] = "Weight_universe"
        elif col in ["market_value", "marketvalue", "mv"]:
            rename_map[col] = "MarketValue"
        elif col in ["price", "last_price"]:
            rename_map[col] = "Price"

    df = df.rename(columns=rename_map)

    # Ensure required columns exist
    if "Ticker" not in df.columns:
        df["Ticker"] = np.nan

    # Optional but nice to have
    if "Company" not in df.columns:
        df["Company"] = df["Ticker"]
    if "Sector" not in df.columns:
        df["Sector"] = ""
    if "Weight_universe" not in df.columns:
        df["Weight_universe"] = np.nan
    if "MarketValue" not in df.columns:
        df["MarketValue"] = np.nan
    if "Price" not in df.columns:
        df["Price"] = np.nan

    # Clean types
    df["Ticker"] = df["Ticker"].astype(str).str.upper().str.strip()
    df["Company"] = df["Company"].astype(str).str.strip()
    df["Sector"] = df["Sector"].astype(str).str.strip()
    df["Weight_universe"] = pd.to_numeric(df["Weight_universe"], errors="coerce")
    df["MarketValue"] = pd.to_numeric(df["MarketValue"], errors="coerce")
    df["Price"] = pd.to_numeric(df["Price"], errors="coerce")

    # Drop empty tickers
    df = df[df["Ticker"] != ""]
    df = df.dropna(subset=["Ticker"])

    # Final column order
    df = df[["Ticker", "Company", "Sector", "Weight_universe", "MarketValue", "Price"]]
    return df


# ------------------------------------------------------------
# LOAD WAVE WEIGHTS (wave_weights.csv)
# ------------------------------------------------------------
def load_wave_weights(path: Path = Path("wave_weights.csv")) -> pd.DataFrame:
    df = safe_read_csv(path)
    if df.empty:
        return df

    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(r"[^a-z0-9]+", "_", regex=True)
    )

    rename_map = {}
    for col in df.columns:
        if col in ["ticker", "symbol"]:
            rename_map[col] = "Ticker"
        elif col in ["wave", "portfolio", "strategy"]:
            rename_map[col] = "Wave"
        elif col in ["weight", "wgt", "alloc"]:
            rename_map[col] = "Weight_wave"

    df = df.rename(columns=rename_map)

    # Ensure required columns
    if "Ticker" not in df.columns:
        df["Ticker"] = np.nan
    if "Wave" not in df.columns:
        df["Wave"] = "SP500_Wave"
    if "Weight_wave" not in df.columns:
        df["Weight_wave"] = 1.0

    df["Ticker"] = df["Ticker"].astype(str).str.upper().str.strip()
    df["Wave"] = df["Wave"].astype(str).str.strip()
    df["Weight_wave"] = pd.to_numeric(df["Weight_wave"], errors="coerce")

    df = df[(df["Ticker"] != "") & df["Ticker"].notna()]
    df = df.dropna(subset=["Weight_wave"])

    # Normalize weights within each wave
    df["Weight_wave"] = df.groupby("Wave")["Weight_wave"].transform(
        lambda x: x / x.sum() if x.sum() else x
    )

    df = df[["Ticker", "Wave", "Weight_wave"]]
    return df


# ------------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------------
universe_df = load_universe()
weights_df = load_wave_weights()

with st.expander("Debug: CSV status"):
    st.write("Universe file rows:", len(universe_df))
    st.write("Wave weights file rows:", len(weights_df))
    st.write("Universe columns:", list(universe_df.columns))
    st.write("Wave weights columns:", list(weights_df.columns))
    st.write("Universe preview:", universe_df.head())
    st.write("Weights preview:", weights_df.head())

if universe_df.empty or weights_df.empty:
    st.error("One or both CSV files have no usable data rows.")
    st.stop()

# ------------------------------------------------------------
# SIDEBAR – WAVE / MODE / CASH CONTROLS
# ------------------------------------------------------------
waves = sorted(weights_df["Wave"].unique())
selected_wave = st.sidebar.selectbox("Choose Wave", waves, index=0)

risk_mode = st.sidebar.selectbox(
    "Risk Mode",
    ["Standard", "Alpha-Minus-Beta", "Private Logic"]
)

cash_buffer_pct = st.sidebar.slider(
    "Cash Buffer (%)",
    min_value=0,
    max_value=50,
    value=5,
    step=1,
    help="Demo control: simulates how much of the Wave is held in SmartSafe / cash."
)

# ------------------------------------------------------------
# BUILD CURRENT WAVE VIEW
# ------------------------------------------------------------
wave_slice = weights_df[weights_df["Wave"] == selected_wave].copy()
merged = wave_slice.merge(universe_df, on="Ticker", how="left")

# If sectors missing, label as Unknown for charts
merged["Sector"] = merged["Sector"].fillna("").replace("", "Unknown")

# Effective equity/cash exposures
total_wave_weight = float(merged["Weight_wave"].sum()) if not merged.empty else 0.0
cash_exposure = cash_buffer_pct / 100.0
equity_exposure = max(0.0, 1.0 - cash_exposure) * total_wave_weight

# ------------------------------------------------------------
# TOP SUMMARY STRIP
# ------------------------------------------------------------
st.subheader(f"Wave: {selected_wave}")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Holdings", len(merged))
c2.metric("Total Wave Weight", f"{total_wave_weight:.4f}")
c3.metric("Equity Exposure", f"{equity_exposure * 100:,.1f}%")
c4.metric("Cash Buffer", f"{cash_buffer_pct:.1f}%")

# Secondary summary – concentration & top names
if not merged.empty:
    sorted_by_weight = merged.sort_values("Weight_wave", ascending=False)
    top1 = sorted_by_weight.iloc[0]
    top5_weight = sorted_by_weight["Weight_wave"].head(5).sum()
    top10_weight = sorted_by_weight["Weight_wave"].head(10).sum()

    c5, c6, c7 = st.columns(3)
    c5.metric("Top Holding", f"{top1['Ticker']} – {top1['Company']}")
    c6.metric("Top 5 Concentration", f"{top5_weight * 100:,.1f}%")
    c7.metric("Top 10 Concentration", f"{top10_weight * 100:,.1f}%")

# ------------------------------------------------------------
# HOLDINGS TABLE
# ------------------------------------------------------------
st.markdown("### Holdings")

display_df = merged.copy()
display_df["Weight"] = display_df["Weight_wave"]

display_cols = ["Ticker", "Company", "Sector", "Weight", "MarketValue", "Price"]
display_cols = [c for c in display_cols if c in display_df.columns]

st.dataframe(
    display_df[display_cols].sort_values("Weight", ascending=False),
    use_container_width=True,
)

# ------------------------------------------------------------
# CHARTS – TOP 10 & SECTOR ALLOCATION
# ------------------------------------------------------------
chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    st.markdown("#### Top 10 Holdings by Weight")
    if not display_df.empty:
        top10 = (
            display_df.sort_values("Weight", ascending=False)
            .head(10)
            .set_index("Ticker")["Weight"]
        )
        st.bar_chart(top10)

with chart_col2:
    st.markdown("#### Sector Allocation (Wave Weights)")
    if "Sector" in display_df.columns and not display_df.empty:
        sector_weights = (
            display_df.groupby("Sector")["Weight"]
            .sum()
            .sort_values(ascending=False)
        )
        st.bar_chart(sector_weights)

# ------------------------------------------------------------
# RISK MODE EXPLANATION (for the demo narrative)
# ------------------------------------------------------------
st.markdown("### Mode Explanation (for Franklin demo)")

if risk_mode == "Standard":
    st.write(
        "In **Standard** mode, the Wave targets full beta to its benchmark with "
        "disciplined rebalancing and tax-efficient execution. The cash buffer is "
        "minimal and primarily operational."
    )
elif risk_mode == "Alpha-Minus-Beta":
    st.write(
        "In **Alpha-Minus-Beta** mode, the Wave dials down market beta (using the cash "
        "buffer and defensive tilts) while preserving as much stock-selection alpha "
        "as possible. This is the capital-preservation profile."
    )
else:
    st.write(
        "In **Private Logic** mode, the Wave applies more aggressive adaptive logic "
        "within guardrails, allowing higher turnover and more dynamic tilts, while "
        "keeping full transparency on every position and trade."
    )