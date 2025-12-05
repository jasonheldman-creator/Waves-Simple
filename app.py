import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import yfinance as yf

# ------------------------------------------------------------
# PAGE SETUP
# ------------------------------------------------------------
st.set_page_config(
    page_title="WAVES Intelligence – S&P Wave Console",
    layout="wide"
)
st.title("WAVES Intelligence – S&P Wave Console")

# Simple manual refresh so you can hit it during the demo
if st.button("🔄 Refresh live data"):
    st.experimental_rerun()

st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

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
# LIVE PRICE FETCH
# ------------------------------------------------------------
def fetch_live_prices(tickers: list[str]) -> dict:
    """Fetch live last prices from Yahoo. Falls back silently if anything fails."""
    prices = {}
    for t in tickers:
        try:
            hist = yf.Ticker(t).history(period="1d", interval="1m")
            if not hist.empty:
                prices[t] = float(hist["Close"].iloc[-1])
        except Exception:
            # If anything fails for a ticker, we just skip it
            continue
    return prices


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
    help="Simulated cash held in SmartSafe / money market."
)

# ------------------------------------------------------------
# BUILD CURRENT WAVE VIEW
# ------------------------------------------------------------
wave_slice = weights_df[weights_df["Wave"] == selected_wave].copy()
merged = wave_slice.merge(universe_df, on="Ticker", how="left")

# If sectors missing, label as Unknown for charts
merged["Sector"] = merged["Sector"].fillna("").replace("", "Unknown")

# ------------------------------------------------------------
# LIVE PRICING & “TRADES”
# ------------------------------------------------------------
# Treat universe Price as "reference" price
merged["Price"] = pd.to_numeric(merged["Price"], errors="coerce")

# Fetch live prices
live_price_map = fetch_live_prices(merged["Ticker"].unique().tolist())
merged["LivePrice"] = merged["Ticker"].map(live_price_map)
# Fallback to static price if live not available
merged["LivePrice"] = merged["LivePrice"].fillna(merged["Price"])

# Assume a notional portfolio value for demo (e.g., $1,000,000)
PORTFOLIO_NOTIONAL = 1_000_000.0

# Shares based on target weights & reference price
valid_price_mask = merged["Price"] > 0
merged["Shares"] = 0.0
merged.loc[valid_price_mask, "Shares"] = (
    merged.loc[valid_price_mask, "Weight_wave"] * PORTFOLIO_NOTIONAL
) / merged.loc[valid_price_mask, "Price"]

# Current market value based on LIVE prices
merged["LiveValue"] = merged["Shares"] * merged["LivePrice"]

total_live_value = merged["LiveValue"].sum()
if total_live_value > 0:
    merged["LiveWeight"] = merged["LiveValue"] / total_live_value
else:
    merged["LiveWeight"] = merged["Weight_wave"]

# Drift vs target
merged["Drift"] = merged["LiveWeight"] - merged["Weight_wave"]

# Simple trade suggestion: if drift > +0.5% → sell, < -0.5% → buy
THRESH = 0.005
merged["TradeAction"] = ""
merged.loc[merged["Drift"] > THRESH, "TradeAction"] = "Sell"
merged.loc[merged["Drift"] < -THRESH, "TradeAction"] = "Buy"

merged["TradeSize_$"] = merged["Drift"] * PORTFOLIO_NOTIONAL
merged["TradeShares"] = merged["TradeSize_$"] / merged["LivePrice"].replace(0, np.nan)

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
# HOLDINGS TABLE (WITH LIVE PRICES)
# ------------------------------------------------------------
st.markdown("### Holdings (Live)")

display_df = merged.copy()
display_df["TargetWeight"] = display_df["Weight_wave"]
display_df["CurrentWeight"] = display_df["LiveWeight"]

display_cols = [
    "Ticker",
    "Company",
    "Sector",
    "TargetWeight",
    "CurrentWeight",
    "Drift",
    "Price",
    "LivePrice",
    "Shares",
    "LiveValue",
]
display_cols = [c for c in display_cols if c in display_df.columns]

st.dataframe(
    display_df[display_cols].sort_values("CurrentWeight", ascending=False),
    use_container_width=True,
)

# ------------------------------------------------------------
# TRADE SUGGESTIONS
# ------------------------------------------------------------
st.markdown("### Trade Suggestions (Engine Output – Demo Only)")

trades = merged[merged["TradeAction"] != ""].copy()
if trades.empty:
    st.write("No trades required – Wave is within drift thresholds.")
else:
    trade_cols = [
        "Ticker",
        "Company",
        "TradeAction",
        "Drift",
        "TradeShares",
        "TradeSize_$",
        "LivePrice",
    ]
    trade_cols = [c for c in trade_cols if c in trades.columns]
    st.dataframe(
        trades[trade_cols].sort_values("TradeSize_$", key=lambda s: s.abs(), ascending=False),
        use_container_width=True,
    )

# ------------------------------------------------------------
# CHARTS – TOP 10 & SECTOR ALLOCATION
# ------------------------------------------------------------
chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    st.markdown("#### Top 10 Holdings by Current Weight")
    if not display_df.empty:
        top10 = (
            display_df.sort_values("CurrentWeight", ascending=False)
            .head(10)
            .set_index("Ticker")["CurrentWeight"]
        )
        st.bar_chart(top10)

with chart_col2:
    st.markdown("#### Sector Allocation (Current Weights)")
    if "Sector" in display_df.columns and not display_df.empty:
        sector_weights = (
            display_df.groupby("Sector")["CurrentWeight"]
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