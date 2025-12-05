import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import yfinance as yf

# ------------------------------------------------------------
# PAGE SETUP & BRANDING
# ------------------------------------------------------------
st.set_page_config(
    page_title="WAVES Intelligence – Wave Engine Console",
    layout="wide"
)

BRAND_BG = "#07072C"       # deep navy
BRAND_CARD = "#111133"     # card background
BRAND_ACCENT = "#30F2A0"   # neon green/teal
BRAND_TEXT = "#F5F7FF"     # light text

st.markdown(
    f"""
    <style>
    /* Overall page */
    .stApp {{
        background: radial-gradient(circle at top left, #101030 0, {BRAND_BG} 45%, #02020a 100%);
        color: {BRAND_TEXT};
        font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text", system-ui, sans-serif;
    }}

    /* Sidebar */
    section[data-testid="stSidebar"] > div {{
        background: linear-gradient(180deg, #050522 0, #02020f 100%);
        border-right: 1px solid #1a1a40;
    }}

    /* Sidebar labels */
    section[data-testid="stSidebar"] label {{
        color: #E0E4FF !important;
        font-weight: 600 !important;
        letter-spacing: 0.02em;
        font-size: 0.8rem;
        text-transform: uppercase;
    }}

    /* Metric cards */
    .stMetric {{
        background-color: {BRAND_CARD};
        padding: 0.6rem 0.8rem;
        border-radius: 0.6rem;
        border: 1px solid #262654;
    }}
    .stMetric > div > div:nth-child(1) {{
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #B6B9FF;
    }}
    .stMetric > div > div:nth-child(2) {{
        font-size: 1.3rem;
        font-weight: 600;
        color: {BRAND_ACCENT};
    }}

    /* Dataframes */
    .stDataFrame, .stDataFrame table {{
        color: {BRAND_TEXT} !important;
        background-color: {BRAND_CARD} !important;
        border-radius: 0.6rem;
    }}
    .stDataFrame [data-testid="stTable"] th {{
        background-color: #181842 !important;
        color: #C0C4FF !important;
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }}

    /* Expander header */
    .streamlit-expanderHeader {{
        background-color: #11112E !important;
        color: #C7CBFF !important;
    }}

    /* Primary buttons */
    button[kind="primary"] {{
        background: linear-gradient(90deg, {BRAND_ACCENT}, #4DF2D2);
        color: #02020a !important;
        border-radius: 999px;
        border: none;
        font-weight: 600;
        padding: 0.4rem 1.1rem;
    }}
    button[kind="primary"]:hover {{
        filter: brightness(1.08);
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# Header / title
st.markdown(
    """
    <div style="display:flex;align-items:baseline;gap:0.6rem;margin-bottom:0.2rem;">
        <div style="font-size:1.5rem;font-weight:650;color:#FFFFFF;">
            WAVES Intelligence&trade; – Wave Engine Console
        </div>
        <div style="font-size:0.9rem;color:#7F84FF;">
            Live multi-wave portfolio engine • Demo view
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Manual refresh for live prices
cols_top = st.columns([1, 2, 2])
with cols_top[0]:
    if st.button("🔄 Refresh live data"):
        st.experimental_rerun()
with cols_top[1]:
    st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
with cols_top[2]:
    st.caption("Note: Prices via Yahoo Finance – demo only, not for trading.")

# ------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------
def safe_read_csv(path: Path) -> pd.DataFrame:
    """Forgiving CSV reader that won't crash on bad lines."""
    try:
        return pd.read_csv(path, engine="python", on_bad_lines="skip")
    except Exception as e:
        st.error(f"Error reading {path.name}: {e}")
        return pd.DataFrame()


def load_universe(path: Path = Path("Master_Stock_Sheet.csv")) -> pd.DataFrame:
    """Load and normalize the master stock universe."""
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

    # Ensure required columns
    if "Ticker" not in df.columns:
        df["Ticker"] = np.nan
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

    df = df[df["Ticker"] != ""]
    df = df.dropna(subset=["Ticker"])

    df = df[["Ticker", "Company", "Sector", "Weight_universe", "MarketValue", "Price"]]
    return df


def load_wave_weights(path: Path = Path("wave_weights.csv")) -> pd.DataFrame:
    """Load and normalize wave weight definitions."""
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

    # Normalize within each wave so they sum to 1.0
    df["Weight_wave"] = df.groupby("Wave")["Weight_wave"].transform(
        lambda x: x / x.sum() if x.sum() else x
    )

    df = df[["Ticker", "Wave", "Weight_wave"]]
    return df


def fetch_live_prices(tickers) -> dict:
    """Fetch live prices from Yahoo Finance. If it fails for a ticker, we skip it."""
    prices = {}
    for t in tickers:
        try:
            hist = yf.Ticker(t).history(period="1d", interval="1m")
            if not hist.empty:
                prices[t] = float(hist["Close"].iloc[-1])
        except Exception:
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
# SIDEBAR – CONTROLS
# ------------------------------------------------------------
waves = sorted(weights_df["Wave"].unique())
selected_wave = st.sidebar.selectbox("Choose Wave", waves, index=0)

risk_mode = st.sidebar.selectbox(
    "Risk Mode",
    ["Standard", "Alpha-Minus-Beta", "Private Logic"],
)

cash_buffer_pct = st.sidebar.slider(
    "Cash Buffer (%)",
    min_value=0,
    max_value=50,
    value=5,
    step=1,
    help="Simulated cash held in SmartSafe / money market.",
)

st.sidebar.markdown(
    """
    <div style="margin-top:1.5rem;padding-top:0.75rem;
                border-top:1px solid #262654;
                color:#8084FF;font-size:0.7rem;
                text-transform:uppercase;letter-spacing:0.15em;">
        WAVES INTELLIGENCE&trade;
    </div>
    """,
    unsafe_allow_html=True,
)

# ------------------------------------------------------------
# BUILD CURRENT WAVE VIEW
# ------------------------------------------------------------
wave_slice = weights_df[weights_df["Wave"] == selected_wave].copy()
merged = wave_slice.merge(universe_df, on="Ticker", how="left")

merged["Sector"] = merged["Sector"].fillna("").replace("", "Unknown")
merged["Price"] = pd.to_numeric(merged["Price"], errors="coerce")

# Live prices
live_price_map = fetch_live_prices(merged["Ticker"].unique().tolist())
merged["LivePrice"] = merged["Ticker"].map(live_price_map)
merged["LivePrice"] = merged["LivePrice"].fillna(merged["Price"])

PORTFOLIO_NOTIONAL = 1_000_000.0

valid_price_mask = merged["Price"] > 0
merged["Shares"] = 0.0
merged.loc[valid_price_mask, "Shares"] = (
    merged.loc[valid_price_mask, "Weight_wave"] * PORTFOLIO_NOTIONAL
) / merged.loc[valid_price_mask, "Price"]

merged["LiveValue"] = merged["Shares"] * merged["LivePrice"]

total_live_value = merged["LiveValue"].sum()
if total_live_value > 0:
    merged["LiveWeight"] = merged["LiveValue"] / total_live_value
else:
    merged["LiveWeight"] = merged["Weight_wave"]

merged["Drift"] = merged["LiveWeight"] - merged["Weight_wave"]

THRESH = 0.005  # 0.5% drift threshold
merged["TradeAction"] = ""
merged.loc[merged["Drift"] > THRESH, "TradeAction"] = "Sell"
merged.loc[merged["Drift"] < -THRESH, "TradeAction"] = "Buy"

merged["TradeSize_$"] = merged["Drift"] * PORTFOLIO_NOTIONAL
merged["TradeShares"] = merged["TradeSize_$"] / merged["LivePrice"].replace(0, np.nan)

total_wave_weight = float(merged["Weight_wave"].sum()) if not merged.empty else 0.0
cash_exposure = cash_buffer_pct / 100.0
equity_exposure = max(0.0, 1.0 - cash_exposure) * total_wave_weight

# Concentration stats
top5_weight = merged["Weight_wave"].nlargest(5).sum() if not merged.empty else 0.0
top10_weight = merged["Weight_wave"].nlargest(10).sum() if not merged.empty else 0.0
top_holding = (
    merged.sort_values("Weight_wave", ascending=False).iloc[0]
    if not merged.empty
    else None
)

# ------------------------------------------------------------
# TOP SUMMARY STRIP
# ------------------------------------------------------------
st.subheader(f"Wave: {selected_wave}")

m1, m2, m3, m4 = st.columns(4)
m1.metric("Holdings", len(merged))
m2.metric("Total Wave Weight", f"{total_wave_weight:.4f}")
m3.metric("Equity Exposure", f"{equity_exposure * 100:,.1f}%")
m4.metric("Cash Buffer", f"{cash_buffer_pct:.1f}%")

if top_holding is not None:
    m5, m6, m7 = st.columns(3)
    m5.metric("Top Holding", f"{top_holding['Ticker']} – {top_holding['Company']}")
    m6.metric("Top 5 Concentration", f"{top5_weight * 100:,.1f}%")
    m7.metric("Top 10 Concentration", f"{top10_weight * 100:,.1f}%")

# ------------------------------------------------------------
# TABS: OVERVIEW / HOLDINGS / TRADES / ANALYTICS
# ------------------------------------------------------------
tab_overview, tab_holdings, tab_trades, tab_analytics = st.tabs(
    ["📊 Overview", "📋 Holdings", "📝 Trades", "📈 Analytics"]
)

display_df = merged.copy()
display_df["TargetWeight"] = display_df["Weight_wave"]
display_df["CurrentWeight"] = display_df["LiveWeight"]

# ---------------- OVERVIEW TAB ----------------
with tab_overview:
    st.markdown(
        "<h4 style='margin-top:0.5rem;margin-bottom:0.4rem;color:#E3E5FF;'>Wave Snapshot</h4>",
        unsafe_allow_html=True,
    )

    o1, o2 = st.columns(2)

    with o1:
        st.markdown(
            "<div style='font-size:0.8rem;color:#B6B9FF;margin-bottom:0.3rem;'>Top 10 by Target Weight</div>",
            unsafe_allow_html=True,
        )
        if not display_df.empty:
            top10 = (
                display_df.sort_values("TargetWeight", ascending=False)
                .head(10)[["Ticker", "Company", "Sector", "TargetWeight"]]
            )
            st.dataframe(top10, use_container_width=True, height=280)

    with o2:
        st.markdown(
            "<div style='font-size:0.8rem;color:#B6B9FF;margin-bottom:0.3rem;'>Top 10 by Current Weight</div>",
            unsafe_allow_html=True,
        )
        if not display_df.empty:
            top10_live = (
                display_df.sort_values("CurrentWeight", ascending=False)
                .head(10)[["Ticker", "Company", "Sector", "CurrentWeight", "Drift"]]
            )
            st.dataframe(top10_live, use_container_width=True, height=280)

    st.markdown(
        "<h4 style='margin-top:1.2rem;margin-bottom:0.4rem;color:#E3E5FF;'>Mode Explanation</h4>",
        unsafe_allow_html=True,
    )

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

# ---------------- HOLDINGS TAB ----------------
with tab_holdings:
    st.markdown(
        "<h4 style='margin-top:0.5rem;margin-bottom:0.4rem;color:#E3E5FF;'>Holdings (Live)</h4>",
        unsafe_allow_html=True,
    )

    holdings_cols = [
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
    holdings_cols = [c for c in holdings_cols if c in display_df.columns]

    st.dataframe(
        display_df[holdings_cols].sort_values("CurrentWeight", ascending=False),
        use_container_width=True,
    )

# ---------------- TRADES TAB ----------------
with tab_trades:
    st.markdown(
        "<h4 style='margin-top:0.5rem;margin-bottom:0.4rem;color:#E3E5FF;'>Trade Suggestions (Demo Engine Output)</h4>",
        unsafe_allow_html=True,
    )

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
            trades[trade_cols].sort_values(
                "TradeSize_$", key=lambda s: s.abs(), ascending=False
            ),
            use_container_width=True,
        )

# ---------------- ANALYTICS TAB ----------------
with tab_analytics:
    a1, a2 = st.columns(2)

    with a1:
        st.markdown(
            "<h4 style='margin-top:0.5rem;margin-bottom:0.4rem;color:#E3E5FF;'>Top 10 by Current Weight</h4>",
            unsafe_allow_html=True,
        )
        if not display_df.empty:
            top10_chart = (
                display_df.sort_values("CurrentWeight", ascending=False)
                .head(10)
                .set_index("Ticker")["CurrentWeight"]
            )
            st.bar_chart(top10_chart)

    with a2:
        st.markdown(
            "<h4 style='margin-top:0.5rem;margin-bottom:0.4rem;color:#E3E5FF;'>Sector Allocation (Current Weights)</h4>",
            unsafe_allow_html=True,
        )
        if "Sector" in display_df.columns and not display_df.empty:
            sector_weights = (
                display_df.groupby("Sector")["CurrentWeight"]
                .sum()
                .sort_values(ascending=False)
            )
            st.bar_chart(sector_weights)