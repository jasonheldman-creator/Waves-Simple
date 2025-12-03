import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# --------------------------------------------------
# Page config
# --------------------------------------------------
st.set_page_config(
    page_title="Waves Simple Console",
    layout="wide"
)

st.title("🌊 WAVES SIMPLE CONSOLE")

st.markdown(
    """
    Upload a **Wave snapshot CSV** to view:
    - Portfolio preview  
    - Key exposure stats  
    - Top holdings & sector charts  
    - SmartSafe™ mode and human override controls  
    """
)

# --------------------------------------------------
# Helper functions
# --------------------------------------------------
def find_column(df: pd.DataFrame, candidates):
    """Return the first matching column name from a list of candidates."""
    lower_map = {c.lower().strip(): c for c in df.columns}
    for name in candidates:
        key = name.lower().strip()
        if key in lower_map:
            return lower_map[key]
    return None


def prepare_weights(df: pd.DataFrame):
    """
    Try to find a weight column.
    If none exists, assume equal weights.
    Returns a copy of df with a 'Weight_pct' column in percent.
    """
    df = df.copy()

    weight_col = find_column(
        df,
        ["Weight_pct", "Weight %", "Weight", "PortfolioWeight", "Port Weight"]
    )

    if weight_col is not None:
        weights = pd.to_numeric(df[weight_col], errors="coerce")
        # If looks like 0-1, scale to percent
        if weights.max() <= 1.01:
            weights = weights * 100.0
        df["Weight_pct"] = weights.fillna(0.0)
    else:
        # Equal weight fallback
        n = len(df)
        df["Weight_pct"] = 100.0 / n if n > 0 else 0.0

    # Normalize to 100% just in case
    total = df["Weight_pct"].sum()
    if total > 0:
        df["Weight_pct"] = df["Weight_pct"] * (100.0 / total)

    return df


def tag_cash(df: pd.DataFrame):
    """
    Try to identify which rows are cash.
    Returns a Series of booleans.
    """
    ticker_col = find_column(df, ["Ticker", "Symbol"])
    asset_col = find_column(df, ["Asset Class", "AssetClass", "Type", "Category"])

    is_cash = pd.Series(False, index=df.index)

    if ticker_col is not None:
        tickers = df[ticker_col].astype(str).str.upper()
        cash_like = ["CASH", "SWVXX", "MMF", "USD CASH"]
        is_cash = is_cash | tickers.isin(cash_like)

    if asset_col is not None:
        assets = df[asset_col].astype(str).str.upper()
        is_cash = is_cash | assets.str.contains("CASH", na=False)

    return is_cash


def compute_basic_stats(df: pd.DataFrame):
    """Compute basic portfolio metrics."""
    df = prepare_weights(df)
    is_cash = tag_cash(df)

    total_holdings = len(df)
    cash_weight = df.loc[is_cash, "Weight_pct"].sum()
    equity_weight = max(0.0, 100.0 - cash_weight)

    largest_pos = df["Weight_pct"].max() if total_holdings > 0 else 0.0

    return {
        "df": df,
        "is_cash": is_cash,
        "total_holdings": total_holdings,
        "cash_weight": cash_weight,
        "equity_weight": equity_weight,
        "largest_pos": largest_pos,
    }


def apply_exposure_overrides(equity_weight, cash_weight, mode, equity_tilt):
    """
    Apply SmartSafe™ mode and human equity tilt.
    equity_weight, cash_weight are in percent (0–100).
    equity_tilt is in percent (-20 to +20).
    Returns adjusted_equity, adjusted_cash (percent).
    """

    mode_factors = {
        "Neutral": 1.00,
        "Defensive": 0.70,
        "Max Safe": 0.40,
    }
    factor = mode_factors.get(mode, 1.0)

    base_equity = equity_weight * factor / 100.0  # convert to 0–1
    tilt_factor = 1.0 + (equity_tilt / 100.0)

    adj_equity = base_equity * tilt_factor
    adj_equity = min(max(adj_equity, 0.0), 1.0)

    adj_cash = 1.0 - adj_equity

    return adj_equity * 100.0, adj_cash * 100.0


# --------------------------------------------------
# Sidebar – controls
# --------------------------------------------------
st.sidebar.header("⚙️ Controls")

smart_mode = st.sidebar.radio(
    "SmartSafe™ mode",
    ["Neutral", "Defensive", "Max Safe"],
    index=0
)

st.sidebar.markdown("---")

equity_tilt = st.sidebar.slider(
    "Equity tilt (human override, %)",
    min_value=-20,
    max_value=20,
    value=0,
    help="Positive = more equity, Negative = more cash"
)

growth_tilt = st.sidebar.slider(
    "Growth style tilt (bps)",
    min_value=-200,
    max_value=200,
    value=0,
    step=25,
    help="Demo-only: used for display; applied if style data exists"
)

value_tilt = st.sidebar.slider(
    "Value style tilt (bps)",
    min_value=-200,
    max_value=200,
    value=0,
    step=25,
    help="Demo-only: used for display; applied if style data exists"
)

st.sidebar.markdown("---")
st.sidebar.caption("This console is **read-only** – no live orders are placed.")

# --------------------------------------------------
# File upload
# --------------------------------------------------
uploaded_file = st.file_uploader(
    "Upload your CSV",
    type=["csv"],
    help="Use your Google Sheets export for a selected Wave."
)

if uploaded_file is None:
    st.info("⬆️ Upload a CSV file to begin.")
    st.stop()

# --------------------------------------------------
# Load and validate data
# --------------------------------------------------
try:
    raw_df = pd.read_csv(uploaded_file)
except Exception as e:
    st.error(f"Could not read CSV: {e}")
    st.stop()

if raw_df.empty:
    st.warning("The uploaded CSV is empty.")
    st.stop()

stats = compute_basic_stats(raw_df)
df = stats["df"]
is_cash = stats["is_cash"]
total_holdings = stats["total_holdings"]
cash_weight = stats["cash_weight"]
equity_weight = stats["equity_weight"]
largest_pos = stats["largest_pos"]

# --------------------------------------------------
# Layout – top: preview & stats
# --------------------------------------------------
preview_tab, charts_tab, overrides_tab = st.tabs(
    ["📄 Preview & Stats", "📊 Charts", "🎛 Overrides & Targets"]
)

# --- Preview & Stats tab ---
with preview_tab:
    st.subheader("Portfolio preview")

    st.dataframe(
        df,
        use_container_width=True,
        height=320
    )

    st.markdown("### Key stats")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total holdings", f"{total_holdings}")
    c2.metric("Equity weight", f"{equity_weight:0.1f}%")
    c3.metric("Cash weight", f"{cash_weight:0.1f}%")
    c4.metric("Largest position", f"{largest_pos:0.1f}%")

    if cash_weight == 0:
        st.caption("⚠️ Cash could not be detected – assuming **100% equity** for now.")

# --- Charts tab ---
with charts_tab:
    st.subheader("Top holdings")

    top_n = 10
    ticker_col = find_column(df, ["Ticker", "Symbol", "Name", "Security"])
    label_col = ticker_col or df.columns[0]

    top_df = df.sort_values("Weight_pct", ascending=False).head(top_n)

    fig_top = px.bar(
        top_df,
        x="Weight_pct",
        y=label_col,
        orientation="h",
        labels={"Weight_pct": "Weight (%)", label_col: "Holding"},
        title=f"Top {min(top_n, len(df))} holdings"
    )
    fig_top.update_layout(yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig_top, use_container_width=True)

    st.markdown("---")
    st.subheader("Sector exposure")

    sector_col = find_column(df, ["Sector", "GICS Sector", "Industry"])

    if sector_col is None:
        st.warning("No sector column found. Add a 'Sector' column to your CSV to enable this chart.")
    else:
        sector_df = (
            df.groupby(sector_col, as_index=False)["Weight_pct"]
            .sum()
            .sort_values("Weight_pct", ascending=False)
        )
        fig_sector = px.pie(
            sector_df,
            names=sector_col,
            values="Weight_pct",
            title="Sector breakdown"
        )
        st.plotly_chart(fig_sector, use_container_width=True)

# --- Overrides & Targets tab ---
with overrides_tab:
    st.subheader("SmartSafe™ exposure & human overrides")

    adj_equity, adj_cash = apply_exposure_overrides(
        equity_weight=equity_weight,
        cash_weight=cash_weight,
        mode=smart_mode,
        equity_tilt=equity_tilt,
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Current equity", f"{equity_weight:0.1f}%")
    c2.metric("Current cash", f"{cash_weight:0.1f}%")
    c3.metric("SmartSafe™ mode", smart_mode)

    st.markdown("### Target exposure after overrides")

    t1, t2 = st.columns(2)
    t1.metric(
        "Target equity",
        f"{adj_equity:0.1f}%",
        delta=f"{adj_equity - equity_weight:0.1f} pts"
    )
    t2.metric(
        "Target cash",
        f"{adj_cash:0.1f}%",
        delta=f"{adj_cash - cash_weight:0.1f} pts"
    )

    st.markdown("---")
    st.markdown("### Style tilts (demo)")

    st.write(
        f"- Growth style tilt: **{growth_tilt} bps**  "
        f"- Value style tilt: **{value_tilt} bps**"
    )

    style_col = find_column(df, ["Style", "Category", "FactorBucket"])
    if style_col is None:
        st.caption(
            "No style column detected. In the full Waves Intelligence™ console, "
            "these tilts would map to factor/style buckets for live rebalancing."
        )
    else:
        st.caption(
            f"Style column detected: **{style_col}** – in future versions, "
            "these tilts will adjust target weights by style bucket."
        )

    st.markdown("---")
    st.caption(
        "This page is designed as a **human money manager dashboard** – "
        "AI manages the Wave, while humans can review, approve, and set guardrails."
    )