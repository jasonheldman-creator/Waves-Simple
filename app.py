import streamlit as st
import pandas as pd
from pathlib import Path

# ---------------------------------------------------------
# Helper functions
# ---------------------------------------------------------

@st.cache_data
def load_default_csv():
    """
    Try to load a default CSV from the repo, e.g. 'live_snapshot.csv'.
    If it doesn't exist, return None and we fall back to upload.
    """
    possible_files = ["live_snapshot.csv", "snapshot.csv", "data.csv"]
    for name in possible_files:
        p = Path(name)
        if p.exists():
            try:
                return pd.read_csv(p)
            except Exception:
                continue
    return None


def build_google_link(ticker: str) -> str:
    """
    Create a Google Finance link for the ticker.
    We use a generic search link so we don't care about exchange codes.
    """
    if not isinstance(ticker, str):
        return ""
    url = f"https://www.google.com/finance?q={ticker}"
    return f"[Google]({url})"


def percent_fmt(x):
    try:
        return f"{float(x):.1f}%"
    except Exception:
        return ""


# ---------------------------------------------------------
# Page setup
# ---------------------------------------------------------

st.set_page_config(
    page_title="WAVES Intelligence™ – Portfolio Wave Console",
    layout="wide",
)

st.markdown(
    "<h2 style='margin-bottom:0'>WAVES INTELLIGENCE™ – PORTFOLIO WAVE CONSOLE</h2>",
    unsafe_allow_html=True,
)

st.write("")

# ---------------------------------------------------------
# Data source: default CSV + optional upload
# ---------------------------------------------------------

st.sidebar.header("Data source")

default_df = load_default_csv()
uploaded_file = st.sidebar.file_uploader(
    "Upload a CSV snapshot (optional)",
    type=["csv"],
    help="Columns like Wave, Ticker, Name, Weight, Sector, Alpha, IsCash, etc.",
)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    source_label = "Uploaded CSV"
elif default_df is not None:
    df = default_df
    source_label = "live_snapshot.csv in repo"
else:
    df = None
    source_label = "No data loaded"

st.sidebar.caption(f"Current source: **{source_label}**")

if df is None or df.empty:
    st.warning(
        "No data loaded yet. Upload a CSV in the sidebar or add `live_snapshot.csv` "
        "to the repo root. The console will update automatically."
    )
    st.stop()

# ---------------------------------------------------------
# Normalize expected columns
# ---------------------------------------------------------

# Try to normalize column names to something predictable
cols_lower = {c.lower(): c for c in df.columns}

def find_col(*options):
    """Return the real column name if any of the options exist (case-insensitive)."""
    for opt in options:
        if opt.lower() in cols_lower:
            return cols_lower[opt.lower()]
    return None

col_wave   = find_col("Wave", "WaveName", "Portfolio")
col_name   = find_col("Name", "Holding", "Security")
col_ticker = find_col("Ticker", "Symbol")
col_weight = find_col("Weight", "PctWeight", "WeightPct")
col_sector = find_col("Sector")
col_alpha  = find_col("Alpha", "AlphaCapture", "Alpha_Capture")
col_is_cash = find_col("IsCash", "CashFlag", "Is_Cash")

# If weight exists, make sure it's numeric
if col_weight:
    df[col_weight] = pd.to_numeric(df[col_weight], errors="coerce")

# ---------------------------------------------------------
# Wave selector – so we can reuse console for any Wave
# ---------------------------------------------------------

if col_wave:
    waves = sorted(df[col_wave].dropna().unique().tolist())
    default_wave = waves[0] if waves else "Wave"
    selected_wave = st.sidebar.selectbox("Select Wave", waves, index=0)
    wave_df = df[df[col_wave] == selected_wave].copy()
else:
    selected_wave = "Wave"
    wave_df = df.copy()

if wave_df.empty:
    st.warning("No rows found for the selected Wave. Check your CSV filters.")
    st.stop()

# ---------------------------------------------------------
# Header / subtitle
# ---------------------------------------------------------

st.markdown(
    f"<h1 style='color:#4CAFEB; margin-top:0'>{selected_wave} (LIVE Demo)</h1>",
    unsafe_allow_html=True,
)
st.caption("Benchmark-aware, AI-directed Wave – rendered in a single screen, Bloomberg-style.")

# ---------------------------------------------------------
# Top holdings + Analytics layout
# ---------------------------------------------------------

left_col, right_col = st.columns([1.2, 1.3])

with left_col:
    st.subheader("Top 10 holdings")

    if col_weight:
        wave_df_sorted = wave_df.sort_values(col_weight, ascending=False)
    else:
        wave_df_sorted = wave_df.copy()

    top10 = wave_df_sorted.head(10).copy()

    # Build display table
    display_cols = {}

    if col_name:
        display_cols["Name"] = top10[col_name]
    if col_ticker:
        display_cols["Ticker"] = top10[col_ticker]

    if col_weight:
        display_cols["Weight"] = top10[col_weight].apply(lambda x: f"{x:.1f}%")
    else:
        display_cols["Weight"] = ""

    # Google links
    if col_ticker:
        display_cols["Link"] = top10[col_ticker].apply(build_google_link)

    if display_cols:
        top_table = pd.DataFrame(display_cols)
        st.markdown("Ranked by Wave weight")
        st.markdown(
            top_table.to_markdown(index=False),
            unsafe_allow_html=True,
        )
    else:
        st.info("Add columns like Ticker / Name / Weight to see the Top-10 table.")

    st.caption(
        "Positive 1D moves render in green; negative in red in future versions. "
        "Click Google for a full Google Finance profile without leaving the console."
    )

with right_col:
    st.subheader("Top-10 by Wave weight")

    if col_ticker and col_weight:
        chart_data = top10[[col_ticker, col_weight]].set_index(col_ticker)
        st.bar_chart(chart_data)
    else:
        st.info("Need Ticker and Weight columns to show this chart.")

    st.write("")
    # Summary tiles
    total_holdings = len(wave_df)
    largest_weight = wave_df[col_weight].max() if col_weight else None
    equity_weight = None
    cash_weight = None

    if col_is_cash and col_weight:
        cash_weight = wave_df.loc[wave_df[col_is_cash] == True, col_weight].sum()
        equity_weight = wave_df[col_weight].sum() - cash_weight

    alpha_capture = None
    if col_alpha:
        alpha_capture = wave_df[col_alpha].mean()

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric("TOTAL HOLDINGS", f"{total_holdings}")
    with c2:
        if equity_weight is not None and cash_weight is not None:
            st.metric(
                "EQUITY vs CASH",
                f"{equity_weight:.0f}% / {cash_weight:.0f}%",
                help="Approximate split based on IsCash flag and Weight column.",
            )
        else:
            st.metric("EQUITY vs CASH", "n/a")
    with c3:
        if largest_weight is not None:
            st.metric(
                "LARGEST POSITION",
                percent_fmt(largest_weight),
                help="Single-name concentration by weight.",
            )
        else:
            st.metric("LARGEST POSITION", "n/a")

    st.write("")
    c4, c5 = st.columns(2)
    with c4:
        if alpha_capture is not None:
            st.metric("ALPHA CAPTURE (EST.)", f"{alpha_capture:.2f}")
        else:
            st.metric("ALPHA CAPTURE (EST.)", "n/a")

# ---------------------------------------------------------
# Sector allocation + weight decay
# ---------------------------------------------------------

st.write("")
st.markdown("---")

sec_col, decay_col = st.columns(2)

with sec_col:
    st.subheader("Sector allocation")

    if col_sector and col_weight:
        sector_data = (
            wave_df
            .dropna(subset=[col_sector, col_weight])
            .groupby(col_sector)[col_weight]
            .sum()
            .sort_values(ascending=False)
        )
        st.bar_chart(sector_data)
        st.caption("No 'Sector' column detected – add one to see sector allocation." if sector_data.empty else "")
    else:
        st.info("Add a 'Sector' column and Weight column to enable sector allocation.")

with decay_col:
    st.subheader("Weight decay curve")

    if col_weight:
        decay_data = (
            wave_df_sorted[[col_weight]]
            .reset_index(drop=True)
            .rename(columns={col_weight: "Weight"})
        )
        decay_data["Rank"] = decay_data.index + 1
        decay_data = decay_data.set_index("Rank")
        st.line_chart(decay_data)
        st.caption("Holding rank vs. Wave weight (shows diversification profile).")
    else:
        st.info("Need a Weight column to show the weight-decay curve.")

# ---------------------------------------------------------
# Mode overview / console status
# ---------------------------------------------------------

st.write("")
st.markdown("---")

st.subheader("Mode overview")
st.write(
    """
**Standard mode** keeps the Wave tightly aligned to its benchmark with controlled
tracking error and strict beta discipline.  
**Alpha-Minus-Beta** and **Private Logic™** modes can be layered on in production
to shift exposure and apply proprietary rebalancing and SmartSafe™ overlays.
"""
)

st.subheader("Console status")
st.write(
    """
- **Read-only** – no real orders are routed from this demo screen.  
- All analytics are calculated directly from the loaded snapshot CSV.  
- Every Wave and mode can be exported to a full institutional console.  
- Quote links open in Google Finance for a clean, broker-neutral view.
"""
)