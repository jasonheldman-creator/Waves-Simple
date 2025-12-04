import streamlit as st
import pandas as pd
from pathlib import Path

# =========================================================
# Helper functions
# =========================================================

def find_col(df: pd.DataFrame, *candidates):
    """Return existing column name matching any candidate (case-insensitive)."""
    cols_lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        key = cand.lower()
        if key in cols_lower:
            return cols_lower[key]
    return None


@st.cache_data
def load_universe() -> pd.DataFrame | None:
    """
    Load the master 5,000-stock universe from Master_Stock_Sheet5.csv.
    Expected columns (flexible): Ticker, Company, Weight, Sector, Market Value, Price, ...
    """
    p = Path("Master_Stock_Sheet5.csv")
    if not p.exists():
        return None

    df = pd.read_csv(p)

    # Normalize key columns
    col_ticker = find_col(df, "Ticker", "Symbol")
    col_name   = find_col(df, "Company", "Name", "Security")
    col_sector = find_col(df, "Sector")
    col_price  = find_col(df, "Price")
    col_mktval = find_col(df, "Market Value", "MarketValue", "MktValue")
    col_weight = find_col(df, "Weight")  # benchmark / index weight, not Wave weight

    # Rename for clearer use downstream
    rename_map = {}
    if col_ticker: rename_map[col_ticker] = "Ticker"
    if col_name:   rename_map[col_name]   = "Company"
    if col_sector: rename_map[col_sector] = "Sector"
    if col_price:  rename_map[col_price]  = "Price"
    if col_mktval: rename_map[col_mktval] = "MarketValue"
    if col_weight: rename_map[col_weight] = "BenchmarkWeight"

    df = df.rename(columns=rename_map)

    # Keep only the most relevant columns for now
    keep_cols = []
    for c in ["Ticker", "Company", "Sector", "Price", "MarketValue", "BenchmarkWeight"]:
        if c in df.columns:
            keep_cols.append(c)
    df = df[keep_cols].dropna(subset=["Ticker"]).drop_duplicates(subset=["Ticker"])

    return df


@st.cache_data
def load_wave_weights(uploaded_file: pd.io.common.FilePath | None) -> pd.DataFrame | None:
    """
    Load wave definitions (Wave / Ticker / Weight).
    If user uploads a CSV, use that; otherwise fall back to wave_weights.csv in repo.
    """
    if uploaded_file is not None:
        try:
            return pd.read_csv(uploaded_file)
        except Exception:
            return None

    p = Path("wave_weights.csv")
    if p.exists():
        try:
            return pd.read_csv(p)
        except Exception:
            return None

    return None


def percent(x):
    try:
        return f"{float(x) * 100:.1f}%"
    except Exception:
        return ""


# =========================================================
# Page config
# =========================================================

st.set_page_config(
    page_title="WAVES Intelligence – Equity Wave Console",
    layout="wide",
)

st.markdown(
    "<h2 style='margin-bottom:0'>WAVES INTELLIGENCE™ – PORTFOLIO WAVE CONSOLE</h2>",
    unsafe_allow_html=True,
)
st.caption("Equity Waves only – benchmark-aware, AI-directed, multi-mode demo.")
st.write("")

# =========================================================
# Sidebar data source controls
# =========================================================

st.sidebar.header("Data sources")

uploaded_wave_file = st.sidebar.file_uploader(
    "Upload wave_weights.csv (optional)",
    type=["csv"],
    help=(
        "Columns: Wave, Ticker, Weight (0–1). "
        "Optional: Alpha, IsCash, any other metrics."
    ),
)

universe_df = load_universe()
waves_df = load_wave_weights(uploaded_wave_file)

# Universe status
if universe_df is None or universe_df.empty:
    st.sidebar.error("Universe file Master_Stock_Sheet5.csv not found or empty.")
    st.error(
        "Cannot find Master_Stock_Sheet5.csv in the app folder. "
        "Export Sheet5 from your Google Sheet as CSV and add it to the repo root."
    )
    st.stop()
else:
    st.sidebar.success("Universe loaded (Master_Stock_Sheet5.csv)")

# Wave weights status
if waves_df is None or waves_df.empty:
    st.sidebar.warning(
        "No wave_weights.csv found or it is empty. "
        "Create wave_weights.csv with columns Wave, Ticker, Weight."
    )
    st.warning(
        "No Wave definitions loaded. The console needs a wave_weights.csv file "
        "with at least Wave, Ticker, and Weight columns."
    )
    st.stop()
else:
    st.sidebar.success("Wave definitions loaded")

# =========================================================
# Normalize wave_weights columns
# =========================================================

col_wave   = find_col(waves_df, "Wave")
col_ticker = find_col(waves_df, "Ticker", "Symbol")
col_wgt    = find_col(waves_df, "Weight", "WaveWeight", "Wgt")
col_alpha  = find_col(waves_df, "Alpha", "AlphaCapture", "Alpha_Capture")
col_is_cash = find_col(waves_df, "IsCash", "CashFlag", "Is_Cash")

if not col_wave or not col_ticker or not col_wgt:
    st.error(
        "wave_weights.csv must contain at least these columns: Wave, Ticker, Weight."
    )
    st.stop()

waves_df = waves_df.rename(columns={
    col_wave: "Wave",
    col_ticker: "Ticker",
    col_wgt: "WaveWeight"
})
if col_alpha:
    waves_df = waves_df.rename(columns={col_alpha: "Alpha"})
if col_is_cash:
    waves_df = waves_df.rename(columns={col_is_cash: "IsCash"})

waves_df["WaveWeight"] = pd.to_numeric(waves_df["WaveWeight"], errors="coerce")

# Drop rows with no ticker or weight
waves_df = waves_df.dropna(subset=["Ticker", "WaveWeight"])

if waves_df.empty:
    st.error("After cleaning, no valid Wave rows remain. Check wave_weights.csv.")
    st.stop()

# =========================================================
# Join waves with universe on Ticker
# =========================================================

joined_df = waves_df.merge(
    universe_df,
    on="Ticker",
    how="left",
    suffixes=("", "_universe"),
)

# We keep all wave rows even if some tickers are missing in the universe
missing_universe = joined_df["Company"].isna().sum()
if missing_universe > 0:
    st.sidebar.warning(
        f"{missing_universe} Wave holdings are missing from the universe file "
        "(Ticker not found in Master_Stock_Sheet5.csv)."
    )

# =========================================================
# Wave selector
# =========================================================

waves = sorted(joined_df["Wave"].dropna().unique().tolist())
if not waves:
    st.error("No Waves found in wave_weights.csv.")
    st.stop()

selected_wave = st.sidebar.selectbox("Select Wave", waves)

wave_df = joined_df[joined_df["Wave"] == selected_wave].copy()
if wave_df.empty:
    st.error("No holdings found for the selected Wave.")
    st.stop()

# =========================================================
# Mode selector
# =========================================================

mode = st.sidebar.radio(
    "Mode",
    ["Standard", "Alpha-Minus-Beta", "Private Logic™"],
)

mode_scale = {
    "Standard": 1.00,
    "Alpha-Minus-Beta": 0.80,   # lower net equity exposure
    "Private Logic™": 1.20,     # more aggressive expression
}.get(mode, 1.00)

wave_df["EffectiveWeight"] = wave_df["WaveWeight"] * mode_scale

# optional alpha / cash flags
has_alpha = "Alpha" in wave_df.columns
has_cash  = "IsCash" in wave_df.columns

# =========================================================
# Header
# =========================================================

st.markdown(
    f"<h1 style='color:#4CAFEB; margin-top:0'>{selected_wave} (LIVE Demo)</h1>",
    unsafe_allow_html=True,
)
st.caption(
    f"Mode: **{mode}** – equities only. In production, this mode flag would drive "
    "risk overlays, SmartSafe™, and rebalancing."
)
st.write("")

# =========================================================
# Top holdings & analytics
# =========================================================

left, right = st.columns([1.3, 1.2])

# ---------- Left: Top-10 table ----------
with left:
    st.subheader("Top 10 holdings")

    wave_sorted = wave_df.sort_values("EffectiveWeight", ascending=False)
    top10 = wave_sorted.head(10).copy()

    table = {}
    if "Company" in top10.columns:
        table["Name"] = top10["Company"]
    table["Ticker"] = top10["Ticker"]
    table["Base weight"] = top10["WaveWeight"].apply(percent)
    table["Mode weight"] = top10["EffectiveWeight"].apply(percent)
    if "Sector" in top10.columns:
        table["Sector"] = top10["Sector"]

    st.caption("Ranked by Wave weight (mode-adjusted).")
    st.dataframe(pd.DataFrame(table), use_container_width=True)

# ---------- Right: chart + metrics ----------
with right:
    st.subheader("Top-10 by Wave weight")

    chart_data = (
        top10[["Ticker", "EffectiveWeight"]]
        .set_index("Ticker")
    )
    st.bar_chart(chart_data)

    total_holdings = len(wave_df)
    largest_weight = wave_df["EffectiveWeight"].max()

    equity_weight = wave_df["WaveWeight"].sum()
    cash_weight = 0.0
    if has_cash:
        cash_mask = wave_df["IsCash"] == True
        cash_weight = wave_df.loc[cash_mask, "WaveWeight"].sum()
        equity_weight = wave_df["WaveWeight"].sum() - cash_weight

    alpha_capture = None
    if has_alpha:
        alpha_capture = wave_df["Alpha"].mean()

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("TOTAL HOLDINGS", f"{total_holdings}")
    with c2:
        if equity_weight or cash_weight:
            st.metric(
                "EQUITY vs CASH (base)",
                f"{equity_weight * 100:.0f}% / {cash_weight * 100:.0f}%",
            )
        else:
            st.metric("EQUITY vs CASH", "n/a")
    with c3:
        st.metric("LARGEST POSITION (mode)", percent(largest_weight))

    c4, c5 = st.columns(2)
    with c4:
        if alpha_capture is not None and pd.notna(alpha_capture):
            st.metric("ALPHA CAPTURE (est.)", f"{alpha_capture:.2f}")
        else:
            st.metric("ALPHA CAPTURE (est.)", "n/a")

# =========================================================
# Sector allocation & weight-decay
# =========================================================

st.markdown("---")
sec_col, decay_col = st.columns(2)

with sec_col:
    st.subheader("Sector allocation")
    if "Sector" in wave_df.columns:
        sector_data = (
            wave_df
            .dropna(subset=["Sector"])
            .groupby("Sector")["EffectiveWeight"]
            .sum()
            .sort_values(ascending=False)
        )
        if not sector_data.empty:
            st.bar_chart(sector_data)
        else:
            st.info("No sector distribution available for this Wave.")
    else:
        st.info("Sector column not present in universe; cannot show sector allocation.")

with decay_col:
    st.subheader("Weight decay curve")
    decay_data = (
        wave_sorted[["EffectiveWeight"]]
        .reset_index(drop=True)
        .rename(columns={"EffectiveWeight": "Weight"})
    )
    decay_data["Rank"] = decay_data.index + 1
    decay_data = decay_data.set_index("Rank")
    st.line_chart(decay_data)

# =========================================================
# Mode overview / console status
# =========================================================

st.markdown("---")
st.subheader("Mode overview")
st.write(
    """
**Standard** – Wave aligned tightly to its benchmark with controlled tracking error  
and strict beta discipline.

**Alpha-Minus-Beta** – same selection logic, but effective equity exposure dialed  
down (e.g., 80%) to make room for SmartSafe™ / hedging overlays.

**Private Logic™** – proprietary overlays for leadership, momentum, and SmartSafe™,  
allowing equity exposure to expand or contract more aggressively.
"""
)

st.subheader("Console status")
st.write(
    """
- This is a **read-only** demo – no real orders are routed.  
- All analytics are calculated directly from the loaded Wave definitions and universe.  
- **Equities only** in this version (crypto and income Waves will be layered later).  
- Wave + Mode selections match how the production engine will be driven.
"""
)