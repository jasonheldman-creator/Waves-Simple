import streamlit as st
import pandas as pd
from pathlib import Path

# ---------------------------------------------------------
# Helpers to load data
# ---------------------------------------------------------

@st.cache_data
def load_universe():
    """
    Load the full equity universe from list.csv if it exists.
    Expected columns (flexible): Ticker, Name, Sector, AssetClass/Type, etc.
    """
    p = Path("list.csv")
    if p.exists():
        try:
            df = pd.read_csv(p)
            return df
        except Exception:
            return None
    return None


@st.cache_data
def load_snapshot_from_repo():
    """
    Load a default live snapshot from the repo, if present.
    You can name it live_snapshot.csv, snapshot.csv, or SP500_PORTFOLIO_FINAL.csv.
    """
    candidates = [
        "live_snapshot.csv",
        "snapshot.csv",
        "SP500_PORTFOLIO_FINAL - Sheet17.csv",
        "SP500_PORTFOLIO_FINAL.csv",
    ]
    for name in candidates:
        p = Path(name)
        if p.exists():
            try:
                return pd.read_csv(p)
            except Exception:
                continue
    return None


def find_col(df, *options):
    """Return the actual column name in df that matches any of the options (case-insensitive)."""
    cols_lower = {c.lower(): c for c in df.columns}
    for opt in options:
        if opt.lower() in cols_lower:
            return cols_lower[opt.lower()]
    return None


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
st.write("Benchmark-aware, AI-directed Waves – equities only for this demo.\n")

# ---------------------------------------------------------
# Load universe + snapshot
# ---------------------------------------------------------

st.sidebar.header("Data source")

universe_df = load_universe()
uploaded_snapshot = st.sidebar.file_uploader(
    "Upload Wave snapshot CSV (optional)",
    type=["csv"],
    help="Rows = holdings. Include columns like Wave, Ticker, Weight, Sector, Alpha, IsCash, etc.",
)

if uploaded_snapshot is not None:
    snap_df = pd.read_csv(uploaded_snapshot)
    source_label = "Uploaded CSV"
else:
    snap_df = load_snapshot_from_repo()
    source_label = "Repo snapshot (e.g. SP500_PORTFOLIO_FINAL)"
st.sidebar.caption(f"Snapshot source: **{source_label}**")

if snap_df is None or snap_df.empty:
    st.warning(
        "No snapshot data loaded yet. Upload a CSV in the sidebar or add "
        "`live_snapshot.csv` (or SP500_PORTFOLIO_FINAL.csv) to the repo root."
    )
    st.stop()

# ---------------------------------------------------------
# Normalize snapshot columns
# ---------------------------------------------------------

col_wave    = find_col(snap_df, "Wave", "WaveName", "Portfolio")
col_ticker  = find_col(snap_df, "Ticker", "Symbol")
col_name    = find_col(snap_df, "Name", "Security", "Holding")
col_weight  = find_col(snap_df, "Weight", "PctWeight", "WeightPct")
col_sector  = find_col(snap_df, "Sector")
col_alpha   = find_col(snap_df, "Alpha", "AlphaCapture", "Alpha_Capture")
col_is_cash = find_col(snap_df, "IsCash", "CashFlag", "Is_Cash")
col_asset   = find_col(snap_df, "AssetClass", "Type", "Asset_Type")

if col_weight:
    snap_df[col_weight] = pd.to_numeric(snap_df[col_weight], errors="coerce")

# ---------------------------------------------------------
# Enrich from universe: Name / Sector / AssetClass
# ---------------------------------------------------------

if universe_df is not None and col_ticker:
    u_ticker   = find_col(universe_df, "Ticker", "Symbol")
    u_name     = find_col(universe_df, "Name", "Security")
    u_sector   = find_col(universe_df, "Sector")
    u_asset    = find_col(universe_df, "AssetClass", "Type", "Asset_Type")

    if u_ticker:
        merge_cols = {u_ticker: "Ticker"}
        tmp_uni = universe_df.rename(columns=merge_cols)

        # Keep only relevant columns
        keep_cols = ["Ticker"]
        if u_name:   keep_cols.append(u_name)
        if u_sector: keep_cols.append(u_sector)
        if u_asset:  keep_cols.append(u_asset)
        tmp_uni = tmp_uni[keep_cols].drop_duplicates(subset=["Ticker"])

        # Sync snapshot tickers into a common column name 'Ticker'
        snap_df = snap_df.rename(columns={col_ticker: "Ticker"})
        snap_df = snap_df.merge(tmp_uni, on="Ticker", how="left")

        # Re-resolve column pointers after merge
        col_ticker = "Ticker"
        if not col_name and u_name:
            col_name = u_name
        if not col_sector and u_sector:
            col_sector = u_sector
        if not col_asset and u_asset:
            col_asset = u_asset

# ---------------------------------------------------------
# Filter to equities only (if possible)
# ---------------------------------------------------------

if col_asset:
    equity_mask = snap_df[col_asset].astype(str).str.contains("equity", case=False, na=False)
    equity_df = snap_df[equity_mask].copy()
else:
    equity_df = snap_df.copy()

if equity_df.empty:
    st.error("Snapshot loaded, but no equity holdings detected. Check your AssetClass/Type column.")
    st.stop()

# ---------------------------------------------------------
# Wave selector (multi-wave support)
# ---------------------------------------------------------

if col_wave and col_wave in equity_df.columns:
    waves = sorted(equity_df[col_wave].dropna().unique().tolist())
    if not waves:
        waves = ["Equity Wave"]
        equity_df[col_wave] = waves[0]
else:
    # If no Wave column, treat everything as one Wave (e.g., S&P 500 Wave)
    col_wave = "Wave"
    equity_df[col_wave] = "S&P 500 Wave"
    waves = ["S&P 500 Wave"]

selected_wave = st.sidebar.selectbox("Select Wave", waves)
wave_df = equity_df[equity_df[col_wave] == selected_wave].copy()

if wave_df.empty:
    st.error("No rows found for the selected Wave.")
    st.stop()

# ---------------------------------------------------------
# Mode selector: Standard / Alpha-Minus-Beta / Private Logic
# ---------------------------------------------------------

mode = st.sidebar.radio(
    "Mode",
    ["Standard", "Alpha-Minus-Beta", "Private Logic™"],
)

# Mode scaling – demo only (for effective weight)
mode_scale = {
    "Standard": 1.00,          # pure benchmark-aligned
    "Alpha-Minus-Beta": 0.80,  # equity dialed down, more cash / protection
    "Private Logic™": 1.20,    # more aggressive expression (for demo)
}.get(mode, 1.00)

if col_weight:
    wave_df["EffectiveWeight"] = wave_df[col_weight] * mode_scale
    eff_col = "EffectiveWeight"
else:
    eff_col = None

# ---------------------------------------------------------
# Header
# ---------------------------------------------------------

st.markdown(
    f"<h1 style='color:#4CAFEB; margin-top:0'>{selected_wave} (LIVE Demo)</h1>",
    unsafe_allow_html=True,
)
st.caption(
    f"Mode: **{mode}** – equities only. "
    "In production, this mode flag would drive risk overlays, SmartSafe™, and rebalancing."
)

# ---------------------------------------------------------
# Top holdings + analytics
# ---------------------------------------------------------

left_col, right_col = st.columns([1.2, 1.3])

with left_col:
    st.subheader("Top 10 holdings")

    # Sort by effective weight if available, else by original weight
    if eff_col:
        wave_sorted = wave_df.sort_values(eff_col, ascending=False)
    elif col_weight:
        wave_sorted = wave_df.sort_values(col_weight, ascending=False)
    else:
        wave_sorted = wave_df.copy()

    top10 = wave_sorted.head(10).copy()

    # Build table
    display = {}

    if col_name and col_name in top10.columns:
        display["Name"] = top10[col_name]
    if col_ticker and col_ticker in top10.columns:
        display["Ticker"] = top10[col_ticker]

    if col_weight:
        display["Base Weight"] = top10[col_weight].apply(
            lambda x: f"{x:.1f}%" if pd.notna(x) else ""
        )
    if eff_col:
        display["Mode Weight"] = top10[eff_col].apply(
            lambda x: f"{x:.1f}%" if pd.notna(x) else ""
        )

    if col_sector and col_sector in top10.columns:
        display["Sector"] = top10[col_sector]

    if display:
        top_table = pd.DataFrame(display)
        st.markdown("Ranked by Wave weight (mode-adjusted)")
        st.dataframe(top_table, use_container_width=True)
    else:
        st.info("Add Ticker / Name / Weight columns in your snapshot to see Top-10.")

with right_col:
    st.subheader("Top-10 by Wave weight")

    weight_col_for_chart = eff_col or col_weight
    if weight_col_for_chart and col_ticker:
        chart_data = top10[[col_ticker, weight_col_for_chart]].set_index(col_ticker)
        st.bar_chart(chart_data)
    else:
        st.info("Need Ticker and Weight to show the Top-10 chart.")

    # Summary metrics
    total_holdings = len(wave_df)
    largest_weight = wave_df[weight_col_for_chart].max() if weight_col_for_chart else None

    equity_weight = None
    cash_weight = None
    if col_is_cash and col_weight:
        cash_weight = wave_df.loc[wave_df[col_is_cash] == True, col_weight].sum()
        equity_weight = wave_df[col_weight].sum() - cash_weight

    alpha_capture = wave_df[col_alpha].mean() if col_alpha and col_alpha in wave_df.columns else None

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("TOTAL HOLDINGS", f"{total_holdings}")
    with c2:
        if equity_weight is not None and cash_weight is not None:
            st.metric(
                "EQUITY vs CASH (base)",
                f"{equity_weight:.0f}% / {cash_weight:.0f}%",
            )
        else:
            st.metric("EQUITY vs CASH", "n/a")
    with c3:
        if largest_weight is not None:
            st.metric("LARGEST POSITION (mode)", f"{largest_weight:.1f}%")
        else:
            st.metric("LARGEST POSITION (mode)", "n/a")

    c4, c5 = st.columns(2)
    with c4:
        if alpha_capture is not None and pd.notna(alpha_capture):
            st.metric("ALPHA CAPTURE (EST.)", f"{alpha_capture:.2f}")
        else:
            st.metric("ALPHA CAPTURE (EST.)", "n/a")

# ---------------------------------------------------------
# Sector allocation & weight-decay
# ---------------------------------------------------------

st.markdown("---")
sec_col, decay_col = st.columns(2)

with sec_col:
    st.subheader("Sector allocation")

    if col_sector and col_sector in wave_df.columns and weight_col_for_chart:
        sector_data = (
            wave_df
            .dropna(subset=[col_sector, weight_col_for_chart])
            .groupby(col_sector)[weight_col_for_chart]
            .sum()
            .sort_values(ascending=False)
        )
        if not sector_data.empty:
            st.bar_chart(sector_data)
        else:
            st.info("No sector data available after filtering.")
    else:
        st.info("Add a Sector column and Weight column in your data to see sector allocation.")

with decay_col:
    st.subheader("Weight decay curve")

    if weight_col_for_chart:
        decay_data = (
            wave_sorted[[weight_col_for_chart]]
            .reset_index(drop=True)
            .rename(columns={weight_col_for_chart: "Weight"})
        )
        decay_data["Rank"] = decay_data.index + 1
        decay_data = decay_data.set_index("Rank")
        st.line_chart(decay_data)
    else:
        st.info("Need a Weight column to show weight-decay.")

# ---------------------------------------------------------
# Mode overview / Console status
# ---------------------------------------------------------

st.markdown("---")
st.subheader("Mode overview")
st.write(
    """
**Standard mode** keeps the Wave tightly aligned to its benchmark with controlled tracking error
and strict beta discipline.

**Alpha-Minus-Beta** dials down net equity exposure (e.g., 80% effective weight here) while keeping
the stock selection logic intact, making room for SmartSafe™ or hedging overlays.

**Private Logic™** represents proprietary leadership / momentum / SmartSafe™ overlays where the
effective equity weight can expand or contract more aggressively.
"""
)

st.subheader("Console status")
st.write(
    """
- **Read-only demo** – no real orders are routed from this screen.  
- All analytics come directly from the snapshot CSV.  
- Equities only for this version (crypto & income Waves can be layered later).  
- Wave + Mode selection here would drive live management in the production engine.
"""
)