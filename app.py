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
def load_default_snapshot() -> pd.DataFrame | None:
    """
    Try to load a default snapshot from the repo.
    You can rename your master file to one of these.
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


def percent(x):
    try:
        return f"{float(x):.1f}%"
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
# Data source (snapshot file)
# =========================================================

st.sidebar.header("Data source")

uploaded = st.sidebar.file_uploader(
    "Upload Wave snapshot CSV (equity holdings)",
    type=["csv"],
    help=(
        "Rows = holdings. Columns like Wave, Ticker, Name, Weight, Sector, "
        "AssetClass, IsCash, Alpha are supported."
    ),
)

if uploaded is not None:
    df = pd.read_csv(uploaded)
    source_label = "Uploaded CSV"
else:
    df = load_default_snapshot()
    source_label = "Default snapshot from repo"

st.sidebar.caption(f"Snapshot source: **{source_label}**")

if df is None or df.empty:
    st.warning(
        "No snapshot data loaded. Upload a CSV in the sidebar or place "
        "`live_snapshot.csv` (or SP500_PORTFOLIO_FINAL.csv) in the repo root."
    )
    st.stop()

# =========================================================
# Column detection and cleaning
# =========================================================

col_wave    = find_col(df, "Wave", "WaveName", "Portfolio")
col_ticker  = find_col(df, "Ticker", "Symbol")
col_name    = find_col(df, "Name", "Security", "Holding")
col_weight  = find_col(df, "Weight", "Weight %", "Weight(%)", "PctWeight", "WeightPct")
col_sector  = find_col(df, "Sector")
col_alpha   = find_col(df, "Alpha", "AlphaCapture", "Alpha_Capture")
col_asset   = find_col(df, "AssetClass", "Type", "Asset_Type")
col_is_cash = find_col(df, "IsCash", "CashFlag", "Is_Cash")

if col_weight:
    df[col_weight] = pd.to_numeric(df[col_weight], errors="coerce")

# Filter to equities only if we can detect an AssetClass / Type column
if col_asset:
    mask_equity = df[col_asset].astype(str).str.contains("equity", case=False, na=False)
    df_equity = df[mask_equity].copy()
else:
    df_equity = df.copy()

if df_equity.empty:
    st.error("Snapshot loaded, but no equity rows detected. Check your AssetClass / Type column.")
    st.stop()

# If there is no Wave column, treat everything as one Wave (e.g., S&P 500 Wave)
if not col_wave:
    col_wave = "Wave"
    df_equity[col_wave] = "S&P 500 Wave"

waves = sorted(df_equity[col_wave].dropna().unique().tolist())
selected_wave = st.sidebar.selectbox("Select Wave", waves)

wave_df = df_equity[df_equity[col_wave] == selected_wave].copy()
if wave_df.empty:
    st.error("No holdings found for the selected Wave.")
    st.stop()

# =========================================================
# Mode selector (Standard / Alpha-Minus-Beta / Private Logic)
# =========================================================

mode = st.sidebar.radio(
    "Mode",
    ["Standard", "Alpha-Minus-Beta", "Private Logic™"],
)

# Simple demo scaling for effective weights
mode_scale = {
    "Standard": 1.00,
    "Alpha-Minus-Beta": 0.80,   # net exposure dialed down
    "Private Logic™": 1.20,     # more aggressive expression
}.get(mode, 1.00)

if col_weight:
    wave_df["EffectiveWeight"] = wave_df[col_weight] * mode_scale
    eff_col = "EffectiveWeight"
else:
    eff_col = None

# =========================================================
# Header
# =========================================================

st.markdown(
    f"<h1 style='color:#4CAFEB; margin-top:0'>{selected_wave} (LIVE Demo)</h1>",
    unsafe_allow_html=True,
)
st.caption(
    f"Mode: **{mode}** – equities only. In production this mode flag would drive "
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
    if eff_col:
        wave_sorted = wave_df.sort_values(eff_col, ascending=False)
    elif col_weight:
        wave_sorted = wave_df.sort_values(col_weight, ascending=False)
    else:
        wave_sorted = wave_df.copy()

    top10 = wave_sorted.head(10).copy()

    table = {}

    if col_name and col_name in top10.columns:
        table["Name"] = top10[col_name]
    if col_ticker and col_ticker in top10.columns:
        table["Ticker"] = top10[col_ticker]
    if col_weight:
        table["Base weight"] = top10[col_weight].apply(percent)
    if eff_col:
        table["Mode weight"] = top10[eff_col].apply(percent)
    if col_sector and col_sector in top10.columns:
        table["Sector"] = top10[col_sector]

    if table:
        st.caption("Ranked by Wave weight (mode-adjusted).")
        st.dataframe(pd.DataFrame(table), use_container_width=True)
    else:
        st.info("Add at least Ticker, Name, and Weight columns to see the Top-10 table.")

# ---------- Right: chart + metrics ----------
with right:
    st.subheader("Top-10 by Wave weight")
    weight_for_chart = eff_col or col_weight

    if weight_for_chart and col_ticker and col_ticker in top10.columns:
        chart_data = top10[[col_ticker, weight_for_chart]].set_index(col_ticker)
        st.bar_chart(chart_data)
    else:
        st.info("Need Ticker and Weight columns to show the Top-10 chart.")

    total_holdings = len(wave_df)

    largest_weight = None
    if weight_for_chart:
        largest_weight = wave_df[weight_for_chart].max()

    equity_weight = None
    cash_weight = None
    if col_is_cash and col_weight:
        cash_mask = wave_df[col_is_cash] == True
        cash_weight = wave_df.loc[cash_mask, col_weight].sum()
        equity_weight = wave_df[col_weight].sum() - cash_weight

    alpha_capture = None
    if col_alpha and col_alpha in wave_df.columns:
        alpha_capture = wave_df[col_alpha].mean()

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
            st.metric("LARGEST POSITION (mode)", percent(largest_weight))
        else:
            st.metric("LARGEST POSITION (mode)", "n/a")

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
    if col_sector and weight_for_chart and col_sector in wave_df.columns:
        sector_data = (
            wave_df
            .dropna(subset=[col_sector, weight_for_chart])
            .groupby(col_sector)[weight_for_chart]
            .sum()
            .sort_values(ascending=False)
        )
        if not sector_data.empty:
            st.bar_chart(sector_data)
        else:
            st.info("No sector distribution available for this Wave.")
    else:
        st.info("Add Sector and Weight columns to your snapshot to see sector allocation.")

with decay_col:
    st.subheader("Weight decay curve")
    if weight_for_chart:
        decay_data = (
            wave_sorted[[weight_for_chart]]
            .reset_index(drop=True)
            .rename(columns={weight_for_chart: "Weight"})
        )
        decay_data["Rank"] = decay_data.index + 1
        decay_data = decay_data.set_index("Rank")
        st.line_chart(decay_data)
    else:
        st.info("Need a Weight column to show weight-decay.")

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
- All analytics are calculated directly from the loaded snapshot CSV.  
- **Equities only** in this version (crypto and income Waves can be added later).  
- Wave + Mode selections are exactly how the production engine will be driven.
"""
)